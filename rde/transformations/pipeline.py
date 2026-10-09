"""Transformation pipeline: apply operators, materialise a new version, record lineage.

Execution is two-pass and streaming:

**Pass 1 (fit)** runs only if some operator needs statistics.  It streams the
frame table once, feeding each fitting operator; nothing is written.

**Pass 2 (apply)** streams again, applying episode selection, then row filters,
then column transforms, writing batches straight to the output Parquet file.
Peak memory is one batch, so the cost is independent of dataset size.

Deferred operators (image resize) contribute to the manifest and the lineage
record but produce no data writes -- see :class:`~rde.transformations.ops.ResizeImages`
for why.

The output version's episodes carry their source episode ids, so the catalog can
answer episode-level provenance without replaying the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

from ..catalog import Catalog
from ..config import Settings, get_settings
from ..ingestion.ingest import episodes_to_table
from ..schemas.core import DatasetManifest, Episode, QualityStatus
from ..storage import BlobStore, ParquetTableStore
from ..versioning import VersionManager, code_fingerprint
from .base import EpisodeFilter, FrameFilter, FrameTransform, Transform, TransformContext


@dataclass
class PipelineResult:
    """What a transformation run changed: versions, counts, dropped episodes."""
    dataset_id: str
    input_version: str
    output_version: str
    operations: list[dict[str, Any]]
    episodes_in: int
    episodes_out: int
    frames_in: int
    frames_out: int
    dropped_episodes: list[int] = field(default_factory=list)
    fitted: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Compact view for the CLI and the run record."""
        return {
            "dataset_id": self.dataset_id,
            "input_version": self.input_version,
            "output_version": self.output_version,
            "operations": [o["name"] for o in self.operations],
            # Display strings, kept because the CLI and run record read them.
            "episodes": f"{self.episodes_in} -> {self.episodes_out}",
            "frames": f"{self.frames_in} -> {self.frames_out}",
            # Structured counts alongside, for instrumentation.  Transform is
            # precisely the stage where in != out, which is what the volume
            # pillar exists to watch: a stage that consumed 19,631 frames and
            # emitted 12 raises no exception and is obviously broken.
            "frames_in": self.frames_in,
            "frames_out": self.frames_out,
            "episodes_in": self.episodes_in,
            "episodes_out": self.episodes_out,
            "dropped_episodes": self.dropped_episodes,
        }


class TransformPipeline:
    """Ordered composition of operators producing a new dataset version."""

    def __init__(
        self,
        operations: Sequence[Transform],
        settings: Settings | None = None,
        catalog: Catalog | None = None,
    ):
        self.operations = list(operations)
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.paths.catalog_db)

    # -- helpers ------------------------------------------------------------
    def _load_episodes(self, dataset_id: str, version: str) -> list[Episode]:
        rows = self.catalog.get_episodes(dataset_id, version)
        tables = ParquetTableStore(self.settings.paths.version_dir(dataset_id, version))
        extra: dict[int, dict] = {}
        if tables.exists("episodes"):
            t = tables.read_table("episodes").to_pydict()
            extra = {
                int(e): {"blob_ids": json.loads(b or "{}"), "attributes": json.loads(a or "{}")}
                for e, b, a in zip(t["episode_id"], t["blob_ids"], t["attributes"])
            }
        out = []
        for r in rows:
            x = extra.get(r["episode_id"], {})
            out.append(
                Episode(
                    episode_id=r["episode_id"], dataset_id=dataset_id, task_id=r["task_id"],
                    frame_start=r["frame_start"], frame_end=r["frame_end"],
                    end_timestamp_s=r["duration_s"],
                    quality_status=QualityStatus(r["quality_status"]),
                    modalities=json.loads(r["modalities"] or "[]"),
                    blob_ids=x.get("blob_ids", {}),
                    attributes=x.get("attributes", {}),
                )
            )
        return out

    def _fingerprint(self) -> str:
        parts = [code_fingerprint(type(op)) for op in self.operations]
        import hashlib

        return hashlib.sha256("".join(parts).encode()).hexdigest()[:16]

    # -- execution ----------------------------------------------------------
    def run(
        self,
        dataset_id: str,
        input_version: str,
        output_version: str,
    ) -> PipelineResult:
        """Fit if needed, then stream the input version into a new output version."""
        manifest = self.catalog.get_manifest(dataset_id, input_version)
        if manifest is None:
            raise LookupError(f"unknown version: {dataset_id}@{input_version}")

        episodes = self._load_episodes(dataset_id, input_version)
        src_tables = ParquetTableStore(
            self.settings.paths.version_dir(dataset_id, input_version)
        )
        dst_tables = ParquetTableStore(
            self.settings.paths.version_dir(dataset_id, output_version),
            compression=self.settings.parquet_compression,
            row_group_size=self.settings.row_group_size,
        )

        ctx = TransformContext(
            manifest=manifest.model_copy(deep=True),
            episodes=episodes,
            settings=self.settings,
            table_store=src_tables,
            blob_store=BlobStore(self.settings.paths.blobs),
        )

        # -- episode selection (before any IO on frames) --------------------
        keep_ids = {e.episode_id for e in episodes}
        for op in self.operations:
            if isinstance(op, EpisodeFilter):
                keep_ids &= set(op.keep([e for e in episodes if e.episode_id in keep_ids], ctx))
        kept_episodes = [e for e in episodes if e.episode_id in keep_ids]
        dropped_ids = {e.episode_id for e in episodes} - keep_ids

        frame_filters = [op for op in self.operations if isinstance(op, FrameFilter)]
        column_ops = [
            op for op in self.operations
            if isinstance(op, FrameTransform) and not op.deferred
        ]
        deferred_ops = [op for op in self.operations if op.deferred]

        ep_filter = (
            pads.field("episode_id").isin(sorted(keep_ids))
            if len(keep_ids) < len(episodes)
            else None
        )

        def source_batches() -> Iterator[pa.RecordBatch]:
            """Fresh iterator over the input frames, episode-filtered."""
            yield from src_tables.iter_batches(
                "frames", filter=ep_filter, batch_size=self.settings.batch_size
            )

        def _apply_frame_filters(batch: pa.RecordBatch) -> pa.RecordBatch | None:
            for f in frame_filters:
                m = f.mask(batch, ctx)
                batch = batch.filter(m)
                if batch.num_rows == 0:
                    return None
            return batch

        def fit_batches() -> Iterator[pa.RecordBatch]:
            """Episode- and row-filtered iterator, used only for fitting.

            Fitting on ``source_batches()`` directly would let a FrameFilter's
            removed rows still influence any fitted statistic (e.g. a
            NormalizeVector mean/std computed over rows DropNonFiniteFrames
            would later drop) -- silently contradicting the operator list's
            declared order. Row filters are applied here as a whole rather
            than replayed up to each fitting operator's exact position, which
            is exact for the current operator set (no shipped operator both
            needs a fit and must see a partially-filtered stream from a
            frame filter declared after it).
            """
            for batch in source_batches():
                filtered = _apply_frame_filters(batch)
                if filtered is not None:
                    yield filtered

        # -- pass 1: fit ----------------------------------------------------
        for op in self.operations:
            if op.needs_fit:
                op.fit(ctx, fit_batches())

        frames_in = src_tables.count_rows("frames")
        frames_out = 0

        # -- pass 2: apply and write ---------------------------------------
        def transformed() -> Iterator[pa.RecordBatch]:
            """Apply row filters then column operators, yielding batches to the writer."""
            nonlocal frames_out
            for batch in source_batches():
                batch = _apply_frame_filters(batch)
                if batch is None:
                    continue
                for op in column_ops:
                    batch = op.apply_batch(batch, ctx)
                frames_out += batch.num_rows
                yield batch

        # Output schema derived by pushing an *empty* batch through the column
        # operators.  Needed because a filter may legitimately match nothing, and
        # an empty result must still produce a readable, correctly-typed table
        # rather than crashing the run or leaving a missing file.
        out_schema = self._output_schema(src_tables, column_ops, ctx)

        vm = VersionManager(self.catalog)
        with vm.run(
            dataset_id=dataset_id,
            operation="transform",
            output_version=output_version,
            input_versions=[input_version],
            parameters={
                "operations": [op.describe() for op in self.operations],
                "row_group_size": self.settings.row_group_size,
            },
            fingerprint=self._fingerprint(),
        ) as run:
            dst_tables.write_batches("frames", transformed(), schema=out_schema)

            # Episode records are rewritten with recomputed frame ranges: after
            # row filtering the old global offsets no longer describe the data.
            counts = self._recount(dst_tables)
            cursor = 0
            new_eps: list[Episode] = []
            for e in kept_episodes:
                n = counts.get(e.episode_id, 0)
                if n == 0:
                    # Every frame was removed by a row-level FrameFilter (the
                    # episode-level filters above already let it through).
                    # Record it as dropped so episodes_in - episodes_out
                    # reconciles with len(dropped_episode_ids).
                    dropped_ids.add(e.episode_id)
                    continue
                ne = e.model_copy(deep=True)
                ne.frame_start, ne.frame_end = cursor, cursor + n
                ne.end_timestamp_s = n / (ctx.manifest.fps or 30.0)
                ne.attributes["source_episode_id"] = e.episode_id
                ne.attributes["source_version"] = input_version
                new_eps.append(ne)
                cursor += n
            dst_tables.write_table("episodes", episodes_to_table(new_eps))

            out_manifest = ctx.manifest
            out_manifest.version = output_version
            out_manifest.source = f"{dataset_id}@{input_version}"
            out_manifest.total_episodes = len(new_eps)
            out_manifest.total_frames = frames_out
            out_manifest.quality_status = QualityStatus.UNKNOWN
            for op in self.operations:
                op.update_manifest(out_manifest, ctx)
            out_manifest.statistics = {
                "frame_table": dst_tables.stats("frames"),
                "fitted": ctx.fitted,
                "deferred_operations": [op.describe() for op in deferred_ops],
            }

            dropped = sorted(dropped_ids)
            run.input_episode_ids = [e.episode_id for e in episodes]
            run.output_episode_ids = [e.episode_id for e in new_eps]
            run.dropped_episode_ids = dropped
            run.metrics = {
                "frames_in": frames_in, "frames_out": frames_out,
                "episodes_in": len(episodes), "episodes_out": len(new_eps),
            }

        self.catalog.register_version(out_manifest, new_eps)
        art = self.settings.paths.artifact_dir(dataset_id, output_version)
        art.mkdir(parents=True, exist_ok=True)
        (art / "manifest.json").write_text(out_manifest.model_dump_json(indent=2))

        return PipelineResult(
            dataset_id=dataset_id,
            input_version=input_version,
            output_version=output_version,
            operations=[op.describe() for op in self.operations],
            episodes_in=len(episodes),
            episodes_out=len(new_eps),
            frames_in=frames_in,
            frames_out=frames_out,
            dropped_episodes=dropped,
            fitted=dict(ctx.fitted),
        )


    @staticmethod
    def _output_schema(src_tables, column_ops, ctx) -> pa.Schema:
        """Schema the pipeline will emit, computed without reading any rows.

        Running the operators over a zero-row batch is exact -- it exercises the
        same code that would rewrite real columns -- and costs nothing.
        """
        schema = src_tables.schema("frames")
        empty = pa.RecordBatch.from_pylist([], schema=schema)
        for op in column_ops:
            empty = op.apply_batch(empty, ctx)
        return empty.schema

    @staticmethod
    def _recount(tables: ParquetTableStore) -> dict[int, int]:
        """Frames per episode in the written table, streamed."""
        counts: dict[int, int] = {}
        for batch in tables.iter_batches("frames", columns=["episode_id"], batch_size=8192):
            ids, n = np.unique(np.asarray(batch.column(0)), return_counts=True)
            for i, c in zip(ids.tolist(), n.tolist()):
                counts[int(i)] = counts.get(int(i), 0) + int(c)
        return counts
