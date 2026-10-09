"""Ingestion orchestrator: source on disk -> canonical ``raw/vN``.

This is the seam where an external format stops existing.  After ``ingest``
returns, every downstream stage reads canonical Parquet, canonical episode
records and a canonical manifest; none of them can tell the data came from
LeRobot.

The pass is single-threaded and streaming by construction: frames flow
adapter -> Arrow batch -> Parquet writer without ever assembling a full table.
That is what keeps a dataset larger than RAM ingestible on this machine.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..catalog import Catalog
from ..config import Settings, get_settings
from ..schemas.core import DatasetManifest, Episode
from ..schemas.lineage import VersionStage
from ..storage import BlobStore, ParquetTableStore
from ..versioning import VersionManager, code_fingerprint
from .base import SourceAdapter, SourceProfile
from .lerobot import LeRobotAdapter

#: Registry of known source formats; extend here to support a new dataset family.
ADAPTERS: dict[str, type[SourceAdapter]] = {"lerobot": LeRobotAdapter}


@dataclass
class IngestResult:
    """Outcome of one ingestion: manifest, counts, blobs, table geometry."""
    manifest: DatasetManifest
    profile: SourceProfile
    version: str
    frames_written: int
    episodes_written: int
    blob_ids: dict[str, str]
    table_stats: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Compact result for the CLI and the pipeline run record."""
        return {
            "dataset_id": self.manifest.dataset_id,
            "version": self.version,
            "source_format": self.manifest.source_format,
            "episodes": self.episodes_written,
            "frames": self.frames_written,
            "modalities": self.manifest.modality_names,
            "blobs": len(self.blob_ids),
            "table_bytes": self.table_stats.get("bytes"),
            "row_groups": self.table_stats.get("row_groups"),
            "anomalies": self.profile.anomalies,
        }


def episodes_to_table(episodes: list[Episode]) -> pa.Table:
    """Serialise canonical episodes to the version's episode table.

    ``attributes`` and ``blob_ids`` are JSON-encoded rather than nested Arrow
    structs: their shape varies by source (video windows exist for LeRobot, not
    for a future depth-only dataset), and a stable flat schema is worth more here
    than queryable nesting for a 50-row table.
    """
    return pa.table(
        {
            "episode_id": pa.array([e.episode_id for e in episodes], pa.int64()),
            "task_id": pa.array([e.task_id for e in episodes], pa.int64()),
            "frame_start": pa.array([e.frame_start for e in episodes], pa.int64()),
            "frame_end": pa.array([e.frame_end for e in episodes], pa.int64()),
            "num_frames": pa.array([e.num_frames for e in episodes], pa.int64()),
            "start_timestamp_s": pa.array([e.start_timestamp_s for e in episodes], pa.float64()),
            "end_timestamp_s": pa.array([e.end_timestamp_s for e in episodes], pa.float64()),
            "duration_s": pa.array([e.duration_s for e in episodes], pa.float64()),
            "outcome": pa.array([e.outcome.value for e in episodes], pa.string()),
            "quality_status": pa.array([e.quality_status.value for e in episodes], pa.string()),
            "modalities": pa.array([json.dumps(e.modalities) for e in episodes], pa.string()),
            "blob_ids": pa.array([json.dumps(e.blob_ids) for e in episodes], pa.string()),
            "attributes": pa.array([json.dumps(e.attributes) for e in episodes], pa.string()),
        }
    )


def ingest(
    source_root: Path,
    dataset_id: str,
    source_format: str = "lerobot",
    version: str = "raw/v1",
    source_uri: str | None = None,
    settings: Settings | None = None,
    catalog: Catalog | None = None,
) -> IngestResult:
    """Convert a source dataset into canonical storage and register it."""
    settings = settings or get_settings()
    settings.paths.ensure()
    catalog = catalog or Catalog(settings.paths.catalog_db)

    adapter_cls = ADAPTERS.get(source_format)
    if adapter_cls is None:
        raise ValueError(
            f"unknown source_format {source_format!r}; known: {sorted(ADAPTERS)}"
        )
    adapter = adapter_cls(Path(source_root), dataset_id, source_uri)

    profile = adapter.inspect()
    blob_store = BlobStore(settings.paths.blobs)
    # Blobs must be registered before frames: frame rows carry blob ids.
    blob_ids = adapter.register_blobs(blob_store)

    manifest = adapter.build_manifest(version)
    episodes = list(adapter.iter_episodes())

    # Written to a staging directory, not the final version_dir directly: a
    # generator raising partway through iter_frame_batches (malformed source
    # data) previously left a real, partial "frames/part-000.parquet" sitting
    # at the version's canonical path with no "episodes" table and no
    # manifest -- indistinguishable from a completed ingest to any caller
    # that only checks ParquetTableStore(...).exists("frames"). The staging
    # dir is only ever renamed into place after every write below succeeds,
    # so the canonical path is either absent or complete, never partial.
    final_dir = settings.paths.version_dir(dataset_id, version)
    staging_dir = final_dir.parent / f".{final_dir.name}.staging-{uuid.uuid4().hex[:8]}"
    tables = ParquetTableStore(
        staging_dir,
        compression=settings.parquet_compression,
        row_group_size=settings.row_group_size,
    )

    vm = VersionManager(catalog)
    with vm.run(
        dataset_id=dataset_id,
        operation="ingest",
        output_version=version,
        input_versions=[],
        parameters={
            "source_format": profile.source_format,
            "source_uri": adapter.source_uri,
            "row_group_size": settings.row_group_size,
            "compression": settings.parquet_compression,
        },
        fingerprint=code_fingerprint(adapter_cls.iter_frame_batches),
    ) as run:
        frames_written = 0

        def counting_batches():
            """Wrap the adapter's batches, tallying rows as they stream past."""
            nonlocal frames_written
            for b in adapter.iter_frame_batches(batch_size=settings.batch_size):
                frames_written += b.num_rows
                yield b

        tables.write_batches(
            "frames", counting_batches(), schema=adapter.canonical_frame_schema()
        )
        tables.write_table("episodes", episodes_to_table(episodes))

        manifest.total_frames = frames_written
        manifest.total_episodes = len(episodes)
        stats = tables.stats("frames")
        manifest.statistics = {
            "frame_table": stats,
            "episode_table": tables.stats("episodes"),
            "blob_bytes": sum(
                blob_store.info(b).size_bytes
                for b in blob_ids.values()
                if blob_store.info(b)
            ),
            "source_bytes": profile.files.get("video_bytes", {}),
        }
        run.output_episode_ids = [e.episode_id for e in episodes]
        run.metrics = {"frames": frames_written, "episodes": len(episodes)}

        # Publish: rename the complete staging directory into place. Kept
        # inside the `with` block so a failure here is captured as a run
        # failure too, rather than a silent gap between "run recorded as ok"
        # and "the data is actually where the catalog says it is". Two
        # concurrent ingests of the same version now race only on this
        # single atomic rename (last one to finish wins outright) rather
        # than on interleaved writes to the same Parquet file.
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            # Re-ingesting the same version is an intended idempotent path
            # (Catalog.register_version's own upsert is designed around
            # exactly this); replace rather than error.
            shutil.rmtree(final_dir)
        os.replace(staging_dir, final_dir)

    catalog.register_version(manifest, episodes)

    # The manifest is also written beside the data so a version directory is
    # self-describing even if the catalog is lost or rebuilt elsewhere.
    art = settings.paths.artifact_dir(dataset_id, version)
    art.mkdir(parents=True, exist_ok=True)
    (art / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (art / "source_profile.json").write_text(json.dumps(profile.summary(), indent=2, default=str))

    return IngestResult(
        manifest=manifest,
        profile=profile,
        version=version,
        frames_written=frames_written,
        episodes_written=len(episodes),
        blob_ids=blob_ids,
        table_stats=stats,
    )
