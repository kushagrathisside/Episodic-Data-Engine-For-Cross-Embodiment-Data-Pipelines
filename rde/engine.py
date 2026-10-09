"""The public interface of the data engine.

Everything above this line is data platform; everything below it is a consumer.
The contract is deliberately narrow::

    engine = DataEngine()
    ds = engine.get_dataset("so100_pickplace", version="training/v1")
    for sample in ds.iter_samples(split="train"):
        ...

A consumer receives a :class:`DatasetView`: manifest, training contract, episode
index, streaming frame access and lazy image decode.  It never learns where the
bytes live, what format they arrived in, or which transformations produced them.
That is the property that lets the model be swapped without touching ingestion,
storage, QC or versioning.

Laziness is the other design rule here.  A :class:`Sample` carries *references*
to images, not pixels; calling :meth:`Sample.image` decodes on demand.  A training
loop that only needs state and action therefore pays nothing for the cameras, and
one that needs images decodes exactly the frames it asks for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

from .catalog import Catalog
from .config import Settings, get_settings
from .schemas.core import DatasetManifest, Episode, QualityStatus
from .schemas.lineage import VersionStage
from .storage import BlobStore, ParquetTableStore, VideoReader


@dataclass
class Sample:
    """One timestep, as handed to a model adapter.

    Images are not decoded until requested -- see :meth:`image`.
    """

    frame_id: int
    episode_id: int
    frame_index: int
    timestamp_s: float
    task_id: int | None
    task: str | None
    observation: dict[str, np.ndarray]
    action: np.ndarray | None
    image_refs: dict[str, tuple[str, float]] = field(default_factory=dict)
    _view: "DatasetView | None" = field(default=None, repr=False)

    def image(self, modality: str, apply_read_transforms: bool = True) -> np.ndarray | None:
        """Decode one camera view for this frame.

        ``apply_read_transforms`` applies the deferred operators the manifest
        declares (resize, scaling) -- the ones the transformation layer chose not
        to materialise.  See :class:`~rde.transformations.ops.ResizeImages`.
        """
        if self._view is None or modality not in self.image_refs:
            return None
        blob_id, ts = self.image_refs[modality]
        return self._view.read_image(
            blob_id, ts, modality=modality, apply_read_transforms=apply_read_transforms
        )

    @property
    def image_modalities(self) -> list[str]:
        """Camera streams available on this sample, still undecoded."""
        return sorted(self.image_refs)


class DatasetView:
    """Read-only handle on one dataset version."""

    def __init__(
        self,
        manifest: DatasetManifest,
        episodes: list[Episode],
        frame_table: ParquetTableStore,
        blob_store: BlobStore,
        settings: Settings,
    ):
        self.manifest = manifest
        self.episodes = episodes
        self._frames = frame_table
        self._blobs = blob_store
        self._settings = settings
        self._readers: dict[str, VideoReader] = {}
        self._tasks = {t.task_id: t.description for t in manifest.tasks}

    # -- identity -----------------------------------------------------------
    @property
    def dataset_id(self) -> str:
        """Dataset this view belongs to."""
        return self.manifest.dataset_id

    @property
    def version(self) -> str:
        """Version this view resolves, e.g. 'training/v1'."""
        return self.manifest.version

    @property
    def contract(self) -> dict[str, Any]:
        """Model-facing schema contract; the interface a model adapter reads."""
        c = self.manifest.attributes.get("training_contract")
        if c:
            return c
        from .dataset_builder.builder import build_training_contract

        return build_training_contract(self.manifest)

    @property
    def fps(self) -> float:
        """Sampling rate declared by the manifest, defaulting to 30."""
        return self.manifest.fps or 30.0

    def __repr__(self) -> str:
        return (
            f"<DatasetView {self.dataset_id}@{self.version} "
            f"episodes={len(self.episodes)} frames={self.manifest.total_frames}>"
        )

    # -- episode access -----------------------------------------------------
    def episode_ids(self, split: str | None = None, task_id: int | None = None) -> list[int]:
        """Episode ids, optionally narrowed to one split or task."""
        out = []
        for e in self.episodes:
            if split is not None and e.attributes.get("split") != split:
                continue
            if task_id is not None and e.task_id != task_id:
                continue
            out.append(e.episode_id)
        return out

    def episode(self, episode_id: int) -> Episode | None:
        """One episode record by id, or None."""
        return next((e for e in self.episodes if e.episode_id == episode_id), None)

    def splits(self) -> dict[str, int]:
        """Episode count per split name."""
        counts: dict[str, int] = {}
        for e in self.episodes:
            s = e.attributes.get("split", "unassigned")
            counts[s] = counts.get(s, 0) + 1
        return counts

    # -- frame access -------------------------------------------------------
    def iter_frames(
        self,
        split: str | None = None,
        episode_ids: Sequence[int] | None = None,
        columns: Sequence[str] | None = None,
        batch_size: int | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Stream raw frame batches, filtered to a split or explicit episodes."""
        ids = list(episode_ids) if episode_ids is not None else self.episode_ids(split=split)
        flt = pads.field("episode_id").isin(ids) if ids else None
        yield from self._frames.iter_batches(
            "frames",
            columns=columns,
            filter=flt,
            batch_size=batch_size or self._settings.batch_size,
        )

    def iter_samples(
        self,
        split: str | None = None,
        episode_ids: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> Iterator[Sample]:
        """Stream :class:`Sample` objects; images stay undecoded until asked for."""
        obs_keys = [
            m.name for m in self.manifest.modalities
            if not m.is_external and m.kind.value != "robot_action"
        ]
        act_key = next(
            (m.name for m in self.manifest.modalities if m.kind.value == "robot_action"), None
        )
        img_keys = [m.name for m in self.manifest.modalities if m.is_external]

        for batch in self.iter_frames(split=split, episode_ids=episode_ids, batch_size=batch_size):
            t = pa.Table.from_batches([batch])
            d = t.to_pydict()
            n = t.num_rows
            for i in range(n):
                obs = {
                    k: np.asarray(d[k][i], dtype=np.float32)
                    for k in obs_keys if k in d and d[k][i] is not None
                }
                act = (
                    np.asarray(d[act_key][i], dtype=np.float32)
                    if act_key and act_key in d and d[act_key][i] is not None
                    else None
                )
                refs = {}
                for k in img_keys:
                    bid, ts = f"{k}__blob_id", f"{k}__ts_in_blob"
                    if bid in d and d[bid][i] and d[ts][i] is not None:
                        refs[k] = (d[bid][i], float(d[ts][i]))
                tid = d.get("task_id", [None] * n)[i]
                yield Sample(
                    frame_id=int(d["frame_id"][i]),
                    episode_id=int(d["episode_id"][i]),
                    frame_index=int(d["frame_index"][i]),
                    timestamp_s=float(d["timestamp_s"][i]),
                    task_id=int(tid) if tid is not None else None,
                    task=self._tasks.get(int(tid)) if tid is not None else None,
                    observation=obs,
                    action=act,
                    image_refs=refs,
                    _view=self,
                )

    def get_episode_frames(self, episode_id: int, columns: Sequence[str] | None = None) -> pa.Table:
        """Whole-episode read.  Bounded by episode length, not dataset size."""
        return self._frames.read_table(
            "frames", columns=columns, filter=pads.field("episode_id") == episode_id
        )

    # -- image access -------------------------------------------------------
    def read_image(
        self,
        blob_id: str,
        timestamp_s: float,
        modality: str | None = None,
        apply_read_transforms: bool = True,
    ) -> np.ndarray | None:
        """Decode one frame, applying deferred read-time transforms."""
        reader = self._readers.get(blob_id)
        if reader is None:
            try:
                path = self._blobs.get(blob_id)
            except KeyError:
                return None
            reader = VideoReader(path, cache_frames=self._settings.video_cache_frames)
            self._readers[blob_id] = reader
        img = reader.read_at(timestamp_s)
        if img is None or not apply_read_transforms or modality is None:
            return img
        return self._apply_read_transforms(img, modality)

    def _apply_read_transforms(self, img: np.ndarray, modality: str) -> np.ndarray:
        spec = self.manifest.modality(modality)
        if spec is None:
            return img
        for op in spec.attributes.get("read_transforms", []) or []:
            if op.get("op") == "resize":
                img = _resize_rgb(img, int(op["height"]), int(op["width"]))
            elif op.get("op") == "normalize":
                mean = np.asarray(op.get("mean", [0, 0, 0]), dtype=np.float32)
                std = np.asarray(op.get("std", [255, 255, 255]), dtype=np.float32)
                img = (img.astype(np.float32) - mean) / np.where(std == 0, 1.0, std)
        return img

    def close(self):
        """Release every open video decoder held by this view.

        Closes every reader unconditionally, even when one raises -- a
        corrupted/truncated container failing on ``container.close()`` used
        to abort the loop immediately, leaking every reader ordered after it.
        """
        errors: list[BaseException] = []
        for r in self._readers.values():
            try:
                r.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        self._readers.clear()
        if errors:
            raise RuntimeError(
                f"{len(errors)} video reader(s) failed to close cleanly: "
                f"{[str(e) for e in errors]}"
            ) from errors[0]

    def __enter__(self) -> "DatasetView":
        return self

    def __exit__(self, *exc):
        self.close()


def _resize_rgb(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Area-average resize with a pure-NumPy fallback.

    Uses PIL when available for quality; otherwise strided sampling, which is
    adequate for the deferred-resize demonstration and keeps the engine free of a
    hard imaging dependency.
    """
    if img.shape[0] == height and img.shape[1] == width:
        return img
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(img).resize((width, height), Image.BILINEAR))
    except ImportError:
        ys = (np.arange(height) * (img.shape[0] / height)).astype(np.int64)
        xs = (np.arange(width) * (img.shape[1] / width)).astype(np.int64)
        return img[np.ix_(ys, xs)]


class DataEngine:
    """Entry point: resolve dataset versions into consumable views."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.paths.ensure()
        self.catalog = Catalog(self.settings.paths.catalog_db)

    # -- discovery ----------------------------------------------------------
    def list_datasets(self) -> list[dict[str, Any]]:
        """Every dataset registered in the catalog."""
        return self.catalog.list_datasets()

    def list_versions(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        """Versions, optionally scoped to one dataset."""
        return self.catalog.list_versions(dataset_id)

    def latest_version(self, dataset_id: str, stage: VersionStage | str) -> str:
        """Most recently created version at a given stage."""
        stage = VersionStage(stage) if isinstance(stage, str) else stage
        versions = [v for v in self.catalog.list_versions(dataset_id) if v["stage"] == stage.value]
        if not versions:
            raise LookupError(f"no {stage.value} version for {dataset_id}")
        return sorted(versions, key=lambda v: v["created_at"])[-1]["version"]

    # -- access -------------------------------------------------------------
    def get_dataset(self, dataset_id: str, version: str = "training/v1") -> DatasetView:
        """Resolve a dataset version into a read-only view.

        This is the call the training layer makes.  It takes a *version*, never a
        path, which is what keeps the physical layout free to change -- and what
        makes a training run reproducible from its recorded version id alone.
        """
        manifest = self.catalog.get_manifest(dataset_id, version)
        if manifest is None:
            raise LookupError(
                f"unknown version {dataset_id}@{version}; "
                f"available: {[v['version'] for v in self.catalog.list_versions(dataset_id)]}"
            )

        # A training version references its source's frame table rather than
        # duplicating it; follow that pointer when present.
        frame_version = manifest.statistics.get("frame_table_version", version)
        frames = ParquetTableStore(
            self.settings.paths.version_dir(dataset_id, frame_version),
            compression=self.settings.parquet_compression,
            row_group_size=self.settings.row_group_size,
        )

        rows = self.catalog.get_episodes(dataset_id, version)
        own_tables = ParquetTableStore(self.settings.paths.version_dir(dataset_id, version))
        extra: dict[int, dict] = {}
        if own_tables.exists("episodes"):
            t = own_tables.read_table("episodes").to_pydict()
            extra = {
                int(e): {"blob_ids": json.loads(b or "{}"), "attributes": json.loads(a or "{}")}
                for e, b, a in zip(t["episode_id"], t["blob_ids"], t["attributes"])
            }
        episodes = []
        for r in rows:
            x = extra.get(r["episode_id"], {})
            episodes.append(
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

        return DatasetView(
            manifest=manifest,
            episodes=episodes,
            frame_table=frames,
            blob_store=BlobStore(self.settings.paths.blobs),
            settings=self.settings,
        )
