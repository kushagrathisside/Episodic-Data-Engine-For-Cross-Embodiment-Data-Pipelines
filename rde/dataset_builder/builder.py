"""Dataset builder: turn a clean version into a training-ready one.

The builder is the last stage that owns data.  Everything after it -- model
adapters, training, evaluation -- consumes a *dataset version* through a stable
interface and never touches files.

What it adds on top of a clean version:

**Splits.**  Assigned by episode, never by frame, and by a stable hash rather
than a shuffle.  See :func:`~rde.transformations.ops.assign_splits` for why both
choices matter.

**A training contract.**  The manifest records the exact input/output shapes,
normalization statistics, deferred read-time transforms and split sizes.  A model
adapter reads that contract instead of rediscovering the schema, which is what
lets a different model be swapped in without touching the data layer.

**An index.**  A compact per-episode table (episode id, split, frame range, task)
that a sampler can load entirely into memory even when the frame table cannot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pyarrow as pa

from ..catalog import Catalog
from ..config import Settings, get_settings
from ..ingestion.ingest import episodes_to_table
from ..schemas.core import DatasetManifest, Episode, QualityStatus
from ..schemas.modality import ModalityKind, StoragePlane
from ..storage import ParquetTableStore
from ..transformations.ops import assign_splits
from ..versioning import VersionManager, code_fingerprint


@dataclass
class BuildResult:
    """Outcome of building one training version: splits, contract, artifacts."""
    dataset_id: str
    input_version: str
    output_version: str
    split_sizes: dict[str, dict[str, int]]
    contract: dict[str, Any]
    episodes: int
    frames: int
    artifacts: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Split sizes and the observation/action keys the contract exposes."""
        return {
            "dataset_id": self.dataset_id,
            "input_version": self.input_version,
            "output_version": self.output_version,
            "episodes": self.episodes,
            "frames": self.frames,
            "splits": self.split_sizes,
            "observation_keys": list(self.contract.get("observation", {})),
            "action_key": self.contract.get("action", {}).get("key"),
        }


def build_training_contract(manifest: DatasetManifest) -> dict[str, Any]:
    """Derive the model-facing contract from the canonical manifest.

    This is the interface boundary of the whole system.  It states, in terms no
    model-specific code has to guess at: which streams are observations, which is
    the action, their shapes and dtypes, what normalization was applied, and what
    transforms remain to be applied at read time.
    """
    observation: dict[str, Any] = {}
    action: dict[str, Any] = {}

    for m in manifest.modalities:
        raw_names = m.attributes.get("element_names")
        entry = {
            "key": m.name,
            "kind": m.kind.value,
            "dtype": m.dtype,
            "shape": list(m.shape),
            "plane": m.plane.value,
            "fps": m.fps,
            "element_names": (
                [n for group in raw_names.values() for n in group]
                if isinstance(raw_names, dict) else raw_names
            ),
            "normalization": m.attributes.get("normalization"),
            "read_transforms": m.attributes.get("read_transforms", []),
        }
        if m.kind is ModalityKind.ROBOT_ACTION:
            action = entry
        else:
            observation[m.name] = entry

    return {
        "schema_version": manifest.schema_version,
        "dataset_id": manifest.dataset_id,
        "version": manifest.version,
        "fps": manifest.fps,
        "robot_type": manifest.robot_type,
        "observation": observation,
        "action": action,
        "tasks": [{"task_id": t.task_id, "description": t.description} for t in manifest.tasks],
        "language_conditioned": len(manifest.tasks) > 0,
    }


class DatasetBuilder:
    """Produces ``training/vN`` from a clean version."""

    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None):
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.paths.catalog_db)

    def build(
        self,
        dataset_id: str,
        input_version: str,
        output_version: str,
        splits: dict[str, float] | None = None,
        seed: int = 0,
        require_quality: Sequence[QualityStatus] | None = None,
        split_mode: str = "quantile",
        episode_weights: dict[int, float] | None = None,
    ) -> BuildResult:
        """Materialise a training version with splits and a contract.

        ``episode_weights`` closes the deployment feedback loop: the flywheel
        nominates episodes the model failed on, and they are *upweighted* for the
        next round rather than dropped.  Weights are stored per episode and
        recorded in lineage, so "training/v2 oversampled these 5 episodes because
        the model failed on them" is a queryable fact, not a lost intention.

        Frame data is *not* rewritten: the clean version's Parquet is already in
        canonical form, and copying it would duplicate storage for no gain.  The
        training version records a pointer to its source table plus its own
        episode index and contract.  This is the same reasoning that keeps video
        blobs shared -- materialise metadata, reference bulk data.
        """
        manifest = self.catalog.get_manifest(dataset_id, input_version)
        if manifest is None:
            raise LookupError(f"unknown version: {dataset_id}@{input_version}")

        rows = self.catalog.get_episodes(dataset_id, input_version)
        # Matches QualityFilter's default elsewhere in the pipeline: UNKNOWN
        # means "never QC'd", not "passed", so it is excluded unless the
        # caller explicitly asks for it via require_quality.
        allowed = set(require_quality or [QualityStatus.PASSED, QualityStatus.WARNING])
        rows = [r for r in rows if QualityStatus(r["quality_status"]) in allowed]
        if not rows:
            raise ValueError(
                f"no episodes in {dataset_id}@{input_version} satisfy "
                f"{[s.value for s in allowed]}"
            )

        src_tables = ParquetTableStore(
            self.settings.paths.version_dir(dataset_id, input_version)
        )
        src_ep = (
            src_tables.read_table("episodes").to_pydict()
            if src_tables.exists("episodes") else {}
        )
        extra = {
            int(e): {"blob_ids": json.loads(b or "{}"), "attributes": json.loads(a or "{}")}
            for e, b, a in zip(
                src_ep.get("episode_id", []), src_ep.get("blob_ids", []),
                src_ep.get("attributes", []),
            )
        }

        splits = splits or {"train": 0.8, "val": 0.1, "test": 0.1}
        ep_ids = [r["episode_id"] for r in rows]
        assignment = assign_splits(ep_ids, splits, seed=seed, mode=split_mode)

        episodes: list[Episode] = []
        for r in rows:
            x = extra.get(r["episode_id"], {})
            attrs = dict(x.get("attributes", {}))
            attrs["split"] = assignment[r["episode_id"]]
            attrs["source_version"] = input_version
            attrs.setdefault("source_episode_id", r["episode_id"])
            if episode_weights:
                attrs["sampling_weight"] = float(
                    episode_weights.get(r["episode_id"], 1.0)
                )
            episodes.append(
                Episode(
                    episode_id=r["episode_id"], dataset_id=dataset_id, task_id=r["task_id"],
                    frame_start=r["frame_start"], frame_end=r["frame_end"],
                    end_timestamp_s=r["duration_s"],
                    quality_status=QualityStatus(r["quality_status"]),
                    modalities=json.loads(r["modalities"] or "[]"),
                    blob_ids=x.get("blob_ids", {}),
                    attributes=attrs,
                )
            )

        split_sizes: dict[str, dict[str, int]] = {}
        for name in splits:
            sel = [e for e in episodes if e.attributes["split"] == name]
            split_sizes[name] = {
                "episodes": len(sel),
                "frames": sum(e.num_frames for e in sel),
            }

        out_manifest = manifest.model_copy(deep=True)
        out_manifest.version = output_version
        out_manifest.source = f"{dataset_id}@{input_version}"
        out_manifest.total_episodes = len(episodes)
        out_manifest.total_frames = sum(e.num_frames for e in episodes)
        out_manifest.quality_status = QualityStatus.PASSED
        contract = build_training_contract(out_manifest)
        out_manifest.statistics = {
            "splits": split_sizes,
            "split_seed": seed,
            "split_fractions": splits,
            "split_mode": split_mode,
            "weighted_episodes": (
                {str(k): v for k, v in episode_weights.items()} if episode_weights else {}
            ),
            # The frame table is referenced, not copied.
            "frame_table_version": input_version,
            "frame_table_path": str(src_tables.path("frames")),
        }
        out_manifest.attributes["training_contract"] = contract

        dst_dir = self.settings.paths.version_dir(dataset_id, output_version)
        dst_tables = ParquetTableStore(
            dst_dir, compression=self.settings.parquet_compression,
            row_group_size=self.settings.row_group_size,
        )

        vm = VersionManager(self.catalog)
        with vm.run(
            dataset_id=dataset_id,
            operation="build_training_dataset",
            output_version=output_version,
            input_versions=[input_version],
            parameters={"splits": splits, "seed": seed, "split_mode": split_mode,
                        "require_quality": [s.value for s in allowed],
                        "episode_weights": (
                            {str(k): v for k, v in episode_weights.items()}
                            if episode_weights else None
                        )},
            fingerprint=code_fingerprint(DatasetBuilder.build),
        ) as run:
            dst_tables.write_table("episodes", episodes_to_table(episodes))
            dst_tables.write_table("index", self._index_table(episodes))
            run.input_episode_ids = ep_ids
            run.output_episode_ids = [e.episode_id for e in episodes]
            run.metrics = {"splits": split_sizes, "frames": out_manifest.total_frames}

        self.catalog.register_version(out_manifest, episodes)

        art = self.settings.paths.artifact_dir(dataset_id, output_version)
        art.mkdir(parents=True, exist_ok=True)
        (art / "manifest.json").write_text(out_manifest.model_dump_json(indent=2))
        (art / "training_contract.json").write_text(json.dumps(contract, indent=2))

        return BuildResult(
            dataset_id=dataset_id,
            input_version=input_version,
            output_version=output_version,
            split_sizes=split_sizes,
            contract=contract,
            episodes=len(episodes),
            frames=out_manifest.total_frames,
            artifacts={
                "manifest": str(art / "manifest.json"),
                "contract": str(art / "training_contract.json"),
            },
        )

    @staticmethod
    def _index_table(episodes: Sequence[Episode]) -> pa.Table:
        """Compact sampler index -- small enough to hold in RAM at any scale."""
        return pa.table(
            {
                "episode_id": pa.array([e.episode_id for e in episodes], pa.int64()),
                "split": pa.array([e.attributes.get("split", "train") for e in episodes]),
                "task_id": pa.array([e.task_id for e in episodes], pa.int64()),
                "frame_start": pa.array([e.frame_start for e in episodes], pa.int64()),
                "frame_end": pa.array([e.frame_end for e in episodes], pa.int64()),
                "num_frames": pa.array([e.num_frames for e in episodes], pa.int64()),
                "duration_s": pa.array([e.duration_s for e in episodes], pa.float64()),
                "sampling_weight": pa.array(
                    [float(e.attributes.get("sampling_weight", 1.0)) for e in episodes],
                    pa.float64(),
                ),
            }
        )
