"""Source adapter contract.

An adapter is the *only* place in the engine allowed to know a source format's
quirks.  Its job is to answer four questions about a dataset on disk and then get
out of the way:

1. What is in it?            -> :meth:`SourceAdapter.inspect`
2. What is its schema?       -> :meth:`SourceAdapter.build_manifest`
3. What are its episodes?    -> :meth:`SourceAdapter.iter_episodes`
4. What are its frames?      -> :meth:`SourceAdapter.iter_frame_batches`

Frames are yielded as Arrow ``RecordBatch``es rather than :class:`Frame` objects.
Constructing 19 631 pydantic models to immediately serialise them back to Parquet
would cost far more than it buys, so the canonical *logical* model is validated at
the boundary (schema construction, episode records) while the bulk path stays
columnar.  :class:`~rde.schemas.core.Frame` remains the interchange type for
single-frame access in exploration and model adapters.

Adding a new source format means implementing this interface -- nothing
downstream of ingestion changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa

from ..schemas.core import DatasetManifest, Episode


@dataclass
class SourceProfile:
    """What an inspection pass learned, before anything is converted.

    Produced by :meth:`SourceAdapter.inspect` and rendered by the CLI, so a user
    can understand an unfamiliar dataset without ingesting it.  ``anomalies``
    carries structural surprises found during inspection -- they become QC
    findings later rather than being silently normalised away here.
    """

    source_format: str
    source_uri: str
    root: Path
    robot_type: str | None = None
    fps: float | None = None
    total_episodes: int = 0
    total_frames: int = 0
    total_tasks: int = 0
    features: dict[str, Any] = field(default_factory=dict)
    files: dict[str, Any] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-safe view for the CLI and run records."""
        return {
            "source_format": self.source_format,
            "source_uri": self.source_uri,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "episodes": self.total_episodes,
            "frames": self.total_frames,
            "tasks": self.total_tasks,
            "features": list(self.features),
            "anomalies": self.anomalies,
        }


class SourceAdapter(ABC):
    """Convert one external dataset layout into the canonical model."""

    #: Stable identifier recorded in manifests and lineage, e.g. ``'lerobot/v3.0'``.
    source_format: str = "unknown"

    def __init__(self, root: Path, dataset_id: str, source_uri: str | None = None):
        self.root = Path(root)
        self.dataset_id = dataset_id
        self.source_uri = source_uri or f"file://{self.root}"

    @abstractmethod
    def inspect(self) -> SourceProfile:
        """Read metadata only; must not decode video or scan every frame."""

    @abstractmethod
    def build_manifest(self, version: str) -> DatasetManifest:
        """Produce the canonical manifest, including the modality registry."""

    @abstractmethod
    def iter_episodes(self) -> Iterator[Episode]:
        """Yield canonical episode records in ascending episode id order."""

    @abstractmethod
    def iter_frame_batches(self, batch_size: int = 2048) -> Iterator[pa.RecordBatch]:
        """Stream canonical frame rows.  Must not materialise the whole table."""

    @abstractmethod
    def register_blobs(self, blob_store) -> dict[str, str]:
        """Admit binary payloads into the blob store; return modality -> blob id."""
