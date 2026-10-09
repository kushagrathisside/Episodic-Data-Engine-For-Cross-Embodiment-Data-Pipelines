"""Canonical dataset / episode / frame model.

These types are the *contract* between every stage of the pipeline.  Ingestion
adapters produce them, validation reads them, transformations rewrite them, the
dataset builder materialises them, and model adapters consume them.  Nothing
downstream of ingestion is allowed to know that LeRobot exists.

Design note -- the Python objects here are the *logical* schema.  The physical
Parquet schema is derived from them in :mod:`rde.storage.tabular`, so the two
cannot silently drift.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .modality import BlobRef, ModalitySpec

#: Bumped whenever the canonical schema changes shape.  Persisted into every
#: dataset manifest so an old store can be recognised and migrated rather than
#: misread.
SCHEMA_VERSION = "1.0.0"


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class QualityStatus(str, Enum):
    """Where an entity stands with respect to the QC gate."""

    UNKNOWN = "unknown"      # not yet assessed
    PASSED = "passed"        # all checks passed
    WARNING = "warning"      # non-blocking findings present
    REJECTED = "rejected"    # blocking findings; excluded from clean datasets
    QUARANTINED = "quarantined"  # held for human review (feedback loop)


class EpisodeOutcome(str, Enum):
    """Task-level result of the demonstration itself, not of data quality."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class Task(BaseModel):
    """A natural-language goal shared by many episodes."""

    task_id: int
    description: str
    source_key: str | None = None


class Frame(BaseModel):
    """One synchronised timestep.

    A frame is the join point of every modality at a single instant.  Inline
    numeric modalities are carried in ``observation`` / ``action``; heavy
    modalities appear as :class:`BlobRef` entries in ``modality_refs``.
    """

    frame_id: int = Field(..., description="Global row id, unique within a dataset.")
    episode_id: int
    frame_index: int = Field(..., description="0-based position within the episode.")
    timestamp_s: float = Field(..., description="Seconds from episode start.")
    task_id: int | None = None

    # Inline numeric planes, keyed by canonical modality name.
    observation: dict[str, list[float]] = Field(default_factory=dict)
    action: dict[str, list[float]] = Field(default_factory=dict)

    # External payload pointers, keyed by canonical modality name.
    modality_refs: dict[str, BlobRef] = Field(default_factory=dict)

    quality_status: QualityStatus = QualityStatus.UNKNOWN

    @field_validator("timestamp_s")
    @classmethod
    def _finite_timestamp(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("timestamp must be finite")
        return v

    @property
    def modalities_present(self) -> set[str]:
        """Every modality this frame actually carries, inline or external."""
        return set(self.observation) | set(self.action) | set(self.modality_refs)


class Episode(BaseModel):
    """A contiguous demonstration: one attempt at one task."""

    episode_id: int
    dataset_id: str
    task_id: int | None = None

    # Frame range is a half-open interval [start, end) into the dataset's global
    # frame index, which lets the builder slice Parquet without opening videos.
    frame_start: int
    frame_end: int

    start_timestamp_s: float = 0.0
    end_timestamp_s: float = 0.0

    outcome: EpisodeOutcome = EpisodeOutcome.UNKNOWN
    quality_status: QualityStatus = QualityStatus.UNKNOWN

    source_dataset: str | None = None
    modalities: list[str] = Field(default_factory=list)
    #: Per-modality blob ids covering this episode (video clips etc.).
    blob_ids: dict[str, str] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        """Frame count; the range is half-open, so end - start."""
        return self.frame_end - self.frame_start

    @property
    def duration_s(self) -> float:
        """Wall-clock length of the demonstration, clamped at zero."""
        return max(0.0, self.end_timestamp_s - self.start_timestamp_s)

    @field_validator("frame_end")
    @classmethod
    def _range_is_ordered(cls, v: int, info) -> int:
        start = info.data.get("frame_start")
        if start is not None and v < start:
            raise ValueError(f"frame_end ({v}) precedes frame_start ({start})")
        return v


class DatasetManifest(BaseModel):
    """Immutable description of one materialised dataset version.

    Every version in the store has exactly one manifest.  It is the object the
    catalog indexes, the lineage graph points at, and the training layer resolves
    when it asks for ``training/v1``.
    """

    dataset_id: str
    version: str = Field(..., description="Namespaced version, e.g. 'clean/v1'.")
    schema_version: str = SCHEMA_VERSION

    source: str = Field(..., description="Origin URI or upstream version id.")
    source_format: str | None = Field(
        default=None, description="e.g. 'lerobot/v3.0' for ingested roots."
    )
    created_at: _dt.datetime = Field(default_factory=_utcnow)

    total_episodes: int = 0
    total_frames: int = 0
    total_tasks: int = 0

    fps: float | None = None
    robot_type: str | None = None

    modalities: list[ModalitySpec] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)

    #: Set by the QC engine once the version has been gated.
    quality_status: QualityStatus = QualityStatus.UNKNOWN
    #: Free-form counters (bytes on disk, rejected episode count, split sizes...).
    statistics: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def version_key(self) -> str:
        """Fully-qualified identity used as the catalog primary key."""
        return f"{self.dataset_id}@{self.version}"

    @property
    def modality_names(self) -> list[str]:
        """Canonical names of every declared stream, in declaration order."""
        return [m.name for m in self.modalities]

    def modality(self, name: str) -> ModalitySpec | None:
        """Look up one declared modality by canonical name, or None."""
        for m in self.modalities:
            if m.name == name:
                return m
        return None

    def modalities_of_kind(self, kind) -> list[ModalitySpec]:
        """Every declared modality of a given ModalityKind."""
        return [m for m in self.modalities if m.kind == kind]
