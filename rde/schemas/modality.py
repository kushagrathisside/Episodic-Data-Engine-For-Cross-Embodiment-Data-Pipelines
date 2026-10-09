"""Modality definitions.

A *modality* is one independently-addressable sensor or actuator stream inside an
episode.  The engine treats modalities as an open registry rather than a fixed
enum baked into the frame schema, because the whole point of the canonical model
is that no dataset is required to carry every modality.  A LeRobot SO-100 recording
has RGB + joint state + joint action; a future dataset may add depth, tactile or a
dexterous-hand state vector, and it must slot in without a schema migration of the
frame table.

Two storage strategies exist, and the distinction drives the storage layer:

``INLINE``   small fixed-width numeric vectors (state, action) live directly in the
             tabular plane as Parquet list columns.
``EXTERNAL`` large binaries (video, images, point clouds) live in the blob store and
             the frame table holds only a *reference* -- blob id plus the index of
             the frame inside that blob.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ModalityKind(str, Enum):
    """Semantic family of a stream, independent of its physical encoding."""

    RGB = "rgb"
    DEPTH = "depth"
    ROBOT_STATE = "robot_state"
    ROBOT_ACTION = "robot_action"
    HAND_STATE = "hand_state"
    TACTILE = "tactile"
    FORCE_TORQUE = "force_torque"
    AUDIO = "audio"
    POINT_CLOUD = "point_cloud"
    LANGUAGE = "language"
    OTHER = "other"


class StoragePlane(str, Enum):
    """Which physical store backs the modality's payload."""

    INLINE = "inline"      # numeric column inside the frame Parquet table
    EXTERNAL = "external"  # blob store, referenced by id from the frame table


#: Kinds whose payload is always large enough to belong in the blob store.
_EXTERNAL_KINDS = {
    ModalityKind.RGB,
    ModalityKind.DEPTH,
    ModalityKind.POINT_CLOUD,
    ModalityKind.AUDIO,
}


class ModalitySpec(BaseModel):
    """Declaration of one stream available in a dataset.

    ``name`` is the canonical key used everywhere downstream (frame references, QC
    records, model adapters).  It is deliberately *not* the source dataset's key --
    :mod:`rde.ingestion` is responsible for mapping e.g. LeRobot's
    ``observation.images.top`` onto the canonical ``rgb.top``.
    """

    name: str = Field(..., description="Canonical stream key, e.g. 'rgb.top'.")
    kind: ModalityKind
    plane: StoragePlane
    dtype: str = Field(..., description="Element dtype, e.g. 'float32' or 'uint8'.")
    shape: tuple[int, ...] = Field(
        default=(),
        description="Per-sample shape excluding time, e.g. (6,) or (480, 640, 3).",
    )
    fps: float | None = Field(
        default=None, description="Nominal sampling rate; None if aperiodic."
    )
    source_key: str | None = Field(
        default=None, description="Original key in the source dataset, for lineage."
    )
    # Free-form, adapter-supplied detail (codec, pixel format, joint names, units...).
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_is_dotted_lowercase(cls, v: str) -> str:
        if not v or v != v.strip():
            raise ValueError("modality name must be non-empty and unpadded")
        if v != v.lower():
            raise ValueError(f"modality name must be lowercase: {v!r}")
        return v

    @property
    def is_external(self) -> bool:
        """True when the payload lives in the blob store rather than a column."""
        return self.plane is StoragePlane.EXTERNAL

    @property
    def element_count(self) -> int:
        """Number of scalars per sample (1 for scalars, 6 for a 6-DoF vector)."""
        n = 1
        for d in self.shape:
            n *= d
        return n

    @classmethod
    def infer_plane(cls, kind: ModalityKind) -> StoragePlane:
        """Default plane for a kind; adapters may override for odd cases."""
        return StoragePlane.EXTERNAL if kind in _EXTERNAL_KINDS else StoragePlane.INLINE


class BlobRef(BaseModel):
    """Pointer from a frame to a payload living in the blob store.

    ``frame_in_blob`` exists because robotics video is stored as *one mp4 per
    episode-chunk*, not one file per frame.  Resolving a single RGB observation is
    therefore (blob id, position inside that blob), and the decoder can seek to it
    without materialising the whole clip -- which is what keeps the engine inside
    its RAM budget.
    """

    modality: str
    blob_id: str = Field(..., description="Content address in the blob store.")
    frame_in_blob: int | None = Field(
        default=None, description="0-based frame position within the container."
    )
    timestamp_in_blob_s: float | None = Field(
        default=None, description="Presentation timestamp used for seeking."
    )
    encoding: Literal["video", "image", "npy", "raw"] = "video"

    def as_row(self) -> dict[str, Any]:
        """Flatten for Parquet storage (structs of scalars, no nesting)."""
        return {
            "modality": self.modality,
            "blob_id": self.blob_id,
            "frame_in_blob": self.frame_in_blob,
            "timestamp_in_blob_s": self.timestamp_in_blob_s,
            "encoding": self.encoding,
        }
