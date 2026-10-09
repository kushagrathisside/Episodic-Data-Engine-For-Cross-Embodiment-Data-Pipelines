"""Canonical data model for the robotics data engine.

Everything downstream of :mod:`rde.ingestion` speaks these types and only these
types.  Source-specific concepts (LeRobot chunk files, HF repo ids, codec quirks)
are confined to adapters and never leak past this boundary.
"""

from .core import (
    SCHEMA_VERSION,
    DatasetManifest,
    Episode,
    EpisodeOutcome,
    Frame,
    QualityStatus,
    Task,
)
from .lineage import (
    LineageGraph,
    LineageNode,
    TransformRun,
    VersionStage,
)
from .modality import BlobRef, ModalityKind, ModalitySpec, StoragePlane
from .qc import CheckScope, CheckStatus, QCRecord, QCReport, Severity

__all__ = [
    "SCHEMA_VERSION",
    "BlobRef",
    "CheckScope",
    "CheckStatus",
    "DatasetManifest",
    "Episode",
    "EpisodeOutcome",
    "Frame",
    "LineageGraph",
    "LineageNode",
    "ModalityKind",
    "ModalitySpec",
    "QCRecord",
    "QCReport",
    "QualityStatus",
    "Severity",
    "StoragePlane",
    "Task",
    "TransformRun",
    "VersionStage",
]
