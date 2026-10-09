"""Ingestion: external formats -> canonical model.

Adapters are the only components that know a source format.  Everything after
:func:`ingest` speaks canonical Parquet, episodes and manifests.
"""

from .base import SourceAdapter, SourceProfile
from .download import SourceLocation, download_hf_dataset
from .ingest import ADAPTERS, IngestResult, ingest
from .lerobot import LeRobotAdapter, canonical_name

__all__ = [
    "ADAPTERS",
    "IngestResult",
    "LeRobotAdapter",
    "SourceAdapter",
    "SourceLocation",
    "SourceProfile",
    "canonical_name",
    "download_hf_dataset",
    "ingest",
]
