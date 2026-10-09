"""Cross-modal synchronization: verify and enforce a common clock."""

from .align import (
    SYNC_CHECKS,
    AlignmentEvidence,
    AlignmentVerifier,
    VideoAlignmentCheck,
    resample_to_grid,
    uniform_grid,
)

__all__ = [
    "SYNC_CHECKS",
    "AlignmentEvidence",
    "AlignmentVerifier",
    "VideoAlignmentCheck",
    "resample_to_grid",
    "uniform_grid",
]
