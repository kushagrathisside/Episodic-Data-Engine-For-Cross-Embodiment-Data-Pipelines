"""Test utilities: synthetic dataset generation and fault injection."""

from .synthetic import (
    inject_discontinuity,
    inject_distribution_shift,
    inject_dropped_frames,
    inject_duplicate_timestamp,
    inject_missing_blob,
    inject_nan,
    inject_non_monotonic,
    inject_stuck_sensor,
    inject_timestamp_gap,
    inject_video_misalignment,
    make_lerobot_dataset,
)

__all__ = [
    "inject_discontinuity",
    "inject_distribution_shift",
    "inject_dropped_frames",
    "inject_duplicate_timestamp",
    "inject_missing_blob",
    "inject_nan",
    "inject_non_monotonic",
    "inject_stuck_sensor",
    "inject_timestamp_gap",
    "inject_video_misalignment",
    "make_lerobot_dataset",
]
