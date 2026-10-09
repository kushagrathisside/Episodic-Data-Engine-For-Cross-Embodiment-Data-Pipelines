"""Cross-modal synchronization and alignment.

Robotics data is multi-clock by nature: joint states arrive from a control loop,
images from a camera pipeline, and the two agree only because someone made them.
This module is where that agreement is *verified* rather than assumed, and where
streams are placed on a common clock when they are not already.

The failure this exists to prevent is specific and severe.  In LeRobot v3.0 every
episode of a camera is concatenated into one container, so a frame's image is
located by ``episode.video_from_timestamp + frame.timestamp``.  Get that
composition wrong and every episode is paired with a *different episode's*
images.  The data stays structurally perfect -- right dtypes, right row counts,
no nulls -- and the model learns from mislabelled observations.  No numeric check
finds it.  So alignment is verified against the pixels themselves.

Two capabilities:

:class:`AlignmentVerifier`  proves the tabular clock and the video clock agree,
                            by decoding and comparing actual frames.
:func:`resample_to_grid`    places an irregularly-sampled stream onto a uniform
                            grid, for datasets whose modalities are not already
                            co-sampled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

from ..schemas.core import Episode
from ..schemas.qc import CheckScope, CheckStatus, QCRecord, Severity
from ..storage.video import VideoReader
from ..validation.base import CheckContext, EpisodeCheck


@dataclass
class AlignmentEvidence:
    """What the verifier observed for one episode/modality pair."""

    episode_id: int
    modality: str
    window_start_s: float
    window_end_s: float
    expected_duration_s: float
    observed_duration_s: float
    first_frame_decoded: bool
    last_frame_decoded: bool
    #: Mean absolute pixel difference between this episode's first frame and the
    #: previous episode's first frame.  Near zero across many episodes means the
    #: lookup is returning the same image regardless of episode -- the smoking
    #: gun for a broken offset.
    distinctness: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def duration_error_s(self) -> float:
        """Absolute mismatch between declared and implied window length."""
        return abs(self.observed_duration_s - self.expected_duration_s)


class VideoAlignmentCheck(EpisodeCheck):
    """Verify each episode's frames resolve to that episode's own video window.

    Four independent pieces of evidence, cheapest first:

    1. **Window arithmetic** -- the declared video window must span the same
       duration the frame count implies.  Catches offsets accumulated across
       episodes.
    2. **Stored-reference agreement** -- the ``<modality>__ts_in_blob`` column
       actually written to the frame table must equal
       ``episode.from_timestamp + frame.timestamp_s``.  This matters more than it
       sounds: that column, not the episode metadata, is what the model adapter
       dereferences at read time.  A check that only re-derives the mapping from
       metadata would confirm the *intent* while the stored data disagreed with
       it.
    3. **Decodability at the stored timestamps** -- the first and last frame must
       decode at the positions actually recorded.
    4. **Distinctness** -- the decoded first frame must differ from the frame the
       *previous* episode maps to, which catches a constant offset that all three
       preceding checks would pass happily.
    """

    name = "sync.video_alignment"
    severity = Severity.ERROR
    description = "Frame timestamps resolve to the correct window of the correct video."
    required_columns = ("timestamp_s", "frame_id", "episode_id")

    def __init__(self, distinctness_threshold: float = 1.0, sample_stride: int = 1):
        #: Mean absolute RGB difference (0-255) below which two frames are
        #: treated as effectively identical.
        self.distinctness_threshold = distinctness_threshold
        #: Check every Nth episode; alignment errors are systemic, so sampling
        #: is sound and keeps the decode cost bounded on large datasets.
        self.sample_stride = max(1, sample_stride)

    def dynamic_columns(self, ctx: CheckContext) -> list[str]:
        """The stored video-reference columns, which are modality-dependent."""
        return [
            f"{m.name}__ts_in_blob"
            for m in ctx.manifest.modalities
            if m.is_external
        ]

    def applies_to(self, ctx: CheckContext) -> bool:
        """Only meaningful when external modalities exist."""
        return any(m.is_external for m in ctx.manifest.modalities)

    def skip_reason(self, ctx: CheckContext) -> str:
        """Explain the skip on the QC record."""
        return "no external (video) modalities to align against"

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Verify window arithmetic, stored refs, decodability and distinctness."""
        if frames.num_rows == 0 or episode.episode_id % self.sample_stride:
            return

        windows = episode.attributes.get("video_windows", {}) or {}
        ts = np.asarray(frames.column("timestamp_s"), dtype=np.float64)
        prev_first: dict[str, np.ndarray] = ctx.shared.setdefault("_align_prev_first", {})
        # A modality with its own declared fps (e.g. SubsampleFrames applied
        # to one camera but not another) is not sampled at the dataset-wide
        # rate; using ctx.fps for it here would raise false window-arithmetic
        # errors -- or worse, mask a real one -- for exactly that modality.
        mod_fps = {m.name: (m.fps or ctx.fps) for m in ctx.manifest.modalities}

        for modality, blob_id in (episode.blob_ids or {}).items():
            win = windows.get(modality)
            if win is None:
                continue
            start, end = float(win["from_timestamp"]), float(win["to_timestamp"])
            fps = mod_fps.get(modality, ctx.fps)

            # (1) window arithmetic
            expected = frames.num_rows / fps
            observed = end - start
            tol = max(2.0 / fps, 0.02 * expected)
            if abs(observed - expected) > tol:
                yield ctx.record(
                    self.name, CheckScope.EPISODE, Severity.ERROR, CheckStatus.FAILED,
                    f"episode {episode.episode_id}: {modality} window spans {observed:.3f}s "
                    f"but {frames.num_rows} frames at {fps} fps imply {expected:.3f}s",
                    episode_id=episode.episode_id, modality=modality,
                    observed=f"{observed:.4f}s", expected=f"{expected:.4f}s +/-{tol:.3f}",
                )

            # (2) the reference column actually stored must match the mapping
            ref_col = f"{modality}__ts_in_blob"
            probe_start = start + float(ts[0])
            probe_end = start + float(ts[-1])
            if ref_col in frames.schema.names:
                stored = np.asarray(frames.column(ref_col), dtype=np.float64)
                derived = start + ts
                finite = np.isfinite(stored) & np.isfinite(derived)
                if not finite.all():
                    yield ctx.record(
                        self.name, CheckScope.EPISODE, Severity.ERROR, CheckStatus.FAILED,
                        f"episode {episode.episode_id}: {modality} has "
                        f"{int((~finite).sum())} unusable video reference(s)",
                        episode_id=episode.episode_id, modality=modality,
                        observed="null/NaN reference", expected="a finite timestamp",
                    )
                else:
                    drift = np.abs(stored - derived)
                    worst = float(drift.max())
                    # Published for every episode, not only failing ones, so
                    # the alignment histogram describes the whole distribution
                    # rather than only its tail.  An offset well inside one
                    # frame period is the healthy case and must be visible as
                    # such -- a metric that only appears when something is
                    # broken cannot show you that nothing is.
                    ctx.shared.setdefault("alignment_offsets", []).append(
                        (modality, worst)
                    )
                    # Half a frame period: anything larger resolves to a
                    # different frame than the one the episode metadata implies.
                    ref_tol = 0.5 / fps
                    if worst > ref_tol:
                        i = int(np.argmax(drift))
                        yield ctx.record(
                            self.name, CheckScope.EPISODE, Severity.ERROR, CheckStatus.FAILED,
                            f"episode {episode.episode_id}: stored {modality} reference "
                            f"drifts up to {worst*1000:.1f} ms from "
                            f"episode_start + frame_timestamp; the frame table points at "
                            f"the wrong video position",
                            episode_id=episode.episode_id, modality=modality,
                            frame_id=int(np.asarray(frames.column("frame_id"))[i]),
                            observed=f"{stored[i]:.4f}s", expected=f"{derived[i]:.4f}s",
                            max_drift_ms=round(worst * 1000, 3),
                            drifted_frames=int((drift > ref_tol).sum()),
                        )
                    # Probe the positions actually stored, not the ideal ones.
                    probe_start, probe_end = float(stored[0]), float(stored[-1])
            else:
                # This is the strongest of the four checks: it is the only one
                # that compares the stored mapping against the frame table
                # rather than re-deriving it from metadata, so silently
                # skipping it (as opposed to falling back to the metadata-only
                # probe_start/probe_end computed above) would make a schema
                # regression invisible. Recorded once per episode/modality,
                # not raised as an error -- the column may legitimately not
                # exist for a modality this dataset version never stored a
                # per-frame reference for.
                yield ctx.record(
                    self.name, CheckScope.EPISODE, Severity.WARNING, CheckStatus.SKIPPED,
                    f"episode {episode.episode_id}: {modality} has no "
                    f"'{ref_col}' column, so the stored-reference check (2) "
                    f"could not run -- only the weaker metadata-derived window "
                    f"was verified",
                    episode_id=episode.episode_id, modality=modality,
                    observed="column absent", expected=ref_col,
                )

            try:
                path = ctx.blob_store.get(blob_id)
            except KeyError:
                continue

            # (3) decodability at the stored positions
            with VideoReader(path, cache_frames=0) as reader:
                first = reader.read_at(probe_start)
                last = reader.read_at(probe_end)
                if first is None or last is None:
                    yield ctx.record(
                        self.name, CheckScope.EPISODE, Severity.ERROR, CheckStatus.FAILED,
                        f"episode {episode.episode_id}: {modality} did not decode at the "
                        f"mapped {'first' if first is None else 'last'} frame position",
                        episode_id=episode.episode_id, modality=modality,
                        observed="no frame", expected="an RGB frame",
                    )
                    continue

                # (4) distinctness from the previous episode's mapped frame
                prev = prev_first.get(modality)
                if prev is not None and prev.shape == first.shape:
                    diff = float(np.abs(prev.astype(np.int16) - first.astype(np.int16)).mean())
                    if diff < self.distinctness_threshold:
                        yield ctx.record(
                            self.name, CheckScope.EPISODE, Severity.ERROR, CheckStatus.FAILED,
                            f"episode {episode.episode_id}: {modality} first frame is "
                            f"pixel-identical to the previous episode's (mean |diff| "
                            f"{diff:.3f}); the video offset is probably not being applied",
                            episode_id=episode.episode_id, modality=modality,
                            observed=f"{diff:.4f}",
                            expected=f">= {self.distinctness_threshold}",
                        )
                prev_first[modality] = first


class AlignmentVerifier:
    """Standalone alignment audit, usable outside a QC run.

    The QC check above reports findings; this returns the raw evidence, which is
    what the exploration CLI and the docs use to *show* that alignment holds
    rather than merely asserting it.
    """

    def __init__(self, blob_store, fps: float, table_store=None):
        self.blob_store = blob_store
        self.fps = fps
        self.table_store = table_store

    def verify_episode(
        self, episode: Episode, num_frames: int, previous_first: dict[str, np.ndarray] | None = None
    ) -> list[AlignmentEvidence]:
        """Collect raw alignment evidence for one episode, per modality."""
        out: list[AlignmentEvidence] = []
        windows = episode.attributes.get("video_windows", {}) or {}
        for modality, blob_id in (episode.blob_ids or {}).items():
            win = windows.get(modality)
            if win is None:
                continue
            start, end = float(win["from_timestamp"]), float(win["to_timestamp"])
            ev = AlignmentEvidence(
                episode_id=episode.episode_id,
                modality=modality,
                window_start_s=start,
                window_end_s=end,
                expected_duration_s=num_frames / self.fps,
                observed_duration_s=end - start,
                first_frame_decoded=False,
                last_frame_decoded=False,
            )
            try:
                path = self.blob_store.get(blob_id)
            except KeyError:
                ev.notes.append("blob missing")
                out.append(ev)
                continue
            with VideoReader(path, cache_frames=0) as reader:
                first = reader.read_at(start)
                last = reader.read_at(max(start, end - 1.0 / self.fps))
                ev.first_frame_decoded = first is not None
                ev.last_frame_decoded = last is not None
                if previous_first is not None and modality in previous_first and first is not None:
                    prev = previous_first[modality]
                    if prev.shape == first.shape:
                        ev.distinctness = float(
                            np.abs(prev.astype(np.int16) - first.astype(np.int16)).mean()
                        )
                if first is not None and previous_first is not None:
                    previous_first[modality] = first
            out.append(ev)
        return out


def resample_to_grid(
    timestamps: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    method: str = "previous",
    max_gap_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Place an irregular stream onto a uniform time grid.

    Returns ``(resampled, valid_mask)``.  Points where the nearest source sample
    is further than ``max_gap_s`` are marked invalid rather than silently
    interpolated across a dropout -- inventing data over a gap is exactly how a
    dataset comes to look cleaner than the robot that produced it.

    ``method``:
      ``previous``  zero-order hold.  Correct for commanded actions and any
                    signal that is piecewise-constant between updates.
      ``linear``    interpolation.  Appropriate for smoothly-varying measured
                    quantities such as joint positions.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if timestamps.size == 0:
        return np.full((grid.size,) + values.shape[1:], np.nan), np.zeros(grid.size, bool)

    two_d = values.ndim == 2
    v = values if two_d else values[:, None]

    idx = np.searchsorted(timestamps, grid, side="right") - 1
    idx_clipped = np.clip(idx, 0, timestamps.size - 1)

    if method == "previous":
        out = v[idx_clipped]
        gap = np.abs(grid - timestamps[idx_clipped])
    elif method == "linear":
        out = np.empty((grid.size, v.shape[1]), dtype=np.float64)
        for j in range(v.shape[1]):
            out[:, j] = np.interp(grid, timestamps, v[:, j])
        nxt = np.clip(idx_clipped + 1, 0, timestamps.size - 1)
        gap = np.minimum(
            np.abs(grid - timestamps[idx_clipped]), np.abs(timestamps[nxt] - grid)
        )
    else:
        raise ValueError(f"unknown resample method: {method!r}")

    valid = (grid >= timestamps[0] - 1e-9) & (grid <= timestamps[-1] + 1e-9)
    if max_gap_s is not None:
        valid &= gap <= max_gap_s
    out = np.where(valid[:, None], out, np.nan)
    return (out if two_d else out[:, 0]), valid


def uniform_grid(duration_s: float, fps: float, start_s: float = 0.0) -> np.ndarray:
    """Uniform timestamp grid, the target clock for resampling."""
    n = int(np.floor(duration_s * fps + 1e-9)) + 1
    return start_s + np.arange(n, dtype=np.float64) / fps


SYNC_CHECKS = [VideoAlignmentCheck()]
