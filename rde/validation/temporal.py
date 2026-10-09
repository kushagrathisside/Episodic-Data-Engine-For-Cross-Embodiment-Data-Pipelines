"""Temporal validation: is time well-behaved?

Temporal defects are the ones most likely to survive into training and quietly
degrade a policy, because the data still *looks* fine -- correct dtype, plausible
values, right row count.  A 200 ms gap where the recorder stalled produces an
action pair that implies an impossible joint velocity, and the model learns it.

Checks here cover monotonicity, duplicate timestamps, gaps, sampling jitter, and
agreement between the tabular clock and the video clock.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pyarrow as pa

from ..schemas.core import Episode
from ..schemas.qc import CheckScope, CheckStatus, QCRecord, Severity
from .base import CheckContext, EpisodeCheck


def _timestamps(frames: pa.Table) -> np.ndarray:
    return np.asarray(frames.column("timestamp_s"), dtype=np.float64)


class MonotonicTimestampCheck(EpisodeCheck):
    """Timestamps must strictly increase within an episode.

    A non-monotonic timestamp means frames are out of order; any windowed model
    input built from them pairs the wrong observation with the wrong action.
    """

    name = "temporal.monotonic"
    severity = Severity.ERROR
    description = "Timestamps strictly increase within each episode."
    required_columns = ("timestamp_s", "frame_id")

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Flag any non-increasing timestamp step."""
        if frames.num_rows < 2:
            return
        ts = _timestamps(frames)
        d = np.diff(ts)
        bad = np.nonzero(d <= 0)[0]
        if bad.size:
            fids = np.asarray(frames.column("frame_id"))
            first = int(bad[0])
            yield ctx.record(
                self.name, CheckScope.EPISODE, self.severity, CheckStatus.FAILED,
                f"episode {episode.episode_id}: {bad.size} non-increasing timestamp "
                f"step(s); first at frame_index {first + 1}",
                episode_id=episode.episode_id, frame_id=int(fids[first + 1]),
                observed=f"dt={d[first]:.6f}s", expected="dt > 0",
                violation_count=int(bad.size),
            )


class DuplicateTimestampCheck(EpisodeCheck):
    """Exactly repeated timestamps indicate a stalled or double-written clock."""

    name = "temporal.duplicates"
    severity = Severity.ERROR
    description = "No two frames in an episode share a timestamp."
    required_columns = ("timestamp_s", "frame_id")

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Flag exactly repeated timestamp values."""
        if frames.num_rows < 2:
            return
        ts = _timestamps(frames)
        uniq, counts = np.unique(ts, return_counts=True)
        dupes = uniq[counts > 1]
        if dupes.size:
            yield ctx.record(
                self.name, CheckScope.EPISODE, self.severity, CheckStatus.FAILED,
                f"episode {episode.episode_id}: {dupes.size} duplicated timestamp value(s)",
                episode_id=episode.episode_id,
                observed=f"{dupes.size} duplicates, first at t={dupes[0]:.4f}s",
                expected="all timestamps unique",
                duplicate_values=[float(x) for x in dupes[:10]],
            )


class SamplingIntervalCheck(EpisodeCheck):
    """Inter-frame intervals must stay close to the declared 1/fps.

    Tolerance is relative and generous by default (25%): recorded robot data
    always jitters a little, and flagging every scheduler hiccup as an error
    would make the QC report unreadable.  What matters is the *distribution* --
    the check reports jitter statistics and only fails when a real gap appears.
    """

    name = "temporal.sampling_interval"
    severity = Severity.WARNING
    description = "Inter-frame intervals stay within tolerance of the nominal period."
    required_columns = ("timestamp_s", "frame_id")

    def __init__(self, rel_tolerance: float = 0.25, max_violation_ratio: float = 0.01):
        self.rel_tolerance = rel_tolerance
        #: Fraction of intervals allowed to breach tolerance before flagging.
        self.max_violation_ratio = max_violation_ratio

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Report interval jitter and flag excessive tolerance breaches."""
        if frames.num_rows < 3:
            return
        ts = _timestamps(frames)
        d = np.diff(ts)
        nominal = 1.0 / ctx.fps
        lo, hi = nominal * (1 - self.rel_tolerance), nominal * (1 + self.rel_tolerance)
        bad = np.nonzero((d < lo) | (d > hi))[0]
        ratio = bad.size / d.size

        ctx.shared.setdefault("interval_stats", {})[episode.episode_id] = {
            "mean": float(d.mean()), "std": float(d.std()),
            "min": float(d.min()), "max": float(d.max()),
        }

        if ratio > self.max_violation_ratio:
            yield ctx.record(
                self.name, CheckScope.EPISODE, self.severity, CheckStatus.WARNED,
                f"episode {episode.episode_id}: {bad.size}/{d.size} intervals "
                f"({ratio:.1%}) outside +/-{self.rel_tolerance:.0%} of {nominal*1000:.1f} ms",
                episode_id=episode.episode_id,
                observed=f"mean={d.mean()*1000:.2f}ms std={d.std()*1000:.2f}ms max={d.max()*1000:.2f}ms",
                expected=f"{nominal*1000:.2f}ms +/-{self.rel_tolerance:.0%}",
                violation_ratio=round(ratio, 4),
            )


class TimestampGapCheck(EpisodeCheck):
    """A large jump means frames are missing, even if indices look contiguous.

    This is the counterpart to ``structural.frame_index_continuity``: the index
    can be perfectly dense while the clock shows a 2 s hole, which means the
    recorder kept counting through a dropout.
    """

    name = "temporal.gaps"
    severity = Severity.ERROR
    description = "No interval exceeds a multiple of the nominal sampling period."
    required_columns = ("timestamp_s", "frame_id")

    def __init__(self, gap_factor: float = 3.0):
        self.gap_factor = gap_factor

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Flag intervals large enough to imply dropped frames."""
        if frames.num_rows < 2:
            return
        ts = _timestamps(frames)
        fids = np.asarray(frames.column("frame_id"))
        d = np.diff(ts)
        threshold = self.gap_factor / ctx.fps
        gaps = np.nonzero(d > threshold)[0]
        for g in gaps[:20]:  # cap: a catastrophic episode should not flood the table
            missing = int(round(d[g] * ctx.fps)) - 1
            yield ctx.record(
                self.name, CheckScope.EPISODE, self.severity, CheckStatus.FAILED,
                f"episode {episode.episode_id}: {d[g]*1000:.1f} ms gap after "
                f"frame_index {int(g)} (~{missing} frames missing)",
                episode_id=episode.episode_id, frame_id=int(fids[g]),
                observed=f"{d[g]*1000:.2f}ms", expected=f"<= {threshold*1000:.2f}ms",
                estimated_missing_frames=missing,
            )


class TimestampOriginCheck(EpisodeCheck):
    """Episode-relative timestamps should start at (or very near) zero.

    Catches an ingestion bug class where episode-relative and file-absolute clocks
    are conflated -- if episode 7 started at 100 s, its timestamps were never
    rebased and every downstream video lookup for it is wrong.
    """

    name = "temporal.origin"
    severity = Severity.WARNING
    description = "Each episode's first timestamp is at the episode origin."
    required_columns = ("timestamp_s",)

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Confirm the episode's clock starts at its own origin."""
        if frames.num_rows == 0:
            return
        t0 = float(_timestamps(frames)[0])
        tol = 1.5 / ctx.fps
        if abs(t0) > tol:
            yield ctx.record(
                self.name, CheckScope.EPISODE, self.severity, CheckStatus.WARNED,
                f"episode {episode.episode_id} starts at t={t0:.3f}s, not the origin",
                episode_id=episode.episode_id,
                observed=f"{t0:.4f}s", expected=f"|t0| <= {tol:.4f}s",
            )


class DurationConsistencyCheck(EpisodeCheck):
    """Frame count, declared duration and timestamp span must agree.

    Three independent statements of the same fact; disagreement means one of them
    is derived from stale or wrong metadata.
    """

    name = "temporal.duration_consistency"
    severity = Severity.WARNING
    description = "num_frames/fps agrees with the observed timestamp span."
    required_columns = ("timestamp_s",)

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Cross-check frame count, fps and observed timestamp span."""
        if frames.num_rows < 2:
            return
        ts = _timestamps(frames)
        observed_span = float(ts[-1] - ts[0])
        expected_span = (frames.num_rows - 1) / ctx.fps
        tol = max(2.0 / ctx.fps, 0.05 * expected_span)
        if abs(observed_span - expected_span) > tol:
            yield ctx.record(
                self.name, CheckScope.EPISODE, self.severity, CheckStatus.WARNED,
                f"episode {episode.episode_id}: timestamp span {observed_span:.3f}s vs "
                f"{expected_span:.3f}s implied by {frames.num_rows} frames at {ctx.fps} fps",
                episode_id=episode.episode_id,
                observed=f"{observed_span:.4f}s", expected=f"{expected_span:.4f}s +/-{tol:.3f}",
            )


TEMPORAL_CHECKS = [
    MonotonicTimestampCheck(),
    DuplicateTimestampCheck(),
    TimestampGapCheck(),
    SamplingIntervalCheck(),
    TimestampOriginCheck(),
    DurationConsistencyCheck(),
]
