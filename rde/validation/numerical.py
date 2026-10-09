"""Numerical validation: are the numbers physically plausible?

Structural and temporal checks establish that the data is *there* and *ordered*.
These establish that it is *believable*.  The failure modes targeted here are the
ones that reach a policy and corrupt it silently:

* NaN/Inf from a dropped sensor read
* values outside the robot's mechanical range (a decoding or unit error)
* discontinuities implying impossible joint velocities
* a stuck sensor -- constant output while the robot is demonstrably moving
* per-episode statistics far outside the dataset's own distribution

Ranges are learned from the dataset rather than hard-coded: this engine must work
for an SO-100 in degrees and an ALOHA arm in radians without a config change.
A robot-specific limits file can override the learned bounds when one exists.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa

from ..schemas.core import Episode
from ..schemas.modality import ModalityKind, StoragePlane
from ..schemas.qc import CheckScope, CheckStatus, QCRecord, Severity
from .base import CheckContext, DatasetCheck, EpisodeCheck


class _VectorReservoir:
    """Uniform random sample of up to ``capacity`` rows from a streamed
    ``(n, dims)`` source, in bounded memory.

    Each incoming row gets an i.i.d. ``Uniform(0, 1)`` key; the rows with the
    top-``capacity`` keys seen so far form a uniform random subset of the
    whole stream -- unlike keeping the first ``capacity`` rows encountered
    (biased toward whatever appears earliest in the table, e.g. the first
    ~18 episodes at the default capacity), and unlike a per-row Algorithm-R
    replacement loop (not vectorizable over a batch, so O(rows) Python-level
    overhead on a 50M-row table). ``rng`` must be seeded deterministically by
    the caller: profiling has to be reproducible for the QC cache fingerprint
    to mean anything.
    """

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self._rng = rng
        self._vals: np.ndarray | None = None
        self._keys: np.ndarray | None = None

    def add(self, batch: np.ndarray) -> None:
        if batch.size == 0:
            return
        keys = self._rng.random(batch.shape[0])
        if self._vals is None:
            vals, all_keys = batch, keys
        else:
            vals = np.vstack([self._vals, batch])
            all_keys = np.concatenate([self._keys, keys])
        if vals.shape[0] > self.capacity:
            idx = np.argpartition(all_keys, -self.capacity)[-self.capacity:]
            vals, all_keys = vals[idx], all_keys[idx]
        self._vals, self._keys = vals, all_keys

    def get(self) -> np.ndarray | None:
        return self._vals


def _vectors(frames: pa.Table, column: str) -> np.ndarray | None:
    """Fixed-size-list column -> (n_frames, width) float array."""
    if column not in frames.schema.names:
        return None
    col = frames.column(column).combine_chunks()
    try:
        arr = np.asarray(col.flatten(), dtype=np.float64)
    except (ValueError, pa.ArrowInvalid):
        return None
    if arr.size == 0:
        return arr.reshape(0, 0)
    width = arr.size // frames.num_rows
    return arr.reshape(frames.num_rows, width)


def _numeric_modalities(ctx: CheckContext) -> list[str]:
    return [
        m.name
        for m in ctx.manifest.modalities
        if m.plane is StoragePlane.INLINE
        and m.kind in (ModalityKind.ROBOT_STATE, ModalityKind.ROBOT_ACTION,
                       ModalityKind.HAND_STATE, ModalityKind.FORCE_TORQUE,
                       ModalityKind.TACTILE)
    ]


class FiniteValueCheck(EpisodeCheck):
    """No NaN or Inf in any numeric modality."""

    name = "numerical.finite"
    severity = Severity.ERROR
    description = "State/action vectors contain no NaN or Inf values."
    required_columns = ("frame_id",)

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Flag NaN or Inf in any numeric modality."""
        fids = np.asarray(frames.column("frame_id")) if "frame_id" in frames.schema.names else None
        for name in _numeric_modalities(ctx):
            v = _vectors(frames, name)
            if v is None or v.size == 0:
                continue
            bad = ~np.isfinite(v)
            if bad.any():
                rows = np.unique(np.nonzero(bad)[0])
                first = int(rows[0])
                yield ctx.record(
                    self.name, CheckScope.EPISODE, self.severity, CheckStatus.FAILED,
                    f"episode {episode.episode_id}: {name} has {int(bad.sum())} "
                    f"non-finite value(s) across {rows.size} frame(s)",
                    episode_id=episode.episode_id, modality=name,
                    frame_id=int(fids[first]) if fids is not None else None,
                    observed=f"{int(bad.sum())} non-finite", expected="all finite",
                    affected_frames=int(rows.size),
                )


class ValueRangeCheck(EpisodeCheck):
    """Values must stay inside plausible per-dimension bounds.

    Bounds come from :class:`RangeProfiler` (learned percentiles, widened by a
    margin), or from a hard ``limits`` table when the robot's spec is known.
    Learned bounds keep the check robot-agnostic.

    **A single sample crossing the bound is not a defect.**  Learned bounds are
    percentiles, so by construction ~1% of the dataset lies outside them.  With
    enough dimensions and long-tailed signals that tautology flags everything:
    on a 14-DoF bimanual dataset with force/torque data (42 dimensions, ~1100
    frames per episode), an early version of this check fired on *all 50
    episodes* -- and a check that flags everything carries no information.

    So a finding requires the violation to be **material**, in one of two ways:

    ``systematic``  more than ``min_violation_ratio`` of the episode's samples
                    fall outside the bound -- the episode really does occupy a
                    different region than the dataset.
    ``egregious``   a single sample lands more than ``far_excursion`` x the
                    dimension's span beyond the bound -- a spike no margin
                    explains.

    A hard ``limits`` table bypasses both: exceeding a mechanical limit is a
    defect at the first sample, no matter how briefly.
    """

    name = "numerical.value_range"
    severity = Severity.WARNING
    description = "Values stay within plausible per-dimension bounds (materially, not marginally)."
    required_columns = ("frame_id",)

    def __init__(
        self,
        margin: float = 0.10,
        limits: dict[str, tuple[Sequence[float], Sequence[float]]] | None = None,
        min_violation_ratio: float = 0.01,
        far_excursion: float = 0.25,
    ):
        self.margin = margin
        self.limits = limits or {}
        self.min_violation_ratio = min_violation_ratio
        self.far_excursion = far_excursion

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Flag values materially outside learned or configured bounds."""
        bounds = ctx.shared.get("value_bounds", {})
        for name in _numeric_modalities(ctx):
            v = _vectors(frames, name)
            if v is None or v.size == 0:
                continue
            hard = name in self.limits
            if hard:
                lo, hi = (np.asarray(x, dtype=np.float64) for x in self.limits[name])
                span = np.maximum(hi - lo, 1e-9)
            elif name in bounds:
                lo = np.asarray(bounds[name]["lo"])
                hi = np.asarray(bounds[name]["hi"])
                span = np.maximum(hi - lo, 1e-9)
                lo, hi = lo - self.margin * span, hi + self.margin * span
            else:
                continue

            finite = np.isfinite(v)
            below = (v < lo) & finite
            above = (v > hi) & finite
            out = below | above
            if not out.any():
                continue

            # How far outside, per dimension, in units of that dimension's span.
            excursion = np.maximum(
                np.where(below, lo - v, 0.0), np.where(above, v - hi, 0.0)
            ) / span
            ratio_per_dim = out.mean(axis=0)
            worst_per_dim = excursion.max(axis=0)

            if hard:
                flagged = np.nonzero(out.any(axis=0))[0]
                why = "exceeds configured limits"
            else:
                systematic = ratio_per_dim > self.min_violation_ratio
                egregious = worst_per_dim > self.far_excursion
                flagged = np.nonzero(systematic | egregious)[0]
                why = "materially outside learned bounds"
            if flagged.size == 0:
                continue  # marginal crossings only -- not a defect

            dims = flagged.tolist()
            yield ctx.record(
                self.name, CheckScope.EPISODE,
                Severity.ERROR if hard else self.severity,
                CheckStatus.FAILED if hard else CheckStatus.WARNED,
                f"episode {episode.episode_id}: {name} {why} in dimension(s) {dims} "
                f"(worst {worst_per_dim[flagged].max():.2f}x span beyond bound, "
                f"{ratio_per_dim[flagged].max():.1%} of samples)",
                episode_id=episode.episode_id, modality=name,
                observed=f"min={np.nanmin(v, axis=0).round(3).tolist()}, "
                         f"max={np.nanmax(v, axis=0).round(3).tolist()}",
                expected=f"lo={np.round(lo,3).tolist()}, hi={np.round(hi,3).tolist()}",
                dimensions=dims,
                violation_ratio=round(float(ratio_per_dim[flagged].max()), 5),
                worst_excursion_spans=round(float(worst_per_dim[flagged].max()), 4),
            )


class DiscontinuityCheck(EpisodeCheck):
    """Step changes implying impossible velocities.

    Threshold is derived from the dataset's own step distribution (a high
    percentile times a factor) rather than an absolute number, because a
    reasonable joint step depends entirely on the robot and the control rate.
    """

    name = "numerical.discontinuity"
    severity = Severity.WARNING
    description = "No single-step jump far outside the dataset's own step distribution."
    required_columns = ("frame_id",)

    def __init__(self, factor: float = 4.0, max_reports: int = 5):
        self.factor = factor
        self.max_reports = max_reports

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Flag single-step jumps far outside the dataset's own step distribution."""
        if frames.num_rows < 3:
            return
        thresholds = ctx.shared.get("step_thresholds", {})
        fids = np.asarray(frames.column("frame_id"))
        for name in _numeric_modalities(ctx):
            thr = thresholds.get(name)
            if thr is None:
                continue
            v = _vectors(frames, name)
            if v is None or v.size == 0:
                continue
            steps = np.abs(np.diff(v, axis=0))
            limit = np.asarray(thr) * self.factor
            hits = np.nonzero((steps > limit).any(axis=1))[0]
            for h in hits[: self.max_reports]:
                dim = int(np.argmax(steps[h] - limit))
                yield ctx.record(
                    self.name, CheckScope.EPISODE, self.severity, CheckStatus.WARNED,
                    f"episode {episode.episode_id}: {name} dim {dim} jumps "
                    f"{steps[h][dim]:.3f} in one step",
                    episode_id=episode.episode_id, modality=name,
                    frame_id=int(fids[h + 1]),
                    observed=f"{steps[h][dim]:.4f}",
                    expected=f"<= {limit[dim]:.4f}",
                    dimension=dim, total_violations=int(hits.size),
                )


class StuckSensorCheck(EpisodeCheck):
    """A dimension that never changes across a whole episode.

    Distinct from a legitimately-still joint: this fires only when the dimension
    is *exactly* constant for the entire episode, which is the signature of a
    sensor that stopped reporting rather than a robot holding position.
    """

    name = "numerical.stuck_sensor"
    severity = Severity.WARNING
    description = "No dimension is exactly constant for an entire episode."
    required_columns = ("frame_id",)

    def __init__(self, min_frames: int = 30):
        self.min_frames = min_frames

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Flag dimensions that never change across a whole episode."""
        if frames.num_rows < self.min_frames:
            return
        for name in _numeric_modalities(ctx):
            v = _vectors(frames, name)
            if v is None or v.size == 0:
                continue
            spread = np.nanmax(v, axis=0) - np.nanmin(v, axis=0)
            stuck = np.nonzero(spread == 0.0)[0]
            if stuck.size:
                yield ctx.record(
                    self.name, CheckScope.EPISODE, self.severity, CheckStatus.WARNED,
                    f"episode {episode.episode_id}: {name} dimension(s) "
                    f"{stuck.tolist()} constant over {frames.num_rows} frames",
                    episode_id=episode.episode_id, modality=name,
                    observed="zero variation", expected="> 0",
                    dimensions=stuck.tolist(),
                    constant_values=[float(v[0, i]) for i in stuck.tolist()],
                )


class RangeProfiler(DatasetCheck):
    """Learn per-dimension bounds and step thresholds from the data itself.

    Not a check in the pass/fail sense -- it emits an INFO record and populates
    ``ctx.shared`` for the episode checks that follow.  It is modelled as a
    DatasetCheck so it participates in the same scheduling and reporting as
    everything else rather than being a hidden pre-step.

    Streams the frame table in batches and keeps only a bounded reservoir per
    dimension (see :class:`_VectorReservoir`), so profiling a dataset larger
    than RAM costs the same memory as profiling a small one -- both for the
    value bounds and for the step thresholds, which stream through the exact
    same reservoir mechanism rather than accumulating without limit.
    """

    name = "numerical.range_profile"
    severity = Severity.INFO
    priority = -100  # must populate ctx.shared before the episode checks run
    description = "Derives per-dimension bounds and step thresholds used by later checks."

    #: Kinds whose signals are impulsive by nature.  Contact forces and tactile
    #: readings spike far beyond their bulk distribution as a matter of physics --
    #: on this ALOHA dataset the true max of a force channel is 2.4x its 99.5th
    #: percentile.  Applying position-like percentiles to them flags normal
    #: contact as an anomaly, so they get wider bounds rather than a special case
    #: buried in the consuming check.
    IMPULSIVE_KINDS = (ModalityKind.FORCE_TORQUE, ModalityKind.TACTILE)
    IMPULSIVE_PCT = (0.02, 99.98)

    def __init__(self, low_pct: float = 0.5, high_pct: float = 99.5, step_pct: float = 99.5,
                 reservoir: int = 20000):
        self.low_pct, self.high_pct, self.step_pct = low_pct, high_pct, step_pct
        #: Cap on retained samples per modality; percentiles need order
        #: statistics, and a bounded reservoir keeps that affordable.
        self.reservoir = reservoir

    def check_dataset(self, ctx: CheckContext) -> Iterator[QCRecord]:
        """Stream the table and derive bounds and step thresholds into ctx.shared."""
        names = _numeric_modalities(ctx)
        if not names or not ctx.table_store.exists("frames"):
            return
        # Fixed seed: two runs over unchanged data must produce identical
        # bounds, or the QC cache fingerprint (rde/qc/cache.py) would treat
        # every re-run as a fresh result even when nothing changed.
        rng = np.random.default_rng(0)
        samples: dict[str, _VectorReservoir] = {
            n: _VectorReservoir(self.reservoir, rng) for n in names
        }
        steps: dict[str, _VectorReservoir] = {
            n: _VectorReservoir(self.reservoir, rng) for n in names
        }

        cols = ["episode_id"] + names
        available = set(ctx.table_store.schema("frames").names)
        cols = [c for c in cols if c in available]

        prev_ep: int | None = None
        tail: dict[str, np.ndarray] = {}
        for batch in ctx.table_store.iter_batches(
            "frames", columns=cols, batch_size=ctx.settings.batch_size
        ):
            t = pa.Table.from_batches([batch])
            eps = np.asarray(t.column("episode_id"))
            for n in names:
                v = _vectors(t, n)
                if v is None or v.size == 0:
                    continue
                samples[n].add(v)
                # Steps are computed within an episode only; a batch boundary in
                # the middle of an episode must not lose the pair that straddles
                # it, and an episode change must not create a phantom jump.
                s = np.abs(np.diff(v, axis=0))
                if prev_ep is not None and eps[0] == prev_ep and n in tail:
                    s = np.vstack([np.abs(v[0] - tail[n])[None, :], s]) if s.size else np.abs(v[0] - tail[n])[None, :]
                if s.size:
                    steps[n].add(s)
                tail[n] = v[-1]
            prev_ep = int(eps[-1])

        kinds = {m.name: m.kind for m in ctx.manifest.modalities}
        bounds: dict[str, dict] = {}
        thresholds: dict[str, list[float]] = {}
        for n in names:
            allv = samples[n].get()
            if allv is not None:
                finite = np.where(np.isfinite(allv), allv, np.nan)
                lo_pct, hi_pct = (
                    self.IMPULSIVE_PCT
                    if kinds.get(n) in self.IMPULSIVE_KINDS
                    else (self.low_pct, self.high_pct)
                )
                bounds[n] = {
                    "lo": np.nanpercentile(finite, lo_pct, axis=0).tolist(),
                    "hi": np.nanpercentile(finite, hi_pct, axis=0).tolist(),
                    "mean": np.nanmean(finite, axis=0).tolist(),
                    "std": np.nanstd(finite, axis=0).tolist(),
                    "percentiles": [lo_pct, hi_pct],
                }
            alls = steps[n].get()
            if alls is not None:
                thresholds[n] = np.nanpercentile(alls, self.step_pct, axis=0).tolist()

        ctx.shared["value_bounds"] = bounds
        ctx.shared["step_thresholds"] = thresholds

        yield ctx.record(
            self.name, CheckScope.DATASET, Severity.INFO, CheckStatus.PASSED,
            f"profiled {len(bounds)} numeric modalities "
            f"({self.low_pct}-{self.high_pct} pct bounds, p{self.step_pct} step thresholds)",
            observed=f"modalities={list(bounds)}", expected="n/a",
            bounds={k: {"lo": np.round(v["lo"], 4).tolist(), "hi": np.round(v["hi"], 4).tolist()}
                    for k, v in bounds.items()},
            step_thresholds={k: np.round(v, 4).tolist() for k, v in thresholds.items()},
        )


class OutlierEpisodeCheck(DatasetCheck):
    """Episodes whose length is a statistical outlier for the dataset.

    Uses the median/MAD rather than mean/std: with 50 episodes a couple of very
    long ones would inflate an SD-based threshold enough to hide themselves.
    """

    name = "numerical.episode_outlier"
    severity = Severity.WARNING
    description = "Episode lengths are within a robust threshold of the dataset median."

    def __init__(self, mad_factor: float = 5.0):
        self.mad_factor = mad_factor

    def check_dataset(self, ctx: CheckContext) -> Iterator[QCRecord]:
        """Flag episode lengths far from the dataset median, using median/MAD."""
        lengths = np.array([e.num_frames for e in ctx.episodes], dtype=np.float64)
        if lengths.size < 5:
            return
        med = float(np.median(lengths))
        mad = float(np.median(np.abs(lengths - med))) or 1.0
        lo, hi = med - self.mad_factor * mad, med + self.mad_factor * mad
        ctx.shared["length_stats"] = {"median": med, "mad": mad, "lo": lo, "hi": hi}
        for ep, n in zip(ctx.episodes, lengths):
            if n < lo or n > hi:
                yield ctx.record(
                    self.name, CheckScope.EPISODE, self.severity, CheckStatus.WARNED,
                    f"episode {ep.episode_id}: {int(n)} frames is an outlier "
                    f"(median {med:.0f}, MAD {mad:.0f})",
                    episode_id=ep.episode_id,
                    observed=int(n), expected=f"[{lo:.0f}, {hi:.0f}]",
                )


NUMERICAL_CHECKS = [
    RangeProfiler(),
    FiniteValueCheck(),
    ValueRangeCheck(),
    DiscontinuityCheck(),
    StuckSensorCheck(),
    OutlierEpisodeCheck(),
]
