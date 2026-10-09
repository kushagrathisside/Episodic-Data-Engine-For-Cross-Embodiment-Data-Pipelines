"""Episode-level health features, computed inside the pass that already runs.

Two things live here.

:class:`EpisodeFeatureProfiler` is a validation check that emits no findings.
It exists purely to reduce each episode to a fixed-length numeric vector while
the QC engine is already streaming that episode's frames.  Registering it as a
check rather than adding a traversal is the whole point: the constraint from
docs/12 Q12 is that observability rides the existing pass, and a second scan of
55,000 frames to compute drift features would violate it for no benefit.

:func:`summarize_episode_health` turns those features plus the run's QC findings
into one :class:`EpisodeHealthRow` per episode.

**Why episode-level.**  The unit of analysis for drift is the episode, never the
frame.  ALOHA records at 50 fps; frame *t* and frame *t+1* of a 14-DoF arm are
nearly the same vector, so a two-sample test over 55,000 frames has enormous
nominal power and will report p < 0.001 for differences that are physically
meaningless.  That is the same error the original ``value_range`` check made --
a valid procedure applied at the wrong unit of analysis, firing on every
episode.  Reducing to one vector per episode first makes the population
genuinely closer to independent, and makes the statistic defensible rather than
merely computed (docs/13 §5).
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa

from ..schemas.core import Episode
from ..schemas.modality import ModalityKind, StoragePlane
from ..schemas.qc import CheckScope, CheckStatus, QCRecord, Severity
from ..validation.base import CheckContext, EpisodeCheck
from .events import EpisodeHealthRow

#: Key under which the profiler publishes ``{episode_id: {feature: value}}``.
SHARED_KEY = "episode_features"

#: Key under which the profiler publishes per-modality sensor-health counters.
#: Accumulated across the episode pass rather than per episode, because these
#: are dataset-level rates and a per-episode breakdown of them belongs in the
#: findings table, not in a metric.
MODALITY_HEALTH_KEY = "modality_health"

#: Check-name prefixes mapped onto the three health sub-scores.  Kept separate
#: rather than averaged into one number because "temporally perfect but
#: misaligned" and "jittery but aligned" are different failures with different
#: owners, and a single score hides which one you have.
_FAMILIES: dict[str, tuple[str, ...]] = {
    "temporal_quality": ("temporal.",),
    "numeric_quality": ("numerical.",),
    "alignment_quality": ("sync.", "structural.video"),
}


def _vectors(frames: pa.Table, column: str) -> np.ndarray | None:
    """Fixed-size-list column -> (n_frames, width) float array, or None."""
    if column not in frames.schema.names:
        return None
    field = frames.schema.field(column)
    width = getattr(field.type, "list_size", None)
    col = frames.column(column).combine_chunks()
    try:
        arr = np.asarray(col.flatten(), dtype=np.float64)
    except (ValueError, pa.ArrowInvalid):
        return None
    if arr.size == 0 or frames.num_rows == 0:
        return None
    if not width or width <= 0:
        # Not a fixed-size-list column with a known width -- fall back to
        # the flattened length, but guard the reshape below explicitly
        # rather than letting a ValueError escape this "never emits a
        # finding" observer and crash the whole episode QC pass.
        width = arr.size // frames.num_rows
    if width * frames.num_rows != arr.size:
        # A fixed-size-list column with nulls in some rows can flatten to
        # fewer elements than num_rows * width; reshaping anyway would
        # either raise (reshape's own ValueError) or, if we'd derived width
        # from arr.size instead of the schema, silently produce a wrong
        # width via truncating integer division. Neither is acceptable from
        # a component whose own docstring promises it emits no findings.
        return None
    return arr.reshape(frames.num_rows, width)


def _finite(a: np.ndarray) -> np.ndarray:
    """Drop non-finite rows so one NaN cannot poison every feature.

    A NaN here would propagate into the drift baseline and silently disable
    comparison, which is the failure mode where telemetry looks healthy while
    it has stopped measuring anything.
    """
    return a[np.isfinite(a).all(axis=1)] if a.ndim == 2 else a[np.isfinite(a)]


def trajectory_features(
    state: np.ndarray | None,
    action: np.ndarray | None,
    dt: float,
) -> dict[str, float]:
    """Reduce one episode's trajectories to a fixed-length feature vector.

    Chosen to be physically interpretable rather than merely available.  Each
    answers a question a robotics engineer would actually ask of a
    demonstration: how far did it travel, how fast, how jerky, how hard did the
    controller push, and did the arm follow what it was told.

    Derivatives are computed with ``np.diff`` over the true sample period rather
    than per-index, so a dataset recorded at 50 fps and one at 30 fps produce
    comparable numbers -- which is required before ALOHA and SO-100 features can
    ever sit in the same table.
    """
    f: dict[str, float] = {}
    dt = max(float(dt), 1e-6)

    if state is not None:
        s = _finite(state)
        if s.shape[0] >= 3:
            vel = np.diff(s, axis=0) / dt
            acc = np.diff(vel, axis=0) / dt
            jerk = np.diff(acc, axis=0) / dt if acc.shape[0] >= 2 else None
            step = np.linalg.norm(np.diff(s, axis=0), axis=1)
            speed = np.linalg.norm(vel, axis=1)
            f["path_length"] = float(step.sum())
            f["state_range"] = float(np.linalg.norm(s.max(axis=0) - s.min(axis=0)))
            f["speed_mean"] = float(speed.mean())
            f["speed_std"] = float(speed.std())
            f["speed_p95"] = float(np.percentile(speed, 95))
            f["accel_mean"] = float(np.linalg.norm(acc, axis=1).mean())
            if jerk is not None and jerk.size:
                f["jerk_mean"] = float(np.linalg.norm(jerk, axis=1).mean())
            # Fraction of the episode spent essentially stationary.  A rising
            # dwell fraction is what a teleoperator hesitating, or a gripper
            # failing to close, looks like in aggregate.
            f["dwell_fraction"] = float((speed < 0.01 * max(speed.max(), 1e-9)).mean())

    if action is not None:
        a = _finite(action)
        if a.shape[0] >= 2:
            mag = np.linalg.norm(a, axis=1)
            f["action_magnitude_mean"] = float(mag.mean())
            f["action_magnitude_p95"] = float(np.percentile(mag, 95))
            f["action_energy"] = float((mag ** 2).sum() * dt)

    if state is not None and action is not None:
        s, a = _finite(state), _finite(action)
        n = min(s.shape[0], a.shape[0])
        if n >= 2 and s.shape[1] == a.shape[1]:
            # How far the commanded pose sits from the achieved pose.  Rises
            # when the arm is loaded, obstructed, or mis-calibrated -- a
            # data-derived indicator of a physical problem, not of a bad file.
            f["state_action_divergence"] = float(
                np.linalg.norm(a[:n] - s[:n], axis=1).mean()
            )
    return f


class EpisodeFeatureProfiler(EpisodeCheck):
    """Computes drift features during the episode pass.  Emits no findings.

    Registered as a check because that is where the streaming pass already is.
    Its severity is INFO and it never yields a record: it is an observer riding
    the validation pass, not a validator.
    """

    name = "observability.episode_features"
    severity = Severity.INFO
    description = "Episode-level trajectory features for drift monitoring."
    required_columns = ("frame_id", "timestamp_s")
    #: After the profilers that seed shared state, before nothing in particular.
    priority = -50

    def dynamic_columns(self, ctx: CheckContext) -> list[str]:
        """State and action columns, whose names depend on the modality set."""
        return [
            m.name
            for m in ctx.manifest.modalities
            if m.plane is StoragePlane.INLINE
            and m.kind in (ModalityKind.ROBOT_STATE, ModalityKind.ROBOT_ACTION,
                           ModalityKind.HAND_STATE, ModalityKind.FORCE_TORQUE)
        ]

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Reduce this episode to a feature vector and publish it."""
        state_col = next(
            (m.name for m in ctx.manifest.modalities
             if m.kind is ModalityKind.ROBOT_STATE), None
        )
        action_col = next(
            (m.name for m in ctx.manifest.modalities
             if m.kind is ModalityKind.ROBOT_ACTION), None
        )
        state = _vectors(frames, state_col) if state_col else None
        action = _vectors(frames, action_col) if action_col else None

        feats = trajectory_features(state, action, 1.0 / max(ctx.fps, 1e-6))
        feats["frame_count"] = float(frames.num_rows)
        feats["duration_s"] = float(episode.duration_s)

        ctx.shared.setdefault(SHARED_KEY, {})[episode.episode_id] = feats
        self._accumulate_modality_health(frames, ctx)
        return iter(())  # an observer, never a validator

    def _accumulate_modality_health(self, frames: pa.Table, ctx: CheckContext) -> None:
        """Accumulate per-modality sensor-health counters across the pass.

        Three data-derived indicators, all cheap because the frames are already
        in memory:

        ``nonfinite``   NaN/Inf count -- a sensor or a transform producing
                        garbage.
        ``saturated``   observations at or beyond the profiled rail.  Counted,
                        not judged: force and torque legitimately hit their
                        rails during contact, so the *rate* is the signal.
        ``abs_max``     running maximum magnitude, kept so the tail ratio
                        (max / p99.5) can be computed at the end.  On the ALOHA
                        dataset that ratio is 2.4x for a force channel and that
                        is normal -- which is precisely why the level cannot be
                        alerted on and a shift in it can.
        """
        health = ctx.shared.setdefault(MODALITY_HEALTH_KEY, {})
        bounds = ctx.shared.get("value_bounds", {})
        for m in ctx.manifest.modalities:
            if m.plane is not StoragePlane.INLINE:
                continue
            v = _vectors(frames, m.name)
            if v is None or v.size == 0:
                continue
            entry = health.setdefault(
                m.name,
                {"kind": m.kind.value, "nonfinite": 0, "saturated": 0,
                 "abs_max": 0.0, "observations": 0},
            )
            finite = np.isfinite(v)
            entry["nonfinite"] += int((~finite).sum())
            entry["observations"] += int(v.size)
            if finite.any():
                entry["abs_max"] = max(entry["abs_max"], float(np.abs(v[finite]).max()))
            b = bounds.get(m.name)
            if b:
                lo = np.asarray(b["lo"], dtype=float)
                hi = np.asarray(b["hi"], dtype=float)
                if lo.shape[0] == v.shape[1]:
                    with np.errstate(invalid="ignore"):
                        entry["saturated"] += int(
                            ((v <= lo) | (v >= hi)).sum()
                        )


def check_bitmask(
    findings: Sequence[QCRecord], check_order: Sequence[str]
) -> int:
    """Pack which checks produced a non-passing finding into one integer.

    Makes a cohort query -- "every episode where alignment failed but structure
    passed" -- a bit test rather than a join against the findings table.  With
    24 checks the mask fits in a 32-bit int, and the ordering is passed in so it
    stays stable as checks are added.
    """
    index = {name: i for i, name in enumerate(check_order)}
    mask = 0
    for f in findings:
        i = index.get(f.check_name)
        if i is not None and f.status in (CheckStatus.FAILED, CheckStatus.WARNED):
            mask |= 1 << i
    return mask


def _family_score(findings: Sequence[QCRecord], prefixes: tuple[str, ...]) -> float:
    """1.0 when a family is clean; lower as findings accumulate.

    Weighted by severity rather than counted, so one ERROR outranks three INFOs.
    Bounded below at 0.0 -- the score says "how bad", not "how many".
    """
    weights = {Severity.CRITICAL: 1.0, Severity.ERROR: 0.5,
               Severity.WARNING: 0.15, Severity.INFO: 0.0}
    penalty = sum(
        weights.get(f.severity, 0.0)
        for f in findings
        if any(f.check_name.startswith(p) for p in prefixes)
        and f.status in (CheckStatus.FAILED, CheckStatus.WARNED)
    )
    return float(max(0.0, 1.0 - penalty))


def summarize_episode_health(
    dataset_id: str,
    version: str,
    qc_execution_id: str,
    run_id: str | None,
    episodes: Sequence[Episode],
    findings_by_episode: dict[int, list[QCRecord]],
    episode_status: dict[int, Any],
    features: dict[int, dict[str, float]],
    expected_modalities: int,
    check_order: Sequence[str],
) -> list[EpisodeHealthRow]:
    """Build one health row per episode from QC output and profiled features."""
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    rows: list[EpisodeHealthRow] = []
    for ep in episodes:
        fs = findings_by_episode.get(ep.episode_id, [])
        feats = features.get(ep.episode_id, {})
        present = len(ep.modalities or [])
        rows.append(EpisodeHealthRow(
            run_id=run_id,
            qc_execution_id=qc_execution_id,
            dataset_id=dataset_id,
            version=version,
            episode_id=ep.episode_id,
            quality_status=getattr(
                episode_status.get(ep.episode_id), "value", "unknown"
            ),
            frame_count=int(feats.get("frame_count", ep.num_frames)),
            duration_s=float(ep.duration_s),
            modalities_present=present,
            modalities_expected=int(expected_modalities),
            missing_ratio=(
                0.0 if not expected_modalities
                else max(0.0, 1.0 - present / expected_modalities)
            ),
            qc_bitmask=check_bitmask(fs, check_order),
            temporal_quality=_family_score(fs, _FAMILIES["temporal_quality"]),
            numeric_quality=_family_score(fs, _FAMILIES["numeric_quality"]),
            alignment_quality=_family_score(fs, _FAMILIES["alignment_quality"]),
            drift_features_json=json.dumps(
                {k: round(v, 6) for k, v in sorted(feats.items())}
            ),
            checked_at=now,
        ))
    return rows


#: QC check name -> the counter its findings feed.  Table-driven so a new check
#: joins the health metrics by adding a row here, not by editing the emitter.
#: Findings are the evidence a check already produced; re-deriving the same
#: facts independently would be a second source of truth about the same data.
_FINDING_COUNTERS: dict[str, str] = {
    "numerical.finite": "rde_data_nonfinite_values_total",
    "numerical.stuck_sensor": "rde_sensor_stuck_intervals_total",
    "numerical.value_range": "rde_sensor_saturation_events_total",
    "structural.frame_index_continuity": "rde_sensor_frame_dropout_total",
    "temporal.gaps": "rde_sensor_frame_dropout_total",
    "structural.video_integrity": "rde_data_decode_failures_total",
}


def _observed_float(value: str | None) -> float | None:
    """Parse a QCRecord's ``observed_value`` back to a number, if it is one.

    Observations are stored as strings so heterogeneous findings (floats,
    counts, shapes, dtype names) share one columnar table.  Values that are not
    numbers are simply skipped -- a shape mismatch has no place in a histogram.
    """
    if value is None:
        return None
    try:
        return float(str(value).rstrip("s"))
    except ValueError:
        return None


def emit_data_health(
    session: Any,
    shared: dict[str, Any],
    records: Sequence[QCRecord],
    modalities: Sequence[Any],
    fps: float,
    dataset_id: str,
) -> None:
    """Route the QC pass's by-products into the data-health metric family.

    Everything here is derived from state the validation pass already produced:
    findings that were already emitted, and profiler output already in
    ``ctx.shared``.  No data is re-read.  That is what keeps the whole
    observability layer at 0.8% of pipeline wall-clock.

    This is the family that distinguishes *"the pipeline ran"* from *"the data
    is usable"*.  A run can succeed with every stage green and produce a dataset
    whose video is five seconds out of sync -- structurally perfect, and ruined.
    """
    if session is None or not getattr(session, "enabled", False):
        return

    kinds = {m.name: getattr(m.kind, "value", str(m.kind)) for m in modalities}

    # -- counters derived from findings ------------------------------------
    for rec in records:
        metric = _FINDING_COUNTERS.get(rec.check_name)
        if metric is None or rec.status not in (CheckStatus.FAILED, CheckStatus.WARNED):
            continue
        session.inc(metric, 1.0,
                    modality=rec.modality or "unknown", dataset_id=dataset_id)

    # -- cross-modal alignment: the sixth pillar ---------------------------
    # Measured for every episode the alignment check evaluated, not only the
    # ones that failed, so the histogram describes the whole distribution
    # rather than only its tail.  An offset well inside one frame period is the
    # healthy case and must be visible as such: a metric that appears only when
    # something is broken cannot show you that nothing is.
    for modality, offset in shared.get("alignment_offsets", []):
        session.observe("rde_data_timestamp_offset_seconds", abs(float(offset)),
                        modality=modality, dataset_id=dataset_id)

    # -- sampling rate and jitter ------------------------------------------
    intervals = shared.get("interval_stats", {})
    if intervals:
        means = [s["mean"] for s in intervals.values() if s.get("mean")]
        if means:
            session.set("rde_data_sampling_interval_seconds",
                        float(np.median(means)),
                        modality="frame_clock", dataset_id=dataset_id)
        expected = 1.0 / max(fps, 1e-9)
        for stats in intervals.values():
            # Jitter as deviation from the *declared* period, not from the
            # observed mean: a stream running uniformly at the wrong rate has
            # near-zero self-consistency and is still wrong.
            if stats.get("mean") is not None:
                session.observe("rde_data_sampling_jitter_seconds",
                                abs(float(stats["mean"]) - expected),
                                modality="frame_clock", dataset_id=dataset_id)

    # -- missingness --------------------------------------------------------
    # Seeded at zero for every modality before applying observed ratios.  The
    # profiler only reports columns that actually had nulls, so without this a
    # clean dataset would emit no series at all -- and "no data" would be
    # indistinguishable from "no problem".
    ratios = shared.get("null_ratios") or {}
    for m in modalities:
        session.set("rde_data_missing_ratio", float(ratios.get(m.name, 0.0)),
                    modality=m.name, dataset_id=dataset_id)

    # -- tail ratio ---------------------------------------------------------
    bounds = shared.get("value_bounds", {})
    for name, entry in (shared.get(MODALITY_HEALTH_KEY) or {}).items():
        b = bounds.get(name)
        if not b:
            continue
        rail = max(
            abs(float(np.max(np.abs(np.asarray(b["hi"], dtype=float))))),
            abs(float(np.max(np.abs(np.asarray(b["lo"], dtype=float))))),
        )
        if rail > 1e-9 and entry["abs_max"] > 0:
            session.set("rde_sensor_tail_ratio", entry["abs_max"] / rail,
                        modality=name, modality_kind=entry["kind"],
                        dataset_id=dataset_id)
