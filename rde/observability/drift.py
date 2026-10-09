"""Distribution drift between two dataset versions, at episode granularity.

The unit of analysis is the **episode**, and that choice is the substance of
this module rather than an implementation detail.

Robotics trajectories are strongly autocorrelated.  ALOHA records at 50 fps, so
frame *t* and frame *t+1* of a 14-DoF arm are nearly the same vector.  A
two-sample test over 55,000 such frames has enormous nominal power and will
return p < 0.001 for differences with no physical meaning -- the same error the
original ``value_range`` check made, a valid procedure applied at the wrong unit
of analysis, firing on every episode.  The statistical literature is explicit
that effective sample size collapses under autocorrelation and that block
methods achieve nominal coverage only when it stays large (docs/13 §5).

So: reduce each episode to a fixed-length feature vector first
(:mod:`rde.observability.health`), then compare *populations of episodes*.
Fifty independent demonstrations is a more honest sample than 55,000 correlated
frames, and the effective sample size is reported next to every statistic --
because a large statistic computed on six effective episodes is noise wearing a
p-value, and publishing it without its ESS is how a drift system loses trust.

Statistics offered:

``psi``          Population Stability Index -- interpretable, banking-standard,
                 thresholds people already have intuitions about.
``wasserstein``  Earth-mover distance, normalised by the baseline's spread so
                 features in different units stay comparable.
``mean_shift``   Difference in means in baseline standard deviations.  The one
                 an engineer can act on without reading a statistics textbook.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .events import DriftReportRow

#: Default alert thresholds.  These are STARTING HEURISTICS, not engineering
#: limits, and are marked as such everywhere they appear.  PSI's 0.1/0.25
#: convention comes from credit-risk practice, not from robotics; it must be
#: recalibrated against a production baseline before anyone pages on it.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "psi": 0.25,
    "wasserstein": 0.5,
    "mean_shift": 2.0,
}

#: Below this many episodes on either side, a comparison is reported but never
#: flagged.  Ten is already generous; it is a floor against absurdity, not a
#: claim of sufficiency.
MIN_EPISODES = 10


def effective_sample_size(x: np.ndarray) -> float:
    """Sample size adjusted for lag-1 autocorrelation.

    ``n_eff = n * (1 - r) / (1 + r)`` -- the standard first-order correction.
    Episode-level features should be close to independent, so ``n_eff`` near
    ``n`` is the expected result; a value well below ``n`` means the episodes
    are not independent either (a single session, one operator, one afternoon),
    and that is worth surfacing rather than hiding.
    """
    n = len(x)
    if n < 3:
        return float(n)
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return float(n)
    r = float((x[:-1] * x[1:]).sum() / denom)
    r = max(min(r, 0.99), -0.99)
    return float(n * (1.0 - r) / (1.0 + r))


def psi(baseline: np.ndarray, candidate: np.ndarray, bins: int | None = None) -> float:
    """Population Stability Index over quantile bins of the baseline.

    Quantile bins rather than equal-width, so a heavy-tailed feature -- which
    force and torque channels are -- does not put 99% of its mass in one bucket
    and report stability it does not have.

    **Bin count scales with sample size, and this is not a detail.**  PSI's
    familiar 0.1/0.25 thresholds come from credit-risk practice, where samples
    run to tens of thousands.  Fixing ``bins=10`` at 50 episodes leaves ~5
    samples per bin, and the estimator's own sampling variance then exceeds
    0.25 on data with no drift at all -- measured, not hypothesised: a null
    comparison of two 50-episode draws from the same distribution returned
    PSI = 0.49.  Borrowing a threshold across three orders of magnitude of
    sample size is the same category of error as applying an IID test to
    autocorrelated frames.

    So bins are chosen to keep roughly ten samples each, floored at 3.  Use
    :func:`calibrate_threshold` to replace the borrowed threshold with one
    measured against your own null.
    """
    if len(baseline) < 2 or len(candidate) < 2:
        return 0.0
    if bins is None:
        bins = int(min(10, max(3, min(len(baseline), len(candidate)) // 10)))
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    b, _ = np.histogram(baseline, bins=edges)
    c, _ = np.histogram(candidate, bins=edges)
    # Laplace smoothing: an empty bin on one side must not make PSI infinite.
    bp = (b + 0.5) / (b.sum() + 0.5 * len(b))
    cp = (c + 0.5) / (c.sum() + 0.5 * len(c))
    return float(np.sum((cp - bp) * np.log(cp / bp)))


def wasserstein(baseline: np.ndarray, candidate: np.ndarray) -> float:
    """1-D earth-mover distance, scaled by the baseline's standard deviation.

    Scaling is what lets ``path_length`` (metres) and ``dwell_fraction``
    (unitless) share a threshold.  Computed from sorted quantiles rather than
    via SciPy, keeping the dependency set unchanged.
    """
    if len(baseline) < 2 or len(candidate) < 2:
        return 0.0
    q = np.linspace(0, 1, 101)
    d = float(np.abs(np.quantile(candidate, q) - np.quantile(baseline, q)).mean())
    scale = float(np.std(baseline))
    return d / scale if scale > 1e-12 else 0.0


def mean_shift(baseline: np.ndarray, candidate: np.ndarray) -> float:
    """Difference in means, in baseline standard deviations.

    Uses a relative-tolerance guard on the denominator rather than an absolute
    one.  An absolute ``1e-9`` guard is what let a float32 constant vector with
    std 7.45e-09 through and put NaNs into the feedback loop's correlation
    inputs -- the same bug class, avoided the same way.
    """
    if len(baseline) < 2 or len(candidate) < 2:
        return 0.0
    b, c = np.asarray(baseline, float), np.asarray(candidate, float)
    sd = float(np.std(b))
    scale = max(abs(float(np.mean(b))), 1.0)
    if sd <= 1e-9 * scale:
        return 0.0 if abs(float(np.mean(c)) - float(np.mean(b))) <= 1e-9 * scale else math.inf
    return float(abs(np.mean(c) - np.mean(b)) / sd)


STATISTICS = {"psi": psi, "wasserstein": wasserstein, "mean_shift": mean_shift}


def calibrate_threshold(
    baseline: np.ndarray,
    statistic: str = "psi",
    quantile: float = 0.95,
    trials: int = 200,
    seed: int = 0,
) -> float:
    """Measure a statistic's null distribution and return its upper quantile.

    Arbitrary thresholds must not be presented as universal robotics
    standards, and the false positive above is why.  This
    replaces a borrowed constant with a measured one: split the baseline in half
    at random many times, compute the statistic on two halves of the *same*
    population, and take the 95th percentile of the resulting values.  Anything
    below that is what this feature does when nothing has changed.

    The result is a **statistical baseline**, not an engineering limit.  It
    describes this feature at this sample size, and must be recomputed when
    either changes.
    """
    fn = STATISTICS.get(statistic)
    x = np.asarray(baseline, dtype=float)
    if fn is None or len(x) < 2 * MIN_EPISODES:
        return DEFAULT_THRESHOLDS.get(statistic, math.inf)
    rng = np.random.default_rng(seed)
    half = len(x) // 2
    values = []
    for _ in range(trials):
        idx = rng.permutation(len(x))
        v = float(fn(x[idx[:half]], x[idx[half: 2 * half]]))
        if math.isfinite(v):
            values.append(v)
    if not values:
        return DEFAULT_THRESHOLDS.get(statistic, math.inf)
    return float(np.quantile(values, quantile))


@dataclass
class DriftResult:
    """One feature, one statistic, with the context needed to trust it."""

    feature: str
    statistic: str
    value: float
    threshold: float
    exceeded: bool
    baseline_n: int
    candidate_n: int
    effective_sample_size: float
    underpowered: bool

    def to_row(
        self, dataset_id: str, baseline_version: str,
        candidate_version: str, run_id: str | None,
    ) -> DriftReportRow:
        return DriftReportRow(
            run_id=run_id,
            dataset_id=dataset_id,
            baseline_version=baseline_version,
            candidate_version=candidate_version,
            feature=self.feature,
            statistic=self.statistic,
            value=round(self.value, 6),
            threshold=self.threshold,
            exceeded=self.exceeded,
            baseline_n=self.baseline_n,
            candidate_n=self.candidate_n,
            effective_sample_size=round(self.effective_sample_size, 3),
            computed_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            details_json=json.dumps({"underpowered": self.underpowered}),
        )


def features_from_rows(rows: Sequence[Any]) -> dict[str, np.ndarray]:
    """Unpack ``EpisodeHealthRow.drift_features_json`` into feature arrays.

    Only features present in *every* episode are kept.  A feature computed for
    some episodes and not others produces a comparison over two different
    populations, which looks like drift and is not.
    """
    parsed: list[dict[str, float]] = []
    for r in rows:
        raw = r.drift_features_json if hasattr(r, "drift_features_json") else r
        try:
            parsed.append(json.loads(raw) if isinstance(raw, str) else dict(raw))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return {}
    common = set(parsed[0])
    for p in parsed[1:]:
        common &= set(p)
    return {
        k: np.array([p[k] for p in parsed], dtype=float)
        for k in sorted(common)
        if all(math.isfinite(float(p[k])) for p in parsed)
    }


def compare(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    statistics: Sequence[str] = ("psi", "mean_shift"),
    thresholds: dict[str, float] | None = None,
    min_episodes: int = MIN_EPISODES,
    calibrated: bool = True,
) -> list[DriftResult]:
    """Compare two episode-feature populations, feature by feature.

    A result is only ``exceeded`` when it both crosses its threshold *and* has
    enough effective sample size to mean anything.  Underpowered comparisons are
    still returned, flagged, and reported -- suppressing them entirely would
    hide that the system is not currently able to detect drift, which is itself
    the thing an operator needs to know.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    out: list[DriftResult] = []
    for feature in sorted(set(baseline) & set(candidate)):
        b, c = baseline[feature], candidate[feature]
        n_eff = min(effective_sample_size(b), effective_sample_size(c))
        underpowered = (
            len(b) < min_episodes or len(c) < min_episodes or n_eff < min_episodes
        )
        for stat_name in statistics:
            fn = STATISTICS.get(stat_name)
            if fn is None:
                continue
            value = float(fn(b, c))
            # Threshold precedence, most trustworthy first:
            #   1. one the caller passed explicitly
            #   2. one measured from this feature's own null distribution
            #   3. the documented starting heuristic
            # A measured null beats a borrowed constant every time there is
            # enough baseline to measure one.
            if thresholds and stat_name in thresholds:
                limit = thresholds[stat_name]
            elif calibrated:
                limit = calibrate_threshold(b, stat_name)
            else:
                limit = th.get(stat_name, math.inf)
            out.append(DriftResult(
                feature=feature,
                statistic=stat_name,
                value=value,
                threshold=limit,
                exceeded=bool(math.isfinite(value) and value > limit and not underpowered),
                baseline_n=len(b),
                candidate_n=len(c),
                effective_sample_size=n_eff,
                underpowered=underpowered,
            ))
    return out


def emit(session: Any, results: Sequence[DriftResult], dataset_id: str,
         baseline_version: str, candidate_version: str) -> None:
    """Route drift results into metrics and event rows.

    ``feature`` is a safe metric label -- it is bounded by the extractor, which
    is code.  The per-version comparison detail goes to event rows, because
    ``dataset_version`` pairs multiply without bound over time.
    """
    if session is None or not getattr(session, "enabled", False):
        return
    for r in results:
        session.set("rde_data_distribution_drift", r.value,
                    feature=r.feature, dataset_id=dataset_id)
        session.set("rde_data_drift_effective_sample_size",
                    r.effective_sample_size,
                    feature=r.feature, dataset_id=dataset_id)
    session.record_drift([
        r.to_row(dataset_id, baseline_version, candidate_version,
                 session.ctx.run_id)
        for r in results
    ])
