"""Feedback loop tests: evaluation output becomes a dataset input."""

from __future__ import annotations

import numpy as np
import pytest

from rde.dataset_builder import DatasetBuilder
from rde.engine import DataEngine
from rde.evaluation import Evaluator
from rde.evaluation.evaluator import FrameOutcome, _safe_corr
from rde.feedback import FailureMode, FeedbackLoop
from rde.model_adapters import build_model
from rde.schemas.core import QualityStatus
from rde.transformations import NormalizeVector, TransformPipeline


@pytest.fixture
def view(settings, catalog, ingested):
    TransformPipeline(
        [NormalizeVector("state.joint"), NormalizeVector("action.joint")],
        settings=settings, catalog=catalog,
    ).run("testds", "raw/v1", "clean/v1")
    DatasetBuilder(settings=settings, catalog=catalog).build(
        "testds", "clean/v1", "training/v1", splits={"train": 0.5, "test": 0.5}, seed=1,
        # Fixture skips QC entirely (goes straight raw -> clean), so episodes
        # are UNKNOWN quality; admit them explicitly rather than relying on
        # DatasetBuilder's default, which excludes UNKNOWN.
        require_quality=[QualityStatus.PASSED, QualityStatus.WARNING, QualityStatus.UNKNOWN],
    )
    v = DataEngine(settings).get_dataset("testds", "training/v1")
    yield v
    v.close()


def _evaluate(view, name="nearest_neighbor"):
    model, adapter = build_model(name, view.contract)
    model.fit(view, split="train")
    return Evaluator().evaluate(view, model, adapter, split="test", persist=False)


def test_safe_corr_returns_none_for_constant_input():
    """float32 constants have non-zero std; the guard must be relative."""
    const = np.full(100, 0.1, dtype=np.float32)
    assert const.std() > 0, "precondition: float32 constant has tiny non-zero std"
    assert _safe_corr(const, np.arange(100.0)) is None


def test_safe_corr_computes_real_correlation():
    x = np.arange(100.0)
    assert _safe_corr(x, 2 * x) == pytest.approx(1.0, abs=1e-6)
    assert _safe_corr(x, -2 * x) == pytest.approx(-1.0, abs=1e-6)


def test_evaluation_emits_per_frame_outcomes(view):
    res = _evaluate(view)
    assert res.outcomes, "no per-frame outcomes emitted"
    assert len(res.outcomes) == res.frames_evaluated
    o = res.outcomes[0]
    assert o.frame_id >= 0 and 0.0 <= o.confidence <= 1.0


def test_metrics_include_interpretable_breakdown(view):
    res = _evaluate(view)
    m = res.metrics
    assert len(m["per_dimension_mae"]) == 6
    assert 0.0 <= m["success_rate"] <= 1.0
    assert m["latency_ms_per_frame"] >= 0


def test_baseline_comparison_is_reported_as_ratio(view):
    res = _evaluate(view, "nearest_neighbor")
    base = _evaluate(view, "mean_action")
    Evaluator().compare(res, base)
    assert res.baseline["mse_ratio"] > 0
    assert res.baseline["improvement"] in ("better", "worse")


def test_failures_are_detected_and_classified(settings, catalog, view):
    res = _evaluate(view)
    report = FeedbackLoop(settings=settings, catalog=catalog).detect(res, error_percentile=80.0)
    assert report.failures, "no failures flagged at the 80th percentile"
    modes = {f.mode for f in report.failures}
    assert modes <= {FailureMode.HIGH_ERROR, FailureMode.LOW_CONFIDENCE, FailureMode.BOTH}


def test_thresholds_are_relative_not_absolute(settings, catalog, view):
    """Thresholds must adapt to the run's own error scale."""
    res = _evaluate(view)
    loop = FeedbackLoop(settings=settings, catalog=catalog)
    strict = loop.detect(res, error_percentile=95.0, persist=False)
    loose = loop.detect(res, error_percentile=50.0, persist=False)
    assert len(loose.failures) > len(strict.failures)


def test_episodes_nominated_with_evidence(settings, catalog, view):
    res = _evaluate(view)
    report = FeedbackLoop(settings=settings, catalog=catalog).detect(
        res, error_percentile=70.0, min_failures_per_episode=1
    )
    assert report.reviews, "no episodes nominated for review"
    r = report.reviews[0]
    assert r.failure_frames > 0
    assert r.total_frames_evaluated >= r.failure_frames
    assert r.recommendation
    assert report.reviews == sorted(report.reviews, key=lambda x: -x.priority)


def test_quarantine_marks_but_does_not_delete(settings, catalog, view):
    """Flagged episodes must remain readable; quarantine is not deletion."""
    res = _evaluate(view)
    loop = FeedbackLoop(settings=settings, catalog=catalog)
    report = loop.detect(res, error_percentile=70.0, min_failures_per_episode=1)
    ids = [r.episode_id for r in report.reviews]
    loop.quarantine_episodes("testds", "training/v1", ids)

    rows = catalog.get_episodes("testds", "training/v1")
    quarantined = [r for r in rows if r["quality_status"] == QualityStatus.QUARANTINED.value]
    assert len(quarantined) == len(ids)
    assert len(rows) == 4, "quarantine removed episodes instead of marking them"


def test_retraining_selection_upweights_rather_than_drops(settings, catalog, view):
    res = _evaluate(view)
    loop = FeedbackLoop(settings=settings, catalog=catalog)
    report = loop.detect(res, error_percentile=70.0, min_failures_per_episode=1)
    sel = loop.build_retraining_selection(report, oversample_factor=3)
    assert sel["flagged_episodes"]
    for ep in sel["flagged_episodes"]:
        assert sel["episode_weights"][ep] == 3.0
    assert "upweighted, not removed" in sel["rationale"]


def test_feedback_artifacts_are_persisted(settings, catalog, view):
    res = _evaluate(view)
    report = FeedbackLoop(settings=settings, catalog=catalog).detect(res, error_percentile=70.0)
    assert report.artifacts
    from pathlib import Path

    assert Path(report.artifacts["review"]).exists()


def test_empty_evaluation_is_handled(settings, catalog, view):
    """A split with no frames must not crash the loop."""
    from rde.evaluation.evaluator import EvaluationResult
    import datetime as dt

    empty = EvaluationResult(
        dataset_id="testds", version="training/v1", split="val", model_name="x",
        adapter_name="y", metrics={}, outcomes=[],
        started_at=dt.datetime.now(dt.timezone.utc),
        finished_at=dt.datetime.now(dt.timezone.utc),
    )
    report = FeedbackLoop(settings=settings, catalog=catalog).detect(empty)
    assert report.failures == [] and report.reviews == []


def test_constant_confidence_does_not_flag_every_frame(settings, catalog, view):
    """A model with a constant confidence (no uncertainty estimate) must be
    judged on error alone; otherwise every frame sits at its own percentile
    and the whole split is flagged as low-confidence."""
    import dataclasses

    res = _evaluate(view)
    flat = dataclasses.replace(
        res, outcomes=[dataclasses.replace(o, confidence=0.5) for o in res.outcomes]
    )
    report = FeedbackLoop(settings=settings, catalog=catalog).detect(
        flat, error_percentile=90.0, persist=False
    )
    assert report.thresholds["confidence_used"] is False
    assert report.thresholds["confidence_threshold"] is None
    assert all(f.mode is FailureMode.HIGH_ERROR for f in report.failures)
    assert len(report.failures) < len(flat.outcomes)


def test_normalized_mae_ignores_constant_dimensions():
    """A dimension constant over the evaluated frames must not blow the metric
    up by dividing by a near-zero spread."""
    from rde.evaluation.evaluator import _normalized_mae

    T = np.column_stack([np.linspace(-1, 1, 32), np.full(32, 0.25) + 1e-9 * np.arange(32)])
    E = np.full_like(T, 0.1)
    value, dims = _normalized_mae(E, T)
    assert dims == 1
    assert value == pytest.approx(0.1 / T[:, 0].std())
    assert _normalized_mae(E, np.zeros_like(T)) == (None, 0)
