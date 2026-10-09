"""Observability must itself be tested, and tested the same way QC is.

A clean dataset cannot prove QC detects defects, which is why the engine ships
nine fault injectors.  The same argument applies here: a telemetry layer that
was only ever exercised on a healthy run proves nothing about the case it exists
for.  So these tests inject failures -- a crashing stage, a dead exporter, a
forbidden label, a shifted distribution -- and assert on what the telemetry
actually says.

The distinction that matters throughout: telemetry records what happened and
never changes it.  Several tests exist purely to prove that a broken
observability layer leaves the pipeline's behaviour untouched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from rde.catalog import Catalog
from rde.observability import (
    EventBuffer,
    EventWriter,
    LabelPolicyError,
    MetricRegistry,
    ObservabilitySession,
    RunContext,
    StageRunRow,
    TextfileExporter,
    build_registry,
    current_session,
    render_prometheus,
    stale_after,
)
from rde.observability.drift import (
    calibrate_threshold,
    compare,
    effective_sample_size,
    psi,
)
from rde.observability.exporters import iter_prom_lines
from rde.observability.health import check_bitmask, trajectory_features
from rde.observability.metrics import normalize_error_class


# -- the label policy ------------------------------------------------------

def test_forbidden_label_raises_with_the_reason():
    """episode_id must be impossible to attach, not merely discouraged.

    The measured cost is 264 billion series at the planning scale; a rule that
    expensive cannot rely on reviewer vigilance.
    """
    r = build_registry(MetricRegistry())
    c = r.get("rde_qc_findings_total")
    with pytest.raises(LabelPolicyError, match="unbounded identifier"):
        c.inc(1.0, episode_id="17")


def test_unknown_label_raises_rather_than_silently_creating_a_series():
    r = build_registry(MetricRegistry())
    with pytest.raises(LabelPolicyError, match="not a known label"):
        r.get("rde_qc_findings_total").inc(1.0, camera_serial="A1B2C3")


def test_safe_labels_are_accepted_and_cardinality_stays_bounded():
    """The whole 24 x 4 x 4 product is still a three-figure series count."""
    r = build_registry(MetricRegistry())
    c = r.get("rde_qc_findings_total")
    for check in (f"check.{i:02d}" for i in range(24)):
        for status in ("passed", "failed", "warned", "skipped"):
            c.inc(1.0, check_name=check, status=status)
    assert c.cardinality == 96
    assert r.cardinality == 96


def test_error_class_vocabulary_is_bounded():
    """An unbounded exception hierarchy would be an unbounded label."""
    assert normalize_error_class("ValueError") == "ValueError"
    assert normalize_error_class("SomeVendorSpecificError") == "other"
    assert normalize_error_class(None) == "none"


# -- metric semantics ------------------------------------------------------

def test_nan_gauge_is_dropped_not_exported():
    """A NaN gauge poisons every aggregate downstream, invisibly.

    Same failure class as the float32 constant-variance bug that put NaNs into
    the feedback loop.  A missing series is visible; a NaN series is not.
    """
    r = build_registry(MetricRegistry())
    g = r.get("rde_data_modality_coverage_ratio")
    g.set(float("nan"), modality="rgb.top")
    assert g.cardinality == 0
    g.set(0.5, modality="rgb.top")
    assert g.value(modality="rgb.top") == 0.5


def test_histogram_buckets_are_cumulative_and_include_inf():
    r = build_registry(MetricRegistry())
    h = r.get("rde_pipeline_stage_duration_seconds")
    for v in (0.002, 0.7, 42.0):
        h.observe(v, stage="validate")
    samples = {s.series_key: s.value for s in h.samples()}
    assert samples['rde_pipeline_stage_duration_seconds_bucket{stage="validate",le="0.005"}'] == 1
    assert samples['rde_pipeline_stage_duration_seconds_bucket{stage="validate",le="1"}'] == 2
    assert samples['rde_pipeline_stage_duration_seconds_bucket{stage="validate",le="+Inf"}'] == 3
    assert samples['rde_pipeline_stage_duration_seconds_count{stage="validate"}'] == 3


def test_counter_rejects_negative_increment():
    r = build_registry(MetricRegistry())
    with pytest.raises(ValueError):
        r.get("rde_qc_findings_total").inc(-1.0, check_name="x")


def test_registry_redeclaration_returns_the_same_instrument():
    """A re-import must not split one metric into two silently."""
    r = MetricRegistry()
    a = r.counter("rde_x_total", "doc")
    b = r.counter("rde_x_total", "doc")
    assert a is b


# -- exporters -------------------------------------------------------------

def test_export_is_atomic_and_parseable(tmp_path: Path):
    r = build_registry(MetricRegistry())
    r.get("rde_qc_findings_total").inc(3.0, check_name="temporal.gaps")
    out = tmp_path / "rde.prom"
    assert TextfileExporter(out, r).export() == out
    parsed = dict(iter_prom_lines(out.read_text()))
    assert parsed['rde_qc_findings_total{check_name="temporal.gaps"}'] == 3.0
    # No temp files survive a successful write.
    assert not list(tmp_path.glob(".*tmp"))


def test_missing_export_is_stale_by_definition(tmp_path: Path):
    """"The run never happened" and "the run happened long ago" both alert.

    This is exactly what a Pushgateway cannot express: it has no `up` metric,
    so absence is indistinguishable from a stale cached push.
    """
    assert stale_after(tmp_path / "never-written.prom", 60.0) is True


def test_exporter_failure_does_not_propagate(tmp_path: Path):
    """A dead exporter degrades observability. It must not fail a run."""
    r = build_registry(MetricRegistry())
    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory")
    exporter = TextfileExporter(blocked / "nested" / "rde.prom", r)
    assert exporter.export() is None  # returns None, does not raise
    assert r.get("rde_observability_export_failures_total").cardinality == 1


def test_render_emits_help_and_type_for_every_instrument():
    text = render_prometheus(build_registry(MetricRegistry()))
    assert text.count("# HELP ") == text.count("# TYPE ")
    assert "# TYPE rde_qc_findings_total counter" in text


# -- the correlation spine -------------------------------------------------

def test_spans_nest_and_carry_parent_ids():
    ctx = RunContext.start("ds", "0.1")
    with ctx.span("outer"):
        with ctx.span("inner"):
            pass
    outer, inner = ctx.spans
    assert outer.parent_span_id is None
    assert inner.parent_span_id == outer.span_id
    assert outer.status == "ok"


def test_span_records_failure_and_reraises():
    ctx = RunContext.start("ds", "0.1")
    with pytest.raises(ValueError):
        with ctx.span("boom"):
            raise ValueError("x")
    assert ctx.spans[0].status == "error"
    assert ctx.spans[0].error_class == "ValueError"


def test_trace_and_span_ids_are_otel_shaped():
    """32 and 16 hex chars, so the ids drop into any OTel backend unmodified."""
    ctx = RunContext.start("ds", "0.1")
    with ctx.span("s"):
        pass
    assert len(ctx.trace_id) == 32 and int(ctx.trace_id, 16) >= 0
    assert len(ctx.spans[0].span_id) == 16


def test_session_is_absent_outside_a_run():
    """Library use must be unaffected: no run, no session, no error."""
    assert current_session() is None


def test_disabled_session_records_nothing_but_still_works():
    s = ObservabilitySession.disabled()
    with s.active():
        with s.stage("ingest") as h:
            h.counts(records_out=100)
    assert s.finish() == {}
    assert s.events.counts() == {}


# -- stage instrumentation -------------------------------------------------

def test_stage_failure_produces_metric_span_and_event(tmp_path: Path):
    sess = ObservabilitySession(
        RunContext.start("ds", "0.1"),
        registry=build_registry(MetricRegistry()),
        event_dir=tmp_path / "events",
    )
    with sess.active():
        with pytest.raises(ValueError):
            with sess.stage("validate"):
                raise ValueError("bad frames")
    sess.finish("error")

    failures = sess.registry.get("rde_pipeline_stage_failures_total")
    assert failures.value(stage="validate", dataset_id="ds",
                          error_class="ValueError") == 1.0
    rows = sess.events.get("stage_runs")
    assert rows[0].status == "error" and rows[0].error_class == "ValueError"
    assert sess.ctx.spans[0].status == "error"


def test_finish_flushes_every_artifact(tmp_path: Path):
    """The documented batch-job failure is losing the last export on exit."""
    sess = ObservabilitySession(
        RunContext.start("ds", "0.1"),
        registry=build_registry(MetricRegistry()),
        event_dir=tmp_path / "events",
        metrics_path=tmp_path / "m" / "rde.prom",
        span_dir=tmp_path / "spans",
    )
    with sess.active():
        with sess.stage("ingest") as h:
            h.counts(records_out=10, episodes=2).bytes(read=1234)
    arts = sess.finish("ok")
    assert set(arts) >= {"pipeline_runs", "stage_runs", "spans", "metrics"}
    for p in arts.values():
        assert Path(p).exists()
    spans = [json.loads(l) for l in Path(arts["spans"]).read_text().splitlines()]
    assert spans[0]["trace_id"] == sess.ctx.trace_id


def test_last_run_gauges_are_set_for_batch_alerting(tmp_path: Path):
    """Counters reset per run; the gauges are what alerts must use."""
    sess = ObservabilitySession(
        RunContext.start("ds", "0.1"), registry=build_registry(MetricRegistry())
    )
    with sess.active():
        with sess.stage("ingest") as h:
            h.counts(records_out=99)
    sess.finish("ok")
    assert sess.registry.get("rde_pipeline_last_run_status").value(dataset_id="ds") == 1.0
    assert sess.registry.get("rde_pipeline_last_run_frames").value(
        stage="ingest", dataset_id="ds") == 99.0
    assert sess.registry.get(
        "rde_pipeline_last_success_timestamp_seconds").value(dataset_id="ds") > 0


# -- event rows ------------------------------------------------------------

def test_event_roundtrip_preserves_schema_on_empty_tables(tmp_path: Path):
    """A zero-row table must keep its columns, or the next append breaks."""
    w = EventWriter(tmp_path)
    assert w.read("drift_reports").num_rows == 0
    assert "effective_sample_size" in w.read("drift_reports").schema.names


def test_event_buffer_flushes_only_non_empty_tables(tmp_path: Path):
    b = EventBuffer()
    b.add("stage_runs", StageRunRow(
        run_id="r", trace_id="t", span_id="s", dataset_id="d", stage="ingest",
        seq=1, started_at="x", finished_at="y", duration_s=1.0, status="ok",
    ))
    written = b.flush(EventWriter(tmp_path), "r")
    assert set(written) == {"stage_runs"}


# -- catalog mirror --------------------------------------------------------

def test_catalog_records_run_history(settings):
    """The query recon G2 could not answer: has validate gotten slower?"""
    cat = Catalog(settings.paths.catalog_db)
    sess = ObservabilitySession(
        RunContext.start("ds", "0.1"),
        registry=build_registry(MetricRegistry()),
        catalog=cat,
    )
    with sess.active():
        with sess.stage("validate") as h:
            h.counts(records_out=5)
    sess.finish("ok")

    runs = cat.run_history("ds")
    assert len(runs) == 1 and runs[0]["status"] == "ok"
    stages = cat.stage_history("ds", "validate")
    assert len(stages) == 1 and stages[0]["records_out"] == 5


def test_qc_records_carry_the_spine(settings, ingested):
    """A finding must be traceable back to the validation that produced it."""
    from rde.qc import QCEngine

    cat = Catalog(settings.paths.catalog_db)
    sess = ObservabilitySession(
        RunContext.start("testds", "0.1"),
        registry=build_registry(MetricRegistry()),
        catalog=cat,
    )
    with sess.active():
        QCEngine(settings=settings, catalog=cat).run("testds", "raw/v1")
    sess.finish("ok")   # the catalog mirror is written on finish

    import sqlite3
    conn = sqlite3.connect(settings.paths.catalog_db)
    rows = conn.execute(
        "SELECT pipeline_run_id, qc_execution_id FROM qc_records"
    ).fetchall()
    assert rows, "expected at least one finding on the synthetic dataset"
    assert all(r[0] == sess.ctx.run_id for r in rows)
    assert all(r[1] and r[1].startswith("qc-") for r in rows)
    assert cat.qc_execution_history("testds")[0]["episodes_total"] > 0


# -- episode health --------------------------------------------------------

def test_trajectory_features_are_sampling_rate_invariant():
    """The same motion at 30 and 60 fps must give the same physical speed.

    Required before ALOHA (50 fps) and SO-100 (30 fps) features can ever be
    compared in one table.
    """
    t30 = np.linspace(0, 1, 31)[:, None] * np.ones((1, 3))
    t60 = np.linspace(0, 1, 61)[:, None] * np.ones((1, 3))
    f30 = trajectory_features(t30, None, 1 / 30)
    f60 = trajectory_features(t60, None, 1 / 60)
    assert f30["speed_mean"] == pytest.approx(f60["speed_mean"], rel=0.05)
    assert f30["path_length"] == pytest.approx(f60["path_length"], rel=0.05)


def test_features_survive_a_nan_row():
    """One NaN must not silently disable the whole drift baseline."""
    s = np.ones((20, 3))
    s[5] = np.nan
    f = trajectory_features(s, None, 1 / 30)
    assert f and all(math.isfinite(v) for v in f.values())


def test_check_bitmask_is_a_stable_bit_test():
    from rde.schemas.qc import CheckScope, CheckStatus, QCRecord, Severity

    order = ["a.one", "b.two", "c.three"]
    rec = QCRecord(
        check_name="b.two", scope=CheckScope.EPISODE, severity=Severity.ERROR,
        status=CheckStatus.FAILED, dataset_id="d", version="raw/v1",
    )
    assert check_bitmask([rec], order) == 0b010
    rec.status = CheckStatus.PASSED
    assert check_bitmask([rec], order) == 0


# -- drift -----------------------------------------------------------------

def test_borrowed_psi_threshold_false_positives_at_episode_scale():
    """The measured reason the 0.25 convention is not used unmodified.

    PSI's 0.1/0.25 thresholds come from credit-risk practice, where samples run
    to tens of thousands.  At 50 episodes with 10 fixed bins the estimator's own
    sampling variance dominates: the *median* null PSI is ~0.30, above the
    threshold, so most comparisons of a distribution against itself would alert.

    Asserted over many draws rather than one seed, because the claim is about
    the estimator's null distribution and a single lucky sample would prove
    nothing either way.
    """
    rng = np.random.default_rng(7)
    nulls = [
        psi(rng.normal(5, 1, 50), rng.normal(5, 1, 50), bins=10)
        for _ in range(200)
    ]
    assert np.mean(np.array(nulls) > 0.25) > 0.4, (
        "expected the borrowed threshold to fire on a large fraction of "
        "no-drift comparisons at episode scale"
    )
    # And it is a small-sample artefact, not a flaw in PSI: at 5,000 samples
    # the same null sits three orders of magnitude below the threshold.
    big = [
        psi(rng.normal(5, 1, 5000), rng.normal(5, 1, 5000), bins=10)
        for _ in range(20)
    ]
    assert max(big) < 0.05


def test_calibrated_drift_has_defensible_operating_characteristics():
    """Measured false-positive and detection rates, not asserted ones.

    Calibration is only worth its complexity if the resulting detector is
    actually good, so this pins the numbers: a null that almost never fires, and
    a one-standard-deviation shift that almost always does.
    """
    rng = np.random.default_rng(11)
    trials = 60

    false_positives = sum(
        any(r.exceeded for r in compare(
            {"f": rng.normal(100, 10, 50)}, {"f": rng.normal(100, 10, 50)}))
        for _ in range(trials)
    )
    detections = sum(
        any(r.exceeded for r in compare(
            {"f": rng.normal(100, 10, 50)}, {"f": rng.normal(110, 10, 50)}))
        for _ in range(trials)
    )
    assert false_positives / trials <= 0.10
    assert detections / trials >= 0.85


def test_calibrated_threshold_exceeds_the_borrowed_constant():
    rng = np.random.default_rng(0)
    base = rng.normal(100, 10, 50)
    assert calibrate_threshold(base, "psi") > 0.25


def test_drift_detects_a_real_shift_and_ignores_a_null():
    rng = np.random.default_rng(0)
    base = {"path_length": rng.normal(100, 10, 60)}
    same = {"path_length": rng.normal(100, 10, 60)}
    moved = {"path_length": rng.normal(150, 10, 60)}
    assert not any(r.exceeded for r in compare(base, same))
    assert all(r.exceeded for r in compare(base, moved))


def test_underpowered_comparison_is_reported_but_never_flagged():
    """A large statistic on six effective episodes is noise wearing a p-value."""
    rng = np.random.default_rng(0)
    base = {"f": rng.normal(0, 1, 5)}
    moved = {"f": rng.normal(9, 1, 5)}
    results = compare(base, moved)
    assert results and all(r.underpowered for r in results)
    assert not any(r.exceeded for r in results)


def test_effective_sample_size_collapses_under_autocorrelation():
    """The measured reason drift is computed per episode, not per frame."""
    rng = np.random.default_rng(0)
    independent = rng.normal(0, 1, 500)
    walk = np.cumsum(rng.normal(0, 1, 500))   # strongly autocorrelated
    assert effective_sample_size(independent) > 300
    assert effective_sample_size(walk) < 30


def test_constant_feature_does_not_produce_infinite_mean_shift():
    """Relative-tolerance guard: the float32 constant-variance bug, avoided."""
    const = np.full(40, 0.1, dtype=np.float32).astype(float)
    results = compare({"f": const}, {"f": const}, statistics=("mean_shift",))
    assert results[0].value == 0.0


# -- the sixth pillar: cross-modal alignment -------------------------------

def test_alignment_offset_is_a_continuous_signal_not_a_binary_check(
    settings, ingested, catalog
):
    """The metric general-purpose data observability does not have.

    Freshness, volume, schema, distribution and lineage all pass on a dataset
    whose video is shifted against its state -- the file is complete, typed,
    non-null and statistically unremarkable, and it is ruined.  This asserts the
    offset is measured continuously, so a healthy dataset is visibly healthy
    rather than merely un-alerting.
    """
    from rde.observability.instruments import build_registry as _br
    from rde.qc import QCEngine
    from rde.testing import inject_video_misalignment

    def mean_offset_ms() -> float:
        sess = ObservabilitySession(
            RunContext.start("testds", "0.1"), registry=_br(MetricRegistry())
        )
        with sess.active():
            QCEngine(settings=settings, catalog=catalog).run(
                "testds", "raw/v1", persist=False
            )
        h = sess.registry.get("rde_data_timestamp_offset_seconds")
        sums = {s.suffix: s.value for s in h.samples() if s.suffix in ("_sum", "_count")}
        if not sums.get("_count"):
            return 0.0
        return 1000.0 * sums["_sum"] / sums["_count"]

    clean = mean_offset_ms()
    assert clean < 1.0, "a clean dataset must measure a sub-millisecond offset"

    inject_video_misalignment(settings.paths.version_dir("testds", "raw/v1"), shift_s=5.0)
    shifted = mean_offset_ms()
    assert shifted > 4000.0, "a 5s shift must show up as a ~5000 ms offset"


# -- data contracts --------------------------------------------------------

def _contract_inputs(settings, catalog, version: str = "raw/v1"):
    """Run QC under a session and return (manifest, result, feature arrays)."""
    from rde.observability.health import SHARED_KEY
    from rde.observability.instruments import build_registry as _br
    from rde.qc import QCEngine

    sess = ObservabilitySession(
        RunContext.start("testds", "0.1"), registry=_br(MetricRegistry())
    )
    with sess.active():
        res = QCEngine(settings=settings, catalog=catalog).run(
            "testds", version, persist=False
        )
    feats = res.shared.get(SHARED_KEY, {})
    keys = sorted({k for f in feats.values() for k in f})
    arrays = {
        k: np.array([f[k] for f in feats.values() if k in f], dtype=float)
        for k in keys if sum(k in f for f in feats.values()) == len(feats)
    }
    return catalog.get_manifest("testds", version), res, arrays


def test_contract_inferred_from_a_version_accepts_that_version(settings, ingested, catalog):
    """A contract must not fail the data it was derived from."""
    from rde.observability import DataContract

    manifest, res, feats = _contract_inputs(settings, catalog)
    contract = DataContract.infer(manifest, res.episode_status, feats, name="t")
    report = contract.evaluate(
        manifest, res.episode_status,
        alignment_offsets=res.shared.get("alignment_offsets"),
    )
    assert report.released, [
        (c.family, c.name, c.observed, c.expected) for c in report.failures
    ]


def test_contract_blocks_a_missing_modality(settings, ingested, catalog):
    """The schema clause is what catches a camera silently disappearing."""
    from rde.observability import DataContract

    manifest, res, feats = _contract_inputs(settings, catalog)
    contract = DataContract.infer(manifest, res.episode_status, feats, name="t")
    contract.required_modalities = contract.required_modalities + ["rgb.nonexistent"]
    report = contract.evaluate(manifest, res.episode_status)
    assert not report.released
    assert any("rgb.nonexistent" in c.name for c in report.blocking)


def test_contract_blocks_an_alignment_violation(settings, ingested, catalog):
    """Tolerance is expressed in frame periods, so it travels between datasets."""
    from rde.observability import DataContract

    manifest, res, feats = _contract_inputs(settings, catalog)
    contract = DataContract.infer(manifest, res.episode_status, feats, name="t")
    period = 1.0 / (manifest.fps or 30.0)
    report = contract.evaluate(
        manifest, res.episode_status,
        alignment_offsets=[("rgb.top", 5.0 * period)],
    )
    assert not report.released
    assert any(c.name == "cross_modal_alignment" for c in report.blocking)


def test_absent_baseline_is_skipped_not_passed(settings, ingested, catalog):
    """"Not evaluated" and "passed" must never render the same.

    A distribution clause with no baseline has not protected anything, and a
    contract that reports it as a pass is worse than one with no clause at all.
    """
    from rde.observability import DataContract
    from rde.observability.contracts import SKIP

    manifest, res, _ = _contract_inputs(settings, catalog)
    contract = DataContract.infer(manifest, res.episode_status, None, name="t")
    report = contract.evaluate(manifest, res.episode_status)
    dist = [c for c in report.clauses if c.family == "distribution"]
    assert dist and all(c.outcome == SKIP for c in dist)
    assert report.released  # a skip does not block, but it is visibly a skip


def test_contract_survives_a_json_roundtrip(settings, ingested, catalog, tmp_path):
    """Contracts are reviewed artifacts, so they must be readable back."""
    from rde.observability import DataContract

    manifest, res, feats = _contract_inputs(settings, catalog)
    contract = DataContract.infer(manifest, res.episode_status, feats, name="t")
    path = contract.save(tmp_path / "contract.json")
    assert DataContract.load(path).to_dict() == contract.to_dict()


def test_underpowered_drift_warns_rather_than_blocking(settings, ingested, catalog):
    """A statistic on too few effective episodes must not gate a release."""
    from rde.observability import DataContract
    from rde.observability.contracts import WARN
    from rde.observability.drift import DriftResult

    manifest, res, feats = _contract_inputs(settings, catalog)
    contract = DataContract.infer(manifest, res.episode_status, feats, name="t")
    feature = next(iter(contract.drift_thresholds), None)
    if feature is None:                     # tiny fixture: nothing calibrated
        pytest.skip("no calibrated features on this fixture")
    huge = DriftResult(
        feature=feature, statistic="psi", value=99.0, threshold=0.1, exceeded=False,
        baseline_n=4, candidate_n=4, effective_sample_size=3.0, underpowered=True,
    )
    report = contract.evaluate(manifest, res.episode_status, drift_results=[huge])
    assert report.released
    assert any(c.outcome == WARN and feature in c.name for c in report.clauses)


# -- incremental validation ------------------------------------------------

def test_cache_hit_reuses_findings_and_is_far_faster(settings, ingested, catalog):
    """Versions are immutable, so re-validating an unchanged one is waste."""
    import time

    from rde.qc import QCEngine

    eng = QCEngine(settings=settings, catalog=catalog, incremental=True)
    t0 = time.perf_counter()
    cold = eng.run("testds", "raw/v1", persist=False)
    cold_s = time.perf_counter() - t0
    assert eng.last_run_cached is False

    t0 = time.perf_counter()
    warm = eng.run("testds", "raw/v1", persist=False)
    warm_s = time.perf_counter() - t0

    assert eng.last_run_cached is True
    assert len(warm.report.records) == len(cold.report.records)
    assert warm.episode_status == cold.episode_status
    assert warm.dataset_status is cold.dataset_status
    assert warm_s < cold_s


def test_editing_a_check_invalidates_the_cache(settings, ingested, catalog):
    """Names are not enough as a key.

    When ``value_range`` was recalibrated its name did not change. A name-keyed
    cache would have served findings computed under the old percentile bounds,
    and the recalibration would have looked like it did nothing.
    """
    from rde.qc.cache import suite_fingerprint
    from rde.validation import default_registry

    reg = default_registry()
    before = suite_fingerprint(reg.all())

    check = reg.get("numerical.value_range")

    class Recalibrated(type(check)):
        """A subclass stands in for an edit to the check's source."""

    reg._checks["numerical.value_range"] = Recalibrated()
    assert suite_fingerprint(reg.all()) != before


def test_in_place_corruption_invalidates_the_cache(settings, ingested, catalog):
    """Immutability is a design invariant, not one the filesystem enforces.

    A fault injector rewrites a version's frames in place. A cache that trusted
    immutability alone would report the pre-injection findings and make the
    fault demonstrations lie.
    """
    from rde.qc import QCEngine
    from rde.testing import inject_nan

    eng = QCEngine(settings=settings, catalog=catalog, incremental=True)
    clean = eng.run("testds", "raw/v1", persist=False)
    eng.run("testds", "raw/v1", persist=False)
    assert eng.last_run_cached is True

    inject_nan(settings.paths.version_dir("testds", "raw/v1"), "state.joint",
               row=3, dim=1)
    after = eng.run("testds", "raw/v1", persist=False)
    assert eng.last_run_cached is False
    assert len(after.report.records) > len(clean.report.records)


def test_blob_deletion_invalidates_even_though_parquet_is_untouched(
    settings, ingested, catalog
):
    """The gap a table-only fingerprint would leave open.

    Deleting a video touches no Parquet byte, so a cache keyed on tables alone
    would report the dataset healthy while its video was gone.
    """
    import json as _json

    from rde.qc import QCEngine
    from rde.storage import ParquetTableStore
    from rde.testing import inject_missing_blob

    eng = QCEngine(settings=settings, catalog=catalog, incremental=True)
    before = eng.run("testds", "raw/v1", persist=False)
    eng.run("testds", "raw/v1", persist=False)
    assert eng.last_run_cached is True

    tables = ParquetTableStore(settings.paths.version_dir("testds", "raw/v1"))
    blob_id = None
    for raw in tables.read_table("episodes", columns=["blob_ids"]).to_pydict()["blob_ids"]:
        mapping = _json.loads(raw or "{}")
        if mapping:
            blob_id = next(iter(mapping.values()))
            break
    if blob_id is None:
        pytest.skip("fixture has no external modalities")

    # Safe here only because `settings` puts the blob store in a temp root;
    # the store is deduplicated, so this is destructive across versions.
    inject_missing_blob(settings.paths.blobs, blob_id)
    after = eng.run("testds", "raw/v1", persist=False)
    assert eng.last_run_cached is False
    assert len(after.report.records) > len(before.report.records)


def test_cache_is_opt_in(settings, ingested, catalog):
    """A stale QC result is a correctness failure in the correctness component."""
    from rde.qc import QCEngine

    eng = QCEngine(settings=settings, catalog=catalog)
    eng.run("testds", "raw/v1", persist=False)
    eng.run("testds", "raw/v1", persist=False)
    assert eng.last_run_cached is False


# -- tiered video health ---------------------------------------------------

def _first_blob(settings, catalog):
    """(path, modality) of one stored video, or None if the fixture has none."""
    import json as _json

    from rde.storage import BlobStore, ParquetTableStore

    tables = ParquetTableStore(settings.paths.version_dir("testds", "raw/v1"))
    if not tables.exists("episodes"):
        return None
    for raw in tables.read_table("episodes", columns=["blob_ids"]).to_pydict()["blob_ids"]:
        mapping = _json.loads(raw or "{}")
        if mapping:
            modality, blob_id = next(iter(mapping.items()))
            return BlobStore(settings.paths.blobs).get(blob_id), modality
    return None


def test_tier0_reads_headers_without_decoding(settings, ingested, catalog):
    """The tier that is cheap enough to run on every file, every pass."""
    from rde.observability.video_health import tier0

    found = _first_blob(settings, catalog)
    if found is None:
        pytest.skip("fixture has no video blobs")
    path, modality = found
    report = tier0(path, "blob", modality)
    assert report.ok and report.width > 0 and report.num_frames > 0
    # The number that justifies the whole tiering scheme.
    assert report.decode_amplification > 1.0


def test_tier0_reports_a_corrupt_container_as_a_result_not_an_exception(tmp_path):
    """An unreadable file is a health finding, not a crash."""
    from rde.observability.video_health import tier0

    bad = tmp_path / "truncated.mp4"
    bad.write_bytes(b"not a video")
    report = tier0(bad, "blob", "rgb.top")
    assert report.ok is False and report.error


def test_tier1_cost_is_set_by_sample_count_not_clip_length(settings, ingested, catalog):
    """The property that makes this affordable at petabyte scale."""
    from rde.observability.video_health import tier1

    found = _first_blob(settings, catalog)
    if found is None:
        pytest.skip("fixture has no video blobs")
    path, modality = found
    few = tier1(path, "blob", modality, positions=(0.1, 0.9))
    many = tier1(path, "blob", modality, positions=(0.1, 0.3, 0.5, 0.7, 0.9))
    assert few.frames_sampled == 2 and many.frames_sampled == 5
    assert len(many.phashes) == 5


def test_phash_identifies_identical_frames(settings, ingested, catalog):
    """Tier 2's job: is this the same picture as that one?"""
    from rde.observability.video_health import _phash

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    assert _phash(frame) == _phash(frame.copy())
    other = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    assert _phash(frame) != _phash(other)


def test_frozen_stream_is_detected_from_sparse_samples():
    """A lens cap and a dead sensor both look like this."""
    from rde.observability.video_health import Tier1Report

    frozen = Tier1Report(blob_id="b", modality="rgb.top", frames_sampled=5,
                         interframe_difference=0.0)
    moving = Tier1Report(blob_id="b", modality="rgb.top", frames_sampled=5,
                         interframe_difference=13.0)
    assert frozen.frozen and not moving.frozen


def test_cohort_summary_catches_a_mid_collection_rig_change():
    """Every file individually valid; the set is not.

    A camera swapped to a different resolution partway through collection is
    invisible per-file and obvious across the cohort.
    """
    from rde.observability.video_health import Tier0Report, cohort_summary

    same = [
        Tier0Report("a", "rgb.top", True, "av1", 640, 480, 100, 3.3, 30, 1000, 90_000),
        Tier0Report("b", "rgb.top", True, "av1", 640, 480, 100, 3.3, 30, 1000, 90_000),
    ]
    mixed = same + [
        Tier0Report("c", "rgb.top", True, "av1", 1280, 720, 100, 3.3, 30, 2000, 270_000),
    ]
    assert cohort_summary(same)["consistent_resolution"] is True
    assert cohort_summary(mixed)["consistent_resolution"] is False
    assert len(cohort_summary(mixed)["resolutions"]) == 2
