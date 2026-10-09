"""Fault-injection tests: prove each check fires on the defect it claims.

This is the most important test module in the project.  The reference dataset is
a curated public release and is genuinely clean, so a QC engine run against it
reports almost nothing -- which is evidence of the *dataset's* quality, not the
engine's.  The only way to demonstrate the checks work is to break the data
deliberately, one defect class at a time, and assert that the right check fires.

Each test also asserts the *converse* where it matters: that a clean dataset
produces no finding from that check.  A check that fires on everything is as
useless as one that fires on nothing.
"""

from __future__ import annotations

import pytest

from rde.qc import QCEngine, QCGate
from rde.schemas.core import QualityStatus
from rde.schemas.qc import CheckStatus, Severity
from rde.testing import (
    inject_discontinuity,
    inject_dropped_frames,
    inject_duplicate_timestamp,
    inject_missing_blob,
    inject_nan,
    inject_non_monotonic,
    inject_stuck_sensor,
    inject_timestamp_gap,
    inject_video_misalignment,
)



def pa_table(d, like):
    """Rebuild a table from a mutated pydict, preserving the original schema."""
    import pyarrow as pa

    return pa.Table.from_pydict(d, schema=like.schema)

def _run(settings, catalog, checks=None):
    return QCEngine(settings=settings, catalog=catalog).run("testds", "raw/v1", checks=checks)


def _findings(result, check_name):
    return [
        r for r in result.report.records
        if r.check_name == check_name and r.status in (CheckStatus.FAILED, CheckStatus.WARNED)
    ]


def test_clean_dataset_passes(settings, catalog, ingested):
    """Baseline: an uncorrupted synthetic dataset raises no blocking finding."""
    res = _run(settings, catalog)
    blocking = res.report.blocking
    assert not blocking, f"clean dataset produced blocking findings: {[b.reason for b in blocking]}"
    assert res.dataset_status in (QualityStatus.PASSED, QualityStatus.WARNING)


def test_nan_detected(settings, catalog, ingested, version_dir):
    inject_nan(version_dir, "state.joint", row=5, dim=2)
    res = _run(settings, catalog)
    f = _findings(res, "numerical.finite")
    assert f, "NaN injected but numerical.finite did not fire"
    assert f[0].severity is Severity.ERROR
    assert f[0].modality == "state.joint"
    assert f[0].episode_id == 0


def test_timestamp_gap_detected(settings, catalog, ingested, version_dir):
    inject_timestamp_gap(version_dir, row=10, gap_s=2.0)
    res = _run(settings, catalog)
    f = _findings(res, "temporal.gaps")
    assert f, "timestamp gap injected but temporal.gaps did not fire"
    assert f[0].details.get("estimated_missing_frames", 0) > 0


def test_non_monotonic_detected(settings, catalog, ingested, version_dir):
    inject_non_monotonic(version_dir, row=8)
    res = _run(settings, catalog)
    assert _findings(res, "temporal.monotonic"), "non-monotonic timestamps not detected"


def test_duplicate_timestamp_detected(settings, catalog, ingested, version_dir):
    inject_duplicate_timestamp(version_dir, row=12)
    res = _run(settings, catalog)
    assert _findings(res, "temporal.duplicates"), "duplicate timestamps not detected"


def test_stuck_sensor_detected(settings, catalog, ingested, version_dir):
    inject_stuck_sensor(version_dir, "state.joint", episode_id=1, dim=0)
    res = _run(settings, catalog)
    f = _findings(res, "numerical.stuck_sensor")
    assert f, "stuck sensor not detected"
    assert any(r.episode_id == 1 for r in f)
    assert 0 in f[0].details.get("dimensions", [])


def test_discontinuity_detected(settings, catalog, ingested, version_dir):
    inject_discontinuity(version_dir, "action.joint", row=15, magnitude=500.0)
    res = _run(settings, catalog)
    assert _findings(res, "numerical.discontinuity") or _findings(res, "numerical.value_range"), \
        "large discontinuity escaped both discontinuity and range checks"


def test_dropped_frames_detected(settings, catalog, ingested, version_dir):
    inject_dropped_frames(version_dir, episode_id=2, drop_from=5, count=3)
    res = _run(settings, catalog)
    assert _findings(res, "structural.frame_index_continuity") or \
           _findings(res, "structural.episode_frame_count"), "dropped frames not detected"


def test_missing_blob_detected(settings, catalog, ingested):
    blob_id = next(iter(ingested.blob_ids.values()))
    inject_missing_blob(settings.paths.blobs, blob_id)
    res = _run(settings, catalog)
    f = _findings(res, "structural.blob_integrity")
    assert f, "deleted blob not detected"
    assert f[0].severity is Severity.CRITICAL


def test_video_misalignment_detected(settings, catalog, ingested, version_dir):
    """The defect no numeric check can see: right numbers, wrong pixels."""
    inject_video_misalignment(version_dir, shift_s=5.0)
    res = _run(settings, catalog)
    numeric = (
        _findings(res, "numerical.finite")
        + _findings(res, "temporal.monotonic")
        + _findings(res, "temporal.gaps")
    )
    assert not numeric, "misalignment should be invisible to numeric/temporal checks"
    assert _findings(res, "sync.video_alignment"), \
        "video misalignment not caught by the alignment check"


def test_gate_rejects_on_blocking_finding(settings, catalog, ingested, version_dir):
    """A blocking finding must move the episode to REJECTED, not merely warn."""
    inject_nan(version_dir, "state.joint", row=5, dim=2)
    res = _run(settings, catalog)
    assert res.episode_status[0] is QualityStatus.REJECTED
    assert 0 in res.report.rejected_episodes


def test_gate_quarantines_on_many_warnings(settings, catalog, ingested, version_dir):
    """max_warnings escalates an accumulation of non-fatal findings to review."""
    inject_stuck_sensor(version_dir, "state.joint", episode_id=1, dim=0)
    inject_stuck_sensor(version_dir, "action.joint", episode_id=1, dim=1)
    eng = QCEngine(settings=settings, catalog=catalog, gate=QCGate(max_warnings=1))
    res = eng.run("testds", "raw/v1")
    assert res.episode_status[1] is QualityStatus.QUARANTINED


def test_findings_are_persisted_and_queryable(settings, catalog, ingested, version_dir):
    """Rejection must be explainable later, not just at the time it happened."""
    inject_nan(version_dir, "state.joint", row=5, dim=2)
    _run(settings, catalog)
    rows = catalog.query_qc("testds", "raw/v1", episode_id=0, check_name="numerical.finite")
    assert rows, "finding was not written to the queryable QC index"
    r = rows[0]
    for field in ("check_name", "severity", "status", "observed_value", "expected_value", "reason"):
        assert r[field] is not None, f"QC record missing required field {field}"


def test_checks_isolated_to_their_defect(settings, catalog, ingested, version_dir):
    """A temporal defect must not raise numerical findings, and vice versa."""
    inject_duplicate_timestamp(version_dir, row=12)
    res = _run(settings, catalog)
    assert _findings(res, "temporal.duplicates")
    assert not _findings(res, "numerical.finite"), \
        "a timestamp defect wrongly produced a numeric finding"


# -- range-check calibration -----------------------------------------------
# These pin a bug found by running a second, structurally different dataset
# (ALOHA: 14-DoF bimanual, 4 cameras, plus force/torque).  Learned bounds are
# percentiles, so ~1% of the dataset lies outside them by construction.  With 42
# dimensions and long-tailed force data that tautology flagged all 50 episodes --
# a check that fires on everything carries no information.


def test_marginal_bound_crossings_are_not_flagged(settings, catalog, ingested, version_dir):
    """A hair over the learned bound is not a defect."""
    import numpy as np
    import pyarrow.parquet as pq

    target = version_dir / "frames" / "part-000.parquet"
    t = pq.read_table(target)
    d = t.to_pydict()
    # Nudge a single sample barely past where the bound will land.
    col = np.array(d["state.joint"], dtype=np.float64)
    span = col[:, 0].max() - col[:, 0].min()
    row = int(np.argmax(col[:, 0]))
    v = list(d["state.joint"][row])
    v[0] = float(col[:, 0].max() + 0.001 * span)
    d["state.joint"][row] = v
    pq.write_table(pa_table(d, t), target)

    res = _run(settings, catalog)
    assert not _findings(res, "numerical.value_range"), \
        "a marginal crossing was reported as a defect"


def test_egregious_excursion_is_flagged(settings, catalog, ingested, version_dir):
    """A spike far beyond the bound is a defect even as a single sample."""
    import numpy as np
    import pyarrow.parquet as pq

    target = version_dir / "frames" / "part-000.parquet"
    t = pq.read_table(target)
    d = t.to_pydict()
    col = np.array(d["state.joint"], dtype=np.float64)
    span = col[:, 0].max() - col[:, 0].min()
    v = list(d["state.joint"][3])
    v[0] = float(col[:, 0].max() + 5.0 * span)
    d["state.joint"][3] = v
    pq.write_table(pa_table(d, t), target)

    res = _run(settings, catalog)
    f = _findings(res, "numerical.value_range")
    assert f, "a 5x-span excursion was not flagged"
    assert f[0].details.get("worst_excursion_spans", 0) > 0.25


def test_impulsive_kinds_get_wider_bounds(settings, catalog, ingested):
    """Force/tactile signals are impulsive; they must not be judged as positions."""
    from rde.validation.numerical import RangeProfiler
    from rde.schemas.modality import ModalityKind

    assert ModalityKind.FORCE_TORQUE in RangeProfiler.IMPULSIVE_KINDS
    assert ModalityKind.TACTILE in RangeProfiler.IMPULSIVE_KINDS
    lo, hi = RangeProfiler.IMPULSIVE_PCT
    assert lo < 0.5 and hi > 99.5, "impulsive bounds must be wider than positional ones"


def test_hard_limits_flag_at_first_sample(settings, catalog, ingested):
    """A configured mechanical limit is not subject to the materiality test."""
    from rde.qc import QCEngine
    from rde.validation import default_registry
    from rde.validation.numerical import ValueRangeCheck
    from rde.schemas.qc import Severity

    registry = default_registry()
    registry._checks["numerical.value_range"] = ValueRangeCheck(
        limits={"state.joint": ([-0.001] * 6, [0.001] * 6)}
    )
    res = QCEngine(settings=settings, catalog=catalog, registry=registry).run(
        "testds", "raw/v1"
    )
    f = _findings(res, "numerical.value_range")
    assert f, "configured hard limits were not enforced"
    assert f[0].severity is Severity.ERROR, "a mechanical-limit breach must be an error"
