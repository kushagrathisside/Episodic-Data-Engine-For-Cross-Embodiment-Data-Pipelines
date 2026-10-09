"""Synchronization tests: alignment verification and resampling."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rde.schemas.core import Episode
from rde.storage import BlobStore, ParquetTableStore
from rde.synchronization import AlignmentVerifier, resample_to_grid, uniform_grid
from rde.testing import inject_video_misalignment


def test_uniform_grid_matches_fps():
    g = uniform_grid(duration_s=1.0, fps=10)
    assert len(g) == 11
    assert np.allclose(np.diff(g), 0.1)


def test_resample_previous_is_zero_order_hold():
    ts = np.array([0.0, 1.0, 2.0])
    v = np.array([[10.0], [20.0], [30.0]])
    out, valid = resample_to_grid(ts, v, np.array([0.5, 1.5]), method="previous")
    assert valid.all()
    assert out[0, 0] == 10.0 and out[1, 0] == 20.0


def test_resample_linear_interpolates():
    ts = np.array([0.0, 1.0])
    v = np.array([[0.0], [10.0]])
    out, valid = resample_to_grid(ts, v, np.array([0.5]), method="linear")
    assert valid.all()
    assert out[0, 0] == pytest.approx(5.0)


def test_resample_refuses_to_invent_data_across_a_gap():
    """Interpolating over a dropout would make data look cleaner than reality."""
    ts = np.array([0.0, 5.0])
    v = np.array([[0.0], [10.0]])
    out, valid = resample_to_grid(ts, v, np.array([2.5]), method="previous", max_gap_s=1.0)
    assert not valid[0]
    assert np.isnan(out[0, 0])


def test_resample_marks_out_of_range_invalid():
    ts = np.array([1.0, 2.0])
    v = np.array([[1.0], [2.0]])
    out, valid = resample_to_grid(ts, v, np.array([0.0, 3.0]), method="previous")
    assert not valid.any()


def test_resample_unknown_method_raises():
    with pytest.raises(ValueError):
        resample_to_grid(np.array([0.0]), np.array([[1.0]]), np.array([0.0]), method="cubic")


def _episodes(settings):
    tables = ParquetTableStore(settings.paths.version_dir("testds", "raw/v1"))
    t = tables.read_table("episodes").to_pydict()
    return [
        Episode(
            episode_id=int(e), dataset_id="testds", frame_start=int(fs), frame_end=int(fe),
            blob_ids=json.loads(b), attributes=json.loads(a),
        )
        for e, fs, fe, b, a in zip(
            t["episode_id"], t["frame_start"], t["frame_end"], t["blob_ids"], t["attributes"]
        )
    ]


def test_alignment_verifier_confirms_correct_mapping(settings, ingested):
    v = AlignmentVerifier(BlobStore(settings.paths.blobs), fps=10.0)
    prev: dict = {}
    evidence = []
    for ep in _episodes(settings):
        evidence.extend(v.verify_episode(ep, ep.num_frames, prev))
    assert evidence
    for ev in evidence:
        assert ev.first_frame_decoded and ev.last_frame_decoded
        assert ev.duration_error_s < 0.15, f"window arithmetic off by {ev.duration_error_s}s"


def test_episodes_map_to_distinct_frames(settings, ingested):
    """Distinctness is what catches a constant offset error."""
    v = AlignmentVerifier(BlobStore(settings.paths.blobs), fps=10.0)
    prev: dict = {}
    distinctness = []
    for ep in _episodes(settings):
        for ev in v.verify_episode(ep, ep.num_frames, prev):
            if ev.distinctness is not None:
                distinctness.append(ev.distinctness)
    assert distinctness, "no cross-episode comparisons were made"
    assert min(distinctness) > 1.0, "episodes resolve to visually identical frames"


def test_stored_reference_drift_is_detected(settings, catalog, ingested):
    """The check must validate the stored column, not just re-derive it."""
    from rde.qc import QCEngine

    version_dir = settings.paths.version_dir("testds", "raw/v1")
    inject_video_misalignment(version_dir, shift_s=3.0)
    res = QCEngine(settings=settings, catalog=catalog).run("testds", "raw/v1")
    findings = [r for r in res.report.records if r.check_name == "sync.video_alignment"]
    assert findings
    assert any("drift" in f.reason for f in findings)
