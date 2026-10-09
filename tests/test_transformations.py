"""Transformation operator tests."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from rde.qc import QCEngine
from rde.schemas.core import QualityStatus
from rde.storage import ParquetTableStore
from rde.transformations import (
    DropNonFiniteFrames,
    EpisodeSubsetFilter,
    LengthFilter,
    NormalizeVector,
    QualityFilter,
    ResizeImages,
    SelectFeatures,
    SubsampleFrames,
    TimestampNormalizer,
    TransformPipeline,
    assign_splits,
)
from rde.testing import inject_nan


def _pipeline(settings, catalog, ops, out="clean/v1"):
    return TransformPipeline(ops, settings=settings, catalog=catalog).run(
        "testds", "raw/v1", out
    )


def test_normalize_produces_zero_mean_unit_std(settings, catalog, ingested):
    res = _pipeline(settings, catalog, [NormalizeVector("state.joint", "mean_std")])
    store = ParquetTableStore(settings.paths.version_dir("testds", "clean/v1"))
    t = store.read_table("frames", columns=["state.joint"])
    v = np.asarray(t.column("state.joint").combine_chunks().flatten()).reshape(t.num_rows, -1)
    assert np.allclose(v.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(v.std(axis=0), 1.0, atol=1e-3)
    assert "normalize.state.joint" in res.fitted


def test_normalize_min_max_bounds(settings, catalog, ingested):
    _pipeline(settings, catalog, [NormalizeVector("state.joint", "min_max")])
    store = ParquetTableStore(settings.paths.version_dir("testds", "clean/v1"))
    t = store.read_table("frames", columns=["state.joint"])
    v = np.asarray(t.column("state.joint").combine_chunks().flatten()).reshape(t.num_rows, -1)
    assert v.min() >= -1.0001 and v.max() <= 1.0001


def test_normalization_stats_recorded_for_reproducibility(settings, catalog, ingested):
    """Fitted statistics must be persisted, not recomputed at training time."""
    res = _pipeline(settings, catalog, [NormalizeVector("action.joint", "mean_std")])
    stats = res.fitted["normalize.action.joint"]
    assert set(stats) >= {"mean", "std", "min", "max", "count"}
    manifest = catalog.get_manifest("testds", "clean/v1")
    assert manifest.modality("action.joint").attributes["normalization"]["scheme"] == "mean_std"


def test_timestamp_normalizer_places_on_uniform_grid(settings, catalog, ingested):
    _pipeline(settings, catalog, [TimestampNormalizer()])
    store = ParquetTableStore(settings.paths.version_dir("testds", "clean/v1"))
    import pyarrow.dataset as ds

    t = store.read_table("frames", filter=ds.field("episode_id") == 1)
    ts = np.asarray(t.column("timestamp_s"))
    fi = np.asarray(t.column("frame_index"))
    assert np.allclose(ts, fi / 10.0)


def test_quality_filter_drops_rejected_episodes(settings, catalog, ingested, version_dir):
    inject_nan(version_dir, "state.joint", row=5, dim=2)
    QCEngine(settings=settings, catalog=catalog).run("testds", "raw/v1")
    res = _pipeline(settings, catalog, [QualityFilter([QualityStatus.PASSED])])
    assert 0 in res.dropped_episodes
    assert res.episodes_out < res.episodes_in


def test_length_filter(settings, catalog, ingested):
    res = _pipeline(settings, catalog, [LengthFilter(min_frames=1000)])
    assert res.episodes_out == 0 or res.frames_out == 0


def test_episode_subset_filter_keeps_only_the_named_episodes(settings, catalog, ingested):
    res = _pipeline(settings, catalog, [EpisodeSubsetFilter([0, 2], reason="unit test subset")])
    assert res.episodes_out == 2
    assert sorted(set(range(4)) - {0, 2}) == res.dropped_episodes


def test_episode_subset_filter_records_reason_in_lineage(settings, catalog, ingested):
    _pipeline(settings, catalog, [EpisodeSubsetFilter([1, 3], reason="reduced-data intervention")])
    runs = catalog.get_runs("testds")
    transform_run = next(r for r in runs if r.operation == "transform")
    op_params = transform_run.parameters["operations"][0]
    assert op_params["parameters"]["reason"] == "reduced-data intervention"
    assert op_params["parameters"]["episode_ids"] == [1, 3]


def test_episode_subset_filter_ignores_ids_outside_the_input_version(settings, catalog, ingested):
    """Requesting an id that does not exist in the input must not raise -- it is simply absent."""
    res = _pipeline(settings, catalog, [EpisodeSubsetFilter([0, 999])])
    assert res.episodes_out == 1


def test_subsample_halves_frames_and_updates_fps(settings, catalog, ingested):
    res = _pipeline(settings, catalog, [SubsampleFrames(stride=2)])
    assert res.frames_out == pytest.approx(res.frames_in / 2, rel=0.05)
    manifest = catalog.get_manifest("testds", "clean/v1")
    assert manifest.fps == 5.0, "declared fps must follow a temporal downsample"


def test_drop_non_finite_removes_bad_rows(settings, catalog, ingested, version_dir):
    inject_nan(version_dir, "state.joint", row=5, dim=2)
    res = _pipeline(settings, catalog, [DropNonFiniteFrames()])
    assert res.frames_out == res.frames_in - 1


def test_select_features_prunes_columns_and_manifest(settings, catalog, ingested):
    _pipeline(settings, catalog, [SelectFeatures(["state.joint", "action.joint"])])
    store = ParquetTableStore(settings.paths.version_dir("testds", "clean/v1"))
    names = store.schema("frames").names
    assert "state.joint" in names
    assert not any(n.startswith("rgb.") for n in names)
    manifest = catalog.get_manifest("testds", "clean/v1")
    assert set(manifest.modality_names) == {"state.joint", "action.joint"}


def test_resize_is_deferred_not_materialized(settings, catalog, ingested):
    """A deferred op must change the manifest, not rewrite stored data."""
    before = ParquetTableStore(settings.paths.version_dir("testds", "raw/v1")).stats("frames")
    _pipeline(settings, catalog, [ResizeImages(64, 64)])
    after = ParquetTableStore(settings.paths.version_dir("testds", "clean/v1")).stats("frames")
    assert after["rows"] == before["rows"]

    manifest = catalog.get_manifest("testds", "clean/v1")
    spec = manifest.modality("rgb.top")
    assert spec.attributes["read_transforms"][0]["op"] == "resize"
    assert tuple(spec.shape[:2]) == (64, 64)


def test_pipeline_recomputes_frame_ranges_after_filtering(settings, catalog, ingested):
    """Dropping rows invalidates old offsets; episodes must be re-indexed."""
    _pipeline(settings, catalog, [SubsampleFrames(stride=2)])
    store = ParquetTableStore(settings.paths.version_dir("testds", "clean/v1"))
    eps = store.read_table("episodes").to_pydict()
    starts = eps["frame_start"]
    ends = eps["frame_end"]
    assert starts[0] == 0
    for i in range(1, len(starts)):
        assert starts[i] == ends[i - 1], "episode frame ranges are not contiguous"


def test_splits_are_deterministic_and_disjoint():
    ids = list(range(50))
    a = assign_splits(ids, {"train": 0.8, "val": 0.2}, seed=1)
    b = assign_splits(ids, {"train": 0.8, "val": 0.2}, seed=1)
    assert a == b
    assert set(a) == set(ids)


def test_quantile_split_gives_exact_proportions():
    ids = list(range(50))
    a = assign_splits(ids, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=3, mode="quantile")
    counts = {k: sum(1 for v in a.values() if v == k) for k in ("train", "val", "test")}
    assert counts == {"train": 40, "val": 5, "test": 5}


def test_split_seed_changes_assignment():
    ids = list(range(50))
    a = assign_splits(ids, {"train": 0.5, "val": 0.5}, seed=1)
    b = assign_splits(ids, {"train": 0.5, "val": 0.5}, seed=2)
    assert a != b


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        assign_splits([1, 2, 3], {"train": 0.5, "val": 0.2})
