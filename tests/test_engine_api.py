"""Public interface tests: the contract consumers depend on."""

from __future__ import annotations

import numpy as np
import pytest

from rde.dataset_builder import DatasetBuilder, build_training_contract
from rde.engine import DataEngine
from rde.schemas.core import QualityStatus
from rde.transformations import NormalizeVector, ResizeImages, TransformPipeline


@pytest.fixture
def training_view(settings, catalog, ingested):
    TransformPipeline(
        [NormalizeVector("state.joint"), NormalizeVector("action.joint"), ResizeImages(32, 32)],
        settings=settings, catalog=catalog,
    ).run("testds", "raw/v1", "clean/v1")
    DatasetBuilder(settings=settings, catalog=catalog).build(
        "testds", "clean/v1", "training/v1", splits={"train": 0.5, "test": 0.5}, seed=1,
        # Fixture skips QC entirely (goes straight raw -> clean), so episodes
        # are UNKNOWN quality; admit them explicitly rather than relying on
        # DatasetBuilder's default, which excludes UNKNOWN.
        require_quality=[QualityStatus.PASSED, QualityStatus.WARNING, QualityStatus.UNKNOWN],
    )
    view = DataEngine(settings).get_dataset("testds", "training/v1")
    yield view
    view.close()


def test_get_dataset_resolves_by_version_not_path(settings, training_view):
    assert training_view.version == "training/v1"
    assert training_view.dataset_id == "testds"
    assert len(training_view.episodes) == 4


def test_unknown_version_raises_with_alternatives(settings, ingested):
    with pytest.raises(LookupError) as exc:
        DataEngine(settings).get_dataset("testds", "training/v99")
    assert "raw/v1" in str(exc.value)


def test_contract_describes_observation_and_action(training_view):
    c = training_view.contract
    assert c["action"]["key"] == "action.joint"
    assert c["action"]["shape"] == [6]
    assert "state.joint" in c["observation"]
    assert "rgb.top" in c["observation"]
    assert c["observation"]["rgb.top"]["plane"] == "external"
    assert c["language_conditioned"] is True


def test_contract_carries_normalization_and_read_transforms(training_view):
    c = training_view.contract
    assert c["action"]["normalization"]["scheme"] == "mean_std"
    assert c["observation"]["rgb.top"]["read_transforms"][0]["op"] == "resize"


def test_splits_are_disjoint_by_episode(training_view):
    train = set(training_view.episode_ids(split="train"))
    test = set(training_view.episode_ids(split="test"))
    assert train and test
    assert not (train & test), "an episode appears in two splits"


def test_iter_samples_yields_canonical_fields(training_view):
    s = next(training_view.iter_samples(split="train"))
    assert s.action is not None and s.action.shape == (6,)
    assert "state.joint" in s.observation
    assert s.task, "language instruction missing from sample"
    assert set(s.image_modalities) == {"rgb.top", "rgb.wrist"}


def test_images_are_lazy(training_view):
    """A sample must carry references, not decoded pixels."""
    s = next(training_view.iter_samples(split="train"))
    assert s.image_refs, "no image references on the sample"
    for blob_id, ts in s.image_refs.values():
        assert isinstance(blob_id, str) and isinstance(ts, float)


def test_image_decode_applies_deferred_resize(training_view):
    s = next(training_view.iter_samples(split="train"))
    img = s.image("rgb.top")
    assert img is not None
    assert img.shape[:2] == (32, 32), "deferred resize was not applied at read time"


def test_image_decode_can_skip_read_transforms(training_view):
    s = next(training_view.iter_samples(split="train"))
    raw = s.image("rgb.top", apply_read_transforms=False)
    assert raw.shape[:2] == (64, 64), "raw decode should return native resolution"


def test_iter_frames_respects_projection(training_view):
    batch = next(training_view.iter_frames(split="train", columns=["frame_id", "episode_id"]))
    assert batch.schema.names == ["frame_id", "episode_id"]


def test_training_version_references_source_frames(settings, catalog, training_view):
    """The training version must not duplicate the frame table."""
    manifest = catalog.get_manifest("testds", "training/v1")
    assert manifest.statistics["frame_table_version"] == "clean/v1"
    own = settings.paths.version_dir("testds", "training/v1") / "frames"
    assert not own.exists(), "training version copied the frame table instead of referencing it"


def test_build_contract_is_pure_function(settings, catalog, ingested):
    manifest = catalog.get_manifest("testds", "raw/v1")
    a = build_training_contract(manifest)
    b = build_training_contract(manifest)
    assert a == b
