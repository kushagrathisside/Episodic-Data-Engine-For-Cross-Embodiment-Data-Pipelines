"""Model layer tests: interchangeability is the property under test."""

from __future__ import annotations

import numpy as np
import pytest

from rde.dataset_builder import DatasetBuilder
from rde.engine import DataEngine
from rde.evaluation import Evaluator
from rde.schemas.core import QualityStatus
from rde.model_adapters import (
    MODEL_REGISTRY,
    MeanActionPolicy,
    NearestNeighborPolicy,
    StateAdapter,
    VisionLanguageAdapter,
    VisionStateAdapter,
    build_model,
)
from rde.transformations import NormalizeVector, ResizeImages, TransformPipeline


@pytest.fixture
def view(settings, catalog, ingested):
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
    v = DataEngine(settings).get_dataset("testds", "training/v1")
    yield v
    v.close()


def _samples(view, split="train", n=8):
    out = []
    for s in view.iter_samples(split=split):
        out.append(s)
        if len(out) >= n:
            break
    return out


def test_state_adapter_shapes(view):
    a = StateAdapter(view.contract)
    mi = a.adapt(_samples(view))
    assert mi.tensors["state"].shape == (8, 6)
    assert mi.frame_ids and len(mi.frame_ids) == 8


def test_vision_adapter_decodes_and_orders_channels(view):
    a = VisionStateAdapter(view.contract, channels_first=True)
    mi = a.adapt(_samples(view, n=4))
    img = mi.tensors["image.rgb.top"]
    assert img.shape == (4, 3, 32, 32), f"unexpected layout {img.shape}"
    assert img.dtype == np.float32


def test_vision_adapter_can_emit_channels_last(view):
    a = VisionStateAdapter(view.contract, channels_first=False)
    mi = a.adapt(_samples(view, n=2))
    assert mi.tensors["image.rgb.top"].shape == (2, 32, 32, 3)


def test_vision_language_adapter_carries_instruction(view):
    a = VisionLanguageAdapter(view.contract)
    mi = a.adapt(_samples(view, n=3))
    assert mi.instruction
    assert len(mi.metadata["instructions"]) == 3


def test_adapters_read_contract_not_hardcoded_keys(view):
    """An adapter must derive its keys from the contract."""
    a = StateAdapter(view.contract)
    assert a.action_key == "action.joint"
    assert a.state_keys == ["state.joint"]
    assert set(a.image_keys) == {"rgb.top", "rgb.wrist"}


def test_mean_action_baseline_predicts_training_mean(view):
    m = MeanActionPolicy()
    m.fit(view, split="train")
    a = StateAdapter(view.contract)
    pred = m.timed_predict(a.adapt(_samples(view, "test", 5)))
    assert pred.actions.shape == (5, 6)
    assert np.allclose(pred.actions[0], pred.actions[1]), "baseline should be input-independent"
    assert pred.latency_ms >= 0


def test_nearest_neighbor_beats_the_baseline(view):
    """The retrieval policy must actually be better, or the harness is wrong."""
    a = StateAdapter(view.contract)
    samples = _samples(view, "test", 16)

    base = MeanActionPolicy(); base.fit(view, split="train")
    knn = NearestNeighborPolicy(k=3); knn.fit(view, split="train")

    truth = np.stack([s.action for s in samples])
    mi = a.adapt(samples)
    e_base = np.abs(base.timed_predict(mi).actions - truth).mean()
    e_knn = np.abs(knn.timed_predict(mi).actions - truth).mean()
    assert e_knn < e_base, f"kNN ({e_knn:.4f}) did not beat baseline ({e_base:.4f})"


def test_predictions_carry_confidence_and_provenance(view):
    m = NearestNeighborPolicy(k=3); m.fit(view, split="train")
    a = StateAdapter(view.contract)
    samples = _samples(view, "test", 6)
    pred = m.timed_predict(a.adapt(samples))
    assert pred.confidence.shape == (6,)
    assert ((pred.confidence >= 0) & (pred.confidence <= 1)).all()
    assert pred.frame_ids == [s.frame_id for s in samples]
    assert pred.model_name == "nearest_neighbor"


def test_models_are_interchangeable_through_one_interface(view):
    """The point of the layer: same code path, different models."""
    ev = Evaluator()
    results = {}
    for name in ("mean_action", "nearest_neighbor"):
        model, adapter = build_model(name, view.contract)
        model.fit(view, split="train")
        results[name] = ev.evaluate(view, model, adapter, split="test", persist=False)
    assert set(results) == {"mean_action", "nearest_neighbor"}
    for r in results.values():
        assert {"mse", "mae", "per_dimension_mae", "success_rate"} <= set(r.metrics)


def test_registry_pairs_models_with_adapters():
    for name, (model_cls, adapter_cls) in MODEL_REGISTRY.items():
        assert hasattr(model_cls, "predict")
        assert hasattr(adapter_cls, "adapt")


def test_unknown_model_raises(view):
    with pytest.raises(KeyError):
        build_model("no_such_model", view.contract)


def test_smolvla_reports_missing_dependency_clearly(view):
    """An unavailable optional model must fail loudly, not degrade silently."""
    from rde.model_adapters import SmolVLAPolicy

    policy = SmolVLAPolicy()
    with pytest.raises((ImportError, RuntimeError, OSError)) as exc:
        policy.load()
    assert "smolvla" in str(exc.value).lower() or "lerobot" in str(exc.value).lower() \
        or "torch" in str(exc.value).lower()


def test_evaluator_episode_ids_scopes_to_an_explicit_slice(view):
    """Stratified evaluation (per-task/per-variation slices) selects an explicit
    episode set rather than a named split; episode_ids must take precedence."""
    ev = Evaluator()
    model, adapter = build_model("mean_action", view.contract)
    model.fit(view, split="train")

    all_test_ids = view.episode_ids(split="test")
    assert all_test_ids, "fixture must have a non-empty test split"
    slice_ids = all_test_ids[:1]

    full = ev.evaluate(view, model, adapter, split="test", persist=False)
    sliced = ev.evaluate(
        view, model, adapter, split="test:slice", episode_ids=slice_ids, persist=False
    )
    assert {o.episode_id for o in sliced.outcomes} == set(slice_ids)
    if len(all_test_ids) > 1:
        assert sliced.frames_evaluated < full.frames_evaluated


def test_smolvla_verify_checkpoint_rejects_incomplete_directory(tmp_path):
    """A corrupted/foreign checkpoint directory must fail loudly, not load partially."""
    from rde.model_adapters import SmolVLAPolicy

    empty_dir = tmp_path / "not_a_checkpoint"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        SmolVLAPolicy.verify_checkpoint(tmp_path / "does_not_exist")
    with pytest.raises(ValueError, match="incomplete"):
        SmolVLAPolicy.verify_checkpoint(empty_dir)


def test_smolvla_verify_checkpoint_accepts_a_well_formed_directory(tmp_path):
    from rde.model_adapters import SmolVLAPolicy

    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "policy_preprocessor.json").write_text("{}")
    (ckpt / "policy_postprocessor.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"")
    (ckpt / "rde_checkpoint_meta.json").write_text('{"checkpoint_source": "lerobot/smolvla_base"}')
    meta = SmolVLAPolicy.verify_checkpoint(ckpt)
    assert meta["checkpoint_source"] == "lerobot/smolvla_base"


def test_evaluator_rejects_action_shape_mismatch(view):
    """Shape drift between model and dataset must be an error, not silent."""
    class BadModel(MeanActionPolicy):
        name = "bad"

        def predict(self, model_input):
            from rde.model_adapters.base import Prediction

            b = model_input.batch_size
            return Prediction(actions=np.zeros((b, 3), np.float32),
                              confidence=np.ones(b, np.float32))

    with pytest.raises(ValueError, match="expects"):
        Evaluator().evaluate(view, BadModel(), StateAdapter(view.contract),
                             split="test", persist=False)
