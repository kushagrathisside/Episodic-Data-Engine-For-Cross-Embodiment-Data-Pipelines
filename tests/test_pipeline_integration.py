"""End-to-end pipeline test: every stage, on a real (synthetic) dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rde.pipeline import Pipeline, PipelineConfig


@pytest.fixture
def config(source_dataset: Path) -> PipelineConfig:
    return PipelineConfig(
        dataset_id="testds",
        source_repo=None,
        source_path=str(source_dataset),
        normalize=("state.joint", "action.joint"),
        image_size=(32, 32),
        keep_quality=("passed",),
        splits={"train": 0.5, "val": 0.25, "test": 0.25},
        split_seed=3,
        model="nearest_neighbor",
        baseline_model="mean_action",
        eval_max_frames=200,
    )


def test_full_pipeline_runs_every_stage(settings, config):
    record = Pipeline(config, settings=settings).run(skip_download=True)
    names = [s["name"] for s in record["stages"]]
    assert names == [
        "ingest", "validate", "transform", "qc_post_transform",
        "build", "evaluate", "feedback",
    ]
    assert all(s["ok"] for s in record["stages"])


def test_pipeline_produces_the_full_version_chain(settings, config, catalog):
    Pipeline(config, settings=settings).run(skip_download=True)
    stages = {v["stage"] for v in catalog.list_versions("testds")}
    assert {"raw", "clean", "training"} <= stages


def test_pipeline_run_record_is_reproducible_input(settings, config):
    """The run record must contain enough to re-run the same pipeline."""
    record = Pipeline(config, settings=settings).run(skip_download=True)
    path = Path(record["run_record"])
    assert path.exists()
    saved = json.loads(path.read_text())
    restored = PipelineConfig(**saved["config"])
    assert restored.dataset_id == config.dataset_id
    assert restored.split_seed == config.split_seed


def test_pipeline_is_idempotent(settings, config, catalog):
    """Re-running must converge on the same state, not accumulate versions."""
    Pipeline(config, settings=settings).run(skip_download=True)
    first = {v["version_key"]: v["total_frames"] for v in catalog.list_versions("testds")}
    Pipeline(config, settings=settings).run(skip_download=True)
    second = {v["version_key"]: v["total_frames"] for v in catalog.list_versions("testds")}
    assert first == second


def test_post_transform_qc_runs_on_the_derived_version(settings, config, catalog):
    """The second QC pass must validate clean/v1, not re-validate raw/v1."""
    record = Pipeline(config, settings=settings).run(skip_download=True)
    post = next(s for s in record["stages"] if s["name"] == "qc_post_transform")
    assert post["detail"]["version"] == "clean/v1"


def test_config_round_trips_through_file(tmp_path: Path, config):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(config.to_dict()))
    loaded = PipelineConfig.from_file(p)
    assert loaded.image_size == config.image_size
    assert loaded.normalize == config.normalize
    assert loaded.splits == config.splits
