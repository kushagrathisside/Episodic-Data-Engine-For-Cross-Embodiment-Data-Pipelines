"""Versioning and lineage: the provenance guarantees the design promises."""

from __future__ import annotations

import pytest

from rde.dataset_builder import DatasetBuilder
from rde.exploration.queries import DatasetExplorer
from rde.qc import QCEngine
from rde.schemas.core import QualityStatus
from rde.schemas.lineage import VersionStage
from rde.testing import inject_nan
from rde.transformations import NormalizeVector, QualityFilter, TransformPipeline
from rde.versioning import next_version


def _full_chain(settings, catalog):
    QCEngine(settings=settings, catalog=catalog).run("testds", "raw/v1")
    TransformPipeline(
        [QualityFilter([QualityStatus.PASSED]), NormalizeVector("state.joint")],
        settings=settings, catalog=catalog,
    ).run("testds", "raw/v1", "clean/v1")
    DatasetBuilder(settings=settings, catalog=catalog).build(
        "testds", "clean/v1", "training/v1", splits={"train": 0.75, "test": 0.25}, seed=7,
        # Fixture skips QC entirely (goes straight raw -> clean), so episodes
        # are UNKNOWN quality; admit them explicitly rather than relying on
        # DatasetBuilder's default, which excludes UNKNOWN.
        require_quality=[QualityStatus.PASSED, QualityStatus.WARNING, QualityStatus.UNKNOWN],
    )


def test_version_stage_parsing():
    assert VersionStage.parse("training/v2") == (VersionStage.TRAINING, 2)
    with pytest.raises(ValueError):
        VersionStage.parse("training")
    with pytest.raises(ValueError):
        VersionStage.parse("training/2")


def test_next_version_increments(settings, catalog, ingested):
    assert next_version(catalog, "testds", VersionStage.RAW) == "raw/v2"
    assert next_version(catalog, "testds", VersionStage.CLEAN) == "clean/v1"


def test_lineage_chain_is_complete(settings, catalog, ingested):
    _full_chain(settings, catalog)
    lin = DatasetExplorer(settings=settings, catalog=catalog).lineage("testds", "training/v1")
    ops = [r["operation"] for r in lin["chain"]]
    assert ops == ["ingest", "transform", "build_training_dataset"]
    assert lin["ancestors"] == ["clean/v1", "raw/v1"]


def test_lineage_records_parameters_and_fingerprint(settings, catalog, ingested):
    _full_chain(settings, catalog)
    lin = DatasetExplorer(settings=settings, catalog=catalog).lineage("testds", "training/v1")
    transform = next(r for r in lin["chain"] if r["operation"] == "transform")
    assert transform["parameters"]["operations"], "operator parameters were not recorded"
    assert transform["code_fingerprint"], "code fingerprint missing; runs are not reproducible"
    assert transform["software_version"]


def test_episode_level_provenance_survives_filtering(settings, catalog, ingested, version_dir):
    """'Which raw episodes reached training/v1?' must be answerable."""
    inject_nan(version_dir, "state.joint", row=5, dim=2)
    _full_chain(settings, catalog)

    lin = DatasetExplorer(settings=settings, catalog=catalog).lineage("testds", "training/v1")
    roots = lin["root_episodes"]
    assert roots, "no contributing root episodes recorded"
    assert 0 not in roots, "a rejected episode was reported as contributing"

    traced = catalog.trace_episodes_to_root("testds", "training/v1")
    assert traced, "episode ancestry table is empty"
    for episode_id, chain in traced.items():
        assert chain, f"episode {episode_id} has no recorded ancestry"


def test_dropped_episodes_are_recorded_not_forgotten(settings, catalog, ingested, version_dir):
    inject_nan(version_dir, "state.joint", row=5, dim=2)
    _full_chain(settings, catalog)
    runs = catalog.get_runs("testds")
    transform = next(r for r in runs if r.operation == "transform")
    assert 0 in transform.dropped_episode_ids
    findings = catalog.query_qc("testds", "raw/v1", episode_id=0)
    assert findings, "episode was dropped without a persisted reason"


def test_failed_run_is_still_recorded(settings, catalog, ingested):
    """A crash mid-pipeline must leave a lineage record, not a silent gap."""
    from rde.versioning import VersionManager

    vm = VersionManager(catalog)
    with pytest.raises(RuntimeError):
        with vm.run("testds", "explode", "clean/v9", ["raw/v1"]) as run:
            raise RuntimeError("boom")
    runs = [r for r in catalog.get_runs("testds") if r.operation == "explode"]
    assert runs and "boom" in runs[0].metrics.get("error", "")


def test_blobs_shared_across_versions_not_copied(settings, catalog, ingested):
    """Derived versions must not duplicate video payloads."""
    from rde.storage import BlobStore

    store = BlobStore(settings.paths.blobs)
    before = store.total_bytes()
    _full_chain(settings, catalog)
    assert store.total_bytes() == before, "a derived version duplicated blob storage"
