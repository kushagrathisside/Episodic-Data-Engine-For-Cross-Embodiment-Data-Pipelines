"""Ingestion tests: source format -> canonical model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pytest

from rde.ingestion import LeRobotAdapter, canonical_name, ingest
from rde.ingestion.lerobot import _flatten_names
from rde.schemas.modality import ModalityKind, StoragePlane
from rde.storage import ParquetTableStore


@pytest.mark.parametrize(
    "source_key,expected_name,expected_kind",
    [
        ("observation.images.top", "rgb.top", ModalityKind.RGB),
        ("observation.images.wrist", "rgb.wrist", ModalityKind.RGB),
        ("observation.state", "state.joint", ModalityKind.ROBOT_STATE),
        ("action", "action.joint", ModalityKind.ROBOT_ACTION),
        ("observation.depth.front", "depth.front", ModalityKind.DEPTH),
        ("observation.effort", "force.effort", ModalityKind.FORCE_TORQUE),
        ("observation.tactile.left", "tactile.left", ModalityKind.TACTILE),
    ],
)
def test_canonical_name_mapping(source_key, expected_name, expected_kind):
    """Source keys map to canonical names by table, not by special case."""
    name, kind = canonical_name(source_key)
    assert name == expected_name
    assert kind == expected_kind


def test_unknown_key_is_kept_not_dropped():
    """An unrecognised feature must survive ingestion, namespaced."""
    name, kind = canonical_name("observation.mystery_sensor")
    assert name.startswith("other.")
    assert kind is ModalityKind.OTHER


def test_flatten_names_passes_through_a_flat_list():
    """Single-limb robots (so100) already give a flat list -- leave it alone."""
    assert _flatten_names(["main_shoulder_pan", "main_gripper"]) == [
        "main_shoulder_pan", "main_gripper",
    ]


def test_flatten_names_flattens_grouped_dict():
    """Multi-limb robots (bimanual ALOHA) group names by sub-limb -- e.g.
    ``{"motors": [...]}`` -- rather than giving a flat list. Every downstream
    reader of element_names (rde.exploration.queries.dimensionality, dataset
    profiling, training contracts) indexes it positionally by dimension, so
    ingestion must flatten this before it ever reaches a manifest."""
    grouped = {"motors": ["left_waist", "left_gripper", "right_waist", "right_gripper"]}
    assert _flatten_names(grouped) == [
        "left_waist", "left_gripper", "right_waist", "right_gripper",
    ]


def test_flatten_names_concatenates_multiple_groups_in_order():
    grouped = {"left": ["l0", "l1"], "right": ["r0", "r1"]}
    assert _flatten_names(grouped) == ["l0", "l1", "r0", "r1"]


def test_inspect_reads_metadata_only(source_dataset: Path):
    adapter = LeRobotAdapter(source_dataset, "testds")
    profile = adapter.inspect()
    assert profile.total_episodes == 4
    assert profile.total_frames == 120
    assert profile.fps == 10
    assert profile.source_format.startswith("lerobot/")
    assert not profile.anomalies


def test_task_description_read_from_pandas_index(source_dataset: Path):
    """tasks.parquet stores the description as __index_level_0__; handle it."""
    adapter = LeRobotAdapter(source_dataset, "testds")
    tasks = adapter._tasks()
    assert tasks and tasks[0].description
    assert "cube" in tasks[0].description.lower()


def test_ingest_produces_canonical_schema(ingested, version_dir: Path):
    store = ParquetTableStore(version_dir)
    schema = store.schema("frames")
    for col in ("frame_id", "episode_id", "frame_index", "timestamp_s", "task_id"):
        assert col in schema.names
    assert "state.joint" in schema.names
    assert "action.joint" in schema.names
    for ref in ("rgb.top__blob_id", "rgb.top__frame_in_blob", "rgb.top__ts_in_blob"):
        assert ref in schema.names


def test_inline_vectors_are_fixed_width(ingested, version_dir: Path):
    """Fixed-size lists make a dimension mismatch a write-time error."""
    schema = ParquetTableStore(version_dir).schema("frames")
    assert schema.field("state.joint").type.list_size == 6
    assert schema.field("action.joint").type.list_size == 6


def test_row_groups_are_bounded_unlike_source(settings, ingested, version_dir: Path):
    """Upstream writes one row group; ingestion must rewrite with many."""
    stats = ParquetTableStore(version_dir).stats("frames")
    assert stats["row_groups"] > 1, "ingestion did not re-chunk the row groups"


def test_video_reference_mapping_is_correct(ingested, version_dir: Path, settings):
    """ts_in_blob must equal episode video start + episode-relative timestamp."""
    store = ParquetTableStore(version_dir)
    frames = store.read_table("frames", filter=ds.field("episode_id") == 2)
    episodes = store.read_table("episodes").to_pydict()
    idx = episodes["episode_id"].index(2)
    windows = json.loads(episodes["attributes"][idx])["video_windows"]
    start = windows["rgb.top"]["from_timestamp"]

    ts = np.asarray(frames.column("timestamp_s"))
    stored = np.asarray(frames.column("rgb.top__ts_in_blob"))
    assert np.allclose(stored, start + ts, atol=1e-6)


def test_blobs_registered_and_shared(ingested, settings):
    """One blob per camera, admitted into the content-addressed store."""
    from rde.storage import BlobStore

    store = BlobStore(settings.paths.blobs)
    assert set(ingested.blob_ids) == {"rgb.top", "rgb.wrist"}
    for blob_id in ingested.blob_ids.values():
        assert store.exists(blob_id)
        assert store.verify(blob_id)


def test_ingest_is_idempotent(settings, source_dataset: Path, catalog, ingested):
    """Re-ingesting must converge, not accumulate duplicates."""
    again = ingest(source_dataset, "testds", "lerobot", "raw/v1",
                   settings=settings, catalog=catalog)
    assert again.frames_written == ingested.frames_written
    versions = [v for v in catalog.list_versions("testds") if v["version"] == "raw/v1"]
    assert len(versions) == 1


def test_manifest_written_beside_data(settings, ingested):
    """A version directory must be self-describing without the catalog."""
    art = settings.paths.artifact_dir("testds", "raw/v1") / "manifest.json"
    assert art.exists()
    manifest = json.loads(art.read_text())
    assert manifest["dataset_id"] == "testds"
    assert manifest["schema_version"]
