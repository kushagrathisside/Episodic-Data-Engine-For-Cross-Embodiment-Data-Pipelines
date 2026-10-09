"""Storage plane tests: blob addressing, Parquet streaming, video decode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from rde.storage import BlobStore, ParquetTableStore, VideoReader, probe, sha256_file


def test_blob_content_addressing_deduplicates(tmp_path: Path):
    """Identical bytes stored twice must occupy one address, not two."""
    store = BlobStore(tmp_path / "blobs")
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"identical payload")
    b.write_bytes(b"identical payload")

    ia = store.put(a, media_type="application/octet-stream")
    ib = store.put(b, media_type="application/octet-stream")
    assert ia.blob_id == ib.blob_id
    assert len(list(store.iter_blobs())) == 1
    assert store.verify(ia.blob_id)


def test_blob_link_mode_avoids_duplication(tmp_path: Path):
    """Link mode must not copy bytes -- the whole point at video scale."""
    store = BlobStore(tmp_path / "blobs")
    src = tmp_path / "big.bin"
    src.write_bytes(b"x" * 100_000)
    info = store.put(src, mode="link")
    stored = store.get(info.blob_id)
    assert stored.stat().st_ino == src.stat().st_ino, "link mode copied instead of linking"


def test_blob_detects_corruption(tmp_path: Path):
    store = BlobStore(tmp_path / "blobs")
    src = tmp_path / "f.bin"
    src.write_bytes(b"original")
    info = store.put(src, mode="copy")
    assert store.verify(info.blob_id)
    store.get(info.blob_id).write_bytes(b"tampered")
    assert not store.verify(info.blob_id), "corrupted blob passed verification"


def test_parquet_row_groups_and_streaming(tmp_path: Path):
    """Writer must honour the configured row-group size, and reads must stream."""
    store = ParquetTableStore(tmp_path / "t", row_group_size=64)
    table = pa.table({"i": pa.array(range(500), pa.int64()),
                      "g": pa.array([i % 5 for i in range(500)], pa.int64())})
    store.write_table("data", table)

    stats = store.stats("data")
    assert stats["rows"] == 500
    assert stats["row_groups"] >= 7, "row groups were not bounded as configured"

    batches = list(store.iter_batches("data", columns=["i"], batch_size=100))
    assert sum(b.num_rows for b in batches) == 500
    assert batches[0].schema.names == ["i"], "column projection was ignored"


def test_parquet_predicate_pushdown(tmp_path: Path):
    import pyarrow.dataset as ds

    store = ParquetTableStore(tmp_path / "t", row_group_size=32)
    store.write_table("data", pa.table({
        "episode_id": pa.array([i // 50 for i in range(500)], pa.int64()),
        "v": pa.array(range(500), pa.int64()),
    }))
    got = store.read_table("data", filter=ds.field("episode_id") == 3)
    assert got.num_rows == 50
    assert set(np.asarray(got.column("episode_id")).tolist()) == {3}


def test_empty_table_still_written(tmp_path: Path):
    """An empty result must produce a readable table, not a missing file."""
    store = ParquetTableStore(tmp_path / "t")
    schema = pa.schema([pa.field("a", pa.int64())])
    store.write_batches("empty", iter([]), schema=schema)
    assert store.exists("empty")
    assert store.read_table("empty").num_rows == 0


def test_video_probe_and_decode(source_dataset: Path):
    """Probe reads headers without decoding; reads return RGB frames."""
    path = next((source_dataset / "videos").rglob("*.mp4"))
    info = probe(path)
    assert info.width > 0 and info.height > 0
    assert info.num_frames > 0

    with VideoReader(path, cache_frames=4) as reader:
        img = reader.read_at(0.0)
        assert img is not None and img.ndim == 3 and img.shape[2] == 3
        assert img.dtype == np.uint8

        seq = list(reader.iter_range(0.0, 0.5))
        assert seq, "sequential range decoded nothing"
        assert all(f.shape == img.shape for _, f in seq)


def test_video_cache_is_bounded(source_dataset: Path):
    """The frame cache must not grow past its budget."""
    path = next((source_dataset / "videos").rglob("*.mp4"))
    with VideoReader(path, cache_frames=3) as reader:
        for i in range(10):
            reader.read_at(i * 0.1)
        assert len(reader._cache) <= 3
