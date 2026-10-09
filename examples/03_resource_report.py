"""Measure and report the engine's actual resource usage.

Every number in docs/09-resource-usage.md comes from this script.  It exists so
the resource claims can be re-verified on other hardware rather than trusted.
"""

from __future__ import annotations

import json
import resource
import subprocess
import time
from pathlib import Path

from rde.config import get_settings
from rde.engine import DataEngine
from rde.storage import BlobStore, ParquetTableStore, VideoReader, probe

DATASET = "so100_pickplace"


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _du(path: Path, count_links: bool = False) -> int:
    """Bytes on disk under ``path``.

    ``du`` deduplicates hardlinks *within a single invocation*, so measuring
    directories separately counts a hardlinked blob once per directory and hides
    the saving entirely.  ``count_links=True`` forces every link to be counted,
    which is what the apparent (pre-dedup) figure needs.
    """
    if not path.exists():
        return 0
    cmd = ["du", "-sb"] + (["--count-links"] if count_links else []) + [str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    return int(out.split()[0]) if out else 0


def main() -> dict:
    s = get_settings()
    p = s.paths
    report: dict = {}

    # -- storage footprint --------------------------------------------------
    blob_store = BlobStore(p.blobs)
    logical_blob_bytes = blob_store.total_bytes()
    report["storage"] = {
        "source_cache_bytes": _du(p.source_cache),
        "blobs_apparent_bytes": logical_blob_bytes,
        "blobs_physical_bytes": _du(p.blobs),
        "tables_bytes": _du(p.tables),
        "artifacts_bytes": _du(p.artifacts),
        "catalog_bytes": p.catalog_db.stat().st_size if p.catalog_db.exists() else 0,
    }
    # The blob store hardlinks into the source cache, so the two planes share
    # extents.  Measure the whole root both ways: with hardlinks deduplicated
    # (real disk) and with every link counted (what a copying store would cost).
    physical = _du(p.data_root)
    apparent = _du(p.data_root, count_links=True)
    report["storage"]["data_root_physical_bytes"] = physical
    report["storage"]["data_root_apparent_bytes"] = apparent
    report["storage"]["blob_dedup_saving_bytes"] = max(0, apparent - physical)

    # -- per-version table sizes -------------------------------------------
    versions = {}
    engine = DataEngine(s)
    for v in engine.list_versions(DATASET):
        store = ParquetTableStore(p.version_dir(DATASET, v["version"]))
        entry = {"episodes": v["total_episodes"], "frames": v["total_frames"]}
        for tbl in ("frames", "episodes", "index"):
            if store.exists(tbl):
                entry[tbl] = store.stats(tbl)
        versions[v["version"]] = entry
    report["versions"] = versions

    # -- video: naive vs streamed ------------------------------------------
    blob_id = next(iter(blob_store.iter_blobs())).blob_id
    path = blob_store.get(blob_id)
    info = probe(path)
    report["video"] = {
        "path_bytes": path.stat().st_size,
        "codec": info.codec,
        "resolution": [info.width, info.height],
        "frames": info.num_frames,
        "duration_s": round(info.duration_s, 2),
        "naive_full_decode_bytes": info.raw_rgb_bytes,
    }

    with VideoReader(path, cache_frames=0) as r:
        t0 = time.perf_counter()
        for ts in (10.0, 100.0, 300.0, 500.0):
            r.read_at(ts)
        report["video"]["random_seek_ms_per_frame"] = round(
            (time.perf_counter() - t0) * 1000 / 4, 2
        )
        t0 = time.perf_counter()
        n = sum(1 for _ in r.iter_range(0.0, 5.0))
        report["video"]["sequential_ms_per_frame"] = round(
            (time.perf_counter() - t0) * 1000 / max(n, 1), 3
        )

    # -- streaming read of the whole frame table ---------------------------
    view = engine.get_dataset(DATASET, "training/v1")
    t0 = time.perf_counter()
    rows = sum(b.num_rows for b in view.iter_frames(split="train"))
    report["frame_scan"] = {
        "rows": rows,
        "seconds": round(time.perf_counter() - t0, 3),
        "peak_rss_mb_after": round(_rss_mb(), 1),
    }

    # -- sample iteration, with and without image decode -------------------
    t0 = time.perf_counter()
    n = 0
    for _ in view.iter_samples(split="test"):
        n += 1
        if n >= 500:
            break
    report["samples_no_images"] = {
        "samples": n, "seconds": round(time.perf_counter() - t0, 3),
    }

    t0 = time.perf_counter()
    n = 0
    for smp in view.iter_samples(split="test"):
        smp.image("rgb.top")
        n += 1
        if n >= 100:
            break
    report["samples_with_images"] = {
        "samples": n, "seconds": round(time.perf_counter() - t0, 3),
        "ms_per_sample": round((time.perf_counter() - t0) * 1000 / max(n, 1), 2),
    }
    view.close()

    report["peak_rss_mb"] = round(_rss_mb(), 1)
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
