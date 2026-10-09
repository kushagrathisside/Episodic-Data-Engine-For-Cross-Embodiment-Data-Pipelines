"""Demonstrate that the QC engine catches defects it claims to catch.

The reference dataset is a curated public release and is genuinely clean, so a QC
run against it reports almost nothing.  That is evidence about the *dataset*, not
about the engine.  This script proves the checks work by breaking the data on
purpose -- one defect class at a time, on a disposable copy of the real ingested
version -- and showing which check fires.

Each fault is injected into a fresh copy, so the results are independent.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from rde.catalog import Catalog
from rde.config import get_settings
from rde.qc import QCEngine
from rde.schemas.qc import CheckStatus
from rde.testing import (
    inject_discontinuity,
    inject_dropped_frames,
    inject_duplicate_timestamp,
    inject_nan,
    inject_non_monotonic,
    inject_stuck_sensor,
    inject_timestamp_gap,
    inject_video_misalignment,
)

DATASET = "so100_pickplace"
SOURCE_VERSION = "raw/v1"
SCRATCH_VERSION = "raw/v99"

FAULTS = [
    ("NaN in state vector", lambda d: inject_nan(d, "state.joint", row=5, dim=2)),
    ("2s timestamp gap", lambda d: inject_timestamp_gap(d, row=10, gap_s=2.0)),
    ("out-of-order timestamps", lambda d: inject_non_monotonic(d, row=8)),
    ("duplicate timestamp", lambda d: inject_duplicate_timestamp(d, row=12)),
    ("stuck joint (whole episode)", lambda d: inject_stuck_sensor(d, "state.joint", 1, 0)),
    ("impossible action jump", lambda d: inject_discontinuity(d, "action.joint", 15, 500.0)),
    ("3 frames dropped mid-episode", lambda d: inject_dropped_frames(d, 2, 5, 3)),
    ("video refs shifted 5s", lambda d: inject_video_misalignment(d, shift_s=5.0)),
]


def _prepare_scratch(settings, catalog) -> Path:
    """Clone the ingested version into a disposable one we can corrupt."""
    src = settings.paths.version_dir(DATASET, SOURCE_VERSION)
    dst = settings.paths.version_dir(DATASET, SCRATCH_VERSION)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    manifest = catalog.get_manifest(DATASET, SOURCE_VERSION)
    manifest.version = SCRATCH_VERSION
    episodes = catalog.get_episodes(DATASET, SOURCE_VERSION)

    from rde.schemas.core import Episode, QualityStatus
    import json

    eps = [
        Episode(
            episode_id=r["episode_id"], dataset_id=DATASET, task_id=r["task_id"],
            frame_start=r["frame_start"], frame_end=r["frame_end"],
            end_timestamp_s=r["duration_s"],
            quality_status=QualityStatus.UNKNOWN,
            modalities=json.loads(r["modalities"] or "[]"),
            attributes=json.loads(r["attributes"] or "{}"),
        )
        for r in episodes
    ]
    catalog.register_version(manifest, eps)
    return dst


def main() -> int:
    settings = get_settings()
    catalog = Catalog(settings.paths.catalog_db)

    if catalog.get_manifest(DATASET, SOURCE_VERSION) is None:
        print(f"{DATASET}@{SOURCE_VERSION} not found -- run the pipeline first:")
        print("  rde run --config configs/so100_pickplace.json")
        return 1

    print("=" * 78)
    print("BASELINE: the real dataset, uncorrupted")
    print("=" * 78)
    scratch = _prepare_scratch(settings, catalog)
    engine = QCEngine(settings=settings, catalog=catalog)
    base = engine.run(DATASET, SCRATCH_VERSION, persist=False)
    blocking = base.report.blocking
    print(f"  checks run       : {len(base.report.checks_run)}")
    print(f"  blocking findings: {len(blocking)}")
    print(f"  status           : {base.dataset_status.value}")

    print("\n" + "=" * 78)
    print("FAULT INJECTION: one defect at a time, fresh copy each run")
    print("=" * 78)
    print(f"{'injected defect':32s} {'detected by':36s} sev")
    print("-" * 78)

    failures = 0
    for label, inject in FAULTS:
        scratch = _prepare_scratch(settings, catalog)
        inject(scratch)
        res = engine.run(DATASET, SCRATCH_VERSION, persist=False)
        fired = [
            r for r in res.report.records
            if r.status in (CheckStatus.FAILED, CheckStatus.WARNED)
        ]
        if not fired:
            print(f"{label:32s} {'** NOT DETECTED **':36s} -")
            failures += 1
            continue
        # Report the most severe check that fired.
        order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
        fired.sort(key=lambda r: order.get(r.severity.value, 9))
        top = fired[0]
        others = len({r.check_name for r in fired}) - 1
        name = top.check_name + (f" (+{others})" if others > 0 else "")
        print(f"{label:32s} {name:36s} {top.severity.value}")

    shutil.rmtree(settings.paths.version_dir(DATASET, SCRATCH_VERSION), ignore_errors=True)
    with catalog.transaction() as conn:
        conn.execute("DELETE FROM versions WHERE version_key=?",
                     (f"{DATASET}@{SCRATCH_VERSION}",))

    print("-" * 78)
    if failures:
        print(f"{failures} defect class(es) went undetected")
        return 1
    print("every injected defect was detected by at least one check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
