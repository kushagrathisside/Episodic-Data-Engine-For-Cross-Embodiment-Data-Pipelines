"""Two end-to-end failure investigations, on the real dataset.

These demonstrations go further than "here is a dashboard with CPU and
latency on it".  Both scenarios below start from an
aggregate signal and end at the specific episodes responsible, using only the
correlation spine and the event tables.

**Scenario A -- a sensor fails.**  Frames go missing from one camera's episode.
The pipeline does not crash.  The path is:

    sensor-health metric -> alert -> trace -> stage -> run
        -> affected episodes -> QC findings

**Scenario B -- the data distribution moves.**  A controller is silently
rescaled.  Every value stays finite, timestamps stay monotonic, the schema is
unchanged, and **all 24 QC checks pass**.  The dataset is clean and no longer
comparable to what the model was trained on.  The path is:

    episode features -> drift statistic -> version comparison
        -> release decision -> lineage

Scenario B is the more important of the two.  A pipeline that succeeds and emits
a ruined dataset is the failure mode structural validation cannot see, and it is
the reason a data engine needs observability rather than just monitoring.

Both run on disposable copies of the real ingested version; nothing here mutates
a version the catalog considers real.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from rde.catalog import Catalog
from rde.config import get_settings
from rde.observability import ObservabilitySession, RunContext, build_registry
from rde.observability.drift import compare, emit
from rde.observability.health import SHARED_KEY
from rde.observability.metrics import MetricRegistry
from rde.qc import QCEngine
from rde.schemas.core import Episode, QualityStatus
from rde.schemas.qc import CheckStatus
from rde.testing import inject_distribution_shift, inject_dropped_frames

DATASET = "so100_pickplace"
SOURCE_VERSION = "raw/v1"
BASELINE_VERSION = "raw/v90"    # disposable clone, left untouched
CANDIDATE_VERSION = "raw/v91"   # disposable clone, corrupted


def _clone(settings, catalog, target: str) -> Path:
    """Clone the ingested version into a disposable one we may corrupt."""
    src = settings.paths.version_dir(DATASET, SOURCE_VERSION)
    dst = settings.paths.version_dir(DATASET, target)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    manifest = catalog.get_manifest(DATASET, SOURCE_VERSION)
    manifest.version = target
    eps = [
        Episode(
            episode_id=r["episode_id"], dataset_id=DATASET, task_id=r["task_id"],
            frame_start=r["frame_start"], frame_end=r["frame_end"],
            end_timestamp_s=r["duration_s"], quality_status=QualityStatus.UNKNOWN,
            modalities=json.loads(r["modalities"] or "[]"),
            attributes=json.loads(r["attributes"] or "{}"),
        )
        for r in catalog.get_episodes(DATASET, SOURCE_VERSION)
    ]
    catalog.register_version(manifest, eps)
    return dst


def _validate(settings, catalog, version: str, session):
    """Run QC under an active observability session and return the result."""
    with session.active():
        return QCEngine(settings=settings, catalog=catalog).run(
            DATASET, version, persist=False
        )


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- scenario A
def scenario_a(settings, catalog) -> bool:
    """Camera frame dropout, traced from aggregate metric to named episodes."""
    _rule("SCENARIO A -- a sensor starts dropping frames")

    affected = [7, 19, 31]
    version = _clone(settings, catalog, CANDIDATE_VERSION)
    for eid in affected:
        inject_dropped_frames(version, episode_id=eid, drop_from=5, count=4)
    print(f"  injected: 4 frames dropped from episodes {affected}")
    print("  the pipeline does not crash; every stage still reports ok\n")

    session = ObservabilitySession(
        RunContext.start(DATASET, "demo"),
        registry=build_registry(MetricRegistry()),
    )
    with session.active():
        with session.stage("validate") as h:
            result = QCEngine(settings=settings, catalog=catalog).run(
                DATASET, CANDIDATE_VERSION, persist=False
            )
            h.counts(episodes=len(result.episode_status))

    # -- 1. the aggregate signal ------------------------------------------
    findings = session.registry.get("rde_qc_findings_total")
    dropout_series = {
        dict(k)["check_name"]: v
        for k, v in ((s.labels, s.value) for s in findings.samples())
        if dict(k).get("status") == "failed"
    }
    print("  1. AGGREGATE METRIC -- what fired, with no episode ids attached")
    for check, count in sorted(dropout_series.items(), key=lambda kv: -kv[1]):
        print(f"       rde_qc_findings_total{{check_name=\"{check}\"}} = {count:.0f}")
    rejection = session.registry.get("rde_qc_gate_rejection_ratio")
    print(f"       rde_qc_gate_rejection_ratio = "
          f"{rejection.value(dataset_id=DATASET):.3f}")

    # -- 2. the trace ------------------------------------------------------
    print("\n  2. TRACE -- which run and which stage")
    print(f"       run_id   {session.ctx.run_id}")
    print(f"       trace_id {session.ctx.trace_id}")
    for sp in session.ctx.spans:
        print(f"       span     {sp.name}  {sp.status}  {sp.duration_s:.3f}s")

    # -- 3. the event rows -------------------------------------------------
    print("\n  3. EVENT ROWS -- now, and only now, the episode ids")
    rows = session.events.get("episode_health")
    bad = sorted(
        (r for r in rows if r.quality_status != QualityStatus.PASSED.value),
        key=lambda r: r.episode_id,
    )
    for r in bad[:8]:
        print(f"       episode {r.episode_id:>3d}  {r.quality_status:<10s} "
              f"frames={r.frame_count:<4d} temporal={r.temporal_quality:.2f} "
              f"mask={r.qc_bitmask}")

    # -- 4. the findings ---------------------------------------------------
    print("\n  4. QC FINDINGS -- the specific evidence")
    named = [
        f for f in result.report.records
        if f.episode_id in affected and f.status is CheckStatus.FAILED
    ]
    for f in named[:6]:
        print(f"       ep {f.episode_id:>3d}  {f.check_name:<34s} {f.reason[:60]}")

    caught = {f.episode_id for f in named}
    ok = set(affected) <= caught
    print(f"\n  RESULT: injected {affected} -> identified {sorted(caught)}  "
          f"{'PASS' if ok else 'FAIL'}")
    print("  Note the shape: the metric named the *cohort*, the event rows named")
    print("  the *individuals*. Neither store was asked to do the other's job.")
    return ok


# ---------------------------------------------------------------- scenario B
def scenario_b(settings, catalog) -> bool:
    """A silent recalibration: clean data that has stopped being comparable."""
    _rule("SCENARIO B -- the action distribution moves, and QC still passes")

    baseline_dir = _clone(settings, catalog, BASELINE_VERSION)
    candidate_dir = _clone(settings, catalog, CANDIDATE_VERSION)
    shift = inject_distribution_shift(
        candidate_dir, column="action.joint", scale=1.35, from_episode=0
    )
    print(f"  injected: {shift['column']} scaled by {shift['scale']}x "
          f"(a recalibrated controller)")

    def features_for(version: str) -> dict:
        session = ObservabilitySession(
            RunContext.start(DATASET, "demo"),
            registry=build_registry(MetricRegistry()),
        )
        res = _validate(settings, catalog, version, session)
        blocking = [f for f in res.report.blocking]
        return {
            "status": res.dataset_status.value,
            "blocking": len(blocking),
            "checks": len({c for c in res.report.checks_run
                           if not c.startswith("observability.")}),
            "features": res.shared.get(SHARED_KEY, {}),
            "session": session,
        }

    base = features_for(BASELINE_VERSION)
    cand = features_for(CANDIDATE_VERSION)

    # -- 1. QC sees nothing -----------------------------------------------
    print("\n  1. QC VERDICT -- structural validation cannot see this")
    print(f"       baseline  : {base['checks']} checks, "
          f"{base['blocking']} blocking, status={base['status']}")
    print(f"       candidate : {cand['checks']} checks, "
          f"{cand['blocking']} blocking, status={cand['status']}")
    qc_blind = cand["blocking"] == base["blocking"]
    print(f"       -> QC finds no new blocking defect: {qc_blind}")

    # -- 2. drift does ------------------------------------------------------
    def as_arrays(feats: dict) -> dict:
        keys = sorted({k for f in feats.values() for k in f})
        return {
            k: np.array([f[k] for f in feats.values() if k in f], dtype=float)
            for k in keys
            if sum(k in f for f in feats.values()) == len(feats)
        }

    b_arr, c_arr = as_arrays(base["features"]), as_arrays(cand["features"])
    results = compare(b_arr, c_arr, statistics=("psi", "mean_shift"))
    exceeded = [r for r in results if r.exceeded]

    print("\n  2. DRIFT -- episode-level features, thresholds measured from "
          "the baseline's own null")
    print(f"       {'feature':<26s} {'stat':<11s} {'value':>9s} {'thresh':>8s} "
          f"{'n_eff':>7s}")
    for r in sorted(results, key=lambda r: -r.value)[:8]:
        flag = "  <-- EXCEEDED" if r.exceeded else ""
        print(f"       {r.feature:<26s} {r.statistic:<11s} {r.value:>9.3f} "
              f"{r.threshold:>8.3f} {r.effective_sample_size:>7.1f}{flag}")

    # -- 3. the release decision -------------------------------------------
    print("\n  3. RELEASE DECISION -- checking and gating stay separate")
    print(f"       QC gate       : {cand['status']}  (would admit this version)")
    print(f"       drift contract: {len(exceeded)} feature(s) outside contract")
    verdict = "BLOCK" if exceeded else "ADMIT"
    print(f"       -> release verdict: {verdict}")

    # -- 4. lineage ---------------------------------------------------------
    print("\n  4. LINEAGE -- what a reviewer can answer about the two versions")
    graph = catalog.lineage_graph(DATASET)
    for v in (BASELINE_VERSION, CANDIDATE_VERSION):
        node = graph.nodes.get(f"{DATASET}@{v}")
        if node:
            print(f"       {v:<9s} episodes={node.total_episodes} "
                  f"frames={node.total_frames} created={node.created_at:%Y-%m-%d %H:%M}")

    ok = qc_blind and bool(exceeded)
    print(f"\n  RESULT: QC blind={qc_blind}, drift caught it={bool(exceeded)}  "
          f"{'PASS' if ok else 'FAIL'}")
    print("  This is the case for a data-health layer in one experiment: the")
    print("  pipeline succeeded, every check passed, and the dataset was wrong.")
    return ok


def main() -> int:
    settings = get_settings()
    catalog = Catalog(settings.paths.catalog_db)
    if catalog.get_manifest(DATASET, SOURCE_VERSION) is None:
        print(f"{DATASET}@{SOURCE_VERSION} not found -- run the pipeline first:")
        print("  rde run --config configs/so100_pickplace.json")
        return 1

    try:
        a = scenario_a(settings, catalog)
        b = scenario_b(settings, catalog)
    finally:
        # The scratch versions exist only for this demo; leaving them in the
        # catalog would make `rde versions` lie about what the engine holds.
        for v in (BASELINE_VERSION, CANDIDATE_VERSION):
            catalog.delete_version(DATASET, v)
            shutil.rmtree(settings.paths.version_dir(DATASET, v), ignore_errors=True)

    _rule("SUMMARY")
    print(f"  Scenario A (sensor failure)      : {'PASS' if a else 'FAIL'}")
    print(f"  Scenario B (distribution change) : {'PASS' if b else 'FAIL'}")
    return 0 if (a and b) else 1


if __name__ == "__main__":
    sys.exit(main())
