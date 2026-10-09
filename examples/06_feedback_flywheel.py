"""Close the loop: model failures become a new dataset version.

    inference -> failure records -> episode selection -> review
        -> new dataset version -> retraining dataset

What is real here: the failure records, the selection query, the quarantine
marking, the new version and its lineage.  What is simulated: the failure
*signal* itself.  There is no robot and no environment, so "failure" means large
prediction error or low model confidence against held-out demonstrations, which
is a proxy for deployment failure, not the thing itself.
"""

from __future__ import annotations

import json

from rde.catalog import Catalog
from rde.config import get_settings
from rde.dataset_builder import DatasetBuilder
from rde.engine import DataEngine
from rde.evaluation import Evaluator
from rde.feedback import FeedbackLoop
from rde.model_adapters import build_model

DATASET = "so100_pickplace"
V1 = "training/v1"
V2 = "training/v2"


def main():
    settings = get_settings()
    catalog = Catalog(settings.paths.catalog_db)
    engine = DataEngine(settings)

    # -- 1. inference ------------------------------------------------------
    print("=" * 78)
    print("1. INFERENCE on the held-out split")
    print("=" * 78)
    view = engine.get_dataset(DATASET, V1)
    model, adapter = build_model("nearest_neighbor", view.contract)
    model.fit(view, split="train")
    result = Evaluator(settings).evaluate(view, model, adapter, split="test", persist=False)
    view.close()
    print(f"  model            : {result.model_name}")
    print(f"  frames evaluated : {result.frames_evaluated}")
    print(f"  MAE              : {result.metrics['mae']:.5f}")
    print(f"  mean confidence  : {result.metrics['mean_confidence']:.3f}")
    print(f"  conf/error corr  : {result.metrics['confidence_error_corr']:.3f}"
          "   (negative = confidence tracks correctness)")

    # -- 2. failure detection ---------------------------------------------
    print("\n" + "=" * 78)
    print("2. FAILURE RECORDS")
    print("=" * 78)
    loop = FeedbackLoop(settings=settings, catalog=catalog)
    report = loop.detect(result, error_percentile=90.0)
    print(f"  thresholds       : {json.dumps(report.thresholds)}")
    print(f"  failure frames   : {len(report.failures)}")
    by_mode = {}
    for f in report.failures:
        by_mode[f.mode.value] = by_mode.get(f.mode.value, 0) + 1
    print(f"  by mode          : {by_mode}")

    # -- 3. episode selection ---------------------------------------------
    print("\n" + "=" * 78)
    print("3. EPISODES NOMINATED FOR REVIEW")
    print("=" * 78)
    print(f"{'ep':>4} {'fail/total':>12} {'mean err':>10} {'mean conf':>10} "
          f"{'priority':>9}  recommendation")
    print("-" * 78)
    for r in report.reviews:
        print(f"{r.episode_id:>4} {f'{r.failure_frames}/{r.total_frames_evaluated}':>12} "
              f"{r.mean_error:>10.5f} {r.mean_confidence:>10.4f} {r.priority:>9.4f}  "
              f"{r.recommendation[:34]}")

    # -- 4. data review ----------------------------------------------------
    print("\n" + "=" * 78)
    print("4. DATA REVIEW: quarantine, do not delete")
    print("=" * 78)
    flagged = [r.episode_id for r in report.reviews]
    loop.quarantine_episodes(DATASET, V1, flagged)
    rows = catalog.get_episodes(DATASET, V1)
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["quality_status"]] = statuses.get(r["quality_status"], 0) + 1
    print(f"  flagged episodes : {flagged}")
    print(f"  episode statuses : {statuses}")
    print(f"  total episodes   : {len(rows)} (unchanged -- nothing was deleted)")

    # -- 5. new dataset version -------------------------------------------
    print("\n" + "=" * 78)
    print("5. NEW TRAINING VERSION with failure-informed sampling")
    print("=" * 78)
    selection = loop.build_retraining_selection(report, oversample_factor=3)
    weights = {int(k): float(v) for k, v in selection["episode_weights"].items()}

    build = DatasetBuilder(settings=settings, catalog=catalog).build(
        DATASET, "clean/v1", V2,
        splits={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=42, episode_weights=weights,
    )
    print(f"  {V2}: {build.episodes} episodes, {build.frames} frames")
    print(f"  splits           : { {k: v['episodes'] for k, v in build.split_sizes.items()} }")
    print(f"  upweighted       : {sorted(k for k, v in weights.items() if v > 1)}")
    print(f"  rationale        : {selection['rationale']}")

    # -- 6. the loop is recorded ------------------------------------------
    print("\n" + "=" * 78)
    print("6. THE LOOP IS IN THE LINEAGE")
    print("=" * 78)
    from rde.exploration.queries import DatasetExplorer

    lin = DatasetExplorer(settings=settings, catalog=catalog).lineage(DATASET, V2)
    for run in lin["chain"]:
        print(f"  {run['operation']:24s} {run['inputs']} -> {run['output']}")
    build_run = lin["chain"][-1]
    ew = build_run["parameters"].get("episode_weights")
    print(f"\n  recorded episode weights: {ew}")
    print("  A later reader can see exactly which episodes were oversampled and why.")


if __name__ == "__main__":
    main()
