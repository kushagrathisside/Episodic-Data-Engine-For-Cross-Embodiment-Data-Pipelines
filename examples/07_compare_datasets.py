"""Run the same engine over two structurally different datasets.

The architectural claim is that the pipeline does not depend on the shape of the
data. The way to test that is not to assert it, but to point the engine at a
second dataset that differs in every dimension that could plausibly break it and
see what had to change.

    SO-100 pick-place      ALOHA static coffee
    ------------------     -------------------
    6-DoF single arm       14-DoF bimanual
    2 cameras              4 cameras
    30 fps                 50 fps
    state + action         state + action + force/torque
    19,631 frames          55,000 frames
    470 MB                 1.63 GB

Adapter code changed: none. What did change was a *check calibration* -- see the
notes at the bottom.
"""

from __future__ import annotations

import json

from rde.catalog import Catalog
from rde.config import get_settings
from rde.engine import DataEngine
from rde.exploration.queries import DatasetExplorer

DATASETS = [("so100_pickplace", "SO-100 pick-place"), ("aloha_coffee", "ALOHA coffee")]


def main():
    settings = get_settings()
    catalog = Catalog(settings.paths.catalog_db)
    explorer = DatasetExplorer(settings=settings, catalog=catalog)
    engine = DataEngine(settings)

    available = [(d, label) for d, label in DATASETS if catalog.get_manifest(d, "raw/v1")]
    if len(available) < 2:
        print("Both datasets must be ingested first:")
        print("  rde run --config configs/so100_pickplace.json")
        print("  rde run --config configs/aloha_static_coffee.json")
        return

    print("=" * 92)
    print("SHAPE OF THE DATA")
    print("=" * 92)
    print(f"{'':22s} {'robot':16s} {'fps':>5s} {'episodes':>9s} {'frames':>9s} "
          f"{'cameras':>8s} {'modalities':>11s}")
    for dsid, label in available:
        o = explorer.overview(dsid, "raw/v1")
        cams = sum(1 for m in o.modalities if m["kind"] == "rgb")
        print(f"{label:22s} {str(o.robot_type):16s} {o.fps:>5.0f} {o.episodes:>9d} "
              f"{o.frames:>9d} {cams:>8d} {len(o.modalities):>11d}")

    print("\n" + "=" * 92)
    print("CANONICAL MODALITIES (derived, not configured)")
    print("=" * 92)
    for dsid, label in available:
        mods = explorer.modality_summary(dsid, "raw/v1")
        print(f"\n  {label}")
        for m in mods:
            dims = m["shape"][0] if len(m["shape"]) == 1 else "x".join(map(str, m["shape"]))
            print(f"    {m['name']:22s} {m['kind']:14s} {m['plane']:9s} {str(dims):>12s}")

    print("\n" + "=" * 92)
    print("THE PIPELINE, UNCHANGED")
    print("=" * 92)
    print(f"{'':22s} {'raw':>18s} {'clean':>18s} {'training splits':>26s}")
    for dsid, label in available:
        versions = {v["version"]: v for v in catalog.list_versions(dsid)}
        raw = versions.get("raw/v1", {})
        clean = versions.get("clean/v1", {})
        train = versions.get("training/v1")
        splits = "-"
        if train:
            manifest = catalog.get_manifest(dsid, "training/v1")
            sizes = manifest.statistics.get("splits", {})
            splits = " / ".join(f"{k}:{v['episodes']}" for k, v in sizes.items())
        raw_eps = f"{raw.get('total_episodes', 0)} ep"
        clean_eps = f"{clean.get('total_episodes', 0)} ep"
        print(f"{label:22s} {raw_eps:>18s} {clean_eps:>18s} {splits:>26s}")

    print("\n" + "=" * 92)
    print("QC: SAME 24 CHECKS, DIFFERENT FINDINGS")
    print("=" * 92)
    for dsid, label in available:
        summary = catalog.qc_summary(dsid, "raw/v1")
        eps = catalog.get_episodes(dsid, "raw/v1")
        from collections import Counter
        statuses = Counter(e["quality_status"] for e in eps)
        print(f"\n  {label}: {dict(statuses)}")
        for r in summary:
            if r["status"] in ("warned", "failed"):
                print(f"    {r['check_name']:34s} {r['severity']:8s} "
                      f"{r['n']:>4d} findings over {r['episodes']:>3d} episodes")

    print("\n" + "=" * 92)
    print("THE MODEL LAYER ALSO ADAPTS")
    print("=" * 92)
    for dsid, label in available:
        if not catalog.get_manifest(dsid, "training/v1"):
            continue
        view = engine.get_dataset(dsid, "training/v1")
        c = view.contract
        state_dims = sum(
            v["shape"][0] for v in c["observation"].values() if v["plane"] == "inline"
        )
        print(f"\n  {label}")
        print(f"    action dim        : {c['action']['shape'][0]}")
        print(f"    state input dim   : {state_dims} (concatenated inline modalities)")
        print(f"    cameras available : {[k for k, v in c['observation'].items() if v['plane']=='external']}")
        print(f"    instruction       : {c['tasks'][0]['description'][:56]!r}")
        view.close()

    print("\n" + "=" * 92)
    print("WHAT ACTUALLY HAD TO CHANGE")
    print("=" * 92)
    print("""
  Adapter code                         none
  Canonical schema                     none - force.effort mapped by the existing rule table
  Storage / catalog / versioning       none
  Model adapters                       none - they read the contract, so 6-DoF -> 14-DoF is data

  Check calibration                    YES. Learned bounds are percentiles, so ~1% of any
                                       dataset lies outside them by construction. At 6 dimensions
                                       that flagged 2 of 50 episodes; at 42 dimensions with
                                       long-tailed force data it flagged all 50 - a check that
                                       fires on everything carries no information.
                                       Fixed two ways: a finding now requires a *material*
                                       violation (systematic, or a large excursion), and the
                                       profiler widens bounds for impulsive kinds, because
                                       contact forces spike by physics rather than by fault.

  k-NN distance computation            YES. The broadcast form allocated 0.63 GB per block at
                                       55 k reference frames and would need 7.2 GB at 500 k.
                                       Rewritten via the gram-matrix identity: 22 MB per block,
                                       independent of dimensionality. Peak RSS 1.58 GB -> 675 MB.

  Both were latent bugs in the original code that only a second dataset exposed.
""")


if __name__ == "__main__":
    main()
