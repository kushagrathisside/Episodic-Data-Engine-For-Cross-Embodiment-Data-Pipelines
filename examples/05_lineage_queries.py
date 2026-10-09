"""Answer the two provenance questions the design exists to make answerable.

    1. Which raw episodes contributed to this training dataset?
    2. What transformations and QC checks were applied before it reached a model?

Both are single queries against the catalog, not an archaeology exercise.
"""

from __future__ import annotations

import json

from rde.catalog import Catalog
from rde.config import get_settings
from rde.exploration.queries import DatasetExplorer

DATASET = "so100_pickplace"
VERSION = "training/v1"


def main():
    settings = get_settings()
    catalog = Catalog(settings.paths.catalog_db)
    explorer = DatasetExplorer(settings=settings, catalog=catalog)

    lineage = explorer.lineage(DATASET, VERSION)

    print("=" * 78)
    print(f"Q1: which raw episodes contributed to {DATASET}@{VERSION}?")
    print("=" * 78)
    roots = lineage["root_episodes"]
    raw_eps = catalog.get_episodes(DATASET, "raw/v1")
    all_raw = {e["episode_id"] for e in raw_eps}
    excluded = sorted(all_raw - set(roots))
    print(f"  raw episodes        : {len(all_raw)}")
    print(f"  contributing        : {len(roots)}")
    print(f"  excluded            : {excluded}")

    for ep in excluded:
        findings = catalog.query_qc(DATASET, "raw/v1", episode_id=ep)
        print(f"\n  episode {ep} was excluded because:")
        for f in findings:
            print(f"     [{f['severity']}] {f['check_name']}")
            print(f"        {f['reason']}")

    print("\n" + "=" * 78)
    print(f"Q2: what was applied before {VERSION} reached a model?")
    print("=" * 78)
    for i, run in enumerate(lineage["chain"], 1):
        print(f"\n  step {i}: {run['operation']}")
        print(f"     {run['inputs'] or '(source)'} -> {run['output']}")
        print(f"     episodes {run['episodes_in']} -> {run['episodes_out']}"
              + (f", dropped {run['dropped']}" if run["dropped"] else ""))
        print(f"     software={run['software_version']} code={run['code_fingerprint']}")
        print(f"     reproducibility fingerprint={run['fingerprint']}")
        params = run["parameters"]
        if "operations" in params:
            for op in params["operations"]:
                mode = "deferred" if op["deferred"] else "materialised"
                print(f"        - {op['name']} ({mode})")
        elif params:
            print(f"        params: {json.dumps(params, default=str)[:180]}")

    print("\n  QC checks applied at each validated stage:")
    for version in ("raw/v1", "clean/v1"):
        summary = catalog.qc_summary(DATASET, version)
        checks = sorted({r["check_name"] for r in summary})
        findings = sum(r["n"] for r in summary)
        print(f"     {version:12s} {len(checks)} checks recorded findings, "
              f"{findings} findings total")

    print("\n" + "=" * 78)
    print("EPISODE-LEVEL ANCESTRY (first 8)")
    print("=" * 78)
    traced = catalog.trace_episodes_to_root(DATASET, VERSION)
    for ep_id, chain in sorted(traced.items())[:8]:
        path = " <- ".join(f"{v}#{e}" for v, e in chain)
        print(f"  training/v1#{ep_id:<4} <- {path}")


if __name__ == "__main__":
    main()
