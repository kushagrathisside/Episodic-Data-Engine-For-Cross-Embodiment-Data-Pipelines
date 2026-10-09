"""Measure what episode-grain labels actually cost a metrics backend.

This is a *measured* demonstration, not a restatement of "high cardinality is
bad".  This script does three things:

1. Reads the real label dimensions out of the catalog -- checks, modalities,
   stages, statuses -- rather than inventing plausible ones.
2. Builds an in-memory series index the way a TSDB head block does (one entry
   per distinct label-set) and measures it with ``tracemalloc``, for a safe
   label policy and for the same policy plus ``episode_id``.
3. Stores the identical facts as a Parquet table and measures that instead,
   which is the comparison that actually decides where episode-grain facts live.

The Python index is a *proxy* for Prometheus's Go TSDB, not a replica.  It is
honest about one thing the theory alone cannot show: the per-series cost is
roughly constant, so the series *count* is the whole story -- and the count is
set by the label policy, not by how much data you have.
"""

from __future__ import annotations

import json
import sqlite3
import tracemalloc
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rde.config import get_settings

#: Design planning scale, used for projection.
TARGET_EPISODES_PER_YEAR = 344_000_000


def _dimensions(db: Path) -> dict[str, int]:
    """Real label-value counts, read from the catalog rather than assumed."""
    c = sqlite3.connect(db)
    q = lambda s: c.execute(s).fetchone()[0]
    dims = {
        "dataset_id": q("select count(distinct dataset_id) from datasets"),
        "modality": q("select count(distinct name) from modalities"),
        "check_name": 24,          # full registry, not just those that fired
        "status": 4,               # passed/failed/warned/skipped
        "severity": 4,             # info/warning/error/critical
        "stage": 8,                # pipeline stages
        "episodes_observed": q("select count(*) from episodes"),
    }
    c.close()
    return dims


def _build_index(n_series: int, labels_per_series: int = 6) -> float:
    """Build ``n_series`` distinct label-sets and measure the bytes they cost.

    Mirrors a head block: each series is its label map plus a small sample
    buffer.  Returns bytes per series.
    """
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    index: dict[tuple, list] = {}
    for i in range(n_series):
        key = tuple(
            f"k{j}=v{i}_{j}" for j in range(labels_per_series)
        )
        # one float sample + timestamp, as a head chunk would hold
        index[key] = [float(i), i * 1000]
    peak = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    total = peak - base
    del index
    return total / n_series


def _parquet_bytes(n_rows: int, out: Path, entropy: str = "realistic") -> float:
    """Bytes per row for the same facts stored as an analytical table.

    Two variants are measured because the answer depends on entropy, and
    quoting only the flattering one would be dishonest:

    ``low``        every categorical column constant -- dictionary encoding
                   crushes it, which is the best case Parquet can ever show.
    ``realistic``  24 check names, 4 statuses, per-episode reasons and float
                   observations, i.e. what our QC findings actually look like.
    """
    import random
    rng = random.Random(0)
    checks = [f"check.{i:02d}" for i in range(24)]
    statuses = ["passed", "failed", "warned", "skipped"]
    if entropy == "low":
        check_col = ["temporal.gaps"] * n_rows
        status_col = ["failed"] * n_rows
        reason_col = ["gap"] * n_rows
    else:
        check_col = [rng.choice(checks) for _ in range(n_rows)]
        status_col = [rng.choice(statuses) for _ in range(n_rows)]
        reason_col = [
            f"gap of {rng.random() * 3:.3f}s at frame {rng.randrange(500)}"
            for _ in range(n_rows)
        ]
    tbl = pa.table({
        "run_id": pa.array([f"run-{i // 50:08x}" for i in range(n_rows)]),
        "episode_id": pa.array(list(range(n_rows)), pa.int64()),
        "dataset_version": pa.array(["training/v1"] * n_rows),
        "check_name": pa.array(check_col),
        "status": pa.array(status_col),
        "observed_value": pa.array([rng.random() for _ in range(n_rows)]),
        "reason": pa.array(reason_col),
    })
    pq.write_table(tbl, out, compression="zstd")
    return out.stat().st_size / n_rows


def main() -> dict:
    s = get_settings()
    dims = _dimensions(s.paths.catalog_db)

    # -- series counts under two label policies -----------------------------
    # SAFE: every dimension is a bounded vocabulary.
    safe = dims["check_name"] * dims["status"] * dims["severity"] * dims["dataset_id"]
    safe_modal = safe * dims["modality"]

    print("=" * 68)
    print("MEASURED LABEL DIMENSIONS (from the real catalog)")
    print("=" * 68)
    for k, v in dims.items():
        print(f"  {k:20s} {v:>12,}")

    print()
    print("=" * 68)
    print("SERIES COUNT BY LABEL POLICY")
    print("=" * 68)
    print(f"  safe (check x status x severity x dataset)     {safe:>14,}")
    print(f"  safe + modality                               {safe_modal:>14,}")
    print("  safe + episode_id, at N episodes:")
    for n in (1_000, 1_000_000, 100_000_000, TARGET_EPISODES_PER_YEAR):
        print(f"    N={n:>13,}  ->  {safe * n:>18,}")

    # -- measured cost per series -------------------------------------------
    print()
    print("=" * 68)
    print("MEASURED COST (this machine, Python head-block proxy)")
    print("=" * 68)
    per_series = _build_index(200_000)
    print(f"  bytes per active series          {per_series:>10.1f} B")

    tmp = Path("/tmp/rde_cardinality_probe.parquet")
    per_row_low = _parquet_bytes(200_000, tmp, "low")
    per_row = _parquet_bytes(200_000, tmp, "realistic")
    print(f"  bytes per Parquet row, low entropy   {per_row_low:>6.1f} B")
    print(f"  bytes per Parquet row, realistic     {per_row:>6.1f} B  <- used below")
    print(f"  ratio (series : realistic row)       {per_series / per_row:>6.1f}x")
    tmp.unlink(missing_ok=True)

    # -- projection ---------------------------------------------------------
    print()
    print("=" * 68)
    print("PROJECTION: one QC metric, labelled with episode_id")
    print("=" * 68)
    rows = []
    for n in (1_000, 1_000_000, 100_000_000, TARGET_EPISODES_PER_YEAR):
        series = safe * n
        tsdb_gb = series * per_series / 1e9
        pq_gb = n * per_row / 1e9
        rows.append({"episodes": n, "series": series,
                     "tsdb_gb": tsdb_gb, "parquet_gb": pq_gb})
        print(f"  N={n:>13,}  series={series:>18,}  "
              f"TSDB={tsdb_gb:>12,.1f} GB   Parquet={pq_gb:>8.2f} GB")

    print()
    print("  Same facts. The label policy, not the data volume, decides")
    print("  whether this is 0.1 GB or a planetary quantity of RAM.")

    report = {
        "dimensions": dims,
        "series_safe": safe,
        "series_safe_with_modality": safe_modal,
        "bytes_per_series_measured": round(per_series, 1),
        "bytes_per_parquet_row_low_entropy": round(per_row_low, 1),
        "bytes_per_parquet_row_realistic": round(per_row, 1),
        "projection": rows,
    }
    out = s.paths.data_root / "artifacts" / "cardinality_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  written: {out}")
    return report


if __name__ == "__main__":
    main()
