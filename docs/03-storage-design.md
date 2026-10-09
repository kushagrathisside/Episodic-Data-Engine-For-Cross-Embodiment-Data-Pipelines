# 03 — Storage Design

## 1. Three Planes, Deliberately Not One Database

The system separates its persistent state into three storage planes, summarized in Table 1.

**Table 1. Storage planes.**

| Plane | Backend | Holds | Mutability |
|---|---|---|---|
| Binary | Content-addressed filesystem (`blobs/`) | Video, images, and (in future) depth and point clouds | Immutable |
| Tabular | Parquet (`tables/`) | State, action, timestamps, QC records | Immutable per version |
| Metadata | SQLite (`catalog.db`) | Datasets, versions, episodes, tasks, lineage, QC index | Mutable, transactional |

Combining these three planes into a single store fails in both directions. Placing mutable QC status inside immutable Parquet would require rewriting files merely to mark one episode as rejected. Placing 19,631 frame rows inside SQLite would discard columnar scanning and predicate pushdown. Each store is instead assigned to the workload it performs well.

DuckDB participates as a fourth component but is not a fourth store: it executes analytical queries directly over the Parquet files, so the `rde explore` command never loads a full frame table into Python.

## 2. Layout

The data root is organized as follows.

```
$RDE_DATA_ROOT/                    default ~/VLAExperiments/rde_data
├── source_cache/                  verbatim upstream downloads, never written to
│   └── lerobot__svla_so100_pickplace/
├── blobs/                         content-addressed binaries
│   └── 7c/7f/7c7f2674….mp4
│       7c/7f/7c7f2674….meta.json
├── tables/<dataset>/<stage>/<vN>/
│   ├── frames/part-000.parquet
│   ├── episodes/part-000.parquet
│   └── index/part-000.parquet     (training versions only)
├── artifacts/<dataset>/<stage>/<vN>/
│   ├── manifest.json
│   ├── qc/{records,summary.json}
│   ├── evaluation/<model>-<split>/
│   └── feedback/<model>/
├── runs/                          machine-readable pipeline run records
├── observability/
│   ├── metrics/rde.prom           Prometheus textfile export (atomic writes)
│   ├── spans/<run_id>.spans.ndjson
│   └── events/                    high-cardinality analytical rows
│       ├── pipeline_runs/<run_id>.parquet
│       ├── stage_runs/<run_id>.parquet
│       ├── qc_executions/<run_id>.parquet
│       ├── episode_health/<run_id>.parquet
│       └── drift_reports/<run_id>.parquet
└── catalog.db
```

The directory `observability/events/` is the system's analytical plane, and it is the same plane used elsewhere in the layout rather than a new one. The telemetry layer did not introduce a new storage technology: episode- and finding-grained facts are stored as Parquet and read through DuckDB, exactly as with every other tabular artifact in this system. This outcome was a finding of the design process rather than a compromise — the high-cardinality store that the cardinality analysis called for had already been built for other purposes and needed only to be routed into.

Each table produces one file per run, because Parquet files cannot be appended to and rewriting an existing file would violate the pipeline's streaming guarantee. Readers glob across these files, which is the standard access pattern for partitioned analytical stores. At fleet scale, the same files would move to ClickHouse over object storage; both engines read Parquet directly, so the row shapes remain unchanged and only the execution location of queries changes.

Data is never stored inside the source tree. This separation is not cosmetic: the binary plane already holds hundreds of megabytes and is designed to scale to terabytes. Locating it inside the code repository would make the repository unclonable and would invite committing large binary files such as a 320 MB video directly to version control. The separation also mirrors a realistic deployment topology, in which code resides in a container image and data resides in object storage, so that `RDE_DATA_ROOT` is the single setting that would be repointed at S3 in such a deployment.

## 3. The Binary Plane

Objects in the binary plane are addressed by the SHA-256 hash of their content. This design has three consequences.

1. **Deduplication.** The same video clip, referenced by `raw/v1`, `clean/v1`, and `training/v1` alike, is stored once.
2. **Immutability.** A blob identifier names one fixed byte sequence permanently, so a lineage record that cites a blob identifier constitutes a guarantee rather than a filename that might later be overwritten.
3. **Integrity verification.** Corruption is detectable by rehashing the stored content; the check `structural.blob_integrity` performs exactly this operation.

Ingestion admits blobs in a link mode: because the source cache already holds the underlying bytes immutably, the store creates a hard link to those bytes rather than copying them. Table 2 gives the measured effect of this choice.

**Table 2. Effect of hard-link deduplication on disk usage.**

| Measurement | Size |
|---|---|
| Data root, hard links deduplicated (actual disk usage) | 472 MB |
| Data root, every link counted independently (cost of copying) | 942 MB |
| Space saved | 470 MB |

The binary-plane API is shaped like an object-store API (`put`, `get`, `exists`, `uri`, with keys sharded as `aa/bb/<hash>`), so that an S3 backend can retain identical identifiers and only the implementation of these four operations needs to change.

## 4. The Tabular Plane

The tabular plane is governed by two rules.

**A full table is never read.** Every read passes through `iter_batches`, which projects the required columns and pushes predicates down into Parquet's own scan. The 19,631 frames in the primary dataset would fit comfortably in memory, but the 50 million frames this design targets at scale would not, and the code is written so that it does not need to change when that threshold is crossed.

**Row-group size is chosen by the engine, not inherited from upstream.** LeRobot ships all 19,631 rows of its frame table in a single row group, so any filter applied to the upstream file decodes the entire group regardless of selectivity. Ingestion rewrites the table with 2,048 rows per group. Table 3 shows the effect of this change.

**Table 3. Row-group structure before and after ingestion.**

| Table | Rows | Row groups | Size | Bytes per row |
|---|---:|---:|---:|---:|
| Upstream `data/…parquet` | 19,631 | 1 | 0.5 MB | — |
| `raw/v1` frames | 19,631 | 20 | 903 KB | 46.0 |
| `clean/v1` frames | 18,668 | 20 | 955 KB | 51.2 |

The `clean/v1` table is slightly larger per row than `raw/v1` because normalization replaces smoothly varying joint angles with zero-mean floating-point values that compress marginally less well. This is a real, if small, cost of materializing normalization.

## 5. What Each Version Materializes

The allocation of storage across versions is the central decision of this design, summarized in Table 4.

**Table 4. Materialization by storage type.**

| Storage type | Materialized per version | Shared across versions |
|---|:---:|:---:|
| Frame table | Yes (a few hundred KB) | |
| Episode table, manifest, QC artifacts | Yes | |
| Video blobs | | Yes, by content address |

A training version goes further still, referencing its source version's frame table rather than copying it. Frames belonging to `training/v1` are read through the `clean/v1` table via a manifest field:

```json
"frame_table_version": "clean/v1"
```

As a result, `training/v1` costs only 12.2 KB — an episode table plus a sampler index — on top of the data it shares with its source version. Promoting a version from raw to training therefore costs kilobytes rather than gigabytes. Had video been copied at every stage instead, this five-stage pipeline would have produced 2.3 GB of identical bytes on the primary dataset alone; at terabyte scale, the same policy would be prohibitive.

## 6. The Metadata Plane

The metadata plane is implemented in SQLite, operated in WAL mode so that exploration queries can read the catalog while a pipeline stage writes to it. Its tables are `datasets`, `versions`, `episodes`, `tasks`, `modalities`, `transform_runs`, `episode_lineage`, and `qc_records`. `episodes`, `tasks`, and `modalities` carry an enforced foreign key to `versions` with `ON DELETE CASCADE`; `transform_runs` and `qc_records` deliberately carry no such constraint, so that a finding or a lineage edge can outlive the version it describes in normal use. `Catalog.delete_version()` — reserved for disposable scratch versions such as the fault-injection demo — deletes `qc_records` and `transform_runs` explicitly within one transaction rather than relying on a constraint that is not there, and `episode_lineage` rows cascade away through their own FK on `transform_runs.run_id`.

The table `qc_records` functions as a queryable mirror; the authoritative, full-detail QC records reside in Parquet alongside the data they describe. Schema changes to the catalog are applied through an explicit migration step, since `CREATE TABLE IF NOT EXISTS` is a no-op against an existing table — a new column therefore requires an explicit `ALTER TABLE`, or an existing catalog will silently discard whatever is written to it. This scenario occurred as a real defect during development and is now covered by a regression test.

## 7. Scaling Path

Table 5 summarizes, for each dimension of the system, its current implementation and what would be required to scale it further.

**Table 5. Scaling path by dimension.**

| Dimension | Current implementation | Unchanged at scale | Requires further work |
|---|---|---|---|
| Frames | 19,600 | Streaming reads, already bounded | — |
| Blob store | Local filesystem | Same interface | S3 backend |
| Catalog | SQLite | Adequate to roughly 10⁶ episodes | PostgreSQL beyond that scale |
| Frame tables | Single file | Parquet dataset partitioning by episode | Parallel writers |
| Processing | Single process | Episode-parallel by construction | A scheduler |
| Metrics | `.prom` textfile | OTLP-shaped instrumentation | Remote write to VictoriaMetrics |
| Event rows | Parquet plus DuckDB | Same row shapes | ClickHouse over object storage |
| Spans | NDJSON files | OpenTelemetry wire-format identifiers already in use | A collector |

None of these transitions requires a change to the logical pipeline; each is a backend substitution behind an interface that already exists.

The final three rows of Table 5 describe the observability plane specifically, and this plane was built to be substitutable rather than to appear elaborate. Its instrumentation follows a counter/gauge/histogram model with an explicit label policy, and span identifiers are already expressed in OpenTelemetry's 128-bit and 64-bit hexadecimal formats — so introducing a collector changes only the transport of this data, not its shape. Standing up ClickHouse and a time-series database to demonstrate the point at proof-of-concept scale would have added infrastructure purely for appearance, a choice that [14-observability-architecture.md](14-observability-architecture.md), Section 7, argues against explicitly.
