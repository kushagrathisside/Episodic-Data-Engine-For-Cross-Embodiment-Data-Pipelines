# 11 — Code Map

This document is an exhaustive walk-through of every module in `rde/`, presented in dependency order. For each module, it states what the module is for, what it exposes, and the non-obvious decision embedded within it.

The other documents in this set explain why the system is shaped as it is. This document explains where the code that implements that shape lives.

This document is a reference rather than a chapter. It is numbered 11 because it closes the sequence of engine documents, but its coverage extends to every module, including the observability layer introduced in [12-observability-recon.md](12-observability-recon.md) through [16-observability-report.md](16-observability-report.md). It is intended to be read beside the source, by lookup, rather than read in sequence, and every claim it makes points at a specific file.

The codebase comprises 47 modules and 14,979 lines, excluding package `__init__` files that only re-export names, and 188 tests (187 passing, 1 skipped).

## 1. How the Layers Stack

Imports point downward only: every import targets a module in the same layer or a lower one, never a higher one. This is what allows any single layer to be replaced without affecting the layers beneath it. Table 1 gives the stack from top to bottom, derived from the package-level import graph of `rde/`.

**Table 1. Module layers, top to bottom.**

| Layer | Modules | Imports from lower layers |
|---|---|---|
| Entry points | `cli.py`, `pipeline.py` | Every layer below |
| Model-facing edge | `feedback`, `evaluation`, `model_adapters` | `engine`, `catalog`, `storage`, `schemas`, `config` |
| Data access | `engine.py` | `dataset_builder`, `catalog`, `storage`, `schemas`, `config` |
| Data consumers | `dataset_builder`, `exploration` | `transformations`, `ingestion`, `versioning`, `catalog`, `storage`, `schemas`, `config` |
| Correctness | `qc`, `validation`, `synchronization` | `observability` (from `qc` only), `catalog`, `storage`, `schemas`, `config` |
| Derivation | `transformations` | `ingestion`, `versioning`, `catalog`, `storage`, `schemas`, `config` |
| Source knowledge | `ingestion` | `versioning`, `catalog`, `storage`, `schemas`, `config` |
| Provenance | `versioning` | `catalog`, `schemas`, `config` |
| Persistence | `catalog`, `storage` | `schemas` (from `catalog` only) |
| Vocabulary | `schemas`, `config` | None |

Within a layer, `feedback` imports `evaluation`, which imports `model_adapters`. The only mutual import is between `validation` and `synchronization`: the alignment checks subclass `validation.base`, and `validation.default_registry` includes them. No module below the model-facing edge imports any module in it, so the data layers carry no knowledge of any model.

The `observability` module sits sideways rather than on top of this stack. It is imported by `pipeline` and `qc`, while its own imports reach only `schemas.core`, `schemas.modality`, `schemas.qc`, `validation.base`, and `storage.video`. It imports no pipeline stage at all, a claim that can be verified directly with the command `grep -rn "from \.\.\(qc\|pipeline\|transformations\|evaluation\|ingestion\)" rde/observability/`, which returns no results. This import direction is what prevents a telemetry change from being able to alter pipeline behavior.

## 2. `config.py` (140 lines)

This module implements the split between code and data, and defines every configuration setting that carries a memory consequence. Its principal symbols are given in Table 2.

**Table 2. Symbols exported by `config.py`.**

| Symbol | Purpose |
|---|---|
| `SOFTWARE_VERSION` | Stamped into every manifest and lineage row |
| `Paths` | Six derived roots under one `data_root` (`source_cache`, `blobs`, `tables`, `artifacts`, `catalog_db`, `runs`), plus `version_dir()`/`artifact_dir()` |
| `Settings` | Engine-wide configuration, a frozen dataclass |
| `get_settings` / `set_settings` | Process-wide accessor; tests substitute an alternate configuration through this pair |

`Paths` derives every location from `RDE_DATA_ROOT`, which defaults to `~/VLAExperiments/rde_data`, deliberately outside the source tree, so that the code repository never contains dataset bytes. Of its accessors, `version_dir(dataset_id, version)` and `artifact_dir(...)` are the two that matter in practice: every stage writes through them rather than constructing paths independently.

The defaults of `Settings` are each chosen for an 8 GB machine, and each represents a memory decision rather than a performance preference, as detailed in Table 3.

**Table 3. `Settings` defaults and their memory rationale.**

| Setting | Default | Rationale |
|---|---|---|
| `row_group_size` | 2,048 | A single row group occupies a few megabytes, so a reader can stream groups rather than whole files. |
| `batch_size` | 1,024 | The number of frames held in memory during validation and transformation. |
| `episode_concurrency` | 1 | The dataset builder holds one episode open in memory at a time. |
| `video_cache_frames` | 64 | At approximately 0.9 MB per frame (480×640×3), the LRU cache is capped at roughly 58 MB. |
| `parquet_compression` | `zstd` | Achieves a better compression ratio than snappy at comparable decode speed. |

## 3. `schemas/` (672 lines, 4 modules)

This package defines the vocabulary of the system. Nothing downstream of ingestion is permitted to know that LeRobot exists; the types defined here are what downstream code speaks instead. Table 4 lists its four modules.

**Table 4. Modules within `schemas/`.**

| Module | Lines | Contents |
|---|---:|---|
| `modality.py` | 142 | `ModalityKind`, `StoragePlane`, `ModalitySpec`, `BlobRef` |
| `core.py` | 191 | `QualityStatus`, `EpisodeOutcome`, `Task`, `Frame`, `Episode`, `DatasetManifest` |
| `qc.py` | 168 | `Severity`, `CheckStatus`, `CheckScope`, `QCRecord`, `QCReport` |
| `lineage.py` | 171 | `VersionStage`, `TransformRun`, `LineageNode`, `LineageGraph` |

### 3.1 `modality.py` — The Extensibility Point

`ModalityKind` is implemented as an open registry rather than a closed union, spanning RGB, DEPTH, ROBOT_STATE, ROBOT_ACTION, HAND_STATE, TACTILE, FORCE_TORQUE, AUDIO, POINT_CLOUD, LANGUAGE, and OTHER. `StoragePlane` is binary: `INLINE`, for a value stored directly in the Parquet row, or `EXTERNAL`, for a value stored in the blob store and referenced by identifier.

This is the single design decision that made supporting the second proof dataset free of adapter changes. The ALOHA dataset is fourteen-degree-of-freedom bimanual, with four cameras and force/torque sensing at 50 fps, while the SO-100 dataset is six-degree-of-freedom with two cameras at 30 fps. Neither required a change to any adapter, because a modality's kind is expressed as data rather than as code.

### 3.2 `core.py` — The Logical Schema

`DatasetManifest` carries `dataset_id`, `version`, `fps`, `robot_type`, the `modalities` list, and the dataset's totals. It is the object every pipeline stage receives, and it is the object the catalog serializes into the `versions.manifest_json` column.

This module's docstring states the rule that keeps the logical and physical schemas from diverging: the Python objects define the logical schema, and the physical Parquet schema is derived from them in `storage/tabular.py`, so the two schemas cannot silently drift apart.

The constant `SCHEMA_VERSION` is persisted into every manifest, so that an older data store is recognized as such rather than misread.

### 3.3 `qc.py` — Three Orthogonal Concepts

This module defines the separation on which every downstream QC-related component depends.

1. **`Severity`** is a property of the check: how serious this class of finding is judged to be. Its levels are INFO, WARNING, ERROR, and CRITICAL.
2. **`CheckStatus`** is the outcome of one particular evaluation: PASSED, FAILED, WARNED, or SKIPPED.
3. **`QCGate`**, defined in `qc/engine.py`, is the policy that converts findings into an episode-level verdict.

A QC record is written only for failures and warnings. Passing checks are instead reflected in counters, since recording "OK" for 19,000 frames across 24 checks would cost more to store than the information is worth.

### 3.4 `lineage.py` — Provenance as a Queryable Graph

`TransformRun` represents one lineage edge: an operation, its input versions, its output version, its parameters, its `software_version`, its `code_fingerprint`, and, critically, the set of episode identifiers that survived it. The method `fingerprint()` computes a stable hash of operation, inputs, parameters, and code together, so that two runs sharing a fingerprint should produce byte-identical output.

`LineageGraph.root_episodes(version)` walks the chain of runs backward, intersecting the surviving episode sets at each step. This is what turns the question "which raw episodes contributed to `training/v1`?" into a graph traversal rather than a manual investigation.

## 4. `storage/` (631 lines, 3 modules)

This package implements the system's three storage planes as three separate modules, deliberately rather than as one database, as listed in Table 5.

**Table 5. Modules within `storage/`.**

| Module | Lines | Plane |
|---|---:|---|
| `blob.py` | 193 | Content-addressed binary storage |
| `tabular.py` | 168 | Parquet |
| `video.py` | 202 | Frame-accurate decode over blobs |

### 4.1 `blob.py` — Content Addressing

`BlobStore.put()` streams a file through SHA-256 in fixed-size chunks (`_HASH_CHUNK` at a time) rather than loading the file into memory, and stores it at a two-level fan-out path. Because the blob identifier is derived from its content, three properties follow directly: deduplication is automatic, since every version referencing the same video shares a single file; `verify()` constitutes a genuine integrity check rather than a size comparison; and the correctness argument behind the incremental-validation cache follows without additional work.

Ingestion hard-links blobs from the source cache rather than copying them, which is why `du` reports 2.0 GB of physical disk usage against 3.9 GB of apparent usage on the reference deployment.

One consequence of this design is worth stating explicitly: the blob store is shared across versions, so deleting a blob damages every version that references it. The fault-injection helper `inject_missing_blob` carries this warning in its own docstring.

### 4.2 `tabular.py` — The Streaming Contract

`ParquetTableStore` exposes `write_batches` and `iter_batches` as its primary interface, with `write_table` and `read_table` provided as conveniences for smaller cases. `iter_batches` accepts `columns=` and `filter=` arguments, which map onto column projection and row-group pruning respectively, so that a check reading only timestamps never causes state vectors to be decoded.

The function `fixed_list()` builds the fixed-size-list Arrow type used for every vector-valued modality, so that a six-degree-of-freedom dataset and a fourteen-degree-of-freedom dataset differ only in a parameter rather than in schema structure.

The method `stats()` returns physical facts about a table — row group count, compressed byte size, and column list — which the resource-usage documentation quotes directly rather than estimating.

### 4.3 `video.py` — Bounded-Memory Decode

`probe()` reads a video container's headers without decoding any frame and returns a `VideoInfo` object, whose `raw_rgb_bytes` property computes what a naive full decode of the container would cost. This property represents the cost that the entire tiered video-health strategy exists to avoid paying: 470 MB stored against 36.2 GB decoded, a factor of 77.

`VideoReader` seeks using PyAV and libdav1d, and holds a bounded LRU cache sized by `video_cache_frames`. Its method `read_at(timestamp)` performs frame-accurate access, while `iter_range()` yields frames rather than accumulating them, so that peak memory is bounded by one frame plus the cache regardless of clip length. Setting `cache_frames=0` disables the LRU cache for one-shot, scattered reads, a mode used by both the alignment check and the Tier 1 video sampler.

## 5. `catalog/catalog.py` (860 lines)

This module implements the metadata plane in SQLite, operated in WAL mode, comprising eleven tables, foreign-key cascades, and 25 public methods. Table 6 lists its tables, plus `qc_cache`, a twelfth table created directly against the same database connection by `QCCache.__init__` in `rde/qc/cache.py` rather than by `catalog.py`'s own schema.

**Table 6. Tables in `catalog.db`.**

| Table | Grain | Added by |
|---|---|---|
| `datasets` | One source dataset | Ingestion |
| `versions` | One `<stage>/vN` | Every stage that produces a version |
| `episodes` | One episode in one version | Ingestion, transformations |
| `tasks` | One task string | Ingestion |
| `modalities` | One channel in one version | Ingestion |
| `transform_runs` | One lineage edge | `VersionManager` |
| `episode_lineage` | One episode's parentage | `VersionManager` |
| `qc_records` | One finding | `QCEngine._persist` |
| `pipeline_runs` | One pipeline execution | `ObservabilitySession.finish` |
| `stage_runs` | One stage within a run | `ObservabilitySession.finish` |
| `qc_executions` | One validation at one time | `ObservabilitySession.finish` |
| `qc_cache` | One memoized validation | `QCCache` |

The final four tables listed are the observability layer's additions to the catalog. They are small and bounded by construction — one row per run, per stage, and per validation — because SQLite functions here as a catalog and must not be pushed toward also serving as a telemetry database. Episode- and finding-grained facts are instead stored in Parquet.

### 5.1 The Migration Mechanism

`_MIGRATIONS` is a tuple of `(table, column, decl)` entries. This mechanism exists because `CREATE TABLE IF NOT EXISTS` is a no-op against an existing table, so adding a column to `SCHEMA_SQL` alone would silently discard every value written to that column on a catalog that already exists. This defect occurred once in practice — lineage episode identifiers were silently lost — and `test_versioning_lineage.py` now constructs an old-schema database directly and asserts that the migration mechanism recovers it correctly.

The migrations currently registered add the columns `input_episodes`, `output_episodes`, `pipeline_run_id`, `qc_execution_id`, and `stage_runs.episodes`.

### 5.2 Query Surface

The catalog's query surface is organized into three groups. Provenance queries are served by `record_run`, `get_runs`, `lineage_graph`, and `trace_episodes_to_root`. QC queries are served by `record_qc`, `clear_qc`, `query_qc`, and `qc_summary`. History queries are served by `run_history`, `stage_history`, and `qc_execution_history` — the three methods that make a question such as "has the `validate` stage gotten slower?" answerable as a query rather than requiring a manual scan over JSON files.

The method `delete_version` exists only for disposable scratch versions. Real versions are immutable; superseding a version means creating the next one rather than modifying the existing one.

## 6. `ingestion/` (902 lines, 4 modules)

This is the only package in the engine with any knowledge of what LeRobot is. Its modules are listed in Table 7.

**Table 7. Modules within `ingestion/`.**

| Module | Lines | Role |
|---|---:|---|
| `base.py` | 102 | `SourceAdapter` abstract base class, `SourceProfile` |
| `download.py` | 113 | Fetches data from the Hugging Face Hub into the source cache |
| `lerobot.py` | 448 | The LeRobot v3.0 adapter |
| `ingest.py` | 185 | Converts adapter output into a canonical version |

### 6.1 `lerobot.py` — Table-Driven Mapping

The mapping from LeRobot source keys to canonical fields is expressed as a table of `(regex, template, ModalityKind)` triples, four of which are shown below.

```
(r"^observation\.images?\.(?P<rest>.+)$", "rgb.{rest}",   ModalityKind.RGB)
(r"^observation\.state$",                 "state.joint",  ModalityKind.ROBOT_STATE)
(r"^observation\.effort$",                "force.effort", ModalityKind.FORCE_TORQUE)
(r"^action$",                             "action.joint", ModalityKind.ROBOT_ACTION)
```

This table-driven design is the reason the ALOHA dataset required zero code changes to ingest: its four cameras and force/torque channel all matched rules that already existed.

This module also handles four quirks specific to the LeRobot v3.0 format, each of which would corrupt a dataset silently if left unhandled, described in Table 8.

**Table 8. LeRobot v3.0 quirks handled during ingestion.**

| # | Quirk | Handling |
|---:|---|---|
| 1 | Concatenated per-camera video | One file holds every episode, so an episode is represented as a window (`from_timestamp` / `to_timestamp`) rather than as a separate file. |
| 2 | Two clock domains | Frame timestamps are episode-relative while video positions are file-absolute; the mapping `episode.from_timestamp + frame.timestamp_s` is written into a stored `<modality>__ts_in_blob` column rather than re-derived at read time. |
| 3 | `__index_level_0__` | Task description strings arrive in a pandas index column bearing this name rather than in a named column. |
| 4 | Single-row-group Parquet | Source files must be re-chunked during ingestion, or every subsequent read pulls the entire file. |

The method `iter_frame_batches` yields Arrow `RecordBatch` objects. Its source code is what `code_fingerprint` hashes into the ingestion lineage row.

## 7. `validation/` (1,597 lines, 6 modules)

This package implements the system's 24 checks as a single streaming pass, organized into the modules listed in Table 9.

**Table 9. Modules within `validation/`.**

| Module | Lines | Checks contained |
|---|---:|---|
| `base.py` | 235 | `CheckContext`, `Check`, `EpisodeCheck`, `DatasetCheck`, `CheckRegistry` |
| `structural.py` | 326 | 6 — schema, blobs, frame counts, index continuity, video, references |
| `temporal.py` | 247 | 6 — monotonic, duplicates, interval, gaps, origin, duration |
| `numerical.py` | 482 | 6 — finite, range, discontinuity, stuck, profiler, outlier |
| `dataset_level.py` | 255 | 5 — length, task, modality, missing, frame-count |
| `__init__.py` | 52 | `default_registry()` |

### 7.1 The Framework

A naive design would let each check scan the frame table independently: with 24 checks, that amounts to 24 full scans, tolerable at 19,000 frames and ruinous at 50 million. `CheckRegistry` instead splits checks by scope, the engine reads the union of every check's declared columns exactly once, and each episode is fanned out to every applicable check.

`Check` declares `required_columns` as a class attribute, but this alone is insufficient, because some column names depend on the dataset's modalities — `rgb.top__ts_in_blob` exists only where a top camera is present. The method `dynamic_columns(ctx)` was added to handle this case, after a failing test revealed that the alignment check was validating intent, by re-deriving the mapping from metadata, while the stored data actually disagreed with that intent.

The `priority` attribute orders checks within a scope. Negative priorities run first and populate `ctx.shared`: `RangeProfiler` runs at priority −100, and the observability feature profiler runs at priority −50.

### 7.2 `numerical.py` — The Calibration Story

`ValueRangeCheck` originally fired on all 50 episodes of the ALOHA dataset. This outcome traces to arithmetic rather than a defect: percentile bounds are tautologically exceeded by approximately 1% of any dataset, and at 42 dimensions, this guarantees that every episode trips at least one bound. The fix was a materiality test, comprising `min_violation_ratio` (1% of an episode's samples) and `far_excursion` (25% beyond the bound), together with kind-aware bounds:

```python
IMPULSIVE_KINDS = (ModalityKind.FORCE_TORQUE, ModalityKind.TACTILE)
IMPULSIVE_PCT   = (0.02, 99.98)
```

Force and torque channels are impulsive by their physical nature, and contact events legitimately populate the far tail of their distributions. The measured result of this fix is a reduction from 64 findings across 50 episodes to 24 findings across 10 episodes on the ALOHA dataset, with the SO-100 result unchanged.

`RangeProfiler` streams the dataset once and publishes `value_bounds` and `step_thresholds` into `ctx.shared`, so that profiling a dataset larger than available RAM costs the same memory as profiling a small one.

## 8. `synchronization/align.py` (371 lines)

This module implements the cross-modal check, and its docstring explains why the check requires four independent pieces of evidence rather than one. `VideoAlignmentCheck` verifies alignment through the following four steps, ordered from cheapest to most expensive.

1. **Window arithmetic.** The declared window must span the duration that the frame count implies. This step catches offsets that accumulate across episodes.
2. **Stored-reference agreement.** The `<modality>__ts_in_blob` column, as actually written, must equal `episode.from_timestamp + frame.timestamp_s`. This step matters more than it might appear to: this column, rather than the episode metadata, is what the model adapter dereferences at read time.
3. **Decodability.** The first and last frames of an episode must decode successfully at their recorded positions.
4. **Distinctness.** The decoded first frame of an episode must differ from the frame that the previous episode's window maps to. A constant offset would otherwise pass all three preceding checks without complaint.

The drift measured in step 2 is published to `ctx.shared["alignment_offsets"]` for every episode, not only for failing ones, which is what allows `rde_data_timestamp_offset_seconds` to be treated as a distribution rather than merely an error log.

`AlignmentVerifier` collects the same evidence for display purposes, and this is what the exploration CLI uses to demonstrate that alignment holds, rather than merely asserting that it does.

## 9. `qc/` (868 lines, 2 modules)

Table 10 lists the two modules in this package.

**Table 10. Modules within `qc/`.**

| Module | Lines | Role |
|---|---:|---|
| `engine.py` | 587 | `QCGate`, `QCResult`, `QCEngine` |
| `cache.py` | 281 | Incremental validation |

### 9.1 `engine.py` — Three Phases, One Pass

The engine executes validation in three phases, summarized in Table 11.

**Table 11. `QCEngine` execution phases.**

| Phase | Condition | Action |
|---|---|---|
| 1 | Dataset checks with priority < 0 | Populate `ctx.shared` |
| 2 | — | Stream episodes, fanning each out to every episode check |
| 3 | Dataset checks with priority ≥ 0 | Compute cross-episode summaries |

The internal method `_iter_episode_frames` issues one filtered scan per episode. Because row groups are bounded, this touches only the groups an episode actually occupies, so peak memory is bounded by one episode's frames — approximately 400 rows — rather than by the size of the table as a whole.

`run_check` converts an exception raised by a check into a `CRITICAL` finding rather than aborting the run: the findings from other checks remain valuable, and a crashing check is itself a QC result worth recording. This method also accumulates `check_seconds`, the instrumentation that made the question "which check is slow?" answerable, and which immediately revealed that two video-related checks account for 96.8% of total validation time.

`QCGate.classify` converts one episode's findings into a `QualityStatus`. Its `block_on` parameter lists the severities that block an episode outright, and its `max_warnings` parameter quarantines an episode that accumulates many non-blocking findings — no single finding is fatal, but a demonstration that trips eight warnings warrants review by a person.

The internal method `_persist` writes findings to Parquet, which is authoritative, and to SQLite, which serves as a queryable mirror, and then stamps the resulting episode- and version-level quality status. It clears existing findings before writing new ones, so that a re-validation replaces rather than appends to the record — which is precisely why the `qc_executions` table exists, to preserve the history that this replacement would otherwise erase.

### 9.2 `cache.py` — Incremental Validation

Because versions are immutable and blobs are content-addressed, re-validating an unchanged version with an unchanged check suite must produce the same findings. The cache key used to detect this condition comprises three fingerprints, described in Table 12.

**Table 12. `QCCache` fingerprint components.**

| Fingerprint | Covers | Rationale |
|---|---|---|
| `suite_fingerprint` | Each check's name and its source code | When `value_range` was recalibrated, its name did not change; a name-keyed cache would have served the old bounds' findings, and the recalibration would have appeared to do nothing. |
| `table_fingerprint` | Size and modification time of every Parquet file | Hashing 470 MB of data to skip 2.7 s of computation costs more than the computation itself; this check is a safety net against the immutability invariant being violated — for instance, by a fault injector rewriting frames in place — rather than the primary correctness argument. |
| `blob_fingerprint` | Existence and size of referenced blobs | The fault injector `inject_missing_blob` deletes a video file without touching any Parquet byte; a table-only cache key would report the dataset healthy while its video was in fact missing. |

Measured directly, incremental validation reduces run time from 2.733 s cold to 0.078 s warm, a 35× speedup. This behavior is opt-in, because a stale QC result is a correctness failure in precisely the component whose entire purpose is correctness.

## 10. `transformations/` (986 lines, 3 modules)

Table 13 lists the modules within this package.

**Table 13. Modules within `transformations/`.**

| Module | Lines | Contents |
|---|---:|---|
| `base.py` | 112 | The `Transform` abstract base class and its three shapes |
| `ops.py` | 543 | Ten operators, plus `assign_splits` |
| `pipeline.py` | 331 | Composition, fitting, and lineage emission |

Transform operators are implemented in one of three shapes, distinguished by what each is permitted to change, as shown in Table 14.

**Table 14. Transform operator shapes.**

| Shape | Effect | Example operators |
|---|---|---|
| `FrameTransform` | Rewrites columns | `NormalizeVector`, `TimestampNormalizer`, `SelectFeatures`, `ResizeImages`, `NormalizeImages` |
| `FrameFilter` | Drops rows | `DropNonFiniteFrames`, `SubsampleFrames` |
| `EpisodeFilter` | Drops whole episodes | `QualityFilter`, `LengthFilter`, `TaskFilter` |

`NormalizeVector` accumulates mean and standard deviation in a streaming pass, so that fitting it costs constant memory regardless of dataset size, and its fitted statistics are recorded in the lineage row — a normalization whose parameters are not persisted cannot be considered reproducible.

`ResizeImages` is deferred: it records the target size in the manifest, and the resize is performed at read time within the model adapter. Materializing resized frames immediately would duplicate the entire binary storage plane to save work that the reader performs at read time regardless.

`TransformPipeline.run` streams data batch by batch, so that peak memory is bounded by one batch and the total cost is independent of dataset size. Its internal method `_output_schema()` pushes an empty `RecordBatch` through the column operators in order to derive a schema, because a filter that happens to match nothing would otherwise leave the writer without any schema at all.

## 11. `versioning/versions.py` (172 lines)

This module is small but load-bearing. The function `code_fingerprint(fn)` hashes a callable's source code directly, since a reproducibility claim is only meaningful when the code producing a result is pinned alongside its parameters; it falls back to a callable's qualified name for C extensions, whose source is not available to hash.

The function `next_version(catalog, dataset_id, stage)` returns the next unused version identifier of the form `<stage>/vN`.

`VersionManager.record(...)` constructs a `TransformRun`, minting a run identifier of the form `run_id = f"{operation}-{uuid4[:12]}"`, and writes both this run record and its associated episode-lineage rows. The field `run.metrics["duration_s"]` is set at this point as well.

## 12. `dataset_builder/builder.py` (295 lines)

This module converts a validated version into a training version. The function `assign_splits`, defined in `transformations/ops.py` at line 449 and invoked from here, splits the dataset by episode rather than by frame. Consecutive frames differ by only one control step, so a frame-level split would place near-duplicate frames on both sides of a split boundary and end up measuring memorization rather than generalization — among the most common ways a robotics benchmark silently misrepresents its own results.

Split assignment is derived from a SHA-256 hash of the episode identifier rather than from a shuffle, so that assignment is reproducible from the seed alone and independent of iteration order. The `group_by` parameter allows episodes sharing a key, such as a recording session or an operator, to be placed in the same split. Two modes trade off exactness against long-term stability, as described in Table 15.

**Table 15. Split assignment modes.**

| Mode | Behavior | Appropriate when |
|---|---|---|
| `quantile` (default) | Ranks episodes by hash and cuts at the target fractions, producing exact split sizes. | At proof-of-concept scale — with 48 episodes, independent assignment routinely produces a 42/3/3 split rather than the intended 38/5/5, and a three-episode validation set carries little statistical meaning. |
| `independent` | Assigns each episode independently by its own hash. | Once the dataset is large enough for the law of large numbers to apply; an episode's split assignment never changes as the dataset grows, so results remain comparable across versions. |

`build_training_contract` writes the manifest that the model layer reads: modality names, dtypes, shapes, normalization statistics, deferred transforms, split membership, and frame rate. This manifest constitutes the interface boundary of the system — a model that consumes this contract is portable across datasets, while one that reads Parquet directly is not.

The episode table is materialized small enough that a sampler can hold it entirely in memory even when the corresponding frame table cannot fit in memory.

## 13. `model_adapters/` (1,091 lines, 3 modules)

Table 16 lists the modules in this package.

**Table 16. Modules within `model_adapters/`.**

| Module | Lines | Role |
|---|---:|---|
| `base.py` | 154 | `ModelInput`, `Prediction`, `ModelInputAdapter`, `Model` |
| `adapters.py` | 133 | Three input shapes |
| `models.py` | 776 | Three models, plus `build_model` |

`Model.predict()` wraps its call in `perf_counter` and stamps the resulting `Prediction.latency_ms`, so that every model's latency is measured identically.

`Prediction` requires a confidence value on every instance. A policy that cannot express how sure it is of a given prediction cannot participate in a feedback loop, so this field is mandatory rather than optional.

The three adapters — `StateAdapter`, `VisionStateAdapter`, and `VisionLanguageAdapter` — together constitute the system's model-independence claim: the engine produces one canonical sample, and the adapter reshapes that sample for whatever a given model requires.

### 13.1 `models.py`

Table 17 summarizes the three model implementations in this module.

**Table 17. Models implemented in `models.py`.**

| Model | Role | Notable implementation detail |
|---|---|---|
| `MeanActionPolicy` | The floor. Predicts the training-set mean action. | — |
| `NearestNeighborPolicy` | Retrieval. | `_squared_distances` uses the Gram-matrix identity `‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b`, because the naive broadcast difference array peaked at 1.58 GB. |
| `SmolVLAPolicy` | A real vision-language-action model. | Three impedance mismatches are absorbed entirely within the adapter and none within the engine: camera slot naming (`observation.images.*` versus this engine's `image.rgb.*`), the LeRobot 0.4 processor pipeline (`make_pre_post_processors`), and a device pin. |

## 14. `evaluation/evaluator.py` (311 lines)

This module scores a model against held-out frames and produces `FrameOutcome` rows. The internal function `_safe_corr` applies a relative, rather than absolute, tolerance to its denominator. An earlier version used an absolute guard of `1e-9`, which allowed a float32 constant vector through undetected — `np.full(n, 0.1, dtype=float32).std()` evaluates to 7.45e-09 — after which `corrcoef` emitted `NaN` directly into the feedback loop's inputs. The same class of bug is avoided the same way in `drift.mean_shift`.

## 15. `feedback/flywheel.py` (339 lines)

This module is where evaluation output becomes data-selection decisions. `FailureMode` classifies each failure, and `FeedbackLoop` nominates episodes for retraining and marks them for human review within the catalog. The industry term for this practice is directed triage, and it constitutes the autonomous-vehicle sector's core data-selection discipline; this system implemented the mechanism before the term for it was known.

`build_retraining_selection` produces the episode list used for `training/vN+1`, which is the point at which the feedback loop closes.

## 16. `observability/` (3,285 lines, 10 modules)

This package implements the system's temporal dimension. Its governing rule, measured rather than merely asserted, is that metrics aggregate populations, events identify individuals, and traces connect them. Table 18 lists its modules.

**Table 18. Modules within `observability/`.**

| Module | Lines | Role |
|---|---:|---|
| `context.py` | 243 | The correlation spine |
| `metrics.py` | 369 | Instruments plus an enforced label policy |
| `instruments.py` | 161 | The 37 declared metrics |
| `events.py` | 326 | High-cardinality analytical rows |
| `exporters.py` | 242 | Textfile `.prom` export, NDJSON spans |
| `session.py` | 450 | Binds everything together; one `with` block per stage |
| `health.py` | 437 | Episode features and data-health emission |
| `drift.py` | 329 | Episode-grain comparison |
| `contracts.py` | 391 | Release policy |
| `video_health.py` | 295 | Tiers 0 through 2 |

### 16.1 `context.py` — The Join Key That Did Not Exist

Reconnaissance into the pre-observability codebase identified the blocking defect that motivated this module: `Pipeline.run()` minted no identifier of its own, and its eight stages each generated an independent `uuid4`, so that nothing joined them together.

`RunContext.start()` mints a `run_id` together with a 128-bit `trace_id` in OpenTelemetry's wire format. `Span` is deliberately coarse-grained: a span per frame would cost more to store than the frames themselves. The run context is held in a `ContextVar` rather than a module-level global, so that concurrent pipeline executions cannot overwrite one another's identity. The accessor `current()` returns `None` outside of an active run, and every consumer of this context is required to handle that case: observability that makes the library harder to use as a library is considered to have failed its purpose.

### 16.2 `metrics.py` — A Policy That Raises

This module defines three classes of metric label, described in Table 19.

**Table 19. Metric label classes.**

| Class | Count | Description |
|---|---:|---|
| `SAFE_LABELS` | 14 | A vocabulary bounded by code or an enum |
| `CONDITIONAL_LABELS` | 6 | Bounded per deployment, growing only by a human decision |
| `FORBIDDEN_LABELS` | 17 | Unbounded by construction |

`LabelPolicy.validate` raises an exception when a forbidden label is used. This rule is judged too consequential to leave to reviewer discipline alone: attaching `episode_id` to a single QC metric produces 264 billion time series and approximately 167 TB of resident memory at the system's planning scale, against 6.02 GB of Parquet storage for the identical underlying facts.

The function `normalize_error_class` collapses exception class names down to an allowlist plus a catch-all `other` category, because an unbounded exception hierarchy is itself an unbounded label, and a crash loop that emits novel exception types would otherwise compound a cardinality incident on top of an outage.

`Gauge.set` drops non-finite values rather than recording them. A `NaN` gauge would silently poison every aggregate that touches it, whereas a missing series is visible as such and a `NaN` series is not.

`Histogram` was chosen over a summary statistic because bucket counts aggregate correctly across multiple workers, whereas quantiles computed independently per worker do not.

### 16.3 `instruments.py` — Thirty-Seven Metrics, Three Families

The 37 declared metrics fall into three families: `rde_pipeline_*` metrics answer whether the system is behaving normally; `rde_qc_*` metrics answer whether validation is behaving normally; and `rde_data_*` and `rde_sensor_*` metrics answer whether the data itself is healthy.

The `rde_pipeline_last_*` gauges exist because counters are the wrong instrument for alerting on a batch job: a batch process exits, so its counters restart, and computing `increase()` over a time window is only correct if every run within that window was actually scraped — a condition a seven-second run will typically fail to satisfy. Alerts in this system are therefore built on the gauges rather than on counters.

### 16.4 `events.py` — Five Row Types

This module defines five row types: `PipelineRunRow`, `StageRunRow`, `QCExecutionRow`, `EpisodeHealthRow`, and `DriftReportRow`. Their Arrow types are inferred from dataclass annotations rather than from the data itself, so that an all-null column retains its declared type instead of collapsing to `null` and breaking the next append operation.

`EventBuffer` collects rows during a run and writes them once at the end of that run: a single run produces hundreds of rows rather than millions, and writing one file per table per run outperforms writing hundreds of small individual appends. Frame-grained data, which would justify a streaming write pattern, is never converted into an event row in the first place.

`EventBuffer.flush` writes each of its five tables inside its own `try`/`except`, rather than one write covering all five. A malformed row that fails to write one table — an `ArrowInvalid` from a type mismatch, for instance — is recorded under a `"_errors"` key in the returned dict rather than raised, so the remaining tables are still written. This matters specifically because the five tables are written in a fixed order: without per-table isolation, a failure on `stage_runs` would silently prevent every table after it, including `episode_health`, from ever reaching disk.

### 16.5 `exporters.py` — The Batch-Job Flush Problem

This module addresses a documented failure mode: the last batch of spans and metrics produced by a run can fail to be exported if the process exits before its telemetry providers have an opportunity to flush. A telemetry layer that silently loses the final stage of every run is worse than having no telemetry layer at all, because it appears to function correctly while actually failing silently.

To address this, every write in this module is synchronous, `flush_all` is registered with `atexit`, and `_atomic_write` writes to a temporary file in the same directory before calling `os.replace` — so that a scraper never observes a half-written exposition file.

The function `stale_after` treats a missing file as stale. Both "the run never happened" and "the run happened but has gone stale" are failure conditions that must trigger an alert, and this is precisely the distinction that a Pushgateway-style system cannot express, since it has no equivalent of an `up` metric.

### 16.6 `session.py` — One `with` Block

`ObservabilitySession.stage(name)` emits a span, a duration histogram sample, a failure counter on error, a peak-RSS gauge, volume counters, and a `stage_runs` row, so that instrumenting a stage costs less code than leaving it uninstrumented. `StageHandle` is the object through which a stage body reports its volume; fields left untouched remain `None` rather than defaulting to a misleading zero.

The function `disabled()` returns a no-op session. Every method on this session swallows its own exceptions: a dead exporter is permitted to degrade observability, but it must never cause an ingestion run to fail. This behavior is inverted, deliberately, from how the rest of the engine treats errors — a QC failure is a result worth recording, while a telemetry failure is not a result at all.

`finish()` gives each of its three exports — the Parquet event tables, the NDJSON spans, and the `.prom` metrics file — its own independent `try`/`except`, rather than one block covering all three, and increments `rde_observability_export_failures_total` on any failure. A failure in one export is therefore never able to suppress the other two, and is always visible in the metrics the same session exports, rather than disappearing without a trace.

### 16.7 `health.py` — Riding the Existing Pass

`EpisodeFeatureProfiler` is registered as a check that emits no findings. Its purpose is to reduce each episode to a fixed-length feature vector while the QC engine is already streaming that episode's frames for other checks. Registering it as a check, rather than adding a separate traversal of the data, is the entire point of this design: a second scan of 55,000 frames would have cost an estimated 30–50%, against a measured actual cost of 1.2%.

The function `trajectory_features` computes path length; the mean, standard deviation, and 95th percentile of speed; acceleration; jerk; dwell fraction; action magnitude and energy; and state–action divergence. Its derivatives use the true sample period rather than a per-index difference, so that a 50 fps dataset and a 30 fps dataset produce comparable numbers.

The function `emit_data_health` routes findings into counters through a lookup table, `_FINDING_COUNTERS`, so that a new check can be joined to the health metrics by adding a row to this table rather than by editing the emitter's logic. Missingness is seeded at 0.0 for every modality at the start of a run, since without this seeding, a clean dataset would emit no series for a given modality, making "no data" indistinguishable from "no problem."

### 16.8 `drift.py` — The Unit of Analysis

The ALOHA dataset records at 50 fps, so frame *t* and frame *t+1* of a fourteen-degree-of-freedom arm are nearly identical vectors. A two-sample statistical test run over 55,000 such frames reports p < 0.001 for differences that carry no physical meaning — the same underlying error that `value_range` originally made, namely a statistically valid procedure applied at the wrong unit of analysis.

For this reason, drift is instead computed over populations of episodes. The function `effective_sample_size` applies the standard lag-1 autocorrelation correction and is published beside every drift statistic the system reports, because a large nominal sample size computed from only six effective episodes is noise wearing the appearance of a p-value.

The function `psi` scales its bin count with sample size, and `calibrate_threshold` measures a statistic's null distribution by repeatedly splitting the baseline dataset in half. Table 20 gives the measured justification for this calibration.

**Table 20. Measured null-distribution behavior of PSI.**

| Sample size (episodes) | Median null PSI, 10 fixed bins | P(PSI > 0.25) |
|---:|---:|---:|
| 50 | 0.301 | 62.5% |
| 5,000 | 0.003 | 0.0% |

The calibrated result achieves a measured 0.5% false-positive rate and 94–98% detection power at a one-standard-deviation shift.

### 16.9 `contracts.py` — The Third Policy Layer

This module implements the third of three policy layers, distinguished in the example below.

```
check    → "episode 17 has a 2.1 s timestamp gap"      (an observation)
gate     → "episode 17 is rejected from clean/v1"      (per episode)
contract → "training/v3 is blocked: action energy
            drifted 3.5× its calibrated null"          (per version)
```

Contracts are organized into five clause families: schema, temporal, quality, distribution, and release.

`DataContract.infer` derives a contract from a version that has actually been inspected. A contract typed in from imagination would instead encode what someone assumed the data looked like, rather than what it was measured to look like.

Three details of this module carry particular weight: `SKIP` is not `PASS`, since a distribution clause with no available baseline has protected nothing at all; alignment tolerance is expressed in frame periods rather than seconds, so that a single contract remains meaningful at both 30 fps and 50 fps; and an underpowered drift comparison warns but never blocks a release.

### 16.10 `video_health.py` — Tiers

Table 21 lists the four tiers of video health checking implemented in this module.

**Table 21. Video health tiers.**

| Tier | Cost | Detects |
|---|---|---|
| 0 — headers only | 1.9 ms per file | Resolution change, codec swap, truncation, frame-count disagreement |
| 1 — sparse decode, 5 frames | 89 ms | Lens cap, dead sensor, frozen stream, blown exposure |
| 2 — perceptual hash per sample | Free (frames already in memory) | Duplicate and frozen content |
| 3 — embeddings / vision-language model | Not implemented | A documented boundary, not an omission |

Frames are sampled at fixed fractions of each clip's duration, so that a 12-second episode and a 90-second episode are compared at the same relative positions within their respective clips, and cost is determined by sample count rather than by clip length.

The function `cohort_summary` answers a question that no single-file check can answer on its own: not "is this file fine" but "do these files agree with one another." A camera swapped mid-collection leaves every individual file valid while leaving the collection as a whole broken.

## 17. `exploration/queries.py` (410 lines)

This module implements DuckDB queries directly over Parquet, exposing twelve query methods covering overview statistics, duration distribution, task summary, modality summary, dimensionality, timestamp frequency, QC summary, failed episodes, and lineage.

The method `sql()` serves as an escape hatch for arbitrary queries. Dotted modality names must be quoted within such queries — for example, as `"state.joint[1]"` — because DuckDB would otherwise parse `state.joint` as a `table.column` reference.

## 18. `engine.py` (387 lines)

This module implements the read-side façade of the system. `DataEngine.get_dataset(id, version)` returns a `DatasetView`, which exposes `contract`, `episode_ids`, `splits`, `iter_frames`, `iter_samples`, `get_episode_frames`, and `read_image`.

The method `read_image` is where deferred transforms are actually applied: a resize recorded in the manifest is performed here, once, at read time.

`DatasetView` is implemented as a context manager, so that video readers are closed deterministically rather than relying on garbage collection.

## 19. `pipeline.py` (352 lines)

This module implements the eight-stage pipeline: download, ingest, validate, transform, `qc_post_transform`, build, evaluate, and feedback.

Validation appears twice by design. The raw-data pass determines what is fit to transform, and the post-transform pass verifies that the transformation itself introduced no defects.

The internal method `_stage(name, fn)` wraps every stage in `self.obs.stage(name)` and extracts volume information from the stage's returned detail dictionary through a lookup table, `_VOLUME_KEYS`, so that a new stage can report its volume by naming its keys consistently rather than by editing the instrumentation code directly. The `StageResult` list returned by a run is kept unchanged in shape: the run record's structure is a published interface, and the observability additions are strictly additive to it.

The method `run()` mints the observability session, threads it through `obs.active()`, and writes a run record carrying both `run_id` and `trace_id`, so that the resulting record functions as a join key rather than as a loose artifact.

## 20. `cli.py` (930 lines)

This module implements the command-line interface using Typer, organized into six sub-applications: `source`, `qc`, `explore`, `version`, `model`, and `obs`.

The `obs` sub-application is the observability surface, exposing the commands `runs`, `stages`, `quality`, `metrics`, `health`, `drift`, `contract`, and `video`.

Two implementation details are worth noting. Rich markup is escaped using `rich.markup.escape`, since a bare string such as `[warning]` is itself valid markup and would otherwise be silently interpreted rather than printed. Additionally, command definitions must precede the `if __name__ == "__main__"` block, since `python -m rde.cli` executes that block at import time.

## 21. `testing/synthetic.py` (380 lines)

The function `make_lerobot_dataset` generates a genuine LeRobot v3.0 layout — including concatenated video, both clock domains, and the `__index_level_0__` column — so that tests exercise the actual source format rather than a simplified mock of it. Most interesting defects in an ingestion layer occur specifically in the handling of format quirks such as these.

This module provides ten fault injectors, listed in Table 22.

**Table 22. Fault injectors and the checks that detect them.**

| Injector | Detected by |
|---|---|
| `inject_nan` | `numerical.finite` |
| `inject_timestamp_gap` | `temporal.gaps` |
| `inject_non_monotonic` | `temporal.monotonic` |
| `inject_duplicate_timestamp` | `temporal.duplicates` |
| `inject_stuck_sensor` | `numerical.stuck_sensor` |
| `inject_discontinuity` | `numerical.discontinuity` |
| `inject_dropped_frames` | `structural.frame_index_continuity` |
| `inject_missing_blob` | `structural.blob_integrity` |
| `inject_video_misalignment` | `sync.video_alignment` |
| `inject_distribution_shift` | Nothing — and that is the point |

The tenth injector breaks nothing that any structural or temporal check can detect. Every value remains finite, timestamps remain monotonic, the schema is unchanged, and all 24 QC checks pass — yet the data has stopped being comparable to what the model was trained on. This is the fault that only distribution monitoring can see.

`inject_missing_blob` is destructive across versions, because the blob store deduplicates content; its docstring states this explicitly, a lesson learned from encountering the consequence directly.

## 22. Where to Start Reading

Table 23 suggests a starting point in the source depending on which aspect of the system a reader wishes to understand.

**Table 23. Reading guide by topic.**

| To understand… | Start with |
|---|---|
| How data becomes canonical | `ingestion/lerobot.py`, then `schemas/core.py` |
| Why storage is organized into three planes | `storage/blob.py`, `storage/tabular.py` |
| How 24 checks cost only one scan | `validation/base.py`, then `qc/engine.py` |
| Why a clean dataset can still be unusable | `synchronization/align.py` |
| What makes a version reproducible | `versioning/versions.py`, `schemas/lineage.py` |
| Why models are interchangeable | `model_adapters/base.py`, `dataset_builder/builder.py` |
| Why `episode_id` is never used as a label | `observability/metrics.py` |
| How drift avoids false alarms | `observability/drift.py` |
| What blocks a release | `observability/contracts.py` |
