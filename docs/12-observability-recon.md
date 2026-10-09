# 12 — Observability Reconnaissance

This document is the first deliverable of the RDE Observability and Data Health project. It is an inventory rather than a design: it records what telemetry the engine emitted prior to this project, where that telemetry landed, what was queryable, and which identifiers were available to carry correlation across records. No code was changed in the course of producing it.

The method used was direct inspection of `rde/` (9,765 lines of code across 51 modules at the time of this survey), the SQLite schema, the on-disk data root at `$RDE_DATA_ROOT`, and four real run records produced by the SO-100 and ALOHA pipelines. Wherever documentation and implementation appeared to disagree, the implementation was treated as ground truth.

## 1. Telemetry Inventory

Table 1 lists every signal the engine emitted about its own execution prior to this project.

**Table 1. Pre-existing telemetry inventory.**

| Signal | Emitted by | Shape | Persisted to | Queryable? |
|---|---|---|---|---|
| Stage name / ok / wall duration | `Pipeline._stage` (`rde/pipeline.py:119`) | `StageResult` | `runs/*.json` | No — file only |
| Stage detail payload | Each stage method | Free-form dict | `runs/*.json` | No — file only |
| Whole-run duration | `Pipeline.run` (`rde/pipeline.py:246`) | Float | `runs/*.json` | No — file only |
| Transform run duration | `TransformRun.duration_s` (`rde/schemas/lineage.py:80`) | Float | `transform_runs.metrics` (JSON text) | Partially — JSON embedded in a column |
| Transform run parameters and fingerprints | `VersionManager` (`rde/versioning/versions.py:105`) | JSON | `transform_runs` | Yes — by column |
| QC findings | `QCEngine.run` (`rde/qc/engine.py:150`) | `QCRecord` × N | `qc_records` and `qc/records.parquet` | Yes — indexed |
| QC pass counters | `QCReport.passed_counts` | Dict | `qc/summary.json` | No — file only |
| QC profile (`ctx.shared`) | `numerical.range_profile` | Nested dict | `qc/summary.json` | No — file only |
| Model inference latency | `rde/model_adapters/base.py:148` | `latency_ms` per call | Evaluation report JSON | No — file only |
| Peak RSS / disk footprint | `examples/03_resource_report.py` | One-shot report | stdout and JSON | No — manual, out-of-band |

Two omissions in this inventory are conspicuous. First, there was no logging: a search for `import logging` across `rde/` returned zero results, and all human-facing output consisted of fifteen `rich.console.print` call sites confined to `cli.py`, meaning nothing was emitted at all when the engine was driven as a library rather than through its CLI. Second, there were no counters, gauges, or histograms of any kind — no metric registry, no aggregation, and no cardinality discipline, because at this stage there was nothing yet to be disciplined about.

## 2. Twelve Guiding Questions

### Q1. What telemetry already exists?

This is answered in full in Section 1. Summarized in one sentence, the engine emitted eight wall-clock stage timings together with a rich set of data-quality facts, and nothing in between these two extremes.

### Q2. Where are stage timings recorded?

Stage timings existed only within `Pipeline._stage` (`rde/pipeline.py:119`), computed as deltas of `datetime.datetime.now()` — wall-clock time rather than `perf_counter`, meaning the recorded figures could be affected by clock adjustments and were not safe against monotonicity violations.

Timing granularity stopped at the eight top-level pipeline stages. Per-check time, per-episode time, per-blob ingest time, and video decode time were all invisible. Because `validate` constituted 2.7 s of a 6.5 s run on the SO-100 dataset — the single dominant stage, internally fanning 24 checks across 50 episodes — the component most worth measuring was precisely the one not measured at all.

### Q3. Where are run records persisted?

Run records were written to `$RDE_DATA_ROOT/runs/<dataset_id>-<YYYYMMDDTHHMMSSZ>.json`, at `rde/pipeline.py:274`. Four such records existed at the time of this survey, and none of them were reflected in the catalog.

The consequence is that a question such as "has the `validate` stage gotten slower over the last twenty runs?" was answerable only by globbing the run directory and parsing each JSON file by hand. No run table, no run index, and no mechanism existed to join a run to the versions it produced, short of re-reading the file directly.

### Q4. Where are QC findings persisted?

QC findings were, deliberately, persisted twice, in `QCEngine._persist` (`rde/qc/engine.py:305`): to Parquet, at `artifacts/<ds>/<stage>/<vN>/qc/records.parquet`, the detailed and authoritative copy; and to SQLite, in the table `qc_records`, a queryable mirror carrying three indexes, on `version_key`, on `(version_key, episode_id)`, and on `(version_key, check_name)`.

One caveat proved particularly consequential for observability design: `_persist` calls `clear_qc` before `record_qc` (`rde/catalog/catalog.py:572`), so that re-validating a version destroys its previous findings outright. The data model in place at the time of this survey expressed "QC describes a version," rather than "QC describes one evaluation of a version at a point in time," and as a direct consequence, a trend line across repeated validations of the same underlying data could not be drawn from what was stored.

### Q5. What is currently queryable?

Eight SQLite tables were queryable at the time of this survey: `datasets`, `versions`, `episodes`, `tasks`, `modalities`, `transform_runs`, `episode_lineage`, and `qc_records`. In addition, DuckDB was already available for frame-level analytics directly over Parquet, through `rde/exploration/queries.py`.

### Q6. What exists only as JSON or loose artifacts?

Pipeline run records, QC pass counters, the QC numeric profile, evaluation reports, feedback reports, and resource reports all existed only as JSON or loose artifact files. In addition, several columns within SQLite itself held JSON blobs inside TEXT columns rather than queryable structured data: `versions.manifest_json`; `episodes.modalities`; `episodes.attributes`; `transform_runs.parameters`, `.metrics`, `.input_episodes`, `.output_episodes`, and `.dropped_episodes`; and `qc_records.details`.

### Q7. What identifiers already exist?

The identifiers present at the time of this survey were `dataset_id`; `version` (of the form `<stage>/v<N>`); `version_key`; `episode_id`; `frame_id`; `task_id`; `blob_id` (the SHA-256 hash of content); `run_id` (of the form `<operation>-<uuid4[:12]>`); `code_fingerprint` (the first sixteen characters of the SHA-256 hash of an operator's source, from `rde/versioning/versions.py:37`); `TransformRun.fingerprint()` (a stable hash of operation, inputs, parameters, and code together); `SOFTWARE_VERSION`; and each modality's `name`.

### Q8. Which identifiers are stable across reruns?

Table 2 classifies the identifiers from Q7 by stability.

**Table 2. Identifier stability.**

| Stable (content- or definition-derived) | Unstable (occurrence-derived) |
|---|---|
| `dataset_id`, `version`, `version_key` | `run_id` — a fresh `uuid4` per execution |
| `episode_id`, `frame_id`, `task_id` | `started_at` / `finished_at` |
| `blob_id` — content-addressed; the most stable identity in the system | The run-record filename |
| `code_fingerprint`, `SOFTWARE_VERSION` | `qc_records.id` (autoincrement) |
| `TransformRun.fingerprint()` | — |

This split already anticipates the metrics/events distinction the eventual design would require. `fingerprint()` identifies a configuration, while `run_id` identifies an occurrence of that configuration being executed. The number of distinct configurations is bounded by the number of distinct code-and-parameter combinations that exist, while the number of occurrences grows without bound. Of the identifiers surveyed, only the former class is a candidate metric label, and even then only conditionally so.

### Q9. Which identifiers suit telemetry correlation?

At the time of this survey, none did — the necessary join key simply did not exist.

`Pipeline.run()` minted no identifier for itself. Each stage that produced a version minted its own independent `uuid4` as a `run_id`, and the run record stored each stage's detail payload without storing that stage's `run_id` — a fact confirmed directly against the record `so100_pickplace-20260902T042848Z.json`, in which no stage detail payload contains a `run_id`. As a result, the eight `TransformRun` rows created by a single pipeline execution carried no column linking them either to one another or to the run record describing the execution that produced them.

This was the single most consequential finding of the entire reconnaissance. Every subsequent design element — the trace model and the `PipelineRun`/`StageRun` schema in particular — depended on a correlation identifier that had to be introduced before any of the rest of the design could be built.

### Q10. What was already observability-friendly?

Seven existing properties of the engine reduced the scope of the work still required, described below.

1. **Streaming, bounded-memory execution.** Observability instrumentation could ride the single pass the engine already performed, rather than requiring a second pass of its own. The constraint that observability must not become the bottleneck is satisfied by construction, provided instrumentation is kept inside the pass that already exists.
2. **Content-addressed blobs.** `blob_id` already functioned as the incremental-validation key that an eventual caching layer would need. Unchanged content was already provably unchanged, without requiring a new fingerprinting mechanism.
3. **Immutable versions.** Drift baselines could be pinned to a specific version and would remain pinned, since a baseline could not silently drift out from under a comparison.
4. **The severity/status/gate separation** (`rde/schemas/qc.py`; `QCGate` at `rde/qc/engine.py:41`). This separation maps directly onto the distinction alerting requires between what was observed, how serious that class of observation is, and what policy dictates should happen as a result. The hard conceptual work here was already done.
5. **Episode-level lineage** (`episode_lineage`, indexed in both directions). The aggregate-to-run-to-episode-to-artifact drilldown that a dashboard would require was already structurally present; it lacked only a time dimension.
6. **`RangeProfiler`.** This component already computed per-modality percentile aggregates in a single streaming pass, with kind-aware bounds for impulsive modalities, and represented approximately 80% of the work a trajectory-drift feature extractor would otherwise require from scratch.
7. **Fault injection.** Nine injectors already existed in `rde/testing/synthetic.py` (lines 209–322), covering NaN injection, timestamp gaps, non-monotonic timestamps, duplicate timestamps, stuck sensors, discontinuities, dropped frames, missing blobs, and video misalignment. Any subsequent observability fault-testing effort could reuse these directly rather than building a new test harness.

In addition, `QCRecord` already carried the columns an analytical event table would need: `episode_id`, `frame_id`, `check_name`, `severity`, `status`, `observed_value`, `expected_value`, `reason`, and `checked_at`.

### Q11. What would have to change?

The following six changes were identified as necessary, ordered by dependency.

1. **Mint a pipeline-level run identifier and trace identifier** inside `Pipeline.run()`, and thread both through every `TransformRun` and every `QCRecord` produced during that run. No other part of the eventual design could function without this change.
2. **Persist run and stage records to the catalog**, rather than only to JSON files, so that runs become a queryable time series rather than a directory listing.
3. **Stop discarding QC history on re-validation**, by scoping findings to a specific QC execution rather than to a version alone.
4. **Introduce a logging layer**, since none existed.
5. **Time sub-stage units** — per check, per episode, per video decode — from within the pass the engine already performs.
6. **Record bytes read and written, and peak RSS, per stage**, since only an out-of-band example script measured these at the time of the survey.

### Q12. What should explicitly not change?

Six existing properties of the system were identified as invariants that any observability design must preserve. Streaming execution and bounded memory must be preserved; no observability feature may require materializing a full dataset. Blob and version immutability must be preserved. Lineage must remain authoritative: the observability layer may extend the provenance system, but it must never become a second, competing source of truth about what produced what. The separation of checking from gating must be preserved. The open modality registry must be preserved; health metrics must not hard-code a fixed sensor list. Model independence must be preserved. And SQLite must not be pushed toward also serving as a telemetry database — it is, and must remain, a catalog.

## 3. Structural Gaps, Consolidated

Table 3 consolidates the structural gaps identified by this reconnaissance and states what each gap blocks.

**Table 3. Structural gaps identified.**

| Gap | Description | Blocks |
|---|---|---|
| G1 | No pipeline-level run identifier or trace identifier; per-stage `run_id` values are unjoinable | The run/stage schema, distributed tracing, dashboard drilldown |
| G2 | Run records are files, not database rows, so no cross-run time series can be queried | Pipeline-level metrics, trend dashboards |
| G3 | QC history is destroyed on re-validation (`clear_qc`) | Quality trending, drift over time |
| G4 | No logging and no metric primitives exist | Every downstream observability capability |
| G5 | Timing granularity stops at eight stages; the dominant stage is unmeasured internally | Stage- and check-level latency analysis |
| G6 | Resource usage is measured only out-of-band, by a manually run example script | Memory, CPU, and I/O monitoring |
| G7 | No time dimension exists anywhere; the catalog answers "what is this dataset?" but never "how has it been behaving?" | Version comparison, drift detection |

Gap G7 is the thematic finding underlying all the others. Prior to this project, the engine was fundamentally a state system: it described datasets as they currently were. The temporal dimension — how the system and its data have been behaving over time — was the dimension it lacked entirely.

## 4. Scope Note for the Next Phase

This reconnaissance supports treating the coming work as building a data-health model, rather than as simply attaching an existing metrics tool such as Prometheus to the engine. The valuable and defensible work lies in the data-health model itself, and the four foundations that model needs — immutable versions, content addressing, persisted findings, and episode lineage — were already present at the time of this survey. What was missing was narrow and concrete: a correlation identifier, a time series of runs, and a metric/event split that respects cardinality.

Gaps G1 through G4 are prerequisites for everything that follows and were implemented before any backend technology was selected. No comparison of backend technologies could be evaluated honestly against a system that could not yet join a stage to the run that executed it.
