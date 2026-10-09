# 14 — Observability Architecture and Backend Decision

This document combines deliverables 3, 5, and 6 of the RDE Observability and Data Health project: the architecture itself, the cardinality analysis supporting it, and the backend decision matrix. It builds on [12-observability-recon.md](12-observability-recon.md), which established what existed prior to this work, and [13-observability-industry-research.md](13-observability-industry-research.md), which established what the available evidence supports.

## 1. Reframing the Problem

Prior to this project, RDE was fundamentally a state system. Its catalog answered the question "what is this dataset, and how was it produced?" with genuine rigor, backed by immutable versions, content-addressed blobs, episode-level lineage, and persisted findings.

It could not, however, answer any question containing the word "when" — for instance, how the pipeline is behaving over time, whether the data population is changing, where quality is degrading, or how to trace a path from an aggregate anomaly back to the exact recording and transformation that caused it.

The observability layer is accordingly best understood as the missing temporal dimension of an existing system, rather than as a monitoring capability bolted onto it after the fact. Every design decision described in this document follows from that framing.

## 2. Three Signal Classes, and the Rule That Separates Them

Every fact this system might want to observe is first classified along a single axis: is the fact about a population, or about an individual? Facts about a population — for example, "the QC failure rate rose threefold" — become metrics, stored with a bounded label vocabulary in a time-series database. Facts about an individual — for example, "episode 193742 failed the `temporal.gaps` check" — become events, stored with unbounded identifiers in Parquet or an equivalent analytical store. A third category, traces, sits across both: it records causality and execution path, answering questions of the form "which stage, in which run, under which configuration."

The rule that governs this classification is stated precisely: a fact may become a metric label only if its value vocabulary is bounded by design, not merely by the current volume of data. The label `check_name` is bounded, because there are 24 checks, and there are 24 checks because 24 were written by hand. The label `episode_id` is unbounded, because it grows with every demonstration ever recorded. The relevant question is never "how many distinct values exist today," but rather "what causes a new value to appear."

This is the same distinction the engine already drew, prior to this project, between `TransformRun.fingerprint()` — a configuration, bounded by the number of distinct code-and-parameter combinations that exist — and `run_id`, an occurrence, unbounded by construction. The reconnaissance phase (Q8) found this split already present in the codebase; the work described here extends it rather than inventing it from scratch.

## 3. Cardinality Analysis, Measured

The figures in this section are the direct output of `python examples/08_cardinality_experiment.py`, run against the real catalog on the development machine, rather than estimates.

### 3.1 Label Dimensions Observed in the Catalog

Table 1 lists the label dimensions found in the catalog and what bounds each one.

**Table 1. Label dimensions and their bounds.**

| Dimension | Observed values | Bounded by |
|---|---:|---|
| `check_name` | 24 | The check registry — code |
| `status` | 4 | The `CheckStatus` enum |
| `severity` | 4 | The `Severity` enum |
| `stage` | 8 | The pipeline definition |
| `modality` | 9 | The modality registry — open, but curated |
| `dataset_id` | 2 | How many datasets exist — grows slowly, by human decision |
| `episode_id` | 324 (at time of writing) | Every recording ever made |

### 3.2 The Cost of Each Labeling Policy

The measured per-series cost of a head-block-style index is 631 bytes per series, using a Python proxy implementation. Prometheus's own documented floor for its `memSeries` struct is approximately 200 bytes before labels and chunks are added, so the true figure in a Go implementation sits between these two values; both are carried through the analysis below.

The measured cost of storing the identical facts as Parquet rows is 17.5 bytes per row at realistic entropy — 24 check names, 4 statuses, varying float observations, and per-episode reason strings — or 11.2 bytes per row if the categorical columns were held constant. The realistic figure of 17.5 bytes per row is used throughout the remainder of this analysis.

Table 2 projects memory cost under increasingly permissive labeling policies.

**Table 2. Projected memory cost by labeling policy.**

| Label policy | Series | RAM at 631 B/series | RAM at 200 B/series |
|---|---:|---:|---:|
| `check × status × severity × dataset` | 768 | 0.5 MB | 0.15 MB |
| … × `modality` | 6,912 | 4.4 MB | 1.4 MB |
| … × `episode_id`, 1,000 episodes | 768,000 | 0.48 GB | 0.15 GB |
| … × `episode_id`, 1,000,000 episodes | 768,000,000 | 485 GB | 154 GB |
| … × `episode_id`, 100,000,000 episodes | 76.8 billion | 48.5 TB | 15.4 TB |
| … × `episode_id`, 344,000,000 episodes/year (planning scale) | 264 billion | 167 TB | 52.8 TB |

The same 344 million facts, stored instead as analytical rows, occupy 6.02 GB of Parquet.

This comparison — 167 TB of RAM against 6 GB of disk, for identical underlying information — is not evidence that one system is categorically better than the other. It reflects the fact that one of the two systems was asked a question it is not shaped to answer. Prometheus pays for cardinality at write time and holds its index resident in memory; a columnar store pays at query time and holds nothing resident.

A further documented behavior makes the steady-state figure above an understatement of the practical risk: short-lived series linger in the head block for hours after they stop receiving samples. A batch pipeline that emits episode-labeled metrics produces almost entirely short-lived series, so the transient peak, not the long-run average, is the relevant hazard.

### 3.3 Classification of Every Candidate Dimension

Table 3 classifies every metric-label dimension considered for this system.

**Table 3. Classification of candidate metric-label dimensions.**

| Dimension | Verdict | Reasoning |
|---|---|---|
| `stage` | Safe | 8 values, fixed by the pipeline definition |
| `check_name` | Safe | 24 values, fixed by the check registry |
| `status`, `severity` | Safe | Bounded by their respective enums, 4 values each |
| `modality` | Safe | The registry is open but curated; growth requires a code change |
| `error_class` | Safe | Bounded by an explicit allowlist plus a catch-all `other` category |
| `dataset_id` | Conditionally safe | Grows by human decision rather than by recording; safe at tens of datasets, worth revisiting at thousands |
| `dataset_version` | Conditionally safe | Bounded per dataset but grows monotonically forever; safe only if old versions stop emitting, which they do in a batch pipeline |
| `software_version` | Conditionally safe | Bounded by release count, but multiplies cardinality when combined with other labels; kept on a dedicated `build_info` gauge rather than on every metric |
| `factory_id`, `robot_id` | Conditionally safe | The cohort dimension that makes a broken site detectable behind an otherwise healthy global average; safe at hundreds of values, worth the cost, but needs review at larger scale |
| `sensor_rig_id` | Conditionally safe | The same argument as above; rig type is preferred over rig instance wherever the question at hand allows it |
| `task_id` | Dangerous | Bounded per dataset but unbounded across a fleet's full task library; a task category should be used instead |
| `episode_id`, `frame_id` | Never | Produces 264 billion series, as shown in Table 2 |
| `run_id`, `trace_id`, `blob_id` | Never | Unbounded by construction; these are join keys, not label dimensions |
| Filename, full path, `code_fingerprint` | Never | Unbounded; `code_fingerprint` belongs on `build_info` and in event rows, not as a metric label |

The governing principle, stated once and applied throughout: metrics aggregate populations, events identify individuals, and traces connect them. Every routing decision in the remainder of this document is an application of that sentence.

## 4. Routing Table: Where Each Existing Fact Goes

Table 4 states, for each fact the engine already produces, where that fact is routed under this architecture.

**Table 4. Routing of existing facts to metric, event, trace, or log.**

| Existing fact | Metric | Event row | Trace | Log |
|---|:---:|:---:|:---:|:---:|
| Stage name / ok / duration | Histogram + counter | `stage_runs` | Span | Yes |
| Whole-run duration | Histogram | `pipeline_runs` | Root span | Yes |
| `TransformRun` parameters and fingerprints | — | `transform_runs` (already existed) | Span attributes | — |
| One QC finding | — | `qc_findings` (already existed) | Span event | — |
| QC findings, counted by check and status | Counter | — | — | — |
| QC pass counters | Counter | — | — | — |
| QC profile (`ctx.shared`) | Selected gauges | `episode_health` | — | — |
| Per-episode quality status | Counter, by status | `episode_health` | — | — |
| Model inference latency | Histogram | Evaluation rows | Span | — |
| Peak RSS / bytes read and written | Gauge + counter | `stage_runs` | Span attributes | — |
| Drift statistic per feature | Gauge (feature is bounded) | `drift_reports` | — | Yes |

Table 5 gives five worked examples showing this routing applied to concrete facts, in the terms used by the original project brief.

**Table 5. Worked routing examples.**

| Fact | Routed as |
|---|---|
| QC failures rose over the last hour | A metric |
| Episode 193742 failed `temporal.gaps` | An event row |
| Run 81, stage `validate`, took 14.2 s | A metric (histogram) and a trace span |
| Frame 4471 had a 2.1 s timestamp gap | An event row (already a `QCRecord`) |
| Many episodes from camera rig X now fail | A metric with a rig label, followed by an analytical query to identify the specific episodes |

The final row of Table 5 illustrates the design in miniature. The metric answers whether something broke and which cohort is affected. The event store answers which specific episodes are affected. The trace answers which specific execution produced the fault. No single store is asked to answer all three questions at once.

## 5. The Correlation Spine

The reconnaissance phase (Q9) identified the blocking defect this architecture had to resolve first: `Pipeline.run()` minted no identifier of its own, and its eight stages each generated an independent `uuid4`, so that nothing joined them together into a single, queryable execution.

Every part of this architecture depends on resolving that defect. `RunContext.start()` mints both identifiers exactly once, at the top of `Pipeline.run()`, and both remain stable for the duration of that run: a `run_id` (`run-` followed by 16 hex characters) and, independently, a 128-bit `trace_id` in OpenTelemetry's wire format — two separately generated random values, not one derived from the other. Each of the pipeline's stages then produces a span under that run's trace. Table 6 shows which downstream lineage or QC record each stage's span is linked to.

**Table 6. Stages, spans, and the records they link to via `pipeline_run_id`.**

| Stage | Span | Linked record |
|---|---|---|
| Download | `span: download` | — |
| Ingest | `span: ingest` | `TransformRun.run_id` |
| Validate | `span: validate` | `QCRecord` rows |
| Transform | `span: transform` | `TransformRun.run_id` |
| Re-validate (`qc_post`) | `span: qc_post` | `QCRecord` rows |
| Build | `span: build` | `TransformRun.run_id` |
| Evaluate | `span: evaluate` | — |
| Feedback | `span: feedback` | — |

All eight spans, together with the run itself, are recorded in the new catalog tables `pipeline_runs` and `stage_runs`, and every existing lineage and QC row gains a `pipeline_run_id` column linking it back to the run that produced it.

This design rests on three required properties. First, `pipeline_run_id` is minted once, at the top of `Pipeline.run()`, and threaded downward; it is never regenerated by a sub-component. Second, it is additive: existing `TransformRun.run_id` values retain their original meaning, and the new column is a foreign key pointing upward rather than a replacement for existing identifiers, so that lineage remains authoritative, consistent with the invariant established in reconnaissance Q12. Third, it is nullable: a component called directly as a library, such as `QCEngine.run` invoked outside a pipeline, must continue to work correctly with a null spine. Observability that makes the library harder to use as a library is considered to have failed its purpose.

## 6. Data Model

The data model described here extends the existing catalog rather than duplicating it. Table 7 lists the five new record types this project introduces; three of them gained a SQLite table in the catalog (the small, bounded run/stage/execution headers), while the two episode- and feature-grained tables are Parquet only, consistent with the invariant, stated in reconnaissance Q12, that SQLite must not be pushed toward also serving as a telemetry database. Every table that already existed gains only a single new column.

**Table 7. New record types in the observability data model.**

| Table | Grain | Catalog (SQLite) | Parquet event log |
|---|---|---|---|
| `pipeline_runs` | One pipeline execution | New | New |
| `stage_runs` | One stage within a run | New | New |
| `qc_executions` | One validation of one version at one point in time | New — resolves reconnaissance gap G3 | New |
| `episode_health` | One episode, in one version, at one QC execution | Not stored here | New |
| `drift_reports` | One feature, compared across two versions | Not stored here | New |
| `transform_runs` | One lineage edge | Existing, plus `pipeline_run_id` | — |
| `qc_records` | One finding | Existing, plus `pipeline_run_id` and `qc_execution_id` | — |

`pipeline_runs`, `stage_runs`, and `qc_executions` are each written to both stores: a mirrored row in the catalog (`pipeline_runs` keyed by `run_id`, and carrying `dataset_id`, `trace_id`, `config_hash`, `software_version`, `started_at`, `finished_at`, and `status`; `stage_runs` carrying `run_id` as a foreign key, along with `stage`, `seq`, `started_at`, `finished_at`, `status`, `error_class`, `records_in`, `records_out`, `bytes_read`, `bytes_written`, `peak_rss_bytes`, and `span_id`; `qc_executions` keyed by `qc_execution_id`, carrying `run_id` as a foreign key, along with `dataset_id`, `version`, `started_at`, `finished_at`, `checks_run`, `findings`, and `dataset_status`), plus the identical row in a Parquet event file. `pipeline_runs` is also the parent of the existing `transform_runs` table, now carrying an added `pipeline_run_id` column.

`episode_health` and `drift_reports` exist only as Parquet event files, written by `EventBuffer`/`EventWriter` in `rde/observability/events.py`; they carry `qc_execution_id` and `run_id`/`dataset_id`/`version` as plain columns rather than as SQL foreign keys, since there is no catalog table on the other end of that relationship. `episode_health` rows carry `qc_execution_id`, `episode_id`, `quality_status`, `qc_bitmask`, `modalities_present`, `modalities_expected`, `frame_count`, `duration_s`, `missing_ratio`, `temporal_quality`, `numeric_quality`, `alignment_quality`, and a `drift_features_json` column. The existing `qc_records` table gains added `qc_execution_id` and `pipeline_run_id` columns.

The table `qc_executions` is what unblocks quality trending. Prior to this project, `_persist` called `clear_qc` and then `record_qc` in sequence, so that re-validating a version erased its history outright, the gap identified as G3 in the reconnaissance. Scoping findings by execution rather than by version alone makes a question such as "has this version's QC outcome changed since the checks were last recalibrated?" answerable — precisely the question a recalibration such as the `value_range` fix described in [05-qc-architecture.md](05-qc-architecture.md) should have been able to answer, and previously could not.

The table `episode_health` functions as the drift feature store: one row per episode per QC execution, holding a fixed-width numeric summary. It is populated directly from the streaming pass the engine already performs, adding no additional traversal of the data.

## 7. Backend Decision Matrix

The candidate backends below are evaluated against the actual constraints of this system: a batch pipeline, bounded memory, running today on a single WSL machine with 8–9 GB of RAM, whose architecture must nonetheless reason correctly about a planning scale of 344 million episodes per year. Table 8 gives the comparison.

**Table 8. Backend comparison.**

| Criterion | Pushgateway | node_exporter textfile | OTLP push | VictoriaMetrics | ClickHouse | DuckDB + Parquet |
|---|---|---|---|---|---|---|
| Short-lived processes | No — no staleness concept | Yes — file persists | Yes — designed for it | Yes — via remote-write | Yes | Yes |
| "Did it run at all?" | No — no `up` metric | Yes — file mtime | Yes — explicit | Yes | Yes | Yes |
| Distributed jobs | No — overwrites | No — per-machine | Yes | Yes | Yes | Partial |
| High cardinality | No | No | No — transport only | Better than Prometheus | Yes | Yes |
| Historical analysis | No | No | Not applicable | Limited — TSDB retention | Yes | Yes |
| Alerting | Partial — via Prometheus | Yes — via Prometheus | Yes | Yes | Partial | No |
| Write amplification | High | None | Low | Low | Low | None |
| Operational complexity | One service | Zero | A collector | One service | A cluster | Zero |
| Petabyte-scale analytics | No | No | No | No | Yes | Limited — single node |
| Prometheus-compatible | Yes | Yes | Yes — via collector | Yes | Partial | No |
| Already present in this repository | No | No | No | No | No | Yes |

### 7.1 The Decision

Metrics are implemented as OTLP-shaped instrumentation, exported through the node_exporter textfile collector for this proof of concept. Three considerations, in order of weight, support this choice.

First, the Pushgateway is rejected on documented grounds rather than on taste. The absence of an `up` metric means that the most important batch-job question — did the nightly run happen at all? — is structurally unanswerable, and the absence of any staleness concept means a failed run's metrics would appear current indefinitely. Either property alone is disqualifying.

Second, the textfile collector provides correct batch semantics with zero additional infrastructure: the pipeline writes a `.prom` file atomically on completion, the file's modification time is itself the freshness signal, and a stale file is visibly stale by that signal alone. On a WSL machine with an 8 GB memory budget and a hard requirement not to crash the host, adding a collector and a time-series database merely to demonstrate a point about metric shape would be precisely the kind of infrastructure built for appearance that the project brief explicitly rules out.

Third, the instrumentation itself is written OTLP-shaped regardless of transport — counters, gauges, and histograms with an explicit label policy — so that the choice of exporter is a substitution rather than a rewrite. Moving to VictoriaMetrics via remote-write, the natural production step, changes exactly one module.

Events are implemented as Parquet, queried through DuckDB — a capability this repository already has. This is worth stating plainly as a finding in its own right: the analytical plane this architecture requires did not need to be built, because it already existed. `ParquetTableStore` already writes bounded row groups with predicate pushdown; `rde/exploration/queries.py` already runs DuckDB queries over them; and QC findings already land in Parquet at `artifacts/<ds>/<version>/qc/records.parquet`. The high-cardinality store the cardinality analysis in Section 3 calls for was not a system to procure, but a system to route existing facts into.

ClickHouse is the correct production answer at the 344-million-episode-per-year planning scale, and the migration path to it is genuine rather than aspirational: both ClickHouse and DuckDB read Parquet directly, so such a migration changes where queries execute, not how the underlying facts are shaped.

Traces are implemented as OTLP spans, exported to file for this proof of concept. The durable value of tracing in this system is not a flame graph for human inspection, but the `trace_id` and `span_id` columns present in the analytical tables themselves. These columns are what convert the path from metric to trace to run to episode to finding into a sequence of database joins, rather than a correlation performed by guesswork over timestamps. The columns are the deliverable; the specific trace backend behind them is interchangeable.

### 7.2 Final Routing Answer

Table 9 gives the final answer to the routing questions posed in the original project brief.

**Table 9. Final routing decisions.**

| Question | Answer |
|---|---|
| What should be pushed to a Pushgateway? | Nothing; metrics are written to a textfile and scraped instead |
| What should be scraped? | The `.prom` textfile; freshness is derived directly from its modification time |
| What goes through OTLP? | The shape of the instrumentation always; the wire protocol itself once a collector is introduced |
| What belongs in a time-series database? | Bounded-label aggregates only: rates, durations, counts, coverage ratios |
| What belongs in ClickHouse or a warehouse? | Everything grained at the episode, run, finding, or blob level |
| What belongs in Parquet or object storage? | The same facts, held durably, and today held authoritatively |
| What must never enter Prometheus? | `episode_id`, `frame_id`, `run_id`, `trace_id`, `blob_id`, file paths, fingerprints, and raw observed values |

## 8. Constraints This Architecture Accepts

The invariants stated in reconnaissance Q12 are restated here as binding design commitments. Instrumentation rides inside the existing streaming pass: no observability feature may add a second traversal of the data or materialize a dataset, and observability that costs more than the pipeline it observes has failed by definition. There are no per-frame metrics: frames are counted, never labeled. Lineage remains authoritative: `pipeline_run_id` is a foreign key pointing upward, never a competing provenance record. SQLite remains a catalog: run and stage rows are metadata about executions — small, bounded, and queryable — while findings at fleet scale belong in Parquet. Checking remains separate from gating, and is now additionally separate from alerting, which functions as a third policy layer over the same underlying findings. The modality registry remains open: health metrics are driven by `ModalityKind`, never by a hard-coded sensor list. Telemetry failure must never fail the pipeline: every exporter is best-effort, and a dead collector degrades observability without ever degrading ingestion.

## 9. Implementation Order

The observability layer was implemented in the following eight steps, in dependency order.

1. **Correlation spine.** `pipeline_run_id` and `trace_id`, threaded through every component, nullable outside a pipeline run.
2. **Catalog migration.** The five new tables, added through the existing `_MIGRATIONS` mechanism, which already existed for exactly this purpose.
3. **Metric primitives.** Counter, gauge, and histogram types, with an enforced label policy that raises an exception on a forbidden label rather than silently accepting it.
4. **Logging.** Introduced from scratch, since none existed prior to this project.
5. **Instrumentation.** Stage timings, byte counts, RSS, and per-check duration, added inside the existing streaming pass.
6. **Exporters.** A textfile `.prom` export, JSON spans, and Parquet event rows — all best-effort, and all flushed synchronously on process exit.
7. **Health and drift.** Population of `episode_health`, and episode-grain drift computation.
8. **Fault-injection demonstrations.** Scenario A (sensor failure) and Scenario B (distribution shift), reusing the nine fault injectors that already existed prior to this project.

Steps 1 through 4 were prerequisites for everything that followed: no comparison of backend technologies could be evaluated honestly on a system that could not yet join a stage to the run that executed it.
