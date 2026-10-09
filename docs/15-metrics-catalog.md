# 15 — Metrics Catalog, Alerts, and Dashboards

This document combines deliverable 4 and part of deliverable 13 of the RDE Observability project. It catalogs all 37 metric instruments, organized into three families. For every instrument, this document states its labels and the resulting cardinality, and, where an instrument drives an alert, whether that alert's threshold is an engineering limit, a statistical baseline, or a starting heuristic. This last distinction is a central purpose of this document: a threshold presented without its provenance is a threshold nobody can meaningfully argue with, and therefore one that nobody should trust.

The catalog can be verified directly against the running code with the command `rde obs metrics --show`.

## 1. Label Vocabulary

The label vocabulary is enforced by `rde/observability/metrics.py`, which raises an exception on a violation rather than merely documenting the rule. Table 1 gives the three classes of label defined by this policy.

**Table 1. Metric label classes.**

| Class | Members | Rule |
|---|---|---|
| Safe | `stage`, `check_name`, `status`, `severity`, `scope`, `modality`, `modality_kind`, `outcome`, `error_class`, `quality_status`, `operation`, `split`, `feature`, `result` | Vocabulary bounded by code or an enum |
| Conditional | `dataset_id`, `dataset_version`, `software_version`, `factory_id`, `robot_id`, `sensor_rig_type` | Bounded per deployment, growing only by human decision; used sparingly and never all at once on a single metric |
| Forbidden | `episode_id`, `frame_id`, `run_id`, `trace_id`, `span_id`, `blob_id`, `task_id`, `code_fingerprint`, `sensor_id`, `sensor_rig_id`, file paths, filenames, `uuid`, timestamps, observed values, reasons | Unbounded by construction; these values are join keys, and belong in event rows rather than metric labels |

The label `error_class` is additionally collapsed to an allowlist plus a catch-all `other` value. An unbounded exception hierarchy is itself an unbounded label, and a crash loop that emits novel exception types would otherwise compound a cardinality incident on top of an outage.

## 2. Pipeline Family — Is the System Behaving Normally?

Table 2 lists the pipeline family's throughput and latency instruments.

**Table 2. Pipeline family: throughput and latency.**

| Metric | Type | Labels | Series | Purpose |
|---|---|---|---|---|
| `rde_pipeline_runs_total` | Counter | `dataset_id`, `outcome` | 2·D | Run volume and success ratio |
| `rde_pipeline_stage_failures_total` | Counter | `stage`, `error_class`, `dataset_id` | 8·13·D | Which stage breaks, and how |
| `rde_pipeline_stage_duration_seconds` | Histogram | `stage`, `dataset_id` | 8·D | Latency distribution per stage |
| `rde_pipeline_run_duration_seconds` | Histogram | `dataset_id`, `stage="all"` | D | End-to-end latency |
| `rde_pipeline_frames_processed_total` | Counter | `stage`, `dataset_id` | 8·D | Throughput |
| `rde_pipeline_episodes_processed_total` | Counter | `stage`, `dataset_id` | 8·D | Throughput |
| `rde_pipeline_bytes_read_total` / `_written_total` | Counter | `stage`, `dataset_id` | 16·D | I/O volume |
| `rde_pipeline_stage_peak_rss_bytes` | Gauge | `stage`, `dataset_id` | 8·D | Which stage raises the memory ceiling |

(Here and throughout this catalog, D denotes the number of distinct datasets under management.)

### 2.1 The Batch-Job Gauges

Table 3 lists a second set of pipeline-family instruments, implemented as gauges specifically because they serve batch-job alerting.

**Table 3. Pipeline family: batch-job gauges.**

| Metric | Type | Labels | Purpose |
|---|---|---|---|
| `rde_pipeline_last_success_timestamp_seconds` | Gauge | `dataset_id` | The freshness signal |
| `rde_pipeline_last_run_timestamp_seconds` | Gauge | `dataset_id` | Whether a run happened at all, success or not |
| `rde_pipeline_last_run_status` | Gauge | `dataset_id` | 1 or 0 |
| `rde_pipeline_last_run_duration_seconds` | Gauge | `stage`, `dataset_id` | The most recent run's per-stage time |
| `rde_pipeline_last_run_frames` / `_episodes` | Gauge | `stage`, `dataset_id` | The volume pillar |

These gauges exist because counters are the wrong instrument for batch-job alerting. A batch process exits, so its counters restart at zero on the next run; Prometheus can handle this as a counter reset, but computing `increase()` over a time window is only correct if every run within that window was actually scraped, and a run that finishes in seven seconds will be missed by a fifteen-second scrape interval more often than not. The counters described in Table 2 remain correct within a single run and form the basis of the event-row aggregates, while the alerts described later in this document are built on the gauges in Table 3 instead.

This distinction is also the concrete reason the Pushgateway was rejected as a backend, as discussed in [14-observability-architecture.md](14-observability-architecture.md). Its documented lack of an `up` metric makes the question "did the nightly run happen?" structurally unanswerable, whereas here the same question reduces to a single expression, because a file has a modification time.

## 3. QC Family — Is Validation Behaving Normally?

Table 4 lists the QC family of instruments.

**Table 4. QC family instruments.**

| Metric | Type | Labels | Series | Purpose |
|---|---|---|---|---|
| `rde_qc_findings_total` | Counter | `check_name`, `severity`, `status`, `scope`, `dataset_id` | ≤ 24·4·4·3·D | What is firing |
| `rde_qc_episode_status_total` | Counter | `quality_status`, `dataset_id` | 5·D | Gate outcome distribution |
| `rde_qc_check_duration_seconds` | Histogram | `check_name`, `dataset_id` | 24·D | Which check is slow |
| `rde_qc_checks_run_total` | Counter | `check_name`, `dataset_id` | 24·D | Suite coverage |
| `rde_qc_gate_rejection_ratio` | Gauge | `dataset_id` | D | Headline quality number |

### 3.1 What This Family Revealed on Its First Run

The instrument `rde_qc_check_duration_seconds` was added because reconnaissance gap G5 showed that timing granularity previously stopped at the eight pipeline stages, leaving the dominant stage internally invisible. Table 5 gives the measured breakdown on the SO-100 dataset, across both QC passes of one run.

**Table 5. Per-check timing, revealed by the new instrumentation.**

| Check | Seconds | Share |
|---|---:|---:|
| `sync.video_alignment` | 2.479 | 48.7% |
| `structural.video_integrity` | 2.449 | 48.1% |
| `observability.episode_features` | 0.059 | 1.2% |
| All 22 other validation checks, combined | 0.106 | 2.1% |
| **Total** | **5.093** | |

Two video-touching checks together account for 96.8% of all validation time, and they run adjacently within the same episode pass — a strong indication that they decode overlapping regions of video and could plausibly share a single decoder. This is a concrete optimization opportunity the engine could not identify before this instrumentation existed, and it was surfaced on the very first run of it.

The third row of Table 5 is the observability layer's own cost, measured at 1.2% of the stage it observes. The constraint that observability must not become a bottleneck is not merely asserted in this design; it is measured directly.

### 3.2 The Cost of Observing

The governing constraint on this entire layer is that observability must not become more expensive than the pipeline it observes. Table 6 gives the measured end-to-end cost on the SO-100 dataset, three runs each on the same machine, with every telemetry component enabled: feature profiling, event rows, Parquet writes, spans, and the metrics export.

**Table 6. End-to-end observability overhead.**

| Configuration | Median wall-clock time |
|---|---:|
| `Pipeline(cfg, observability=False)` | 6.011 s |
| `Pipeline(cfg, observability=True)` | 6.061 s |
| Overhead | +0.050 s (+0.8%) |

This measurement can be reproduced with the following script:

```python
import statistics, time
from rde.pipeline import Pipeline, PipelineConfig

cfg = PipelineConfig.from_file("configs/so100_pickplace.json")
for obs in (False, True):
    t = [ (lambda t0=time.perf_counter(): (Pipeline(cfg, observability=obs).run(),
           time.perf_counter() - t0)[1])() for _ in range(3) ]
    print(obs, round(statistics.median(t), 3))
```

An overhead of 0.8% is affordable specifically because of where the instrumentation is placed: the feature profiler is implemented as a check that rides the episode pass already streaming every frame, and event rows are buffered in memory and written once per run rather than appended per episode. An implementation that instead added a second traversal of the data to compute drift features would have cost an estimated 30–50%, and would have violated the bounded-memory guarantee on which the engine is built.

## 4. Data-Health Family — Is the Data Usable?

This family covers a question general-purpose observability tooling does not address: a pipeline can succeed perfectly by every operational measure and still emit a ruined dataset. Table 7 lists its core instruments.

**Table 7. Data-health family instruments.**

| Metric | Type | Labels | Purpose |
|---|---|---|---|
| `rde_data_modality_coverage_ratio` | Gauge | `modality`, `dataset_id` | Is a camera silently absent? |
| `rde_data_missing_ratio` | Gauge | `modality`, `dataset_id` | Completeness |
| `rde_data_sampling_interval_seconds` | Gauge | `modality`, `dataset_id` | Declared rate versus observed rate |
| `rde_data_sampling_jitter_seconds` | Histogram | `modality`, `dataset_id` | Clock stability |
| `rde_data_timestamp_offset_seconds` | Histogram | `modality`, `dataset_id` | Cross-modal alignment |
| `rde_data_decode_failures_total` | Counter | `modality`, `dataset_id` | Video health |
| `rde_data_nonfinite_values_total` | Counter | `modality`, `dataset_id` | NaN/Inf rate |
| `rde_data_episode_duration_seconds` | Histogram | `dataset_id` | Population shape |
| `rde_data_episode_frame_count` | Histogram | `dataset_id` | Population shape |

Modality coverage is computed from what episodes actually declare, rather than from the manifest alone. A modality present in the manifest but absent from every episode is exactly the silent failure this metric is designed to catch.

`rde_data_timestamp_offset_seconds` uses histogram buckets at 1, 5, 10, 33, and 67 ms, because at 30 fps a single frame period is 33 ms; the sub-frame buckets are what distinguish a tolerable offset from a genuine desynchronization. This instrument implements the sixth pillar argued for in [13-observability-industry-research.md](13-observability-industry-research.md), Section 7: the canonical five pillars — freshness, volume, schema, distribution, and lineage — all pass on a dataset whose video has been shifted five seconds against its recorded state.

This metric is recorded for every episode the alignment check evaluates, not only for episodes that fail, because a metric that appears only when something is broken cannot demonstrate that nothing currently is. Table 8 shows the measured effect of a synthetic misalignment fault on the SO-100 dataset, across 50 episodes and 2 cameras.

**Table 8. Measured alignment offset under fault injection.**

| Condition | Mean offset |
|---|---:|
| Clean | 0.0000 ms |
| After `inject_video_misalignment(shift_s=5.0)` | 5000.0000 ms |

This metric behaves as a continuous signal rather than a binary verdict, which is what allows an offset drifting from 2 ms to 20 ms to be observed while it remains below the 33 ms threshold that would constitute an outright failure. This property is asserted directly in `tests/test_observability.py::test_alignment_offset_is_a_continuous_signal_not_a_binary_check`.

### 4.1 Sensor Health, Inferred From the Data

The instruments in Table 9 are framed explicitly as data-derived health indicators, never as hardware diagnosis — a dataset engine has no access to hardware telemetry and should not present itself as though it does.

**Table 9. Sensor-health instruments.**

| Metric | Type | Labels | Indicates |
|---|---|---|---|
| `rde_sensor_frame_dropout_total` | Counter | `modality`, `dataset_id` | A camera or link dropping frames |
| `rde_sensor_stuck_intervals_total` | Counter | `modality`, `dataset_id` | A frozen channel |
| `rde_sensor_saturation_events_total` | Counter | `modality`, `dataset_id` | Rail hits |
| `rde_sensor_tail_ratio` | Gauge | `modality`, `dataset_id` | Maximum value divided by the 99.5th percentile |

`rde_sensor_tail_ratio` merits particular explanation. On the ALOHA dataset, a force channel's maximum value was measured at 2.4 times its 99.5th percentile, and this is normal behavior: force and torque signals are impulsive by their physical nature, so contact events legitimately populate the far tail of their distributions. Alerting directly on the level of this ratio would page an operator on every successful grasp. The meaningful signal is a shift in the ratio over time, not the ratio's absolute value — the same lesson that forced the `value_range` recalibration described in [05-qc-architecture.md](05-qc-architecture.md), where percentile bounds originally fired on all 50 ALOHA episodes because approximately 1% of any dataset is tautologically outside a 99th-percentile bound, and at 42 dimensions this guarantees that every episode trips at least one.

### 4.2 Drift

Table 10 lists the two drift-related instruments in this family.

**Table 10. Drift instruments.**

| Metric | Type | Labels | Purpose |
|---|---|---|---|
| `rde_data_distribution_drift` | Gauge | `feature`, `dataset_id` | The drift statistic for one episode-level feature |
| `rde_data_drift_effective_sample_size` | Gauge | `feature`, `dataset_id` | Published beside the drift statistic at all times |

The label `feature` is a safe label under the policy in Table 1, since it is bounded by the feature extractor, which is code.

## 5. Self-Observation

Table 11 lists two instruments the observability layer uses to observe itself.

**Table 11. Self-observation instruments.**

| Metric | Type | Purpose |
|---|---|---|
| `rde_observability_series_count` | Gauge | Total active metric series |
| `rde_observability_export_failures_total` | Counter | Count of failed telemetry exports |

A metric layer that cannot report its own cost is precisely the kind of layer that produces surprises. `rde_observability_series_count` is what allows a cardinality regression to be caught during code review rather than in production. Measured on a real SO-100 run, the system produces 667 active series across 37 metrics — a figure that is bounded, and small enough to inspect directly.

## 6. Alert Design

### 6.1 Threshold Provenance

Every threshold used in alerting belongs to one of four classes, described in Table 12. Stating which class a given threshold belongs to is not a matter of pedantry: it determines who is permitted to change the threshold, and on what evidence.

**Table 12. Threshold provenance classes.**

| Class | Meaning | Who may change it |
|---|---|---|
| Engineering limit | Derived from physics or a hardware specification | Nobody, absent new hardware |
| Statistical baseline | Measured directly from this system's own null distribution | Recomputed automatically |
| SLO | A negotiated promise between teams | The team that made the promise |
| Starting heuristic | A defensible guess, pending calibration | Anyone, given supporting data |

### 6.2 Proposed Alerts

Table 13 lists the alerts proposed for this system, each labeled with its threshold class.

**Table 13. Proposed alerts.**

| Alert | Expression (sketch) | Threshold class |
|---|---|---|
| Pipeline stopped | `time() - rde_pipeline_last_success_timestamp_seconds > 86400` | SLO — how stale may training data become? |
| Run failed | `rde_pipeline_last_run_status == 0` | Engineering — a failure is a failure |
| Volume collapse | `rde_pipeline_last_run_frames < 0.5 × 7-day median` | Statistical baseline |
| Stage regression | `rde_pipeline_last_run_duration_seconds > 3 × 7-day median` | Statistical baseline |
| Rejection spike | `rde_qc_gate_rejection_ratio > baseline + 3σ` | Statistical baseline |
| Modality vanished | `rde_data_modality_coverage_ratio < contract_min` | Contract |
| Sampling rate wrong | `abs(observed − declared) / declared > 0.02` | Starting heuristic |
| Alignment offset | `histogram_quantile(0.99, timestamp_offset) > 1 frame period` | Engineering — derived from fps |
| Drift outside contract | `rde_data_distribution_drift > calibrated_threshold` | Statistical baseline |
| Telemetry stale | Export file modification time older than twice the expected interval | SLO |
| Cardinality regression | `rde_observability_series_count > 5000` | Starting heuristic |

Only two of the eleven proposed alerts in Table 13 rest on a starting heuristic, and both are labeled as such explicitly. Wherever a threshold could instead be measured, it was measured, as described in Section 7.

### 6.3 Avoiding Five Pages for One Failure

A single failing camera propagates through the system as a chain of five distinct symptoms, each of which corresponds to its own metric.

1. A camera firmware fault causes frame dropout, reflected in `rde_sensor_frame_dropout_total`.
2. Frame dropout produces decode anomalies, reflected in `rde_data_decode_failures_total`.
3. Decode anomalies produce alignment errors, reflected in `rde_data_timestamp_offset_seconds`.
4. Alignment errors produce QC failures, reflected in `rde_qc_findings_total`.
5. QC failures produce episode rejections, reflected in `rde_qc_gate_rejection_ratio`.

Five metrics move in response to a single underlying fault. The alerting design pages on the root cause rather than on the full chain: downstream alerts are inhibited while an upstream alert in the same chain is already firing, and the full five-metric chain is what the investigation dashboard displays after the page, rather than what generates five separate pages.

Scenario A, in `examples/09_observability_scenarios.py`, walks exactly this path on real data, tracing from an aggregate metric that names no specific episode down to the three episodes that were actually damaged.

## 7. Calibrated Thresholds, and Why They Are Not Optional

`rde/observability/drift.py` measures a statistic's null distribution directly, rather than borrowing a constant threshold from elsewhere. The justification for this is measured rather than theoretical.

The familiar PSI thresholds of 0.1 and 0.25 originate in credit-risk practice, where sample sizes typically run to tens of thousands. Table 14 shows what happens when these thresholds are applied at episode scale instead, using 10 fixed histogram bins.

**Table 14. Null-distribution behavior of PSI by sample size.**

| Sample size (n) | Median null PSI | P(null PSI > 0.25) |
|---:|---:|---:|
| 50 | 0.301 | 62.5% |
| 100 | 0.161 | 17.8% |
| 500 | 0.032 | 0.0% |
| 5,000 | 0.003 | 0.0% |

At episode scale, the borrowed threshold fires on 62.5% of comparisons of a distribution against itself, and the median null value already sits above the conventional "significant drift" line. An alert built on this threshold would be pure noise, and, more troublingly, would resemble a working drift detector for as long as nobody investigated a specific alert closely.

Adaptive binning, at approximately 10 samples per bin, reduces this false-positive rate to 15.5%. Calibrating against the baseline's own measured null distribution completes the correction. Table 15 gives the measured operating characteristics of the resulting detector, over 200 trials at n = 50.

**Table 15. Operating characteristics of the calibrated drift detector.**

| Condition | PSI fires | Mean-shift statistic fires |
|---|---:|---:|
| No drift | 0.5% | 0.0% |
| 0.5σ shift | 30.5% | 37.5% |
| 1.0σ shift | 94.0% | 98.0% |
| 2.0σ shift | 100% | 100% |

This result can be reproduced with `pytest tests/test_observability.py -k drift`.

## 8. Dashboards

Five dashboards are defined for this system, described below.

**Dashboard 1 — Pipeline.** Displays run outcomes over time, stage latency by stage, failure rate by `error_class`, throughput, peak RSS by stage, and time since last success. This last panel is the one that matters most, and it is the one a Pushgateway-based design cannot draw at all.

**Dashboard 2 — Dataset health.** Displays episodes by quality status, the rejection-ratio trend, a modality-coverage heatmap, and missingness, together with three sub-scores — temporal, numeric, and alignment quality — kept deliberately separate from one another, since "temporally perfect but misaligned" and "jittery but aligned" are different failure modes with different owners.

**Dashboard 3 — Sensor health.** Displays dropout by modality, decode failures, sampling jitter, the change in tail ratio over time, and a per-cohort breakdown by `factory_id` and `robot_id`. This cohort split is what prevents a healthy global average from concealing a single broken site.

**Dashboard 4 — Distribution health.** Displays drift per feature together with its effective sample size shown beside it, episode duration and frame-count distributions, task mix, and the pinned, named baseline version each comparison is measured against.

**Dashboard 5 — Run investigation.** This dashboard implements the drill-down path as a sequence of database joins rather than a correlation performed by timestamp guesswork. Starting from a metric anomaly, carrying cohort labels such as `stage`, `check_name`, and `modality`, the path proceeds as follows.

1. The anomaly's `trace_id` links to its spans, identifying the failing stage.
2. The stage's `run_id` links to `stage_runs`, and from there to `qc_executions`.
3. The `qc_execution_id` links to `episode_health`, identifying the specific named episodes affected.
4. Those episodes' rows in `qc_records` identify the specific findings, each of which links via `blob_id` to the underlying artifact.

Every link in this four-step path is an existing foreign key. This is precisely what the correlation spine, described in [14-observability-architecture.md](14-observability-architecture.md), Section 5, was built to provide, and it is the reason that spine had to be built before anything else in this layer.

## 9. Retention

Table 16 states the retention policy for each category of telemetry data.

**Table 16. Retention policy.**

| Data | Store | Retention | Rationale |
|---|---|---|---|
| Metrics, raw | Time-series database | 15 days | Operational debugging window |
| Metrics, 5-minute rollup | Time-series database | 13 months | Year-over-year seasonality |
| `pipeline_runs`, `stage_runs` | Catalog and Parquet | Indefinite | Small, bounded, and the trend itself is the value being retained |
| `qc_executions` | Catalog and Parquet | Indefinite | Quality history; the fix for reconnaissance gap G3 |
| `episode_health` | Parquet | Indefinite | The drift feature store |
| `qc_records` | Parquet, with a catalog mirror | Indefinite | Already the engine's established policy |
| Spans | NDJSON, moved to object storage | 30 days | Investigation window |

Retaining this data indefinitely is affordable specifically because the cost was measured rather than assumed: 344 million episodes' worth of health rows occupies 6 GB of Parquet, as shown in [14-observability-architecture.md](14-observability-architecture.md), Section 3. The reason to delete telemetry is cost, and at this measured cost profile, there is not a great deal of cost to justify deleting it for.
