# 08 — An End-to-End Run

This document presents a complete, annotated execution of the pipeline against the real dataset. It can be reproduced with the following command:

```bash
rde run --config configs/so100_pickplace.json
```

## 0. Configuration

The run uses the following configuration:

```json
{
  "dataset_id": "so100_pickplace",
  "source_repo": "lerobot/svla_so100_pickplace",
  "raw_version": "raw/v1", "clean_version": "clean/v1", "training_version": "training/v1",
  "normalize": ["state.joint", "action.joint"], "normalize_scheme": "mean_std",
  "image_size": [224, 224],
  "keep_quality": ["passed"],
  "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
  "split_seed": 42, "split_mode": "quantile",
  "model": "nearest_neighbor", "baseline_model": "mean_action"
}
```

## 1. Download to Source Cache (0.2 s)

The download stage transfers 470.1 MB across 30 files into `$RDE_DATA_ROOT/source_cache/`. This location is never inside the source tree, and it is never written to again after the download completes.

## 2. Ingest to `raw/v1` (0.6 s)

Ingestion produces the following summary:

```json
{"episodes": 50, "frames": 19631, "source_format": "lerobot/v3.0",
 "modalities": ["action.joint", "state.joint", "rgb.top", "rgb.wrist"],
 "blobs": 2, "table_bytes": 903442, "row_groups": 20, "anomalies": []}
```

The two source videos are hard-linked into the content-addressed store, and the 19,631 frame rows are rewritten from upstream's single row group into 20 row groups. LeRobot, as a source format, ceases to exist as a concept downstream of this stage.

## 3. Validate `raw/v1` (2.7 s)

Validation reports the following outcome:

```
checks run: 24   findings: 9   by_severity: {info: 4, warning: 5}
rejected: []     warned: [9, 31]     status: warning
episodes: 48 passed, 2 warning
```

There are zero blocking findings, indicating a genuinely clean dataset. The two warnings that are recorded are specific and interpretable:

```
episode  9  numerical.value_range     action.joint exits learned bounds, dim [3] (wrist_flex)
episode  9  numerical.value_range     state.joint  exits learned bounds, dim [3]
episode 31  numerical.episode_outlier 575 frames vs median 380, MAD 22
episode 31  numerical.value_range     action.joint exits learned bounds, dim [2] (elbow_flex)
episode 31  numerical.value_range     state.joint  exits learned bounds, dim [2]
```

This validation pass also records the learned range profile, the episode length distribution (minimum 326, median 380, maximum 575, a 1.76× spread), confirmation that the dataset is single-task, and confirmation of zero null values across 13 columns and 19,631 rows.

## 4. Transform to `clean/v1` (0.1 s)

The transform stage applies the following operations:

```
operations: quality_filter · timestamp.normalize ·
            normalize.state.joint · normalize.action.joint · images.resize(deferred)
episodes:   50 → 48        frames: 19,631 → 18,668        dropped: [9, 31]
```

The configuration setting `keep_quality: ["passed"]` causes the two warned episodes to be dropped at this stage. The fitted normalization statistics are recorded for reproducibility:

```
state.joint mean [14.425, -56.579, 55.142, 63.096, 85.675, 9.301]
state.joint std  [28.182,  34.416, 19.380, 16.775, 12.554, 9.727]
```

The `images.resize` operation is deferred: no video is re-encoded at this stage. The manifest records the intent to resize, and the model adapter applies the resize at decode time.

## 5. Re-Validate `clean/v1` (2.5 s)

The second validation pass reports:

```
checks run: 24   findings: 6   rejected: []   warned: [36]
```

A new warning appears here that did not appear in the validation of `raw/v1`. Normalization changed the distribution of the underlying values, so the range profiler relearned its bounds, and a different episode became the statistical outlier under the new distribution. This is a true statement about the transformed data, and it demonstrates precisely why the second validation pass exists: it verifies that the transformation introduced no defects, and in this instance it correctly reports what the transformation changed.

## 6. Build to `training/v1` (0.02 s)

The build stage produces:

```
episodes 48    frames 18,668
splits   train 38 (14,919)   val 5 (1,841)   test 5 (1,908)
contract observation=[state.joint, rgb.top, rgb.wrist]   action=action.joint
```

Splits are assigned by episode rather than by frame, since consecutive frames differ by only one control step; a frame-level split would place near-duplicate frames on both sides of a split boundary, which would measure memorization rather than generalization. The `quantile` split mode is used because it yields exact proportions, which matters at 48 episodes, where independent hashing would otherwise produce an uneven 42/3/3 split.

This stage costs 12.2 KB. The frame table is referenced from `clean/v1` rather than copied.

## 7. Evaluate (1.3 s)

Evaluation of the nearest-neighbor model against the test split produces:

```
model nearest_neighbor   adapter state   split test   frames 1,908
  MSE               0.02385        baseline (mean_action)  0.90799
  MAE               0.09748        baseline                0.78940
  normalized MAE    0.10446
  success @0.1      51.6%
  mean confidence   0.374
  confidence/error correlation   -0.790
  latency           0.0138 ms/frame
  → 2.6% of baseline MSE (38× better)
```

The correlation of −0.79 between confidence and error is, in this context, more informative than the MSE value alone: it demonstrates that the model's confidence tracks its correctness, which is what makes the low-confidence signal used in the next stage meaningful rather than decorative.

## 8. Feedback (0.003 s)

The feedback stage identifies episodes for targeted data collection:

```
thresholds     error p90 = 0.1847,  confidence p10 = 0.0289
failure frames 279     {high_error: 88, low_confidence: 88, both: 103}
episodes flagged 5

  ep   fail/total   mean err  mean conf  priority  recommendation
  27       94/326    0.11854     0.3686    0.3225  collect more demonstrations
  14       87/375    0.11456     0.2887    0.2586  collect more demonstrations
  29       49/374    0.09820     0.3343    0.1439  collect more demonstrations
  42       32/417    0.07205     0.4824    0.0823  collect more demonstrations
   3       17/416    0.09043     0.3798    0.0446  collect more demonstrations
```

The thresholds used here are percentiles computed over this specific run rather than fixed absolute values, since what constitutes a large error depends on the data's units, its normalization, and the model producing the predictions.

## 9. Closing the Loop to `training/v2`

The demonstration script `examples/06_feedback_flywheel.py` continues the run as follows:

```
quarantine    episodes [27, 14, 29, 42, 3]
statuses      {passed: 42, quarantined: 5, warning: 1}   total 48 — nothing deleted
build         training/v2, 48 episodes, weights {3:3.0, 14:3.0, 27:3.0, 29:3.0, 42:3.0}
lineage       build_training_dataset ['clean/v1'] -> training/v2
              recorded episode weights: {'3': 2.0, '14': 2.0, ...}
```

Flagged episodes are upweighted rather than removed. A model's failure on a given episode constitutes evidence about the model at least as often as it constitutes evidence about the data, and deleting such episodes would discard exactly the examples that the next training round needs most.

## 10. Totals

Table 1 summarizes the run as a whole, including the feedback-loop retraining step in §9 that builds `training/v2`. This is a superset of the plain `rde run` pipeline that [README §1](../README.md#1-what-it-does-on-real-data) and [09-resource-usage.md](09-resource-usage.md) report (7.0 s), which stops after evaluating and generating feedback on `training/v1` and does not build a second training version — the two totals are not measuring the same run.

**Table 1. End-to-end run summary.**

| Metric | Value |
|---|---|
| Wall-clock time | 7.8 s |
| Peak RSS | 537 MB |
| Versions produced | `raw/v1` → `clean/v1` → `training/v1` (and `training/v2`) |
| Tabular data written | 1.9 MB |
| Video re-encoded | 0 bytes |
| Data deleted | None |
| Telemetry emitted | 667 series, 4 event tables, 8 spans |
| Observability overhead | +0.8% |

## 11. Telemetry Produced by the Run

Every execution also writes a correlated telemetry record, organized as follows.

```
rde_data/observability/
├── metrics/rde.prom                       667 series, 37 instruments
├── spans/run-<id>.spans.ndjson            8 spans, one per stage
└── events/
    ├── pipeline_runs/run-<id>.parquet     1 row  — the run header
    ├── stage_runs/run-<id>.parquet        8 rows — timings, volume, peak RSS
    ├── qc_executions/run-<id>.parquet     2 rows — one per validation pass
    └── episode_health/run-<id>.parquet   98 rows — 50 + 48 episode summaries
```

Mirrored headers of this telemetry are also written to the catalog. The run record itself carries a `run_id` and a `trace_id`, which makes it usable as a join key rather than a loose artifact:

```json
{
  "dataset_id": "so100_pickplace",
  "run_id":  "run-1a7808af436b6d72",
  "trace_id": "e3e7c7414163729f4033089e61025cb3",
  "stages": [ ... ]
}
```

Every lineage row and every QC finding produced by this run carries the same `run_id`, so the question "show me everything that happened in this execution" reduces to a single `WHERE` clause. Prior to the observability layer, the eight lineage rows produced by a run carried no column linking them together.

## 12. Verifying the Run Afterward

The following commands inspect the data produced by the run:

```bash
rde version list so100_pickplace
rde version lineage so100_pickplace training/v1
rde version trace   so100_pickplace training/v1
rde qc findings     so100_pickplace raw/v1 --episode 9
rde explore overview so100_pickplace training/v1
rde explore dims     so100_pickplace raw/v1
rde explore sql      so100_pickplace "SELECT episode_id, count(*) FROM frames GROUP BY 1 LIMIT 5"
```

The following commands inspect the telemetry produced by the run:

```bash
rde obs runs     so100_pickplace          # did it run, how long, under which config
rde obs stages   so100_pickplace          # has any stage gotten slower
rde obs quality  so100_pickplace          # has the QC verdict changed over time
rde obs health   so100_pickplace raw/v1   # per-episode health, worst first
rde obs metrics                           # the exposition, and whether it is stale
rde obs video    so100_pickplace raw/v1   # container health without decoding
rde obs contract so100_pickplace raw/v1   # infer a release contract
```

Two demonstration scripts walk through the full investigation path end to end: one traces a camera dropout from an aggregate metric down to the three damaged episodes, and the other traces a distribution shift that all twenty-four QC checks pass and that only the drift contract catches. Both are run with:

```bash
make observability
```
