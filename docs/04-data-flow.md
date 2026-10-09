# 04 — Data Flow and Stage Contracts

Every stage in the pipeline has an explicit input contract and output contract. This is what allows each stage to be run, tested, and replaced independently of the others.

## 1. The Pipeline as a Sequence of Stages

Table 1 enumerates the pipeline stages in execution order, from data acquisition through evaluation and feedback.

**Table 1. Pipeline stages, in execution order.**

| Order | Stage | Reads | Produces |
|---|---|---|---|
| 1 | Download | Source repository or local path | `SourceLocation`, via `download_hf_dataset()` |
| 2 | Source cache | — | Verbatim upstream bytes, never subsequently mutated |
| 3 | Ingest | Source cache | `raw/vN`: canonical Parquet, blobs, manifest, and catalog rows, via `SourceAdapter.inspect()` / `.build_manifest()` / `.iter_episodes()` / `.iter_frame_batches()` |
| 4 | Validate | `raw/vN` | `QCResult`: findings and a per-episode status, via `QCEngine.run()` |
| 5 | Transform | Validated `raw/vN` | `clean/vN`: filtered, normalized, timestamp-rebased data with deferred operations recorded in the manifest, via `TransformPipeline.run()` |
| 6 | Re-validate | `clean/vN` | A second `QCResult`, confirming the transform introduced no defects, via a second `QCEngine.run()` |
| 7 | Build | `clean/vN` | `training/vN`: splits and a training contract, with the frame table referenced rather than copied, via `DatasetBuilder.build()` |
| 8 | Serve | `training/vN` | `DatasetView`, via `DataEngine.get_dataset(version)`, consumed either by exploration tooling (CLI, DuckDB) or by a model adapter |
| 9 | Adapt | `Sample[]` | `ModelInput`, via `ModelInputAdapter.adapt(samples)` |
| 10 | Predict | `ModelInput` | `Prediction` (actions and confidence), via `Model.predict()` |
| 11 | Evaluate | Predictions against held-out demonstrations | `EvaluationResult` and per-frame outcomes, via `Evaluator.evaluate()` |
| 12 | Detect feedback | `EvaluationResult` | `FailureRecords` and `EpisodeReviews`, via `FeedbackLoop.detect()` |
| 13 | Select for retraining | Feedback | Episode weights, via `build_retraining_selection()` |
| 14 | Rebuild | Episode weights | `training/vN+1`, via `DatasetBuilder.build(episode_weights=…)` |

Stage 14 closes the loop back to stage 7, so that the output of evaluation becomes an input to the next build rather than a terminal report.

## 2. The Observability Layer, Running Alongside

A separate process runs alongside every stage listed in Table 1, without ever appearing in the data path itself. `ObservabilitySession.for_run()` mints a `run_id` and a `trace_id` once, at the start of a run. Each stage subsequently contributes a span, a duration histogram, volume counters, and a `stage_runs` row; each QC pass contributes per-check timing, findings counters, and `episode_health` rows; and, on completion, the session writes a catalog mirror, Parquet event files, spans, and a `.prom` export.

The telemetry plane is a side-channel, not a stage: it observes work that is already happening and never causes any of its own. Running the pipeline with `Pipeline(cfg, observability=False)` produces byte-identical data to a run with observability enabled, and the measured cost of leaving observability on is **+0.8%**.

## 3. Stage Contracts

Table 2 states the formal contract for each stage: what it consumes, what it produces, and what it guarantees to callers.

**Table 2. Stage contracts.**

| Stage | Input | Output | Guarantee |
|---|---|---|---|
| Download | Repository identifier | `SourceLocation` | Resumable; the cache is never written to afterward |
| Ingest | Source root, dataset identifier, version | `IngestResult` | Streaming; no source-format concept leaks past this stage |
| Validate | Dataset identifier, version | `QCResult` | One pass over the frames, evaluating N checks; nothing is deleted |
| Transform | Version and operator list | `PipelineResult` | Streaming; lineage is recorded; deferred operations are declared |
| Build | Clean version, split specification | `BuildResult` | Episode-level splits; a training contract is emitted |
| Get dataset | Dataset identifier, version | `DatasetView` | Lazy image loading; version-addressed, never path-addressed |
| Adapt | `Sample[]` | `ModelInput` | Reads the training contract; never hard-codes field names |
| Evaluate | View, model, adapter | `EvaluationResult` | Per-frame outcomes are retained |
| Feedback | `EvaluationResult` | `FeedbackReport` | Relative thresholds; failing episodes are quarantined, not deleted |
| Observe | Any stage | Span, metrics, and event rows | Best-effort; never fails a stage; nullable outside a run |
| Contract | Manifest, QC result, and drift measurements | `ContractReport` | A per-version release verdict; `SKIP` is distinct from `PASS` |

## 4. Why Validation Runs Twice

The `validate` stage and the `qc_post_transform` stage invoke the same QC engine against two different versions of the data. This duplication is not redundant, because the two passes answer different questions:

1. The raw-data pass determines what is fit to transform.
2. The post-transform pass verifies that the transformation itself introduced no new defects.

This second pass earns its keep on real data. Applied to `clean/v1`, it reports a warning for episode 36 that the pass over `raw/v1` did not report. Normalization changes the distribution of the underlying values, so the range profiler relearns its bounds, and a different episode becomes the statistical outlier under the new distribution. This is a true statement about the transformed data, and precisely the kind of fact one needs before training on it.

## 5. The Correlation Spine

Before the observability layer was introduced, `Pipeline.run()` minted no identifier for itself. Each stage that produced a version generated an independent `uuid4`, so the eight lineage rows written by a single execution carried no column linking them together, and a question such as "show me everything that happened in run 81" had no answer.

`RunContext.start()` now mints one `run_id` and one OpenTelemetry-shaped `trace_id` at the start of a run and threads both values down through every component that participates in it:

```
pipeline_run_id ─┬─▶ stage_runs        (one row per stage)
                 ├─▶ transform_runs    (+ pipeline_run_id, nullable)
                 ├─▶ qc_executions     (one row per validation)
                 │      └─▶ qc_records (+ qc_execution_id, nullable)
                 │      └─▶ episode_health
                 └─▶ spans             (trace_id / span_id, NDJSON)
```

This design rests on three deliberate properties.

1. **Minted once.** Sub-components receive the run context; they never create one of their own.
2. **Additive.** `TransformRun.run_id` retains its original meaning as the identity of one lineage edge, and `pipeline_run_id` is a foreign key pointing upward from it. Lineage remains the authoritative provenance record.
3. **Nullable.** `QCEngine.run()`, called directly from a notebook, has no pipeline above it and must behave identically to a run invoked through the full pipeline. Observability that makes the library harder to use as a library is considered to have failed its purpose.

## 6. Streaming at Every Stage

No stage in the pipeline materializes a full table in memory. Ingestion passes data from adapter to Arrow batch to Parquet writer; validation performs one filtered scan per episode, reading only the required columns; transformation performs a fit pass followed by an apply pass, one batch at a time; and evaluation iterates over samples in batches, decoding images only when the model adapter requests them.

As a consequence, peak memory usage is bounded by the size of a single batch — a few hundred rows — regardless of the overall size of the dataset.

## 7. Failure Handling

The system's response to failure differs by the kind of failure encountered, as summarized below.

1. **A check that raises an exception** becomes a `CRITICAL` finding rather than an aborted run. The findings from other checks continue to matter, and the crash itself is recorded as a QC result.
2. **A transform that raises an exception** still writes its lineage record, with the error captured in the `metrics` field. A pipeline that crashed partway through is exactly the situation in which its lineage record is most needed.
3. **A stage that fails** is recorded in the run record together with its error and duration. Since the introduction of the observability layer, it is additionally recorded as a `stage_runs` row with a normalized `error_class`, as an errored span, and as an increment to the counter `rde_pipeline_stage_failures_total`.
4. **A telemetry export that fails** does none of the above: it increments `rde_observability_export_failures_total` and returns. This behavior is inverted, deliberately, from every other failure mode in the engine. A QC failure is a result worth recording; a telemetry failure is not a result at all, and an ingestion run must never fail merely because a metrics collector is unreachable.
