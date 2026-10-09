# 16 — Observability: Final Report

This document is the capstone of the RDE Observability and Data Health project. It is intended to be read after [12-observability-recon.md](12-observability-recon.md), [13-observability-industry-research.md](13-observability-industry-research.md), [14-observability-architecture.md](14-observability-architecture.md), and [15-metrics-catalog.md](15-metrics-catalog.md), in that order.

For the code itself — every module, what it exposes, and the non-obvious decision embedded within it — see [11-code-map.md](11-code-map.md).

## 1. Summary

Prior to this project, the engine could answer the question "what is this dataset, and how was it produced?" It could not answer any question containing the word "when." This project added that missing dimension, governed throughout by one measured rule: metrics aggregate populations, events identify individuals, and traces connect them.

## 2. What the Engine Already Did, and What Observability Adds

Table 1 summarizes, for each area of the system, what capability already existed prior to this project and what this project added to it.

**Table 1. Existing capability versus what observability adds.**

| Already present | Added by this project |
|---|---|
| 24 QC checks in one streaming pass | Visibility into which check is slow, and how findings trend over time |
| Findings persisted to Parquet and SQLite | Findings scoped to a specific validation, so history survives a re-run |
| Episode-level lineage with code fingerprints | A `pipeline_run_id` joining every row a single execution produced |
| Immutable versions, content-addressed blobs | Incremental validation that exploits both, at a measured 35× speedup |
| Severity / status / gate separation | A third policy layer — contracts — that gates a version rather than an episode |
| `RangeProfiler` percentiles | Episode-level drift detection with calibrated thresholds |
| Nine fault injectors | A tenth injector that breaks nothing checkable, plus tests for observability itself |
| Bounded-memory video access | A tiered strategy under which health checking costs 89 ms rather than 36 GB |

## 3. What Was Implemented

The package `rde/observability/` comprises 3,243 lines across ten modules, listed in Table 2.

**Table 2. Modules implemented in `rde/observability/`.**

| Module | Responsibility |
|---|---|
| `context` | The correlation spine — one run identifier and one OpenTelemetry-shaped trace identifier, threaded through the system, nullable outside a run |
| `metrics` | Counter, gauge, and histogram types, with an enforced label policy |
| `instruments` | The 37 declared instruments, organized into three families |
| `events` | High-cardinality analytical rows, backed by Parquet |
| `exporters` | Atomic textfile `.prom` export and NDJSON spans, flushed synchronously on exit |
| `session` | Binds the above components together; a single call site is one `with` block |
| `health` | Episode feature extraction and data-health emission, performed inside the existing pass |
| `drift` | Episode-grain comparison, using calibrated thresholds |
| `contracts` | Release policy — the third layer, following checking and gating |
| `video_health` | Tiers 0 through 2 of video health checking |

Beyond this package, the project also added `rde/qc/cache.py` for incremental validation, three new catalog tables (`pipeline_runs`, `stage_runs`, `qc_executions`) plus two Parquet-only event tables (`episode_health`, `drift_reports`) that have no catalog counterpart, the `rde obs` command group with eight subcommands, two demonstration scripts, and 52 tests.

Across the whole repository, the system now comprises 14,979 lines of engine code, 187 passing tests with 1 skipped, 18 architecture documents, and 9 runnable examples.

## 4. What Was Measured

Every figure in Table 3 was produced directly on the development machine, against the real SO-100 dataset, by a script present in the repository. None of these figures are projections.

**Table 3. Measured results.**

| Category | Measurement | Value | Reproduced by |
|---|---|---|---|
| Cardinality | Series produced by safe labels | 768 | `examples/08` |
| Cardinality | Series if `episode_id` is added, at 344M episodes/year | 264,192,000,000 | `examples/08` |
| Cardinality | Bytes per active series (head-block proxy) | 631 B | `examples/08` |
| Cardinality | Bytes per Parquet row, at realistic entropy | 17.5 B | `examples/08` |
| Cardinality | Same facts, time-series database versus Parquet, at 344M episodes | 167 TB versus 6.02 GB | `examples/08` |
| Drift calibration | P(PSI > 0.25) on no-drift data, n = 50, 10 fixed bins | 62.5% | `pytest -k psi_threshold` |
| Drift calibration | Same, at n = 5,000 | 0.0% | Same |
| Drift calibration | Calibrated detector, false-positive rate | 0.5% | `pytest -k operating_characteristics` |
| Drift calibration | Calibrated detector, detection rate at 1σ | 94–98% | Same |
| Cost | Share of QC time spent in two video checks | 96.8% of 5.093 s | `rde obs metrics --show` |
| Cost | Feature profiler cost | 1.2% of the QC stage | Same |
| Cost | Whole observability layer, end to end | +0.8% (6.011 s → 6.061 s) | [15-metrics-catalog.md](15-metrics-catalog.md), Section 3 |
| Cost | Active series on a real run | 667, across 37 metrics | `rde obs metrics` |
| Incremental validation | Cold validation | 2.733 s | `rde qc run … --incremental` |
| Incremental validation | Warm (unchanged version) | 0.078 s | Same |
| Incremental validation | Speedup | 35× | Same |
| Video | SO-100, stored versus fully decoded | 470 MB / 36.2 GB (77×) | `rde obs video … --tier 0` |
| Video | Tier 0 cost | 1.9 ms per file | Same |
| Video | Tier 1 cost | 89 ms per episode-camera | `--tier 1` |
| Video | Tier 1 peak memory | One frame, approximately 0.9 MB | Same |
| Alignment | Mean cross-modal offset, clean data | 0.0000 ms | `pytest -k alignment_offset` |
| Alignment | Same, after a 5 s video shift | 5000.0000 ms | Same |

## 5. What Is Only Projected

The figures in Table 4 are stated separately from those in Table 3, because conflating measured and projected figures is how a proof of concept comes to oversell itself.

**Table 4. Projected, not measured, figures.**

| Projection | Basis for the projection | What remains unverified |
|---|---|---|
| 167 TB of time-series-database memory at 344M episodes | Measured per-series cost multiplied by a computed series count | No time-series database was actually run at this scale; the per-series figure is a Python proxy, roughly three times Prometheus's documented 200-byte `memSeries` floor. Even at that floor, the projected figure is 52.8 TB, still disqualifying. |
| 6.02 GB of Parquet for the same facts | Measured per-row cost multiplied by row count | Not written at scale; compression ratios shift with real data entropy. |
| ClickHouse as the production event store | Both ClickHouse and DuckDB read Parquet directly, so row shapes carry over unchanged | Never deployed; the migration path is argued for, not performed. |
| Tier 1 video cost at approximately 9 s per version | Measured per-reference cost multiplied by reference count | Not run across a full fleet; I/O contention at scale is unmodeled. |
| Moving from textfile export to VictoriaMetrics as "one module" | The instrumentation is already OTLP-shaped | No remote-write path was actually built. |
| Alert thresholds proposed in [15-metrics-catalog.md](15-metrics-catalog.md), Section 6 | Two are explicitly labeled starting heuristics; the rest derive from contracts, engineering limits, or measured baselines | None of these alerts has run against production traffic. |

## 6. The Research Questions, Answered

This section answers each of the twelve research questions posed at the start of the project.

**Q1. What does observability of robotics data capture that infrastructure observability does not?** Cross-modal temporal alignment. All five canonical pillars of data observability — freshness, volume, schema, distribution, and lineage — pass on a dataset whose video has been shifted five seconds against its recorded state. Measured directly, the alignment metric moves from 0.0000 ms to 5000.0000 ms under exactly this fault.

**Q2. How is pipeline failure distinguished from pipeline success accompanied by bad data?** By two separate metric families with two separate owners: `rde_pipeline_*` answers the first question, and `rde_data_*` answers the second. Scenario B in the demonstration suite illustrates the gap directly: every stage completes successfully, all 24 checks pass, and the resulting dataset is nonetheless ruined.

**Q3. How is systematic semantic corruption detected when schema, counts, and null rates all pass?** Through distribution monitoring at episode granularity, gated by a contract. A 1.35× rescale of `action.joint` passes every structural check and is correctly blocked by the contract layer, which names the specific features responsible.

**Q4. How is multimodal alignment monitored continuously, rather than checked once?** By measuring the offset on every episode the alignment check evaluates, not only on episodes that fail outright. A metric that appears only when something is already broken cannot demonstrate that nothing currently is, and cannot show a drift from 2 ms to 20 ms while both values remain under the 33 ms failure threshold.

**Q5. How are trajectory distributions compared when frames are autocorrelated?** By reducing each episode to a fixed-length feature vector and comparing populations of episodes, rather than comparing frames directly. Fifty independent demonstrations constitute a more defensible sample than 55,000 correlated frames drawn from those same demonstrations. Effective sample size is published beside every drift statistic, and a comparison judged underpowered issues a warning rather than a block.

**Q6. How is video quality monitored without decoding petabytes of footage?** Through tiering. Container headers, checked at 1.9 ms per file, catch cohort-wide failures; five sampled frames per clip, checked at 89 ms, catch per-clip failures; and embeddings are documented as a further tier but deliberately left unbuilt.

**Q7. How are heavy-tailed force and tactile signals monitored without generating false alarms?** By never alerting on the absolute level of such a signal. On the ALOHA dataset, a force channel's maximum value is 2.4 times its 99.5th percentile, and this is simply a successful grasp rather than an anomaly. `rde_sensor_tail_ratio` exists so that a shift in this ratio over time, rather than its absolute value, is the signal that drives an alert — the same lesson that forced the `value_range` recalibration described in [05-qc-architecture.md](05-qc-architecture.md).

**Q8. Where do episode-level facts live?** In Parquet, queried today by DuckDB and, at fleet scale, intended to be queried by ClickHouse. The measured difference in storage cost for the same facts is 167 TB against 6.02 GB.

**Q9. What must a Prometheus-style time-series database never contain?** `episode_id`, `frame_id`, `run_id`, `trace_id`, `blob_id`, `task_id`, file paths, code fingerprints, and raw observed values. This is enforced by a label policy that raises an exception rather than merely documenting the rule.

**Q10. When are traces more useful than metrics?** When the question at hand is which specific execution produced a result, rather than how often something happens. The durable value here is the `trace_id` column present in the analytical tables, which converts the drill-down process into a sequence of database joins rather than a correlation performed by timestamp guesswork.

**Q11. How does the observability layer remain cheaper than the pipeline it observes?** By riding the pass the engine already performs, rather than adding a new one. This is measured at +0.8% overhead. A second traversal of the data to compute drift features separately would have cost an estimated 30–50% and would have broken the engine's bounded-memory guarantee.

**Q12. How is a broken cohort prevented from hiding inside a healthy global average?** Through the labels `factory_id`, `robot_id`, and `sensor_rig_type`, which are accepted as conditionally safe specifically for this purpose — their cardinality cost is judged worth paying. The corresponding per-cohort dashboard panel is designed but not populated with real data, because this proof of concept operates with only one factory and two robots.

## 7. Remaining Limitations

The following ten limitations are stated plainly, on the view that a proof of concept claiming no limitations is not being fully honest about its own scope.

1. **Single-machine deployment.** Every measurement in this report was taken on one WSL machine with an approximately 9 GB memory cap. Distributed telemetry concerns — collector fan-in, clock skew between workers, and partial-run reconciliation — remain unaddressed.
2. **No time-series database was actually run.** The textfile export is real and well-formed, but no Prometheus instance has scraped it, so nothing validates the alert expressions in [15-metrics-catalog.md](15-metrics-catalog.md), Section 6, end to end.
3. **Cohort labels are untested at cardinality.** `factory_id` and `robot_id` are accepted by the label policy but exercised by nothing in practice, because this project's dataset has exactly one factory and two robots.
4. **Drift features are trajectory-only.** No appearance-based drift is wired into the drift path; Tier 2 perceptual hashes are computed but not yet fed into the `compare()` function.
5. **The cache's table fingerprint uses size and modification time, not content.** This is sound protection against accidental change, but not against a determined adversary or a filesystem with coarse-grained timestamps.
6. **`rde_data_sampling_jitter_seconds` uses per-episode means**, so within-episode jitter is not individually visible. The underlying per-frame interval distribution exists inside the profiler but is not currently surfaced as its own metric.
7. **Contracts operate per stage and carry no notion of intent.** A contract inferred from `raw/v1` correctly rejects `clean/v1` because a transform resized images — a real and, in this case, intended schema change. No mechanism yet exists to express "this specific change was reviewed and approved."
8. **Tier 3 video checking is documented but not built.**
9. **Retention is a policy stated in a document**, not a policy enforced by any running system.
10. **The dashboard is a static snapshot.** [`docs/dashboard/`](dashboard/) renders real, measured telemetry, but it embeds that telemetry directly rather than querying a live backend; there is no time-series database behind it, and no Grafana configuration is shipped alongside it.

## 8. Productionization Roadmap

The following four stages are ordered by dependency — by what each stage unblocks — rather than by which is most appealing to build first.

**Stage 1 (weeks): make it real.** Stand up VictoriaMetrics and scrape the existing textfile export. Validate every alert expression in [15-metrics-catalog.md](15-metrics-catalog.md) against actual series. Ship the five dashboards described there. Add remote-write support to the exporter; since the instrumentation is already OTLP-shaped, this is a new module rather than a rewrite of existing code.

**Stage 2 (months): make it distributed.** Replace SQLite with PostgreSQL for the catalog. Emit spans over OTLP to a real collector. Solve partial-run reconciliation, since a run that dies mid-stage currently leaves an open span and no corresponding `pipeline_runs` row. Add per-worker labels, and verify that the cardinality budget established in [14-observability-architecture.md](14-observability-architecture.md) still holds under them.

**Stage 3 (quarters): make it fleet-scale.** Move event rows to ClickHouse over object storage, retaining Parquet as the interchange format throughout. Introduce the `factory_id` and `robot_id` cohort labels in earnest, and measure their real cardinality rather than relying on the projection in Table 4. Wire the feedback flywheel further upstream, into edge-level data selection, so that data health informs what to upload from the edge — the only affordable strategy at a planning scale of 3 to 10 PB per year.

**Stage 4: close the loop.** Build Tier 3 video checking: embeddings over sampled frames, appearance drift, and semantic coverage. Build a contract-approval workflow, so that an intended schema change becomes a reviewable event rather than simply a blocked release.

## 9. Definition of Done

Table 5 lists every requirement of this project and its current status.

**Table 5. Definition-of-done checklist.**

| Requirement | Status |
|---|---|
| Repository architecture inspected | Met — [12-observability-recon.md](12-observability-recon.md) |
| Existing telemetry inventoried | Met — [12-observability-recon.md](12-observability-recon.md), Section 1 |
| Industry research performed | Met — [13-observability-industry-research.md](13-observability-industry-research.md) |
| Claims classified as documented, inferred, or assumed | Met — [13-observability-industry-research.md](13-observability-industry-research.md), seven evidence classes |
| Observability architecture documented | Met — [14-observability-architecture.md](14-observability-architecture.md) |
| Metrics taxonomy exists | Met — [15-metrics-catalog.md](15-metrics-catalog.md), 37 instruments |
| Cardinality analysis exists | Met — [14-observability-architecture.md](14-observability-architecture.md), Section 3, measured directly |
| Backend comparison exists | Met — [14-observability-architecture.md](14-observability-architecture.md), Section 7 |
| Data-health event schema exists | Met — `events.py`, 5 tables |
| Trace model exists | Met — `context.py` |
| Distribution-drift strategy exists | Met — `drift.py`, calibrated |
| Video sampling strategy exists | Met — `video_health.py`, Tiers 0–2 |
| Heavy-tail strategy exists | Met — `rde_sensor_tail_ratio`, [15-metrics-catalog.md](15-metrics-catalog.md), Section 4 |
| Data-contract strategy exists | Met — `contracts.py` |
| Incremental validation strategy exists | Met — `rde/qc/cache.py`, 35× |
| Fault-injection tests exist | Met — 52 observability tests |
| Operational dashboard exists | Met — [`docs/dashboard/`](dashboard/), built from measured telemetry |
| Two end-to-end failure demonstrations exist | Met — `examples/09`, both pass |
| No per-episode cardinality explosion | Met — enforced; the policy raises an exception |
| Existing lineage remains authoritative | Met — the spine is a foreign key pointing upward |
| Streaming and bounded memory remain intact | Met — +0.8% overhead, measured |
| All new behavior has test coverage | Met — 185 tests passing, 1 skipped |
| Documentation explains why, not only what | Met — five documents, plus [11-code-map.md](11-code-map.md) for what |

Twenty-two of twenty-three requirements are fully met. The remaining item, Tier 3 video checking, is a documented boundary rather than an unnoticed omission; and the operational dashboard, while built and populated with genuinely measured data, remains a static snapshot rather than a live view backed by a running query engine.

## 10. The Claim This Project Actually Supports

This project's contribution is not "here is a monitoring stack." Monitoring stacks are inexpensive to assemble, and the one built here is deliberately small — a textfile export and some Parquet files — because the analytical plane this design required already existed in the repository, and adding ClickHouse merely to make a point would have constituted infrastructure built for appearance rather than for need.

The claim this project actually supports is narrower, and more useful: a multimodal robotics data engine has a failure mode that general-purpose data observability tooling cannot see, in which the pipeline succeeds, every structural check passes, and the resulting dataset is nonetheless wrong. Catching this failure mode requires treating cross-modal alignment as a continuously measured quantity, monitoring distribution at a unit of analysis that respects temporal correlation in the underlying data, and calibrating thresholds against the data's own null distribution rather than borrowing them from an unrelated field.

All three of these requirements are implemented in this system, and each is demonstrated directly against a fault that every other check in the engine passes without complaint.
