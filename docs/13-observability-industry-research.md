# 13 — Industry Research: Observability for Robotics Data

This document is the second deliverable of the RDE Observability and Data Health project. Every claim below carries an evidence-class tag, defined in Table 1, indicating the strength and origin of the evidence behind it.

**Table 1. Evidence-class tags used in this document.**

| Tag | Meaning |
|---|---|
| [DOCUMENTED] | Stated in official or primary documentation, or in a vendor's own technical writing about their own system |
| [VENDOR] | Vendor marketing or a vendor technical blog describing a market rather than the vendor's own internals — directionally useful, but commercially motivated |
| [ACADEMIC] | Peer-reviewed or standard statistical literature |
| [INFERRED] | A reading of public evidence that is not stated outright in that evidence |
| [STANDARD] | Widely held engineering practice |
| [RECOMMENDATION] | A judgment made specifically for this system |
| [UNKNOWN] | Not publicly documented; searched for and not found |

## 1. The Most Important Finding: The Evidence Is Not There

The research brief for this project asked that evidence be prioritized from Waymo, Nuro, Cruise, Zoox, Tesla, and Uber ATG. No primary engineering documentation from any of these six companies concerning data-pipeline observability or data-quality architecture could be located. [UNKNOWN]

Searches for their data infrastructure return, almost exclusively, three categories of result: interview-preparation content farms (such as interviewquery.com, datainterview.com, and techprep.app) that describe what interview questions allegedly cover; data-labeling and triage vendors writing about the autonomous-vehicle market generally; and press coverage of fleet expansion.

The frequently repeated figures in this space — for example, that Waymo generates 1.8 TB per vehicle-hour, or that autonomous-vehicle fleets produce 11–152 TB per vehicle-day — trace back to iMerit, a data-services vendor, rather than to the autonomous-vehicle companies themselves. These figures should be treated as [VENDOR] order-of-magnitude claims rather than as measurements.

This finding has a direct consequence for the present project: no architectural decision in this system should be justified on the grounds that "this is what Waymo does," because that justification cannot be supported by available evidence. What can be supported is reasoning from documented tool behavior, documented statistical results, and measurements taken directly on this system. The remainder of this document is confined to those three sources of evidence.

## 2. Foxglove — the Closest Thing to a Primary Source

Foxglove is the only company writing publicly and technically about robotics data observability as a discipline in its own right. Its claims concern its own product, and are therefore treated as [DOCUMENTED] for what its own system does.

Foxglove describes its architecture as requiring four layers. The data model is described as "a time-aware system based on the principle of composition over inheritance." The data layer must handle "asynchronous data streams from multiple diverse sources" and must index across multiple timestamps, including both `receive_time` and `log_time`. The data-semantics layer must understand spatial relationships, including "projections and transform hierarchies." The visualization engine must render both 2D and 3D content, live and historical alike.

Foxglove's product comprises several components. MCAP is an open multimodal logging container format, released in 2022. The Data Platform provides petabyte-scale storage, search, and query, deployable in the cloud, on-premises, or air-gapped. The Foxglove Agent runs directly on the robot, making data importable as soon as it is logged, with selective range import from edge to cloud. SDKs are provided in Python, Rust, and C++.

Two findings from this source materially changed the design of this project's observability layer.

First, **two clock domains are treated as a first-class concept.** Foxglove indexes on both `receive_time` and `log_time`. This engine already confronts exactly this distinction, in the form of LeRobot v3.0's episode-relative versus file-absolute video clocks, which is precisely what the `sync.video_alignment` check exists to catch. Prior to this research, the distinction was treated purely as a validation problem rather than as an observable quantity in its own right. [RECOMMENDATION] Clock-domain skew should be treated as a continuously monitored metric, not solely as a pass/fail check.

Second, **selective edge-to-cloud import is the scaling answer.** At the planning scale stated in the project brief — 3 to 10 PB per year — it is not feasible to ingest everything and validate it later. The decision of what to upload from the edge is itself informed by data health. This has the same shape as this system's feedback flywheel, applied one step further upstream.

## 3. Data Triage — the Autonomous-Vehicle Industry's Actual Discipline

The following description, from iMerit's series on autonomous-vehicle data triage, is [VENDOR] evidence but is conceptually sound: "Triage addresses the problem by filtering, prioritizing, and routing data for further review, annotation, and potential use in model training." A more advanced form, directed triage, is described as follows: "Directed triage extends this workflow by using AI to identify model weaknesses and request targeted scenarios, while human experts validate and act on those requests."

This is exactly what `rde/feedback/flywheel.py` in this system already does: it nominates episodes for retraining based on where the model failed, and flags them for human review in the catalog. This system implemented directed triage without having a name for it.

[RECOMMENDATION] The mechanism should be named explicitly. The feedback flywheel is not a peripheral demonstration; it implements the autonomous-vehicle industry's core data-selection discipline, and it should be instrumented accordingly, with metrics covering nomination rate, review queue depth, review latency, and acceptance rate.

## 4. Batch-Job Telemetry: What the Tools Actually Document

### 4.1 Pushgateway — [DOCUMENTED], and the Documentation Says No

The Prometheus project's own guidance is unusually direct on this point: "The only valid use case for the Pushgateway is for capturing the outcome of a service-level batch job." Table 2 lists the documented limitations of the Pushgateway that are relevant to this system.

**Table 2. Pushgateway limitations relevant to this project.**

| Limitation | Consequence for this system |
|---|---|
| No concept of staleness; pushed metrics persist indefinitely until explicitly deleted | A failed run's metrics would remain visible and appear current indefinitely. |
| No `up` metric for pushers | The question "did the nightly ingest run at all?" — the single most important batch-job question — becomes unanswerable. |
| A single point of failure caching many subjects | One Pushgateway outage produces an alert storm across every dataset simultaneously. |
| Not an aggregator, but a metrics cache | Concurrent episode workers would overwrite one another's values rather than sum them. |
| Series removal is the operator's responsibility | Manual lifecycle management is required per dataset version, indefinitely. |

The documented alternative for jobs tied to a long-lived machine is the node_exporter textfile collector.

[RECOMMENDATION] The Pushgateway is rejected for this system. This is not a matter of preference; the absence of an `up` metric and the absence of any staleness concept are each independently disqualifying for a pipeline whose most important failure mode is simply not running at all.

### 4.2 OpenTelemetry for Batch Workloads — [STANDARD], With One Sharp Edge

Industry guidance converges on tracing a pipeline "as a single unit of work while capturing detailed metrics at each stage." One sharp edge in this guidance is worth quoting directly, because it applies to this system without modification: "ETL jobs are different from web services in that they have a defined start and end rather than running continuously. This means your last batch of spans and metrics might never get exported because the process exits before the telemetry providers have a chance to flush."

[RECOMMENDATION] Any exporter built for this system must flush synchronously on process exit, and this behavior must itself be covered by a fault-injection test. A telemetry layer that silently loses the final stage of every run is worse than having no telemetry at all, because it appears to function correctly while actually failing.

It is also [STANDARD] practice to separate pipelines by signal type — traces, metrics, and logs — using different batch sizes and timeouts for each, since their volume and latency requirements differ substantially.

### 4.3 High Cardinality — [DOCUMENTED] Mechanics, [VENDOR] Framing

ClickHouse's published analysis of this topic is commercially motivated, but its description of the underlying mechanism is accurate and consistent with the Prometheus source code. Prometheus creates a distinct time series for each unique combination of metric name and label values, and pays this cost at write time: the internal `memSeries` struct alone occupies approximately 200 bytes, before accounting for labels, chunks, or metadata. Under cardinality explosion, the in-memory head block can reach tens or hundreds of gigabytes of RAM, and short-lived series can linger in memory for hours after they stop receiving new data. ClickHouse's own worked example illustrates the scale of the problem: 1,000 hosts times 100 applications times 50 endpoints times 5 status codes produces 25 million distinct time series from only four label dimensions.

By contrast, ClickHouse defers this cost to query time, storing data cheaply in columnar format but consuming memory during `GROUP BY` aggregations over high-cardinality dimensions.

[INFERRED] The correct reading of this comparison is not that ClickHouse is simply superior to Prometheus. Rather, the two systems have opposite cost curves, and the resulting design question is which facts belong in which system — precisely the metrics/events split addressed in the project brief. This system's own cardinality figures are computed directly in [14-observability-architecture.md](14-observability-architecture.md) rather than assumed from any vendor's illustrative example.

## 5. Drift on Autocorrelated Data — [ACADEMIC], and the Real Trap

The moving block bootstrap (Künsch, 1989; Liu and Singh, 1992) exists precisely because resampling individual observations independently destroys temporal dependence in the data; contiguous blocks are resampled instead, which preserves the dependence structure within each block. Three results from this literature bear directly on this project. Effective sample size collapses under autocorrelation: with *q* dependent observations, the effective sample size is smaller than *q* by an amount governed by the correlation structure of the data. The block bootstrap achieves its nominal coverage only at larger effective sample sizes, and underperforms when effective sample size is small. Block-length selection is itself an open problem with a dedicated body of literature of its own.

This is why the trap matters concretely for this project. The ALOHA dataset records at 50 fps, so frame *t* and frame *t+1* of a fourteen-degree-of-freedom arm are nearly identical vectors. A two-sample Kolmogorov–Smirnov test run over 55,000 such frames has enormous nominal statistical power, and will return p < 0.001 for differences that carry no physical meaning whatsoever. A per-frame, independent-and-identically-distributed drift test on this kind of data is not a conservative choice; it is a false-positive generator, and it fails in exactly the way this system's `value_range` check originally failed — a statistically valid procedure applied at the wrong unit of analysis, firing on every episode regardless of whether anything meaningful had changed.

[RECOMMENDATION] The unit of analysis for drift in this system is the episode, not the frame. Each episode is reduced to a fixed-length feature vector first, and populations of these episode-level vectors are then compared. Fifty episodes of genuinely independent demonstrations constitute a more honest sample than 55,000 correlated frames drawn from those same fifty demonstrations, and this choice makes the resulting statistics defensible rather than merely computable.

## 6. Robotics-Specific Telemetry Practice

Several existing tools address robotics telemetry directly, and are [DOCUMENTED] for what they do. ROS 2 diagnostics is the established mechanism for hardware health monitoring, covering motor performance, battery degradation, and sensor status. `ros-opentelemetry` exists as a community library providing OpenTelemetry-based distributed tracing for ROS 2. ROS 2 quality-of-service profiles deliberately differentiate stream classes, such that command traffic, telemetry, and high-rate sensor streams are not treated identically — state estimation, mission status, and telemetry traffic may favor stronger delivery guarantees over raw throughput.

One directly relevant source, a ROSCon talk titled "What is Robotics Observability? Scaling ROS from Prototype to Production," could not be evaluated: its PDF is behind an access-control layer that could not be reached during this research, and this document accordingly does not characterize a talk that could not be read in full. [UNKNOWN]

[INFERRED] The important gap in existing robotics telemetry practice is that ROS 2 diagnostics observes the robot live, in the moment, and says nothing about a recorded dataset after the fact. A real and largely unaddressed space exists between "the robot's motor was running hot," which is fleet telemetry, and "this dataset is fit to train on," which is data health. This project occupies precisely that gap.

## 7. Data Observability's Five Pillars, and Where Robotics Breaks Them

[STANDARD] The canonical five pillars of data observability are freshness, volume, schema, distribution, and lineage. Data contracts are described in this literature as "a set of verifiable assertions applied to an individual data asset — schema checks, quality rules, and freshness SLAs." The consensus view in this literature holds that contracts prevent structural breaks while observability catches freshness, volume, and distribution drift; the two mechanisms are complementary, and neither substitutes for the other.

Table 3 maps these five pillars honestly onto the current state of this system.

**Table 3. The five canonical pillars, mapped onto this system.**

| Pillar | Current state in this system | Remaining gap |
|---|---|---|
| Schema | Strong — enforced by `structural.frame_schema`, the manifest, and the modality registry | No contract versioning between dataset versions |
| Volume | Partial — episode and frame counts are recorded in the catalog | No expectation, no trend, and no alert defined over these counts |
| Lineage | Strong — episode-level, code-fingerprinted, and queryable as a DAG | No time dimension |
| Distribution | Partial — covered by `RangeProfiler` and `numerical.episode_outlier` | Within-version only; no cross-version comparison |
| Freshness | Absent | No concept of data age or expected arrival time exists at all |

A sixth concern, absent from the canonical five pillars, is specifically required by robotics data: **temporal alignment**, the correctness of the relationship between modalities. This system's `inject_video_misalignment` fault shifts video references by five seconds while leaving the dataset structurally perfect: schema-valid, complete, non-null, and distributionally unremarkable by every one of the five canonical pillars. All five pillars pass under this fault, and the data is nonetheless ruined.

[RECOMMENDATION] Cross-modal alignment should be treated as a sixth pillar specific to multimodal robotics data, and it is the one general-purpose data-observability tooling is structurally unable to see. It is also, not coincidentally, the most defensible contribution this project makes.

## 8. Consequences of This Research for the Design

Table 4 summarizes how each finding in this document shaped a specific design decision in the resulting observability layer.

**Table 4. Research findings and their design consequences.**

| Finding | Design consequence |
|---|---|
| No primary autonomous-vehicle evidence exists | Justify design decisions from tool documentation and this system's own measurements; never from an unverifiable claim about what a specific company does |
| Foxglove indexes two clock domains | Treat clock skew as a continuously monitored quantity, not solely as a pass/fail check |
| Pushgateway has no `up` metric | Reject Pushgateway; "did it run?" must remain answerable |
| OpenTelemetry batch jobs can lose their final flush | The exporter must flush synchronously on exit, and this behavior must be fault-tested |
| Prometheus pays its cardinality cost at write time | Episode-grain facts must never become metric labels; compute cardinality figures directly rather than borrowing them |
| The block bootstrap and effective-sample-size collapse | Drift's unit of analysis is the episode, not the frame |
| Directed triage is the autonomous-vehicle industry's core discipline | Instrument the existing feedback flywheel as a first-class loop |
| The five canonical pillars miss cross-modal alignment | Add alignment as a sixth pillar; treat it as this project's differentiator |

## Sources

1. Foxglove. "Moving into a New Era of Robotics Tooling with Multimodal Data Observability." https://foxglove.dev/blog/moving-into-a-new-era-of-robotics-tooling
2. Foxglove. "Foxglove 2.0: Unifying Robotics Observability." https://foxglove.dev/blog/foxglove-2-0-unifying-robotics-observability
3. Foxglove. "The Agentic Data Platform for Physical AI." https://foxglove.dev/product
4. Prometheus. "When to Use the Pushgateway." https://prometheus.io/docs/practices/pushing/
5. ClickHouse. "ClickHouse vs Prometheus for High Cardinality, Part 1." https://clickhouse.com/blog/clickhouse-vs-promethous-high-cardinality-p1-understanding-the-problem
6. OneUptime. "How to Monitor Data Pipeline ETL Jobs with OpenTelemetry." https://oneuptime.com/blog/post/2026-02-06-monitor-data-pipeline-etl-jobs-opentelemetry/view
7. "The Impact of Bootstrap Methods on Time Series Analysis." Statistical Science. https://projecteuclid.org/journals/statistical-science/volume-18/issue-2/The-Impact-of-Bootstrap-Methods-on-Time-Series-Analysis/10.1214/ss/1063994977.pdf
8. "Block Length Selection in the Bootstrap for Time Series." https://www.sciencedirect.com/science/article/abs/pii/S0167947399000146
9. `ros-opentelemetry`. https://github.com/szobov/ros-opentelemetry
10. DataHub. "What Are Data Contracts?" https://datahub.com/blog/the-what-why-and-how-of-data-contracts/
11. Monte Carlo. "Data Contracts Explained." https://montecarlo.ai/blog-data-contracts-explained
12. SYNQ. "The Complete Guide to Data Quality." https://www.synq.io/blog/the-complete-guide-to-data-quality
13. iMerit. "Sensor Data Triage Strategies for Scalable Autonomous Vehicle Training." https://imerit.ai/resources/blog/sensor-data-triage-strategies-for-scalable-autonomous-vehicle-training/
