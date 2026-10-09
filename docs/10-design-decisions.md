# 10 — Design Decisions and Trade-offs

This document catalogs the significant design decisions made in this system. Each entry states the decision, the alternative that was considered, the reasoning behind the choice, and the cost the choice imposes.

## Part I — The Engine

### 1. Three Storage Planes, Not One Database

The system stores content-addressed blobs, tabular data in Parquet, and metadata in SQLite as three separate planes, rather than storing everything in a single database such as Postgres or a single format such as Parquet.

Combining these planes fails in both directions. Placing mutable QC status inside immutable Parquet would require rewriting files merely to mark one episode as rejected, while placing 19,631 frame rows inside SQLite would discard columnar scanning and predicate pushdown.

This decision costs three separate consistency domains rather than one. The catalog can drift from the state of the disk, which is why every version directory also carries a self-describing `manifest.json`, from which the catalog can be rebuilt if necessary.

### 2. Blobs Shared Across Versions; Only Metadata Materialized

Video is stored once, addressed by content hash, and derived versions reference it rather than copying it. Training versions go further still, referencing their source version's frame table as well. The alternative considered was to have each version own a complete, independent copy of its data.

Copying video at every stage would turn this five-stage pipeline into 2.3 GB of identical bytes on the primary dataset alone. The measured saving from sharing is 470 MB (472 MB of physical storage against 942 MB of apparent storage). At the terabyte scale this design targets, copying would be disqualifying on cost alone.

This decision costs self-containment: a version is not independently complete, and deleting `clean/v1`'s frame table would break `training/v1`. Proper garbage collection would require reference counting, which is not implemented. This is an acceptable gap for a proof of concept in which nothing is deleted, but it is the first addition a production deployment would need.

### 3. Hard Links Into the Source Cache

Ingestion uses `os.link`, with a copy fallback when the source and destination reside on different filesystems, rather than always copying data into the store.

The source cache already holds its bytes immutably, so copying would double 470 MB of storage today, and terabytes of storage in the future, for no benefit.

This decision costs a filesystem constraint: hard links require the source and destination to reside on the same filesystem, and a caller who mutated the source cache directly would corrupt the blob store. This risk is mitigated by treating the source cache as read-only in practice and by the `structural.blob_integrity` check, which rehashes stored content on demand.

### 4. Image Resize Deferred, Not Materialized

The `ResizeImages` transform records its intent in the manifest, and the model adapter applies the resize at decode time, rather than re-encoding video at the target resolution immediately.

Materializing a resize immediately would cost minutes of CPU time and hundreds of megabytes of storage to save microseconds per frame at read time, and it would bake a single resolution into storage permanently — so that a second model requiring 256×256 images would need a second copy of the dataset. Lineage records the resize operator identically regardless of when it is materialized.

This decision costs read-time CPU on every training epoch (7.09 ms per sample when decoding is required). For a training run that reads the same frames hundreds of times, a materialized cache would outperform this approach. The preferable long-term design is likely to support both strategies, with the manifest recording which is in effect; this is not yet implemented.

### 5. Splits by Episode, Assigned by Rank of Hash

Dataset splits are assigned at the episode level, using `quantile` mode by default, rather than at the frame level, by random shuffling, or by independent per-episode hashing.

Frame-level splitting is among the most common ways a robotics benchmark silently misrepresents its own results: consecutive frames differ by only one control step, so a frame-level split places near-duplicate frames on both sides of the split boundary and ends up measuring memorization rather than generalization.

Between the two episode-level hashing strategies considered, a genuine trade-off exists, summarized in Table 1.

**Table 1. Episode-level split modes compared.**

| Mode | Produces exact split sizes | Stable as the dataset grows |
|---|:---:|:---:|
| `quantile` (default) | Yes¹ | No — adding episodes can reassign existing ones |
| `independent` | No | Yes — an episode's split assignment never changes |

At 48 episodes, independent hashing produced a 42/3/3 split rather than the intended 38/5/5. A three-episode validation set is too small to be statistically meaningful, so exactness is the priority at this scale. Past a few hundred episodes, `independent` mode would be the preferable default, and the configuration flag to select it already exists.

¹ Exact only when `group_by` is not used. `group_by` keeps correlated episodes (e.g. the same recording session) on one side of a split, which `quantile` mode enforces by ranking and cutting whole groups rather than individual episodes — a group larger than a split's remaining budget pushes that split over its target, so leakage-safety and exact sizing are not simultaneously guaranteed once `group_by` is set.

### 6. Thresholds Learned From the Data, Not Configured

A profiler check derives per-dimension percentile bounds and step thresholds directly from the data and stores them for later checks to use, rather than relying on a robot-specific limits file supplied by the operator.

The engine must operate correctly on an SO-100 arm measured in degrees and an ALOHA arm measured in radians without any configuration change, and absolute limits would be incorrect as soon as a new robot type is introduced.

This decision costs specificity: learned bounds describe the dataset at hand, so a systematically miscalibrated dataset defines its own errors as normal. The profiler cannot, for example, detect that every joint reads ten degrees too high across an entire dataset. A hard-limits table is supported as an explicit override for this reason and should be supplied whenever a robot's true specification is known.

### 7. Checking Separated From Gating

Checks report findings, and a separate `QCGate` component decides what those findings mean for an episode's status, rather than having each check return a pass/fail verdict directly.

As a consequence, the same set of findings can produce both a strict training dataset and a permissive exploration dataset without re-running validation — on a large dataset, this is the difference between a policy change costing seconds and costing hours.

This decision costs one additional concept for a new user of the system to learn. The trade-off is judged worthwhile.

### 8. Nothing Is Deleted; Rejection Is a Marking

Rejected and quarantined episodes remain on disk together with the findings that caused their status, rather than being filtered out destructively.

The question "why is episode 9 missing from `training/v1`?" must remain answerable months after the fact, and destructive filtering would make it unanswerable.

This decision costs monotonically growing storage, and the catalog accumulates findings over time. This cost is negligible at the scale of this project, but at terabyte scale it would require a retention policy, which is not yet implemented.

### 9. One Streaming Pass, Many Checks

The engine streams each episode once and fans it out to every registered check, rather than allowing each check to scan the data independently.

Independent scans are O(checks × frames) — twenty-four full scans of the dataset in this system — which is tolerable at 19,000 frames and ruinous at 50 million.

This decision costs flexibility: checks cannot choose their own access pattern, and a check requiring an unusual column projection must declare it explicitly through `dynamic_columns`. This mechanism exists because a test caught the alignment check silently reading columns that the engine was not otherwise projecting.

### 10. Arrow Batches on the Bulk Path, Pydantic at the Boundary

Ingestion adapters yield Arrow `RecordBatch` objects, with the `Frame` type reserved for single-sample access, rather than validating every frame through a Pydantic model.

Constructing 19,631 Pydantic model instances merely to serialize them immediately back to Parquet would cost far more than the validation returns in value.

This decision costs per-row type validation on the bulk path. This gap is mitigated by fixed-size list types, which convert a dimension mismatch into a write-time error, and by the structural schema check, which validates the resulting table.

### 11. Confidence Is Mandatory on `Prediction`

Every model adapter is required to express a confidence value with each prediction, rather than treating confidence as an optional field.

The feedback loop selects low-confidence examples for review, so a model that cannot express uncertainty cannot participate in the feedback flywheel. The measured confidence/error correlation for the nearest-neighbor model is −0.79.

This decision costs an obligation on every model integration: models without a native notion of uncertainty must produce some confidence value. `SmolVLAPolicy` addresses this by returning a constant value and stating this fact explicitly in `metadata.confidence_note`, rather than emitting a plausible-looking number the feedback loop would then trust. An honest constant is judged preferable to dishonest calibration.

### 12. A Retrieval Policy as the Reference Model

The reference model is a nearest-neighbor retrieval policy over state, evaluated alongside a mean-action baseline, rather than a small trained neural policy or SmolVLA alone.

The model is explicitly not the focus of this project, but a genuinely working model demonstrates the model-integration interface more convincingly than a stub implementation would. The nearest-neighbor model runs on CPU alone in approximately one second and reaches 2.6% of the baseline's MSE. The mean-action baseline is not decorative: an absolute MSE computed on normalized actions is close to meaningless on its own, and it is easy to report an impressive-looking number that the mean baseline also achieves.

This decision costs generality: the nearest-neighbor model performs well specifically because the data is single-task with heavily overlapping trajectories, and it would not generalize to more complex tasks. Its result says nothing about how a genuine vision-language-action model would perform, and it is not intended to.

### 13. Optional Heavy Dependencies

`torch` and `lerobot` are placed behind an optional `vla` extra and imported only inside the `load()` method of the SmolVLA adapter, rather than being required dependencies of the package.

As a result, the data engine and its 180 tests run in seconds without requiring a multi-gigabyte dependency installation. Requiring `torch` merely to validate a Parquet file would amount to an architectural admission that the data layer depends on the model layer — precisely the coupling this design exists to avoid.

This decision costs test coverage: SmolVLA itself is not exercised in continuous integration. The test `test_smolvla_reports_missing_dependency_clearly` at minimum pins the failure mode when the dependency is absent.

### 14. Fault Injection as the QC Proof

The QC engine's correctness is demonstrated through a synthetic dataset generator together with a set of `inject_*` helper functions that introduce specific defects, rather than by trusting that the checks work because they execute without error.

The reference dataset used in this project is genuinely clean, so a passing QC run on it proves nothing about the correctness of the engine itself. Deliberately introducing defects, one class at a time, is the only way to demonstrate both that a check fires on the defect it claims to detect, and that it remains silent otherwise.

This decision costs ongoing maintenance of the fault-injection generator as real code, and the injected faults are synthetic — they may not resemble every way a real recording system actually fails in the field.

### 15. Range Findings Require Materiality, Not a Bound Crossing

A learned-bound violation is reported only when it is systematic or egregious; configured hard limits, by contrast, fire at the first sample that exceeds them. The alternative considered was to report any value outside a learned bound as a finding.

Percentile bounds are exceeded by approximately 1% of the data by definition. At 42 dimensions, this means essentially every episode contains some crossing, and an unmediated check degrades into noise — measured directly, this configuration flagged 50 of 50 episodes on the ALOHA dataset.

This decision costs sensitivity to a genuine but brief single-sample excursion that falls between the two materiality thresholds. This is judged the correct trade-off for a learned bound. Where real physical limits are known, the hard-limits path deliberately has no materiality test at all, because a mechanical limit breach matters even for a single sample.

### 16. Kind-Aware Bounds for Impulsive Signals

Force/torque and tactile channels are profiled at p0.02/p99.98 rather than the p0.5/p99.5 used for other modalities. The alternative considered was a single percentile pair applied uniformly across every modality.

Contact forces spike as a matter of physics. The true maximum of a force channel in the ALOHA dataset is 2.4 times its 99.5th percentile, so positional percentiles calibrated for smooth signals report ordinary contact events as anomalies. Placing this distinction inside the profiler keeps it out of every check that subsequently consumes the profiler's output.

This decision costs precision: a per-kind percentile pair is a crude proxy for "this signal is heavy-tailed." A distribution-shape test would be more principled than a lookup keyed on modality kind, but such a test would also be untestable against the only two datasets available in this project, so the simpler rule, justified by physical reasoning, is preferred for now.

### 17. Gram-Matrix Distances in the Nearest-Neighbor Policy

Pairwise distances in the nearest-neighbor policy are computed using the identity `||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b`, rather than the more directly readable broadcast expression `norm(refs[None] - q[:, None], axis=2)`.

The broadcast expression allocates an intermediate array of shape `(queries, refs, dims)`, which costs 0.63 GB per block at 55,000 reference frames and a projected 7.2 GB at 500,000 reference frames — a failure mode that does not degrade gracefully on WSL. The Gram-matrix expansion requires only an array of shape `(queries, refs)`, costing 22 MB and a projected 0.26 GB for the same two cases respectively, independent of the number of dimensions. Measured directly, this change reduced peak RSS from 1.58 GB to 675 MB and evaluation time from 7.05 s to 1.19 s.

This decision costs code clarity, and the expansion is numerically less stable than a direct difference for points that are very close together. This is mitigated by clipping small negative values before taking the square root, and the surrounding code carries a comment explaining the trade-off.

### 18. A Hard Memory Ceiling Around Data-Touching Commands

The script `scripts/capped` wraps data-touching commands in a cgroup v2 scope with a configured `MemoryMax`, rather than trusting the streaming design to hold under all conditions.

Every access path in the engine is bounded by design, but the nearest-neighbor regression described in decision 17 is direct proof that "designed to be bounded" and "is bounded" are different claims. On WSL specifically, an out-of-memory condition can bring down the entire virtual machine rather than only the offending process, so the cost of being wrong about a bound is losing the entire working environment.

This decision costs portability: the mechanism is Linux/systemd-specific, and the script degrades to running commands uncapped elsewhere, with a warning. It also has the effect of masking memory leaks by killing them before they can be fully characterized — the cap is a safety net, not a substitute for the resource measurements in [09-resource-usage.md](09-resource-usage.md).

## Part II — The Observability Layer

Decisions 19 through 24 concern the telemetry plane, introduced in [12-observability-recon.md](12-observability-recon.md) through [16-observability-report.md](16-observability-report.md). Each entry below is written to stand alone, though the supporting evidence for its measurements is presented in full in those documents.

### 19. The Metric Label Policy Raises Rather Than Documents

`LabelPolicy.validate` raises a `LabelPolicyError` when an unbounded identifier is used as a metric label, rather than relying on a naming convention enforced through code review.

This rule is judged too consequential to leave to discipline alone. Attaching `episode_id` to a single QC metric produces 264 billion time series at the system's planning scale — a measured 167 TB of resident memory, against 6.02 GB of columnar rows holding the identical facts. A convention catches this mistake in code review most of the time; a runtime exception catches it every time, at the call site, with the offending count stated in the error message.

The distinction the policy enforces is not "how many values exist today" but "what causes a new value to appear." The label `check_name` is bounded because there are twenty-four checks, and there are twenty-four checks because twenty-four were written by hand. The label `episode_id` is unbounded because a new value appears every time any robot records a new demonstration.

This decision costs friction: a legitimately new label dimension requires an explicit edit to `SAFE_LABELS` rather than simply being used at the call site. This friction is the intended effect of the policy, not an accidental cost.

### 20. Drift Is Compared Per Episode, Never Per Frame

Drift monitoring reduces each episode to a fixed-length feature vector and compares populations of these episode-level vectors, rather than running two-sample tests directly over the frame-level distribution.

The ALOHA dataset records at 50 fps, so frame *t* and frame *t+1* of a fourteen-degree-of-freedom arm are nearly identical vectors. A statistical test run over 55,000 such frames has enormous nominal statistical power and returns p < 0.001 for differences that carry no physical meaning. This is the same underlying error that decision 15 addresses in a different context — a statistically valid procedure applied at the wrong unit of analysis — and left uncorrected here, it would have produced a drift detector that fired constantly.

Fifty independent demonstrations constitute a more honest sample than 55,000 correlated frames drawn from those same fifty demonstrations. The effective sample size is published beside every drift statistic the system reports, and a comparison judged underpowered issues a warning rather than a block.

This decision costs granularity: episode-level features are lossy by construction, and a defect confined to a small number of frames within an otherwise normal episode will not move them. This gap is precisely what frame-level QC checks are designed to catch, so the two mechanisms are complementary rather than redundant.

### 21. Drift Thresholds Are Measured, Not Borrowed

`calibrate_threshold` estimates a drift statistic's null distribution by repeatedly splitting the baseline dataset and uses the 95th percentile of that null distribution as its threshold, rather than adopting the conventional PSI thresholds of 0.1 and 0.25 used in industry practice.

Table 2 shows the result of 200 null comparisons — comparisons of a distribution against itself — at a sample size of 50 episodes with 10 fixed histogram bins.

**Table 2. Measured null-distribution behavior of PSI at episode scale.**

| Sample size (episodes) | Median null PSI | P(PSI > 0.25) |
|---:|---:|---:|
| 50 | 0.301 | 62.5% |
| 5,000 | 0.003 | 0.0% |

The conventional threshold of 0.25 originates in credit-risk modeling practice, where sample sizes typically run to tens of thousands. At episode scale, this threshold fires on 62.5% of comparisons of a distribution against itself, and the median null value already sits above the conventional "significant drift" line. An alert built on this threshold would be noise that resembles a working detector until someone investigates a specific alert.

The calibrated threshold instead achieves a measured 0.5% false-positive rate and 94–98% detection power at a one-standard-deviation shift.

This decision costs computation: calibration requires 200 resamples per feature, and the resulting threshold describes a specific feature at a specific sample size — it must be recomputed whenever either changes.

### 22. Incremental Validation Is Opt-In

Incremental validation is enabled explicitly via `QCEngine(incremental=True)` and is off by default, rather than being enabled automatically on the grounds that the underlying correctness argument is sound.

The correctness argument genuinely is sound: versions are immutable, blobs are content-addressed, and the cache key covers the check suite's source code as well as both data planes. Nonetheless, a stale QC result constitutes a correctness failure in precisely the component whose entire purpose is correctness, and the consequence of such a failure is a dataset shipped on findings that no longer describe it. Reusing a cached result should be a decision an operator makes deliberately, not a default the system inherits silently.

This decision costs the 35× speedup measured in [09-resource-usage.md](09-resource-usage.md) on any run that does not explicitly request incremental validation, which is to say, most runs.

### 23. Video Health Is Tiered, and Tier 3 Is Deliberately Absent

Video health checking proceeds through container-header inspection (1.9 ms), then five sampled frames per clip (89 ms), then perceptual hashing, and stops there. The alternative considered was to add a further tier using vision embeddings over sampled frames, semantic clustering, or inspection by a vision-language model.

The SO-100 dataset is 470 MB stored and 36.2 GB when fully decoded. At a projected fleet scale of 3–10 PB per year, any strategy that decodes video in full costs more than the model training it is meant to support. Tier 0 alone already catches the failure modes a changed recording rig produces — resolution changes, codec swaps, truncated files, and frame-count disagreements — and these failures tend to affect whole cohorts of episodes simultaneously.

A further tier would require a model, a dedicated GPU budget, and its own sampling policy, and every question this project actually asks of video health is answered by the tiers already implemented. This boundary is documented explicitly so that it reads as a considered decision rather than an unnoticed omission.

This decision costs semantic coverage: none of the implemented tiers detects a semantic change such as a new object in frame, a rearranged scene, or a different lighting setup — precisely the class of change a further tier would be built to catch.

### 24. Telemetry Failures Are Swallowed

Every telemetry exporter and metric call catches its own exceptions internally and returns, rather than allowing such exceptions to propagate like every other error in the engine.

This behavior inverts the engine's usual stance, in which a crashing QC check becomes a `CRITICAL` finding rather than a silent pass. The asymmetry is intentional: a QC failure is a result worth recording, whereas a telemetry failure is not a result at all. An ingestion run that terminates because a metrics collector happens to be unreachable would have converted an observability outage into a data outage.

The telemetry layer instead observes its own failures, through the counter `rde_observability_export_failures_total`, so that this swallowing behavior remains visible rather than silent.

This decision costs quiet degradation: a systematically broken exporter fails without raising an alarm on its own. This is mitigated by a freshness signal — a missing or stale `.prom` file is itself treated as an alert condition.

## Part III — What Would Be Changed Next

The following eleven items represent the most significant gaps between this proof of concept and a production-ready system, in no particular order of priority.

1. **Reference counting and garbage collection for blobs.** Sharing blobs without reference counting means that nothing can ever be safely deleted.
2. **Episode-parallel processing.** Sixteen cores are available and only one is used; episodes are independent by construction, so this improvement is a scheduler away from being realized.
3. **Partitioned frame tables.** A single Parquet file per version is adequate to roughly 10⁷ frames; beyond that scale, partitioning by episode range would be required to support parallel writers.
4. **A materialized-image cache**, offered alongside deferred resize and selected per dataset, with the choice recorded in the manifest.
5. **A real robot-limits registry**, so that numerical checks can use absolute bounds where a robot's specification is known, rather than always relying on learned bounds.
6. **A PostgreSQL catalog**, appropriate past approximately 10⁶ episodes; the storage interface already isolates the catalog implementation from its callers.
7. **Point-cloud and tactile adapters.** The modality kinds and storage planes for these already exist in the schema, but nothing currently exercises them — extensibility along these dimensions is claimed by design but not yet proven in practice.
8. **A real metrics backend.** The `.prom` export is correct and well-formed but has never been scraped; no alert expression described in [15-metrics-catalog.md](15-metrics-catalog.md) has been validated against a live time-series database.
9. **Cohort labels at cardinality.** The labels `factory_id` and `robot_id` are accepted by the label policy but are exercised by nothing, since this project's dataset has exactly one of each — and these are precisely the dimensions that would otherwise allow a healthy global average to conceal a single broken site.
10. **Appearance drift.** Tier 2 perceptual hashes are computed but not yet fed into the `compare()` drift function, so drift monitoring today is limited to trajectory features alone.
11. **Contract approval.** A contract inferred from `raw/v1` correctly rejects `clean/v1` because a transform resized images — a genuine and intended schema change. No mechanism yet exists to express "this specific change was reviewed and approved."
