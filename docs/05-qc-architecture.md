# 05 — QC Architecture

## 1. Execution Model: One Pass, Many Checks

A naive validation framework allows each check to scan the frame table independently, incurring O(checks × frames) of I/O. With twenty-four checks, this amounts to twenty-four full scans of the dataset — tolerable at 19,000 frames, but ruinous at 50 million. The engine instead streams each episode once and fans it out to every episode-scope check registered with the engine.

Execution proceeds in three phases, described in Table 1.

**Table 1. QC engine execution phases.**

| Phase | Scope | Description |
|---|---|---|
| 1 | Pre-dataset checks (negative priority) | The range profiler learns per-dimension bounds and step thresholds, storing them in `ctx.shared` so that later checks have a basis for comparison. |
| 2 | Episode pass | One filtered scan per episode, reading only the union of columns the registered checks declare; every episode-scope check observes every episode. |
| 3 | Post-dataset checks | Checks over distributions, coverage, and cross-episode invariants. |

Checks declare their column requirements in two ways: `required_columns` for a static list, and `dynamic_columns(ctx)` for columns whose names depend on the dataset's modalities — for instance, `rgb.top__ts_in_blob` exists only when a top camera is present. The engine reads the union of all declared columns and nothing more, so that a timestamp-only check never causes state vectors to be decoded unnecessarily.

## 2. The Twenty-Four Checks

The checks are organized into five families, summarized in Table 2.

**Table 2. QC check families.**

| Family | Count | Question answered | Checks |
|---|---:|---|---|
| Structural | 6 | Does the data physically exist and hold together? | `frame_schema`, `blob_integrity`, `episode_references`, `episode_frame_count`, `frame_index_continuity`, `video_integrity` |
| Temporal | 6 | Is time well-behaved? | `monotonic`, `duplicates`, `gaps`, `sampling_interval`, `origin`, `duration_consistency` |
| Numerical | 6 | Are the numbers physically plausible? | `range_profile`, `finite`, `value_range`, `discontinuity`, `stuck_sensor`, `episode_outlier` |
| Dataset-level | 5 | Is the collection fit for use? | `length_distribution`, `task_distribution`, `modality_availability`, `missing_data`, `frame_count_consistency` |
| Cross-modal | 1 | Do the streams agree with one another? | `sync.video_alignment` |

By severity, the twenty-four checks comprise 2 critical, 10 error, 10 warning, and 2 informational checks.

## 3. Calibration: A Check That Fires on Everything Is Useless

Learned bounds are expressed as percentiles, and it follows as a tautology that roughly 1% of any dataset lies outside them by construction. This property is harmless at low dimensionality and becomes lethal at high dimensionality.

This effect was measured directly. The `value_range` check flagged 2 of 50 episodes on the six-degree-of-freedom SO-100 dataset. Applied to a fourteen-degree-of-freedom bimanual ALOHA dataset — 42 numeric dimensions, including long-tailed force/torque channels — the same check flagged all 50 episodes, producing 64 findings, because across 42 dimensions and roughly 1,100 frames per episode, some sample crosses some percentile boundary essentially always.

This was corrected in two independent ways.

**Materiality.** A finding is now raised only when a violation is either systematic (more than `min_violation_ratio` of an episode's samples lie out of bounds) or egregious (a single sample exceeds the bound by more than `far_excursion` times the dimension's span). A sample marginally over the line is not treated as a defect. A configured table of hard limits bypasses both conditions, since exceeding a mechanical limit is a defect from the first sample, however brief.

**Kind-aware bounds.** `RangeProfiler` widens its learned percentiles for `FORCE_TORQUE` and `TACTILE` channels, using p0.02/p99.98 rather than p0.5/p99.5. Contact signals are impulsive as a matter of physics: on the ALOHA data, the true maximum of a force channel is 2.4 times its 99.5th percentile. Judging such channels by positional percentiles calibrated for smooth signals reports ordinary contact events as anomalies.

Table 3 summarizes the effect of these two corrections.

**Table 3. Effect of materiality and kind-aware bounds on findings volume.**

| Configuration | Findings | Episodes flagged |
|---|---:|---|
| Before correction | 64 | 50 of 50 |
| After correction | 24 | 10 of 50 |
| SO-100 (regression check) | 9 → 9 | 2 → 2 (unchanged) |

The corrected configuration is more discriminating on the ALOHA data without any loss of sensitivity on the original SO-100 data. Four regression tests in `tests/test_validation_faults.py` pin both directions of this behavior.

## 4. Thresholds Are Learned, Not Hard-Coded

The check `numerical.range_profile` streams the frame table and derives per-dimension percentile bounds and step thresholds, which the checks `value_range` and `discontinuity` subsequently use. The engine must operate correctly on an SO-100 arm measured in degrees and an ALOHA arm measured in radians without any configuration change, so absolute limits would be incorrect as soon as a new robot type is introduced. A robot-specific table of hard limits may override the learned bounds where one exists.

The profiler accumulates only aggregate statistics over a bounded reservoir, so profiling a dataset larger than available RAM costs the same memory as profiling a small one.

## 5. Severity, Status, and Gate

Three distinct concepts govern how a finding is interpreted, and the engine keeps them deliberately separate.

1. **Severity** is a property of the check: how serious this class of finding is judged to be.
2. **Status** is the outcome of one particular evaluation: passed, failed, warned, or skipped.
3. **`QCGate`** is the policy that converts a set of findings into an episode-level status.

Separating checking from gating means that the same set of findings can produce both a strict training dataset and a permissive exploration dataset without re-running validation:

```python
QCGate(block_on=(Severity.ERROR, Severity.CRITICAL),
       max_warnings=3,              # escalate an accumulation to QUARANTINED
       advisory_checks=("numerical.episode_outlier",))
```

The status `QUARANTINED` exists because an episode with no single fatal finding but eight accumulated warnings deserves review by a person, rather than silent inclusion in the training set.

### 5.1 The Third Layer: Contracts

A gate decides about an episode. A **contract** decides about a version — whether it may be released to consumers at all. These are three different questions with three different owners, and collapsing them produces a system that is either unable to ship a dataset that is 99% fine, or unable to stop one that is subtly wrong. Table 4 illustrates the distinction with a concrete example at each level.

**Table 4. Check, gate, and contract, illustrated.**

| Layer | Example statement | Scope |
|---|---|---|
| Check | "Episode 17 has a 2.1 s timestamp gap." | An observation |
| Gate | "Episode 17 is rejected from `clean/v1`." | Per episode |
| Contract | "`training/v3` is blocked: action energy drifted 3.5× its calibrated null." | Per version |

`DataContract` (`rde/observability/contracts.py`) organizes its evaluation into five clause families: schema, temporal, quality, distribution, and release. Contracts are inferred from a version that has actually been inspected, never typed in from expectation:

```bash
$ rde obs contract so100_pickplace raw/v1 --candidate clean/v1
```

Three properties of the contract layer carry particular weight.

1. **`SKIP` is not `PASS`.** A distribution clause with no available baseline has not passed evaluation; it has not been evaluated at all. Rendering the two identically is a mechanism by which a contract quietly stops protecting anything.
2. **Alignment tolerance is expressed in frame periods, not seconds**, so that a single contract definition remains meaningful for both a 30 fps and a 50 fps dataset.
3. **Underpowered drift measurements warn but never block.** A statistic computed from three effective episodes must not be permitted to gate a release.

Further detail is given in [14-observability-architecture.md](14-observability-architecture.md).

## 6. Nothing Is Deleted

Every finding is persisted as a record carrying the fields `episode_id`, `frame_id`, `check_name`, `severity`, `observed_value`, `expected_value`, `status`, and `reason`. These records are written to Parquet, which is authoritative, and mirrored into SQLite for querying. Rejected episodes remain on disk; the clean version of the dataset simply does not select them. As a result, a question such as "why is episode 9 missing from `training/v1`?" is answerable as a query:

```bash
$ rde qc findings so100_pickplace raw/v1 --episode 9
```

A re-run of validation replaces prior findings rather than appending to them, so the record converges to the current state of the data instead of accumulating duplicate entries over time.

This convergence carries a cost that the observability work surfaced: re-validating a version erases its QC history. A question such as "did my check recalibration help, and where?" was previously unanswerable, because the pre-change findings no longer existed once the version was re-validated. The table `qc_executions` resolves this by scoping findings to a specific validation run rather than to a version alone, so that a history of validation headers accumulates even though the underlying findings continue to be replaced on each run:

```bash
$ rde obs quality so100_pickplace
```

## 7. Skipped Is Not Silent

A check that does not apply to a given dataset emits a `SKIPPED` record together with a reason. The question "was tactile sensing validated?" is thereby answered with "no, because the dataset has no tactile modality," rather than with ambiguous absence.

## 8. Proving the Checks Work

The reference dataset used in this project is a curated public release and is, by inspection, genuinely clean: across all twenty-four checks it produces zero blocking findings and nine informational or warning findings. This outcome is evidence about the quality of the dataset, not about the correctness of the engine.

The checks are instead validated by fault injection, implemented in `rde/testing/synthetic.py` and demonstrated in `examples/02_fault_injection_demo.py`. Each defect class is introduced into a disposable copy of the real, already-ingested dataset version, and the resulting findings are compared against expectation. Table 5 summarizes the injected defects and the checks that detect them.

**Table 5. Fault injection results.**

| Injected defect | Detected by | Additional findings | Severity |
|---|---|---:|---|
| NaN in state vector | `numerical.finite` | +2 | Error |
| 2 s timestamp gap | `temporal.gaps` | +4 | Error |
| Out-of-order timestamps | `temporal.monotonic` | +3 | Error |
| Duplicate timestamp | `temporal.monotonic` | +4 | Error |
| Stuck joint, whole episode | `numerical.stuck_sensor` | +2 | Warning |
| Impossible action jump | `numerical.value_range` | +2 | Warning |
| Three frames dropped mid-episode | `structural.episode_frame_count` | +6 | Error |
| Video references shifted 5 s | `sync.video_alignment` | +2 | Error |

Fourteen further tests in `tests/test_validation_faults.py` assert both directions of this relationship: that the correct check fires on the injected defect, and that a clean dataset produces no finding from it. A check that fires on everything is exactly as useless as one that fires on nothing.

## 9. The Defect That Only Pixels Reveal

The check `sync.video_alignment` is the most consequential check in the suite, because the defect it targets is invisible to every other check. If the video references are shifted by five seconds, the data remains structurally perfect: correct dtypes, correct row counts, no null values, monotonic timestamps, and plausible joint values. Every numeric and temporal check passes, and every frame nonetheless points at the wrong image.

The check assembles four pieces of evidence, ordered from cheapest to most expensive to compute.

1. **Window arithmetic.** The declared window duration must equal `frames/fps`.
2. **Stored-reference agreement.** The `<modality>__ts_in_blob` column, as actually written, must equal `episode.from_timestamp + frame.timestamp_s`, within half a frame period. This step matters more than it might appear to: this column, rather than the episode metadata, is what the model adapter dereferences at read time. A check that only re-derived the mapping from metadata would confirm the intended mapping while the stored data disagreed with it — a gap that was discovered by a failing test rather than anticipated in the design.
3. **Decodability at the stored positions**, rather than at the theoretically ideal positions.
4. **Distinctness.** The decoded first frame of an episode must differ from the frame that the previous episode's window maps to. This is the criterion that catches a constant offset, which the first three checks would pass without complaint.

Measured on the reference dataset, window arithmetic is exact to 0.0 ms across all 50 episodes, and cross-episode pixel distinctness ranges from 4.3 to 11.9 mean absolute RGB difference, well above the configured threshold of 1.0.

## 10. An Observer Riding the Pass

`EpisodeFeatureProfiler` is registered as a check that emits no findings. Its purpose is to reduce each episode to a fixed-length feature vector while the engine is already streaming that episode's frames for other checks; these features are subsequently used for episode-level drift monitoring.

This profiler is attached at run time only when a run is being observed, so that the validation suite otherwise remains twenty-four checks and the reported `checks_run` count does not inflate. Riding the existing pass, rather than adding a separate traversal of the data, is the central design choice here: a second scan of 55,000 frames purely to compute drift features would have cost an estimated 30–50%. Measured directly, the actual cost is 1.2%.

The same pass also accumulates per-modality sensor health statistics — non-finite value counts, rail-hit counts, and a running absolute maximum for the tail ratio — because the frames are already resident in memory during this pass, and discarding them only to re-read the data later would be wasteful.

## 11. Cost

**Table 6. QC engine cost summary.**

| Measurement | Value |
|---|---|
| 24 checks over 50 episodes / 19,631 frames | 2.7 s |
| Video integrity mode | Probes both ends of each episode by default; full decode is opt-in |
| Peak RSS during QC | Bounded by one episode's frames (approximately 400 rows) |

The flag `--deep-video` escalates video integrity checking to a full decode, when an integrity audit is judged worth the additional minutes it costs.

### 11.1 Where the 2.7 Seconds Actually Goes

Per-check timing was invisible until the observability layer added it; stage-level granularity previously stopped at the eight pipeline stages, leaving the dominant stage internally opaque. Table 7 shows the measured breakdown across both QC passes of one run.

**Table 7. Per-check timing breakdown.**

| Check | Seconds | Share |
|---|---:|---:|
| `sync.video_alignment` | 2.479 | 48.7% |
| `structural.video_integrity` | 2.449 | 48.1% |
| `observability.episode_features` | 0.059 | 1.2% |
| All 22 other checks, combined | 0.106 | 2.1% |

Two video-touching checks together account for 96.8% of total validation time, and they run adjacently over the same episode pass — a strong indication that they decode overlapping regions of video and could plausibly share a decoder. This is a concrete optimization opportunity that the engine could not identify before it gained the ability to time itself:

```bash
$ rde obs metrics --show | grep qc_check_duration
```

### 11.2 Not Re-Checking What Provably Has Not Changed

Versions are immutable and blobs are content-addressed, so re-validating an unchanged version with an unchanged suite of checks must produce the same findings. `QCCache` (`rde/qc/cache.py`) keys its cache on three fingerprints, described in Table 8.

**Table 8. `QCCache` fingerprint components.**

| Fingerprint | Covers | Rationale |
|---|---|---|
| Suite | Each check's name, its class source, and its configured parameters (`__dict__`) | When `value_range` was recalibrated by passing different constructor arguments (e.g. `min_violation_ratio=0.05`), its name and class source did not change; a fingerprint over name and source alone would have served the old bounds' findings, and the recalibration would have appeared to do nothing. |
| Tables | Size and modification time of every Parquet file | Hashing 470 MB of data to skip 2.7 s of computation would cost more than the computation itself. This check is a safety net against the immutability invariant being violated — for example, by a fault injector rewriting frames in place — rather than the primary correctness argument. |
| Blobs | Existence and size of referenced blobs | The fault injector `inject_missing_blob` deletes a video file without touching any Parquet byte; a table-only cache key would report the dataset healthy while its video was in fact missing. |

```bash
$ rde qc run so100_pickplace raw/v1 --incremental
$ rde qc cache so100_pickplace
```

Measured directly, incremental validation completes in 0.078 s against a cold run of 2.733 s, a 35× speedup. This behavior is opt-in, because a stale QC result is a correctness failure in precisely the component whose entire purpose is correctness; reusing a prior result should be a decision an operator makes deliberately, not a default the system inherits.
