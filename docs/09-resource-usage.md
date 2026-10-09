# 09 — Resource Usage

All figures in this document were measured on the development machine described in Table 1. They can be reproduced with `python examples/03_resource_report.py`.

**Table 1. Development machine specification.**

| Component | Specification |
|---|---|
| RAM | 10 GB total, approximately 8 GB usable |
| GPU | RTX 4060 Laptop, 8 GB VRAM (unused; the pipeline is CPU-only) |
| CPU | 16 cores |
| Python | 3.11.9 |

## 1. Headline Figures

Table 2 summarizes the resource profile of a full pipeline run.

**Table 2. Headline resource figures.**

| Metric | Value |
|---|---|
| Dataset size on disk | 470 MB |
| Estimated cost of naive full video decode | 18.1 GB |
| Actual peak RSS, full pipeline | 488 MB |
| Full pipeline wall-clock time | 7.0 s |
| Observability overhead, end to end | +0.8% |
| Incremental re-validation speedup | 35× (2.733 s → 0.078 s) |

The 18.1 GB figure is the central justification for the streaming design: it corresponds to 19,631 frames at 640×480×3 bytes, per camera. This exceeds the machine's available RAM by a factor of 2.3, so any approach that decodes an entire video into an array fails immediately on a dataset of only 470 MB, well before any consideration of terabyte-scale data.

## 2. Scaling Across Two Datasets

Table 3 compares resource usage across the two proof datasets used in this project.

**Table 3. Resource usage across datasets.**

| Property | SO-100 pick-place | ALOHA coffee |
|---|---:|---:|
| Source size | 470 MB | 1.63 GB |
| Episodes / frames | 50 / 19,631 | 50 / 55,000 |
| State and action dimensions | 6 + 6 | 14 + 14 (plus 14 force) |
| Cameras | 2 | 4 |
| Sampling rate | 30 fps | 50 fps |
| Wall-clock time | 7.0 s | 12.9 s |
| Peak RSS | 488 MB | 675 MB |
| Canonical frame table size | 903 KB | 5.5 MB |

Between the two datasets, frame count grew by a factor of 2.8 and dimensionality grew by a factor of 2.3, while peak RSS grew by only a factor of 1.26. This sublinear growth is direct evidence that the streaming design functions as intended: memory usage tracks batch size and the size of one episode, not the size of the dataset overall.

This result did not hold from the outset. The first run against the ALOHA dataset peaked at 1.58 GB, a regression traced to the nearest-neighbor policy's broadcast distance computation, which allocated 0.63 GB per block. The computation was subsequently rewritten using the Gram-matrix identity, described in [10-design-decisions.md](10-design-decisions.md#17-gram-matrix-distances-in-the-nearest-neighbor-policy). Table 4 shows the effect of this change.

**Table 4. Effect of the Gram-matrix rewrite on the nearest-neighbor evaluation stage.**

| Metric | Before | After |
|---|---:|---:|
| Peak RSS | 1.58 GB | 675 MB |
| Evaluate stage duration | 7.05 s | 1.19 s |
| Projected cost at 500,000 reference frames, per block | 7.2 GB | 0.26 GB |

## 3. The Memory Guard

The script `scripts/capped` runs a command inside a cgroup v2 scope with a hard `MemoryMax` limit:

```bash
scripts/capped rde run --config configs/so100_pickplace.json   # 6 GB default
RDE_MEM_MAX=3G scripts/capped python examples/03_resource_report.py
```

This guard exists because, although the engine is designed to stream data and remain memory-bounded, the nearest-neighbor regression described in Section 2 demonstrates that "designed to" and "does" are separate claims requiring independent verification. On WSL specifically, an out-of-memory condition can bring down the entire virtual machine rather than only the offending process. Under the memory cap, a runaway stage is instead terminated on its own with SIGKILL (exit code 137), and the host system survives. This behavior has been verified directly, by allocating memory past the configured limit and confirming that the process is killed as expected.

Every data-touching `make` target in this project uses this guard.

## 4. Stage Timings

Table 5 gives the measured duration of each pipeline stage.

**Table 5. Per-stage timing.**

| Stage | Seconds | Notes |
|---|---:|---|
| Download | 0.2 | Cached; approximately 60 s cold over the network |
| Ingest | 0.6 | 19,631 frames converted to canonical Parquet; 2 blobs hard-linked |
| Validate | 2.7 | 24 checks over 50 episodes |
| Transform | 0.1 | Fit and apply, streaming |
| Re-validate (`qc_post_transform`) | 2.5 | Second full check suite |
| Build | 0.02 | Metadata only; the frame table is referenced, not copied |
| Evaluate | 0.43 | Nearest-neighbor fit plus 1,908 test frames, using Gram-matrix distances |
| Feedback | 0.003 | — |
| **Total** | **7.0** | 488 MB peak RSS |

Per-stage volume and peak RSS are now recorded automatically rather than measured by hand, as shown in the following output:

```bash
$ rde obs stages so100_pickplace
stage              status  seconds  frames   episodes  rss_mb
download           ok        0.103       ·          ·    13.9
ingest             ok        0.334  19,631         50    59.0
validate           ok        2.687       ·         50   169.8
transform          ok        0.088  19,631→18,668   48    19.8
qc_post_transform  ok        2.656       ·         48    30.8
build              ok        0.024  18,668         48     0.0
evaluate           ok        0.421   1,908          ·    57.5
feedback           ok        0.006       ·          ·     0.0
```

The `transform` row is the reason the volume columns exist: it records 19,631 frames in and 18,668 out, with 48 episodes surviving. Under this instrumentation, a stage that consumed 19,631 frames and emitted only 12 would be immediately visible as broken, rather than passing silently.

### 4.1 Inside the Dominant Stage

Stage-level granularity stopped at the eight rows above until per-check timing was added, leaving the most expensive stage internally opaque. Table 6 shows the measured breakdown across both QC passes.

**Table 6. Per-check timing within validation.**

| Check | Seconds | Share |
|---|---:|---:|
| `sync.video_alignment` | 2.479 | 48.7% |
| `structural.video_integrity` | 2.449 | 48.1% |
| `observability.episode_features` | 0.059 | 1.2% |
| All 22 other checks, combined | 0.106 | 2.1% |

Two video-touching checks together account for 96.8% of validation time. Because they run adjacently within the same episode pass, this is strong evidence that they decode overlapping regions of video and could plausibly share a single decoder.

## 5. What Observability Costs

The governing constraint on the telemetry plane is that it must not become more expensive than the pipeline it observes. Table 7 shows median wall-clock time across three runs of the SO-100 dataset, with every telemetry component enabled: feature profiling, event rows, Parquet writes, spans, and the `.prom` export.

**Table 7. Observability overhead.**

| Configuration | Median wall-clock time |
|---|---:|
| `Pipeline(cfg, observability=False)` | 6.011 s |
| `Pipeline(cfg, observability=True)` | 6.061 s |
| Overhead | +0.050 s (+0.8%) |

This overhead is affordable because of where the instrumentation is placed. The feature profiler is implemented as a check that rides the episode pass already streaming every frame, and event rows are buffered in memory and written once per run rather than appended per episode. An implementation that instead added a second traversal of the data to compute drift features would have cost an estimated 30–50% and would have broken the bounded-memory guarantee.

Telemetry storage is likewise bounded: a real run produces 667 active metric series across 37 instruments, a bound that holds because unbounded identifiers are refused outright at the metric layer rather than merely discouraged by convention.

## 6. Not Redoing Provably Unchanged Work

Versions are immutable, so re-validating an unchanged version with an unchanged check suite is pure waste — and the `validate` and `qc_post_transform` stages together account for 86% of this pipeline's wall-clock time. Table 8 shows the measured effect of incremental validation.

**Table 8. Incremental validation speedup.**

| Condition | Seconds |
|---|---:|
| Cold validation | 2.733 |
| Warm, unchanged version | 0.078 |
| Speedup | 35× |

Cache invalidation is exact rather than optimistic. An in-place NaN injection correctly invalidates the cache, with findings increasing from 9 to 10; deleting a referenced blob likewise invalidates the cache, even though no byte of the Parquet table changed, with findings increasing from 10 to 11.

## 7. Video: What Tiering Saves

Table 9 summarizes the cost of the tiered approach to video health checking.

**Table 9. Video tiering cost.**

| Metric | Value |
|---|---:|
| SO-100 video, stored | 470 MB |
| SO-100 video, fully decoded | 36.2 GB (77×) |
| Tier 0 (container headers only) | 1.9 ms per file |
| Tier 1 (5 sampled frames per clip) | 89 ms per episode-camera |
| Tier 1 peak memory | One frame, approximately 0.9 MB |
| Tier 1, across the whole version | Approximately 9 s |

Tier 0 already detects a resolution change, a codec swap, a truncated file, and a frame-count disagreement — the class of failures a changed recording rig produces, which tend to affect whole cohorts of episodes at once. Frames are sampled at fixed fractions of a clip's duration, so the cost of Tier 1 is set by the sample count rather than by clip length: a 90-second episode costs the same to check as a 12-second one, which is the property that makes this approach affordable at petabyte scale.

```bash
$ rde obs video so100_pickplace raw/v1 --tier 0
$ rde obs video so100_pickplace raw/v1 --tier 1
```

## 8. Storage

Table 10 breaks down the components of on-disk storage for the primary dataset.

**Table 10. Storage breakdown.**

| Component | Size |
|---|---:|
| Source cache (verbatim upstream) | 470.1 MB |
| Blobs, logical | 469.5 MB |
| Tables (all 4 versions) | 1.9 MB |
| Artifacts (manifests, QC, evaluation, feedback) | 98.9 KB |
| `catalog.db` | 262 KB |
| Data root, hard links deduplicated | 472.4 MB |
| Data root, every link counted independently | 941.9 MB |
| Saved by content-addressed hard linking | 469.5 MB |

Table 11 gives the tabular storage cost of each individual version.

**Table 11. Per-version tabular storage.**

| Version | Episodes | Frames | Frame table | Episode table | Index |
|---|---:|---:|---|---|---|
| `raw/v1` | 50 | 19,631 | 903 KB (20 row groups) | 8.2 KB | — |
| `clean/v1` | 48 | 18,668 | 955 KB (20 row groups) | 8.5 KB | — |
| `training/v1` | 48 | 18,668 | Referenced from `clean/v1` | 8.7 KB | 3.5 KB |

A training version costs 12.2 KB. The entire four-version chain adds 1.9 MB of tabular data on top of the 470 MB of shared video.

## 9. Video Decode Performance

Table 12 gives measured video decode performance.

**Table 12. Video decode performance.**

| Metric | Value |
|---|---|
| Container | 320.4 MB / 149.2 MB, AV1 (`libdav1d`), 640×480, 19,631 frames, 654.4 s |
| Random access (seek and decode) | 4.75 ms per frame |
| Sequential access (`iter_range`) | 1.67 ms per frame |
| Frame cache | 64 frames, approximately 59 MB, LRU, bounded |

Sequential access is approximately 2.8 times faster per frame than random access, which is why the API exposes both, with its documentation stating when each should be used: `read_at` for sampling and QC spot checks, and `iter_range` whenever a whole episode is being exported.

## 10. Read Paths

Table 13 gives the measured cost of common read patterns.

**Table 13. Read-path performance.**

| Operation | Result |
|---|---:|
| Stream all 14,919 train frames | 0.012 s |
| 500 samples, no image decode | 0.005 s |
| 100 samples with `rgb.top` decode | 0.709 s (7.09 ms per sample) |
| Peak RSS during a frame scan | 222.6 MB |

Lazy image loading is worth approximately a factor of 1,400 on the sample path when images are not required by the consumer.

## 11. How the Resource Budget Is Held

The resource budget stated above is achieved through the following seven design choices.

1. **Streaming everywhere.** Data flows from adapter to Arrow batch to Parquet writer; validation reads one episode at a time; transformation applies one batch at a time.
2. **Column projection.** Each check declares the columns it requires, and the engine reads the union of these declarations and nothing more.
3. **Bounded row groups.** Row groups of 2,048 rows allow predicate pushdown to skip irrelevant data rather than decoding an entire table.
4. **Frame-accurate video seeking**, used in place of full video decode.
5. **Bounded caches.** The video frame cache is an LRU cache bounded in frame count, and the profiler's reservoir is capped in size.
6. **Hard-linked blobs**, used in place of file copies.
7. **Reference rather than duplication.** Training versions point at their source version's frame table rather than copying it.
8. **Deferred image operations.** Resizing is applied at read time rather than by re-encoding stored video.

## 12. Configuration

The following environment variables adjust the resource/performance trade-off:

```bash
RDE_DATA_ROOT=/mnt/big/rde        # relocate all data
RDE_ROW_GROUP_SIZE=4096           # larger groups, fewer seeks
RDE_BATCH_SIZE=512                # smaller batches, lower peak RSS
```

## 13. Honest Limits

The following limitations of the current implementation are stated explicitly.

1. **Single-process execution.** Sixteen cores are available on the development machine, and only one is used. Episode-level parallelism is structurally available, since episodes are independent of one another, but it is not implemented.
2. **The GPU is unused.** The 8 GB of available VRAM performs no work in this system; the pipeline is I/O- and decode-bound, and the reference models are CPU-only by design.
3. **The 488 MB peak RSS figure is dominated by library imports, not data.** The libraries `pyarrow`, `duckdb`, `av`, and `numpy` together account for most of this figure. The data-dependent component of memory usage is small and bounded, which is the property that matters for scaling behavior.
4. **The telemetry backend is a file.** The `.prom` export is real and parseable, but no Prometheus instance has scraped it in this project, so none of the alert expressions described in [15-metrics-catalog.md](15-metrics-catalog.md), Section 6, has been validated end to end.
5. **Cardinality projections are computed, not observed.** The figure of 167 TB at 344 million episodes per year is derived from a measured per-series cost multiplied by a computed series count; no time-series database was actually run at that scale.
6. **The system has not been tested beyond 1.63 GB.** The design targets datasets larger than available RAM, and every access path is bounded by construction, but the largest dataset actually run through the system is the 1.63 GB ALOHA dataset. LeRobot datasets of 14 GB and 46 GB would constitute the real test of this design, and two of them (`behavior1k` and `metaworld_mt50`) would additionally exercise code paths this engine has not yet encountered: multi-file cameras, and images stored inside Parquet rather than as video.
