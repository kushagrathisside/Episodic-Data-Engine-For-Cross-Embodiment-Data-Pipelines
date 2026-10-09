# 06 — Versioning and Lineage

## 1. Versions

A version is named `<stage>/v<N>` — for example, `raw/v1`, `validated/v1`, `clean/v1`, `transformed/v1`, `training/v1`, or `training/v2`. A version's identity is the pair `dataset_id@version`.

Every version is immutable. With the exception of an ingested root version, every version is produced by exactly one recorded `TransformRun`, which is what makes provenance a directed acyclic graph rather than an informally maintained record.

## 2. The Two Questions This Design Answers

The lineage design exists to make two specific questions answerable, each in a single call (`examples/05_lineage_queries.py`).

### 2.1 "Which raw episodes contributed to this training dataset?"

A representative query returns the following result.

```
raw episodes        : 50
contributing        : 48
excluded            : [9, 31]

episode 9 was excluded because:
   [warning] numerical.value_range
      episode 9: action.joint exits learned bounds in dimension(s) [3]
episode 31 was excluded because:
   [warning] numerical.episode_outlier
      episode 31: 575 frames is an outlier (median 380, MAD 22)
```

Dataset-level edges alone cannot answer this question, because a clean version drops individual episodes: the statement "`clean/v1` came from `raw/v1`" is not sufficiently precise. For this reason, `TransformRun` carries the set of surviving episode identifiers, and the catalog maintains a dedicated `episode_lineage` table that links each episode across versions, as illustrated below for two episodes.

| Training episode | Derived from (clean) | Derived from (raw) |
|---|---|---|
| `training/v1#0` | `clean/v1#0` | `raw/v1#0` |
| `training/v1#1` | `clean/v1#1` | `raw/v1#1` |

### 2.2 "What was applied before this reached the model?"

A representative query returns a step-by-step record of every stage applied to the data, together with the QC findings recorded at each stage.

```
step 1: ingest              (source) -> raw/v1        episodes 0 -> 50
   software=0.1.0  code=d58f9de310837848  fingerprint=be45fb09ed2db70c
   params: source_format=lerobot/v3.0, row_group_size=2048, compression=zstd

step 2: transform           ['raw/v1'] -> clean/v1    episodes 50 -> 48, dropped [9, 31]
   software=0.1.0  code=700ae571e4c55400  fingerprint=313d63120127b3a0
      - episodes.quality_filter   (materialised)
      - timestamp.normalize       (materialised)
      - normalize.state.joint     (materialised)
      - normalize.action.joint    (materialised)
      - images.resize             (deferred)

step 3: build_training_dataset ['clean/v1'] -> training/v1  episodes 48 -> 48
   software=0.1.0  code=f0a5313c032e2107  fingerprint=8ec7ebf6ec283417
   params: splits={train:0.8,val:0.1,test:0.1}, seed=42, split_mode=quantile

QC checks applied:
   raw/v1     6 checks recorded findings, 9 findings total
   clean/v1   5 checks recorded findings, 6 findings total
```

## 3. What Makes a Run Reproducible

Each `TransformRun` records the fields listed in Table 1.

**Table 1. `TransformRun` fields and their purpose.**

| Field | Purpose |
|---|---|
| `operation`, `input_versions`, `output_version` | Defines the lineage edge |
| `parameters` | Every operator and its arguments, including fitted normalization statistics |
| `software_version` | The package version at the time of the run |
| `code_fingerprint` | The SHA-256 hash of the operator's source code |
| `fingerprint()` | A stable hash of operation, inputs, parameters, and code, taken together |
| `input`/`output`/`dropped_episode_ids` | Episode-level provenance |
| `metrics`, timings | A record of what actually happened during the run |
| `pipeline_run_id` | Identifies which pipeline execution produced this edge; nullable |

The code fingerprint matters because a reproducibility claim is meaningless if the code that produced a result is not pinned alongside its parameters. Changing an operator's implementation changes its fingerprint, so a subsequent run is visibly distinguishable from a prior one even when its parameters are identical.

Fitted statistics are stored rather than recomputed at training time. A normalizer that recomputes its own mean at training time is not reproducing the original dataset; it is producing a new dataset that happens to resemble it.

### 3.1 Stable Identity Versus Occurrence Identity

This design distinguishes two kinds of identifier that are easy to conflate, described in Table 2.

**Table 2. Identifier types and their properties.**

| Identifier | Derived from | Grows with | Safe as a metric label? |
|---|---|---|---|
| `fingerprint()` | Operation, inputs, parameters, and code | The number of distinct configurations | Conditionally |
| `run_id` | `uuid4` | Every execution | Never |
| `blob_id` | File content | The number of distinct content values | Never |

The value `fingerprint()` identifies a configuration, while `run_id` identifies an occurrence of that configuration being executed. The number of distinct configurations is bounded by the number of distinct code-and-parameter combinations that exist, whereas the number of occurrences grows without bound over time. This distinction predates the observability layer and is precisely the metrics/events split on which the telemetry plane is built; see [14-observability-architecture.md](14-observability-architecture.md), Section 2.

## 4. Failures Are Recorded Too

A run that raises an exception is still persisted, with the error captured in its metrics field:

```python
with vm.run(dataset_id, "transform", "clean/v9", ["raw/v1"]) as run:
    raise RuntimeError("boom")
# → run persisted with metrics={"error": "RuntimeError: boom"}
```

A pipeline that crashed partway through its execution is exactly the situation in which the record of what it was doing is most needed. This behavior is covered by a regression test.

## 5. The Time Dimension

Lineage answers the question of what produced a given version. It cannot, by itself, answer any question that depends on *when* — how long a run took, whether runs are getting slower over time, or whether a quality verdict has changed since the previous week. Four catalog tables, added by the observability layer, supply this time dimension without altering lineage's authority as the provenance record. They are summarized in Table 3.

**Table 3. Catalog tables supplying the time dimension.**

| Table | Grain | Question it answers |
|---|---|---|
| `pipeline_runs` | One execution | Did it run? How long did it take? Under which configuration hash? |
| `stage_runs` | One stage within a run | Has the `validate` stage gotten slower? How much volume moved through it? |
| `qc_executions` | One validation at one point in time | Did data quality change, and when? |
| `qc_cache` | One memoized validation | Can this validation be skipped entirely? |

```bash
$ rde obs runs so100_pickplace
$ rde obs stages so100_pickplace --stage validate
$ rde obs quality so100_pickplace
```

Lineage remains authoritative throughout. `pipeline_run_id` is a nullable foreign key pointing upward from `transform_runs` and `qc_records` — a grouping key layered over the provenance record, not a second provenance record. It is nullable because a `TransformRun` or `QCRecord` produced by a direct library call legitimately has no pipeline above it, and a provenance system that broke under library use would be a worse outcome than one with this gap.

## 6. Storage Consequence

Only tabular data and manifests are materialized per version; video blobs are shared across versions by content address. `training/v1` goes further still, referencing `clean/v1`'s frame table rather than copying it, at a cost of 12.2 KB on top of the data it shares.

This property is asserted directly by a test: building the full version chain must not increase the value returned by `BlobStore.total_bytes()`.

## 7. Where the Feedback Loop Closes

`training/v2` is produced by the same builder used for `training/v1`, supplied with `episode_weights` drawn from the feedback report. These weights are recorded directly in lineage:

```
build_training_dataset ['clean/v1'] -> training/v2
   recorded episode weights: {'3': 2.0, '14': 2.0, '27': 2.0, '29': 2.0, '42': 2.0}
```

As a result, the statement "`training/v2` oversampled these five episodes because the model failed on them" is a queryable fact rather than an intention that has been lost to history.

The industry term for this practice is *directed triage* — using observed model weakness to request targeted data — and it constitutes the core data-selection discipline of the autonomous-vehicle sector. This system implemented the mechanism before the term for it was known; supporting evidence is given in [13-observability-industry-research.md](13-observability-industry-research.md), Section 2.

## 8. Releasing a Version

Lineage records how a version was produced. A **contract** determines whether that version may be consumed. `DataContract` evaluates schema, temporal, quality, and distribution clauses together into a single verdict:

```bash
$ rde obs contract so100_pickplace raw/v1 --candidate raw/v2
contract so100_pickplace@raw/v1 vs so100_pickplace@raw/v2
  verdict : BLOCKED
  clauses : 30 (2 failed, 3 warned)
    [FAIL ] distribution.drift[action_energy]        observed=3.5163 expected=<= 0.5481
    [FAIL ] distribution.drift[action_magnitude_p95] observed=3.5163 expected=<= 0.4481
```

This example reflects a real evaluation, triggered by a 1.35× rescale of `action.joint` corresponding to a silently recalibrated controller. All twenty-four QC checks pass on the affected data: every value remains finite, timestamps remain monotonic, and the schema is unchanged. The data is clean by every structural and temporal measure, yet it has stopped being comparable to the data the model was trained on — and only the distribution contract detects this.

## 9. Two Scoping Limits

**Episode-level lineage assumes one input version.** `VersionManager.run` maps each output episode back to the identically-numbered episode of `input_versions[0]`. This is exact for every transform shipped in this codebase, since none currently accepts more than one input version. A future multi-input-version transform — a dataset-merge operator, for instance — would need its own lineage-mapping logic; used unmodified, this mechanism would silently fabricate an ancestry edge to the wrong input version rather than raise an error.

**`next_version()` is not safe under concurrent execution.** Computing the next `<stage>/v<N>` is a non-atomic read-then-write: list existing versions, take the maximum, add one. Two pipeline runs targeting the same dataset and stage at the same time can compute the same version number, and the catalog's upsert-on-conflict then lets the second writer overwrite the first's data with no error raised. This system is designed for sequential, single-process use; concurrent writers to the same dataset are not supported.
