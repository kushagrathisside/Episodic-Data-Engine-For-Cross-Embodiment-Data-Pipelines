# Episodic Data Engine

This is an end-to-end data platform that turns raw multimodal robot demonstrations into validated, synchronized, queryable, versioned, training-ready datasets, and exposes them through one stable interface that different models can consume without requiring any change to the data layer itself.

The system rests on a single organizing principle: **data is the product, and models are consumers of it.** No model dictates the architecture. Ingestion, storage, validation, transformation, and versioning contain no reference to any model; swapping the model requires only a new file in `rde/model_adapters/` and nothing else.

<p align="center">
  <img src="docs/diagrams/00-context-dfd.svg" alt="Context diagram: dataset sources and a data engineer feed the engine, which serves dataset versions to a training layer and reports to exploration; the training layer returns evaluation failures" width="820">
</p>

**Figure 1. System context.** The engine sits between dataset sources and any training layer. Evaluation failures flow back into the engine as inputs to the next dataset version. The companion repository [`vla-finetune`](https://github.com/kushagrathisside/VLAFinetuningExperiments) is one such training layer: it fine-tunes SmolVLA on versions produced here.

## 1. What It Does, on Real Data

The engine is built and measured against [`lerobot/svla_so100_pickplace`](https://huggingface.co/datasets/lerobot/svla_so100_pickplace), a dataset of a real SO-100 arm performing pick-and-place: 50 episodes, 19,631 frames, 30 fps, six-degree-of-freedom joint state and action, two 640×480 AV1 cameras, and 470 MB on disk.

The pipeline runs eleven stages in sequence: download, ingest, validate, synchronize, transform, QC, version, build, model adapter, evaluate, and feedback. One command runs all of it:

```bash
rde run --config configs/so100_pickplace.json
```

A real run against this dataset produces the following output.

```
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━┓
┃ stage             ┃ ok   ┃ seconds ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━┩
│ download          │ True │ 0.204   │
│ ingest            │ True │ 0.565   │   50 episodes, 19,631 frames
│ validate          │ True │ 2.659   │   24 checks, 9 findings
│ transform         │ True │ 0.085   │   50 → 48 episodes
│ qc_post_transform │ True │ 2.521   │   re-validated after transform
│ build             │ True │ 0.025   │   train 38 / val 5 / test 5
│ evaluate          │ True │ 0.426   │   MSE 0.0239 (2.6% of baseline)
│ feedback          │ True │ 0.003   │   279 failure frames, 5 episodes flagged
└───────────────────┴──────┴─────────┘
```

The full run completes in 7.0 s wall-clock time at 488 MB peak RSS, against a dataset whose video would require an estimated 18.1 GB to decode naively, on a machine with only 8 GB of usable RAM.

## 2. Proven Portable: A Second, Structurally Different Dataset

The architecture's central claim — that the pipeline does not depend on the shape of the underlying data — is only meaningful if a second dataset actually tests it. The engine was accordingly pointed at [`lerobot/aloha_static_coffee`](https://huggingface.co/datasets/lerobot/aloha_static_coffee), which differs from the primary dataset along every dimension that could plausibly break the design, as shown in Table 1.

**Table 1. The two proof datasets compared.**

| Property | SO-100 pick-place | ALOHA coffee |
|---|---|---|
| Robot | Six-degree-of-freedom single arm | Fourteen-degree-of-freedom bimanual |
| Cameras | 2 | 4 |
| Rate | 30 fps | 50 fps |
| Modalities | State and action | State, action, and force/torque |
| Frames | 19,631 | 55,000 |
| Size | 470 MB | 1.63 GB |

```bash
make pipeline-aloha          # 12.9 s, 675 MB peak
```

No adapter code had to change to support the second dataset. The field `observation.effort` mapped to `force.effort` through the existing rule table, and the transition from six to fourteen degrees of freedom was absorbed as a change in data rather than in code, because model adapters read the training contract at run time instead of hard-coding field names.

This second dataset did surface two latent defects that the first dataset could never have exposed. Both are now fixed and covered by regression tests.

**The range check was miscalibrated.** Learned bounds are expressed as percentiles, so approximately 1% of any dataset lies outside them by construction. At six dimensions, this flagged 2 of 50 episodes; at 42 dimensions, with long-tailed force data included, it flagged all 50 — a check that fires on everything carries no information at all. This was fixed in two ways: a finding now requires a material violation, either systematic or a large excursion relative to the dimension's span, and the profiler widens its bounds for impulsive signal kinds, since contact forces spike as a matter of physics rather than as a matter of fault. The measured result is a reduction from 64 findings across 50 episodes to 24 findings across 10 episodes, with the SO-100 results unchanged.

**The nearest-neighbor distance computation would have brought the machine down.** Its original broadcast formulation allocated 0.63 GB per block at 55,000 reference frames, and would have required a projected 7.2 GB at 500,000 reference frames. It was rewritten using the Gram-matrix identity, reducing the allocation to 22 MB per block, independent of dimensionality. Table 2 shows the measured effect.

**Table 2. Effect of the Gram-matrix rewrite.**

| Metric | Before | After |
|---|---:|---:|
| Peak RSS | 1.58 GB | 675 MB |
| Evaluate stage duration | 7.05 s | 1.19 s |

```bash
make compare                 # side-by-side, and what changed
```

## 3. Three Properties That Matter

### 3.1 It Processes Data Larger Than RAM

Nothing in this system is ever fully materialized in memory. Video is decoded frame-accurately by seeking, at approximately 4.8 ms for random access and 1.7 ms for sequential access per frame; Parquet is read in bounded row groups with column projection and predicate pushdown; and every pipeline stage streams its data rather than loading it wholesale.

**Table 3. Memory footprint versus a naive approach.**

| Metric | Value |
|---|---:|
| Naive full video decode | 18.1 GB |
| Actual peak RSS, full pipeline | 488 MB |

### 3.2 It Never Silently Discards Data

Every QC finding is persisted as a queryable record, carrying `episode_id`, `check_name`, `severity`, `observed_value`, `expected_value`, and `reason`. Rejected episodes remain on disk, together with the evidence that caused their rejection.

```bash
$ rde explore failed so100_pickplace raw/v1
episode 9 [warning] 388 frames, 12.93s
   [warning] numerical.value_range: episode 9: action.joint exits learned bounds in dimension(s) [3]
episode 31 [warning] 575 frames, 19.17s
   [warning] numerical.episode_outlier: episode 31: 575 frames is an outlier (median 380, MAD 22)
```

Validation comprises 24 checks spanning structural, temporal, numerical, dataset-level, and cross-modal alignment concerns. Because the reference dataset is genuinely clean, these checks are instead proven by [fault injection](examples/02_fault_injection_demo.py): every defect class is deliberately introduced into a disposable copy of the data and confirmed to be caught, as shown in Table 4.

**Table 4. Fault injection results.**

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

The final row of Table 4 is the most consequential: shifted video references leave the data structurally perfect — correct dtypes, correct row counts, no null values — while every frame nonetheless points at the wrong image. Only pixel-level alignment verification is capable of catching this defect.

### 3.3 Provenance Is a Query, Not an Investigation

```bash
$ rde version lineage so100_pickplace training/v1
ingest                 [] -> raw/v1                episodes 0->50
transform              ['raw/v1'] -> clean/v1      episodes 50->48  dropped=[9, 31]
build_training_dataset ['clean/v1'] -> training/v1 episodes 48->48
```

```bash
$ rde version trace so100_pickplace training/v1
training/v1#0  <- clean/v1#0 <- raw/v1#0
training/v1#1  <- clean/v1#1 <- raw/v1#1
...
```

Both of the design's central provenance questions — which raw episodes contributed to this training set, and what was applied before the data reached the model — are answerable in a single call, down to per-operator parameters, fitted normalization statistics, software version, and a code fingerprint.

## 4. Quick Start

```bash
git clone https://github.com/kushagrathisside/Episodic-Data-Engine-For-Cross-Embodiment-Data-Pipelines episodic-data-engine
cd episodic-data-engine
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .

rde info                                                    # show layout
rde source download lerobot/svla_so100_pickplace            # 470 MB → data root
rde run --config configs/so100_pickplace.json               # full pipeline
rde explore overview so100_pickplace training/v1
```

Dataset bytes never land in the source tree. The data root defaults to `~/VLAExperiments/rde_data/` and can be redirected by setting `RDE_DATA_ROOT`.

### 4.1 Memory Guard

Every data-touching `make` target runs under `scripts/capped`, which wraps the command in a cgroup v2 scope with a hard `MemoryMax`. The engine is designed to stream and remain memory-bounded, but "designed to" is not the same guarantee as "is guaranteed to," and on WSL specifically, an out-of-memory condition does not merely kill the offending process — it can take down the entire virtual machine. Under this cap, a runaway stage is instead terminated on its own, with SIGKILL and exit code 137, and the host machine survives.

```bash
scripts/capped rde run --config configs/so100_pickplace.json   # 6 GB default
RDE_MEM_MAX=3G scripts/capped python examples/03_resource_report.py
```

### 4.2 Using a Dataset

```python
from rde import DataEngine

ds = DataEngine().get_dataset("so100_pickplace", version="training/v1")

for sample in ds.iter_samples(split="train"):
    sample.observation["state.joint"]   # (6,) normalized
    sample.action                       # (6,) normalized
    sample.task                         # "Pick up the cube and place it in the box."
    img = sample.image("rgb.top")       # decoded on demand, resized per contract
```

Images remain as references until explicitly requested, so a state-only policy pays no cost for video decode at all: 500 samples without images take 5 ms, while the same samples with image decode take 7.1 ms each.

## 5. Architecture

Figure 2 shows the engine's layered decomposition: the order in which data moves from a source adapter to its consumers. Only the ingestion layer knows the source format. The observability package sits outside the stack: it observes every stage, but only `qc` and `pipeline` import it.

<p align="center">
  <img src="docs/diagrams/01-layered-view.svg" alt="Layered decomposition: source adapters, canonical contract, three persistence planes, stateless processing stages, and consumption, with the observability package alongside" width="900">
</p>

**Figure 2. Layered architecture.** Arrows show the direction of data flow. Imports also point one way: nothing in the data layers imports a model-facing module, and the engine imports nothing from the model layer. The exact import stack is Table 1 of [11 — Code map](docs/11-code-map.md). Processing stages are stateless: all state lives in the persistence planes. The level-1 data-flow diagram, the run sequence, and the remaining nine diagrams are indexed in [00 — System architecture](docs/00-architecture.md).

Storage is organized into three planes, deliberately rather than one database. Video blobs are content-addressed and shared across versions; only tabular data and manifests are materialized per version. As a result, promoting `raw/v1` to `training/v1` costs kilobytes rather than gigabytes, and content-addressed hard linking saves 470 MB on this dataset alone (472 MB of physical storage against 942 MB of apparent storage). Table 5 summarizes the three planes.

**Table 5. Storage planes.**

| Plane | Store | Holds |
|---|---|---|
| Binary | Content-addressed filesystem (`blobs/`) | Video, images, and, in future, point clouds |
| Tabular | Parquet (`tables/`) | State, action, timestamps, QC records |
| Metadata | SQLite (`catalog.db`) | Versions, episodes, lineage, QC index |

Figure 3 shows the metadata plane in full: the twelve tables of `catalog.db` and the foreign-key relationships between them.

<p align="center">
  <img src="docs/diagrams/06-catalog-erd.svg" alt="Entity-relationship diagram of the twelve catalog tables" width="760">
</p>

**Figure 3. Catalog entity-relationship diagram.** Eight tables describe data (datasets, versions, episodes, tasks, modalities, QC records, transform runs, episode lineage); four describe executions and were added by the observability layer (pipeline runs, stage runs, QC executions, QC cache).

Analytical queries run through DuckDB directly over the Parquet files, so exploration never requires loading a full frame table into Python.

## 6. Documentation

The documentation is organized into three parts. Only the first two are meant to be read as sequences; the code map is a standalone reference.

**Part I — the engine.** These documents should be read in order, since each builds on the previous one.

| Document | Covers |
|---|---|
| [00 — System architecture](docs/00-architecture.md) | All 13 diagrams: DFDs, the entity-relationship diagram, use cases, the run sequence, the state machine, and the telemetry plane |
| [01 — Source dataset](docs/01-source-dataset.md) | What LeRobot v3.0 actually looks like on disk |
| [02 — Canonical schema](docs/02-canonical-schema.md) | The internal model every stage speaks |
| [03 — Storage design](docs/03-storage-design.md) | The three storage planes, and why they are separate |
| [04 — Data flow](docs/04-data-flow.md) | Stage contracts, end to end |
| [05 — QC architecture](docs/05-qc-architecture.md) | Checks, gating, contracts, and fault injection |
| [06 — Versioning and lineage](docs/06-versioning-lineage.md) | Provenance guarantees |
| [07 — Model integration](docs/07-model-integration.md) | The model-interchangeability boundary |
| [08 — End-to-end run](docs/08-end-to-end-run.md) | A full, annotated execution |
| [09 — Resource usage](docs/09-resource-usage.md) | Measured resource figures |
| [10 — Design decisions](docs/10-design-decisions.md) | Every major trade-off, and what would be changed next |

**Part II — the observability layer.** A later, self-contained project, also read in order: inventory, evidence, design, catalog, and conclusion.

| Document | Covers |
|---|---|
| [12 — Observability reconnaissance](docs/12-observability-recon.md) | What telemetry existed beforehand, and the seven structural gaps |
| [13 — Industry research](docs/13-observability-industry-research.md) | Evidence, classified by strength — including where no evidence exists |
| [14 — Observability architecture](docs/14-observability-architecture.md) | The metrics/events/traces split, measured cardinality, and the backend decision |
| [15 — Metrics catalog](docs/15-metrics-catalog.md) | 37 instruments, alert threshold provenance, dashboards |
| [16 — Observability report](docs/16-observability-report.md) | What was measured, what is projected, and what remains missing |

**Reference documents.** These are not read in sequence; they cover every module in the system, including those introduced in Part II.

| Document | Covers |
|---|---|
| [11 — Code map](docs/11-code-map.md) | Every module in `rde/`: what it exposes, and the non-obvious decision inside it |
| [17 — CLI reference](docs/17-cli-reference.md) | A handover manual covering all 35 commands, every option and default, the `PipelineConfig` schema, and Makefile targets |
| [Dashboard](docs/dashboard/) | The data health console, built from real telemetry |

## 7. Diagrams and Dashboard

Two sets of visual artifacts accompany this repository, listed in Table 6. Both are built from the real system, using no mock data and no rendered-rather-than-decoded imagery, and both are static, self-contained files checked into the repository — no external hosting required.

**Table 6. Local visual artifacts.**

| Artifact | Description |
|---|---|
| [Diagrams](docs/diagrams/) | Thirteen self-contained SVGs: DFDs, the canonical entity-relationship model, the catalog entity-relationship diagram, use cases, the run sequence, the episode state machine, and the observability plane. Indexed in [00 — System architecture](docs/00-architecture.md). |
| [RDE Data Health Console](docs/dashboard/data-health-console.html) | The operational dashboard: stage timings, per-check cost, a fifty-cell episode health strip, cross-modal alignment, tiered video health, a release-contract verdict, drift calibration, and the cardinality projection. Every figure shown is reproducible by a command in this repository — see [`docs/dashboard/`](docs/dashboard/) for how. |

To inspect what's actually present in the data root at any point — real decoded video frames, storage layout, per-camera specifications, duration distributions, real state-vector samples, QC findings, and the full version lineage — use the `rde explore` commands (`rde explore overview`, `rde explore dims`, `rde explore failed`, ...); see [17 — CLI reference](docs/17-cli-reference.md).

## 8. Examples

Nine runnable examples demonstrate specific claims made throughout the documentation.

```bash
python examples/01_inspect_source.py      # inspect LeRobot without ingesting
python examples/02_fault_injection_demo.py # prove the QC checks fire
python examples/03_resource_report.py     # measure resource usage
python examples/04_swap_models.py         # model interchangeability
python examples/05_lineage_queries.py     # answer the provenance questions
python examples/06_feedback_flywheel.py   # close the data flywheel
python examples/07_compare_datasets.py    # same engine, two very different datasets
python examples/08_cardinality_experiment.py   # what episode_id as a label costs
python examples/09_observability_scenarios.py  # sensor failure + silent drift, end to end
```

These examples are also reachable through `make`: `make pipeline`, `make faults`, `make compare`, `make flywheel`, and `make test`, among others. Run `make help` for the full list of targets.

## 9. Tests

```bash
pytest tests -q      # 188 tests, 79% line coverage
```

Tests run against a synthetic dataset generated in the real LeRobot v3.0 layout (`rde/testing/`), using no network access and no mocks of the source format, since format-handling defects are where the most interesting bugs in an ingestion layer tend to live.

## 10. Scope

This system is a data-engineering proof of concept, and its success criterion is the pipeline itself, not model accuracy. Three things are explicitly not claimed.

1. **No robot and no environment.** The feedback loop's notion of "failure" refers to prediction error on held-out demonstrations, used as a proxy for deployment failure in the absence of a physical robot.
2. **No trained vision-language-action model ships with the engine.** `SmolVLAPolicy` is a working adapter for a real pretrained checkpoint and supports fine-tuning, but requires the optional `vla` extra. The reference models used by `rde run` are a mean-action baseline and a nearest-neighbor retrieval policy that outperforms it by a factor of 38. Fine-tuning experiments and their measured results are in the companion repository [`vla-finetune`](https://github.com/kushagrathisside/VLAFinetuningExperiments).
3. **Single-node execution.** The logical pipeline is designed to scale out, as discussed in [10-design-decisions.md](docs/10-design-decisions.md), but it has not been run in a distributed configuration.
