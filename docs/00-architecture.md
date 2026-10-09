# 00 — System Architecture

This document is a component-by-component overview of the engine, presented through a set of formal diagrams. It is intended as a visual entry point; the detailed treatment of each component lives in the document that its section links to.

## Guide to the Document Set

The documentation is organized into three parts, only the first of which is meant to be read as a sequence.

| Range | Content | Reading order |
|---|---|---|
| 00–10 | The engine | Sequential — each document builds on the previous one |
| 11 | Code map | Reference — consult at any point |
| 12–16 | Observability | Sequential — a later, self-contained addition |

Thirteen self-contained SVG diagrams accompany the documents and are collected in [`diagrams/`](diagrams/). Each carries its own palette and adapts to light or dark backgrounds.

The organizing principle behind every design decision in this system is that **data is the product and models are consumers of it**. No model dictates the architecture: ingestion, storage, validation, transformation, and versioning contain no reference to any model.

Table 1 summarizes the scale of the implementation at the time of writing.

**Table 1. System scale.**

| Quantity | Value |
|---|---|
| Components | 15 |
| Python files | 63 (15,324 lines, including package `__init__` files) |
| QC checks | 24 |
| Metric instruments | 37 |
| Catalog entities | 12 |
| Storage planes | 3 |
| Datasets proven | 2 (6-DoF/2-camera; 14-DoF/4-camera with force) |

This document provides the visual, component-by-component overview. For an exhaustive, module-by-module walk-through of the source — what every file exposes and the non-obvious decision inside it — see [11-code-map.md](11-code-map.md).

---

## 1. System Context

The system has three classes of neighbor: **sources**, which publish raw demonstrations; **operators**, who configure and interrogate the system; and **consumers**, who train models on its output. The boundary with consumers is bidirectional. That return arrow represents the data flywheel, and it is the property that distinguishes a pipeline from a platform.

![Context diagram — DFD level 0](diagrams/00-context-dfd.svg)

*Figure 1. Context diagram (DFD level 0).*

---

## 2. Layered View

The system rests on a single structural invariant, from which the rest of the design follows: **imports point downward, never upward.** Storage has no knowledge of validation; validation has no knowledge of models; the model layer imports the engine, and the engine imports nothing from the model layer. This directional discipline, rather than any plugin mechanism, is what makes the model swappable. The precise layer stack, derived from the import graph, is Table 1 of [11 — Code map](11-code-map.md).

Figure 2 is the coarser view: the order in which data moves through the system, from a source adapter to its consumers.

![Layered decomposition](diagrams/01-layered-view.svg)

*Figure 2. Layered decomposition. Arrows show the direction of data flow; the import rule is stated over the finer stack in the code map.*

**Placement of the observability layer.** The layered diagram predates the telemetry plane, which does not sit naturally on a vertical axis. Table 2 states its import relationships directly.

**Table 2. Import relationships of `rde/observability/`.**

| Direction | Modules |
|---|---|
| Imported by | `pipeline`, `qc` |
| Imports | `schemas`, `validation.base`, `storage.video` |
| Imports from any pipeline stage | None |

Because observability imports no stage, a change to telemetry code cannot alter pipeline behavior. This is also why the layer can be disabled outright: `Pipeline(cfg, observability=False)` produces byte-identical output to a run with observability enabled.

---

## 3. Canonical Data Model

The entities in the canonical model are, for the most part, the expected ones. The decision that carries the most weight is treating **modality as an open registry rather than a fixed frame schema** — no dataset is required to carry every stream. A six-degree-of-freedom arm with two cameras and a fourteen-degree-of-freedom bimanual rig with four cameras and force/torque sensing are represented as the same model with different registry contents. This is why extending the system to the second dataset required no adapter changes.

![Canonical entity model](diagrams/02-canonical-model.svg)

*Figure 3. Canonical entity model.*

The field `outcome` (did the demonstration succeed?) is kept deliberately separate from `quality_status` (is the recording sound?), because the two facts call for different downstream responses.

Further detail is given in [02-canonical-schema.md](02-canonical-schema.md).

---

## 4. Data Flow — Level 1

Every process in the pipeline reads one version of the data and writes another. No stage reaches into another stage's memory or files; the persistent stores are the only shared state, and the catalog is the only one of those stores that is mutable.

![Level-1 data flow diagram](diagrams/03-dataflow-level1.svg)

*Figure 4. Level-1 data flow diagram.*

Processes 3 and 5 in the diagram both invoke the same QC engine, applied to different versions of the data: the raw-data pass determines what is fit to transform, and the second pass verifies that the transformation itself introduced no defects.

Further detail is given in [04-data-flow.md](04-data-flow.md).

---

## 5. Run Sequence

The interaction of greatest consequence in this diagram is the **generator chain**: the source adapter yields Arrow batches, the writer consumes them as they arrive, and at no point does any component hold the entire table in memory.

![Ingest and validate sequence](diagrams/04-run-sequence.svg)

*Figure 5. Ingest and validate sequence.*

In the diagram, the amber loop represents the generator chain, and the teal loop represents single-pass validation, in which one filtered read per episode serves all fourteen episode-scope checks rather than requiring fourteen separate full scans.

---

## 6. Storage — Three Planes

A single storage backend cannot satisfy the requirements of this system in either direction. Placing mutable QC status inside immutable Parquet files would require rewriting files merely to mark one episode as rejected. Placing frame-level rows inside SQLite would discard columnar scanning and predicate pushdown.

![Storage planes](diagrams/05-storage-planes.svg)

*Figure 6. Storage planes.*

Video is stored once regardless of how many dataset versions reference it; only tabular data and manifests are duplicated per version. As a result, a training version costs **12.2 KB** on top of the data it shares with earlier versions.

Further detail is given in [03-storage-design.md](03-storage-design.md).

---

## 7. Catalog — Entity Relationships

The catalog is small, mutable, and relational, in deliberate contrast to the bulk storage planes, which are large and immutable. Composite keys on `version_key` ensure that an episode belongs to a specific version rather than floating free of one, and a dedicated `episode_lineage` table exists because dataset-level edges cannot, by themselves, answer questions about episode-level provenance.

![Catalog ER diagram](diagrams/06-catalog-erd.svg)

*Figure 7. Catalog entity-relationship diagram.*

In the diagram, solid lines denote enforced foreign keys with `ON DELETE CASCADE`; dashed lines denote logical references that carry no database constraint. The tables `qc_records` and `transform_runs` carry no such constraint against `versions`, deliberately — a finding is allowed to survive the version it describes in normal use. `Catalog.delete_version()`, used only for disposable scratch versions (fault-injection and observability demos), deletes both explicitly rather than relying on a constraint that does not exist, so that a later reuse of the same version string cannot collide with an orphaned row from the deleted run; `episode_lineage` rows then cascade away via their FK to `transform_runs`.

**Four tables added by the observability layer.** The observability layer introduced a time dimension that the original eight catalog tables did not carry. Table 3 lists the four additions.

**Table 3. Observability tables added to the catalog.**

| Table | Grain | Question it answers |
|---|---|---|
| `pipeline_runs` | One execution | Did it run? How long did it take? Under which configuration? |
| `stage_runs` | One stage within a run | Has the `validate` stage gotten slower? How much volume moved? |
| `qc_executions` | One validation at one point in time | Did data quality change, and when? |
| `qc_cache` | One memoized validation | Can this validation be skipped? |

These tables are deliberately small and bounded, holding one row per run, per stage, and per validation. Episode- and finding-grained facts remain in Parquet, because SQLite serves as the catalog and must not be pushed toward also serving as a telemetry database. The measured justification appears in [14-observability-architecture.md](14-observability-architecture.md), Section 3: representing episode-grained facts as metric labels would cost 167 TB of resident memory at the system's planning scale, against 6.02 GB for the same facts stored as columnar rows.

The correlation spine threads through all four tables. `pipeline_run_id` is a nullable foreign key pointing *upward* on `transform_runs` and `qc_records`. It is nullable because a component invoked directly as a library has no pipeline above it, and it points upward because lineage remains the authoritative provenance record rather than being superseded by a second one.

---

## 8. Validation and the QC Engine

The QC engine executes twenty-four checks organized into five families, scheduled by scope. Its central design goal is **one pass over the data, evaluating many checks**, since the naive alternative — one scan per check — is O(checks × frames), equivalent to twenty-four full scans of the dataset.

![QC engine level-2 data flow](diagrams/07-qc-engine-level2.svg)

*Figure 8. QC engine, level-2 data flow diagram.*

Three concepts are kept separate throughout the engine: **severity** is a property of the check itself; **status** is the outcome of one particular evaluation; and **the gate** is the policy that converts findings into episode-level statuses. Keeping these separate means that the same set of findings can produce both a strict training dataset and a permissive exploration dataset without re-running validation.

Further detail is given in [05-qc-architecture.md](05-qc-architecture.md).

---

## 9. Model Adapters

Model adapters depend on the engine, and the engine never depends on a model. Adding support for a new model requires only a new adapter file and a registry entry.

![Model adapter boundary](diagrams/08-model-boundary.svg)

*Figure 9. Model adapter boundary.*

Adapters read the training contract at run time rather than hard-coding field names, so a six-degree-of-freedom, two-camera dataset and a fourteen-degree-of-freedom, four-camera dataset drive identical adapter code. The `confidence` field on `Prediction` is mandatory because the feedback loop selects low-confidence frames for review.

Further detail is given in [07-model-integration.md](07-model-integration.md).

---

## 10. Evaluation and Feedback

Evaluation output functions as an **input to the data pipeline**, not as a terminal report.

![The data flywheel](diagrams/09-feedback-flywheel.svg)

*Figure 10. The data flywheel.*

Flagged episodes are upweighted rather than removed, on the reasoning that a model failing on a given episode is evidence about the model at least as often as it is evidence about the data. Because there is no physical robot in this system, what is simulated is the failure *signal*: "failure" here denotes prediction error or low confidence measured against held-out demonstrations.

---

## 11. Use Cases

![Use case diagram](diagrams/10-use-cases.svg)

*Figure 11. Use case diagram.*

The three human actors divide along the system's structure: the data engineer operates the producing half, the ML engineer operates the consuming half, and the platform engineer operates the observability plane. Closing the feedback loop includes evaluating a model, which is why that relation is drawn as «include».

---

## 12. Episode Lifecycle

![Episode quality state machine](diagrams/11-episode-states.svg)

*Figure 12. Episode quality state machine.*

The gate's four guards mirror `QCGate.classify` exactly: a failed check at a blocking severity rejects, more than `max_warnings` warnings quarantines, any remaining warning yields WARNING, and otherwise the episode passes. The two dashed review transitions have no command-line interface; they are performed by calling `Catalog.set_episode_quality` directly. No transition in this state machine destroys data. A rejected episode remains in its source version alongside the findings that caused its rejection; the clean version of the dataset simply does not select it. This design is what makes a question such as "why is episode 9 missing from `training/v1`?" answerable as a query rather than as a mystery.

Further detail is given in [06-versioning-lineage.md](06-versioning-lineage.md).

---

## 13. The Observability Plane

![The observability plane](diagrams/12-observability-plane.svg)

*Figure 13. The observability plane.*

The telemetry layer functions as a **side-channel rather than a stage**: it observes work that is already happening and never causes any of its own. This is why it can be disabled entirely, and why its measured cost is only +0.8%.

The routing decision at the center of the diagram is the layer's central design choice, and it is enforced in code rather than left to convention because of a specific, measured consequence: labeling a single QC metric with `episode_id` would produce 264 billion time series and approximately 167 TB of resident memory at the system's planning scale, against 6.02 GB for the identical facts stored as columnar rows. Accordingly, `LabelPolicy` raises an exception if `episode_id` is used as a metric label.

Further detail is given in [14-observability-architecture.md](14-observability-architecture.md) and [15-metrics-catalog.md](15-metrics-catalog.md).

---

## 14. Diagram Coverage

All thirteen diagrams are current as of the observability layer. Five were revised to accommodate it, and each revision is additive: the engine-level content each diagram already carried is unchanged. Table 4 summarizes what was added to each.

**Table 4. Diagrams revised for the observability layer.**

| Diagram | Addition |
|---|---|
| `01-layered-view` | The `observability` rail, drawn beside the layer stack rather than inside it, showing both import directions |
| `03-dataflow-level1` | The telemetry side-channel, stores D5–D7, and the catalog mirror |
| `06-catalog-erd` | The tables `pipeline_runs`, `stage_runs`, `qc_executions`, and `qc_cache`, together with the two spine columns added to `qc_records` |
| `10-use-cases` | A platform-engineer actor and four associated use cases: monitor, investigate, gate a release, and detect drift |
| `12-observability-plane` | A new diagram showing the population/individual routing rule and its three destination stores |

Three previously published figures were found to be factually incorrect during this revision and have been corrected. Diagram `07` stated the episode-check split as four structural checks and three numerical checks; the correct split is three structural and four numerical. Diagram `09` stated that flagged episodes were upweighted by a factor of three; the correct factor is two. Diagram `05` stated the training index size as 3.2 KB; the measured value is 3.5 KB.

A later audit, which rendered every SVG and checked its text against the code, made the corrections listed in Table 5.

**Table 5. Corrections from the rendered audit.**

| Diagram | Correction |
|---|---|
| `01-layered-view` | Arrows relabelled as data flow: the bands do not form an import hierarchy, and the import stack is now stated in [11 — Code map](11-code-map.md). The adapter is labelled LeRobot v3.0 only, and its contract is labelled as five methods. |
| `03-dataflow-level1` | Connectors re-routed so that no label crosses a line. `qc_cache` removed from the list of event headers mirrored into the catalog, because it is not an event table. |
| `07-qc-engine-level2` | Phase 3 runs nine dataset checks, not ten; the tenth, the range profiler, runs in Phase 1. |
| `10-use-cases` | Trace lineage reassigned to the data engineer, the ML engineer moved so that no association line crosses a use case, and the evaluate/feedback relation corrected to «include». |
| `11-episode-states` | Guards aligned with `QCGate.classify`, and manual review transitions marked as such. |
| `00`, `02`, `04`, `05`, `06`, `08`, `09` | Text overflow, overlapping lifeline headers, and label collisions removed. |

Text overflow and label collision are invisible when inspecting the source markup and become obvious only in a rendered image, so each diagram must be rendered and inspected after every edit.

---

## 15. Cross-Cutting Concern: Memory Discipline

The reference machine for this project has approximately 8 GB of usable RAM. Naively decoding a single camera stream in full would require an estimated 18.1 GB, so every access path in the system is bounded by construction rather than by convention.

**Table 6. Peak resident memory across the two proof datasets.**

| Metric | SO-100 | ALOHA | Growth factor |
|---|---:|---:|---:|
| Frames | 19,631 | 55,000 | 2.8× |
| Numeric dimensions | 12 | 42 | 3.5× |
| Cameras | 2 | 4 | 2.0× |
| Peak RSS | 488 MB | 675 MB | 1.38× |

The sublinear growth in peak RSS relative to frame count is direct evidence that the streaming design functions as intended: resident memory tracks batch size and the size of one episode, not the size of the dataset as a whole.

The observability layer preserves this discipline, and the claim is measured rather than asserted. Instrumentation rides the same pass that already streams every frame, so the entire telemetry plane costs **+0.8% end to end** (a median of three runs moved from 6.011 s to 6.061 s). Computing drift features in a second traversal of the data would instead have cost an estimated 30–50% and would have broken the bounded-memory guarantee outright.

Further detail is given in [09-resource-usage.md](09-resource-usage.md), [10-design-decisions.md](10-design-decisions.md), and [15-metrics-catalog.md](15-metrics-catalog.md).

---

## 16. Maintaining the Diagrams

The SVG files in `docs/diagrams/` are the source of truth and are edited directly. Each one is self-contained: it inlines its palette as SVG-local custom properties, paints its own background, and carries a `prefers-color-scheme` block, so it renders correctly both as a standalone file and through an `<img>` tag in light or dark mode.

After any edit, render the diagram to PNG and inspect it, setting the window height above the SVG's own height:

```bash
google-chrome --headless --hide-scrollbars --screenshot=out.png \
  --window-size=1060,1446 docs/diagrams/03-dataflow-level1.svg
```
