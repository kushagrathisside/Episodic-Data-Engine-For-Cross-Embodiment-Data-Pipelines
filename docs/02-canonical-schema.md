# 02 — The Canonical Schema

The canonical schema is the contract observed by every stage of the pipeline. Ingestion adapters produce it, validation reads it, transformations rewrite it, the dataset builder materializes it, and model adapters consume it. As a consequence, nothing downstream of ingestion has any knowledge that LeRobot exists.

The schema is defined in `rde/schemas/`. It carries a version number, `1.0.0`, which is recorded in every manifest so that an older store is recognized as such rather than misread.

## 1. Entities

The schema comprises four entities, related as shown below: a dataset manifest describes the modalities and tasks present in a dataset; each episode represents one demonstration attempt; and each frame represents one synchronized timestep within an episode.

```
DatasetManifest ──┬── ModalitySpec[]    what streams exist
                  └── Task[]            language goals
        │
        ├── Episode[]                   one demonstration attempt
        │       └── frame range, video windows, quality status
        │
        └── Frame                       one synchronised timestep
                ├── observation{}       inline numeric, by canonical name
                ├── action{}            inline numeric
                └── modality_refs{}     BlobRef → external payload
```

### 1.1 Dataset (`DatasetManifest`)

The dataset manifest carries the following fields: `dataset_id`, `version`, `schema_version`, `source`, `source_format`, `created_at`, `total_episodes`, `total_frames`, `total_tasks`, `fps`, `robot_type`, `modalities[]`, `tasks[]`, `quality_status`, and `statistics`.

A dataset's identity is the pair `dataset_id@version` — for example, `so100_pickplace@training/v1`.

### 1.2 Episode

Each episode carries the fields `episode_id`, `dataset_id`, `task_id`, `frame_start`/`frame_end` (a half-open range), `start`/`end_timestamp_s`, `outcome`, `quality_status`, `modalities[]`, `blob_ids{}`, and `attributes{}`.

The field `outcome` (success, failure, or unknown) is kept separate from `quality_status` (unknown, passed, warning, rejected, or quarantined) because "the demonstration failed" and "the recording is corrupt" are different facts that call for different downstream responses. LeRobot carries no success label for its demonstrations, so `outcome` is recorded here as `unknown` — an honest representation of missing information rather than a fabricated value.

### 1.3 Frame

Each frame carries the fields `frame_id`, `episode_id`, `frame_index`, `timestamp_s`, `task_id`, `observation{}`, `action{}`, and `modality_refs{}`.

### 1.4 Modality — the Extensibility Point

```python
class ModalityKind(str, Enum):
    RGB, DEPTH, ROBOT_STATE, ROBOT_ACTION, HAND_STATE,
    TACTILE, FORCE_TORQUE, AUDIO, POINT_CLOUD, LANGUAGE, OTHER

class StoragePlane(str, Enum):
    INLINE    # numeric column in the frame table
    EXTERNAL  # blob store, referenced by id
```

Modalities form an open registry rather than a fixed frame schema, and this is the central design decision of the canonical model: no dataset is required to carry every modality. The primary dataset used in this project carries RGB, joint state, and joint action. A dataset that instead carries depth, tactile sensing, or a dexterous hand can be represented without a schema migration, because the storage-plane assignment follows from the modality kind, and the physical Parquet schema is derived from the registry rather than hand-written.

Tactile and force-torque are declared as modality kinds even though no data of these kinds is present in the primary dataset. This is intentional: the registry provides an extensible slot for these modalities rather than fabricating readings that were never collected.

## 2. Mapping from LeRobot to the Canonical Schema

The mapping from LeRobot's source keys to canonical fields is table-driven, defined in `rde/ingestion/lerobot.py`, so that supporting a new stream requires no code change. Table 1 gives the mapping used for the datasets in this project.

**Table 1. LeRobot-to-canonical field mapping.**

| Source key | Canonical field | Kind | Storage plane |
|---|---|---|---|
| `action` | `action.joint` | robot_action | inline |
| `observation.state` | `state.joint` | robot_state | inline |
| `observation.images.top` | `rgb.top` | rgb | external |
| `observation.images.wrist` | `rgb.wrist` | rgb | external |
| `observation.depth.*` | `depth.*` | depth | external |
| `observation.effort` | `force.effort` | force_torque | inline |
| `observation.tactile.*` | `tactile.*` | tactile | inline |
| (unrecognized) | `other.<key>` | other | inline |

A source key that the mapping table does not recognize is retained under a namespaced `other.<key>` field rather than dropped. Silently discarding a feature that was not anticipated in advance is a common mechanism by which data is lost.

## 3. Physical Frame Schema

The physical frame schema is derived automatically from the modality registry, which prevents the logical and physical schemas from drifting apart.

```
frame_id                int64
episode_id              int64
frame_index              int64
timestamp_s             double
task_id                 int64
action.joint            fixed_size_list<float>[6]
state.joint             fixed_size_list<float>[6]
rgb.top__blob_id        string
rgb.top__frame_in_blob  int64
rgb.top__ts_in_blob     double
rgb.wrist__…            (same three columns)
```

Two decisions in this schema merit explicit statement.

**Fixed-size lists.** Dimensionality is encoded in the column type itself, so a six-degree-of-freedom column cannot silently accept a seven-element row. This converts a class of dimensionality bug from a training-time failure into a write-time error.

**Flattened external references.** References to external payloads are flattened into three scalar columns rather than represented as a nested struct. Flat columns dictionary-encode efficiently — the blob identifier repeats 19,631 times in the primary dataset, so its dictionary contains only two entries — and they support predicate pushdown, neither of which nested structs do well.

## 4. Batches Rather Than Objects on the Bulk Path

Ingestion adapters yield Arrow `RecordBatch` objects rather than individual `Frame` objects. Constructing 19,631 Pydantic model instances merely to serialize them immediately back to Parquet would incur far more cost than it returns. Accordingly, the canonical model is validated at the boundary of the system — during manifest construction and episode-record creation — while the bulk data path itself remains columnar throughout. The `Frame` type remains available as the interchange type for single-frame access in exploration tools and model adapters.

## 5. A Second Schema Family: Event Rows

The entities described above characterize *data*. The observability layer adds a second family of five dataclasses, defined in `rde/observability/events.py`, that characterize *executions*. Table 2 summarizes these row types.

**Table 2. Observability event row types.**

| Row type | Grain | Destination |
|---|---|---|
| `PipelineRunRow` | One pipeline execution | Parquet and `pipeline_runs` |
| `StageRunRow` | One stage within a run | Parquet and `stage_runs` |
| `QCExecutionRow` | One validation at one point in time | Parquet and `qc_executions` |
| `EpisodeHealthRow` | One episode at one validation | Parquet only |
| `DriftReportRow` | One feature, compared across two versions | Parquet only |

These five types are implemented as plain dataclasses rather than as Pydantic models. This is a deliberate choice: they are written far more often than they are validated, and their Arrow types are inferred from their type annotations rather than from the data itself, so that an all-null column retains its declared type instead of collapsing to `null` and breaking the next append operation.

Nested payloads are stored within a single `*_json` column so that the schema remains stable as new health features are introduced; adding a drift feature should not require a table migration.

The final two row types, `EpisodeHealthRow` and `DriftReportRow`, are Parquet-only by design. Both are episode-grained, meaning that their volume grows with the size of the fleet, and SQLite functions here as a catalog rather than as a telemetry store. The measured justification for this separation is given in [14-observability-architecture.md](14-observability-architecture.md), Section 3.
