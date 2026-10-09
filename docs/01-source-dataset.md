# 01 — The Source Dataset

Two datasets are used in this project. This document covers the primary one, `lerobot/svla_so100_pickplace`. The second, `lerobot/aloha_static_coffee` (fourteen-degree-of-freedom bimanual, four cameras, 50 fps, force/torque sensing, 55,000 frames, 1.63 GB), serves as the portability test; see the project README and `examples/07_compare_datasets.py`.

Every claim in this document was established by inspecting `lerobot/svla_so100_pickplace` directly, rather than by reading LeRobot's own documentation. The inspection can be reproduced with the following command:

```bash
python examples/01_inspect_source.py
```

## 1. Overview

The dataset records a real SO-100 five-degree-of-freedom arm with a gripper performing a single pick-and-place task. Table 1 summarizes its scale.

**Table 1. Source dataset summary.**

| Property | Value |
|---|---|
| Codebase version | `v3.0` |
| Robot | `so100` |
| Episodes | 50 |
| Frames | 19,631 |
| Sampling rate | 30 fps |
| Tasks | 1 ("Pick up the cube and place it in the box.") |
| Size on disk | 470 MB across 30 files |

## 2. On-Disk Layout

The dataset is organized into a metadata directory, a single frame-table Parquet file, and two per-camera video files, as shown below.

```
lerobot__svla_so100_pickplace/
├── meta/
│   ├── info.json                          2.6 KB   schema, features, fps, paths
│   ├── stats.json                         5.8 KB   dataset-level statistics
│   ├── tasks.parquet                      2.3 KB   task_index → description
│   └── episodes/chunk-000/file-000.parquet  76 KB  per-episode metadata
├── data/chunk-000/file-000.parquet        0.5 MB   all 19,631 frames
└── videos/
    ├── observation.images.top/chunk-000/file-000.mp4    320.4 MB
    └── observation.images.wrist/chunk-000/file-000.mp4  149.2 MB
```

## 3. Frame Table

The frame table is a single Parquet file containing 19,631 rows. Its columns are described in Table 2.

**Table 2. Frame table schema.**

| Column | Type | Meaning |
|---|---|---|
| `index` | int64 | Global frame identifier |
| `episode_index` | int64 | Identifies the containing episode |
| `frame_index` | int64 | Position of the frame within its episode |
| `timestamp` | float | Seconds elapsed from the start of the episode |
| `task_index` | int64 | Task identifier |
| `observation.state` | list\<float\> | Six joint positions, in degrees |
| `action` | list\<float\> | Six commanded joint positions |

The six joints are named `main_shoulder_pan`, `main_shoulder_lift`, `main_elbow_flex`, `main_wrist_flex`, `main_wrist_roll`, and `main_gripper`.

Measured joint state spans approximately −89.8° to +123.1°, depending on the joint, while the gripper channel ranges from 0 to 34.7. No `NaN` or `Inf` values occur anywhere in the table, and sampling intervals are exactly 1/30 s throughout.

## 4. Episode Metadata

The episode metadata table carries one row per episode. Its columns of primary interest are described in Table 3.

**Table 3. Episode metadata columns.**

| Column | Meaning |
|---|---|
| `dataset_from_index` / `dataset_to_index` | Half-open global frame range covered by the episode |
| `length` | Frame count |
| `tasks` | List of task strings associated with the episode |
| `videos/<key>/from_timestamp` / `to_timestamp` | The episode's time window inside the concatenated per-camera video |
| `stats/<feature>/{min,max,mean,std,count}` | Per-episode statistics for the named feature |

## 5. Four Properties That Require Care

Inspection of the source dataset surfaced four properties that are easy to overlook and that have direct consequences for the design of the ingestion and validation layers.

### 5.1 Videos Are Concatenated by Camera, Not by Episode

Each camera contributes a single video file — 320 MB in the case of the top camera — holding all 50 episodes, 19,631 frames, and 654.37 seconds of footage. An individual episode is therefore represented as a time window inside a shared file rather than as a file of its own.

The consequence is that resolving a single RGB observation requires a pair `(blob, timestamp)`, not a file path. Naively decoding the full container would require an estimated 18.1 GB of RGB data, more than twice the RAM available on the target machine. The blob-plane design described in [03-storage-design.md](03-storage-design.md) follows directly from this constraint.

### 5.2 Two Clocks Are Present, and They Are Not the Same Clock

The `timestamp` column in the frame table is episode-relative: every episode begins at 0.0. The `from_timestamp` field in the episode metadata is file-absolute. The two are related by the following mapping:

```
video_timestamp = episode.from_timestamp + frame.timestamp
```

An error in this mapping causes every episode to be paired with a different episode's images. Critically, the data remains structurally perfect under this error — correct dtypes, correct row counts, no null values — so a model trains without complaint on mislabeled observations. This is the reason the `sync.video_alignment` check verifies alignment against decoded pixels rather than relying on arithmetic alone.

The mapping has been verified directly: episode windows are contiguous, and their durations match `frames/fps` to within 0.0 ms across all 50 episodes, with both cameras sharing identical windows.

### 5.3 Task Descriptions Are Stored in the Pandas Index, Not a Named Column

The file `tasks.parquet` exposes the columns `task_index` and `__index_level_0__`; the task description itself is stored as the pandas index rather than as a named column. A reader that expects a `task` column finds nothing. This case is handled explicitly in `rde/ingestion/lerobot.py`.

### 5.4 All 19,631 Rows Occupy a Single Row Group

A single row group defeats predicate pushdown entirely, since any filter still requires decoding the entire group. Ingestion addresses this by rewriting the table with bounded row groups of 2,048 rows each (20 groups in total), so that episode-level filters touch only the groups they require.

## 6. Video Codec

Both video files are encoded in AV1 (`libdav1d`). Decoding therefore requires a working AV1 decoder, which can be confirmed with the following check:

```python
import av; print([c for c in av.codecs_available if "av1" in c.lower()])
```

Measured decode performance on the target machine is 4.8 ms per random-access frame and 1.7 ms per sequential frame.
