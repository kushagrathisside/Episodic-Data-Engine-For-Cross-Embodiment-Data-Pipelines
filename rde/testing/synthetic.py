"""Synthetic LeRobot-format dataset generator.

Two jobs:

1. **Testing without the network.**  The full suite runs against a real dataset
   generated on disk in the actual v3.0 layout -- concatenated per-camera video,
   episode metadata with video windows, the ``__index_level_0__`` task column --
   so tests exercise the real adapter code paths rather than mocks.

2. **Fault injection.**  A curated public dataset is clean, which is exactly the
   wrong thing to validate a QC engine against.  ``inject_*`` helpers introduce
   one defect class at a time so each check can be shown to fire on the defect it
   claims to catch, and to stay silent otherwise.

Generated episodes follow smooth sinusoidal joint trajectories with per-episode
phase offsets, so numeric checks see plausible robot-like data rather than noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _encode_video(path: Path, frames: np.ndarray, fps: int) -> None:
    """Write frames to an mp4, preferring a widely-available codec."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for codec in ("libx264", "mpeg4"):
        try:
            with av.open(str(path), mode="w") as container:
                stream = container.add_stream(codec, rate=fps)
                stream.width = frames.shape[2]
                stream.height = frames.shape[1]
                stream.pix_fmt = "yuv420p"
                for f in frames:
                    frame = av.VideoFrame.from_ndarray(f, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            return
        except Exception as exc:  # try the next codec
            last_error = exc
            continue
    raise RuntimeError(f"no usable video encoder available: {last_error}")


def make_lerobot_dataset(
    root: Path,
    num_episodes: int = 4,
    frames_per_episode: int = 30,
    fps: int = 10,
    state_dim: int = 6,
    height: int = 64,
    width: int = 64,
    cameras: Sequence[str] = ("top", "wrist"),
    tasks: Sequence[str] = ("Pick up the cube and place it in the box.",),
    seed: int = 0,
) -> Path:
    """Write a complete LeRobot v3.0 dataset to ``root``.

    Layout matches the real thing, including the details that trip up naive
    readers: one concatenated video per camera, episode-relative frame timestamps
    against file-absolute video windows, and the pandas index column in
    ``tasks.parquet``.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    total_frames = num_episodes * frames_per_episode

    states, actions, ts, fi, ei, gi, ti = [], [], [], [], [], [], []
    for ep in range(num_episodes):
        phase = ep * 0.7
        t = np.arange(frames_per_episode) / fps
        # Smooth, robot-like joint trajectories rather than noise.
        s = np.stack(
            [np.sin(2 * np.pi * 0.3 * t + phase + d * 0.4) * (10 + 5 * d)
             for d in range(state_dim)], axis=1
        ).astype(np.float32)
        a = (s + rng.normal(0, 0.05, s.shape)).astype(np.float32)
        states.append(s)
        actions.append(a)
        ts.append(t.astype(np.float32))
        fi.append(np.arange(frames_per_episode, dtype=np.int64))
        ei.append(np.full(frames_per_episode, ep, dtype=np.int64))
        gi.append(np.arange(ep * frames_per_episode, (ep + 1) * frames_per_episode, dtype=np.int64))
        ti.append(np.full(frames_per_episode, ep % len(tasks), dtype=np.int64))

    frame_table = pa.table({
        "action": pa.array([a.tolist() for a in np.vstack(actions)], pa.list_(pa.float32())),
        "observation.state": pa.array([s.tolist() for s in np.vstack(states)], pa.list_(pa.float32())),
        "timestamp": pa.array(np.concatenate(ts), pa.float32()),
        "frame_index": pa.array(np.concatenate(fi), pa.int64()),
        "episode_index": pa.array(np.concatenate(ei), pa.int64()),
        "index": pa.array(np.concatenate(gi), pa.int64()),
        "task_index": pa.array(np.concatenate(ti), pa.int64()),
    })
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(frame_table, root / "data" / "chunk-000" / "file-000.parquet")

    # Videos: all episodes concatenated per camera, as v3.0 does.
    for cam in cameras:
        vid = np.zeros((total_frames, height, width, 3), dtype=np.uint8)
        for i in range(total_frames):
            ep = i // frames_per_episode
            # Distinct per-episode appearance so alignment checks are meaningful.
            vid[i, :, :, 0] = (30 + 50 * ep) % 256
            vid[i, :, :, 1] = (i * 7) % 256
            vid[i, :, :, 2] = (i * 3 + 60 * ep) % 256
            vid[i, 5:15, 5 + (i % max(1, width - 20)):15 + (i % max(1, width - 20)), :] = 255
        _encode_video(root / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4",
                      vid, fps)

    ep_meta: dict[str, list] = {
        "episode_index": [], "data/chunk_index": [], "data/file_index": [],
        "dataset_from_index": [], "dataset_to_index": [], "tasks": [], "length": [],
        "meta/episodes/chunk_index": [], "meta/episodes/file_index": [],
    }
    for cam in cameras:
        k = f"videos/observation.images.{cam}"
        ep_meta[f"{k}/chunk_index"] = []
        ep_meta[f"{k}/file_index"] = []
        ep_meta[f"{k}/from_timestamp"] = []
        ep_meta[f"{k}/to_timestamp"] = []

    for ep in range(num_episodes):
        f0 = ep * frames_per_episode
        f1 = f0 + frames_per_episode
        ep_meta["episode_index"].append(ep)
        ep_meta["data/chunk_index"].append(0)
        ep_meta["data/file_index"].append(0)
        ep_meta["dataset_from_index"].append(f0)
        ep_meta["dataset_to_index"].append(f1)
        ep_meta["tasks"].append([tasks[ep % len(tasks)]])
        ep_meta["length"].append(frames_per_episode)
        ep_meta["meta/episodes/chunk_index"].append(0)
        ep_meta["meta/episodes/file_index"].append(0)
        for cam in cameras:
            k = f"videos/observation.images.{cam}"
            ep_meta[f"{k}/chunk_index"].append(0)
            ep_meta[f"{k}/file_index"].append(0)
            ep_meta[f"{k}/from_timestamp"].append(f0 / fps)
            ep_meta[f"{k}/to_timestamp"].append(f1 / fps)

    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(ep_meta),
                   root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # Task descriptions live in the pandas index column, as upstream writes them.
    pq.write_table(
        pa.table({
            "task_index": pa.array(list(range(len(tasks))), pa.int64()),
            "__index_level_0__": pa.array(list(tasks)),
        }),
        root / "meta" / "tasks.parquet",
    )

    features: dict[str, Any] = {
        "action": {"dtype": "float32", "shape": [state_dim],
                   "names": [f"joint_{i}" for i in range(state_dim)], "fps": float(fps)},
        "observation.state": {"dtype": "float32", "shape": [state_dim],
                              "names": [f"joint_{i}" for i in range(state_dim)], "fps": float(fps)},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": float(fps), "video.height": height, "video.width": width,
                     "video.channels": 3, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "has_audio": False},
        }
    for k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[k] = {"dtype": "float32" if k == "timestamp" else "int64",
                       "shape": [1], "names": None}

    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "robot_type": "synthetic_arm",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }, indent=2))
    return root


# -- fault injection -------------------------------------------------------
def _rewrite_frames(version_dir: Path, mutate) -> None:
    """Load, mutate and rewrite a canonical frame table in place."""
    target = version_dir / "frames" / "part-000.parquet"
    table = pq.read_table(target)
    pq.write_table(mutate(table), target)


def inject_nan(version_dir: Path, column: str = "state.joint",
               row: int = 5, dim: int = 2) -> dict:
    """Put a NaN into one element of one frame -> numerical.finite."""
    def mutate(t: pa.Table) -> pa.Table:
        d = t.to_pydict()
        v = list(d[column][row])
        v[dim] = float("nan")
        d[column][row] = v
        return pa.Table.from_pydict(d, schema=t.schema)

    _rewrite_frames(version_dir, mutate)
    return {"fault": "nan", "column": column, "row": row, "dim": dim}


def inject_timestamp_gap(version_dir: Path, row: int = 10, gap_s: float = 2.0) -> dict:
    """Shift all timestamps after ``row`` forward -> temporal.gaps."""
    def mutate(t: pa.Table) -> pa.Table:
        d = t.to_pydict()
        ep = d["episode_id"][row]
        for i in range(len(d["timestamp_s"])):
            if d["episode_id"][i] == ep and i >= row:
                d["timestamp_s"][i] += gap_s
        return pa.Table.from_pydict(d, schema=t.schema)

    _rewrite_frames(version_dir, mutate)
    return {"fault": "timestamp_gap", "row": row, "gap_s": gap_s}


def inject_non_monotonic(version_dir: Path, row: int = 8) -> dict:
    """Swap two adjacent timestamps -> temporal.monotonic."""
    def mutate(t: pa.Table) -> pa.Table:
        d = t.to_pydict()
        d["timestamp_s"][row], d["timestamp_s"][row + 1] = (
            d["timestamp_s"][row + 1], d["timestamp_s"][row]
        )
        return pa.Table.from_pydict(d, schema=t.schema)

    _rewrite_frames(version_dir, mutate)
    return {"fault": "non_monotonic", "row": row}


def inject_duplicate_timestamp(version_dir: Path, row: int = 12) -> dict:
    """Copy a timestamp onto its neighbour -> temporal.duplicates."""
    def mutate(t: pa.Table) -> pa.Table:
        d = t.to_pydict()
        d["timestamp_s"][row + 1] = d["timestamp_s"][row]
        return pa.Table.from_pydict(d, schema=t.schema)

    _rewrite_frames(version_dir, mutate)
    return {"fault": "duplicate_timestamp", "row": row}


def inject_stuck_sensor(version_dir: Path, column: str = "state.joint",
                        episode_id: int = 1, dim: int = 0) -> dict:
    """Freeze one dimension for a whole episode -> numerical.stuck_sensor."""
    def mutate(t: pa.Table) -> pa.Table:
        d = t.to_pydict()
        const = None
        for i in range(len(d[column])):
            if d["episode_id"][i] != episode_id:
                continue
            v = list(d[column][i])
            const = v[dim] if const is None else const
            v[dim] = const
            d[column][i] = v
        return pa.Table.from_pydict(d, schema=t.schema)

    _rewrite_frames(version_dir, mutate)
    return {"fault": "stuck_sensor", "episode_id": episode_id, "dim": dim}


def inject_discontinuity(version_dir: Path, column: str = "action.joint",
                         row: int = 15, magnitude: float = 500.0) -> dict:
    """Add a large single-step jump -> numerical.discontinuity / value_range."""
    def mutate(t: pa.Table) -> pa.Table:
        d = t.to_pydict()
        v = list(d[column][row])
        v[0] += magnitude
        d[column][row] = v
        return pa.Table.from_pydict(d, schema=t.schema)

    _rewrite_frames(version_dir, mutate)
    return {"fault": "discontinuity", "row": row, "magnitude": magnitude}


def inject_dropped_frames(version_dir: Path, episode_id: int = 2,
                          drop_from: int = 5, count: int = 3) -> dict:
    """Delete rows mid-episode -> structural.frame_index_continuity."""
    def mutate(t: pa.Table) -> pa.Table:
        d = t.to_pydict()
        keep = [
            i for i in range(t.num_rows)
            if not (d["episode_id"][i] == episode_id
                    and drop_from <= d["frame_index"][i] < drop_from + count)
        ]
        return t.take(keep)

    _rewrite_frames(version_dir, mutate)
    return {"fault": "dropped_frames", "episode_id": episode_id,
            "drop_from": drop_from, "count": count}


def inject_distribution_shift(
    version_dir: Path,
    column: str = "action.joint",
    scale: float = 1.6,
    offset: float = 0.0,
    from_episode: int = 0,
) -> dict:
    """Scale and offset a modality for later episodes -> a *distribution* fault.

    Unlike the other nine injectors, this one breaks nothing checkable.  Every
    value stays finite, timestamps stay monotonic, frame counts stay consistent,
    the schema is unchanged and no blob goes missing -- so all 24 QC checks pass
    and the dataset is, by every structural definition, clean.

    That is the point.  It reproduces the failure a data engine actually fears:
    a re-calibrated robot, a swapped gripper, a firmware change that silently
    rescales a controller.  The data remains valid and stops being comparable
    to the data the model was trained on.  Nothing but distribution monitoring
    sees it, which is what makes it the right fault to prove drift detection
    against (docs/14 §1).
    """
    def mutate(t: pa.Table) -> pa.Table:
        d = t.to_pydict()
        for i in range(len(d[column])):
            if d["episode_id"][i] < from_episode:
                continue
            d[column][i] = [v * scale + offset for v in d[column][i]]
        return pa.Table.from_pydict(d, schema=t.schema)

    _rewrite_frames(version_dir, mutate)
    return {"fault": "distribution_shift", "column": column, "scale": scale,
            "offset": offset, "from_episode": from_episode}


def inject_missing_blob(blob_root: Path, blob_id: str) -> dict:
    """Delete a stored blob -> structural.blob_integrity.

    .. warning::
       Destructive **across versions**.  The blob store is content-addressed and
       deduplicated, so one file backs every version that references that
       content -- deleting it corrupts the original alongside the scratch clone,
       and the only repair is re-ingesting from the source cache.  Use this
       against a throwaway blob root, never the shared one.  This is why
       ``examples/02_fault_injection_demo.py``, which clones the *tables* into a
       scratch version but shares the blob plane, does not include it.
    """
    shard = blob_root / blob_id[:2] / blob_id[2:4]
    removed = []
    for p in shard.glob(f"{blob_id}*"):
        if not p.name.endswith(".meta.json"):
            p.unlink()
            removed.append(str(p))
    return {"fault": "missing_blob", "blob_id": blob_id, "removed": removed}


def inject_video_misalignment(version_dir: Path, shift_s: float = 5.0) -> dict:
    """Shift every video reference by a constant -> sync.video_alignment.

    The nastiest defect class: structurally perfect data whose images belong to
    the wrong moment.  Only the pixel-distinctness check catches it.
    """
    def mutate(t: pa.Table) -> pa.Table:
        d = t.to_pydict()
        for col in list(d):
            if col.endswith("__ts_in_blob"):
                d[col] = [None if v is None else v + shift_s for v in d[col]]
        return pa.Table.from_pydict(d, schema=t.schema)

    _rewrite_frames(version_dir, mutate)
    return {"fault": "video_misalignment", "shift_s": shift_s}
