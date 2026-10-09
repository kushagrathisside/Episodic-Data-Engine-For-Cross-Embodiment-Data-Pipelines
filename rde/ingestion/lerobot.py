"""LeRobot dataset adapter (codebase version v3.0).

Everything format-specific about LeRobot is confined to this file.  The details
that matter, established by inspecting the real dataset rather than trusting the
docs:

**Videos are concatenated per camera, not per episode.**  ``svla_so100_pickplace``
stores all 50 episodes of the top camera in one 320 MB mp4.  An episode is a
*time window* inside it, given by ``videos/<key>/from_timestamp`` and
``to_timestamp`` in the episode metadata.  Frame-level access is therefore
(blob id, global timestamp), which is exactly what :class:`BlobRef` encodes.

**Frame timestamps are episode-relative; video timestamps are file-absolute.**
The mapping is ``video_ts = episode.from_timestamp + frame.timestamp``.  Getting
this wrong silently pairs each episode's actions with the previous episode's
images -- a defect no numeric check would catch, so
:mod:`rde.synchronization` verifies it explicitly.

**tasks.parquet stores the description as the pandas index**, arriving as a column
literally named ``__index_level_0__``.  Handled here rather than leaking outward.

**All 19 631 rows sit in one Parquet row group upstream**, which defeats predicate
pushdown.  Ingestion rewrites with bounded row groups.

Key mapping from LeRobot names to canonical modality names is table-driven, so a
dataset with depth or tactile streams maps without code changes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from ..schemas.core import DatasetManifest, Episode, EpisodeOutcome, Task
from ..schemas.modality import ModalityKind, ModalitySpec, StoragePlane
from .base import SourceAdapter, SourceProfile

# -- canonical naming ------------------------------------------------------
# Ordered rules: first match wins.  (pattern, canonical template, kind)
_KEY_RULES: list[tuple[re.Pattern, str, ModalityKind]] = [
    (re.compile(r"^observation\.images?\.(?P<rest>.+)$"), "rgb.{rest}", ModalityKind.RGB),
    (re.compile(r"^observation\.depth\.(?P<rest>.+)$"), "depth.{rest}", ModalityKind.DEPTH),
    (re.compile(r"^observation\.state$"), "state.joint", ModalityKind.ROBOT_STATE),
    (re.compile(r"^observation\.environment_state$"), "state.environment", ModalityKind.ROBOT_STATE),
    (re.compile(r"^observation\.effort$"), "force.effort", ModalityKind.FORCE_TORQUE),
    (re.compile(r"^observation\.velocity$"), "state.velocity", ModalityKind.ROBOT_STATE),
    (re.compile(r"^observation\.hand\.(?P<rest>.+)$"), "hand.{rest}", ModalityKind.HAND_STATE),
    (re.compile(r"^observation\.tactile\.?(?P<rest>.*)$"), "tactile.{rest}", ModalityKind.TACTILE),
    (re.compile(r"^action$"), "action.joint", ModalityKind.ROBOT_ACTION),
]

#: LeRobot bookkeeping columns -- carried as canonical fields, not modalities.
_STRUCTURAL_KEYS = {
    "timestamp", "frame_index", "episode_index", "index", "task_index", "next.done",
    "next.reward", "next.success",
}


def canonical_name(source_key: str) -> tuple[str, ModalityKind]:
    """Map a LeRobot feature key onto a canonical modality name and kind."""
    for pattern, template, kind in _KEY_RULES:
        m = pattern.match(source_key)
        if m:
            rest = m.groupdict().get("rest") or ""
            name = template.format(rest=rest).rstrip(".")
            return name, kind
    # Unknown key: keep it, namespaced, rather than dropping data silently.
    return f"other.{source_key.replace('.', '_')}", ModalityKind.OTHER


def _flatten_names(names: Any) -> list[str]:
    """Normalize a LeRobot feature's ``names`` into a flat, index-aligned list.

    Single-limb robots (``svla_so100_pickplace``) store this as a flat list
    already. Multi-limb robots (bimanual ALOHA's ``observation.state.names``)
    store it grouped by sub-limb instead -- ``{"motors": [...14 names...]}``,
    or potentially multiple groups. Every downstream consumer of
    ``element_names`` (exploration queries, dataset profiling, reports)
    expects a flat list aligned to the modality's dimension index, so
    normalize once here rather than at every read site.
    """
    if isinstance(names, dict):
        flat: list[str] = []
        for group in names.values():
            flat.extend(group)
        return flat
    return list(names)


class LeRobotAdapter(SourceAdapter):
    """Adapter for on-disk LeRobot datasets."""

    source_format = "lerobot"

    def __init__(self, root: Path, dataset_id: str, source_uri: str | None = None):
        super().__init__(root, dataset_id, source_uri)
        self._info: dict[str, Any] | None = None
        self._episodes_table: pa.Table | None = None
        self._blob_ids: dict[str, str] = {}

    # -- raw metadata -------------------------------------------------------
    @property
    def info(self) -> dict[str, Any]:
        """Parsed meta/info.json, memoised. Raises if this is not a LeRobot root."""
        if self._info is None:
            p = self.root / "meta" / "info.json"
            if not p.exists():
                raise FileNotFoundError(f"not a LeRobot dataset (no meta/info.json): {self.root}")
            self._info = json.loads(p.read_text())
        return self._info

    @property
    def codebase_version(self) -> str:
        """LeRobot format version, e.g. 'v3.0'."""
        return str(self.info.get("codebase_version", "unknown"))

    def _episode_meta(self) -> pa.Table:
        """Episode metadata, concatenated across chunk files (50 rows here)."""
        if self._episodes_table is None:
            meta_dir = self.root / "meta" / "episodes"
            files = sorted(meta_dir.rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no episode metadata under {meta_dir}")
            self._episodes_table = pa.concat_tables([pq.read_table(f) for f in files])
        return self._episodes_table

    def _data_files(self) -> list[Path]:
        return sorted((self.root / "data").rglob("*.parquet"))

    def _video_files(self) -> dict[str, Path]:
        """Map LeRobot video feature key -> single container path.

        v3.0 emits one file per camera for this dataset size.  Multi-file cameras
        are rejected loudly rather than half-supported, because silently ingesting
        only the first file would lose data.
        """
        out: dict[str, Path] = {}
        vroot = self.root / "videos"
        if not vroot.is_dir():
            return out
        for key, feat in self.info.get("features", {}).items():
            if feat.get("dtype") != "video":
                continue
            files = sorted((vroot / key).rglob("*.mp4"))
            if not files:
                continue
            if len(files) > 1:
                raise NotImplementedError(
                    f"{key}: {len(files)} video files; multi-file cameras need the "
                    "per-file blob mapping to be extended before ingesting."
                )
            out[key] = files[0]
        return out

    def _tasks(self) -> list[Task]:
        """Read tasks, tolerating the pandas-index column name."""
        p = self.root / "meta" / "tasks.parquet"
        if not p.exists():
            return []
        t = pq.read_table(p).to_pydict()
        idx = t.get("task_index", [])
        desc_col = next(
            (c for c in ("task", "__index_level_0__", "description") if c in t), None
        )
        descs = t.get(desc_col, []) if desc_col else [""] * len(idx)
        return [
            Task(task_id=int(i), description=str(d), source_key=desc_col)
            for i, d in zip(idx, descs)
        ]

    # -- inspection ---------------------------------------------------------
    def inspect(self) -> SourceProfile:
        """Read metadata only; never decodes video or scans every frame."""
        info = self.info
        anomalies: list[str] = []

        data_files = self._data_files()
        video_files = self._video_files()

        # Row-group geometry drives whether streaming reads can skip work.
        rg_info = []
        for f in data_files:
            md = pq.ParquetFile(f).metadata
            rg_info.append(
                {"file": f.name, "rows": md.num_rows, "row_groups": md.num_row_groups}
            )
            if md.num_row_groups == 1 and md.num_rows > 50_000:
                anomalies.append(
                    f"{f.name}: {md.num_rows} rows in a single row group; "
                    "predicate pushdown will be ineffective until rewritten"
                )

        declared_eps = int(info.get("total_episodes", 0))
        actual_eps = self._episode_meta().num_rows
        if declared_eps != actual_eps:
            anomalies.append(
                f"info.json declares {declared_eps} episodes; metadata holds {actual_eps}"
            )

        declared_frames = int(info.get("total_frames", 0))
        actual_frames = sum(r["rows"] for r in rg_info)
        if declared_frames != actual_frames:
            anomalies.append(
                f"info.json declares {declared_frames} frames; data holds {actual_frames}"
            )

        return SourceProfile(
            source_format=f"{self.source_format}/{self.codebase_version}",
            source_uri=self.source_uri,
            root=self.root,
            robot_type=info.get("robot_type"),
            fps=float(info["fps"]) if info.get("fps") else None,
            total_episodes=actual_eps,
            total_frames=actual_frames,
            total_tasks=len(self._tasks()),
            features=info.get("features", {}),
            files={
                "data_files": rg_info,
                "video_files": {k: str(v) for k, v in video_files.items()},
                "video_bytes": {k: v.stat().st_size for k, v in video_files.items()},
            },
            anomalies=anomalies,
            raw_metadata={
                k: v for k, v in info.items() if k != "features"
            },
        )

    # -- canonical schema ---------------------------------------------------
    def _modalities(self) -> list[ModalitySpec]:
        specs: list[ModalitySpec] = []
        for key, feat in self.info.get("features", {}).items():
            if key in _STRUCTURAL_KEYS:
                continue
            name, kind = canonical_name(key)
            is_video = feat.get("dtype") == "video"
            plane = StoragePlane.EXTERNAL if is_video else ModalitySpec.infer_plane(kind)
            attrs: dict[str, Any] = {}
            if feat.get("names"):
                attrs["element_names"] = _flatten_names(feat["names"])
            if feat.get("info"):
                attrs["video"] = feat["info"]
            specs.append(
                ModalitySpec(
                    name=name,
                    kind=kind,
                    plane=plane,
                    dtype="uint8" if is_video else str(feat.get("dtype", "float32")),
                    shape=tuple(feat.get("shape", ()) or ()),
                    fps=float(feat.get("fps") or self.info.get("fps") or 0) or None,
                    source_key=key,
                    attributes=attrs,
                )
            )
        return specs

    def build_manifest(self, version: str) -> DatasetManifest:
        """Canonical manifest, including the derived modality registry."""
        info = self.info
        profile = self.inspect()
        return DatasetManifest(
            dataset_id=self.dataset_id,
            version=version,
            source=self.source_uri,
            source_format=f"{self.source_format}/{self.codebase_version}",
            total_episodes=profile.total_episodes,
            total_frames=profile.total_frames,
            total_tasks=profile.total_tasks,
            fps=profile.fps,
            robot_type=info.get("robot_type"),
            modalities=self._modalities(),
            tasks=self._tasks(),
            attributes={
                "codebase_version": self.codebase_version,
                "splits": info.get("splits"),
                "chunks_size": info.get("chunks_size"),
                "ingest_anomalies": profile.anomalies,
            },
        )

    # -- blobs --------------------------------------------------------------
    def register_blobs(self, blob_store) -> dict[str, str]:
        """Hardlink each camera's container into the blob store.

        Link mode matters at scale: copying would double 470 MB now and terabytes
        later, for bytes that already exist immutably in the source cache.
        """
        mapping: dict[str, str] = {}
        for source_key, path in self._video_files().items():
            name, _ = canonical_name(source_key)
            feat = self.info["features"][source_key]
            binfo = blob_store.put(
                path,
                media_type="video/mp4",
                mode="link",
                attributes={
                    "modality": name,
                    "source_key": source_key,
                    "codec": (feat.get("info") or {}).get("video.codec"),
                    "dataset_id": self.dataset_id,
                },
            )
            mapping[name] = binfo.blob_id
        self._blob_ids = mapping
        return mapping

    # -- episodes -----------------------------------------------------------
    def iter_episodes(self) -> Iterator[Episode]:
        """Canonical episodes, carrying video windows in attributes."""
        t = self._episode_meta().to_pydict()
        n = len(t["episode_index"])
        tasks = {task.description: task.task_id for task in self._tasks()}
        video_keys = list(self._video_files())
        fps = float(self.info.get("fps") or 30.0)

        for i in range(n):
            ep_id = int(t["episode_index"][i])
            f0 = int(t["dataset_from_index"][i])
            f1 = int(t["dataset_to_index"][i])

            ep_tasks = t.get("tasks", [[]])[i] or []
            task_id = tasks.get(ep_tasks[0]) if ep_tasks else None

            # Video windows are absolute offsets into the concatenated container.
            windows: dict[str, dict[str, float]] = {}
            for vk in video_keys:
                fk, tk = f"videos/{vk}/from_timestamp", f"videos/{vk}/to_timestamp"
                if fk in t and tk in t:
                    cname, _ = canonical_name(vk)
                    windows[cname] = {
                        "from_timestamp": float(t[fk][i]),
                        "to_timestamp": float(t[tk][i]),
                    }

            length = int(t.get("length", [f1 - f0])[i])
            yield Episode(
                episode_id=ep_id,
                dataset_id=self.dataset_id,
                task_id=task_id,
                frame_start=f0,
                frame_end=f1,
                start_timestamp_s=0.0,
                end_timestamp_s=length / fps,
                outcome=EpisodeOutcome.UNKNOWN,  # LeRobot records no success label
                source_dataset=self.source_uri,
                modalities=[m.name for m in self._modalities()],
                blob_ids=dict(self._blob_ids),
                attributes={
                    "video_windows": windows,
                    "tasks": ep_tasks,
                    "declared_length": length,
                },
            )

    # -- frames -------------------------------------------------------------
    def canonical_frame_schema(self) -> pa.Schema:
        """Physical schema of the canonical frame table.

        Inline vectors become fixed-size lists so dimensionality is enforced by
        the schema.  External modalities are flattened into three scalar columns
        each -- flat columns dictionary-encode and predicate-push far better than
        a nested struct, and blob ids repeat heavily so the dictionary is tiny.
        """
        fields = [
            pa.field("frame_id", pa.int64()),
            pa.field("episode_id", pa.int64()),
            pa.field("frame_index", pa.int64()),
            pa.field("timestamp_s", pa.float64()),
            pa.field("task_id", pa.int64()),
        ]
        for m in self._modalities():
            if m.plane is StoragePlane.INLINE:
                width = m.element_count
                fields.append(pa.field(m.name, pa.list_(pa.float32(), width)))
            else:
                fields += [
                    pa.field(f"{m.name}__blob_id", pa.string()),
                    pa.field(f"{m.name}__frame_in_blob", pa.int64()),
                    pa.field(f"{m.name}__ts_in_blob", pa.float64()),
                ]
        return pa.schema(fields)

    def _episode_offsets(self) -> dict[int, dict[str, float]]:
        """episode_id -> {modality: video window start}, for timestamp rebasing."""
        offsets: dict[int, dict[str, float]] = {}
        for ep in self.iter_episodes():
            offsets[ep.episode_id] = {
                name: w["from_timestamp"]
                for name, w in ep.attributes.get("video_windows", {}).items()
            }
        return offsets

    def iter_frame_batches(self, batch_size: int = 2048) -> Iterator[pa.RecordBatch]:
        """Stream source frames, converted to the canonical schema."""
        schema = self.canonical_frame_schema()
        offsets = self._episode_offsets()
        fps = float(self.info.get("fps") or 30.0)
        inline = [m for m in self._modalities() if m.plane is StoragePlane.INLINE]
        external = [m for m in self._modalities() if m.plane is StoragePlane.EXTERNAL]
        if external and not self._blob_ids:
            # register_blobs() populates self._blob_ids; every external
            # modality's *__blob_id column below silently becomes None
            # without it (ingest.py's documented call order, enforced here
            # rather than only by a comment there), corrupting the frame
            # table with no error at ingest time or read time.
            raise RuntimeError(
                "iter_frame_batches() called before register_blobs(): "
                f"{[m.name for m in external]} would be written with a null "
                "blob id. Call register_blobs(blob_store) first."
            )

        wanted = ["index", "episode_index", "frame_index", "timestamp", "task_index"]
        wanted += [m.source_key for m in inline]

        for f in self._data_files():
            pf = pq.ParquetFile(f)
            available = set(pf.schema_arrow.names)
            cols = [c for c in wanted if c in available]
            for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
                d = batch.to_pydict()
                n = batch.num_rows
                eps = [int(x) for x in d["episode_index"]]
                ts = [float(x) for x in d["timestamp"]]

                arrays: dict[str, Any] = {
                    "frame_id": [int(x) for x in d["index"]],
                    "episode_id": eps,
                    "frame_index": [int(x) for x in d["frame_index"]],
                    "timestamp_s": ts,
                    "task_id": [int(x) for x in d.get("task_index", [0] * n)],
                }
                for m in inline:
                    arrays[m.name] = d.get(m.source_key, [None] * n)

                for m in external:
                    blob_id = self._blob_ids.get(m.name)
                    abs_ts: list[float | None] = []
                    frame_no: list[int | None] = []
                    for ep_id, t in zip(eps, ts):
                        base = offsets.get(ep_id, {}).get(m.name)
                        if base is None:
                            abs_ts.append(None)
                            frame_no.append(None)
                        else:
                            a = base + t
                            abs_ts.append(a)
                            frame_no.append(int(round(a * fps)))
                    arrays[f"{m.name}__blob_id"] = [blob_id] * n
                    arrays[f"{m.name}__frame_in_blob"] = frame_no
                    arrays[f"{m.name}__ts_in_blob"] = abs_ts

                yield pa.RecordBatch.from_arrays(
                    [pa.array(arrays[f.name], type=f.type) for f in schema],
                    schema=schema,
                )
