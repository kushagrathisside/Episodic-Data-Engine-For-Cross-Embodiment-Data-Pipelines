"""Dataset exploration queries.

Answers the questions a person actually asks of an unfamiliar robotics dataset --
how many episodes, which failed QC, what is the duration distribution, what are
the state dimensions, show me examples of task X.

Two engines, each used where it is strongest:

**SQLite (catalog)** for metadata joins: versions, episode status, QC findings,
lineage.  Small, indexed, transactional.

**DuckDB (over Parquet)** for analytics on frame-scale data: percentiles over
19 k rows, per-episode aggregates, distribution histograms.  DuckDB reads the
Parquet files directly with projection and predicate pushdown, so a query over
columns it does not need costs nothing -- and it never loads the table into
Python memory.

The split matters: putting frame data in SQLite would throw away columnar scans,
and putting mutable QC status in Parquet would require rewriting files to mark an
episode rejected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..catalog import Catalog
from ..config import Settings, get_settings
from ..storage import ParquetTableStore


@dataclass
class DatasetOverview:
    """Headline facts about one version: counts, modalities, storage."""
    dataset_id: str
    version: str
    stage: str
    created_at: str
    episodes: int
    frames: int
    tasks: int
    fps: float | None
    robot_type: str | None
    duration_s: float
    modalities: list[dict[str, Any]]
    quality_status: str
    storage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON rendering."""
        return self.__dict__.copy()


class DatasetExplorer:
    """Read-only query surface over one dataset's versions."""

    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None):
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.paths.catalog_db)

    # -- helpers ------------------------------------------------------------
    def _frames_glob(self, dataset_id: str, version: str) -> str:
        """Parquet path DuckDB should scan, following the training pointer."""
        manifest = self.catalog.get_manifest(dataset_id, version)
        frame_version = version
        if manifest is not None:
            frame_version = manifest.statistics.get("frame_table_version", version)
        return str(self.settings.paths.version_dir(dataset_id, frame_version) / "frames" / "*.parquet")

    def _duckdb(self):
        import duckdb

        return duckdb.connect(":memory:")

    def sql(self, dataset_id: str, version: str, query: str) -> list[dict[str, Any]]:
        """Run arbitrary SQL against the frame table, exposed as ``frames``.

        Intended for exploration, not for pipeline code -- pipeline stages use
        the typed stores so their access patterns stay reviewable.
        """
        con = self._duckdb()
        try:
            con.execute(
                f"CREATE VIEW frames AS SELECT * FROM read_parquet('{self._frames_glob(dataset_id, version)}')"
            )
            cur = con.execute(query)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            con.close()

    # -- overview -----------------------------------------------------------
    def overview(self, dataset_id: str, version: str) -> DatasetOverview:
        """Episodes, frames, tasks, modalities, quality and storage footprint."""
        manifest = self.catalog.get_manifest(dataset_id, version)
        if manifest is None:
            raise LookupError(f"unknown version: {dataset_id}@{version}")
        rows = self.catalog.list_versions(dataset_id)
        meta = next((r for r in rows if r["version"] == version), {})
        episodes = self.catalog.get_episodes(dataset_id, version)

        tables = ParquetTableStore(self.settings.paths.version_dir(dataset_id, version))
        storage: dict[str, Any] = {}
        for name in ("frames", "episodes", "index"):
            if tables.exists(name):
                storage[name] = tables.stats(name)
        # A training version references its source's frame table instead of
        # copying it; report where the frames actually live so the storage
        # figures are not silently missing the bulk of the data.
        frame_version = manifest.statistics.get("frame_table_version")
        if frame_version and frame_version != version:
            ref = ParquetTableStore(self.settings.paths.version_dir(dataset_id, frame_version))
            if ref.exists("frames"):
                storage["frames"] = {**ref.stats("frames"), "referenced_from": frame_version}
        # Blob footprint is computed from the version's own references rather
        # than read from the manifest: only ingestion records it there, so a
        # derived version would otherwise report null despite referencing the
        # same (shared) video.
        from ..storage import BlobStore

        blob_store = BlobStore(self.settings.paths.blobs)
        own_tables = ParquetTableStore(self.settings.paths.version_dir(dataset_id, version))
        blob_ids: set[str] = set()
        if own_tables.exists("episodes"):
            for raw in own_tables.read_table("episodes", columns=["blob_ids"]).to_pydict()["blob_ids"]:
                blob_ids.update(json.loads(raw or "{}").values())
        storage["blob_bytes"] = sum(
            info.size_bytes for b in blob_ids if (info := blob_store.info(b))
        )
        storage["blob_count"] = len(blob_ids)
        storage["blobs_shared"] = True  # content-addressed; not owned by this version

        return DatasetOverview(
            dataset_id=dataset_id,
            version=version,
            stage=meta.get("stage", ""),
            created_at=meta.get("created_at", ""),
            episodes=len(episodes),
            frames=sum(e["num_frames"] for e in episodes),
            tasks=len(manifest.tasks),
            fps=manifest.fps,
            robot_type=manifest.robot_type,
            duration_s=sum(e["duration_s"] for e in episodes),
            modalities=[
                {
                    "name": m.name, "kind": m.kind.value, "plane": m.plane.value,
                    "dtype": m.dtype, "shape": list(m.shape), "fps": m.fps,
                }
                for m in manifest.modalities
            ],
            quality_status=meta.get("quality_status", "unknown"),
            storage=storage,
        )

    # -- distributions ------------------------------------------------------
    def duration_distribution(self, dataset_id: str, version: str) -> dict[str, Any]:
        """Episode duration and frame-count statistics, plus a histogram."""
        eps = self.catalog.get_episodes(dataset_id, version)
        if not eps:
            return {}
        import numpy as np

        d = np.array([e["duration_s"] for e in eps], dtype=float)
        f = np.array([e["num_frames"] for e in eps], dtype=float)
        return {
            "episodes": len(eps),
            "duration_s": {
                "min": round(float(d.min()), 3), "max": round(float(d.max()), 3),
                "mean": round(float(d.mean()), 3), "median": round(float(np.median(d)), 3),
                "p10": round(float(np.percentile(d, 10)), 3),
                "p90": round(float(np.percentile(d, 90)), 3),
                "total": round(float(d.sum()), 1),
            },
            "frames": {
                "min": int(f.min()), "max": int(f.max()),
                "mean": round(float(f.mean()), 1), "median": float(np.median(f)),
                "total": int(f.sum()),
            },
            "histogram": self._histogram(d.tolist(), bins=10),
        }

    @staticmethod
    def _histogram(values: list[float], bins: int = 10) -> list[dict[str, Any]]:
        import numpy as np

        counts, edges = np.histogram(np.asarray(values), bins=bins)
        return [
            {"lo": round(float(edges[i]), 3), "hi": round(float(edges[i + 1]), 3),
             "count": int(counts[i])}
            for i in range(len(counts))
        ]

    def task_summary(self, dataset_id: str, version: str) -> list[dict[str, Any]]:
        """Per-task episode, frame and duration totals."""
        manifest = self.catalog.get_manifest(dataset_id, version)
        descriptions = {t.task_id: t.description for t in (manifest.tasks if manifest else [])}
        eps = self.catalog.get_episodes(dataset_id, version)
        agg: dict[int, dict[str, Any]] = {}
        for e in eps:
            a = agg.setdefault(
                e["task_id"],
                {"task_id": e["task_id"], "description": descriptions.get(e["task_id"], "?"),
                 "episodes": 0, "frames": 0, "duration_s": 0.0},
            )
            a["episodes"] += 1
            a["frames"] += e["num_frames"]
            a["duration_s"] += e["duration_s"]
        for a in agg.values():
            a["duration_s"] = round(a["duration_s"], 1)
        return sorted(agg.values(), key=lambda a: -a["episodes"])

    def modality_summary(self, dataset_id: str, version: str) -> list[dict[str, Any]]:
        """Declared modalities with coverage and applied transforms."""
        manifest = self.catalog.get_manifest(dataset_id, version)
        if manifest is None:
            return []
        eps = self.catalog.get_episodes(dataset_id, version)
        total = len(eps) or 1
        out = []
        for m in manifest.modalities:
            present = sum(1 for e in eps if m.name in json.loads(e["modalities"] or "[]"))
            raw_names = m.attributes.get("element_names")
            out.append({
                "name": m.name, "kind": m.kind.value, "plane": m.plane.value,
                "dtype": m.dtype, "shape": list(m.shape), "fps": m.fps,
                "coverage": f"{present}/{total}",
                "element_names": (
                    [n for group in raw_names.values() for n in group]
                    if isinstance(raw_names, dict) else raw_names
                ),
                "normalized": bool(m.attributes.get("normalization")),
                "read_transforms": m.attributes.get("read_transforms", []),
            })
        return out

    def dimensionality(self, dataset_id: str, version: str) -> dict[str, Any]:
        """State/action dimensionality and per-dimension statistics via DuckDB."""
        manifest = self.catalog.get_manifest(dataset_id, version)
        if manifest is None:
            return {}
        out: dict[str, Any] = {}
        for m in manifest.modalities:
            if m.is_external:
                continue
            width = m.element_count
            # Canonical modality names are dotted ("state.joint"), which SQL
            # would otherwise parse as table.column -- always quote them.
            col = '"' + m.name.replace('"', '""') + '"'
            selects = ", ".join(
                f"min({col}[{i+1}]) AS min_{i}, max({col}[{i+1}]) AS max_{i}, "
                f"avg({col}[{i+1}]) AS mean_{i}, stddev({col}[{i+1}]) AS std_{i}"
                for i in range(width)
            )
            try:
                rows = self.sql(dataset_id, version, f'SELECT {selects} FROM frames')
            except Exception as exc:
                out[m.name] = {"dimensions": width, "error": str(exc)[:120]}
                continue
            r = rows[0] if rows else {}
            # Ingestion normalizes element_names to a flat list (see
            # rde.ingestion.lerobot._flatten_names), but manifests written
            # before that normalization existed may still carry the raw,
            # dict-grouped form (e.g. bimanual ALOHA's {"motors": [...]});
            # tolerate that shape here rather than crash on old data.
            raw_names = m.attributes.get("element_names")
            names = (
                [n for group in raw_names.values() for n in group]
                if isinstance(raw_names, dict) else raw_names
            )
            out[m.name] = {
                "dimensions": width,
                "element_names": names,
                "per_dimension": [
                    {
                        "index": i,
                        "name": (names[i] if names and i < len(names) else None),
                        "min": round(r.get(f"min_{i}", 0.0) or 0.0, 4),
                        "max": round(r.get(f"max_{i}", 0.0) or 0.0, 4),
                        "mean": round(r.get(f"mean_{i}", 0.0) or 0.0, 4),
                        "std": round(r.get(f"std_{i}", 0.0) or 0.0, 4),
                    }
                    for i in range(width)
                ],
            }
        return out

    def timestamp_frequency(self, dataset_id: str, version: str) -> dict[str, Any]:
        """Observed sampling frequency, measured rather than trusted."""
        rows = self.sql(
            dataset_id, version,
            """
            WITH d AS (
                SELECT episode_id,
                       timestamp_s - lag(timestamp_s) OVER (
                           PARTITION BY episode_id ORDER BY frame_index) AS dt
                FROM frames
            )
            SELECT count(*) AS intervals,
                   avg(dt) AS mean_dt, stddev(dt) AS std_dt,
                   min(dt) AS min_dt, max(dt) AS max_dt,
                   median(dt) AS median_dt
            FROM d WHERE dt IS NOT NULL
            """,
        )
        r = rows[0] if rows else {}
        mean_dt = r.get("mean_dt") or 0.0
        manifest = self.catalog.get_manifest(dataset_id, version)
        return {
            "declared_fps": manifest.fps if manifest else None,
            "observed_fps": round(1.0 / mean_dt, 4) if mean_dt else None,
            "intervals": r.get("intervals"),
            "mean_dt_ms": round(mean_dt * 1000, 4),
            "median_dt_ms": round((r.get("median_dt") or 0.0) * 1000, 4),
            "std_dt_ms": round((r.get("std_dt") or 0.0) * 1000, 6),
            "min_dt_ms": round((r.get("min_dt") or 0.0) * 1000, 4),
            "max_dt_ms": round((r.get("max_dt") or 0.0) * 1000, 4),
        }

    # -- quality ------------------------------------------------------------
    def qc_summary(self, dataset_id: str, version: str) -> list[dict[str, Any]]:
        """QC findings grouped by check and severity."""
        return self.catalog.qc_summary(dataset_id, version)

    def failed_episodes(self, dataset_id: str, version: str) -> list[dict[str, Any]]:
        """Episodes that failed QC, each with the findings that rejected them."""
        eps = self.catalog.get_episodes(dataset_id, version)
        bad = [e for e in eps if e["quality_status"] in ("rejected", "quarantined", "warning")]
        out = []
        for e in bad:
            findings = self.catalog.query_qc(dataset_id, version, episode_id=e["episode_id"])
            out.append({
                "episode_id": e["episode_id"],
                "status": e["quality_status"],
                "num_frames": e["num_frames"],
                "duration_s": round(e["duration_s"], 2),
                "findings": [
                    {"check": f["check_name"], "severity": f["severity"],
                     "reason": f["reason"], "observed": f["observed_value"]}
                    for f in findings
                ],
            })
        return out

    def episodes_with_missing_frames(self, dataset_id: str, version: str) -> list[dict[str, Any]]:
        """Episodes flagged by the structural/temporal gap checks."""
        out = []
        for check in ("structural.frame_index_continuity", "temporal.gaps",
                      "structural.episode_frame_count"):
            for f in self.catalog.query_qc(dataset_id, version, check_name=check):
                out.append({
                    "episode_id": f["episode_id"], "check": f["check_name"],
                    "reason": f["reason"], "observed": f["observed_value"],
                    "expected": f["expected_value"],
                })
        return out

    # -- examples -----------------------------------------------------------
    def examples_for_task(
        self, dataset_id: str, version: str, task_id: int, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Sample episodes for one task, with split and quality status."""
        eps = self.catalog.get_episodes(dataset_id, version, task_id=task_id)
        manifest = self.catalog.get_manifest(dataset_id, version)
        desc = next(
            (t.description for t in (manifest.tasks if manifest else []) if t.task_id == task_id),
            None,
        )
        return [
            {
                "episode_id": e["episode_id"], "task_id": task_id, "task": desc,
                "num_frames": e["num_frames"], "duration_s": round(e["duration_s"], 2),
                "quality_status": e["quality_status"],
                "split": json.loads(e["attributes"] or "{}").get("split"),
            }
            for e in eps[:limit]
        ]

    # -- lineage ------------------------------------------------------------
    def lineage(self, dataset_id: str, version: str) -> dict[str, Any]:
        """Full provenance chain for a version, plus contributing raw episodes."""
        graph = self.catalog.lineage_graph(dataset_id)
        chain = graph.path_to_root(version)
        runs = [
            {
                "operation": r.operation,
                "run_id": r.run_id,
                "inputs": r.input_versions,
                "output": r.output_version,
                "started_at": r.started_at.isoformat(),
                "duration_s": round(r.duration_s or 0.0, 3),
                "software_version": r.software_version,
                "code_fingerprint": r.code_fingerprint,
                "fingerprint": r.fingerprint(),
                "episodes_in": len(r.input_episode_ids),
                "episodes_out": len(r.output_episode_ids),
                "dropped": r.dropped_episode_ids,
                "parameters": r.parameters,
            }
            for r in chain
        ]
        return {
            "dataset_id": dataset_id,
            "version": version,
            "ancestors": graph.ancestors(version),
            "chain": runs,
            "root_episodes": graph.root_episodes(version),
        }
