"""Metadata catalog -- the system of record for *what exists*.

Deliberately separate from both the blob store and the Parquet plane.  Blobs and
Parquet files are immutable bulk data; the catalog is small, mutable, relational
and transactional.  Forcing all three into one store would mean either putting
mutable status flags inside immutable Parquet (requiring rewrites to mark an
episode rejected) or putting 19 k frame rows into SQLite (throwing away columnar
scans).  Splitting them lets each do what it is good at.

SQLite is the backend because the catalog is metadata-scale -- tens of versions,
thousands of episodes -- and an embedded, transactional, zero-ops store is the
right size of tool.  DuckDB handles the analytical queries *over Parquet* in
:mod:`rde.exploration`; the two are complementary, not redundant.

Foreign keys cascade from ``versions``, so deleting a version cleans up its
episodes, tasks, modalities and QC index in one transaction.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..schemas.core import DatasetManifest, Episode, QualityStatus
from ..schemas.lineage import LineageGraph, LineageNode, TransformRun, VersionStage
from ..schemas.qc import QCRecord

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id      TEXT PRIMARY KEY,
    source          TEXT,
    source_format   TEXT,
    robot_type      TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS versions (
    version_key     TEXT PRIMARY KEY,
    dataset_id      TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    version         TEXT NOT NULL,
    stage           TEXT NOT NULL,
    schema_version  TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    total_episodes  INTEGER NOT NULL DEFAULT 0,
    total_frames    INTEGER NOT NULL DEFAULT 0,
    total_tasks     INTEGER NOT NULL DEFAULT 0,
    fps             REAL,
    quality_status  TEXT NOT NULL DEFAULT 'unknown',
    manifest_json   TEXT NOT NULL,
    UNIQUE (dataset_id, version)
);
CREATE INDEX IF NOT EXISTS idx_versions_stage ON versions(dataset_id, stage);

CREATE TABLE IF NOT EXISTS episodes (
    version_key     TEXT NOT NULL REFERENCES versions(version_key) ON DELETE CASCADE,
    episode_id      INTEGER NOT NULL,
    task_id         INTEGER,
    frame_start     INTEGER NOT NULL,
    frame_end       INTEGER NOT NULL,
    num_frames      INTEGER NOT NULL,
    duration_s      REAL NOT NULL,
    outcome         TEXT NOT NULL DEFAULT 'unknown',
    quality_status  TEXT NOT NULL DEFAULT 'unknown',
    modalities      TEXT,
    attributes      TEXT,
    PRIMARY KEY (version_key, episode_id)
);
CREATE INDEX IF NOT EXISTS idx_episodes_quality ON episodes(version_key, quality_status);
CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(version_key, task_id);

CREATE TABLE IF NOT EXISTS tasks (
    version_key     TEXT NOT NULL REFERENCES versions(version_key) ON DELETE CASCADE,
    task_id         INTEGER NOT NULL,
    description     TEXT NOT NULL,
    PRIMARY KEY (version_key, task_id)
);

CREATE TABLE IF NOT EXISTS modalities (
    version_key     TEXT NOT NULL REFERENCES versions(version_key) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    kind            TEXT NOT NULL,
    plane           TEXT NOT NULL,
    dtype           TEXT NOT NULL,
    shape           TEXT NOT NULL,
    fps             REAL,
    source_key      TEXT,
    attributes      TEXT,
    PRIMARY KEY (version_key, name)
);

CREATE TABLE IF NOT EXISTS transform_runs (
    run_id            TEXT PRIMARY KEY,
    dataset_id        TEXT NOT NULL,
    operation         TEXT NOT NULL,
    input_versions    TEXT NOT NULL,
    output_version    TEXT NOT NULL,
    parameters        TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    software_version  TEXT,
    code_fingerprint  TEXT,
    metrics           TEXT,
    dropped_episodes  TEXT,
    input_episodes    TEXT,
    output_episodes   TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_output ON transform_runs(output_version);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON transform_runs(dataset_id);

-- Episode-level provenance: answers "which raw episodes reached training/v1?"
-- without replaying every transformation.
CREATE TABLE IF NOT EXISTS episode_lineage (
    run_id              TEXT NOT NULL REFERENCES transform_runs(run_id) ON DELETE CASCADE,
    output_version      TEXT NOT NULL,
    output_episode_id   INTEGER NOT NULL,
    input_version       TEXT NOT NULL,
    input_episode_id    INTEGER NOT NULL,
    PRIMARY KEY (run_id, output_version, output_episode_id, input_version, input_episode_id)
);
CREATE INDEX IF NOT EXISTS idx_eplin_out ON episode_lineage(output_version, output_episode_id);
CREATE INDEX IF NOT EXISTS idx_eplin_in ON episode_lineage(input_version, input_episode_id);

-- Queryable index of QC findings.  The authoritative, full-detail records live in
-- Parquet next to the data; this mirror exists so the CLI can answer
-- "which episodes failed QC?" with a join instead of a scan.
CREATE TABLE IF NOT EXISTS qc_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    version_key     TEXT NOT NULL,
    dataset_id      TEXT NOT NULL,
    version         TEXT NOT NULL,
    check_name      TEXT NOT NULL,
    scope           TEXT NOT NULL,
    severity        TEXT NOT NULL,
    status          TEXT NOT NULL,
    episode_id      INTEGER,
    frame_id        INTEGER,
    modality        TEXT,
    observed_value  TEXT,
    expected_value  TEXT,
    reason          TEXT,
    details         TEXT,
    checked_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qc_version ON qc_records(version_key);
CREATE INDEX IF NOT EXISTS idx_qc_episode ON qc_records(version_key, episode_id);
CREATE INDEX IF NOT EXISTS idx_qc_check ON qc_records(version_key, check_name);

-- Observability: the time dimension the rest of this schema does not have.
-- The catalog answers "what is this dataset and how was it produced?"; these
-- three tables are what make "how has it been behaving?" a query rather than a
-- directory listing of JSON files (docs/12, G2 and G7).
--
-- Deliberately small and bounded: one row per run, one per stage, one per QC
-- execution.  The episode- and finding-grained facts stay in Parquet, because
-- SQLite is a catalog and must not be pushed toward being a telemetry store.
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id           TEXT PRIMARY KEY,
    trace_id         TEXT NOT NULL,
    dataset_id       TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    duration_s       REAL,
    status           TEXT NOT NULL,
    software_version TEXT,
    config_hash      TEXT,
    host             TEXT,
    pid              INTEGER
);
CREATE INDEX IF NOT EXISTS idx_runs_dataset_time
    ON pipeline_runs(dataset_id, started_at);

CREATE TABLE IF NOT EXISTS stage_runs (
    run_id         TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    seq            INTEGER NOT NULL,
    stage          TEXT NOT NULL,
    span_id        TEXT,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    duration_s     REAL,
    status         TEXT NOT NULL,
    error_class    TEXT,
    records_in     INTEGER,
    records_out    INTEGER,
    episodes       INTEGER,
    bytes_read     INTEGER,
    bytes_written  INTEGER,
    peak_rss_bytes INTEGER,
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_stage_runs_stage ON stage_runs(stage, started_at);

CREATE TABLE IF NOT EXISTS qc_executions (
    qc_execution_id       TEXT PRIMARY KEY,
    run_id                TEXT,
    dataset_id            TEXT NOT NULL,
    version               TEXT NOT NULL,
    started_at            TEXT NOT NULL,
    finished_at           TEXT,
    duration_s            REAL,
    checks_run            INTEGER NOT NULL DEFAULT 0,
    findings              INTEGER NOT NULL DEFAULT 0,
    episodes_total        INTEGER NOT NULL DEFAULT 0,
    episodes_passed       INTEGER NOT NULL DEFAULT 0,
    episodes_warned       INTEGER NOT NULL DEFAULT 0,
    episodes_rejected     INTEGER NOT NULL DEFAULT 0,
    episodes_quarantined  INTEGER NOT NULL DEFAULT 0,
    dataset_status        TEXT NOT NULL,
    gate_policy           TEXT
);
CREATE INDEX IF NOT EXISTS idx_qc_exec_version
    ON qc_executions(dataset_id, version, started_at);
"""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class Catalog:
    """Transactional metadata store over SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        # WAL lets exploration read while a pipeline stage writes.
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Transactional connection; commits on success, rolls back on exception."""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    #: Columns added after the first release, applied to existing databases.
    #: ``CREATE TABLE IF NOT EXISTS`` is a no-op on an existing table, so new
    #: columns need an explicit ALTER or an old catalog silently loses the data
    #: written to them.
    _MIGRATIONS: tuple[tuple[str, str, str], ...] = (
        ("transform_runs", "input_episodes", "TEXT"),
        ("transform_runs", "output_episodes", "TEXT"),
        # The correlation spine.  Nullable, because a TransformRun or QCRecord
        # produced outside a pipeline -- a direct library call -- legitimately
        # has no run above it (docs/14 §5).
        ("transform_runs", "pipeline_run_id", "TEXT"),
        ("qc_records", "pipeline_run_id", "TEXT"),
        ("qc_records", "qc_execution_id", "TEXT"),
        ("stage_runs", "episodes", "INTEGER"),
    )

    def _init_schema(self):
        with self.transaction() as conn:
            conn.executescript(SCHEMA_SQL)
            for table, column, decl in self._MIGRATIONS:
                existing = {
                    r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
                if existing and column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # -- datasets & versions ------------------------------------------------
    def register_dataset(self, manifest: DatasetManifest) -> None:
        """Insert or update the dataset row. Idempotent."""
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO datasets (dataset_id, source, source_format, robot_type, created_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(dataset_id) DO UPDATE SET
                       source=excluded.source,
                       source_format=excluded.source_format,
                       robot_type=excluded.robot_type""",
                (
                    manifest.dataset_id,
                    manifest.source,
                    manifest.source_format,
                    manifest.robot_type,
                    _now(),
                ),
            )

    def register_version(
        self, manifest: DatasetManifest, episodes: list[Episode] | None = None
    ) -> str:
        """Insert (or replace) a version and its episode/task/modality rows.

        Replace-on-conflict is intentional: re-running a stage should converge on
        the same catalog state rather than accumulating duplicates.
        """
        self.register_dataset(manifest)
        stage, _ = VersionStage.parse(manifest.version)
        vk = manifest.version_key

        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO versions
                   (version_key, dataset_id, version, stage, schema_version, created_at,
                    total_episodes, total_frames, total_tasks, fps, quality_status, manifest_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(version_key) DO UPDATE SET
                       total_episodes=excluded.total_episodes,
                       total_frames=excluded.total_frames,
                       total_tasks=excluded.total_tasks,
                       quality_status=excluded.quality_status,
                       manifest_json=excluded.manifest_json""",
                (
                    vk,
                    manifest.dataset_id,
                    manifest.version,
                    stage.value,
                    manifest.schema_version,
                    manifest.created_at.isoformat(),
                    manifest.total_episodes,
                    manifest.total_frames,
                    manifest.total_tasks,
                    manifest.fps,
                    manifest.quality_status.value,
                    manifest.model_dump_json(),
                ),
            )

            conn.execute("DELETE FROM tasks WHERE version_key=?", (vk,))
            conn.executemany(
                "INSERT INTO tasks (version_key, task_id, description) VALUES (?,?,?)",
                [(vk, t.task_id, t.description) for t in manifest.tasks],
            )

            conn.execute("DELETE FROM modalities WHERE version_key=?", (vk,))
            conn.executemany(
                """INSERT INTO modalities
                   (version_key, name, kind, plane, dtype, shape, fps, source_key, attributes)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        vk,
                        m.name,
                        m.kind.value,
                        m.plane.value,
                        m.dtype,
                        json.dumps(list(m.shape)),
                        m.fps,
                        m.source_key,
                        json.dumps(m.attributes),
                    )
                    for m in manifest.modalities
                ],
            )

            if episodes is not None:
                conn.execute("DELETE FROM episodes WHERE version_key=?", (vk,))
                conn.executemany(
                    """INSERT INTO episodes
                       (version_key, episode_id, task_id, frame_start, frame_end,
                        num_frames, duration_s, outcome, quality_status, modalities, attributes)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    [
                        (
                            vk,
                            e.episode_id,
                            e.task_id,
                            e.frame_start,
                            e.frame_end,
                            e.num_frames,
                            e.duration_s,
                            e.outcome.value,
                            e.quality_status.value,
                            json.dumps(e.modalities),
                            json.dumps(e.attributes),
                        )
                        for e in episodes
                    ],
                )
        return vk

    def get_manifest(self, dataset_id: str, version: str) -> DatasetManifest | None:
        """Full manifest for one version, or None if unknown."""
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT manifest_json FROM versions WHERE version_key=?",
                (f"{dataset_id}@{version}",),
            ).fetchone()
        return DatasetManifest.model_validate_json(row["manifest_json"]) if row else None

    def list_versions(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        """Version rows, oldest first, optionally scoped to one dataset."""
        q = """SELECT version_key, dataset_id, version, stage, created_at,
                      total_episodes, total_frames, quality_status FROM versions"""
        args: tuple = ()
        if dataset_id:
            q += " WHERE dataset_id=?"
            args = (dataset_id,)
        q += " ORDER BY created_at"
        with self.transaction() as conn:
            return [dict(r) for r in conn.execute(q, args).fetchall()]

    def list_datasets(self) -> list[dict[str, Any]]:
        """Every registered dataset, oldest first."""
        with self.transaction() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM datasets ORDER BY created_at"
                ).fetchall()
            ]

    def version_exists(self, dataset_id: str, version: str) -> bool:
        """Whether a given dataset@version has been registered."""
        with self.transaction() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM versions WHERE version_key=?",
                    (f"{dataset_id}@{version}",),
                ).fetchone()
                is not None
            )

    def set_version_quality(
        self, dataset_id: str, version: str, status: QualityStatus
    ) -> None:
        """Stamp the gate's verdict onto a whole version."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE versions SET quality_status=? WHERE version_key=?",
                (status.value, f"{dataset_id}@{version}"),
            )

    # -- episodes -----------------------------------------------------------
    def get_episodes(
        self,
        dataset_id: str,
        version: str,
        quality_status: QualityStatus | None = None,
        task_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Episode rows, optionally filtered by quality status or task."""
        q = "SELECT * FROM episodes WHERE version_key=?"
        args: list[Any] = [f"{dataset_id}@{version}"]
        if quality_status is not None:
            q += " AND quality_status=?"
            args.append(quality_status.value)
        if task_id is not None:
            q += " AND task_id=?"
            args.append(task_id)
        q += " ORDER BY episode_id"
        with self.transaction() as conn:
            return [dict(r) for r in conn.execute(q, args).fetchall()]

    def set_episode_quality(
        self,
        dataset_id: str,
        version: str,
        statuses: dict[int, QualityStatus],
    ) -> None:
        """Apply per-episode gate verdicts in one transaction."""
        vk = f"{dataset_id}@{version}"
        with self.transaction() as conn:
            conn.executemany(
                "UPDATE episodes SET quality_status=? WHERE version_key=? AND episode_id=?",
                [(s.value, vk, eid) for eid, s in statuses.items()],
            )

    # -- lineage ------------------------------------------------------------
    def record_run(self, run: TransformRun, episode_map: dict[int, int] | None = None):
        """Persist a lineage edge, plus optional output->input episode mapping."""
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO transform_runs
                   (run_id, dataset_id, operation, input_versions, output_version,
                    parameters, started_at, finished_at, software_version,
                    code_fingerprint, metrics, dropped_episodes,
                    input_episodes, output_episodes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id) DO UPDATE SET
                       finished_at=excluded.finished_at,
                       metrics=excluded.metrics,
                       input_episodes=excluded.input_episodes,
                       output_episodes=excluded.output_episodes""",
                (
                    run.run_id,
                    run.dataset_id,
                    run.operation,
                    json.dumps(run.input_versions),
                    run.output_version,
                    json.dumps(run.parameters, default=str),
                    run.started_at.isoformat(),
                    run.finished_at.isoformat() if run.finished_at else None,
                    run.software_version,
                    run.code_fingerprint,
                    json.dumps(run.metrics, default=str),
                    json.dumps(run.dropped_episode_ids),
                    json.dumps(run.input_episode_ids),
                    json.dumps(run.output_episode_ids),
                ),
            )
            if episode_map:
                src_version = run.input_versions[0] if run.input_versions else ""
                conn.executemany(
                    """INSERT OR IGNORE INTO episode_lineage
                       (run_id, output_version, output_episode_id, input_version, input_episode_id)
                       VALUES (?,?,?,?,?)""",
                    [
                        (run.run_id, run.output_version, out_id, src_version, in_id)
                        for out_id, in_id in episode_map.items()
                    ],
                )

    def get_runs(self, dataset_id: str) -> list[TransformRun]:
        """Every lineage edge for a dataset, oldest first."""
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM transform_runs WHERE dataset_id=? ORDER BY started_at",
                (dataset_id,),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                TransformRun(
                    run_id=r["run_id"],
                    dataset_id=r["dataset_id"],
                    operation=r["operation"],
                    input_versions=json.loads(r["input_versions"]),
                    output_version=r["output_version"],
                    parameters=json.loads(r["parameters"]),
                    started_at=_dt.datetime.fromisoformat(r["started_at"]),
                    finished_at=(
                        _dt.datetime.fromisoformat(r["finished_at"])
                        if r["finished_at"]
                        else None
                    ),
                    software_version=r["software_version"] or "",
                    code_fingerprint=r["code_fingerprint"] or "",
                    metrics=json.loads(r["metrics"] or "{}"),
                    dropped_episode_ids=json.loads(r["dropped_episodes"] or "[]"),
                    input_episode_ids=json.loads(
                        (r["input_episodes"] if "input_episodes" in r.keys() else None) or "[]"
                    ),
                    output_episode_ids=json.loads(
                        (r["output_episodes"] if "output_episodes" in r.keys() else None) or "[]"
                    ),
                )
            )
        return out

    def lineage_graph(self, dataset_id: str) -> LineageGraph:
        """Materialise the provenance DAG: version nodes plus run edges."""
        nodes: dict[str, LineageNode] = {}
        for v in self.list_versions(dataset_id):
            nodes[v["version_key"]] = LineageNode(
                version_key=v["version_key"],
                dataset_id=v["dataset_id"],
                version=v["version"],
                stage=VersionStage(v["stage"]),
                created_at=_dt.datetime.fromisoformat(v["created_at"]),
                total_episodes=v["total_episodes"],
                total_frames=v["total_frames"],
            )
        return LineageGraph(
            dataset_id=dataset_id, nodes=nodes, edges=self.get_runs(dataset_id)
        )

    def trace_episodes_to_root(
        self, dataset_id: str, version: str
    ) -> dict[int, list[tuple[str, int]]]:
        """Map each episode of ``version`` to its ancestry, oldest step last.

        This is the query that makes "which raw episodes contributed to
        training/v1?" a single traversal rather than a forensic exercise.
        """
        # episode_lineage carries no dataset_id column of its own -- version
        # strings are per-dataset (e.g. every dataset's first clean version is
        # "clean/v1"), so a query keyed on output_version/input_version alone
        # would silently mix or cross-return another dataset's provenance
        # chain whenever two datasets happen to share a version string, which
        # is the common case, not an edge case. Every row is reachable only
        # through a transform_runs row that does carry dataset_id, so joining
        # through it scopes both the seed query and the recursive walk
        # without a schema migration.
        result: dict[int, list[tuple[str, int]]] = {}
        with self.transaction() as conn:
            current = conn.execute(
                """SELECT DISTINCT el.output_episode_id
                   FROM episode_lineage el
                   JOIN transform_runs tr ON tr.run_id = el.run_id
                   WHERE tr.dataset_id=? AND el.output_version=?""",
                (dataset_id, version),
            ).fetchall()
            for row in current:
                eid = row["output_episode_id"]
                chain: list[tuple[str, int]] = []
                probe_v, probe_e = version, eid
                for _ in range(50):  # depth guard against a malformed cycle
                    parent = conn.execute(
                        """SELECT el.input_version, el.input_episode_id
                           FROM episode_lineage el
                           JOIN transform_runs tr ON tr.run_id = el.run_id
                           WHERE tr.dataset_id=? AND el.output_version=?
                                 AND el.output_episode_id=?
                           LIMIT 1""",
                        (dataset_id, probe_v, probe_e),
                    ).fetchone()
                    if parent is None or not parent["input_version"]:
                        break
                    chain.append((parent["input_version"], parent["input_episode_id"]))
                    probe_v, probe_e = parent["input_version"], parent["input_episode_id"]
                result[eid] = chain
        return result

    # -- QC index -----------------------------------------------------------
    def record_qc(
        self,
        records: list[QCRecord],
        pipeline_run_id: str | None = None,
        qc_execution_id: str | None = None,
    ) -> None:
        """Append findings to the queryable QC mirror.

        The two id arguments are the correlation spine (docs/14 §5).  Both are
        optional and default to NULL, because ``QCEngine.run`` called directly
        as a library has no pipeline above it -- observability that made the
        library harder to use would have failed.
        """
        if not records:
            return
        with self.transaction() as conn:
            conn.executemany(
                """INSERT INTO qc_records
                   (version_key, dataset_id, version, check_name, scope, severity, status,
                    episode_id, frame_id, modality, observed_value, expected_value,
                    reason, details, checked_at, pipeline_run_id, qc_execution_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        f"{r.dataset_id}@{r.version}",
                        r.dataset_id,
                        r.version,
                        r.check_name,
                        r.scope.value,
                        r.severity.value,
                        r.status.value,
                        r.episode_id,
                        r.frame_id,
                        r.modality,
                        r.observed_value,
                        r.expected_value,
                        r.reason,
                        json.dumps(r.details, default=str) if r.details else None,
                        r.checked_at.isoformat(),
                        pipeline_run_id,
                        qc_execution_id,
                    )
                    for r in records
                ],
            )

    def clear_qc(self, dataset_id: str, version: str) -> None:
        """Drop prior findings so a re-run replaces rather than duplicates them."""
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM qc_records WHERE version_key=?",
                (f"{dataset_id}@{version}",),
            )

    def query_qc(
        self,
        dataset_id: str,
        version: str,
        episode_id: int | None = None,
        check_name: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        """QC findings, optionally narrowed by episode, check or severity."""
        q = "SELECT * FROM qc_records WHERE version_key=?"
        args: list[Any] = [f"{dataset_id}@{version}"]
        for col, val in (
            ("episode_id", episode_id),
            ("check_name", check_name),
            ("severity", severity),
        ):
            if val is not None:
                q += f" AND {col}=?"
                args.append(val)
        q += " ORDER BY episode_id, check_name"
        with self.transaction() as conn:
            return [dict(r) for r in conn.execute(q, args).fetchall()]

    def qc_summary(self, dataset_id: str, version: str) -> list[dict[str, Any]]:
        """Findings grouped by check, severity and status, with episode counts."""
        with self.transaction() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    """SELECT check_name, severity, status, COUNT(*) AS n,
                              COUNT(DISTINCT episode_id) AS episodes
                       FROM qc_records WHERE version_key=?
                       GROUP BY check_name, severity, status ORDER BY n DESC""",
                    (f"{dataset_id}@{version}",),
                ).fetchall()
            ]

    def delete_version(self, dataset_id: str, version: str) -> bool:
        """Remove one version and everything that cascades from it.

        Episodes, tasks and modalities are removed by the FK cascade declared
        in the schema; QC records are not (they carry no FK, deliberately, so a
        finding survives its version being dropped in normal use) and so are
        deleted explicitly here.

        Intended for disposable scratch versions -- the fault-injection and
        observability demos clone a real version, corrupt the clone, and must
        leave the catalog telling the truth afterwards.  It is not part of the
        normal lifecycle: real versions are immutable, and superseding one means
        creating the next, never deleting the last.
        """
        version_key = f"{dataset_id}@{version}"
        with self.transaction() as conn:
            conn.execute("DELETE FROM qc_records WHERE version_key = ?", (version_key,))
            # transform_runs/episode_lineage carry no FK to `versions`, so they
            # would otherwise survive this delete. Left in place, a later
            # reuse of the same version string -- exactly what happens when a
            # scratch version is cloned, corrupted, and dropped for a
            # fault-injection or observability demo -- collides with the
            # orphaned run: two transform_runs rows claim the same
            # (dataset_id, output_version), and lineage_graph/
            # trace_episodes_to_root can no longer tell which one is current.
            # episode_lineage rows for the deleted run cascade away with it
            # (ON DELETE CASCADE on episode_lineage.run_id).
            conn.execute(
                "DELETE FROM transform_runs WHERE dataset_id=? AND output_version=?",
                (dataset_id, version),
            )
            cur = conn.execute(
                "DELETE FROM versions WHERE version_key = ?", (version_key,)
            )
            return cur.rowcount > 0

    # -- observability ------------------------------------------------------
    def record_pipeline_run(self, row: Any) -> None:
        """Upsert one pipeline-run header.

        Idempotent on ``run_id`` so a run that writes at start and again at
        finish updates in place rather than duplicating.
        """
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO pipeline_runs
                   (run_id, trace_id, dataset_id, started_at, finished_at,
                    duration_s, status, software_version, config_hash, host, pid)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id) DO UPDATE SET
                       finished_at=excluded.finished_at,
                       duration_s=excluded.duration_s,
                       status=excluded.status""",
                (row.run_id, row.trace_id, row.dataset_id, row.started_at,
                 row.finished_at, row.duration_s, row.status,
                 row.software_version, row.config_hash, row.host, row.pid),
            )

    def record_stage_runs(self, rows: Sequence[Any]) -> None:
        """Insert this run's stage rows, replacing any prior attempt."""
        if not rows:
            return
        with self.transaction() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO stage_runs
                   (run_id, seq, stage, span_id, started_at, finished_at,
                    duration_s, status, error_class, records_in, records_out,
                    episodes, bytes_read, bytes_written, peak_rss_bytes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [(r.run_id, r.seq, r.stage, r.span_id, r.started_at,
                  r.finished_at, r.duration_s, r.status, r.error_class,
                  r.records_in, r.records_out, r.episodes, r.bytes_read,
                  r.bytes_written, r.peak_rss_bytes) for r in rows],
            )

    def record_qc_executions(self, rows: Sequence[Any]) -> None:
        """Insert QC execution headers.

        Appended, never cleared.  This is the fix for the history loss recon
        found (G3): ``qc_records`` is replaced on re-validation, but the
        execution headers accumulate, so "how did this version's QC outcome
        change over time?" survives a re-run.
        """
        if not rows:
            return
        with self.transaction() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO qc_executions
                   (qc_execution_id, run_id, dataset_id, version, started_at,
                    finished_at, duration_s, checks_run, findings,
                    episodes_total, episodes_passed, episodes_warned,
                    episodes_rejected, episodes_quarantined, dataset_status,
                    gate_policy)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [(r.qc_execution_id, r.run_id, r.dataset_id, r.version,
                  r.started_at, r.finished_at, r.duration_s, r.checks_run,
                  r.findings, r.episodes_total, r.episodes_passed,
                  r.episodes_warned, r.episodes_rejected,
                  r.episodes_quarantined, r.dataset_status, r.gate_policy)
                 for r in rows],
            )

    def run_history(self, dataset_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Recent pipeline runs, newest first."""
        with self.transaction() as conn:
            return [dict(r) for r in conn.execute(
                """SELECT * FROM pipeline_runs WHERE dataset_id = ?
                   ORDER BY started_at DESC LIMIT ?""",
                (dataset_id, limit),
            ).fetchall()]

    def stage_history(
        self, dataset_id: str, stage: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Stage executions joined to their runs, newest first.

        The query recon G2 could not answer: *has ``validate`` gotten slower?*
        """
        sql = """SELECT s.*, p.dataset_id, p.started_at AS run_started_at
                 FROM stage_runs s JOIN pipeline_runs p ON p.run_id = s.run_id
                 WHERE p.dataset_id = ?"""
        params: list[Any] = [dataset_id]
        if stage:
            sql += " AND s.stage = ?"
            params.append(stage)
        sql += " ORDER BY p.started_at DESC, s.seq ASC LIMIT ?"
        params.append(limit)
        with self.transaction() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def qc_execution_history(
        self, dataset_id: str, version: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """QC executions for a dataset, newest first.  Quality over time."""
        sql = "SELECT * FROM qc_executions WHERE dataset_id = ?"
        params: list[Any] = [dataset_id]
        if version:
            sql += " AND version = ?"
            params.append(version)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        with self.transaction() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
