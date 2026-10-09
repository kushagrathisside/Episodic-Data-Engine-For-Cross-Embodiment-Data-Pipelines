"""Analytical event rows: the facts that are too identifying to be metrics.

The cardinality measurement (docs/14 §3) draws the line this module implements.
Attaching ``episode_id`` to one QC metric costs 264 billion series and ~167 TB
of resident memory at the design planning scale.  The identical facts, stored
as columnar rows, measure **6 GB**.  Not because one store is better -- because
a TSDB pays for cardinality at write time and holds an index resident, while a
columnar store pays at query time and holds nothing.

So: *metrics aggregate populations; events identify individuals.*  Everything
here is an individual.

Storage follows the pattern the engine already uses for QC findings -- Parquet
as the authoritative detailed record, SQLite as the queryable mirror.  That is
not a new decision; it is the existing one, applied to a new fact type.  At
fleet scale the Parquet side migrates to ClickHouse and the queries move, but
the row shapes do not, because both engines read Parquet.

Rows are flat and typed.  Nested structures go in a single JSON column so the
schema stays stable as new health features are added -- a new drift feature must
not require a table migration.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


def _iso(dt: _dt.datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


@dataclass
class PipelineRunRow:
    """One pipeline execution.  The root of every correlation join.

    ``config_hash`` and ``software_version`` are what make two runs comparable:
    a duration change means something different when the config changed.
    """

    run_id: str
    trace_id: str
    dataset_id: str
    started_at: str
    finished_at: str | None
    duration_s: float | None
    status: str
    software_version: str
    config_hash: str
    host: str
    pid: int


@dataclass
class StageRunRow:
    """One stage within one run.

    ``records_in``/``records_out`` are the volume pillar at stage granularity:
    a stage that consumed 55,000 frames and emitted 12 is not a failure by any
    exception-based definition, and is obviously broken by this one.
    """

    run_id: str
    trace_id: str
    span_id: str
    dataset_id: str
    stage: str
    seq: int
    started_at: str
    finished_at: str | None
    duration_s: float | None
    status: str
    error_class: str | None = None
    records_in: int | None = None
    records_out: int | None = None
    #: Episodes touched. Separate from ``records_*`` because a frame and an
    #: episode are different units, and conflating them makes "18,668 records"
    #: and "48 episodes" look like the same measurement.
    episodes: int | None = None
    bytes_read: int | None = None
    bytes_written: int | None = None
    peak_rss_bytes: int | None = None
    attributes_json: str = "{}"


@dataclass
class EpisodeHealthRow:
    """One episode's health at one QC execution.

    The drift feature store.  One fixed-width numeric summary per episode, which
    is what makes episode-level drift comparison possible without re-reading
    frames -- and episode-level is the only defensible unit, because adjacent
    frames at 50 fps are not independent observations (docs/13 §5).

    ``qc_bitmask`` packs which checks fired into one integer, so cohort queries
    ("every episode where alignment failed but structure passed") are a bit-test
    rather than a join against the findings table.
    """

    run_id: str | None
    qc_execution_id: str
    dataset_id: str
    version: str
    episode_id: int
    quality_status: str
    frame_count: int
    duration_s: float
    modalities_present: int
    modalities_expected: int
    missing_ratio: float
    qc_bitmask: int = 0
    #: Per-family sub-scores in [0, 1]; 1.0 is clean.  Kept separate rather than
    #: averaged because "temporally perfect but misaligned" and "jittery but
    #: aligned" are different failures with different owners.
    temporal_quality: float = 1.0
    numeric_quality: float = 1.0
    alignment_quality: float = 1.0
    #: Episode-level drift features, JSON. Schema-free by design: adding a
    #: feature must not migrate the table.
    drift_features_json: str = "{}"
    checked_at: str = ""


@dataclass
class DriftReportRow:
    """One feature compared between two dataset versions.

    ``effective_sample_size`` is recorded next to the statistic, not derived
    later, because a drift number without it is uninterpretable: the block
    bootstrap achieves nominal coverage only at larger effective sample sizes
    (docs/13 §5), so a large statistic on 6 effective episodes is noise wearing
    a p-value.
    """

    run_id: str | None
    dataset_id: str
    baseline_version: str
    candidate_version: str
    feature: str
    statistic: str
    value: float
    threshold: float | None
    exceeded: bool
    baseline_n: int
    candidate_n: int
    effective_sample_size: float
    computed_at: str
    details_json: str = "{}"


@dataclass
class QCExecutionRow:
    """One validation of one version at one point in time.

    This row is what unblocks quality *trending*.  Today ``QCEngine._persist``
    clears a version's findings before writing new ones, so re-validating erases
    history (docs/12, G3) -- and the question that recalibration should have been
    able to answer ("did my check change help, and where?") is unanswerable.
    Scoping findings by execution makes it a query.
    """

    qc_execution_id: str
    run_id: str | None
    dataset_id: str
    version: str
    started_at: str
    finished_at: str | None
    duration_s: float | None
    checks_run: int
    findings: int
    episodes_total: int
    episodes_passed: int
    episodes_warned: int
    episodes_rejected: int
    episodes_quarantined: int
    dataset_status: str
    gate_policy: str = ""


#: Table name -> row class.  Drives both the Parquet writer and the catalog.
EVENT_TABLES: dict[str, type] = {
    "pipeline_runs": PipelineRunRow,
    "stage_runs": StageRunRow,
    "qc_executions": QCExecutionRow,
    "episode_health": EpisodeHealthRow,
    "drift_reports": DriftReportRow,
}


def rows_to_table(rows: Sequence[Any]) -> pa.Table:
    """Convert a homogeneous list of row dataclasses to an Arrow table.

    Types are inferred from the dataclass annotations rather than from the data,
    so an all-null column keeps its declared type instead of becoming ``null``
    and breaking the next append.
    """
    if not rows:
        raise ValueError("cannot infer a schema from zero rows; use empty_table")
    cls = type(rows[0])
    dicts = [asdict(r) for r in rows]
    return pa.table({
        name: pa.array([d[name] for d in dicts], type=_arrow_type(cls, name))
        for name in dicts[0]
    })


def empty_table(cls: type) -> pa.Table:
    """A zero-row table with the correct schema, for the no-events case."""
    fields = getattr(cls, "__annotations__", {})
    return pa.table({n: pa.array([], type=_arrow_type(cls, n)) for n in fields})


_ARROW_BY_ANNOTATION = {
    "int": pa.int64(),
    "float": pa.float64(),
    "bool": pa.bool_(),
    "str": pa.string(),
}


def _arrow_type(cls: type, name: str) -> pa.DataType:
    """Map a dataclass annotation to an Arrow type; unions become nullable."""
    ann = str(getattr(cls, "__annotations__", {}).get(name, "str"))
    base = ann.replace("| None", "").strip()
    return _ARROW_BY_ANNOTATION.get(base, pa.string())


class EventWriter:
    """Append-only Parquet event log, partitioned by table name.

    One file per table per run.  Appending to a single Parquet file is not
    supported by the format, and rewriting it would break the streaming
    guarantee the engine is built on -- so files accumulate and readers use a
    dataset glob, which is how partitioned analytical stores work anyway.
    """

    def __init__(self, directory: Path, compression: str = "zstd"):
        self.directory = Path(directory)
        self.compression = compression

    def write(self, table_name: str, rows: Sequence[Any], run_id: str) -> Path | None:
        """Write one table's rows.  Returns the path, or None if there were none."""
        if not rows:
            return None
        out_dir = self.directory / table_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{run_id}.parquet"
        pq.write_table(rows_to_table(rows), out, compression=self.compression)
        return out

    def read(self, table_name: str) -> pa.Table:
        """Read every partition of one table.  Empty table if none exist."""
        out_dir = self.directory / table_name
        files = sorted(out_dir.glob("*.parquet")) if out_dir.exists() else []
        if not files:
            return empty_table(EVENT_TABLES[table_name])
        return pa.concat_tables([pq.read_table(f) for f in files], promote_options="default")


class EventBuffer:
    """Collects rows during a run and writes them once at the end.

    Buffered rather than streamed because a run produces hundreds of rows, not
    millions -- one row per stage, one per episode -- and one Parquet file per
    table per run beats hundreds of tiny appends.  The frame-grain data that
    *would* justify streaming never becomes an event row in the first place.
    """

    def __init__(self) -> None:
        self._rows: dict[str, list[Any]] = {name: [] for name in EVENT_TABLES}

    def add(self, table_name: str, row: Any) -> None:
        """Buffer one row for ``table_name``; unknown tables are rejected."""
        if table_name not in self._rows:
            raise KeyError(f"unknown event table {table_name!r}")
        self._rows[table_name].append(row)

    def extend(self, table_name: str, rows: Sequence[Any]) -> None:
        """Buffer several rows for one table."""
        for r in rows:
            self.add(table_name, r)

    def get(self, table_name: str) -> list[Any]:
        """A copy of one table's buffered rows."""
        return list(self._rows[table_name])

    def counts(self) -> dict[str, int]:
        """Row count per non-empty table, for run summaries and tests."""
        return {k: len(v) for k, v in self._rows.items() if v}

    def flush(self, writer: EventWriter, run_id: str) -> dict[str, str]:
        """Write every non-empty table.  Returns table name -> path.

        Each table is written independently: a write failure on one (a
        malformed row raising ``ArrowInvalid``, say) must not take the
        others down with it. These rows are the audit trail, not disposable
        telemetry -- losing ``episode_health`` because ``stage_runs`` failed
        to write would be a second, compounding failure on top of the first.
        A failed table is reported under the ``"_errors"`` key rather than
        raised, so a caller can log/count it without a run-ending exception.
        """
        written: dict[str, str] = {}
        errors: dict[str, str] = {}
        for name, rows in self._rows.items():
            try:
                path = writer.write(name, rows, run_id)
            except Exception as exc:  # noqa: BLE001
                errors[name] = f"{type(exc).__name__}: {exc}"
                continue
            if path is not None:
                written[name] = str(path)
        if errors:
            written["_errors"] = json.dumps(errors, sort_keys=True)
        return written


def json_column(value: Any) -> str:
    """Serialise a nested payload for a ``*_json`` column."""
    return json.dumps(value, default=str, sort_keys=True)
