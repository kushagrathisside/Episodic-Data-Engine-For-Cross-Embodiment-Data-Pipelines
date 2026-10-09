"""QC engine: runs the check suite, persists findings, gates versions.

Execution model -- one streaming pass, three phases:

1. **pre-dataset checks** (priority < 0) populate ``ctx.shared``; the range
   profiler learns bounds here so episode checks have something to compare to.
2. **episode pass** streams the frame table grouped by episode and fans each
   episode out to every episode check.  One read, N checks.
3. **post-dataset checks** summarise distributions and cross-episode invariants.

Gating policy is deliberately separate from checking.  A check reports what it
saw; the :class:`QCGate` decides what that means for downstream use.  The same
findings can therefore produce a strict training dataset and a permissive
exploration dataset without re-running validation.

Nothing is deleted.  Rejected episodes are marked, their findings are persisted,
and the clean version simply does not select them -- so "why is episode 17
missing from training/v1?" is answered by a query, not an archaeology dig.
"""

from __future__ import annotations

import datetime as _dt
import json
import time as _time
import uuid as _uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow as pa
import pyarrow.dataset as pads

from ..catalog import Catalog
from .cache import (
    QCCache,
    blob_fingerprint,
    cached_report,
    suite_fingerprint,
    table_fingerprint,
)
from ..config import Settings, get_settings
from ..observability.events import QCExecutionRow
from ..observability.health import (
    SHARED_KEY,
    EpisodeFeatureProfiler,
    emit_data_health,
    summarize_episode_health,
)
from ..observability.session import current_session
from ..schemas.core import DatasetManifest, Episode, QualityStatus
from ..schemas.qc import CheckStatus, QCRecord, QCReport, Severity
from ..storage import BlobStore, ParquetTableStore
from ..validation import CheckContext, CheckRegistry, default_registry


@dataclass
class QCGate:
    """Policy that turns findings into per-episode quality statuses.

    ``block_on`` lists the severities that reject an episode.  ``max_warnings``
    quarantines an episode that accumulates many non-blocking findings -- no
    single one is fatal, but a demonstration tripping eight warnings deserves a
    human look rather than silent inclusion.
    """

    block_on: tuple[Severity, ...] = (Severity.ERROR, Severity.CRITICAL)
    max_warnings: int | None = None
    #: Checks whose findings never gate, whatever their severity.
    advisory_checks: tuple[str, ...] = ()

    def classify(self, records: Sequence[QCRecord]) -> QualityStatus:
        """Turn one episode's findings into a quality status."""
        blocking = [
            r for r in records
            if r.check_name not in self.advisory_checks
            and r.status is CheckStatus.FAILED
            and r.severity in self.block_on
        ]
        if blocking:
            return QualityStatus.REJECTED
        warnings = [
            r for r in records
            if r.check_name not in self.advisory_checks
            and r.status is CheckStatus.WARNED
        ]
        if self.max_warnings is not None and len(warnings) > self.max_warnings:
            return QualityStatus.QUARANTINED
        if warnings:
            return QualityStatus.WARNING
        return QualityStatus.PASSED


@dataclass
class QCResult:
    """Findings, per-episode statuses and the artifacts they were written to."""
    report: QCReport
    episode_status: dict[int, QualityStatus] = field(default_factory=dict)
    dataset_status: QualityStatus = QualityStatus.UNKNOWN
    artifacts: dict[str, str] = field(default_factory=dict)
    shared: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Report summary plus the gated dataset and per-episode statuses."""
        counts: dict[str, int] = {}
        for s in self.episode_status.values():
            counts[s.value] = counts.get(s.value, 0) + 1
        return {
            **self.report.summary(),
            "dataset_status": self.dataset_status.value,
            "episodes_by_status": counts,
        }


def qc_records_to_table(records: Sequence[QCRecord]) -> pa.Table:
    """Serialise findings to the QC Parquet table (the authoritative copy)."""
    rows = [r.as_row() for r in records]
    cols = [
        "check_name", "scope", "severity", "status", "dataset_id", "version",
        "episode_id", "frame_id", "modality", "observed_value", "expected_value",
        "reason", "details", "checked_at",
    ]
    types = {
        "episode_id": pa.int64(), "frame_id": pa.int64(),
        "checked_at": pa.timestamp("us", tz="UTC"),
    }
    return pa.table(
        {c: pa.array([r.get(c) for r in rows], type=types.get(c, pa.string())) for c in cols}
    )


class QCEngine:
    """Runs a check suite over one dataset version."""

    def __init__(
        self,
        settings: Settings | None = None,
        catalog: Catalog | None = None,
        registry: CheckRegistry | None = None,
        gate: QCGate | None = None,
        collect_features: bool | None = None,
        incremental: bool = False,
    ):
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.paths.catalog_db)
        self.registry = registry or default_registry()
        self.gate = gate or QCGate()
        #: Set by ``_emit_telemetry``; stamped onto the persisted findings so a
        #: finding can be traced back to the validation that produced it.
        self._last_execution_id: str | None = None
        #: Whether to attach the episode-feature profiler.  ``None`` means
        #: "follow the session" -- profile when a run is being observed.  Set
        #: explicitly by callers that want the features without the telemetry,
        #: such as contract inference, which needs the baseline's distribution
        #: but is not itself a pipeline run.
        self.collect_features = collect_features
        #: Reuse a previous validation when the version, its blobs and the check
        #: suite are all unchanged.  **Opt-in, deliberately.**  A stale QC result
        #: is a correctness failure in the component whose entire job is
        #: correctness, so reusing one should be a decision somebody made rather
        #: than a default they inherited.  See ``rde/qc/cache.py`` for what the
        #: key covers and what it does not.
        self.incremental = incremental
        self._cache = QCCache(self.catalog) if incremental else None
        #: Whether the last ``run()`` was served from cache.
        self.last_run_cached = False

    # -- episode streaming --------------------------------------------------
    def _iter_episode_frames(
        self, tables: ParquetTableStore, episodes: list[Episode], columns: list[str]
    ) -> Iterator[tuple[Episode, pa.Table]]:
        """Yield (episode, its frames) reading only the requested columns.

        Uses one filtered scan per episode.  With bounded row groups this touches
        only the groups an episode occupies, and peak memory is one episode's
        frames -- ~400 rows here -- rather than the whole table.
        """
        available = set(tables.schema("frames").names)
        cols = [c for c in columns if c in available]
        for ep in episodes:
            frames = tables.read_table(
                "frames",
                columns=cols or None,
                filter=pads.field("episode_id") == ep.episode_id,
            )
            yield ep, frames

    def run(
        self,
        dataset_id: str,
        version: str,
        checks: Sequence[str] | None = None,
        persist: bool = True,
    ) -> QCResult:
        """Validate a version and return findings plus per-episode statuses."""
        manifest = self.catalog.get_manifest(dataset_id, version)
        if manifest is None:
            raise LookupError(f"unknown version: {dataset_id}@{version}")

        ep_rows = self.catalog.get_episodes(dataset_id, version)
        episodes = [
            Episode(
                episode_id=r["episode_id"], dataset_id=dataset_id, task_id=r["task_id"],
                frame_start=r["frame_start"], frame_end=r["frame_end"],
                end_timestamp_s=r["duration_s"],
                quality_status=QualityStatus(r["quality_status"]),
                modalities=json.loads(r["modalities"] or "[]"),
                attributes=json.loads(r["attributes"] or "{}"),
            )
            for r in ep_rows
        ]

        tables = ParquetTableStore(
            self.settings.paths.version_dir(dataset_id, version),
            compression=self.settings.parquet_compression,
            row_group_size=self.settings.row_group_size,
        )
        # blob_ids live in the episode Parquet table rather than the catalog,
        # because they are bulk per-episode data, not queryable metadata.
        if tables.exists("episodes"):
            et = tables.read_table("episodes", columns=["episode_id", "blob_ids"]).to_pydict()
            blob_map = {
                int(e): json.loads(b or "{}")
                for e, b in zip(et["episode_id"], et["blob_ids"])
            }
            for ep in episodes:
                ep.blob_ids = blob_map.get(ep.episode_id, {})

        registry = self.registry.select(checks)
        session = current_session()

        # -- incremental short-circuit --------------------------------------
        self.last_run_cached = False
        fingerprints: tuple[str, str] | None = None
        if self._cache is not None:
            version_dir = self.settings.paths.version_dir(dataset_id, version)
            blob_ids = [
                b for ep in episodes for b in (ep.blob_ids or {}).values()
            ]
            plane_hash = (
                table_fingerprint(version_dir)
                + blob_fingerprint(BlobStore(self.settings.paths.blobs), blob_ids)
            )
            suite_hash = suite_fingerprint(registry.all())
            fingerprints = (suite_hash, plane_hash)
            hit = self._cache.lookup(dataset_id, version, suite_hash, plane_hash)
            if hit is not None:
                records_c, statuses_c, ds_status, checks_run_c = hit
                self.last_run_cached = True
                result = QCResult(
                    report=cached_report(dataset_id, version, records_c, checks_run_c),
                    episode_status=statuses_c,
                    dataset_status=ds_status,
                    shared={"cache": "hit"},
                )
                if session is not None and session.enabled:
                    session.inc("rde_qc_checks_run_total", 0.0,
                                check_name="cache", dataset_id=dataset_id)
                # A cache hit is a fast re-derivation of the same findings,
                # not a reason to skip writing them: the caller asked for
                # persist=True to get a catalog/Parquet record out of this
                # call, and a cache-only version would otherwise sit at
                # quality_status=UNKNOWN forever if nothing else ever calls
                # run() without a cache hit for it.
                if persist:
                    result.artifacts = self._persist(
                        dataset_id, version, result, tables,
                        pipeline_run_id=session.ctx.run_id if session and session.enabled else None,
                        qc_execution_id=self._last_execution_id,
                    )
                return result
        want_features = (
            self.collect_features
            if self.collect_features is not None
            else (session is not None and session.enabled)
        )
        if want_features:
            # The feature profiler is observability, not validation, so it is
            # never in the default suite -- the engine's 24 checks stay 24.  It
            # is attached only when a run is being observed, and it rides the
            # episode pass that is about to happen anyway rather than adding a
            # traversal of its own.
            try:
                registry.register(EpisodeFeatureProfiler())
            except ValueError:
                pass  # already attached by a caller

        ctx = CheckContext(
            manifest=manifest,
            episodes=episodes,
            table_store=tables,
            blob_store=BlobStore(self.settings.paths.blobs),
            settings=self.settings,
            version_dir=self.settings.paths.version_dir(dataset_id, version),
        )

        report = QCReport(
            dataset_id=dataset_id,
            version=version,
            started_at=_dt.datetime.now(_dt.timezone.utc),
        )
        records: list[QCRecord] = []

        #: check_name -> seconds accumulated across the whole episode pass.
        #: Recon G5 found timing granularity stopped at the eight pipeline
        #: stages, so the dominant stage -- validate, 2.7s of a 6.8s run -- was
        #: internally invisible.  This is what makes "which check is slow?"
        #: answerable.
        check_seconds: dict[str, float] = {}

        def run_check(check, producer):
            """Execute one check, converting an exception into a finding.

            A crashing check must not abort the run: the other checks' findings
            are still valuable, and the crash itself is a QC result worth
            recording.
            """
            if not check.applies_to(ctx):
                records.append(
                    ctx.record(
                        check.name, check.scope, Severity.INFO, CheckStatus.SKIPPED,
                        check.skip_reason(ctx),
                    )
                )
                return
            t_check = _time.perf_counter()
            try:
                records.extend(producer())
            except Exception as exc:
                records.append(
                    ctx.record(
                        check.name, check.scope, Severity.CRITICAL, CheckStatus.FAILED,
                        f"check raised {type(exc).__name__}: {exc}",
                        observed=type(exc).__name__, expected="clean execution",
                    )
                )
            finally:
                check_seconds[check.name] = (
                    check_seconds.get(check.name, 0.0)
                    + (_time.perf_counter() - t_check)
                )

        # phase 1 -- profilers that seed ctx.shared
        for check in registry.dataset_checks("pre"):
            report.checks_run.append(check.name)
            run_check(check, lambda c=check: list(c.check_dataset(ctx)))

        # phase 2 -- single streaming pass over episodes
        ep_checks = registry.episode_checks()
        if ep_checks:
            columns = sorted(
                {"frame_id", "episode_id", "frame_index", "timestamp_s"}
                | {c for chk in ep_checks for c in chk.required_columns}
                | {c for chk in ep_checks for c in chk.dynamic_columns(ctx)}
                | {m.name for m in manifest.modalities if not m.is_external}
            )
            for check in ep_checks:
                report.checks_run.append(check.name)
            applicable = []
            for check in ep_checks:
                if check.applies_to(ctx):
                    applicable.append(check)
                else:
                    records.append(
                        ctx.record(
                            check.name, check.scope, Severity.INFO,
                            CheckStatus.SKIPPED, check.skip_reason(ctx),
                        )
                    )
            for ep, frames in self._iter_episode_frames(tables, episodes, columns):
                for check in applicable:
                    run_check(
                        check,
                        lambda c=check, e=ep, f=frames: list(c.check_episode(e, f, ctx)),
                    )

        # phase 3 -- cross-episode summaries
        for check in registry.dataset_checks("post"):
            report.checks_run.append(check.name)
            run_check(check, lambda c=check: list(c.check_dataset(ctx)))

        report.records = records
        report.finished_at = _dt.datetime.now(_dt.timezone.utc)

        # -- gating ---------------------------------------------------------
        by_episode: dict[int, list[QCRecord]] = {ep.episode_id: [] for ep in episodes}
        for r in records:
            if r.episode_id is not None and r.episode_id in by_episode:
                by_episode[r.episode_id].append(r)
        episode_status = {eid: self.gate.classify(rs) for eid, rs in by_episode.items()}

        dataset_blocking = [
            r for r in records
            if r.episode_id is None and r.is_blocking
        ]
        if dataset_blocking:
            dataset_status = QualityStatus.REJECTED
        elif any(s is QualityStatus.REJECTED for s in episode_status.values()):
            dataset_status = QualityStatus.WARNING
        elif any(s is QualityStatus.WARNING for s in episode_status.values()):
            dataset_status = QualityStatus.WARNING
        else:
            dataset_status = QualityStatus.PASSED

        result = QCResult(
            report=report,
            episode_status=episode_status,
            dataset_status=dataset_status,
            shared=dict(ctx.shared),
        )

        if session is not None and session.enabled:
            self._emit_telemetry(
                session, result, report, registry, ctx,
                by_episode, check_seconds, episodes, manifest,
            )

        if self._cache is not None and fingerprints is not None:
            self._cache.store(
                dataset_id, version, fingerprints[0], fingerprints[1],
                result.report.records, result.episode_status,
                result.dataset_status, result.report.checks_run,
            )

        if persist:
            result.artifacts = self._persist(
                dataset_id, version, result, tables,
                pipeline_run_id=session.ctx.run_id if session and session.enabled else None,
                qc_execution_id=self._last_execution_id,
            )
        return result

    # -- telemetry ----------------------------------------------------------
    def _emit_telemetry(
        self, session, result: QCResult, report: QCReport,
        registry: CheckRegistry, ctx: CheckContext,
        by_episode: dict[int, list[QCRecord]], check_seconds: dict[str, float],
        episodes: list[Episode], manifest: DatasetManifest,
    ) -> None:
        """Route this validation's output into metrics and event rows.

        Applies the split measured in docs/14 §3.  Counts by check, severity and
        status become metrics -- a bounded label product of 24 x 4 x 4.  The
        individual findings and per-episode health rows, which carry episode
        ids, become event rows.  The same facts, routed by whether they identify
        a population or an individual.
        """
        ds = ctx.dataset_id

        # -- metrics: bounded labels only ----------------------------------
        for rec in report.records:
            session.inc(
                "rde_qc_findings_total", 1.0,
                check_name=rec.check_name, severity=rec.severity.value,
                status=rec.status.value, scope=rec.scope.value, dataset_id=ds,
            )
        for eid, status in result.episode_status.items():
            session.inc(
                "rde_qc_episode_status_total", 1.0,
                quality_status=getattr(status, "value", str(status)), dataset_id=ds,
            )
        for name, seconds in check_seconds.items():
            session.observe(
                "rde_qc_check_duration_seconds", seconds,
                check_name=name, dataset_id=ds,
            )
            session.inc("rde_qc_checks_run_total", 1.0, check_name=name, dataset_id=ds)

        total = max(len(episodes), 1)
        rejected = sum(
            1 for s in result.episode_status.values() if s is QualityStatus.REJECTED
        )
        session.set("rde_qc_gate_rejection_ratio", rejected / total, dataset_id=ds)

        # -- data-health metrics -------------------------------------------
        # Coverage is computed from what the episodes actually declare rather
        # than from the manifest alone: a modality present in the manifest but
        # absent from every episode is precisely the silent failure worth
        # catching.
        expected = [m.name for m in manifest.modalities]
        for mod in expected:
            present = sum(1 for ep in episodes if mod in (ep.modalities or []))
            session.set(
                "rde_data_modality_coverage_ratio", present / total,
                modality=mod, dataset_id=ds,
            )
        for ep in episodes:
            session.observe(
                "rde_data_episode_duration_seconds", float(ep.duration_s),
                dataset_id=ds,
            )
            session.observe(
                "rde_data_episode_frame_count", float(ep.num_frames), dataset_id=ds,
            )

        # Sensor and alignment health, derived entirely from findings the pass
        # already produced and profiler state already in ctx.shared -- no data
        # is re-read.
        emit_data_health(
            session, ctx.shared, report.records, manifest.modalities, ctx.fps, ds,
        )

        # -- event rows: the identifying facts ------------------------------
        qc_execution_id = f"qc-{_uuid.uuid4().hex[:12]}"
        self._last_execution_id = qc_execution_id
        statuses = list(result.episode_status.values())
        session.record_qc_execution(QCExecutionRow(
            qc_execution_id=qc_execution_id,
            run_id=session.ctx.run_id,
            dataset_id=ds,
            version=ctx.version,
            started_at=report.started_at.isoformat(),
            finished_at=(
                report.finished_at.isoformat() if report.finished_at else None
            ),
            duration_s=(
                (report.finished_at - report.started_at).total_seconds()
                if report.finished_at else None
            ),
            # Observability checks are excluded: the validation suite is 24
            # checks, and a telemetry observer riding the pass must not inflate
            # the number the QC report claims to have run.
            checks_run=len({
                n for n in report.checks_run if not n.startswith("observability.")
            }),
            findings=len(report.records),
            episodes_total=len(episodes),
            episodes_passed=statuses.count(QualityStatus.PASSED),
            episodes_warned=statuses.count(QualityStatus.WARNING),
            episodes_rejected=statuses.count(QualityStatus.REJECTED),
            episodes_quarantined=statuses.count(QualityStatus.QUARANTINED),
            dataset_status=result.dataset_status.value,
            gate_policy=",".join(sev.value for sev in self.gate.block_on),
        ))
        session.record_episode_health(summarize_episode_health(
            dataset_id=ds,
            version=ctx.version,
            qc_execution_id=qc_execution_id,
            run_id=session.ctx.run_id,
            episodes=episodes,
            findings_by_episode=by_episode,
            episode_status=result.episode_status,
            features=ctx.shared.get(SHARED_KEY, {}),
            expected_modalities=len(expected),
            check_order=sorted({c.name for c in registry.all()}),
        ))

    def _persist(
        self, dataset_id: str, version: str, result: QCResult,
        tables: ParquetTableStore,
        pipeline_run_id: str | None = None,
        qc_execution_id: str | None = None,
    ) -> dict[str, str]:
        """Write findings to Parquet + catalog, and stamp episode statuses."""
        art_dir: Path = self.settings.paths.artifact_dir(dataset_id, version) / "qc"
        art_dir.mkdir(parents=True, exist_ok=True)

        store = ParquetTableStore(art_dir, compression=self.settings.parquet_compression)
        records_path = store.write_table("records", qc_records_to_table(result.report.records))

        summary_path = art_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    **result.summary(),
                    # Underscore keys are cross-check scratch (decoded frames
                    # held for the alignment comparison, for instance). They are
                    # working state, not results, and serialising them bloats the
                    # artifact with a truncated repr of a numpy array.
                    "profile": {
                        k: v for k, v in result.shared.items()
                        if not k.startswith("_")
                    },
                },
                indent=2, default=str,
            )
        )

        # Re-runs replace prior findings rather than appending to them.
        self.catalog.clear_qc(dataset_id, version)
        self.catalog.record_qc(
            result.report.records, pipeline_run_id, qc_execution_id
        )
        self.catalog.set_episode_quality(dataset_id, version, result.episode_status)
        self.catalog.set_version_quality(dataset_id, version, result.dataset_status)

        return {"records": str(records_path), "summary": str(summary_path)}
