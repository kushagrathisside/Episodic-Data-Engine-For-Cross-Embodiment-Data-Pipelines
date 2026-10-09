"""One object that owns a run's telemetry, so call sites stay one line long.

The engine's modules should not each learn how to mint spans, pick label sets,
build event rows and flush exporters.  :class:`ObservabilitySession` binds those
together and exposes two verbs -- :meth:`stage` and :meth:`record_*` -- which is
the whole surface most call sites need.

Design constraints inherited from the reconnaissance (docs/12, Q12):

* **Rides the existing pass.**  Nothing here traverses data.  It observes work
  that is already happening; it never causes work.
* **Optional.**  ``ObservabilitySession.disabled()`` returns a session whose
  methods are no-ops, so the engine runs identically without telemetry.  This
  is what keeps the library usable as a library.
* **Never fatal.**  Telemetry failures are swallowed.  A dead exporter degrades
  observability; it must not fail an ingest.
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import hashlib
import json
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .context import RunContext, use
from .events import (
    DriftReportRow,
    EpisodeHealthRow,
    EventBuffer,
    EventWriter,
    PipelineRunRow,
    QCExecutionRow,
    StageRunRow,
)
from .exporters import SpanExporter, TextfileExporter
from .instruments import build_registry
from .metrics import MetricRegistry, normalize_error_class


#: The active session, set by ``ObservabilitySession.active()``.
#: A ContextVar rather than a module global so concurrent pipelines -- the
#: episode-parallel execution the scaling path anticipates -- cannot overwrite
#: each other's telemetry.
_CURRENT_SESSION: contextvars.ContextVar["ObservabilitySession | None"] = (
    contextvars.ContextVar("rde_observability_session", default=None)
)


def current_session() -> "ObservabilitySession | None":
    """The active session, or ``None`` outside a pipeline run.

    Every caller must handle ``None``.  ``QCEngine.run`` invoked directly from a
    notebook has no session around it and must behave identically.
    """
    return _CURRENT_SESSION.get()


def _peak_rss_bytes() -> int:
    """Peak RSS of this process, in bytes.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS.  It is also a
    high-water mark for the whole process, never decreasing -- so a per-stage
    figure is the *delta*, and is meaningful only as "this stage pushed the peak
    up by X", not as "this stage used X".  Recorded because a stage that raises
    the ceiling is exactly the stage that will OOM first at scale.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def config_hash(payload: Any) -> str:
    """Stable short hash of a run's configuration.

    Two runs are only comparable if their configuration matched; a duration
    change across a config change means nothing.  Sorted keys so dict ordering
    never produces a spurious difference.
    """
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class ObservabilitySession:
    """Telemetry for one pipeline execution."""

    def __init__(
        self,
        ctx: RunContext,
        registry: MetricRegistry | None = None,
        event_dir: Path | None = None,
        metrics_path: Path | None = None,
        span_dir: Path | None = None,
        enabled: bool = True,
        catalog: Any = None,
    ):
        self.ctx = ctx
        self.enabled = enabled
        #: Optional catalog mirror.  Parquet stays authoritative for the
        #: detailed rows; the catalog gets the small bounded headers so
        #: "has validate gotten slower?" is a SQL query (docs/12, G2).
        self.catalog = catalog
        self.registry = registry or build_registry()
        self.events = EventBuffer()
        self.event_writer = EventWriter(event_dir) if event_dir else None
        self.span_exporter = SpanExporter(span_dir) if span_dir else None
        self.metrics_exporter = (
            TextfileExporter(metrics_path, self.registry) if metrics_path else None
        )
        self._stage_seq = 0
        self._artifacts: dict[str, str] = {}

    # -- construction -------------------------------------------------------
    @classmethod
    def disabled(cls) -> "ObservabilitySession":
        """A session that records nothing.  Used when telemetry is switched off."""
        return cls(
            RunContext.start("<disabled>", "0"),
            registry=MetricRegistry(),
            enabled=False,
        )

    @classmethod
    def for_run(
        cls,
        dataset_id: str,
        software_version: str,
        paths: Any,
        config: Any = None,
        registry: MetricRegistry | None = None,
        catalog: Any = None,
    ) -> "ObservabilitySession":
        """Build a session wired to the engine's standard data-root layout."""
        ctx = RunContext.start(
            dataset_id, software_version, config_hash(config) if config else ""
        )
        obs_root = Path(paths.data_root) / "observability"
        return cls(
            ctx=ctx,
            registry=registry,
            event_dir=obs_root / "events",
            metrics_path=obs_root / "metrics" / "rde.prom",
            span_dir=obs_root / "spans",
            catalog=catalog,
        )

    # -- metric helpers -----------------------------------------------------
    def _counter(self, name: str):
        return self.registry.get(name)

    def inc(self, metric: str, amount: float = 1.0, **labels: str) -> None:
        """Increment a counter, swallowing any policy or export error."""
        if not self.enabled:
            return
        inst = self._counter(metric)
        if inst is None:
            return
        try:
            inst.inc(amount, **labels)
        except Exception:  # noqa: BLE001
            pass

    def observe(self, metric: str, value: float, **labels: str) -> None:
        """Record a histogram observation."""
        if not self.enabled:
            return
        inst = self._counter(metric)
        if inst is None:
            return
        try:
            inst.observe(value, **labels)
        except Exception:  # noqa: BLE001
            pass

    def set(self, metric: str, value: float, **labels: str) -> None:
        """Set a gauge."""
        if not self.enabled:
            return
        inst = self._counter(metric)
        if inst is None:
            return
        try:
            inst.set(value, **labels)
        except Exception:  # noqa: BLE001
            pass

    # -- stages -------------------------------------------------------------
    @contextmanager
    def stage(self, name: str, **attributes: Any) -> Iterator["StageHandle"]:
        """Instrument one pipeline stage.

        Emits a span, a duration histogram sample, a failure counter on error,
        and a ``stage_runs`` event row -- from one ``with`` block, so that
        instrumenting a stage stays cheaper than not instrumenting it.
        """
        if not self.enabled:
            yield StageHandle(self, name, None)
            return

        self._stage_seq += 1
        seq = self._stage_seq
        rss_before = _peak_rss_bytes()
        t0 = time.perf_counter()
        handle = StageHandle(self, name, None)
        started = _dt.datetime.now(_dt.timezone.utc)
        error_class: str | None = None
        status = "ok"

        with self.ctx.span(f"stage.{name}", **attributes) as sp:
            handle.span = sp
            try:
                yield handle
            except BaseException as exc:
                status = "error"
                error_class = type(exc).__name__
                raise
            finally:
                elapsed = time.perf_counter() - t0
                rss_after = _peak_rss_bytes()
                labels = {"stage": name, "dataset_id": self.ctx.dataset_id}
                self.observe("rde_pipeline_stage_duration_seconds", elapsed, **labels)
                if status == "error":
                    self.inc(
                        "rde_pipeline_stage_failures_total", 1.0,
                        error_class=normalize_error_class(error_class), **labels,
                    )
                self.set("rde_pipeline_stage_peak_rss_bytes", float(rss_after), **labels)
                self.set("rde_pipeline_last_run_duration_seconds", elapsed, **labels)
                if handle.records_out is not None:
                    self.set("rde_pipeline_last_run_frames",
                             float(handle.records_out), **labels)
                if handle.episodes is not None:
                    self.set("rde_pipeline_last_run_episodes",
                             float(handle.episodes), **labels)
                if handle.records_out is not None:
                    self.inc(
                        "rde_pipeline_frames_processed_total",
                        float(handle.records_out), **labels,
                    )
                if handle.episodes is not None:
                    self.inc(
                        "rde_pipeline_episodes_processed_total",
                        float(handle.episodes), **labels,
                    )
                if handle.bytes_read:
                    self.inc("rde_pipeline_bytes_read_total",
                             float(handle.bytes_read), **labels)
                if handle.bytes_written:
                    self.inc("rde_pipeline_bytes_written_total",
                             float(handle.bytes_written), **labels)

                self.events.add("stage_runs", StageRunRow(
                    run_id=self.ctx.run_id,
                    trace_id=self.ctx.trace_id,
                    span_id=sp.span_id if sp else "",
                    dataset_id=self.ctx.dataset_id,
                    stage=name,
                    seq=seq,
                    started_at=started.isoformat(),
                    finished_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                    duration_s=round(elapsed, 6),
                    status=status,
                    error_class=error_class,
                    records_in=handle.records_in,
                    records_out=handle.records_out,
                    episodes=handle.episodes,
                    bytes_read=handle.bytes_read,
                    bytes_written=handle.bytes_written,
                    peak_rss_bytes=max(rss_after - rss_before, 0),
                    attributes_json=json.dumps(handle.attributes, default=str),
                ))

    # -- event recording ----------------------------------------------------
    def record_qc_execution(self, row: QCExecutionRow) -> None:
        """Buffer one QC-execution header for this run."""
        if self.enabled:
            self.events.add("qc_executions", row)

    def record_episode_health(self, rows: list[EpisodeHealthRow]) -> None:
        """Buffer one health row per episode."""
        if self.enabled:
            self.events.extend("episode_health", rows)

    def record_drift(self, rows: list[DriftReportRow]) -> None:
        """Buffer drift comparisons between two dataset versions."""
        if self.enabled:
            self.events.extend("drift_reports", rows)

    # -- lifecycle ----------------------------------------------------------
    @contextmanager
    def active(self) -> Iterator["ObservabilitySession"]:
        """Bind this session and its run context for the duration of the block."""
        token = _CURRENT_SESSION.set(self)
        try:
            with use(self.ctx):
                yield self
        finally:
            _CURRENT_SESSION.reset(token)

    def finish(self, status: str = "ok") -> dict[str, str]:
        """Close the run, emit run-level metrics, and flush every exporter.

        Flushing here rather than relying on ``atexit`` alone is deliberate:
        the documented batch-job failure is losing the final export because the
        process exits first (docs/13 §4.2).  ``atexit`` remains registered as a
        backstop for the abnormal path.
        """
        if not self.enabled:
            return {}
        self.ctx.finish(status)

        labels = {"dataset_id": self.ctx.dataset_id, "outcome": status}
        self.inc("rde_pipeline_runs_total", 1.0, **labels)
        self.observe(
            "rde_pipeline_run_duration_seconds",
            float(self.ctx.duration_s or 0.0),
            dataset_id=self.ctx.dataset_id, stage="all",
        )
        now = time.time()
        self.set("rde_pipeline_last_run_timestamp_seconds", now,
                 dataset_id=self.ctx.dataset_id)
        self.set("rde_pipeline_last_run_status", 1.0 if status == "ok" else 0.0,
                 dataset_id=self.ctx.dataset_id)
        self.set("rde_pipeline_last_run_duration_seconds",
                 float(self.ctx.duration_s or 0.0),
                 dataset_id=self.ctx.dataset_id, stage="all")
        if status == "ok":
            self.set("rde_pipeline_last_success_timestamp_seconds", now,
                     dataset_id=self.ctx.dataset_id)

        self.events.add("pipeline_runs", PipelineRunRow(
            run_id=self.ctx.run_id,
            trace_id=self.ctx.trace_id,
            dataset_id=self.ctx.dataset_id,
            started_at=self.ctx.started_at.isoformat(),
            finished_at=(
                self.ctx.finished_at.isoformat() if self.ctx.finished_at else None
            ),
            duration_s=self.ctx.duration_s,
            status=status,
            software_version=self.ctx.software_version,
            config_hash=self.ctx.config_hash,
            host=self.ctx.host,
            pid=self.ctx.pid,
        ))

        # Mirror the bounded headers into the catalog before flushing Parquet.
        # Wrapped because a catalog write failure must degrade observability,
        # never fail a run that has already completed its actual work.
        if self.catalog is not None:
            try:
                self.catalog.record_pipeline_run(self.events.get("pipeline_runs")[-1])
                self.catalog.record_stage_runs(self.events.get("stage_runs"))
                self.catalog.record_qc_executions(self.events.get("qc_executions"))
            except Exception:  # noqa: BLE001
                pass

        # Each exporter gets its own try/except: these three are independent
        # I/O paths (Parquet event tables, NDJSON spans, a Prometheus text
        # file), and a failure in one must not skip the other two the way a
        # single shared try/except previously did -- a bad row in one event
        # table silently dropped that run's spans and metrics too, in the one
        # subsystem whose entire design principle is "never silently discard
        # data". Every failure is counted so it is visible in the metrics
        # this same session exports, rather than vanishing without a trace.
        artifacts: dict[str, str] = {}
        if self.event_writer is not None:
            try:
                artifacts.update(self.events.flush(self.event_writer, self.ctx.run_id))
            except Exception:  # noqa: BLE001
                self.inc("rde_observability_export_failures_total", 1.0, result="fail")
        if self.span_exporter is not None:
            try:
                p = self.span_exporter.export(self.ctx)
                if p:
                    artifacts["spans"] = str(p)
                else:
                    self.inc("rde_observability_export_failures_total", 1.0, result="fail")
            except Exception:  # noqa: BLE001
                self.inc("rde_observability_export_failures_total", 1.0, result="fail")
        if self.metrics_exporter is not None:
            try:
                p = self.metrics_exporter.export()
                if p:
                    artifacts["metrics"] = str(p)
                # TextfileExporter.export() already counts its own failures.
            except Exception:  # noqa: BLE001
                self.inc("rde_observability_export_failures_total", 1.0, result="fail")
        self._artifacts = artifacts
        return artifacts

    @property
    def artifacts(self) -> dict[str, str]:
        """Paths written by the last :meth:`finish`, keyed by what they hold."""
        return dict(self._artifacts)


class StageHandle:
    """Handle a stage body uses to report volume and byte counts.

    Populated by the stage, read by the session when the block closes -- so a
    stage that has nothing to report simply does not touch it, and the fields
    stay ``None`` rather than a misleading zero.
    """

    def __init__(self, session: ObservabilitySession, name: str, span: Any):
        self.session = session
        self.name = name
        self.span = span
        self.records_in: int | None = None
        self.records_out: int | None = None
        self.episodes: int | None = None
        self.bytes_read: int | None = None
        self.bytes_written: int | None = None
        self.attributes: dict[str, Any] = {}

    def counts(
        self,
        records_in: int | None = None,
        records_out: int | None = None,
        episodes: int | None = None,
    ) -> "StageHandle":
        """Report record and episode volume for this stage."""
        if records_in is not None:
            self.records_in = int(records_in)
        if records_out is not None:
            self.records_out = int(records_out)
        if episodes is not None:
            self.episodes = int(episodes)
        return self

    def bytes(self, read: int | None = None, written: int | None = None) -> "StageHandle":
        """Report I/O volume for this stage."""
        if read is not None:
            self.bytes_read = int(read)
        if written is not None:
            self.bytes_written = int(written)
        return self

    def attr(self, **kwargs: Any) -> "StageHandle":
        """Attach descriptive attributes to the span and the event row."""
        self.attributes.update(kwargs)
        if self.span is not None:
            self.span.attributes.update(kwargs)
        return self
