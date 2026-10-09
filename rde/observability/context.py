"""The correlation spine: one id per pipeline execution, threaded everywhere.

Reconnaissance (docs/12, Q9) found the blocking defect this module fixes.
``Pipeline.run()`` minted no identifier for itself; each stage that produced a
version generated an independent ``uuid4``, and nothing joined them.  The eight
lineage rows written by one execution had no column linking them to each other,
so "show me everything that happened in run 81" was unanswerable.

Three properties, each of which is a deliberate constraint rather than an
accident of implementation:

**Minted once.**  ``RunContext.start`` is called at the top of a pipeline run
and nowhere else.  Sub-components receive the context; they never create one.

**Additive.**  ``TransformRun.run_id`` keeps its existing meaning as the
identity of one lineage edge.  ``pipeline_run_id`` is a foreign key *upward*.
Lineage remains the authoritative provenance record (docs/12, Q12); this is a
grouping key over it, not a replacement for it.

**Nullable.**  ``current()`` returns ``None`` outside a run, and every consumer
must handle that.  ``QCEngine.run`` called directly from a notebook has no
pipeline around it and must keep working.  Observability that makes the library
harder to use as a library has failed.

The context is held in a :class:`~contextvars.ContextVar` rather than a module
global so that concurrent pipelines -- the episode-parallel execution the
scaling path anticipates -- do not overwrite each other's identity.
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import os
import secrets
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

#: Set for the duration of a pipeline run; ``None`` outside one.
_CURRENT: contextvars.ContextVar["RunContext | None"] = contextvars.ContextVar(
    "rde_run_context", default=None
)


def _trace_id() -> str:
    """128-bit id, hex, matching the OpenTelemetry trace-id wire format.

    We are not emitting OTLP today (docs/14, §7), but the *shape* is free to
    adopt now and expensive to retrofit later.  A 32-hex-character trace id
    drops into any OTel backend without translation.
    """
    return secrets.token_hex(16)


def _span_id() -> str:
    """64-bit id, hex -- the OpenTelemetry span-id wire format."""
    return secrets.token_hex(8)


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


@dataclass
class Span:
    """One timed unit of execution within a run.

    Deliberately coarse.  There is no span per frame, and the reason is
    arithmetic: at 135B frames/year a per-frame span costs more to store than
    the frames themselves.  Spans mark stages and their major sub-units; detail
    below that granularity belongs in event rows.
    """

    name: str
    span_id: str
    parent_span_id: str | None
    started_at: _dt.datetime
    finished_at: _dt.datetime | None = None
    status: str = "running"
    error_class: str | None = None
    #: Low-cardinality descriptive attributes.  High-cardinality ids are allowed
    #: here -- unlike on metrics -- because spans are events, not aggregates.
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form, for the span exporter."""
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_s": self.duration_s,
            "status": self.status,
            "error_class": self.error_class,
            "attributes": self.attributes,
        }


@dataclass
class RunContext:
    """Identity and accumulated telemetry for one pipeline execution."""

    run_id: str
    trace_id: str
    dataset_id: str
    software_version: str
    started_at: _dt.datetime
    host: str
    pid: int
    config_hash: str = ""
    finished_at: _dt.datetime | None = None
    status: str = "running"

    spans: list[Span] = field(default_factory=list)
    #: Stack of open spans; the last entry is the current parent.
    _open: list[Span] = field(default_factory=list, repr=False)

    # -- identity -----------------------------------------------------------
    @classmethod
    def start(
        cls,
        dataset_id: str,
        software_version: str,
        config_hash: str = "",
    ) -> "RunContext":
        """Mint a new run identity.  Called once per pipeline execution."""
        return cls(
            run_id=f"run-{_trace_id()[:16]}",
            trace_id=_trace_id(),
            dataset_id=dataset_id,
            software_version=software_version,
            started_at=_now(),
            host=socket.gethostname(),
            pid=os.getpid(),
            config_hash=config_hash,
        )

    @property
    def duration_s(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def current_span(self) -> Span | None:
        """Innermost open span, or None at the run's top level."""
        return self._open[-1] if self._open else None

    # -- spans --------------------------------------------------------------
    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        """Open a span, close it on exit, record the exception class on failure.

        The exception is re-raised.  Observability records what happened; it
        never changes what happens.
        """
        sp = Span(
            name=name,
            span_id=_span_id(),
            parent_span_id=self.current_span.span_id if self.current_span else None,
            started_at=_now(),
            attributes=dict(attributes),
        )
        self.spans.append(sp)
        self._open.append(sp)
        try:
            yield sp
        except BaseException as exc:
            sp.status = "error"
            sp.error_class = type(exc).__name__
            raise
        else:
            sp.status = "ok"
        finally:
            sp.finished_at = _now()
            self._open.pop()

    def finish(self, status: str = "ok") -> "RunContext":
        """Close the run.  Idempotent."""
        if self.finished_at is None:
            self.finished_at = _now()
            self.status = status
        return self

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe run header, without the spans."""
        return {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "dataset_id": self.dataset_id,
            "software_version": self.software_version,
            "config_hash": self.config_hash,
            "host": self.host,
            "pid": self.pid,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_s": self.duration_s,
            "status": self.status,
        }


def current() -> RunContext | None:
    """The active run context, or ``None`` when not inside a pipeline run.

    Every caller must handle ``None``.  That is the point: the engine stays
    usable as a library, and instrumentation degrades to a no-op rather than
    raising.
    """
    return _CURRENT.get()


@contextmanager
def use(ctx: RunContext) -> Iterator[RunContext]:
    """Bind ``ctx`` as the active run context for the duration of the block."""
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span | None]:
    """Open a span on the active run, or do nothing if there is no run.

    Module-level convenience so call sites deep in the engine can be
    instrumented without threading a context object through every signature.
    """
    ctx = current()
    if ctx is None:
        yield None
        return
    with ctx.span(name, **attributes) as sp:
        yield sp
