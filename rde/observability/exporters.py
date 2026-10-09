"""Getting telemetry out of a process that is about to exit.

Batch jobs break the assumptions exporters are usually written under.  The
documented failure (docs/13 §4.2) is blunt:

    "your last batch of spans and metrics might never get exported because the
    process exits before the telemetry providers have a chance to flush."

A telemetry layer that silently loses the final stage of every run is worse than
none, because it looks like it works.  So every exporter here writes
synchronously, and :func:`flush_all` is registered with ``atexit``.

Two rules hold throughout:

**Atomic or absent.**  Files are written to a temporary path and ``os.replace``d
into position.  A scraper must never read a half-written exposition; on POSIX
the rename is atomic, so it sees either the previous complete file or the new
complete one.

**Best-effort, always.**  An export failure increments a counter and returns.
A dead collector degrades observability; it must never fail an ingest.  This is
inverted from how the rest of the engine treats errors, and deliberately so:
QC failures are results worth recording, but telemetry failures are not results
at all.
"""

from __future__ import annotations

import atexit
import json
import os
import tempfile
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Iterable

from .context import RunContext
from .metrics import MetricRegistry

#: Weak references to callables run by :func:`flush_all`, in registration
#: order. Weak so a long-running process that constructs many short-lived
#: exporters (one TextfileExporter per ObservabilitySession, say) doesn't
#: keep every one -- and the MetricRegistry each holds -- alive for the rest
#: of the process merely because it was registered here once; each entry
#: self-removes once its owner is garbage collected.
_FLUSHERS: list[weakref.ref] = []


def register_flusher(fn: Callable[[], None]) -> Callable[[], None]:
    """Add a callable to the exit-flush chain.

    Returns an ``unregister()`` callable the owner can call once it has
    already flushed explicitly (e.g. ``ObservabilitySession.finish()``), to
    drop the entry immediately rather than waiting for garbage collection.
    """
    ref = weakref.WeakMethod(fn) if hasattr(fn, "__self__") else weakref.ref(fn)
    _FLUSHERS.append(ref)

    def unregister() -> None:
        try:
            _FLUSHERS.remove(ref)
        except ValueError:
            pass

    return unregister


def flush_all() -> None:
    """Run every registered flusher, swallowing individual failures.

    Registered with ``atexit`` at import.  One broken exporter must not prevent
    the others from writing.
    """
    for ref in list(_FLUSHERS):
        fn = ref()
        if fn is None:
            try:
                _FLUSHERS.remove(ref)
            except ValueError:
                pass
            continue
        try:
            fn()
        except Exception:  # noqa: BLE001 - telemetry never propagates
            pass


atexit.register(flush_all)


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via a same-directory temp file.

    Same directory matters: ``os.replace`` is only atomic within a filesystem,
    and /tmp is frequently a different one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def render_prometheus(registry: MetricRegistry) -> str:
    """Render a registry in the Prometheus text exposition format.

    Emitted even when a metric holds no samples: an explicit ``# TYPE`` with no
    series tells a reader the metric exists and is currently zero-valued, which
    is different from the metric having been removed.
    """
    lines: list[str] = []
    for inst in registry:
        lines.append(f"# HELP {inst.name} {inst.documentation}")
        lines.append(f"# TYPE {inst.name} {inst.kind}")
        for sample in inst.samples():
            value = sample.value
            rendered = repr(value) if isinstance(value, float) else str(value)
            lines.append(f"{sample.series_key} {rendered}")
    return "\n".join(lines) + "\n"


class TextfileExporter:
    """Write metrics where node_exporter's textfile collector will find them.

    Chosen over the Pushgateway on documented grounds (docs/13 §4.1, docs/14 §7):
    the Pushgateway provides no ``up`` metric and no staleness, so *"did the
    nightly run happen at all?"* -- the most important question about a batch
    job -- becomes structurally unanswerable.  A file has an mtime.  Freshness
    is free, staleness is visible, and the infrastructure cost is zero, which
    matters on a box with an 8 GB budget.

    The ``.prom`` suffix is required by node_exporter; files without it are
    ignored.

    **Counter semantics across runs.**  A batch process exits, so its counters
    restart at zero next run.  Prometheus treats that as a counter reset and
    ``rate()`` handles it, but ``increase()`` over a window is only correct if
    every run was scraped -- and a run that finishes in seven seconds will be
    missed by a fifteen-second scrape interval more often than not.  So the
    counters here are honest *within* a run and are the basis of the event-row
    aggregates, while the signals worth alerting on are the
    ``rde_pipeline_last_*`` gauges: a timestamp to measure freshness against, a
    status, a duration, and a volume.  This is the standard batch-job shape, and
    it is why "did the nightly run happen?" is answerable here and is not
    answerable through a Pushgateway.
    """

    def __init__(self, path: Path, registry: MetricRegistry):
        self.path = Path(path)
        self.registry = registry
        register_flusher(self.export)

    def export(self) -> Path | None:
        """Render and write.  Returns the path, or None if the write failed."""
        try:
            gauge = self.registry.get("rde_observability_series_count")
            if gauge is not None:
                # Set twice: the first call creates this gauge's own series, so
                # only the second reading includes it. A self-counting metric
                # that undercounts itself by one is a small lie, but it is the
                # metric whose job is to be trusted about cardinality.
                gauge.set(float(self.registry.cardinality))
                gauge.set(float(self.registry.cardinality))
            _atomic_write(self.path, render_prometheus(self.registry))
            return self.path
        except Exception:  # noqa: BLE001
            failures = self.registry.get("rde_observability_export_failures_total")
            if failures is not None:
                try:
                    failures.inc(1.0, result="fail")
                except Exception:  # noqa: BLE001
                    pass
            return None


class SpanExporter:
    """Write a run's spans as newline-delimited JSON.

    NDJSON rather than an OTLP payload because the POC has no collector to send
    to, and the durable value of tracing here is not the flame graph -- it is
    the ``trace_id`` and ``span_id`` sitting in the analytical tables, which
    turn metric -> trace -> run -> episode -> finding into a series of joins
    (docs/14 §7).  Those columns are the deliverable; the backend is
    interchangeable, and the ids are already OTel-shaped.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def export(self, ctx: RunContext) -> Path | None:
        """Write ``<run_id>.spans.ndjson``.  Returns the path, or None."""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            out = self.directory / f"{ctx.run_id}.spans.ndjson"
            payload = [
                {"trace_id": ctx.trace_id, "run_id": ctx.run_id, **sp.to_dict()}
                for sp in ctx.spans
            ]
            _atomic_write(out, "\n".join(json.dumps(p) for p in payload) + "\n")
            return out
        except Exception:  # noqa: BLE001
            return None


class ConsoleExporter:
    """Human-readable run summary, for the CLI and for tests."""

    def render(self, ctx: RunContext, registry: MetricRegistry) -> str:
        rows = [
            f"run {ctx.run_id}  trace {ctx.trace_id[:16]}…  "
            f"{ctx.status}  {ctx.duration_s or 0.0:.2f}s"
        ]
        for sp in ctx.spans:
            if sp.parent_span_id is None:
                rows.append(
                    f"  {sp.name:<20s} {sp.status:<6s} {sp.duration_s or 0.0:>8.3f}s"
                )
        rows.append(f"  {'series':<20s} {registry.cardinality:>15d}")
        return "\n".join(rows)


def stale_after(path: Path, max_age_s: float) -> bool:
    """Whether a textfile export is older than ``max_age_s``.

    The freshness primitive the Pushgateway cannot provide.  A missing file is
    stale by definition -- "the run never happened" and "the run happened long
    ago" are both failures, and both must alert.
    """
    try:
        return (time.time() - path.stat().st_mtime) > max_age_s
    except FileNotFoundError:
        return True


def iter_prom_lines(text: str) -> Iterable[tuple[str, float]]:
    """Parse an exposition back into (series_key, value) pairs.

    Used by the tests to assert on what was actually exported rather than on
    what the in-memory registry believes -- the two diverge exactly when the
    exporter is broken, which is the case worth catching.
    """
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, val = line.rpartition(" ")
        try:
            yield key, float(val)
        except ValueError:
            continue


def summarize(registry: MetricRegistry) -> dict[str, Any]:
    """Compact per-instrument view: kind, series count, documentation."""
    return {
        inst.name: {
            "kind": inst.kind,
            "series": inst.cardinality,
            "help": inst.documentation,
        }
        for inst in registry
    }
