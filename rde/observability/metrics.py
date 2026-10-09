"""Metric primitives with an *enforced* label policy.

The cardinality analysis in docs/14 is not advisory.  Attaching ``episode_id``
to one QC metric produces 264 billion series at the design planning scale --
measured at 167 TB of resident memory, against 6 GB of Parquet holding the same
facts (``examples/08_cardinality_experiment.py``).  A rule that costly cannot be
left to reviewer discipline, so it is enforced here: :class:`LabelPolicy` raises
on a forbidden label instead of silently accepting it.

The distinction being enforced is not "how many values are there today" but
"what makes a new value appear":

* ``check_name`` is bounded -- 24 checks exist because 24 were written.  A new
  one requires a code change.
* ``episode_id`` is unbounded -- a new value appears every time a robot records
  a demonstration.

Bounded dimensions may be labels.  Unbounded ones are join keys, and belong in
event rows (``rde/observability/events.py``) where a columnar store pays for
them at query time instead of holding them resident.

Instrument shapes follow the OpenTelemetry/Prometheus model -- counter, gauge,
histogram -- so that swapping the exporter for OTLP is a transport change and
not a rewrite (docs/14, §7).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping

#: Dimensions whose value vocabulary is bounded by code or by an enum, and which
#: may therefore appear as metric labels.
SAFE_LABELS: frozenset[str] = frozenset({
    "stage",          # 8, fixed by the pipeline definition
    "check_name",     # 24, fixed by the check registry
    "status",         # 4, CheckStatus
    "severity",       # 4, Severity
    "scope",          # 3, CheckScope
    "modality",       # curated registry; growth is a code change
    "modality_kind",  # ModalityKind enum
    "outcome",        # ok / error
    "error_class",    # exception type, allowlisted below
    "quality_status", # QualityStatus enum
    "operation",      # transformation operator name
    "split",          # train / val / test
    "feature",        # drift feature name, from a fixed extractor
    "result",         # hit / miss, pass / fail
})

#: Bounded per deployment but growing monotonically.  Permitted, with the
#: understanding that each one multiplies the series count of every metric it
#: is attached to -- so they are used sparingly and never all at once.
CONDITIONAL_LABELS: frozenset[str] = frozenset({
    "dataset_id",       # grows by human decision, not by recording
    "dataset_version",  # bounded per dataset; old versions stop emitting
    "software_version", # bounded by releases -- prefer build_info
    "factory_id",       # the cohort dimension that makes a broken site visible
    "robot_id",
    "sensor_rig_type",  # rig *type*, deliberately not rig instance
})

#: Unbounded by construction.  These identify individuals, which is what event
#: rows are for.  Listed explicitly so the error message can say why.
FORBIDDEN_LABELS: frozenset[str] = frozenset({
    "episode_id", "frame_id", "run_id", "trace_id", "span_id", "blob_id",
    "task_id", "code_fingerprint", "path", "file", "filename", "uuid",
    "timestamp", "observed_value", "reason", "sensor_id", "sensor_rig_id",
})

#: Exception names allowed through as ``error_class``.  Anything else collapses
#: to ``other``: an unbounded exception hierarchy is an unbounded label, and a
#: crash loop emitting novel exception types would otherwise be a cardinality
#: incident on top of an outage.
KNOWN_ERROR_CLASSES: frozenset[str] = frozenset({
    "ValueError", "KeyError", "TypeError", "OSError", "FileNotFoundError",
    "LookupError", "RuntimeError", "MemoryError", "TimeoutError",
    "ArrowInvalid", "PermissionError", "NotImplementedError",
})

#: Latency buckets in seconds, spanning a sub-millisecond check to a stage that
#: runs for minutes.  Explicit rather than generated: bucket boundaries are part
#: of the metric's contract, and changing them invalidates stored history.
DURATION_BUCKETS: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0,
)

#: Byte-count buckets, 1 KB to 100 GB.
SIZE_BUCKETS: tuple[float, ...] = tuple(10.0 ** e for e in range(3, 12))


class LabelPolicyError(ValueError):
    """Raised when a metric is given a label that would explode cardinality."""


def normalize_error_class(name: str | None) -> str:
    """Collapse an arbitrary exception name to a bounded vocabulary."""
    if not name:
        return "none"
    return name if name in KNOWN_ERROR_CLASSES else "other"


@dataclass(frozen=True)
class LabelPolicy:
    """Validates label sets against the safe/conditional/forbidden split."""

    allow_conditional: bool = True

    def validate(self, labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        """Check and canonicalise a label set.

        Returns the labels sorted, so that the same logical series always
        produces the same key regardless of keyword ordering at the call site.
        """
        for key in labels:
            if key in FORBIDDEN_LABELS:
                raise LabelPolicyError(
                    f"{key!r} is an unbounded identifier and must never be a metric "
                    f"label -- at 344M episodes/year this costs ~167 TB of resident "
                    f"memory (see docs/14 §3). Record it as an event row instead: "
                    f"rde.observability.events."
                )
            if key in CONDITIONAL_LABELS:
                if not self.allow_conditional:
                    raise LabelPolicyError(
                        f"{key!r} is a conditionally-safe label, disallowed by this "
                        f"policy. Drop it or widen the policy deliberately."
                    )
                continue
            if key not in SAFE_LABELS:
                raise LabelPolicyError(
                    f"{key!r} is not a known label. Add it to SAFE_LABELS or "
                    f"CONDITIONAL_LABELS in rde/observability/metrics.py after "
                    f"establishing that its value vocabulary is bounded by code, "
                    f"not by data volume."
                )
        return tuple(sorted((k, str(v)) for k, v in labels.items()))


@dataclass
class Sample:
    """One exported series: a metric name, a label set and a value."""
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float
    suffix: str = ""

    @property
    def series_key(self) -> str:
        """Prometheus-format identity: ``name{label="value",...}``."""
        lbl = ",".join(f'{k}="{v}"' for k, v in self.labels)
        return f"{self.name}{self.suffix}{{{lbl}}}" if lbl else f"{self.name}{self.suffix}"


class _Instrument:
    """Shared plumbing: name, help text, policy, and a lock over the series map."""

    kind = "untyped"

    def __init__(self, name: str, documentation: str, policy: LabelPolicy | None = None):
        self.name = name
        self.documentation = documentation
        self.policy = policy or LabelPolicy()
        self._lock = threading.Lock()

    def _key(self, labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        return self.policy.validate(labels)

    def samples(self) -> Iterator[Sample]:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def cardinality(self) -> int:
        """Distinct series this instrument currently holds.

        Exposed because a metric layer that cannot report its own cardinality
        is the one that surprises you.
        """
        raise NotImplementedError


class Counter(_Instrument):
    """Monotonically increasing total.  Rates are derived at query time."""

    kind = "counter"

    def __init__(self, name: str, documentation: str, policy: LabelPolicy | None = None):
        super().__init__(name, documentation, policy)
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Add ``amount`` (must be non-negative) to the labelled series."""
        if amount < 0:
            raise ValueError("counters increase only; use a gauge for a signed value")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        """Current total for one labelled series; 0.0 if never incremented."""
        return self._values.get(self._key(labels), 0.0)

    def samples(self) -> Iterator[Sample]:
        """One sample per labelled series, in stable label order."""
        for key, val in sorted(self._values.items()):
            yield Sample(self.name, key, val)

    @property
    def cardinality(self) -> int:
        """Distinct series this counter holds."""
        return len(self._values)


class Gauge(_Instrument):
    """Point-in-time value that may rise or fall."""

    kind = "gauge"

    def __init__(self, name: str, documentation: str, policy: LabelPolicy | None = None):
        super().__init__(name, documentation, policy)
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        """Record ``value``.  NaN is dropped rather than exported.

        A NaN gauge silently poisons every downstream aggregate it touches --
        which is the same failure mode as the float32 constant-variance bug that
        put NaNs into the feedback loop's correlation inputs.  Dropping is
        visible as a missing series; exporting is invisible until it corrupts.
        """
        if not math.isfinite(value):
            return
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)

    def value(self, **labels: str) -> float | None:
        """Current value for one labelled series, or None if never set."""
        return self._values.get(self._key(labels))

    def samples(self) -> Iterator[Sample]:
        """One sample per labelled series, in stable label order."""
        for key, val in sorted(self._values.items()):
            yield Sample(self.name, key, val)

    @property
    def cardinality(self) -> int:
        """Distinct series this gauge holds."""
        return len(self._values)


class Histogram(_Instrument):
    """Cumulative bucket counts, plus sum and count.

    Chosen over a summary because bucket counts aggregate across workers and
    quantiles do not -- and the scaling path runs episodes in parallel.
    """

    kind = "histogram"

    def __init__(
        self,
        name: str,
        documentation: str,
        buckets: Iterable[float] = DURATION_BUCKETS,
        policy: LabelPolicy | None = None,
    ):
        super().__init__(name, documentation, policy)
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[tuple[tuple[str, str], ...], list[int]] = {}
        self._sums: dict[tuple[tuple[str, str], ...], float] = {}

    def observe(self, value: float, **labels: str) -> None:
        """Record one observation.  Non-finite values are dropped."""
        if not math.isfinite(value):
            return
        key = self._key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            self._sums[key] = self._sums.get(key, 0.0) + value
            # +Inf bucket is the last slot; cumulative form is built at export.
            idx = len(self.buckets)
            for i, edge in enumerate(self.buckets):
                if value <= edge:
                    idx = i
                    break
            counts[idx] += 1

    def samples(self) -> Iterator[Sample]:
        """Cumulative ``_bucket`` samples plus ``_sum`` and ``_count``."""
        for key, counts in sorted(self._counts.items()):
            cumulative = 0
            for edge, count in zip(self.buckets, counts):
                cumulative += count
                yield Sample(
                    self.name, key + (("le", _fmt(edge)),), cumulative, "_bucket"
                )
            total = cumulative + counts[-1]
            yield Sample(self.name, key + (("le", "+Inf"),), total, "_bucket")
            yield Sample(self.name, key, self._sums.get(key, 0.0), "_sum")
            yield Sample(self.name, key, total, "_count")

    @property
    def cardinality(self) -> int:
        """Distinct series this histogram holds, counting bucket sets once."""
        return len(self._counts)


def _fmt(v: float) -> str:
    """Prometheus bucket-edge formatting: integral edges lose the '.0'."""
    return str(int(v)) if float(v).is_integer() else repr(v)


class MetricRegistry:
    """Holds every instrument and knows its own total cardinality."""

    def __init__(self, policy: LabelPolicy | None = None):
        self.policy = policy or LabelPolicy()
        self._instruments: dict[str, _Instrument] = {}

    def counter(self, name: str, documentation: str) -> Counter:
        return self._register(Counter(name, documentation, self.policy))

    def gauge(self, name: str, documentation: str) -> Gauge:
        return self._register(Gauge(name, documentation, self.policy))

    def histogram(
        self, name: str, documentation: str, buckets: Iterable[float] = DURATION_BUCKETS
    ) -> Histogram:
        return self._register(Histogram(name, documentation, buckets, self.policy))

    def _register(self, inst: _Instrument):
        """Idempotent: re-declaring an existing metric returns the original.

        Modules declare their metrics at import time; a re-import must not
        produce a second instrument that silently splits the series.
        """
        existing = self._instruments.get(inst.name)
        if existing is not None:
            if type(existing) is not type(inst):
                raise ValueError(
                    f"metric {inst.name!r} already registered as {existing.kind}"
                )
            return existing
        self._instruments[inst.name] = inst
        return inst

    def get(self, name: str) -> _Instrument | None:
        """One instrument by name, or None if it was never declared."""
        return self._instruments.get(name)

    def __iter__(self) -> Iterator[_Instrument]:
        """Instruments in name order, so exports are byte-stable."""
        return iter(sorted(self._instruments.values(), key=lambda i: i.name))

    @property
    def cardinality(self) -> int:
        """Total series across every instrument.

        The number this layer must keep small.  Exported as a metric in its own
        right so that a cardinality regression is itself observable.
        """
        return sum(i.cardinality for i in self._instruments.values())

    def clear(self) -> None:
        """Drop every instrument.  For tests and for per-run isolation."""
        self._instruments.clear()
