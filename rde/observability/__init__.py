"""Observability: the temporal dimension the catalog does not have.

The engine's catalog answers *"what is this dataset and how was it produced?"*
with real rigour -- immutable versions, content-addressed blobs, episode-level
lineage, persisted findings.  It cannot answer any question containing the word
*"when"*.

This package adds that, under one governing rule, measured rather than assumed
(docs/14 §3):

    Metrics aggregate populations.  Events identify individuals.
    Traces connect them.

Attaching ``episode_id`` to a single QC metric costs 264 billion series and
~167 TB of resident memory at 344M episodes/year.  The identical facts as
columnar rows measure 6 GB.  :mod:`rde.observability.metrics` enforces that rule
rather than documenting it.

Modules
-------
``context``      Correlation spine: one run id + trace id, threaded, nullable.
``metrics``      Counter/gauge/histogram with an *enforced* label policy.
``instruments``  The declared metric catalog -- every series the engine may emit.
``events``       High-cardinality analytical rows, Parquet-backed.
``exporters``    Textfile ``.prom``, NDJSON spans; atomic, flushed on exit.
``session``      Binds the above so a call site is one ``with`` block.
``health``       Episode-level health summaries and drift features.
``drift``        Episode-grain distribution comparison between versions.
``contracts``    Release policy: the third layer after checking and gating.
"""

from .context import RunContext, Span, current, span, use
from .contracts import (
    Clause,
    ContractReport,
    DataContract,
    render_report,
)
from .events import (
    DriftReportRow,
    EpisodeHealthRow,
    EventBuffer,
    EventWriter,
    PipelineRunRow,
    QCExecutionRow,
    StageRunRow,
)
from .exporters import (
    SpanExporter,
    TextfileExporter,
    render_prometheus,
    stale_after,
)
from .instruments import REGISTRY, build_registry
from .metrics import (
    CONDITIONAL_LABELS,
    FORBIDDEN_LABELS,
    SAFE_LABELS,
    Counter,
    Gauge,
    Histogram,
    LabelPolicy,
    LabelPolicyError,
    MetricRegistry,
)
from .session import (
    ObservabilitySession,
    StageHandle,
    config_hash,
    current_session,
)

__all__ = [
    "RunContext", "Span", "current", "span", "use",
    "DataContract", "ContractReport", "Clause", "render_report",
    "MetricRegistry", "Counter", "Gauge", "Histogram",
    "LabelPolicy", "LabelPolicyError",
    "SAFE_LABELS", "CONDITIONAL_LABELS", "FORBIDDEN_LABELS",
    "REGISTRY", "build_registry",
    "EventBuffer", "EventWriter", "PipelineRunRow", "StageRunRow",
    "QCExecutionRow", "EpisodeHealthRow", "DriftReportRow",
    "TextfileExporter", "SpanExporter", "render_prometheus", "stale_after",
    "ObservabilitySession", "StageHandle", "config_hash", "current_session",
]
