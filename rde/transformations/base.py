"""Composable transformation operators.

Every transformation is an independent object that declares its parameters and
reports what it did.  A pipeline is an ordered list of them.  That structure is
what makes lineage real rather than decorative: the recorded
``(operation, parameters, code fingerprint, input version)`` tuple is sufficient
to reconstruct the output, because there is nowhere else for behaviour to hide.

Three operator kinds, distinguished by what they act on:

:class:`FrameTransform`   rewrites columns of the frame table, streaming.
:class:`FrameFilter`      drops rows by predicate.
:class:`EpisodeFilter`    drops whole episodes.

Operators that need dataset statistics (normalizers) implement :meth:`fit`, which
gets one streaming pass before the write pass.  Fitted parameters are recorded in
the lineage entry, so a training-time normalizer can be reproduced exactly rather
than recomputed and hoped to match.

**Deferred transforms.**  Image resizing is deliberately *not* a materialising
operator.  Re-encoding 470 MB of AV1 to write a resized copy would cost minutes
and gigabytes for a result the model adapter can produce in microseconds at
decode time.  Such operators declare themselves via
:attr:`Transform.deferred` and record their parameters into the manifest, where
the model adapter reads and applies them.  The lineage is identical; only the
materialisation point differs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import pyarrow as pa

from ..schemas.core import DatasetManifest, Episode


@dataclass
class TransformContext:
    """State shared across a pipeline run."""

    manifest: DatasetManifest
    episodes: list[Episode]
    settings: Any
    table_store: Any = None
    blob_store: Any = None
    #: Statistics and fitted parameters, keyed by operator name.
    fitted: dict[str, Any] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        """Sampling rate of the input version."""
        return self.manifest.fps or 30.0


class Transform(ABC):
    """Base class for all transformation operators."""

    name: str = "unnamed"
    description: str = ""
    #: True when the operator is applied at read time by the model adapter
    #: instead of rewriting stored data.
    deferred: bool = False
    #: True when the operator must see the data once before it can be applied.
    needs_fit: bool = False

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Everything needed to reproduce this operator's behaviour."""

    def fit(self, ctx: TransformContext, batches: Iterator[pa.RecordBatch]) -> None:
        """Consume one streaming pass to learn parameters.  Default: no-op."""
        return None

    def update_manifest(self, manifest: DatasetManifest, ctx: TransformContext) -> None:
        """Record the operator's effect on the canonical schema/metadata."""
        return None

    def describe(self) -> dict[str, Any]:
        """Name, parameters and materialisation mode, for the lineage record."""
        return {
            "name": self.name,
            "description": self.description,
            "deferred": self.deferred,
            "parameters": self.parameters(),
        }


class FrameTransform(Transform):
    """Rewrites frame-table columns, one batch at a time."""

    @abstractmethod
    def apply_batch(self, batch: pa.RecordBatch, ctx: TransformContext) -> pa.RecordBatch:
        """Return the transformed batch.  Must not accumulate state across calls."""


class FrameFilter(Transform):
    """Drops rows from the frame table."""

    @abstractmethod
    def mask(self, batch: pa.RecordBatch, ctx: TransformContext) -> pa.Array:
        """Boolean mask, True to keep."""


class EpisodeFilter(Transform):
    """Selects which episodes survive into the output version."""

    @abstractmethod
    def keep(self, episodes: Sequence[Episode], ctx: TransformContext) -> list[int]:
        """Return the episode ids to retain."""
