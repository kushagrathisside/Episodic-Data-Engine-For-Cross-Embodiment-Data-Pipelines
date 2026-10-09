"""Validation framework: checks, context, and the registry.

Design constraint that shapes everything here: **one pass over the frame data,
many checks**.  A naive framework lets each check scan the table itself, which is
O(checks x frames) IO.  With 20 checks that is 20 full scans -- tolerable at 19 k
frames, ruinous at 50 M.  So the engine streams episodes once and fans each
episode's frames out to every episode-scope check.

Checks come in two scopes:

:class:`EpisodeCheck`  sees one episode's frames as an Arrow table.  Emits
                       episode- or frame-level findings.
:class:`DatasetCheck`  runs once with aggregates over all episodes.  Emits
                       distribution- and completeness-level findings.

Every check declares the columns it needs, so the engine reads the *union* of
required columns and nothing more -- a check that only looks at timestamps never
causes state vectors to be decoded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

import pyarrow as pa

from ..schemas.core import DatasetManifest, Episode
from ..schemas.qc import CheckScope, CheckStatus, QCRecord, Severity


@dataclass
class CheckContext:
    """Everything a check may read.

    Checks are given stores rather than loaded data so they can decide their own
    access pattern -- a video check needs the blob store and a decoder, a
    numerical check needs only the columns the engine already streamed.
    """

    manifest: DatasetManifest
    episodes: list[Episode]
    table_store: Any                       # ParquetTableStore
    blob_store: Any                        # BlobStore
    settings: Any                          # Settings
    version_dir: Any = None
    #: Cross-check scratch space (e.g. per-episode stats computed once).
    shared: dict[str, Any] = field(default_factory=dict)

    @property
    def dataset_id(self) -> str:
        """Dataset under validation."""
        return self.manifest.dataset_id

    @property
    def version(self) -> str:
        """Version under validation."""
        return self.manifest.version

    @property
    def fps(self) -> float:
        """Declared sampling rate, defaulting to 30."""
        return self.manifest.fps or 30.0

    def record(
        self,
        check_name: str,
        scope: CheckScope,
        severity: Severity,
        status: CheckStatus,
        reason: str,
        episode_id: int | None = None,
        frame_id: int | None = None,
        modality: str | None = None,
        observed: Any = None,
        expected: Any = None,
        **details: Any,
    ) -> QCRecord:
        """Build a fully-populated QC record with the context filled in."""
        return QCRecord(
            check_name=check_name,
            scope=scope,
            severity=severity,
            status=status,
            dataset_id=self.dataset_id,
            version=self.version,
            episode_id=episode_id,
            frame_id=frame_id,
            modality=modality,
            observed_value=None if observed is None else str(observed),
            expected_value=None if expected is None else str(expected),
            reason=reason,
            details=details,
        )


class Check(ABC):
    """Base for all validation checks."""

    #: Stable identifier; appears in every emitted record and in the CLI.
    name: str = "unnamed"
    #: Default severity for findings this check produces.
    severity: Severity = Severity.WARNING
    #: Human-readable purpose, surfaced by ``rde qc list-checks``.
    description: str = ""
    #: Frame-table columns the check needs; the engine reads their union.
    required_columns: Sequence[str] = ()
    #: Execution order within a scope, ascending.  Checks that populate
    #: ``ctx.shared`` for others (the range profiler) use a negative priority so
    #: their output exists before dependants run.
    priority: int = 0

    def dynamic_columns(self, ctx: CheckContext) -> list[str]:
        """Extra frame columns needed that cannot be named statically.

        ``required_columns`` is a class attribute, so it cannot reference columns
        whose names depend on the dataset's modalities -- ``rgb.top__ts_in_blob``
        exists only for datasets that have a top camera.  Checks that need such
        columns declare them here instead, and the engine adds them to the
        projection it reads.
        """
        return []

    def applies_to(self, ctx: CheckContext) -> bool:
        """Whether the check is meaningful for this dataset.

        Returning False produces a SKIPPED record rather than a silent no-op, so
        "was tactile validated?" is answerable with "no, the dataset has none"
        instead of ambiguous absence.
        """
        return True

    def skip_reason(self, ctx: CheckContext) -> str:
        """Human-readable reason shown on the SKIPPED record."""
        return "not applicable to this dataset"


class EpisodeCheck(Check):
    """A check evaluated once per episode against that episode's frames."""

    scope = CheckScope.EPISODE

    @abstractmethod
    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Yield findings for one episode.  Yield nothing when it is clean."""


class DatasetCheck(Check):
    """A check evaluated once over the whole dataset."""

    scope = CheckScope.DATASET

    @abstractmethod
    def check_dataset(self, ctx: CheckContext) -> Iterator[QCRecord]:
        """Yield dataset-level findings."""


class CheckRegistry:
    """Named collection of checks, assembled into validation suites."""

    def __init__(self):
        self._checks: dict[str, Check] = {}

    def register(self, check: Check) -> Check:
        """Add a check. Duplicate names are rejected, not silently replaced."""
        if check.name in self._checks:
            raise ValueError(f"duplicate check name: {check.name}")
        self._checks[check.name] = check
        return check

    def add_all(self, checks: Iterable[Check]) -> "CheckRegistry":
        """Register several checks; returns self for chaining."""
        for c in checks:
            self.register(c)
        return self

    def get(self, name: str) -> Check:
        """One check by name."""
        return self._checks[name]

    def all(self) -> list[Check]:
        """Every registered check."""
        return list(self._checks.values())

    def episode_checks(self) -> list[EpisodeCheck]:
        """Episode-scope checks, ordered by priority."""
        return sorted(
            (c for c in self._checks.values() if isinstance(c, EpisodeCheck)),
            key=lambda c: c.priority,
        )

    def dataset_checks(self, phase: str = "all") -> list[DatasetCheck]:
        """Dataset checks, optionally split around the episode pass.

        ``phase='pre'`` returns the profilers that must populate ``ctx.shared``
        before episode checks run; ``'post'`` returns those that summarise after.
        """
        checks = sorted(
            (c for c in self._checks.values() if isinstance(c, DatasetCheck)),
            key=lambda c: c.priority,
        )
        if phase == "pre":
            return [c for c in checks if c.priority < 0]
        if phase == "post":
            return [c for c in checks if c.priority >= 0]
        return checks

    def required_columns(self) -> list[str]:
        """Union of the static columns every episode check declares."""
        cols: set[str] = set()
        for c in self.episode_checks():
            cols.update(c.required_columns)
        return sorted(cols)

    def select(self, names: Sequence[str] | None) -> "CheckRegistry":
        """Sub-registry containing only the named checks.

        Always returns a fresh registry, even when ``names`` is falsy --
        returning ``self`` in that case would let a caller that mutates the
        result (e.g. registering an extra observability-only check) silently
        and permanently change what this registry runs on every future call.
        """
        r = CheckRegistry()
        for n in (names or self._checks.keys()):
            r.register(self._checks[n])
        return r

    def __len__(self) -> int:
        return len(self._checks)

    def __iter__(self):
        return iter(self._checks.values())
