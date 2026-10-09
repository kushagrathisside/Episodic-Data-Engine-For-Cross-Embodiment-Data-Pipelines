"""Machine-readable quality-control records.

The engine never silently drops data.  Every finding -- structural, temporal,
numerical or dataset-level -- becomes a :class:`QCRecord` that is persisted
alongside the data it describes.  That record is what makes "why was episode 17
excluded from training/v1?" an answerable question months later.

Severity vs status:

* ``severity`` is a property of the *check* -- how bad this class of finding is.
* ``status``   is the outcome of *this evaluation* -- pass, fail, warn, skip.

A record is written for failures and warnings.  Passing checks are aggregated into
counters rather than stored per-frame, because 19k frames x 20 checks of "OK" is
noise that costs more to store than it is worth.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """How much a finding should be allowed to affect downstream use."""

    INFO = "info"          # observation only, never gates
    WARNING = "warning"    # tolerated; episode keeps flowing but is flagged
    ERROR = "error"        # blocking; episode is rejected from clean datasets
    CRITICAL = "critical"  # dataset-level breakage; halts the run

    @property
    def is_blocking(self) -> bool:
        """True when this record both failed and did so at a blocking severity."""
        return self in (Severity.ERROR, Severity.CRITICAL)


class CheckStatus(str, Enum):
    """Outcome of one evaluation, as distinct from the check's severity."""
    PASSED = "passed"
    FAILED = "failed"
    WARNED = "warned"
    SKIPPED = "skipped"  # e.g. modality absent in this dataset


class CheckScope(str, Enum):
    """Granularity the check operates at -- drives how it is scheduled."""

    FRAME = "frame"
    EPISODE = "episode"
    DATASET = "dataset"


class QCRecord(BaseModel):
    """One finding, at one scope, from one check.

    Field names match the contract required by the pipeline spec so the table can
    be queried directly:
    ``episode_id, frame_id, check_name, severity, observed_value, expected_value,
    status, reason``.
    """

    check_name: str
    scope: CheckScope
    severity: Severity
    status: CheckStatus

    dataset_id: str
    version: str
    episode_id: int | None = None
    frame_id: int | None = None
    modality: str | None = None

    # Kept as strings so heterogeneous observations (floats, counts, shapes,
    # dtype names) live in one columnar table without a union type.
    observed_value: str | None = None
    expected_value: str | None = None

    reason: str = ""
    #: Structured payload for findings that need more than two scalars.
    details: dict[str, Any] = Field(default_factory=dict)
    checked_at: _dt.datetime = Field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc)
    )

    @property
    def is_blocking(self) -> bool:
        """Whether a finding at this severity should reject an episode."""
        return self.status is CheckStatus.FAILED and self.severity.is_blocking

    def as_row(self) -> dict[str, Any]:
        """Flat dict for the Parquet QC table."""
        import json

        return {
            "check_name": self.check_name,
            "scope": self.scope.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "episode_id": self.episode_id,
            "frame_id": self.frame_id,
            "modality": self.modality,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "reason": self.reason,
            "details": json.dumps(self.details, default=str) if self.details else None,
            "checked_at": self.checked_at,
        }


class QCReport(BaseModel):
    """Aggregate outcome of one validation run over one dataset version."""

    dataset_id: str
    version: str
    started_at: _dt.datetime
    finished_at: _dt.datetime | None = None

    records: list[QCRecord] = Field(default_factory=list)
    #: check_name -> count of entities that passed (not stored per-entity).
    passed_counts: dict[str, int] = Field(default_factory=dict)
    checks_run: list[str] = Field(default_factory=list)

    @property
    def blocking(self) -> list[QCRecord]:
        """Findings that failed at a blocking severity."""
        return [r for r in self.records if r.is_blocking]

    @property
    def rejected_episodes(self) -> set[int]:
        """Episode ids with at least one blocking finding."""
        return {
            r.episode_id
            for r in self.records
            if r.is_blocking and r.episode_id is not None
        }

    @property
    def warned_episodes(self) -> set[int]:
        """Episode ids with warnings but nothing blocking."""
        return {
            r.episode_id
            for r in self.records
            if r.status is CheckStatus.WARNED and r.episode_id is not None
        }

    def summary(self) -> dict[str, Any]:
        """Counts by severity and check, plus the gated episode lists."""
        by_sev: dict[str, int] = {}
        by_check: dict[str, int] = {}
        for r in self.records:
            by_sev[r.severity.value] = by_sev.get(r.severity.value, 0) + 1
            by_check[r.check_name] = by_check.get(r.check_name, 0) + 1
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "checks_run": len(self.checks_run),
            "findings": len(self.records),
            "by_severity": by_sev,
            "by_check": by_check,
            "rejected_episodes": sorted(self.rejected_episodes),
            "warned_episodes": sorted(self.warned_episodes),
        }
