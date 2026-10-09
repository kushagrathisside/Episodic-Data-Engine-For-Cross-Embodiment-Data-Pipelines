"""Incremental validation: don't re-check data that provably has not changed.

The engine's central storage invariant is that **versions are immutable**.
``raw/v1`` is written once; superseding it means creating ``clean/v1``, never
editing it in place.  Blobs are content-addressed, so a blob's identity *is* its
content.

That invariant has a consequence nobody was collecting: re-validating an
unchanged version with an unchanged check suite must produce the same findings,
so running it again is not caution, it is waste.  On this dataset ``validate``
is 2.7 s of a 6.2 s run and ``qc_post_transform`` another 2.6 s -- 86% of the
pipeline -- and 97% of that is two video checks.

The cache key is therefore:

    (version_key, suite fingerprint, table fingerprint)

``suite fingerprint``  SHA-256 over each check's name and the source of its
                       class.  Edit a check and every cached entry that used it
                       is invalidated -- which is what makes recalibrating
                       ``value_range`` safe rather than silently reusing results
                       computed under the old bounds.
``table fingerprint``  size and mtime of every Parquet file in the version
                       directory.  **This is a safety net, not the correctness
                       argument.**  Immutability is a design invariant, not
                       something the filesystem enforces: a fault injector
                       rewrites a version's frames in place, and a cache that
                       trusted immutability alone would happily report the
                       pre-injection findings and make the fault demos lie.

Cached entries live in the catalog and store the findings themselves, so a hit
returns a fully-formed :class:`QCResult` and not merely a "still fine".
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Sequence

from ..schemas.core import QualityStatus
from ..schemas.qc import CheckScope, CheckStatus, QCRecord, QCReport, Severity


def suite_fingerprint(checks: Sequence[Any]) -> str:
    """Hash the identity, implementation, *and configuration* of a check suite.

    Names alone are not enough.  When ``value_range`` was recalibrated its name
    did not change, and a name-keyed cache would have served findings computed
    under the old percentile bounds -- the recalibration would have looked like
    it did nothing.  Hashing the source means a changed check invalidates
    exactly the entries that used it -- but recalibration is not always a code
    edit: a check retuned by passing different constructor arguments (e.g.
    ``ValueRangeCheck(min_violation_ratio=0.05)``) has identical class source
    to the untuned check, so the class hash alone would still serve stale,
    pre-tuning findings. The check's own ``__dict__`` is folded in too, so a
    parameter change invalidates the cache exactly like a code change does.
    """
    parts: list[str] = []
    for check in sorted(checks, key=lambda c: c.name):
        try:
            src = inspect.getsource(type(check))
        except (OSError, TypeError):
            src = type(check).__qualname__
        try:
            config = repr(sorted(vars(check).items(), key=lambda kv: kv[0]))
        except TypeError:
            config = repr(vars(check))
        digest = hashlib.sha256((src + "\0" + config).encode()).hexdigest()[:16]
        parts.append(f"{check.name}:{digest}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def blob_fingerprint(blob_store: Any, blob_ids: Sequence[str]) -> str:
    """Existence and size of the blobs a version references.

    The table fingerprint does not cover the binary plane, and that gap is
    reachable: ``inject_missing_blob`` deletes a video without touching a single
    Parquet byte, so a cache keyed on tables alone would serve the pre-deletion
    findings and report a dataset healthy while its video was gone.

    Resolution goes through the store rather than reconstructing its layout, so
    a change to the fan-out scheme cannot silently turn every lookup into a
    "missing" that still hashes consistently.

    Only the referenced blobs are stat-ed, not the whole store, so the cost is
    proportional to the version rather than to everything ever ingested.
    """
    h = hashlib.sha256()
    for blob_id in sorted(set(blob_ids)):
        try:
            path = blob_store.resolve(blob_id)
            h.update(
                f"{blob_id}:{path.stat().st_size}".encode() if path
                else f"{blob_id}:missing".encode()
            )
        except OSError:
            h.update(f"{blob_id}:unreadable".encode())
    return h.hexdigest()[:16]


def table_fingerprint(version_dir: Path) -> str:
    """Cheap change detector over a version's Parquet files.

    Size and mtime rather than content: hashing 470 MB to decide whether to skip
    2.7 s of work would cost more than the work.  This is deliberately a
    heuristic -- it exists to catch the case where the immutability invariant is
    violated (in-place fault injection, a manual edit), not to prove content
    equality, which the version contract already asserts.
    """
    if not version_dir.exists():
        return "missing"
    h = hashlib.sha256()
    for p in sorted(version_dir.rglob("*.parquet")):
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS qc_cache (
    cache_key        TEXT PRIMARY KEY,
    dataset_id       TEXT NOT NULL,
    version          TEXT NOT NULL,
    suite_hash       TEXT NOT NULL,
    table_hash       TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    records_json     TEXT NOT NULL,
    statuses_json    TEXT NOT NULL,
    dataset_status   TEXT NOT NULL,
    checks_run_json  TEXT NOT NULL,
    hits             INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_qc_cache_version ON qc_cache(dataset_id, version);
"""


def cache_key(dataset_id: str, version: str, suite: str, tables: str) -> str:
    """Stable key over everything a validation result depends on.

    ``tables`` is expected to already fold in the blob fingerprint; they are
    combined by the caller so that both planes are covered by one key.
    """
    return hashlib.sha256(
        f"{dataset_id}@{version}|{suite}|{tables}".encode()
    ).hexdigest()[:24]


class QCCache:
    """Store and retrieve validation results keyed by what they depend on."""

    def __init__(self, catalog: Any):
        self.catalog = catalog
        with self.catalog.transaction() as conn:
            conn.executescript(CACHE_SCHEMA)

    def lookup(
        self, dataset_id: str, version: str, suite: str, tables: str
    ) -> tuple[list[QCRecord], dict[int, QualityStatus], QualityStatus, list[str]] | None:
        """Return a cached validation, or None on a miss.

        A hit increments a counter, so cache effectiveness is itself
        observable -- a cache whose hit rate nobody can see is a cache nobody
        should trust.
        """
        key = cache_key(dataset_id, version, suite, tables)
        with self.catalog.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM qc_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE qc_cache SET hits = hits + 1 WHERE cache_key = ?", (key,)
            )
        records = [QCRecord(**_revive(r)) for r in json.loads(row["records_json"])]
        statuses = {
            int(k): QualityStatus(v) for k, v in json.loads(row["statuses_json"]).items()
        }
        return (
            records,
            statuses,
            QualityStatus(row["dataset_status"]),
            json.loads(row["checks_run_json"]),
        )

    def store(
        self,
        dataset_id: str,
        version: str,
        suite: str,
        tables: str,
        records: Sequence[QCRecord],
        statuses: dict[int, QualityStatus],
        dataset_status: QualityStatus,
        checks_run: Sequence[str],
    ) -> str:
        """Persist one validation result.  Overwrites an entry with the same key."""
        key = cache_key(dataset_id, version, suite, tables)
        with self.catalog.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO qc_cache
                   (cache_key, dataset_id, version, suite_hash, table_hash,
                    created_at, records_json, statuses_json, dataset_status,
                    checks_run_json, hits)
                   VALUES (?,?,?,?,?,?,?,?,?,?,0)""",
                (
                    key, dataset_id, version, suite, tables,
                    _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    json.dumps([_flatten(r) for r in records]),
                    json.dumps({str(k): v.value for k, v in statuses.items()}),
                    dataset_status.value,
                    json.dumps(sorted(set(checks_run))),
                ),
            )
        return key

    def invalidate(self, dataset_id: str, version: str | None = None) -> int:
        """Drop cached results for a dataset, or for one of its versions."""
        sql = "DELETE FROM qc_cache WHERE dataset_id = ?"
        params: list[Any] = [dataset_id]
        if version:
            sql += " AND version = ?"
            params.append(version)
        with self.catalog.transaction() as conn:
            return conn.execute(sql, params).rowcount

    def stats(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        """Cached entries with their hit counts."""
        sql = ("SELECT dataset_id, version, suite_hash, table_hash, created_at, "
               "hits, length(records_json) AS bytes FROM qc_cache")
        params: list[Any] = []
        if dataset_id:
            sql += " WHERE dataset_id = ?"
            params.append(dataset_id)
        sql += " ORDER BY created_at DESC"
        with self.catalog.transaction() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _flatten(r: QCRecord) -> dict[str, Any]:
    """QCRecord -> JSON-safe dict, enums as values."""
    return {
        "check_name": r.check_name, "scope": r.scope.value,
        "severity": r.severity.value, "status": r.status.value,
        "dataset_id": r.dataset_id, "version": r.version,
        "episode_id": r.episode_id, "frame_id": r.frame_id,
        "modality": r.modality, "observed_value": r.observed_value,
        "expected_value": r.expected_value, "reason": r.reason,
        "details": r.details, "checked_at": r.checked_at.isoformat(),
    }


def _revive(d: dict[str, Any]) -> dict[str, Any]:
    """JSON dict -> QCRecord kwargs, enums restored."""
    return {
        **d,
        "scope": CheckScope(d["scope"]),
        "severity": Severity(d["severity"]),
        "status": CheckStatus(d["status"]),
        "checked_at": _dt.datetime.fromisoformat(d["checked_at"]),
    }


def cached_report(
    dataset_id: str, version: str, records: list[QCRecord], checks_run: list[str]
) -> QCReport:
    """Rebuild a QCReport from cached findings.

    ``started_at``/``finished_at`` are set to now with zero duration, which is
    honest: this validation did not take any time, because it did not happen.
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    return QCReport(
        dataset_id=dataset_id, version=version,
        started_at=now, finished_at=now,
        records=records, checks_run=list(checks_run),
    )
