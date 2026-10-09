"""Catalog tests: the metadata system of record."""

from __future__ import annotations

import pytest

from rde.catalog import Catalog
from rde.schemas.core import QualityStatus
from rde.schemas.qc import CheckScope, CheckStatus, QCRecord, Severity


def test_register_and_read_manifest(settings, catalog, ingested):
    m = catalog.get_manifest("testds", "raw/v1")
    assert m is not None
    assert m.total_episodes == 4
    assert m.modality_names


def test_versions_are_listed_with_stage(settings, catalog, ingested):
    versions = catalog.list_versions("testds")
    assert versions[0]["stage"] == "raw"
    assert versions[0]["total_frames"] == 120


def test_episode_quality_updates_persist(settings, catalog, ingested):
    catalog.set_episode_quality("testds", "raw/v1", {0: QualityStatus.REJECTED})
    rows = catalog.get_episodes("testds", "raw/v1", quality_status=QualityStatus.REJECTED)
    assert [r["episode_id"] for r in rows] == [0]


def test_qc_records_round_trip(settings, catalog, ingested):
    rec = QCRecord(
        check_name="test.check", scope=CheckScope.EPISODE, severity=Severity.ERROR,
        status=CheckStatus.FAILED, dataset_id="testds", version="raw/v1",
        episode_id=1, observed_value="3", expected_value="4", reason="mismatch",
    )
    catalog.record_qc([rec])
    rows = catalog.query_qc("testds", "raw/v1", check_name="test.check")
    assert len(rows) == 1
    assert rows[0]["reason"] == "mismatch"


def test_clear_qc_prevents_duplicate_accumulation(settings, catalog, ingested):
    rec = QCRecord(
        check_name="test.check", scope=CheckScope.DATASET, severity=Severity.INFO,
        status=CheckStatus.PASSED, dataset_id="testds", version="raw/v1",
    )
    catalog.record_qc([rec])
    catalog.clear_qc("testds", "raw/v1")
    catalog.record_qc([rec])
    assert len(catalog.query_qc("testds", "raw/v1", check_name="test.check")) == 1


def test_migration_adds_columns_to_existing_db(tmp_path):
    """An older catalog must gain new columns rather than silently lose data."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """CREATE TABLE transform_runs (
               run_id TEXT PRIMARY KEY, dataset_id TEXT, operation TEXT,
               input_versions TEXT, output_version TEXT, parameters TEXT,
               started_at TEXT, finished_at TEXT, software_version TEXT,
               code_fingerprint TEXT, metrics TEXT, dropped_episodes TEXT);"""
    )
    conn.commit()
    conn.close()

    Catalog(db)  # opening must migrate
    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(transform_runs)").fetchall()}
    conn.close()
    assert {"input_episodes", "output_episodes"} <= cols


def test_cascade_delete_cleans_children(settings, catalog, ingested):
    with catalog.transaction() as conn:
        conn.execute("DELETE FROM versions WHERE version_key=?", ("testds@raw/v1",))
    assert catalog.get_episodes("testds", "raw/v1") == []
