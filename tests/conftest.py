"""Shared fixtures.

Every test runs against a real synthetic dataset written to a temp directory in
the actual LeRobot v3.0 layout and ingested through the real adapter -- no mocks.
Mocking the source format would defeat the purpose: most of the interesting bugs
in an ingestion layer are in the format handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rde.catalog import Catalog
from rde.config import Paths, Settings, set_settings
from rde.ingestion import ingest
from rde.testing import make_lerobot_dataset


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Engine settings redirected entirely into a temp data root."""
    s = Settings(
        paths=Paths(tmp_path / "rde_data"),
        row_group_size=64,   # small, so row-group behaviour is exercised
        batch_size=32,
    )
    s.paths.ensure()
    set_settings(s)
    return s


@pytest.fixture
def source_dataset(tmp_path: Path) -> Path:
    return make_lerobot_dataset(
        tmp_path / "source", num_episodes=4, frames_per_episode=30, fps=10
    )


@pytest.fixture
def catalog(settings: Settings) -> Catalog:
    return Catalog(settings.paths.catalog_db)


@pytest.fixture
def ingested(settings: Settings, source_dataset: Path, catalog: Catalog):
    """A canonical raw/v1 built from the synthetic source."""
    return ingest(
        source_dataset, "testds", "lerobot", "raw/v1",
        settings=settings, catalog=catalog,
    )


@pytest.fixture
def version_dir(settings: Settings, ingested) -> Path:
    return settings.paths.version_dir("testds", "raw/v1")
