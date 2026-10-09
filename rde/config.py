"""Runtime configuration and on-disk layout.

Two roots, kept deliberately separate:

``CODE``  the repository you are reading -- versioned in git, small, no payloads.
``DATA``  every byte the engine ingests or produces -- caches, blobs, Parquet
          tables, catalog db, artifacts.  Defaults to a sibling of the repo and is
          overridable with ``RDE_DATA_ROOT``.

Keeping data out of the source tree is not cosmetic.  The blob store holds
hundreds of MB now and is designed to hold TB later; letting it grow inside the
package directory would make the repo unclonable, poison editor indexing, and
tempt someone into committing a 500 MB mp4.  The split also mirrors how this
would deploy for real -- code in an image, data in an object store -- so the
`DATA` root is the single thing you repoint at S3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Bumped with the package; recorded in every lineage run for reproducibility.
SOFTWARE_VERSION = "0.1.0"

_DEFAULT_DATA_ROOT = Path.home() / "VLAExperiments" / "rde_data"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem layout for one engine instance."""

    data_root: Path

    @property
    def source_cache(self) -> Path:
        """Untouched downloads exactly as the upstream published them."""
        return self.data_root / "source_cache"

    @property
    def blobs(self) -> Path:
        """Content-addressed binary plane (videos, images, point clouds)."""
        return self.data_root / "blobs"

    @property
    def tables(self) -> Path:
        """Structured numeric plane: <dataset_id>/<stage>/<vN>/..."""
        return self.data_root / "tables"

    @property
    def artifacts(self) -> Path:
        """Derived non-tabular outputs: QC reports, stats, manifests, exports."""
        return self.data_root / "artifacts"

    @property
    def catalog_db(self) -> Path:
        """SQLite metadata catalog (datasets, versions, lineage, QC index)."""
        return self.data_root / "catalog.db"

    @property
    def runs(self) -> Path:
        """Per-run logs and machine-readable execution records."""
        return self.data_root / "runs"

    def version_dir(self, dataset_id: str, version: str) -> Path:
        """Table directory for one version; ``version`` is '<stage>/v<N>'."""
        return self.tables / dataset_id / version

    def artifact_dir(self, dataset_id: str, version: str) -> Path:
        """Artifact directory for one version: manifests, QC, evaluations."""
        return self.artifacts / dataset_id / version

    def ensure(self) -> "Paths":
        """Create every root directory. Idempotent; returns self."""
        for p in (
            self.source_cache,
            self.blobs,
            self.tables,
            self.artifacts,
            self.runs,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class Settings:
    """Engine-wide knobs.  Every default here is chosen for an 8 GB RAM box."""

    paths: Paths = field(
        default_factory=lambda: Paths(_env_path("RDE_DATA_ROOT", _DEFAULT_DATA_ROOT))
    )

    #: Rows per Parquet row group.  Small enough that a single group is a few MB,
    #: which is what lets readers stream batches instead of whole files.
    row_group_size: int = 2048
    #: Frames processed per in-memory batch during validation/transformation.
    batch_size: int = 1024
    #: Episodes held open at once by the builder.
    episode_concurrency: int = 1

    parquet_compression: str = "zstd"
    #: Decoded-frame LRU budget for the video reader, in frames (480x640x3 ~ 0.9 MB).
    video_cache_frames: int = 64

    hf_token: str | None = None
    software_version: str = SOFTWARE_VERSION

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the environment, honouring RDE_* overrides."""
        return cls(
            paths=Paths(_env_path("RDE_DATA_ROOT", _DEFAULT_DATA_ROOT)),
            row_group_size=int(os.environ.get("RDE_ROW_GROUP_SIZE", 2048)),
            batch_size=int(os.environ.get("RDE_BATCH_SIZE", 1024)),
            hf_token=os.environ.get("HF_TOKEN"),
        )


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton, initialised from the environment."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.from_env()
    return _SETTINGS


def set_settings(settings: Settings) -> None:
    """Override settings -- used by tests to redirect the data root to tmp."""
    global _SETTINGS
    _SETTINGS = settings
