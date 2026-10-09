"""Acquisition of source datasets into the immutable source cache.

The cache is deliberately a *verbatim* copy of what upstream published.  Nothing
in the engine ever writes back into it.  That gives two things:

* re-ingestion is cheap and offline -- you can rebuild every derived version from
  the cache without touching the network;
* the "did the engine corrupt this, or did it arrive that way?" question stays
  answerable, because the pristine bytes are still on disk.

Downloads are resumable and go straight to the data root configured in
:mod:`rde.config`, never into the source tree.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings, get_settings


@dataclass
class SourceLocation:
    """Where a source dataset landed, plus how to identify it later."""

    source_uri: str          # e.g. 'hf://lerobot/svla_so100_pickplace@main'
    local_path: Path
    revision: str | None = None
    size_bytes: int = 0
    file_count: int = 0

    def to_json(self) -> dict:
        """Serialise for the .rde_source.json marker."""
        return {
            "source_uri": self.source_uri,
            "local_path": str(self.local_path),
            "revision": self.revision,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
        }


def _dir_stats(path: Path) -> tuple[int, int]:
    """(total bytes, file count) following symlinks into the HF blob store."""
    total = 0
    count = 0
    for p in path.rglob("*"):
        if p.is_file() or (p.is_symlink() and p.resolve().is_file()):
            try:
                total += p.stat().st_size
            except OSError:
                continue
            count += 1
    return total, count


def download_hf_dataset(
    repo_id: str,
    revision: str = "main",
    settings: Settings | None = None,
    allow_patterns: list[str] | None = None,
    force: bool = False,
) -> SourceLocation:
    """Snapshot a Hugging Face dataset repo into ``paths.source_cache``.

    ``allow_patterns`` lets a caller pull metadata only -- useful for inspecting a
    large dataset's schema before committing to its video payload.
    """
    from huggingface_hub import snapshot_download

    settings = settings or get_settings()
    settings.paths.ensure()

    target = settings.paths.source_cache / repo_id.replace("/", "__")
    marker = target / ".rde_source.json"

    if marker.exists() and not force and allow_patterns is None:
        try:
            meta = json.loads(marker.read_text())
        except (OSError, ValueError):
            # A marker from a process killed mid-write is a cache miss, not
            # a crash: fall through and re-download.
            meta = None
        # The marker is keyed by repo_id alone (the cache dir doesn't
        # encode revision), so it must be checked against the requested
        # revision explicitly -- otherwise a later call for a different
        # revision would silently return this (unrelated) snapshot.
        if meta is not None and meta.get("revision") == revision:
            return SourceLocation(
                source_uri=meta["source_uri"],
                local_path=Path(meta["local_path"]),
                revision=meta.get("revision"),
                size_bytes=meta.get("size_bytes", 0),
                file_count=meta.get("file_count", 0),
            )

    # hf_transfer gives a large speedup on multi-hundred-MB video payloads,
    # but huggingface_hub raises at download time if this is set without the
    # optional package installed -- only enable it when it's actually there.
    import importlib.util

    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(target),
        allow_patterns=allow_patterns,
        token=settings.hf_token,
    )
    local_path = Path(local)
    size, count = _dir_stats(local_path)

    loc = SourceLocation(
        source_uri=f"hf://{repo_id}@{revision}",
        local_path=local_path,
        revision=revision,
        size_bytes=size,
        file_count=count,
    )
    if allow_patterns is None:
        marker.write_text(json.dumps(loc.to_json(), indent=2))
    return loc
