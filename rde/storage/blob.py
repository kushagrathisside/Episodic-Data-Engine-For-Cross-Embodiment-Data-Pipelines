"""Content-addressed blob store for large binary payloads.

Videos, images and (later) point clouds live here; nothing numeric does.  The
store is addressed by SHA-256 of the content, which buys three properties the
engine relies on:

*Deduplication* -- the same clip referenced by ``raw/v1``, ``clean/v1`` and
``training/v1`` is stored once.  This is why derived versions cost kilobytes
instead of copying 470 MB of video per stage.

*Immutability* -- a blob id names exactly one byte sequence forever, so a lineage
record that cites a blob id is a real guarantee, not a filename that might have
been overwritten.

*Integrity* -- corruption is detectable by rehashing, and the QC layer uses that.

Ingestion defaults to ``link`` mode: the source cache already holds the bytes, so
the store hardlinks rather than duplicating them.  ``copy`` is available when the
source is temporary.  The public API is deliberately object-store shaped
(``put``/``get``/``exists``/``uri``) so swapping the backend for S3 touches this
file and nothing else.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

_log = logging.getLogger(__name__)

_HASH_CHUNK = 4 * 1024 * 1024  # 4 MB reads keep hashing off the RAM budget


@dataclass(frozen=True)
class BlobInfo:
    """Metadata sidecar for one stored blob."""

    blob_id: str
    size_bytes: int
    media_type: str
    original_name: str | None = None
    attributes: dict | None = None

    def to_json(self) -> dict:
        """Serialise for the sidecar written next to the blob."""
        return {
            "blob_id": self.blob_id,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "original_name": self.original_name,
            "attributes": self.attributes or {},
        }


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


class BlobStore:
    """Local-filesystem object store with content addressing.

    Layout mirrors what an object store would use as a key prefix, so a future
    S3 backend can keep identical ids::

        blobs/<aa>/<bb>/<full-sha256>[.ext]
        blobs/<aa>/<bb>/<full-sha256>.meta.json
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- addressing ---------------------------------------------------------
    def _shard(self, blob_id: str) -> Path:
        return self.root / blob_id[:2] / blob_id[2:4]

    def path_for(self, blob_id: str, suffix: str = "") -> Path:
        """Sharded path for a blob id; the shard mirrors an object-store key prefix."""
        return self._shard(blob_id) / f"{blob_id}{suffix}"

    def _meta_path(self, blob_id: str) -> Path:
        return self._shard(blob_id) / f"{blob_id}.meta.json"

    def uri(self, blob_id: str) -> str:
        """Stable identifier safe to persist in manifests and lineage."""
        return f"blob://{blob_id}"

    # -- writes -------------------------------------------------------------
    def put(
        self,
        source: Path,
        media_type: str = "application/octet-stream",
        mode: Literal["link", "copy"] = "link",
        attributes: dict | None = None,
    ) -> BlobInfo:
        """Admit a file into the store and return its :class:`BlobInfo`.

        Idempotent: re-putting identical bytes is a no-op that returns the
        existing id, which is what makes re-ingestion cheap.
        """
        source = Path(source)
        blob_id = sha256_file(source)
        suffix = source.suffix
        dest = self.path_for(blob_id, suffix)
        info = BlobInfo(
            blob_id=blob_id,
            size_bytes=source.stat().st_size,
            media_type=media_type,
            original_name=source.name,
            attributes=attributes,
        )

        if dest.exists():
            return info  # already stored; content address guarantees identity

        dest.parent.mkdir(parents=True, exist_ok=True)
        # A unique per-call name (not a fixed `dest + ".tmp"`) so two
        # concurrent put()s for the same not-yet-stored blob_id don't write
        # through the same path with independent buffered writers, which
        # could publish a corrupted file at `dest`.
        fd, tmp_name = tempfile.mkstemp(
            dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp"
        )
        os.close(fd)
        tmp = Path(tmp_name)
        tmp.unlink()  # mkstemp pre-creates the file; os.link requires the
                       # destination to not exist, so free the reserved name
                       # while keeping it (cryptographically) unique.
        try:
            if mode == "link":
                try:
                    os.link(source, tmp)          # same filesystem: zero extra bytes
                except OSError as exc:
                    # os.link is all-or-nothing: a failed attempt never
                    # creates `tmp`, so the copy fallback can reuse it as-is.
                    if exc.errno != errno.EXDEV:
                        # Anything other than "cross-device link" (permission
                        # error, quota, a misconfigured mount) silently
                        # degrading "zero extra bytes" into a full copy would
                        # accumulate unbounded, invisible duplication -- the
                        # exact cost this store exists to avoid. Surface it.
                        _log.warning(
                            "blob store: hardlink failed for %s (%s), falling "
                            "back to a full copy",
                            source, exc,
                        )
                    shutil.copy2(source, tmp)     # cross-device: fall back to copy
            else:
                shutil.copy2(source, tmp)
            os.replace(tmp, dest)                 # atomic publish
        finally:
            tmp.unlink(missing_ok=True)  # no-op once replaced into dest

        self._meta_path(blob_id).write_text(json.dumps(info.to_json(), indent=2))
        return info

    # -- reads --------------------------------------------------------------
    def exists(self, blob_id: str) -> bool:
        """Whether the blob is present, whatever extension it was stored under."""
        return self.resolve(blob_id) is not None

    def resolve(self, blob_id: str) -> Path | None:
        """Locate the stored file regardless of the extension it was given."""
        shard = self._shard(blob_id)
        if not shard.is_dir():
            return None
        for p in shard.iterdir():
            if p.name.startswith(blob_id) and not p.name.endswith((".meta.json", ".tmp")):
                return p
        return None

    def get(self, blob_id: str) -> Path:
        """Path to the stored blob. Raises KeyError if absent."""
        p = self.resolve(blob_id)
        if p is None:
            raise KeyError(f"blob not found: {blob_id}")
        return p

    def info(self, blob_id: str) -> BlobInfo | None:
        """Read the sidecar metadata, or None if the blob is unknown."""
        mp = self._meta_path(blob_id)
        if not mp.exists():
            return None
        d = json.loads(mp.read_text())
        return BlobInfo(
            blob_id=d["blob_id"],
            size_bytes=d["size_bytes"],
            media_type=d["media_type"],
            original_name=d.get("original_name"),
            attributes=d.get("attributes"),
        )

    def verify(self, blob_id: str) -> bool:
        """Rehash stored bytes and confirm they still match the address."""
        p = self.resolve(blob_id)
        return p is not None and sha256_file(p) == blob_id

    def iter_blobs(self) -> Iterator[BlobInfo]:
        """Every blob in the store, read from the sidecars."""
        for meta in self.root.rglob("*.meta.json"):
            d = json.loads(meta.read_text())
            yield BlobInfo(
                blob_id=d["blob_id"],
                size_bytes=d["size_bytes"],
                media_type=d["media_type"],
                original_name=d.get("original_name"),
                attributes=d.get("attributes"),
            )

    def total_bytes(self) -> int:
        """Logical size of everything stored, ignoring hardlink sharing."""
        return sum(b.size_bytes for b in self.iter_blobs())
