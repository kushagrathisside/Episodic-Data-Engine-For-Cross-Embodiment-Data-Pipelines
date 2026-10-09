"""Structural validation: does the data physically exist and hold together?

These checks run before any numeric interpretation, because a NaN check on a
table that failed to parse is meaningless.  They answer:

* are the files there, and do they still hash to their content address?
* does the Parquet schema match what the manifest promises?
* do episode frame ranges agree with the frames actually stored?
* do the video containers open, and do they hold the frames they claim?

Video integrity is the expensive one.  Fully decoding 19 631 AV1 frames to prove
they exist costs minutes; the check therefore verifies the *header* count against
the manifest for every episode, and does a bounded decode probe at episode
boundaries.  ``deep=True`` escalates to a full decode when a caller is willing to
pay for it.
"""

from __future__ import annotations

from typing import Iterator

import pyarrow as pa

from ..schemas.core import Episode
from ..schemas.modality import StoragePlane
from ..schemas.qc import CheckScope, CheckStatus, QCRecord, Severity
from .base import CheckContext, DatasetCheck, EpisodeCheck


class FrameTableSchemaCheck(DatasetCheck):
    """Stored Parquet schema must match the manifest's modality registry."""

    name = "structural.frame_schema"
    severity = Severity.CRITICAL
    description = "Frame table contains every column the manifest declares, with the declared width."

    def check_dataset(self, ctx: CheckContext) -> Iterator[QCRecord]:
        """Compare the stored Parquet schema against the manifest's registry."""
        if not ctx.table_store.exists("frames"):
            yield ctx.record(
                self.name, CheckScope.DATASET, Severity.CRITICAL, CheckStatus.FAILED,
                "frame table is missing from the version directory",
                observed="absent", expected="frames/*.parquet",
            )
            return

        schema = ctx.table_store.schema("frames")
        names = set(schema.names)

        for col in ("frame_id", "episode_id", "frame_index", "timestamp_s"):
            if col not in names:
                yield ctx.record(
                    self.name, CheckScope.DATASET, Severity.CRITICAL, CheckStatus.FAILED,
                    f"required canonical column {col!r} missing",
                    observed="absent", expected=col,
                )

        for m in ctx.manifest.modalities:
            if m.plane is StoragePlane.INLINE:
                if m.name not in names:
                    yield ctx.record(
                        self.name, CheckScope.DATASET, Severity.ERROR, CheckStatus.FAILED,
                        f"declared modality {m.name!r} has no column",
                        modality=m.name, observed="absent", expected=m.name,
                    )
                    continue
                # A fixed-size list encodes dimensionality in the type itself, so
                # a width mismatch is caught here rather than at training time.
                field = schema.field(m.name)
                width = getattr(field.type, "list_size", None)
                if width is not None and width != m.element_count:
                    yield ctx.record(
                        self.name, CheckScope.DATASET, Severity.ERROR, CheckStatus.FAILED,
                        f"{m.name}: stored width {width} != declared {m.element_count}",
                        modality=m.name, observed=width, expected=m.element_count,
                    )
            else:
                for suffix in ("__blob_id", "__frame_in_blob", "__ts_in_blob"):
                    if f"{m.name}{suffix}" not in names:
                        yield ctx.record(
                            self.name, CheckScope.DATASET, Severity.ERROR, CheckStatus.FAILED,
                            f"external modality {m.name!r} missing column {suffix}",
                            modality=m.name, observed="absent",
                            expected=f"{m.name}{suffix}",
                        )


class BlobPresenceCheck(DatasetCheck):
    """Every referenced blob must exist and still match its content address."""

    name = "structural.blob_integrity"
    severity = Severity.CRITICAL
    description = "Referenced video blobs exist and hash to their recorded content address."

    def __init__(self, verify_hash: bool = False):
        # Rehashing 470 MB costs ~2 s; opt-in so routine QC stays fast while an
        # integrity audit can still be demanded.
        self.verify_hash = verify_hash

    def check_dataset(self, ctx: CheckContext) -> Iterator[QCRecord]:
        """Confirm every referenced blob exists, optionally rehashing it."""
        seen: set[str] = set()
        for ep in ctx.episodes:
            for modality, blob_id in (ep.blob_ids or {}).items():
                if blob_id in seen:
                    continue
                seen.add(blob_id)
                if not ctx.blob_store.exists(blob_id):
                    yield ctx.record(
                        self.name, CheckScope.DATASET, Severity.CRITICAL, CheckStatus.FAILED,
                        f"blob {blob_id[:12]} referenced by {modality} is not in the store",
                        modality=modality, observed="missing", expected="present",
                        blob_id=blob_id,
                    )
                elif self.verify_hash and not ctx.blob_store.verify(blob_id):
                    yield ctx.record(
                        self.name, CheckScope.DATASET, Severity.CRITICAL, CheckStatus.FAILED,
                        f"blob {blob_id[:12]} content no longer matches its hash",
                        modality=modality, observed="hash mismatch", expected=blob_id[:12],
                        blob_id=blob_id,
                    )


class EpisodeFrameCountCheck(EpisodeCheck):
    """Frames actually stored must match the episode's declared range."""

    name = "structural.episode_frame_count"
    severity = Severity.ERROR
    description = "Rows present for an episode equal frame_end - frame_start."
    required_columns = ("frame_id", "episode_id", "frame_index")

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Compare rows stored against the episode's declared range."""
        actual = frames.num_rows
        declared = episode.num_frames
        if actual != declared:
            yield ctx.record(
                self.name, CheckScope.EPISODE, self.severity, CheckStatus.FAILED,
                f"episode {episode.episode_id}: {actual} rows stored, {declared} declared",
                episode_id=episode.episode_id, observed=actual, expected=declared,
            )
            return
        if actual == 0:
            yield ctx.record(
                self.name, CheckScope.EPISODE, Severity.ERROR, CheckStatus.FAILED,
                f"episode {episode.episode_id} contains no frames",
                episode_id=episode.episode_id, observed=0, expected=">0",
            )


class FrameIndexContinuityCheck(EpisodeCheck):
    """frame_index must be a dense 0..n-1 sequence within an episode.

    A gap here means frames were dropped during recording or ingestion.  Unlike a
    timestamp gap it is unambiguous -- the index is definitionally contiguous --
    so this is an ERROR rather than a warning.
    """

    name = "structural.frame_index_continuity"
    severity = Severity.ERROR
    description = "frame_index runs 0..n-1 with no gaps or duplicates."
    required_columns = ("frame_index", "frame_id")

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Confirm frame_index is a dense 0..n-1 sequence."""
        if frames.num_rows == 0:
            return
        import numpy as np

        idx = np.asarray(frames.column("frame_index"))
        expected = np.arange(len(idx))
        if not np.array_equal(idx, expected):
            missing = sorted(set(expected.tolist()) - set(idx.tolist()))
            dupes = len(idx) - len(set(idx.tolist()))
            yield ctx.record(
                self.name, CheckScope.EPISODE, self.severity, CheckStatus.FAILED,
                f"episode {episode.episode_id}: frame_index is not dense 0..{len(idx)-1}",
                episode_id=episode.episode_id,
                observed=f"first_gap={missing[0] if missing else 'none'}, duplicates={dupes}",
                expected=f"0..{len(idx)-1}",
                missing_count=len(missing), duplicate_count=dupes,
            )


class VideoIntegrityCheck(EpisodeCheck):
    """Video containers open, and hold the frames their episodes reference."""

    name = "structural.video_integrity"
    severity = Severity.ERROR
    description = "Referenced video windows are decodable and long enough for the episode."
    required_columns = ("episode_id",)

    def __init__(self, deep: bool = False, probe_frames: int = 2):
        #: deep=True decodes the entire episode window; otherwise probe the ends,
        #: which catches truncation and unreadable streams at ~1% of the cost.
        self.deep = deep
        self.probe_frames = probe_frames

    def applies_to(self, ctx: CheckContext) -> bool:
        """Only meaningful when the dataset declares external modalities."""
        return any(m.is_external for m in ctx.manifest.modalities)

    def skip_reason(self, ctx: CheckContext) -> str:
        """Explain the skip on the QC record."""
        return "dataset declares no external (video) modalities"

    def check_episode(
        self, episode: Episode, frames: pa.Table, ctx: CheckContext
    ) -> Iterator[QCRecord]:
        """Open each referenced window and confirm it is decodable and long enough."""
        from ..storage.video import VideoReader

        windows = episode.attributes.get("video_windows", {}) or {}
        for modality, blob_id in (episode.blob_ids or {}).items():
            win = windows.get(modality)
            if win is None:
                yield ctx.record(
                    self.name, CheckScope.EPISODE, Severity.ERROR, CheckStatus.FAILED,
                    f"episode {episode.episode_id}: no video window recorded for {modality}",
                    episode_id=episode.episode_id, modality=modality,
                    observed="absent", expected="from/to timestamp",
                )
                continue
            try:
                path = ctx.blob_store.get(blob_id)
            except KeyError:
                continue  # reported by BlobPresenceCheck; do not double-count

            start, end = float(win["from_timestamp"]), float(win["to_timestamp"])
            try:
                with VideoReader(path, cache_frames=0) as reader:
                    info = reader.info
                    if end > info.duration_s + 1.0 / max(ctx.fps, 1.0):
                        yield ctx.record(
                            self.name, CheckScope.EPISODE, Severity.ERROR, CheckStatus.FAILED,
                            f"episode {episode.episode_id}: window ends at {end:.2f}s "
                            f"but {modality} is only {info.duration_s:.2f}s long",
                            episode_id=episode.episode_id, modality=modality,
                            observed=f"{end:.3f}s", expected=f"<= {info.duration_s:.3f}s",
                        )
                        continue
                    if self.deep:
                        decoded = reader.count_decodable(start, end)
                        if decoded < frames.num_rows:
                            yield ctx.record(
                                self.name, CheckScope.EPISODE, Severity.ERROR, CheckStatus.FAILED,
                                f"episode {episode.episode_id}: {modality} yielded "
                                f"{decoded} frames for {frames.num_rows} rows",
                                episode_id=episode.episode_id, modality=modality,
                                observed=decoded, expected=frames.num_rows,
                            )
                    else:
                        # Probe both ends: a truncated or corrupt tail is the
                        # common failure and is invisible from the header alone.
                        eps = 1.0 / max(ctx.fps, 1.0)
                        for probe_ts in (start, max(start, end - eps)):
                            if reader.read_at(probe_ts) is None:
                                yield ctx.record(
                                    self.name, CheckScope.EPISODE, Severity.ERROR,
                                    CheckStatus.FAILED,
                                    f"episode {episode.episode_id}: {modality} not "
                                    f"decodable at {probe_ts:.3f}s",
                                    episode_id=episode.episode_id, modality=modality,
                                    observed="decode returned nothing",
                                    expected="an RGB frame",
                                )
                                break
            except Exception as exc:
                yield ctx.record(
                    self.name, CheckScope.EPISODE, Severity.ERROR, CheckStatus.FAILED,
                    f"episode {episode.episode_id}: {modality} failed to open ({type(exc).__name__})",
                    episode_id=episode.episode_id, modality=modality,
                    observed=str(exc)[:200], expected="readable container",
                )


class EpisodeReferenceCheck(DatasetCheck):
    """Episode ranges must tile the frame table without gaps or overlaps."""

    name = "structural.episode_references"
    severity = Severity.ERROR
    description = "Episode frame ranges are consistent, non-overlapping and cover the table."

    def check_dataset(self, ctx: CheckContext) -> Iterator[QCRecord]:
        """Confirm episode ranges tile the frame table without gaps or overlap."""
        eps = sorted(ctx.episodes, key=lambda e: e.frame_start)
        for a, b in zip(eps, eps[1:]):
            if b.frame_start < a.frame_end:
                yield ctx.record(
                    self.name, CheckScope.DATASET, Severity.ERROR, CheckStatus.FAILED,
                    f"episodes {a.episode_id} and {b.episode_id} overlap in frame range",
                    episode_id=b.episode_id,
                    observed=f"[{b.frame_start},{b.frame_end}) vs [{a.frame_start},{a.frame_end})",
                    expected="disjoint ranges",
                )

        total_declared = sum(e.num_frames for e in ctx.episodes)
        actual = ctx.table_store.count_rows("frames") if ctx.table_store.exists("frames") else 0
        if total_declared != actual:
            yield ctx.record(
                self.name, CheckScope.DATASET, Severity.ERROR, CheckStatus.FAILED,
                f"episode ranges account for {total_declared} frames; table holds {actual}",
                observed=actual, expected=total_declared,
            )

        ids = [e.episode_id for e in ctx.episodes]
        if len(ids) != len(set(ids)):
            yield ctx.record(
                self.name, CheckScope.DATASET, Severity.CRITICAL, CheckStatus.FAILED,
                "duplicate episode ids in the episode table",
                observed=len(ids), expected=len(set(ids)),
            )


STRUCTURAL_CHECKS = [
    FrameTableSchemaCheck(),
    BlobPresenceCheck(),
    EpisodeReferenceCheck(),
    EpisodeFrameCountCheck(),
    FrameIndexContinuityCheck(),
    VideoIntegrityCheck(),
]
