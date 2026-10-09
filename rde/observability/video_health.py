"""Video health without decoding petabytes.

The engine's own measurement makes the case: SO-100 is **470 MB on disk and
18.1 GB decoded** — 38x. At the design planning scale of 3–10 PB/year, a
monitoring strategy that decodes everything costs more than the training it
supports, and would do so every time anyone asked a question.

So video health is tiered, and each tier is a deliberate trade of cost for
what it can see:

``TIER 0``  no decode at all. Container headers only: codec, resolution, frame
            count, duration, declared fps, and whether the container opens.
            Microseconds per file, and it already catches a resolution change,
            a codec swap, a truncated file and a frame-count disagreement — the
            failures that come from a changed recording rig, which are the ones
            that hit whole cohorts at once.

``TIER 1``  sparse decode. A handful of frames spread across the file:
            brightness, contrast, sharpness, black-frame ratio, clipping, and
            frame-to-frame difference. Catches a lens cap, a dead sensor, a
            frozen stream, a blown exposure. Cost is set by the sample count,
            not by the file's length.

``TIER 2``  perceptual hashes over the sampled frames — a 64-bit fingerprint
            per frame. Catches duplicate and frozen content, and gives a cheap
            appearance-drift signal comparable across versions.

``TIER 3``  vision embeddings, semantic clustering, VLM inspection.
            **Not implemented here**, and deliberately: it needs a model, a GPU
            budget and a sampling policy of its own, and every question this
            project actually asks is answered by tiers 0–2. Documented so the
            boundary is a decision rather than an omission.

The sampling policy matters as much as the tiers. Frames are sampled at fixed
*fractions* of the file rather than at fixed timestamps, so a 12-second episode
and a 90-second one are compared at the same relative positions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..storage.video import VideoReader, probe

#: Default sample positions, as fractions of the clip. Includes both ends
#: because that is where truncation and trailing black frames live, and where
#: a fixed-stride sampler would never look.
DEFAULT_POSITIONS: tuple[float, ...] = (0.02, 0.25, 0.5, 0.75, 0.98)

#: Below this mean 0-255 level a frame counts as black. 8 rather than 0: a
#: lens cap and a powered-down sensor produce sensor noise, not true zero.
BLACK_LEVEL = 8.0

#: Above this fraction of pixels at 255, a frame is blown out.
CLIP_FRACTION = 0.05


@dataclass
class Tier0Report:
    """Container facts, obtained without decoding a single frame."""

    blob_id: str
    modality: str
    ok: bool
    codec: str = ""
    width: int = 0
    height: int = 0
    num_frames: int = 0
    duration_s: float = 0.0
    fps: float = 0.0
    file_bytes: int = 0
    #: What a full decode would have cost. The number that justifies the tiers.
    raw_rgb_bytes: int = 0
    error: str = ""

    @property
    def decode_amplification(self) -> float:
        """Decoded size divided by stored size."""
        return self.raw_rgb_bytes / self.file_bytes if self.file_bytes else 0.0


@dataclass
class Tier1Report:
    """Appearance statistics from a handful of decoded frames."""

    blob_id: str
    modality: str
    frames_sampled: int = 0
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    contrast_mean: float = 0.0
    sharpness_mean: float = 0.0
    black_frame_ratio: float = 0.0
    clipped_frame_ratio: float = 0.0
    #: Mean absolute difference between consecutive *samples*. Near zero means
    #: the stream is not changing -- a frozen camera, or a robot that never
    #: moved. Which of those it is takes the state trajectory to disambiguate.
    interframe_difference: float = 0.0
    decode_failures: int = 0
    #: Tier 2, computed from the same decoded frames because they are already
    #: in memory and throwing them away to re-decode later would be absurd.
    phashes: list[str] = field(default_factory=list)

    @property
    def frozen(self) -> bool:
        """Whether consecutive samples are effectively identical."""
        return self.frames_sampled >= 2 and self.interframe_difference < 0.5

    @property
    def duplicate_samples(self) -> int:
        """Sampled frames sharing a perceptual hash with an earlier one."""
        return len(self.phashes) - len(set(self.phashes)) if self.phashes else 0


def tier0(path: Path, blob_id: str = "", modality: str = "") -> Tier0Report:
    """Container-level health. No frames are decoded.

    Cheap enough to run on every file on every pass, which is what makes it the
    right place to catch a changed recording rig: a resolution or codec change
    hits an entire cohort at once, and finding it here costs nothing.
    """
    try:
        info = probe(path)
    except Exception as exc:  # noqa: BLE001 - a bad container is a result
        return Tier0Report(
            blob_id=blob_id, modality=modality, ok=False,
            error=f"{type(exc).__name__}: {exc}",
            file_bytes=path.stat().st_size if path.exists() else 0,
        )
    return Tier0Report(
        blob_id=blob_id, modality=modality, ok=True,
        codec=info.codec, width=info.width, height=info.height,
        num_frames=info.num_frames, duration_s=info.duration_s, fps=info.fps,
        file_bytes=path.stat().st_size if path.exists() else 0,
        raw_rgb_bytes=info.raw_rgb_bytes,
    )


def _phash(frame: np.ndarray, size: int = 8) -> str:
    """64-bit difference hash of one frame.

    A dHash rather than a DCT pHash: it needs no transform, survives
    re-encoding, and answers the only question asked of it here — "is this
    frame the same picture as that one?".
    """
    grey = frame.mean(axis=2) if frame.ndim == 3 else frame.astype(float)
    h, w = grey.shape
    ys = np.linspace(0, h - 1, size).astype(int)
    xs = np.linspace(0, w - 1, size + 1).astype(int)
    small = grey[np.ix_(ys, xs)]
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def _frame_stats(frame: np.ndarray) -> tuple[float, float, float, bool, bool]:
    """(brightness, contrast, sharpness, is_black, is_clipped) for one frame."""
    grey = frame.mean(axis=2) if frame.ndim == 3 else frame.astype(float)
    brightness = float(grey.mean())
    contrast = float(grey.std())
    # Sharpness as the variance of a discrete Laplacian. A blurred or defocused
    # frame has little high-frequency content and a low value; computed with
    # numpy slicing so no imaging dependency is added for one statistic.
    lap = (
        -4.0 * grey[1:-1, 1:-1]
        + grey[:-2, 1:-1] + grey[2:, 1:-1]
        + grey[1:-1, :-2] + grey[1:-1, 2:]
    ) if grey.shape[0] > 2 and grey.shape[1] > 2 else np.zeros((1, 1))
    sharpness = float(lap.var())
    clipped = float((frame >= 254).mean()) > CLIP_FRACTION
    return brightness, contrast, sharpness, brightness < BLACK_LEVEL, clipped


def tier1(
    path: Path,
    blob_id: str = "",
    modality: str = "",
    window: tuple[float, float] | None = None,
    positions: Sequence[float] = DEFAULT_POSITIONS,
) -> Tier1Report:
    """Sparse-decode appearance health over a few frames.

    ``window`` restricts sampling to one episode's slice of a concatenated
    per-camera video, which is how LeRobot v3.0 stores them -- without it, an
    episode's health would be computed from whatever else shares the file.

    Cost is ``len(positions)`` decodes regardless of clip length, so a 90-second
    episode costs the same as a 12-second one. That is the property that makes
    this affordable at petabyte scale, and the reason positions are fractions.
    """
    report = Tier1Report(blob_id=blob_id, modality=modality)
    try:
        info = probe(path)
    except Exception:  # noqa: BLE001
        report.decode_failures = len(positions)
        return report

    start, end = window if window else (0.0, info.duration_s)
    span = max(end - start, 0.0)
    if span <= 0:
        return report

    brightness, contrast, sharpness = [], [], []
    blacks = clipped = 0
    previous: np.ndarray | None = None
    diffs: list[float] = []

    # cache_frames=0: these are one-shot reads at scattered positions, so an
    # LRU would hold frames that will never be asked for again.
    with VideoReader(path, cache_frames=0) as reader:
        for frac in positions:
            ts = start + frac * span
            try:
                frame = reader.read_at(ts)
            except Exception:  # noqa: BLE001
                frame = None
            if frame is None:
                report.decode_failures += 1
                continue
            report.frames_sampled += 1
            b, c, s, is_black, is_clipped = _frame_stats(frame)
            brightness.append(b)
            contrast.append(c)
            sharpness.append(s)
            blacks += int(is_black)
            clipped += int(is_clipped)
            report.phashes.append(_phash(frame))
            if previous is not None and previous.shape == frame.shape:
                diffs.append(
                    float(np.abs(
                        previous.astype(np.int16) - frame.astype(np.int16)
                    ).mean())
                )
            previous = frame

    n = max(report.frames_sampled, 1)
    report.brightness_mean = float(np.mean(brightness)) if brightness else 0.0
    report.brightness_std = float(np.std(brightness)) if brightness else 0.0
    report.contrast_mean = float(np.mean(contrast)) if contrast else 0.0
    report.sharpness_mean = float(np.mean(sharpness)) if sharpness else 0.0
    report.black_frame_ratio = blacks / n
    report.clipped_frame_ratio = clipped / n
    report.interframe_difference = float(np.mean(diffs)) if diffs else 0.0
    return report


def cohort_summary(reports: Sequence[Tier0Report]) -> dict[str, Any]:
    """Aggregate Tier 0 across a version, surfacing rig inconsistency.

    The question this answers is not "is this file fine" but "do all these files
    agree" -- a single camera swapped to a different resolution mid-collection
    produces a dataset where every file is individually valid and the set is
    not.
    """
    ok = [r for r in reports if r.ok]
    resolutions = sorted({f"{r.width}x{r.height}" for r in ok})
    codecs = sorted({r.codec for r in ok})
    return {
        "files": len(reports),
        "unreadable": sum(1 for r in reports if not r.ok),
        "resolutions": resolutions,
        "codecs": codecs,
        "consistent_resolution": len(resolutions) <= 1,
        "consistent_codec": len(codecs) <= 1,
        "stored_bytes": sum(r.file_bytes for r in ok),
        "decoded_bytes": sum(r.raw_rgb_bytes for r in ok),
        "decode_amplification": round(
            sum(r.raw_rgb_bytes for r in ok) / max(sum(r.file_bytes for r in ok), 1), 1
        ),
    }


def emit(session: Any, reports: Sequence[Tier1Report], dataset_id: str) -> None:
    """Route Tier 1 output into the sensor-health metrics."""
    if session is None or not getattr(session, "enabled", False):
        return
    for r in reports:
        if r.decode_failures:
            session.inc("rde_data_decode_failures_total", float(r.decode_failures),
                        modality=r.modality, dataset_id=dataset_id)
        if r.frozen:
            session.inc("rde_sensor_stuck_intervals_total", 1.0,
                        modality=r.modality, dataset_id=dataset_id)
        if r.black_frame_ratio > 0:
            session.inc("rde_sensor_frame_dropout_total",
                        float(r.black_frame_ratio * max(r.frames_sampled, 1)),
                        modality=r.modality, dataset_id=dataset_id)


def as_dicts(reports: Sequence[Any]) -> list[dict[str, Any]]:
    """Dataclass reports -> JSON-safe dicts."""
    return [asdict(r) for r in reports]
