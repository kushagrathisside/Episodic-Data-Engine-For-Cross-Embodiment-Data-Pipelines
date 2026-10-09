"""The standard operator library.

Each operator is small, independently testable, and records enough in its
parameters to be reproduced.  Grouped by what they touch:

*Columns*   :class:`NormalizeVector`, :class:`TimestampNormalizer`,
            :class:`SelectFeatures`
*Rows*      :class:`DropNonFiniteFrames`, :class:`SubsampleFrames`
*Episodes*  :class:`QualityFilter`, :class:`LengthFilter`, :class:`TaskFilter`,
            :class:`EpisodeSubsetFilter`
*Deferred*  :class:`ResizeImages`, :class:`NormalizeImages`
*Splits*    :func:`assign_splits`
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..schemas.core import DatasetManifest, Episode, QualityStatus
from ..schemas.modality import StoragePlane
from .base import EpisodeFilter, FrameFilter, FrameTransform, TransformContext


def _vector_column(batch: pa.RecordBatch, name: str) -> np.ndarray | None:
    """Fixed-size-list column -> (n, width) float64 array.

    Width is read from the field's declared ``list_size`` rather than
    inferred from ``flat.size // batch.num_rows``: on an empty batch the
    latter is always 0, which silently discards the column's real width and
    corrupts schema derivation (``TransformPipeline._output_schema`` probes
    operators with an empty batch to learn the output dtype/shape).
    """
    if name not in batch.schema.names:
        return None
    i = batch.schema.get_field_index(name)
    col = batch.column(i)
    width = batch.schema.field(i).type.list_size
    flat = np.asarray(col.flatten(), dtype=np.float64)
    return flat.reshape(batch.num_rows, width)


def _replace_column(batch: pa.RecordBatch, name: str, values: np.ndarray) -> pa.RecordBatch:
    """Write a (n, width) array back into a fixed-size-list column."""
    i = batch.schema.get_field_index(name)
    field = batch.schema.field(i)
    width = values.shape[1]
    arr = pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1).astype(np.float32)), width
    )
    cols = list(batch.columns)
    cols[i] = arr
    schema = batch.schema.set(i, pa.field(field.name, arr.type))
    return pa.RecordBatch.from_arrays(cols, schema=schema)


# -- column operators ------------------------------------------------------
class NormalizeVector(FrameTransform):
    """Normalise a numeric modality using dataset statistics.

    Two schemes, and the choice matters for robot data:

    ``mean_std``   zero-mean unit-variance.  The default for joint states.
    ``min_max``    scale to [-1, 1] using observed extremes.  Preferred when a
                   policy head expects bounded outputs.

    Zero-variance dimensions are passed through unchanged rather than divided by
    epsilon: a joint that never moves should stay constant, not explode into
    numerical noise that looks like signal.
    """

    needs_fit = True

    def __init__(self, modality: str, scheme: str = "mean_std", eps: float = 1e-8):
        if scheme not in ("mean_std", "min_max"):
            raise ValueError(f"unknown scheme: {scheme!r}")
        self.modality = modality
        self.scheme = scheme
        self.eps = eps
        self.name = f"normalize.{modality}"
        self.description = f"Normalise {modality} with {scheme} statistics."
        self._stats: dict[str, list[float]] = {}

    def parameters(self) -> dict[str, Any]:
        """Scheme, epsilon and the fitted statistics needed to reproduce this."""
        return {"modality": self.modality, "scheme": self.scheme, "eps": self.eps,
                "statistics": self._stats}

    def fit(self, ctx: TransformContext, batches: Iterator[pa.RecordBatch]) -> None:
        """Streaming accumulation -- constant memory regardless of dataset size."""
        n = 0
        total = None
        total_sq = None
        vmin = vmax = None
        for batch in batches:
            v = _vector_column(batch, self.modality)
            if v is None or v.size == 0:
                continue
            finite = np.isfinite(v)
            v = np.where(finite, v, 0.0)
            counts = finite.sum(axis=0)
            if total is None:
                total = np.zeros(v.shape[1])
                total_sq = np.zeros(v.shape[1])
                vmin = np.full(v.shape[1], np.inf)
                vmax = np.full(v.shape[1], -np.inf)
                n = np.zeros(v.shape[1])
            total += v.sum(axis=0)
            total_sq += (v ** 2).sum(axis=0)
            n += counts
            masked = np.where(finite, v, np.nan)
            vmin = np.fmin(vmin, np.nanmin(masked, axis=0))
            vmax = np.fmax(vmax, np.nanmax(masked, axis=0))
        if total is None:
            return
        n = np.maximum(n, 1)
        mean = total / n
        var = np.maximum(total_sq / n - mean ** 2, 0.0)
        self._stats = {
            "mean": mean.tolist(), "std": np.sqrt(var).tolist(),
            "min": vmin.tolist(), "max": vmax.tolist(), "count": int(n[0]),
        }
        ctx.fitted[self.name] = self._stats

    def apply_batch(self, batch: pa.RecordBatch, ctx: TransformContext) -> pa.RecordBatch:
        """Apply the fitted normalization; constant dimensions pass through."""
        v = _vector_column(batch, self.modality)
        if v is None or v.size == 0 or not self._stats:
            return batch
        if self.scheme == "mean_std":
            mean = np.asarray(self._stats["mean"])
            std = np.asarray(self._stats["std"])
            scale = np.where(std > self.eps, std, 1.0)  # constant dims pass through
            out = (v - mean) / scale
        else:
            lo = np.asarray(self._stats["min"])
            hi = np.asarray(self._stats["max"])
            span = hi - lo
            scale = np.where(span > self.eps, span, 1.0)
            out = 2.0 * (v - lo) / scale - 1.0
            out = np.where(span > self.eps, out, 0.0)
        return _replace_column(batch, self.modality, out)

    def update_manifest(self, manifest: DatasetManifest, ctx: TransformContext) -> None:
        """Record the scheme and statistics on the modality spec."""
        m = manifest.modality(self.modality)
        if m is not None:
            m.attributes["normalization"] = {"scheme": self.scheme, **self._stats}


class TimestampNormalizer(FrameTransform):
    """Rebase each episode's timestamps to start at zero on a uniform grid.

    Idempotent for already-clean data.  It exists because ingestion must not
    assume the source rebased anything, and downstream windowing code is entitled
    to assume ``timestamp_s == frame_index / fps``.
    """

    def __init__(self, fps: float | None = None):
        self.fps = fps
        self.name = "timestamp.normalize"
        self.description = "Rebase episode timestamps onto a uniform 1/fps grid from zero."

    def parameters(self) -> dict[str, Any]:
        """Target rate, or None to inherit the dataset's."""
        return {"fps": self.fps}

    def apply_batch(self, batch: pa.RecordBatch, ctx: TransformContext) -> pa.RecordBatch:
        """Rewrite timestamps as frame_index / fps."""
        if "frame_index" not in batch.schema.names or "timestamp_s" not in batch.schema.names:
            return batch
        fps = self.fps or ctx.fps
        fi = np.asarray(batch.column(batch.schema.get_field_index("frame_index")))
        new_ts = fi.astype(np.float64) / fps
        i = batch.schema.get_field_index("timestamp_s")
        cols = list(batch.columns)
        cols[i] = pa.array(new_ts, type=pa.float64())
        return pa.RecordBatch.from_arrays(cols, schema=batch.schema)


class SelectFeatures(FrameTransform):
    """Keep only the named modalities, dropping the rest from the frame table.

    Real use: a state-only policy does not need two camera reference columns, and
    dropping them makes the training export smaller and the contract explicit.
    """

    def __init__(self, keep: Sequence[str]):
        self.keep = list(keep)
        self.name = "features.select"
        self.description = f"Retain only modalities: {', '.join(self.keep)}."

    def parameters(self) -> dict[str, Any]:
        """The modalities being retained."""
        return {"keep": self.keep}

    def apply_batch(self, batch: pa.RecordBatch, ctx: TransformContext) -> pa.RecordBatch:
        """Drop every column not carrying a retained modality."""
        canonical = {"frame_id", "episode_id", "frame_index", "timestamp_s", "task_id"}
        keep_cols = []
        for name in batch.schema.names:
            base = name.split("__")[0]
            if name in canonical or base in self.keep:
                keep_cols.append(name)
        return batch.select(keep_cols)

    def update_manifest(self, manifest: DatasetManifest, ctx: TransformContext) -> None:
        """Prune the modality registry to match."""
        manifest.modalities = [m for m in manifest.modalities if m.name in self.keep]


# -- row operators ---------------------------------------------------------
class DropNonFiniteFrames(FrameFilter):
    """Remove rows containing NaN/Inf in any numeric modality."""

    def __init__(self, modalities: Sequence[str] | None = None):
        self.modalities = list(modalities) if modalities else None
        self.name = "frames.drop_non_finite"
        self.description = "Drop frames whose numeric modalities contain NaN or Inf."

    def parameters(self) -> dict[str, Any]:
        """Modalities inspected, or None for every inline one."""
        return {"modalities": self.modalities}

    def mask(self, batch: pa.RecordBatch, ctx: TransformContext) -> pa.Array:
        """Keep rows whose numeric modalities are all finite."""
        names = self.modalities or [
            m.name for m in ctx.manifest.modalities if m.plane is StoragePlane.INLINE
        ]
        keep = np.ones(batch.num_rows, dtype=bool)
        for n in names:
            v = _vector_column(batch, n)
            if v is None or v.size == 0:
                continue
            keep &= np.isfinite(v).all(axis=1)
        return pa.array(keep)


class SubsampleFrames(FrameFilter):
    """Keep every Nth frame -- a temporal downsample.

    Note this changes the effective control rate; the manifest fps is updated so
    downstream consumers are not silently misled about the sampling frequency.
    """

    def __init__(self, stride: int = 2, phase: int = 0):
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride, self.phase = stride, phase
        self.name = "frames.subsample"
        self.description = f"Keep every {stride}th frame (phase {phase})."

    def parameters(self) -> dict[str, Any]:
        """Stride and phase of the temporal downsample."""
        return {"stride": self.stride, "phase": self.phase}

    def mask(self, batch: pa.RecordBatch, ctx: TransformContext) -> pa.Array:
        """Keep every Nth frame within each episode."""
        fi = np.asarray(batch.column(batch.schema.get_field_index("frame_index")))
        return pa.array(((fi - self.phase) % self.stride) == 0)

    def update_manifest(self, manifest: DatasetManifest, ctx: TransformContext) -> None:
        """Divide the declared fps, so consumers are not misled."""
        if manifest.fps:
            manifest.fps = manifest.fps / self.stride
        for m in manifest.modalities:
            if m.fps:
                m.fps = m.fps / self.stride


# -- episode operators -----------------------------------------------------
class QualityFilter(EpisodeFilter):
    """Keep only episodes whose QC status is acceptable.

    This is the operator that turns ``validated/vN`` into ``clean/vN``.  It drops
    nothing from disk -- rejected episodes remain in the source version with
    their findings intact; the clean version simply does not reference them.
    """

    def __init__(self, allowed: Sequence[QualityStatus] | None = None):
        self.allowed = list(allowed or [QualityStatus.PASSED, QualityStatus.WARNING])
        self.name = "episodes.quality_filter"
        self.description = (
            f"Keep episodes with status in {[s.value for s in self.allowed]}."
        )

    def parameters(self) -> dict[str, Any]:
        """The quality statuses being admitted."""
        return {"allowed": [s.value for s in self.allowed]}

    def keep(self, episodes: Sequence[Episode], ctx: TransformContext) -> list[int]:
        """Episode ids whose QC status is acceptable."""
        return [e.episode_id for e in episodes if e.quality_status in self.allowed]


class LengthFilter(EpisodeFilter):
    """Keep episodes whose frame count lies in a range."""

    def __init__(self, min_frames: int | None = None, max_frames: int | None = None):
        self.min_frames, self.max_frames = min_frames, max_frames
        self.name = "episodes.length_filter"
        self.description = f"Keep episodes with {min_frames} <= frames <= {max_frames}."

    def parameters(self) -> dict[str, Any]:
        """Inclusive frame-count bounds."""
        return {"min_frames": self.min_frames, "max_frames": self.max_frames}

    def keep(self, episodes: Sequence[Episode], ctx: TransformContext) -> list[int]:
        """Episode ids inside the length range."""
        out = []
        for e in episodes:
            if self.min_frames is not None and e.num_frames < self.min_frames:
                continue
            if self.max_frames is not None and e.num_frames > self.max_frames:
                continue
            out.append(e.episode_id)
        return out


class EpisodeSubsetFilter(EpisodeFilter):
    """Keep exactly an explicit set of episode ids.

    The general-purpose selection mechanism for curated dataset variants --
    a reduced-data or deliberately-imbalanced training set -- whose selection
    rule does not reduce to quality, length or task (e.g. "these 24 episodes,
    chosen because a dataset profile showed this task's short-trajectory bucket
    was under-represented"). The rule that produced the id list is the caller's
    to justify and belongs in ``reason``, which is recorded here so the
    resulting version's lineage states *why* these episodes and not others --
    not just that some were excluded.
    """

    def __init__(self, episode_ids: Sequence[int], reason: str = ""):
        self.episode_ids = sorted({int(e) for e in episode_ids})
        self.reason = reason
        self.name = "episodes.subset_filter"
        self.description = f"Keep an explicit subset of {len(self.episode_ids)} episodes."

    def parameters(self) -> dict[str, Any]:
        """The explicit id list, its size and the stated selection rationale."""
        return {"episode_ids": self.episode_ids, "count": len(self.episode_ids), "reason": self.reason}

    def keep(self, episodes: Sequence[Episode], ctx: TransformContext) -> list[int]:
        """Episode ids present in both the input version and the requested subset."""
        wanted = set(self.episode_ids)
        return [e.episode_id for e in episodes if e.episode_id in wanted]


class TaskFilter(EpisodeFilter):
    """Keep episodes belonging to specific tasks."""

    def __init__(self, task_ids: Sequence[int]):
        self.task_ids = list(task_ids)
        self.name = "episodes.task_filter"
        self.description = f"Keep episodes for tasks {self.task_ids}."

    def parameters(self) -> dict[str, Any]:
        """The task ids being retained."""
        return {"task_ids": self.task_ids}

    def keep(self, episodes: Sequence[Episode], ctx: TransformContext) -> list[int]:
        """Episode ids belonging to the selected tasks."""
        return [e.episode_id for e in episodes if e.task_id in self.task_ids]


# -- deferred (read-time) operators ----------------------------------------
class ResizeImages(FrameTransform):
    """Declare a read-time image resize.

    Deliberately deferred.  Materialising a resize means re-encoding every video:
    minutes of CPU and hundreds of MB written, to save a resize that costs
    microseconds per frame at load time.  Worse, it bakes one resolution into
    storage, so a second model wanting a different input size needs a second
    copy of the whole dataset.

    So the operator records its intent in the manifest and
    :mod:`rde.model_adapters` applies it during decode.  Lineage is unaffected --
    the transformation is recorded identically either way.
    """

    deferred = True

    def __init__(self, height: int, width: int, modalities: Sequence[str] | None = None,
                 interpolation: str = "bilinear"):
        self.height, self.width = height, width
        self.modalities = list(modalities) if modalities else None
        self.interpolation = interpolation
        self.name = "images.resize"
        self.description = f"Resize image modalities to {height}x{width} at read time."

    def parameters(self) -> dict[str, Any]:
        """Target size, interpolation and the modalities affected."""
        return {"height": self.height, "width": self.width,
                "modalities": self.modalities, "interpolation": self.interpolation}

    def apply_batch(self, batch: pa.RecordBatch, ctx: TransformContext) -> pa.RecordBatch:
        """No-op: deferred to read time, so no stored data changes."""
        return batch  # deferred: no stored data changes

    def update_manifest(self, manifest: DatasetManifest, ctx: TransformContext) -> None:
        """Record the resize as a read transform and update the shape."""
        targets = self.modalities or [m.name for m in manifest.modalities if m.is_external]
        for m in manifest.modalities:
            if m.name in targets:
                m.attributes.setdefault("read_transforms", []).append(
                    {"op": "resize", "height": self.height, "width": self.width,
                     "interpolation": self.interpolation}
                )
                if len(m.shape) == 3:
                    m.shape = (self.height, self.width, m.shape[2])


class NormalizeImages(FrameTransform):
    """Declare a read-time image scaling/normalisation."""

    deferred = True

    def __init__(self, mean: Sequence[float] = (0.0, 0.0, 0.0),
                 std: Sequence[float] = (255.0, 255.0, 255.0),
                 modalities: Sequence[str] | None = None):
        self.mean, self.std = list(mean), list(std)
        self.modalities = list(modalities) if modalities else None
        self.name = "images.normalize"
        self.description = "Scale image pixels at read time."

    def parameters(self) -> dict[str, Any]:
        """Per-channel mean and scale, and the modalities affected."""
        return {"mean": self.mean, "std": self.std, "modalities": self.modalities}

    def apply_batch(self, batch: pa.RecordBatch, ctx: TransformContext) -> pa.RecordBatch:
        """No-op: deferred to read time."""
        return batch

    def update_manifest(self, manifest: DatasetManifest, ctx: TransformContext) -> None:
        """Record the scaling as a read transform."""
        targets = self.modalities or [m.name for m in manifest.modalities if m.is_external]
        for m in manifest.modalities:
            if m.name in targets:
                m.attributes.setdefault("read_transforms", []).append(
                    {"op": "normalize", "mean": self.mean, "std": self.std}
                )


# -- splitting -------------------------------------------------------------
def assign_splits(
    episode_ids: Sequence[int],
    fractions: dict[str, float],
    seed: int = 0,
    group_by: dict[int, Any] | None = None,
    mode: str = "quantile",
) -> dict[int, str]:
    """Deterministically assign episodes to named splits.

    Splitting is done **by episode, never by frame**.  Frames within an episode
    are massively autocorrelated -- consecutive frames differ by one control
    step -- so a frame-level split puts near-duplicates on both sides and
    produces a validation score that measures memorisation, not generalisation.
    This is the single most common way a robotics benchmark silently lies.

    Assignment derives from a hash of the episode id rather than a shuffle, so it
    is reproducible from the seed alone and does not depend on iteration order.

    Two modes, trading exactness against stability:

    ``quantile`` (default)
        Rank episodes by hash and cut at the requested fractions.  Gives *exact*
        split sizes, which matters at POC scale -- with 48 episodes, independent
        per-episode assignment routinely lands 42/3/3 instead of 38/5/5, and a
        3-episode validation set is too small to mean anything.
    ``independent``
        Assign each episode by its own hash, independent of the others.  Sizes
        vary, but an episode's split never changes as the dataset grows, so
        results stay comparable across dataset versions.  Prefer this once the
        dataset is large enough for the law of large numbers to do the work.

    ``group_by`` keeps correlated episodes (e.g. same recording session) on
    one side of the split, whichever mode is used. Under ``quantile`` this
    means groups, not raw episode counts, are what gets cut at the target
    fractions -- a group larger than a split's remaining budget pushes that
    split over its target, so split sizes are only approximate when
    ``group_by`` is set.
    """
    total = sum(fractions.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    names = list(fractions)
    edges = np.cumsum([fractions[n] for n in names])

    def _u(eid: int) -> float:
        key = f"{seed}:{group_by[eid] if group_by else eid}"
        h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        return (h % 10_000_000) / 10_000_000

    if mode == "independent":
        return {
            eid: (names[int(np.searchsorted(edges, u, side="right"))] if (u := _u(eid)) < 1.0
                  else names[-1])
            for eid in episode_ids
        }
    if mode != "quantile":
        raise ValueError(f"unknown split mode: {mode!r}")

    if group_by:
        # Rank whole groups, not individual episodes, and cut only at group
        # boundaries -- a raw episode-count cut can otherwise fall inside a
        # tied-hash block of same-group episodes, splitting the group across
        # train/val/test and defeating the reason group_by exists. This
        # trades exact split sizes for the leakage guarantee: a group larger
        # than one split's remaining budget pushes that split over target.
        groups: dict[Any, list[int]] = {}
        for eid in episode_ids:
            groups.setdefault(group_by[eid], []).append(eid)
        ranked_groups = sorted(groups, key=_u)
        n = len(episode_ids)
        cuts = [int(round(e * n)) for e in edges]
        out = {}
        name_idx = 0
        cum = 0
        for key in ranked_groups:
            for eid in groups[key]:
                out[eid] = names[name_idx]
            cum += len(groups[key])
            while name_idx < len(names) - 1 and cum >= cuts[name_idx]:
                name_idx += 1
        return out

    ranked = sorted(episode_ids, key=_u)
    n = len(ranked)
    cuts = [int(round(e * n)) for e in edges]
    out: dict[int, str] = {}
    start = 0
    for name, stop in zip(names, cuts):
        for eid in ranked[start:stop]:
            out[eid] = name
        start = stop
    for eid in ranked[start:]:  # rounding remainder goes to the last split
        out[eid] = names[-1]
    return out
