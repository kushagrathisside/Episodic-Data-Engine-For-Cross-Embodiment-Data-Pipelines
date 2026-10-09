"""Dataset-level validation: is the collection as a whole fit for training?

Episode-level checks catch broken episodes.  These catch broken *datasets* -- a
collection where every episode is individually valid but the set is unusable:
one task dominating 95% of frames, a camera present in only half the episodes, a
length distribution so skewed that batching is pathological.

These findings are usually WARNINGs.  They rarely mean "reject this data"; they
mean "know this before you train on it, and record that you knew".
"""

from __future__ import annotations

from collections import Counter
from typing import Iterator

import numpy as np

from ..schemas.modality import StoragePlane
from ..schemas.qc import CheckScope, CheckStatus, QCRecord, Severity
from .base import CheckContext, DatasetCheck


class EpisodeLengthDistributionCheck(DatasetCheck):
    """Report the length distribution and flag pathological spread."""

    name = "dataset.length_distribution"
    severity = Severity.INFO
    description = "Summarises episode length distribution; warns on extreme spread."

    def __init__(self, max_ratio: float = 10.0):
        self.max_ratio = max_ratio

    def check_dataset(self, ctx: CheckContext) -> Iterator[QCRecord]:
        """Summarise episode lengths; warn on extreme spread."""
        lengths = np.array([e.num_frames for e in ctx.episodes], dtype=np.float64)
        if lengths.size == 0:
            yield ctx.record(
                self.name, CheckScope.DATASET, Severity.CRITICAL, CheckStatus.FAILED,
                "dataset contains no episodes", observed=0, expected=">0",
            )
            return

        stats = {
            "count": int(lengths.size),
            "min": int(lengths.min()), "max": int(lengths.max()),
            "mean": round(float(lengths.mean()), 1),
            "median": float(np.median(lengths)),
            "p05": float(np.percentile(lengths, 5)),
            "p95": float(np.percentile(lengths, 95)),
            "std": round(float(lengths.std()), 1),
            "total_frames": int(lengths.sum()),
        }
        ctx.shared["length_distribution"] = stats

        ratio = float(lengths.max() / max(lengths.min(), 1))
        status = CheckStatus.WARNED if ratio > self.max_ratio else CheckStatus.PASSED
        severity = Severity.WARNING if status is CheckStatus.WARNED else Severity.INFO
        yield ctx.record(
            self.name, CheckScope.DATASET, severity, status,
            f"episode lengths: min={stats['min']} median={stats['median']:.0f} "
            f"max={stats['max']} (max/min={ratio:.1f}x)",
            observed=f"{ratio:.2f}x spread", expected=f"<= {self.max_ratio}x",
            **stats,
        )


class TaskDistributionCheck(DatasetCheck):
    """Flag task imbalance, which silently biases anything trained on the set."""

    name = "dataset.task_distribution"
    severity = Severity.WARNING
    description = "Reports per-task episode/frame share and warns on dominance."

    def __init__(self, max_share: float = 0.90):
        self.max_share = max_share

    def check_dataset(self, ctx: CheckContext) -> Iterator[QCRecord]:
        """Report per-task share; warn when one task dominates."""
        counts = Counter(e.task_id for e in ctx.episodes)
        frames = Counter()
        for e in ctx.episodes:
            frames[e.task_id] += e.num_frames
        total_eps = sum(counts.values()) or 1
        total_frames = sum(frames.values()) or 1

        descriptions = {t.task_id: t.description for t in ctx.manifest.tasks}
        dist = {
            str(tid): {
                "description": descriptions.get(tid, "<unknown>"),
                "episodes": n,
                "episode_share": round(n / total_eps, 4),
                "frames": frames[tid],
                "frame_share": round(frames[tid] / total_frames, 4),
            }
            for tid, n in counts.most_common()
        }
        ctx.shared["task_distribution"] = dist

        if not counts:
            return
        top_id, top_n = counts.most_common(1)[0]
        share = top_n / total_eps
        # A single-task dataset is a legitimate design, not an imbalance defect;
        # say so explicitly instead of raising a warning nobody can act on.
        if len(counts) == 1:
            yield ctx.record(
                self.name, CheckScope.DATASET, Severity.INFO, CheckStatus.PASSED,
                f"single-task dataset: {descriptions.get(top_id, top_id)!r} "
                f"({total_eps} episodes)",
                observed="1 task", expected="n/a", distribution=dist,
            )
        elif share > self.max_share:
            yield ctx.record(
                self.name, CheckScope.DATASET, Severity.WARNING, CheckStatus.WARNED,
                f"task {top_id} covers {share:.1%} of episodes across {len(counts)} tasks",
                observed=f"{share:.1%}", expected=f"<= {self.max_share:.0%}",
                distribution=dist,
            )
        else:
            yield ctx.record(
                self.name, CheckScope.DATASET, Severity.INFO, CheckStatus.PASSED,
                f"{len(counts)} tasks, largest share {share:.1%}",
                observed=f"{len(counts)} tasks", expected="n/a", distribution=dist,
            )


class ModalityAvailabilityCheck(DatasetCheck):
    """Every declared modality must actually be present in every episode.

    Partial modality coverage is a real and nasty condition: a dataset where the
    wrist camera cut out for 8 of 50 episodes will train, and will quietly teach
    the model that the wrist view is sometimes meaningless.
    """

    name = "dataset.modality_availability"
    severity = Severity.WARNING
    description = "Declared modalities are present across all episodes."

    def check_dataset(self, ctx: CheckContext) -> Iterator[QCRecord]:
        """Confirm every declared modality is present in every episode."""
        declared = {m.name for m in ctx.manifest.modalities}
        external = {m.name for m in ctx.manifest.modalities if m.plane is StoragePlane.EXTERNAL}
        total = len(ctx.episodes) or 1

        coverage: dict[str, float] = {}
        for name in sorted(declared):
            if name in external:
                present = sum(1 for e in ctx.episodes if name in (e.blob_ids or {}))
            else:
                present = sum(1 for e in ctx.episodes if name in (e.modalities or []))
            coverage[name] = round(present / total, 4)
            if present == 0:
                yield ctx.record(
                    self.name, CheckScope.DATASET, Severity.ERROR, CheckStatus.FAILED,
                    f"modality {name!r} is declared but present in no episode",
                    modality=name, observed="0%", expected="100%",
                )
            elif present < total:
                yield ctx.record(
                    self.name, CheckScope.DATASET, Severity.WARNING, CheckStatus.WARNED,
                    f"modality {name!r} present in {present}/{total} episodes",
                    modality=name, observed=f"{present}/{total}", expected=f"{total}/{total}",
                    missing_episodes=[
                        e.episode_id for e in ctx.episodes
                        if name not in ((e.blob_ids or {}) if name in external else (e.modalities or []))
                    ][:50],
                )
        ctx.shared["modality_coverage"] = coverage


class MissingDataCheck(DatasetCheck):
    """Null-rate per column across the frame table.

    Streams the table and accumulates counts only, so the cost is independent of
    dataset size.
    """

    name = "dataset.missing_data"
    severity = Severity.WARNING
    description = "Reports the fraction of null values per frame-table column."

    def __init__(self, max_null_ratio: float = 0.001):
        self.max_null_ratio = max_null_ratio

    def check_dataset(self, ctx: CheckContext) -> Iterator[QCRecord]:
        """Stream the table and report the null rate per column."""
        if not ctx.table_store.exists("frames"):
            return
        schema = ctx.table_store.schema("frames")
        nulls = {n: 0 for n in schema.names}
        total = 0
        for batch in ctx.table_store.iter_batches(
            "frames", batch_size=ctx.settings.batch_size
        ):
            total += batch.num_rows
            for i, name in enumerate(batch.schema.names):
                nulls[name] += batch.column(i).null_count

        if total == 0:
            return
        ratios = {k: round(v / total, 6) for k, v in nulls.items() if v}
        ctx.shared["null_ratios"] = ratios
        for name, ratio in ratios.items():
            if ratio > self.max_null_ratio:
                yield ctx.record(
                    self.name, CheckScope.DATASET, Severity.WARNING, CheckStatus.WARNED,
                    f"column {name!r} is {ratio:.3%} null ({nulls[name]}/{total})",
                    modality=name if name in {m.name for m in ctx.manifest.modalities} else None,
                    observed=f"{ratio:.4%}", expected=f"<= {self.max_null_ratio:.3%}",
                    null_count=nulls[name], total_rows=total,
                )
        if not ratios:
            yield ctx.record(
                self.name, CheckScope.DATASET, Severity.INFO, CheckStatus.PASSED,
                f"no nulls across {len(schema.names)} columns and {total} rows",
                observed="0 nulls", expected="0 nulls",
            )


class FrameCountConsistencyCheck(DatasetCheck):
    """Manifest totals must match what is physically stored."""

    name = "dataset.frame_count_consistency"
    severity = Severity.ERROR
    description = "Manifest episode/frame totals agree with the stored tables."

    def check_dataset(self, ctx: CheckContext) -> Iterator[QCRecord]:
        """Cross-check manifest totals against what is physically stored."""
        if not ctx.table_store.exists("frames"):
            return
        actual_frames = ctx.table_store.count_rows("frames")
        if actual_frames != ctx.manifest.total_frames:
            yield ctx.record(
                self.name, CheckScope.DATASET, Severity.ERROR, CheckStatus.FAILED,
                f"manifest declares {ctx.manifest.total_frames} frames; "
                f"table holds {actual_frames}",
                observed=actual_frames, expected=ctx.manifest.total_frames,
            )
        if len(ctx.episodes) != ctx.manifest.total_episodes:
            yield ctx.record(
                self.name, CheckScope.DATASET, Severity.ERROR, CheckStatus.FAILED,
                f"manifest declares {ctx.manifest.total_episodes} episodes; "
                f"catalog holds {len(ctx.episodes)}",
                observed=len(ctx.episodes), expected=ctx.manifest.total_episodes,
            )


DATASET_CHECKS = [
    EpisodeLengthDistributionCheck(),
    TaskDistributionCheck(),
    ModalityAvailabilityCheck(),
    MissingDataCheck(),
    FrameCountConsistencyCheck(),
]
