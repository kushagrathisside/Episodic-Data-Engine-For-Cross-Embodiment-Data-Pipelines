"""Deployment feedback loop -- the data flywheel.

Simulates the cycle the platform exists to serve::

    inference -> failure / low-confidence example -> failure record
        -> episode selection -> data review -> new dataset version -> retraining

The engineering claim being demonstrated is narrow and specific: **evaluation
output is an input to the data pipeline, not a terminal report.**  A failure
record is a first-class artifact with the same provenance discipline as a QC
finding, episode selection is a query over it, and the resulting dataset is a
normal version with normal lineage -- so "which episodes were added because the
model failed on them?" is answerable by the same lineage traversal as everything
else.

What is *not* simulated: an actual robot.  There is no rollout, no environment,
no success signal from the world.  Failures here are prediction errors against
held-out demonstrations, which is a proxy.  The loop's mechanics are real; the
failure signal's origin is not, and the docs say so.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np

from ..catalog import Catalog
from ..config import Settings, get_settings
from ..evaluation.evaluator import EvaluationResult
from ..schemas.core import QualityStatus


class FailureMode(str, Enum):
    """Why a frame was flagged.  Different modes warrant different responses."""

    HIGH_ERROR = "high_error"            # model was confidently wrong
    LOW_CONFIDENCE = "low_confidence"    # model knew it did not know
    BOTH = "both"                        # wrong and aware -- clearest signal


@dataclass
class FailureRecord:
    """One flagged inference, with everything needed to act on it."""

    frame_id: int
    episode_id: int
    dataset_id: str
    version: str
    model_name: str
    mode: FailureMode
    error: float
    confidence: float
    detected_at: _dt.datetime = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc)
    )

    def as_row(self) -> dict[str, Any]:
        """Flat dict for the failures Parquet table."""
        return {
            "frame_id": self.frame_id, "episode_id": self.episode_id,
            "dataset_id": self.dataset_id, "version": self.version,
            "model_name": self.model_name, "mode": self.mode.value,
            "error": self.error, "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class EpisodeReview:
    """An episode nominated for review, with the evidence that nominated it."""

    episode_id: int
    failure_frames: int
    total_frames_evaluated: int
    mean_error: float
    max_error: float
    mean_confidence: float
    modes: dict[str, int]
    priority: float
    recommendation: str

    @property
    def failure_rate(self) -> float:
        """Flagged frames as a fraction of frames evaluated for this episode."""
        return self.failure_frames / max(self.total_frames_evaluated, 1)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view, with the failure rate materialised."""
        return {**self.__dict__, "failure_rate": round(self.failure_rate, 4)}


@dataclass
class FeedbackReport:
    """Flagged frames, nominated episodes and the thresholds that selected them."""
    dataset_id: str
    version: str
    model_name: str
    failures: list[FailureRecord]
    reviews: list[EpisodeReview]
    thresholds: dict[str, float]
    artifacts: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Failure counts by mode, flagged episodes and the thresholds used."""
        by_mode: dict[str, int] = defaultdict(int)
        for f in self.failures:
            by_mode[f.mode.value] += 1
        return {
            "dataset": f"{self.dataset_id}@{self.version}",
            "model": self.model_name,
            "failure_frames": len(self.failures),
            "by_mode": dict(by_mode),
            "episodes_flagged": len(self.reviews),
            "thresholds": self.thresholds,
            "top_episodes": [r.episode_id for r in self.reviews[:5]],
        }


class FeedbackLoop:
    """Turns evaluation outcomes into failure records and dataset actions."""

    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None):
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.paths.catalog_db)

    # -- detection ----------------------------------------------------------
    def detect(
        self,
        result: EvaluationResult,
        error_percentile: float = 90.0,
        confidence_threshold: float | None = None,
        min_failures_per_episode: int = 3,
        persist: bool = True,
    ) -> FeedbackReport:
        """Flag failures and nominate episodes for review.

        The error threshold is a *percentile of this run*, not an absolute
        number: what counts as a large error depends on the action units, the
        normalisation, and the model.  A fixed threshold would either flag
        everything or nothing the moment either changes.

        Confidence is used only when it carries information.  A model that
        reports a constant (``SmolVLAPolicy`` and ``MeanActionPolicy`` do) would
        otherwise put every frame at or below its own percentile and flag the
        whole split as low-confidence.  In that case detection falls back to
        error alone and ``thresholds["confidence_used"]`` records why, unless
        the caller passes an explicit ``confidence_threshold``.
        """
        if not result.outcomes:
            return FeedbackReport(
                dataset_id=result.dataset_id, version=result.version,
                model_name=result.model_name, failures=[], reviews=[],
                thresholds={},
            )

        errors = np.array([o.error for o in result.outcomes])
        confs = np.array([o.confidence for o in result.outcomes])
        err_thr = float(np.percentile(errors, error_percentile))
        if confidence_threshold is not None:
            conf_thr: float | None = float(confidence_threshold)
        elif float(np.ptp(confs)) > 0.0:
            conf_thr = float(np.percentile(confs, 100 - error_percentile))
        else:
            conf_thr = None  # constant confidence: uninformative, ignore it

        failures: list[FailureRecord] = []
        for o in result.outcomes:
            high_err = o.error >= err_thr
            low_conf = conf_thr is not None and o.confidence <= conf_thr
            if not (high_err or low_conf):
                continue
            mode = (
                FailureMode.BOTH if (high_err and low_conf)
                else FailureMode.HIGH_ERROR if high_err
                else FailureMode.LOW_CONFIDENCE
            )
            failures.append(
                FailureRecord(
                    frame_id=o.frame_id, episode_id=o.episode_id,
                    dataset_id=result.dataset_id, version=result.version,
                    model_name=result.model_name, mode=mode,
                    error=o.error, confidence=o.confidence,
                )
            )

        reviews = self._nominate(result, failures, min_failures_per_episode)
        report = FeedbackReport(
            dataset_id=result.dataset_id, version=result.version,
            model_name=result.model_name, failures=failures, reviews=reviews,
            thresholds={
                "error_percentile": error_percentile,
                "error_threshold": round(err_thr, 6),
                "confidence_threshold": round(conf_thr, 6) if conf_thr is not None else None,
                "confidence_used": conf_thr is not None,
                "min_failures_per_episode": min_failures_per_episode,
            },
        )
        if persist:
            report.artifacts = self._persist(report)
        return report

    @staticmethod
    def _nominate(
        result: EvaluationResult,
        failures: Sequence[FailureRecord],
        min_failures: int,
    ) -> list[EpisodeReview]:
        """Aggregate frame failures into episode-level review candidates.

        Episode-level because that is the unit a human can actually review and
        the unit the dataset is built from.  A scattering of bad frames across
        many episodes is a model problem; a concentration inside one episode is
        usually a *data* problem, and that distinction is the whole point.
        """
        per_ep_failures: dict[int, list[FailureRecord]] = defaultdict(list)
        for f in failures:
            per_ep_failures[f.episode_id].append(f)

        per_ep_all: dict[int, list] = defaultdict(list)
        for o in result.outcomes:
            per_ep_all[o.episode_id].append(o)

        reviews: list[EpisodeReview] = []
        for ep_id, fs in per_ep_failures.items():
            if len(fs) < min_failures:
                continue
            allo = per_ep_all.get(ep_id, [])
            errs = np.array([o.error for o in allo]) if allo else np.array([0.0])
            confs = np.array([o.confidence for o in allo]) if allo else np.array([0.0])
            modes: dict[str, int] = defaultdict(int)
            for f in fs:
                modes[f.mode.value] += 1
            rate = len(fs) / max(len(allo), 1)
            # Priority blends how often it fails with how badly, so a short
            # episode that fails everywhere outranks a long one with a few spikes.
            priority = round(float(rate * (1.0 + errs.mean())), 6)
            reviews.append(
                EpisodeReview(
                    episode_id=ep_id,
                    failure_frames=len(fs),
                    total_frames_evaluated=len(allo),
                    mean_error=round(float(errs.mean()), 6),
                    max_error=round(float(errs.max()), 6),
                    mean_confidence=round(float(confs.mean()), 6),
                    modes=dict(modes),
                    priority=priority,
                    recommendation=(
                        "inspect for data defects; failure concentrated in this episode"
                        if rate > 0.5 else
                        "collect more demonstrations covering this state region"
                    ),
                )
            )
        return sorted(reviews, key=lambda r: -r.priority)

    # -- action -------------------------------------------------------------
    def quarantine_episodes(
        self, dataset_id: str, version: str, episode_ids: Sequence[int]
    ) -> int:
        """Mark nominated episodes for human review in the catalog.

        Quarantine, not rejection: the model failing on an episode is evidence
        about the *model* at least as often as about the data.  Deleting it would
        discard exactly the examples the next training round needs most.
        """
        self.catalog.set_episode_quality(
            dataset_id, version, {int(e): QualityStatus.QUARANTINED for e in episode_ids}
        )
        return len(episode_ids)

    def build_retraining_selection(
        self,
        report: FeedbackReport,
        oversample_factor: int = 2,
        include_all: bool = True,
    ) -> dict[str, Any]:
        """Produce a weighted episode selection for the next training version.

        Rather than dropping data, flagged episodes are *upweighted*: the model
        struggled there, so the next round should see them more often.  The
        selection is data, not action -- it is handed to the dataset builder,
        which records it in lineage like any other parameter.
        """
        flagged = [r.episode_id for r in report.reviews]
        weights: dict[int, float] = {}
        if include_all:
            for f in report.failures:
                weights.setdefault(f.episode_id, 1.0)
        for ep in flagged:
            weights[ep] = float(oversample_factor)
        return {
            "source_version": report.version,
            "model": report.model_name,
            "flagged_episodes": flagged,
            "episode_weights": weights,
            "oversample_factor": oversample_factor,
            "rationale": "episodes with concentrated model failures are upweighted, not removed",
        }

    # -- persistence --------------------------------------------------------
    def _persist(self, report: FeedbackReport) -> dict[str, str]:
        import pyarrow as pa

        from ..storage import ParquetTableStore

        out = (
            self.settings.paths.artifact_dir(report.dataset_id, report.version)
            / "feedback" / report.model_name
        )
        out.mkdir(parents=True, exist_ok=True)

        rows = [f.as_row() for f in report.failures]
        store = ParquetTableStore(out, compression=self.settings.parquet_compression)
        store.write_table(
            "failures",
            pa.table({
                "frame_id": pa.array([r["frame_id"] for r in rows], pa.int64()),
                "episode_id": pa.array([r["episode_id"] for r in rows], pa.int64()),
                "model_name": pa.array([r["model_name"] for r in rows]),
                "mode": pa.array([r["mode"] for r in rows]),
                "error": pa.array([r["error"] for r in rows], pa.float64()),
                "confidence": pa.array([r["confidence"] for r in rows], pa.float64()),
                "detected_at": pa.array([r["detected_at"] for r in rows]),
            }),
        )
        (out / "review.json").write_text(
            json.dumps(
                {"summary": report.summary(),
                 "reviews": [r.to_dict() for r in report.reviews]},
                indent=2, default=str,
            )
        )
        return {"failures": str(out / "failures"), "review": str(out / "review.json")}
