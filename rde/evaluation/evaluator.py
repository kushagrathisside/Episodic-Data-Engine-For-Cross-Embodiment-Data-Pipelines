"""Evaluation harness.

Consumes a dataset *version* and a model, produces metrics and -- crucially for
the flywheel -- a per-frame record of where the model did badly or was unsure.
That record is the input to :mod:`rde.feedback`.

Metrics are chosen to be interpretable for continuous control:

``mse`` / ``mae``       raw error in action units.
``per_dimension_mae``   which joint the model gets wrong; an aggregate number
                        hides that the gripper is the failure and the arm is fine.
``normalized_mae``      MAE divided by the action's own standard deviation, so
                        dimensions with different ranges are comparable.
``success_rate``        fraction of frames within a tolerance -- a proxy, and
                        labelled as one.

Every run is compared against a baseline where one is supplied, because an
absolute MSE on normalised actions is close to meaningless on its own.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..config import Settings, get_settings
from ..engine import DatasetView, Sample
from ..model_adapters.base import Model, ModelInputAdapter, Prediction



def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation, or None when either side is effectively constant.

    An absolute variance threshold is not safe here.  ``np.full(n, 0.1,
    dtype=float32).std()`` is 7.45e-09 rather than 0 -- float32 rounding -- so a
    constant confidence vector slips past a 1e-9 guard and ``corrcoef`` then
    divides by a near-zero sigma and returns NaN.  A NaN correlation is worse
    than a missing one: the feedback loop would read it as a real (if useless)
    signal.  So the test is *relative* to each vector's own scale, and both sides
    are checked, not just one.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size != a.size:
        return None
    for v in (a, b):
        scale = max(abs(float(np.mean(v))), 1.0)
        if float(np.std(v)) <= 1e-6 * scale:
            return None
    with np.errstate(invalid="ignore", divide="ignore"):
        c = float(np.corrcoef(a, b)[0, 1])
    return None if not np.isfinite(c) else c


@dataclass
class FrameOutcome:
    """Per-frame evaluation record; the raw material of the feedback loop."""

    frame_id: int
    episode_id: int
    error: float
    confidence: float
    per_dim_error: list[float] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        """Flat dict for the per-frame outcomes Parquet table."""
        return {
            "frame_id": self.frame_id,
            "episode_id": self.episode_id,
            "error": self.error,
            "confidence": self.confidence,
        }


@dataclass
class EvaluationResult:
    """Metrics plus the per-frame outcomes the feedback loop consumes."""
    dataset_id: str
    version: str
    split: str
    model_name: str
    adapter_name: str
    metrics: dict[str, Any]
    outcomes: list[FrameOutcome]
    started_at: _dt.datetime
    finished_at: _dt.datetime
    baseline: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def frames_evaluated(self) -> int:
        """Number of frames scored."""
        return len(self.outcomes)

    def summary(self) -> dict[str, Any]:
        """Headline metrics plus the baseline comparison, if attached."""
        out = {
            "dataset": f"{self.dataset_id}@{self.version}",
            "split": self.split,
            "model": self.model_name,
            "adapter": self.adapter_name,
            "frames": self.frames_evaluated,
            "duration_s": round((self.finished_at - self.started_at).total_seconds(), 2),
            **{k: v for k, v in self.metrics.items() if not isinstance(v, list)},
        }
        if self.baseline:
            out["vs_baseline"] = self.baseline
        return out


class Evaluator:
    """Runs a model over a dataset version and scores it."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def evaluate(
        self,
        view: DatasetView,
        model: Model,
        adapter: ModelInputAdapter,
        split: str = "test",
        batch_size: int = 64,
        max_frames: int | None = None,
        success_tolerance: float = 0.1,
        persist: bool = True,
        episode_ids: Sequence[int] | None = None,
    ) -> EvaluationResult:
        """Run a model over one split and score it, retaining per-frame outcomes.

        ``episode_ids``, when given, evaluates exactly that episode set instead of
        a named split -- the mechanism stratified evaluation (per-task, per
        -variation, a fixed "hard cases" slice, ...) uses. It takes precedence
        over ``split`` for frame selection, matching
        :meth:`~rde.engine.DatasetView.iter_samples`'s own semantics; ``split`` is
        still recorded on the result and used to name persisted artifacts, so
        callers evaluating a slice should pass a descriptive label (e.g.
        ``"test:task_pickplace"``) rather than leaving it as a plain split name.
        """
        started = _dt.datetime.now(_dt.timezone.utc)
        act_key = view.contract["action"]["key"]

        errors: list[np.ndarray] = []
        confidences: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        outcomes: list[FrameOutcome] = []
        latencies: list[float] = []
        seen = 0

        for batch in adapter.batched(
            view.iter_samples(split=split, episode_ids=episode_ids), batch_size
        ):
            if max_frames is not None and seen >= max_frames:
                break
            truth = np.stack([
                s.action if s.action is not None
                else np.zeros(adapter.action_dim, np.float32)
                for s in batch
            ])
            mi = adapter.adapt(batch)
            pred = model.timed_predict(mi)
            latencies.append(pred.latency_ms / max(len(batch), 1))

            pa_ = pred.actions
            if pa_.shape != truth.shape:  # defensive: shape drift is a real bug source
                raise ValueError(
                    f"model returned actions {pa_.shape}, dataset expects {truth.shape}"
                )
            err = pa_ - truth
            errors.append(err)
            truths.append(truth)
            confidences.append(np.asarray(pred.confidence, np.float32))

            per_frame = np.abs(err).mean(axis=1)
            for i, s in enumerate(batch):
                outcomes.append(
                    FrameOutcome(
                        frame_id=s.frame_id,
                        episode_id=s.episode_id,
                        error=float(per_frame[i]),
                        confidence=float(pred.confidence[i]),
                        per_dim_error=np.abs(err[i]).round(5).tolist(),
                    )
                )
            seen += len(batch)

        finished = _dt.datetime.now(_dt.timezone.utc)
        if not errors:
            return EvaluationResult(
                dataset_id=view.dataset_id, version=view.version, split=split,
                model_name=model.name, adapter_name=adapter.name,
                metrics={"error": f"no frames in split {split!r}"}, outcomes=[],
                started_at=started, finished_at=finished,
            )

        E = np.vstack(errors)
        T = np.vstack(truths)
        C = np.concatenate(confidences)
        # Normalising by the ground truth's own spread makes per-dimension errors
        # comparable and stops a wide-range joint from dominating the headline.
        spread = T.std(axis=0)
        spread = np.where(spread > 1e-8, spread, 1.0)

        metrics: dict[str, Any] = {
            "mse": float((E ** 2).mean()),
            "mae": float(np.abs(E).mean()),
            "rmse": float(np.sqrt((E ** 2).mean())),
            "per_dimension_mae": np.abs(E).mean(axis=0).round(5).tolist(),
            "normalized_mae": float((np.abs(E).mean(axis=0) / spread).mean()),
            "success_rate": float((np.abs(E).mean(axis=1) < success_tolerance).mean()),
            "success_tolerance": success_tolerance,
            "mean_confidence": float(C.mean()),
            # None when confidence is constant -- see _safe_corr.
            "confidence_error_corr": _safe_corr(C, np.abs(E).mean(axis=1)),
            "latency_ms_per_frame": round(float(np.mean(latencies)), 4),
            "action_dim": int(E.shape[1]),
        }

        result = EvaluationResult(
            dataset_id=view.dataset_id, version=view.version, split=split,
            model_name=model.name, adapter_name=adapter.name,
            metrics=metrics, outcomes=outcomes,
            started_at=started, finished_at=finished,
        )
        if persist:
            result.artifacts = self._persist(result, model, adapter)
        return result

    def compare(
        self, result: EvaluationResult, baseline: EvaluationResult
    ) -> EvaluationResult:
        """Attach a baseline comparison.

        Reported as ratios: an MSE that is 0.4x the baseline's means something,
        an MSE of 0.031 does not.
        """
        result.baseline = {
            "model": baseline.model_name,
            "mse_ratio": round(result.metrics["mse"] / max(baseline.metrics["mse"], 1e-12), 4),
            "mae_ratio": round(result.metrics["mae"] / max(baseline.metrics["mae"], 1e-12), 4),
            "baseline_mse": round(baseline.metrics["mse"], 6),
            "baseline_mae": round(baseline.metrics["mae"], 6),
            "improvement": (
                "better" if result.metrics["mse"] < baseline.metrics["mse"] else "worse"
            ),
        }
        return result

    def _persist(
        self, result: EvaluationResult, model: Model, adapter: ModelInputAdapter
    ) -> dict[str, str]:
        import pyarrow as pa

        from ..storage import ParquetTableStore

        out = (
            self.settings.paths.artifact_dir(result.dataset_id, result.version)
            / "evaluation" / f"{result.model_name}-{result.split}"
        )
        out.mkdir(parents=True, exist_ok=True)

        store = ParquetTableStore(out, compression=self.settings.parquet_compression)
        rows = [o.as_row() for o in result.outcomes]
        store.write_table(
            "outcomes",
            pa.table({
                "frame_id": pa.array([r["frame_id"] for r in rows], pa.int64()),
                "episode_id": pa.array([r["episode_id"] for r in rows], pa.int64()),
                "error": pa.array([r["error"] for r in rows], pa.float64()),
                "confidence": pa.array([r["confidence"] for r in rows], pa.float64()),
            }),
        )
        (out / "metrics.json").write_text(
            json.dumps(
                {
                    "summary": result.summary(),
                    "metrics": result.metrics,
                    "model": model.describe(),
                    "adapter": adapter.describe(),
                },
                indent=2, default=str,
            )
        )
        return {"outcomes": str(out / "outcomes"), "metrics": str(out / "metrics.json")}
