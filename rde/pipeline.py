"""End-to-end pipeline orchestration.

Wires the stages into the flow the design calls for::

    DOWNLOAD -> INGEST -> VALIDATE -> SYNCHRONIZE -> TRANSFORM -> QC
             -> VERSION -> BUILD -> MODEL ADAPTER -> EVALUATE -> FEEDBACK

Each stage is independently callable; this module exists so the whole thing can
be run reproducibly from a config file, and so the docs have a single place that
shows the composition rather than describing it.

Validation and QC appear as one stage here rather than two.  The distinction the
spec draws -- validate the raw import, then QC after transformation -- is a
distinction of *when the suite runs*, not of what it does, so the same engine is
invoked at both points with different gates.  Running it twice is deliberate:
the raw pass decides what is fit to transform, the post-transform pass verifies
the transformation did not itself introduce defects.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .catalog import Catalog
from .config import Settings, get_settings
from .dataset_builder import DatasetBuilder
from .engine import DataEngine
from .evaluation import Evaluator
from .feedback import FeedbackLoop
from .ingestion import download_hf_dataset, ingest
from .model_adapters import build_model
from .observability import ObservabilitySession
from .qc import QCEngine, QCGate
from .schemas.core import QualityStatus
from .schemas.qc import Severity
from .transformations import (
    NormalizeVector,
    QualityFilter,
    ResizeImages,
    TimestampNormalizer,
    TransformPipeline,
)


@dataclass
class PipelineConfig:
    """Everything needed to reproduce a full run."""

    dataset_id: str
    source_repo: str | None = None
    source_path: str | None = None
    source_format: str = "lerobot"

    raw_version: str = "raw/v1"
    clean_version: str = "clean/v1"
    training_version: str = "training/v1"

    normalize: tuple[str, ...] = ("state.joint", "action.joint")
    normalize_scheme: str = "mean_std"
    image_size: tuple[int, int] | None = (224, 224)
    keep_quality: tuple[str, ...] = ("passed",)

    splits: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    split_seed: int = 42
    split_mode: str = "quantile"

    model: str = "nearest_neighbor"
    baseline_model: str | None = "mean_action"
    eval_split: str = "test"
    eval_max_frames: int | None = 2000

    deep_video_check: bool = False

    @classmethod
    def from_file(cls, path: Path | str) -> "PipelineConfig":
        """Load a config from JSON, restoring tuple-typed fields."""
        data = json.loads(Path(path).read_text())
        if "image_size" in data and data["image_size"]:
            data["image_size"] = tuple(data["image_size"])
        if "normalize" in data:
            data["normalize"] = tuple(data["normalize"])
        if "keep_quality" in data:
            data["keep_quality"] = tuple(data["keep_quality"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict; round-trips back through from_file."""
        d = dict(self.__dict__)
        d["image_size"] = list(self.image_size) if self.image_size else None
        d["normalize"] = list(self.normalize)
        d["keep_quality"] = list(self.keep_quality)
        return d


@dataclass
class StageResult:
    """One pipeline stage: name, success, duration and its result payload."""
    name: str
    ok: bool
    duration_s: float
    detail: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Runs the full flow, recording each stage."""

    def __init__(
        self,
        config: PipelineConfig,
        settings: Settings | None = None,
        observability: bool = True,
    ):
        self.config = config
        self.settings = settings or get_settings()
        self.settings.paths.ensure()
        self.catalog = Catalog(self.settings.paths.catalog_db)
        self.stages: list[StageResult] = []
        self.observability_enabled = observability
        #: Replaced with a live session in ``run()``.  Starting disabled means a
        #: stage invoked directly -- ``Pipeline(cfg).ingest()`` -- still works
        #: with no run around it, which is how the examples and tests use it.
        self.obs = ObservabilitySession.disabled()

    #: Detail-dict keys a stage may report, mapped onto the volume counters.
    #: Table-driven so a new stage reports volume by naming its keys the same
    #: way the others do, rather than by editing the instrumentation.
    #: Ordered: the explicit ``*_in``/``*_out`` keys win over the display keys,
    #: because a stage that transforms reports ``"19631 -> 18668"`` for humans
    #: and integers for instruments.
    _VOLUME_KEYS = (
        ("frames", "records_out"),
        ("episodes", "episodes"),
        ("frames_in", "records_in"),
        ("frames_out", "records_out"),
        ("episodes_out", "episodes"),
    )

    def _stage(self, name: str, fn):
        """Run one stage, recording it both locally and as telemetry.

        The local ``StageResult`` list is kept as-is: the run record's shape is
        a published interface, and observability must not change what the
        pipeline returns.  The span, metrics and event row are additive.
        """
        t0 = _dt.datetime.now()
        with self.obs.stage(name, dataset_id=self.config.dataset_id) as handle:
            try:
                detail = fn() or {}
                ok = True
            except Exception as exc:
                detail = {"error": f"{type(exc).__name__}: {exc}"}
                ok = False
                self.stages.append(
                    StageResult(name, ok, (_dt.datetime.now() - t0).total_seconds(), detail)
                )
                raise
            counts = {
                target: detail[key]
                for key, target in self._VOLUME_KEYS
                if isinstance(detail.get(key), int)
            }
            if counts:
                handle.counts(**counts)
            if isinstance(detail.get("size_mb"), (int, float)):
                handle.bytes(read=int(detail["size_mb"] * 1e6))
        self.stages.append(
            StageResult(name, ok, (_dt.datetime.now() - t0).total_seconds(), detail)
        )
        return detail

    # -- stages -------------------------------------------------------------
    def download(self) -> dict[str, Any]:
        """Fetch the source into the cache, or skip if a local path was given."""
        if not self.config.source_repo:
            return {"skipped": "no source_repo; using source_path"}
        loc = download_hf_dataset(self.config.source_repo, settings=self.settings)
        self.config.source_path = str(loc.local_path)
        return {"path": str(loc.local_path), "size_mb": round(loc.size_bytes / 1e6, 1),
                "files": loc.file_count}

    def ingest(self) -> dict[str, Any]:
        """Convert the source into the canonical raw version."""
        if not self.config.source_path:
            raise ValueError("source_path is not set; run download() first")
        res = ingest(
            Path(self.config.source_path),
            self.config.dataset_id,
            self.config.source_format,
            self.config.raw_version,
            source_uri=(
                f"hf://{self.config.source_repo}" if self.config.source_repo else None
            ),
            settings=self.settings,
            catalog=self.catalog,
        )
        return res.summary()

    def validate(self, version: str | None = None, gate: QCGate | None = None) -> dict[str, Any]:
        """Run the full check suite, including cross-modal alignment."""
        version = version or self.config.raw_version
        registry = None
        if self.config.deep_video_check:
            from .validation import default_registry
            from .validation.structural import VideoIntegrityCheck

            registry = default_registry()
            registry._checks["structural.video_integrity"] = VideoIntegrityCheck(deep=True)
        eng = QCEngine(
            settings=self.settings, catalog=self.catalog, registry=registry, gate=gate
        )
        res = eng.run(self.config.dataset_id, version)
        # ``episodes`` is picked up by the stage instrumentation's volume map,
        # so the stage row records what was validated rather than leaving the
        # dominant stage with a null volume.
        return {**res.summary(), "episodes": len(res.episode_status)}

    def transform(self) -> dict[str, Any]:
        """Apply the configured operator chain, producing the clean version."""
        ops = [
            QualityFilter([QualityStatus(q) for q in self.config.keep_quality]),
            TimestampNormalizer(),
        ]
        ops += [
            NormalizeVector(m, self.config.normalize_scheme) for m in self.config.normalize
        ]
        if self.config.image_size:
            ops.append(ResizeImages(*self.config.image_size))
        pipe = TransformPipeline(ops, settings=self.settings, catalog=self.catalog)
        return pipe.run(
            self.config.dataset_id, self.config.raw_version, self.config.clean_version
        ).summary()

    def build(self) -> dict[str, Any]:
        """Produce the training version with splits and a contract."""
        b = DatasetBuilder(settings=self.settings, catalog=self.catalog)
        return b.build(
            self.config.dataset_id,
            self.config.clean_version,
            self.config.training_version,
            splits=self.config.splits,
            seed=self.config.split_seed,
            split_mode=self.config.split_mode,
        ).summary()

    def evaluate(self) -> dict[str, Any]:
        """Fit and score the configured model, against a baseline if set."""
        engine = DataEngine(self.settings)
        ev = Evaluator(self.settings)

        with engine.get_dataset(self.config.dataset_id, self.config.training_version) as view:
            model, adapter = build_model(self.config.model, view.contract)
            model.fit(view, split="train")
            result = ev.evaluate(
                view, model, adapter, split=self.config.eval_split,
                max_frames=self.config.eval_max_frames,
            )

            if self.config.baseline_model:
                bmodel, badapter = build_model(self.config.baseline_model, view.contract)
                bmodel.fit(view, split="train")
                bres = ev.evaluate(
                    view, bmodel, badapter, split=self.config.eval_split,
                    max_frames=self.config.eval_max_frames, persist=False,
                )
                ev.compare(result, bres)

        self._last_eval = result
        return result.summary()

    def feedback(self) -> dict[str, Any]:
        """Detect failures from the last evaluation and build a retraining selection."""
        result = getattr(self, "_last_eval", None)
        if result is None:
            raise RuntimeError("run evaluate() before feedback()")
        loop = FeedbackLoop(settings=self.settings, catalog=self.catalog)
        report = loop.detect(result)
        selection = loop.build_retraining_selection(report)
        self._last_feedback = report
        self._last_selection = selection
        return {**report.summary(), "retraining_selection": selection}

    # -- composition --------------------------------------------------------
    def run(self, skip_download: bool = False) -> dict[str, Any]:
        """Execute every stage and return a machine-readable run record."""
        started = _dt.datetime.now(_dt.timezone.utc)
        self.obs = (
            ObservabilitySession.for_run(
                self.config.dataset_id,
                self.settings.software_version,
                self.settings.paths,
                config=self.config.to_dict(),
                catalog=self.catalog,
            )
            if self.observability_enabled
            else ObservabilitySession.disabled()
        )
        status = "ok"
        with self.obs.active():
            try:
                if not skip_download:
                    self._stage("download", self.download)
                self._stage("ingest", self.ingest)
                self._stage("validate", lambda: self.validate(self.config.raw_version))
                self._stage("transform", self.transform)
                # Second QC pass: proves the transformation did not introduce defects.
                self._stage(
                    "qc_post_transform",
                    lambda: self.validate(self.config.clean_version),
                )
                self._stage("build", self.build)
                self._stage("evaluate", self.evaluate)
                self._stage("feedback", self.feedback)
            except Exception:
                status = "error"
                self.obs.finish(status)
                raise
        telemetry = self.obs.finish(status)
        finished = _dt.datetime.now(_dt.timezone.utc)

        record = {
            "dataset_id": self.config.dataset_id,
            # The correlation spine.  Every lineage row, QC finding, span and
            # event row emitted by this execution carries the same run_id, so
            # the run record is now a join key rather than a loose artifact.
            "run_id": self.obs.ctx.run_id,
            "trace_id": self.obs.ctx.trace_id,
            "config": self.config.to_dict(),
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_s": round((finished - started).total_seconds(), 2),
            "stages": [
                {"name": s.name, "ok": s.ok, "duration_s": round(s.duration_s, 3),
                 "detail": s.detail}
                for s in self.stages
            ],
            "versions": self.catalog.list_versions(self.config.dataset_id),
        }
        run_path = (
            self.settings.paths.runs
            / f"{self.config.dataset_id}-{started.strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        run_path.parent.mkdir(parents=True, exist_ok=True)
        record["telemetry"] = telemetry
        run_path.write_text(json.dumps(record, indent=2, default=str))
        record["run_record"] = str(run_path)
        return record
