"""Model adapter contract.

This is the seam that keeps the data platform independent of any model.  The
flow the spec calls for::

    Dataset -> ModelInputAdapter -> Model -> Prediction -> Evaluation

An adapter's whole job is to translate the *canonical* representation into
whatever tensor layout, key naming and normalisation a particular model demands.
Nothing above it changes when the model changes.

The direction of the dependency is the point: adapters import from the data
engine, never the reverse.  Ingestion, storage, QC and versioning contain no
reference to any model, so a new model is a new file in this package and nothing
else.

Adapters read the *training contract* (:func:`rde.dataset_builder.build_training_contract`)
rather than hard-coding key names, so a dataset with different modality names --
a bimanual robot with ``rgb.left``/``rgb.right``, say -- works without editing the
adapter.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..engine import DatasetView, Sample


@dataclass
class ModelInput:
    """Whatever one model needs for one (or a batch of) timestep(s)."""

    #: Named arrays, model-specific: 'state', 'pixels', 'input_ids'...
    tensors: dict[str, np.ndarray] = field(default_factory=dict)
    #: Natural-language goal, when the model is language-conditioned.
    instruction: str | None = None
    #: Provenance so a prediction can be traced back to its source frames.
    frame_ids: list[int] = field(default_factory=list)
    episode_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        """Rows in this input, taken from the first tensor."""
        for v in self.tensors.values():
            return int(v.shape[0])
        return 0


@dataclass
class Prediction:
    """A model's output, plus what evaluation and the feedback loop need.

    ``confidence`` is required rather than optional: the deployment feedback loop
    selects low-confidence examples for review, so a model that cannot express
    uncertainty cannot participate in the flywheel.  Models without a native
    notion of confidence must derive one and say how in ``metadata``.
    """

    actions: np.ndarray                     # (batch, action_dim)
    confidence: np.ndarray                  # (batch,) in [0, 1]
    frame_ids: list[int] = field(default_factory=list)
    episode_ids: list[int] = field(default_factory=list)
    latency_ms: float = 0.0
    model_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelInputAdapter(ABC):
    """Canonical :class:`Sample` -> model-specific :class:`ModelInput`."""

    name: str = "unnamed"

    def __init__(self, contract: dict[str, Any]):
        self.contract = contract
        self.action_key = contract.get("action", {}).get("key")
        self.action_dim = int(
            (contract.get("action", {}).get("shape") or [0])[0]
        )
        self.observation_keys = list(contract.get("observation", {}))
        self.image_keys = [
            k for k, v in contract.get("observation", {}).items()
            if v.get("plane") == "external"
        ]
        self.state_keys = [
            k for k, v in contract.get("observation", {}).items()
            if v.get("plane") == "inline"
        ]

    @abstractmethod
    def adapt(self, samples: Sequence[Sample]) -> ModelInput:
        """Convert a batch of canonical samples into one model input."""

    def describe(self) -> dict[str, Any]:
        """What this adapter consumes and produces, for the run record."""
        return {
            "adapter": self.name,
            "consumes": {
                "state_keys": self.state_keys,
                "image_keys": self.image_keys,
                "language": self.contract.get("language_conditioned", False),
            },
            "produces_action_dim": self.action_dim,
        }

    @staticmethod
    def batched(samples: Iterable[Sample], batch_size: int) -> Iterable[list[Sample]]:
        """Chunk a sample stream into fixed-size lists."""
        buf: list[Sample] = []
        for s in samples:
            buf.append(s)
            if len(buf) >= batch_size:
                yield buf
                buf = []
        if buf:
            yield buf


class Model(ABC):
    """A policy that maps :class:`ModelInput` to :class:`Prediction`."""

    name: str = "unnamed"
    #: Adapter class this model expects; used by the registry to pair them.
    adapter_class: type[ModelInputAdapter] | None = None

    def fit(self, view: DatasetView, split: str = "train", **kwargs) -> None:
        """Optional training/fitting step.  Default: nothing to learn."""
        return None

    @abstractmethod
    def predict(self, model_input: ModelInput) -> Prediction:
        """Run inference.  Implementations must populate ``confidence``."""

    def describe(self) -> dict[str, Any]:
        """Model identity and configuration, recorded alongside metrics."""
        return {"model": self.name, "type": type(self).__name__}

    def timed_predict(self, model_input: ModelInput) -> Prediction:
        """Wrap :meth:`predict` with latency measurement."""
        t0 = time.perf_counter()
        pred = self.predict(model_input)
        pred.latency_ms = (time.perf_counter() - t0) * 1000.0
        pred.model_name = pred.model_name or self.name
        if not pred.frame_ids:
            pred.frame_ids = list(model_input.frame_ids)
        if not pred.episode_ids:
            pred.episode_ids = list(model_input.episode_ids)
        return pred
