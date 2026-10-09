"""Concrete input adapters.

Three, spanning the range of what robot policies consume:

:class:`StateAdapter`          proprioception only.  No image decode, so it is
                               the fastest path and the right baseline.
:class:`VisionStateAdapter`    images + state.  Decodes lazily, applies the
                               manifest's deferred read transforms.
:class:`VisionLanguageAdapter` images + state + the task instruction, i.e. the
                               VLA input shape.

Each reads the training contract for key names and shapes; none contains a
literal ``"observation.state"`` or ``"rgb.top"``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..engine import Sample
from .base import ModelInput, ModelInputAdapter


class StateAdapter(ModelInputAdapter):
    """Proprioceptive state only -- concatenated inline modalities."""

    name = "state"

    def adapt(self, samples: Sequence[Sample]) -> ModelInput:
        """Concatenate every inline state modality into one (batch, dim) array."""
        rows = []
        for s in samples:
            parts = []
            for k in self.state_keys:
                if k in s.observation:
                    parts.append(np.asarray(s.observation[k], dtype=np.float32))
                else:
                    # Zero-fill to the contract's declared width rather than
                    # omitting the modality, mirroring VisionStateAdapter's
                    # missing-camera handling below. Omitting it silently
                    # shortens this sample's concatenated vector relative to
                    # samples where the modality IS present, which either
                    # crashes np.stack() mid-batch on a mixed batch or, if
                    # every sample in a batch happens to be missing it
                    # uniformly, silently produces a shorter-than-declared
                    # state vector that mismatches what the model was told
                    # to expect.
                    spec = self.contract["observation"].get(k, {})
                    width = int((spec.get("shape") or [0])[0])
                    parts.append(np.zeros(width, dtype=np.float32))
            rows.append(np.concatenate(parts) if parts else np.zeros(0, np.float32))
        state = np.stack(rows) if rows else np.zeros((0, 0), np.float32)
        return ModelInput(
            tensors={"state": state},
            instruction=samples[0].task if samples else None,
            frame_ids=[s.frame_id for s in samples],
            episode_ids=[s.episode_id for s in samples],
            metadata={"state_keys": self.state_keys},
        )


class VisionStateAdapter(ModelInputAdapter):
    """Camera views plus state.

    Images are decoded here and nowhere earlier: the samples arriving from
    :meth:`DatasetView.iter_samples` carry references, and this is the point at
    which a model actually needs pixels.  A state-only model therefore never pays
    for video decode.
    """

    name = "vision_state"

    def __init__(self, contract: dict[str, Any], cameras: Sequence[str] | None = None,
                 channels_first: bool = True, dtype: str = "float32"):
        super().__init__(contract)
        self.cameras = list(cameras) if cameras else list(self.image_keys)
        #: PyTorch convention is NCHW; keep it configurable for JAX/TF models.
        self.channels_first = channels_first
        self.dtype = dtype

    def _stack_images(self, samples: Sequence[Sample], camera: str) -> np.ndarray:
        imgs = []
        for s in samples:
            img = s.image(camera)
            if img is None:
                spec = self.contract["observation"].get(camera, {})
                shape = spec.get("shape") or [224, 224, 3]
                img = np.zeros(tuple(int(x) for x in shape), dtype=np.uint8)
            imgs.append(img)
        arr = np.stack(imgs)
        if arr.dtype == np.uint8 and self.dtype != "uint8":
            arr = arr.astype(np.float32) / 255.0
        if self.channels_first and arr.ndim == 4:
            arr = np.transpose(arr, (0, 3, 1, 2))
        return arr

    def adapt(self, samples: Sequence[Sample]) -> ModelInput:
        """State plus one decoded, laid-out tensor per camera."""
        base = StateAdapter(self.contract).adapt(samples)
        tensors = dict(base.tensors)
        for cam in self.cameras:
            tensors[f"image.{cam}"] = self._stack_images(samples, cam)
        return ModelInput(
            tensors=tensors,
            instruction=base.instruction,
            frame_ids=base.frame_ids,
            episode_ids=base.episode_ids,
            metadata={"cameras": self.cameras, "channels_first": self.channels_first,
                      "state_keys": self.state_keys},
        )

    def describe(self) -> dict[str, Any]:
        """Adds the camera list and tensor layout to the base description."""
        d = super().describe()
        d["consumes"]["cameras"] = self.cameras
        d["layout"] = "NCHW" if self.channels_first else "NHWC"
        return d


class VisionLanguageAdapter(VisionStateAdapter):
    """VLA input: cameras + state + the natural-language task instruction.

    The instruction comes from the canonical task table, so a dataset with many
    tasks conditions per-sample rather than per-dataset.
    """

    name = "vision_language"

    def adapt(self, samples: Sequence[Sample]) -> ModelInput:
        """Vision and state plus the per-sample task instruction."""
        mi = super().adapt(samples)
        instructions = [s.task or "" for s in samples]
        mi.tensors["instruction_index"] = np.asarray(
            [s.task_id if s.task_id is not None else -1 for s in samples], dtype=np.int64
        )
        mi.metadata["instructions"] = instructions
        mi.instruction = instructions[0] if instructions else None
        return mi

    def describe(self) -> dict[str, Any]:
        """Marks the adapter as language-conditioned."""
        d = super().describe()
        d["consumes"]["language"] = True
        return d
