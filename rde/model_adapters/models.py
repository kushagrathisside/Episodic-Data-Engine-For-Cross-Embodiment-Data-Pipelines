"""Reference models.

The engine's claim is that models are interchangeable, and the way to demonstrate
that is with models that genuinely differ in what they consume -- not three
wrappers around the same tensor.

:class:`MeanActionPolicy`      state-independent baseline.  Establishes the floor
                               any real model must beat.
:class:`NearestNeighborPolicy` behaviour retrieval: finds the closest training
                               state and replays its action.  Real, non-trivial,
                               CPU-only, and a genuinely strong baseline on
                               single-task demonstration data.
:class:`SmolVLAPolicy`         adapter for the actual SmolVLA checkpoint, used
                               when the optional heavy dependencies are present.

All three implement the same :class:`~rde.model_adapters.base.Model` interface
and are driven by the same evaluation code.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..engine import DatasetView
from .adapters import StateAdapter, VisionLanguageAdapter, VisionStateAdapter
from .base import Model, ModelInput, ModelInputAdapter, Prediction

#: Files a checkpoint saved by :meth:`SmolVLAPolicy.save_checkpoint` must have.
#: Used to fail loudly on a corrupted or foreign directory rather than loading
#: something partial and misattributing the result to a working checkpoint.
_CHECKPOINT_REQUIRED_FILES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "rde_checkpoint_meta.json",
)


def _policy_features_from_contract(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate a dataset's training contract into SmolVLA's ``input_features``/
    ``output_features`` (a ``{key: PolicyFeature}`` pair).

    ``lerobot/smolvla_base`` ships pre-configured for whatever embodiment it was
    itself tutorial-trained on (so100_pickplace: 6-DoF state/action, 2 cameras).
    Loading it as-is against a *different* dataset -- a different action
    dimensionality, a different number of cameras -- silently keeps that
    embodiment's shapes, which is wrong rather than merely suboptimal (a
    14-DoF bimanual action forced through a 6-wide head does not just lose
    accuracy, it cannot be represented). SmolVLA pads state/action internally
    to a fixed ``max_state_dim``/``max_action_dim`` budget (32) regardless of
    the real per-dataset width, so declaring the *real* width here changes
    what the model treats as real signal vs. padding -- it does not resize
    any weight tensor. Camera count is unconstrained the same way: each
    camera is pushed through the same shared vision encoder, so declaring
    more or fewer cameras only changes how many times that encoder runs per
    forward pass, not any parameter's shape. Both are verified against
    lerobot 0.4.4's own modeling_smolvla.py, not assumed.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature

    action = contract.get("action") or {}
    action_dim = int((action.get("shape") or [0])[0])
    if not action_dim:
        raise ValueError("dataset contract has no usable action shape")
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}

    input_features: dict[str, Any] = {}
    observation = contract.get("observation") or {}

    # Must match ModelInputAdapter.state_keys exactly (base.py): *every* inline
    # modality concatenated in observation-dict order, not just the one named
    # "robot_state". so100_pickplace only has state.joint (6-dim) inline, so
    # this coincides with "robot_state" there; aloha_coffee also has
    # force.effort (14-dim) as a second inline modality, concatenated onto
    # state.joint into a 28-dim vector by StateAdapter.adapt -- declaring only
    # the 14-dim robot_state part here would silently mismatch what the
    # adapter actually hands the model.
    state_entries = [e for e in observation.values() if e.get("plane") == "inline"]
    state_dim = sum(int((e.get("shape") or [0])[0]) for e in state_entries)
    if state_dim:
        input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))

    camera_entries = sorted(
        (e for e in observation.values() if e.get("plane") == "external"), key=lambda e: e["key"]
    )
    for i, cam in enumerate(camera_entries):
        h, w, c = cam["shape"]  # canonical rde image shape is (H, W, C); lerobot wants (C, H, W)
        input_features[f"observation.images.camera{i + 1}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(c, h, w)
        )
    if not camera_entries:
        raise ValueError("dataset contract has no camera modalities")
    return input_features, output_features


def _dataset_stats_from_contract(
    contract: dict[str, Any], input_features: dict[str, Any], output_features: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Fresh mean/std normalization statistics for the STATE and ACTION features,
    sourced from *this* dataset's own ingested statistics -- never from the
    pretrained checkpoint's saved processor, which encodes a different
    embodiment's joint distribution and would silently misnormalize this one.
    Images use SmolVLA's IDENTITY normalization mode and need no stats.
    """
    import torch

    stats: dict[str, dict[str, Any]] = {}
    observation = contract.get("observation") or {}
    if "observation.state" in input_features:
        # Same inline-modality set, same order, as _policy_features_from_contract
        # and StateAdapter.adapt -- concatenating stats out of that order would
        # normalize each dimension against the wrong modality's mean/std.
        state_entries = [e for e in observation.values() if e.get("plane") == "inline"]
        means: list[float] = []
        stds: list[float] = []
        for e in state_entries:
            norm = e.get("normalization") or {}
            means.extend(norm.get("mean", []))
            stds.extend(norm.get("std", []))
        if len(means) == input_features["observation.state"].shape[0]:
            stats["observation.state"] = {
                "mean": torch.tensor(means, dtype=torch.float32),
                "std": torch.tensor(stds, dtype=torch.float32),
            }
    action = contract.get("action") or {}
    norm = action.get("normalization") or {}
    if "mean" in norm and "std" in norm:
        stats["action"] = {
            "mean": torch.tensor(norm["mean"], dtype=torch.float32),
            "std": torch.tensor(norm["std"], dtype=torch.float32),
        }
    return stats



def _squared_distances(refs: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Pairwise squared distances, without materialising a 3-D difference array.

    The obvious ``norm(refs[None] - queries[:, None], axis=2)`` allocates a
    ``(queries, refs, dims)`` intermediate.  At 44 k reference frames and a batch
    of 64 that is 0.63 GB *per block*; at 500 k references it is 7.2 GB, which on
    WSL does not merely fail -- it can take the VM down.

    Expanding ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b needs only a
    ``(queries, refs)`` matrix: 22 MB and 0.26 GB for the same two cases, and
    independent of dimensionality.  Small negatives from floating-point
    cancellation are clipped, since the caller takes a square root.
    """
    q2 = np.einsum("ij,ij->i", queries, queries)[:, None]
    r2 = np.einsum("ij,ij->i", refs, refs)[None, :]
    d2 = q2 + r2 - 2.0 * (queries @ refs.T)
    return np.maximum(d2, 0.0, out=d2)

class MeanActionPolicy(Model):
    """Predicts the training-set mean action regardless of input.

    Exists to calibrate every other number.  A model that cannot beat this on a
    metric has learned nothing about that metric, and it is startlingly easy to
    report an impressive-looking MSE that the mean also achieves.
    """

    name = "mean_action"
    adapter_class = StateAdapter

    def __init__(self):
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, view: DatasetView, split: str = "train", **kwargs) -> None:
        """Stream the train split and accumulate the mean action."""
        total = None
        total_sq = None
        n = 0
        act_key = view.contract.get("action", {}).get("key")
        for batch in view.iter_frames(split=split, columns=[act_key]):
            import pyarrow as pa

            t = pa.Table.from_batches([batch])
            v = np.asarray(t.column(act_key).combine_chunks().flatten(), dtype=np.float64)
            v = v.reshape(t.num_rows, -1)
            total = v.sum(0) if total is None else total + v.sum(0)
            total_sq = (v ** 2).sum(0) if total_sq is None else total_sq + (v ** 2).sum(0)
            n += t.num_rows
        if n:
            self._mean = total / n
            self._std = np.sqrt(np.maximum(total_sq / n - self._mean ** 2, 0.0))

    def predict(self, model_input: ModelInput) -> Prediction:
        """Return the training mean, tiled across the batch."""
        b = model_input.batch_size
        dim = self._mean.shape[0] if self._mean is not None else 0
        actions = np.tile(self._mean if self._mean is not None else np.zeros(dim), (b, 1))
        return Prediction(
            actions=actions.astype(np.float32),
            # Constant, deliberately low: this model has no basis for confidence.
            confidence=np.full(b, 0.1, dtype=np.float32),
            model_name=self.name,
            metadata={"note": "state-independent baseline"},
        )

    def describe(self) -> dict[str, Any]:
        """Model identity plus the fitted mean."""
        return {
            "model": self.name, "type": "baseline",
            "mean_action": None if self._mean is None else np.round(self._mean, 4).tolist(),
        }


class NearestNeighborPolicy(Model):
    """Retrieval policy: nearest training state, replay its action.

    Strong on single-task demonstration data because the state trajectories of
    repeated attempts overlap heavily.  That strength is the point -- it makes a
    learned policy prove it is doing more than retrieval.

    Confidence is derived from the retrieval distance, mapped through the
    distance distribution seen during fitting: a query far from anything in the
    training set gets low confidence, which is exactly the signal the feedback
    loop needs to flag novel situations for review.
    """

    name = "nearest_neighbor"
    adapter_class = StateAdapter

    def __init__(self, k: int = 5, max_reference_frames: int = 50_000):
        self.k = k
        #: Cap on retained reference frames, so memory stays bounded on datasets
        #: far larger than this one.
        self.max_reference_frames = max_reference_frames
        self._states: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._dist_ref: float = 1.0

    def fit(self, view: DatasetView, split: str = "train", **kwargs) -> None:
        """Load a bounded reference set and calibrate the distance scale."""
        import pyarrow as pa

        contract = view.contract
        act_key = contract["action"]["key"]
        state_keys = [k for k, v in contract["observation"].items() if v.get("plane") == "inline"]
        cols = [*state_keys, act_key]

        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        kept = 0
        for batch in view.iter_frames(split=split, columns=cols):
            t = pa.Table.from_batches([batch])
            n = t.num_rows
            s = np.concatenate(
                [np.asarray(t.column(k).combine_chunks().flatten(), np.float32).reshape(n, -1)
                 for k in state_keys], axis=1)
            a = np.asarray(t.column(act_key).combine_chunks().flatten(), np.float32).reshape(n, -1)
            if kept + n > self.max_reference_frames:
                room = max(0, self.max_reference_frames - kept)
                s, a = s[:room], a[:room]
            if len(s):
                states.append(s)
                actions.append(a)
                kept += len(s)
            if kept >= self.max_reference_frames:
                break

        if not states:
            return
        self._states = np.vstack(states)
        self._actions = np.vstack(actions)
        # Per-dimension scaling so a wide-range joint does not dominate the
        # distance metric purely because of its units.
        std = self._states.std(axis=0)
        self._scale = np.where(std > 1e-6, std, 1.0)
        sample = self._states[:: max(1, len(self._states) // 1000)]
        d = self._pairwise_min(sample)
        self._dist_ref = float(np.percentile(d, 90)) or 1.0

    def _pairwise_min(self, queries: np.ndarray) -> np.ndarray:
        """Distance to nearest *other* reference point, for calibration."""
        q = (queries / self._scale).astype(np.float64)
        r = (self._states / self._scale).astype(np.float64)
        out = np.empty(len(q), dtype=np.float32)
        for i in range(0, len(q), 128):
            block = q[i : i + 128]
            d2 = _squared_distances(r, block)
            # index 1: nearest *other* point (index 0 is the point itself)
            part = np.partition(d2, 1, axis=1)
            out[i : i + len(block)] = np.sqrt(part[:, 1])
        return out

    def predict(self, model_input: ModelInput) -> Prediction:
        """Inverse-distance-weighted blend of the k nearest training actions."""
        state = model_input.tensors.get("state")
        if self._states is None or state is None or len(state) == 0:
            b = model_input.batch_size
            dim = self._actions.shape[1] if self._actions is not None else 0
            return Prediction(
                actions=np.zeros((b, dim), np.float32),
                confidence=np.zeros(b, np.float32),
                model_name=self.name,
            )

        q = (state / self._scale).astype(np.float64)
        r = (self._states / self._scale).astype(np.float64)
        actions = np.empty((len(q), self._actions.shape[1]), dtype=np.float32)
        conf = np.empty(len(q), dtype=np.float32)

        # Blocked to keep the distance matrix bounded regardless of batch size.
        for i in range(0, len(q), 64):
            block = q[i : i + 64]
            d2 = _squared_distances(r, block)
            k = min(self.k, d2.shape[1])
            idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
            rows = np.arange(len(block))[:, None]
            dk = np.sqrt(d2[rows, idx])
            # Inverse-distance weighting: closer neighbours dominate the blend.
            w = 1.0 / (dk + 1e-6)
            w = w / w.sum(axis=1, keepdims=True)
            actions[i : i + len(block)] = (self._actions[idx] * w[:, :, None]).sum(axis=1)
            nearest = dk.min(axis=1)
            conf[i : i + len(block)] = np.exp(-nearest / max(self._dist_ref, 1e-6))

        return Prediction(
            actions=actions,
            confidence=np.clip(conf, 0.0, 1.0),
            model_name=self.name,
            metadata={"k": self.k, "reference_frames": int(len(self._states))},
        )

    def describe(self) -> dict[str, Any]:
        """Model identity, k, reference size and the confidence formula."""
        return {
            "model": self.name, "type": "retrieval", "k": self.k,
            "reference_frames": 0 if self._states is None else int(len(self._states)),
            "confidence": "exp(-d_nearest / p90(d_intra))",
        }


class SmolVLAPolicy(Model):
    """Adapter for the real SmolVLA checkpoint.

    Kept import-light: ``torch`` and ``lerobot`` are only touched inside
    :meth:`load`, so the engine and its tests run without a 2.5 GB dependency
    tree.  If the checkpoint is unavailable this raises with an actionable
    message rather than silently degrading -- a model that quietly becomes a stub
    is worse than one that refuses to start.

    The point of this class is that it required no change anywhere else: it reads
    the same contract and returns the same :class:`Prediction` as the baselines.
    """

    name = "smolvla"
    adapter_class = VisionLanguageAdapter

    def __init__(self, checkpoint: str = "lerobot/smolvla_base", device: str | None = None):
        self.checkpoint = checkpoint
        self.device = device
        self._policy = None
        self._torch = None
        #: Populated on the first predict: checkpoint slot -> our canonical camera.
        self._camera_map: dict[str, str] = {}
        self._pre = None
        self._post = None
        self._checkpoint_meta: dict[str, Any] | None = None

    def load(self, contract: dict[str, Any] | None = None) -> "SmolVLAPolicy":
        """Import torch and lerobot, then load the checkpoint onto a device.

        ``contract`` is the *target* dataset's training contract
        (:attr:`DatasetView.contract`). When given, the checkpoint's own
        declared input/output shapes and normalization statistics are
        overridden with this dataset's real ones before weights are loaded --
        required whenever the target dataset's embodiment differs from
        whatever ``self.checkpoint`` was itself configured for (see
        :func:`_policy_features_from_contract`). When omitted (the default,
        and always the case for :meth:`from_checkpoint` reloading one of our
        own already-correctly-configured saved checkpoints), the checkpoint's
        own saved config and processor pipeline are used unchanged.
        """
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _P
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SmolVLA needs the optional model extras. Install with:\n"
                "  uv pip install 'torch' 'lerobot[smolvla]'\n"
                f"(original error: {exc})"
            ) from exc
        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        config = None
        dataset_stats = None
        if contract is not None:
            config = PreTrainedConfig.from_pretrained(self.checkpoint)
            input_features, output_features = _policy_features_from_contract(contract)
            config.input_features = input_features
            config.output_features = output_features
            dataset_stats = _dataset_stats_from_contract(contract, input_features, output_features)

        self._policy = _P.from_pretrained(self.checkpoint, config=config).to(self.device).eval()

        # LeRobot 0.4 moved normalisation and language tokenisation out of the
        # policy and into a processor pipeline shipped with the checkpoint
        # (policy_preprocessor.json / policy_postprocessor.json). Both
        # predict_action_chunk and fit's forward pass now expect
        # already-tokenised language, so the pipeline is not optional.
        #
        # Worth noting where this landed: a breaking change in a model library
        # cost one method in one adapter. Nothing in ingestion, storage, QC,
        # versioning or the dataset builder moved.
        # The shipped pipeline pins device="cuda"; override it so the same code
        # runs on a CPU-only box. Everything else in the pipeline -- normalisation
        # statistics, the language tokeniser -- comes from the checkpoint by
        # default, which is the point: those must match what the model was
        # trained with. When `contract` names a *different* dataset, its own
        # fresh statistics are supplied instead (`pretrained_path` omitted, so
        # the checkpoint's saved -- wrong-embodiment -- stats are never read).
        device_override = {"device_processor": {"device": self.device}}
        if dataset_stats is not None:
            self._pre, self._post = make_pre_post_processors(
                self._policy.config,
                dataset_stats=dataset_stats,
                preprocessor_overrides=device_override,
                postprocessor_overrides=device_override,
            )
        else:
            self._pre, self._post = make_pre_post_processors(
                self._policy.config,
                pretrained_path=self.checkpoint,
                preprocessor_overrides=device_override,
                postprocessor_overrides=device_override,
            )
        return self

    def _map_inputs(self, model_input: ModelInput) -> dict[str, Any]:
        """Translate a canonical :class:`ModelInput` into checkpoint-shaped tensors.

        Checkpoint-specific key mapping lives here, and only here, shared by
        :meth:`predict` and :meth:`fit` so the two paths cannot silently diverge.

        The engine names cameras canonically -- ``rgb.top``, ``rgb.wrist`` --
        because those names mean something about the robot. SmolVLA's released
        checkpoint names its inputs positionally: ``observation.images.camera1``
        ..``camera3``, fixed at pretraining time and unrelated to any particular
        rig. Neither name is wrong; they answer different questions.

        Translating between them is exactly this layer's job. The alternative --
        renaming our modalities to match one checkpoint -- would let a model
        dictate the data model, which is the coupling the whole design exists to
        prevent, and would break the moment a second checkpoint used different
        names.
        """
        torch = self._torch
        expected = sorted(getattr(self._policy.config, "image_features", {}) or {})
        ours = sorted(k for k in model_input.tensors if k.startswith("image."))

        batch: dict[str, Any] = {}
        for slot, key in zip(expected, ours):          # positional, cameras in name order
            batch[slot] = torch.as_tensor(
                model_input.tensors[key], device=self.device, dtype=torch.float32
            )
        if not batch:
            raise ValueError(
                f"no image features matched. checkpoint expects {expected}, "
                f"adapter produced {ours}"
            )
        self._camera_map = dict(zip(expected, ours))

        state = model_input.tensors.get("state")
        if state is not None:
            batch["observation.state"] = torch.as_tensor(
                state, device=self.device, dtype=torch.float32
            )
        batch["task"] = model_input.metadata.get(
            "instructions", [model_input.instruction or ""] * model_input.batch_size
        )
        return batch

    def predict(self, model_input: ModelInput) -> Prediction:
        """Run the checkpoint and return the first step of a freshly computed action chunk.

        Uses ``predict_action_chunk`` rather than ``select_action`` deliberately.
        ``select_action`` maintains an internal, stateful action queue intended
        for a *single continuous rollout*: it runs the model once every
        ``n_action_steps`` calls and replays cached actions from a queue on the
        calls in between -- correct for driving one robot through consecutive
        timesteps of the same episode, wrong for this engine's evaluation
        harness, which calls ``predict()`` once per *independent* batch of
        held-out dataset frames with no temporal relationship between calls.
        Reusing the queue across such batches silently replays a stale,
        previously-computed chunk against the wrong frames -- caught in
        practice by a batch-shape mismatch the one time a batch size changed
        mid-evaluation, but silently wrong (just not visibly so) on every
        same-sized batch before that. ``predict_action_chunk`` performs a fresh
        forward pass on every call and holds no state between calls, which is
        what an evaluation contract over independent batches requires. A real
        closed-loop rollout, were one available in this project, would need its
        own call path that manages ``select_action``'s queue deliberately --
        not this method.
        """
        if self._policy is None:
            raise RuntimeError("call load() before predict()")
        torch = self._torch
        # Force eval mode rather than trusting the caller left it there:
        # fit() puts the module in train() mode and only restores eval()
        # after the loop completes normally (see fit()'s try/finally), so a
        # caller that catches a fit() exception and calls predict() straight
        # after would otherwise silently get dropout/BatchNorm-affected,
        # non-deterministic predictions with no indication anything is wrong.
        self._policy.eval()
        batch = self._map_inputs(model_input)
        with torch.no_grad():
            actions = self._policy.predict_action_chunk(self._pre(batch))
            actions = self._post(actions)
        a = actions.detach().cpu().numpy()
        if a.ndim == 3:  # (batch, chunk, action_dim) -> first step
            a = a[:, 0, :]
        return Prediction(
            actions=a.astype(np.float32),
            # SmolVLA emits no calibrated uncertainty; declare that rather than
            # inventing a number the feedback loop would then trust.
            confidence=np.full(len(a), 0.5, dtype=np.float32),
            model_name=self.name,
            metadata={"checkpoint": self.checkpoint, "device": self.device,
                      "camera_map": getattr(self, "_camera_map", {}),
                      "confidence_note": "uncalibrated constant; model exposes no uncertainty"},
        )

    # -- fine-tuning ----------------------------------------------------------
    def fit(
        self,
        view: DatasetView,
        split: str = "train",
        *,
        steps: int = 200,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 0,
        episode_ids: Sequence[int] | None = None,
        episode_weights: dict[int, float] | None = None,
        seed: int = 0,
        optimizer_state: dict[str, Any] | None = None,
        on_step: Callable[[dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Real gradient-descent fine-tuning against a dataset split.

        Builds SmolVLA's expected training batch -- images, state, the raw task
        string (tokenised by the checkpoint's own preprocessor, exactly as
        :meth:`predict` uses it) and an *action chunk* of ``config.chunk_size``
        consecutive ground-truth actions per anchor frame (SmolVLA is a
        flow-matching policy trained against a trajectory segment, not a single
        step; see ``modeling_smolvla.forward``'s ``(batch, num_steps, motors)``
        loss shape). Steps past an episode's end are padded with its last action
        and masked out of the loss via ``actions_id_pad`` -- the exact key
        ``forward`` reads, matched against the installed checkpoint rather than
        assumed.

        ``episode_weights`` implements oversampling for the data flywheel and for
        a "rebalanced" data intervention: frames from a weighted episode are drawn
        proportionally more often, in place rather than duplicated on disk.

        Mirrors the forward/backward/step sequence lerobot's own
        ``scripts/lerobot_train.py`` uses (``preprocessor(batch) -> policy.forward
        -> backward -> clip -> optimizer.step``), but built on our own config and
        our own weighted sampler rather than lerobot's ``TrainPipelineConfig`` /
        dataloader, so the experiment config stays the single source of truth.
        """
        if self._policy is None:
            self.load()
        torch = self._torch

        adapter = VisionLanguageAdapter(view.contract)
        ids = list(episode_ids) if episode_ids is not None else view.episode_ids(split=split)
        if not ids:
            raise ValueError(f"no episodes resolved for split={split!r}, episode_ids={episode_ids!r}")

        act_key = view.contract.get("action", {}).get("key")
        if not act_key:
            raise ValueError("dataset contract has no action key; cannot fine-tune")
        chunk_size = max(1, int(getattr(self._policy.config, "chunk_size", 1)))

        # Bounded by the episodes being trained on, not dataset size: one
        # (episode_length, action_dim) float32 array per episode.
        episode_actions: dict[int, np.ndarray] = {}
        for eid in ids:
            table = view.get_episode_frames(eid, columns=["frame_index", act_key])
            d = table.to_pydict()
            if not d["frame_index"]:
                continue
            order = np.argsort(d["frame_index"])
            arr = np.asarray(d[act_key], dtype=np.float32)[order]
            episode_actions[int(eid)] = arr.reshape(len(order), -1)
        if not episode_actions:
            raise ValueError(f"no action data found for episodes {ids}")

        samples = list(view.iter_samples(split=None, episode_ids=list(episode_actions)))
        if not samples:
            raise ValueError(f"no frames resolved for episodes {list(episode_actions)}")

        weights = np.asarray(
            [(episode_weights or {}).get(s.episode_id, 1.0) for s in samples], dtype=np.float64
        )
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("sample weights must be finite and sum to a positive value")
        probs = weights / weights.sum()

        optimizer = torch.optim.AdamW(
            self._policy.get_optim_params(), lr=learning_rate, weight_decay=weight_decay
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        def _lr_scale(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            span = max(1, steps - warmup_steps)
            progress = min(1.0, (step - warmup_steps) / span)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_scale)
        rng = np.random.default_rng(seed)

        self._policy.train()
        history: list[dict[str, Any]] = []
        optimizer.zero_grad()
        try:
            for step in range(steps):
                micro_losses: list[float] = []
                for _ in range(max(1, grad_accum_steps)):
                    idx = rng.choice(len(samples), size=batch_size, p=probs)
                    batch_samples = [samples[i] for i in idx]
                    mi = adapter.adapt(batch_samples)
                    batch = self._map_inputs(mi)
                    actions = np.zeros((len(batch_samples), chunk_size, next(iter(episode_actions.values())).shape[1]),
                                        dtype=np.float32)
                    pad = np.zeros((len(batch_samples), chunk_size), dtype=bool)
                    for i, s in enumerate(batch_samples):
                        arr = episode_actions[s.episode_id]
                        start = s.frame_index
                        end = min(start + chunk_size, len(arr))
                        n = max(0, end - start)
                        if n > 0:
                            actions[i, :n] = arr[start:end]
                        if n < chunk_size:
                            actions[i, n:] = arr[-1] if len(arr) else 0.0
                            pad[i, n:] = True
                    batch["action"] = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
                    batch["actions_id_pad"] = torch.as_tensor(pad, device=self.device)
                    batch = self._pre(batch)
                    loss, _out = self._policy.forward(batch)
                    (loss / max(1, grad_accum_steps)).backward()
                    micro_losses.append(float(loss.detach().cpu()))
                grad_norm = None
                if max_grad_norm and max_grad_norm > 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), max_grad_norm)
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                record = {
                    "step": step,
                    "loss": float(np.mean(micro_losses)),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "grad_norm": grad_norm,
                }
                history.append(record)
                if on_step is not None:
                    on_step(record)
        finally:
            # Always restore eval mode, even on a mid-loop exception (OOM, a
            # malformed batch): predict() also forces eval() defensively now,
            # but leaving the module in train() mode after a failed fit() is
            # still the wrong resting state for any other code touching
            # self._policy directly (e.g. torch.save, another eval() call
            # elsewhere that assumes fit() cleaned up after itself).
            self._policy.eval()

        return {
            "steps": steps,
            "final_loss": history[-1]["loss"] if history else None,
            "history": history,
            "optimizer_state": optimizer.state_dict(),
        }

    # -- checkpointing ----------------------------------------------------------
    def save_checkpoint(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Save weights plus the exact pre/post-processor pipelines used to train them.

        Both are required to reload correctly: the checkpoint's normalisation
        statistics and language tokenizer live in the processor pipelines, not in
        the model weights, and a checkpoint reloaded with a *different* processor
        (e.g. the base model's defaults) would silently misnormalise every input.
        """
        if self._policy is None:
            raise RuntimeError("nothing to save: call load() (and typically fit()) first")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._policy.save_pretrained(path)
        self._pre.save_pretrained(path)
        self._post.save_pretrained(path)
        meta = {
            "checkpoint_source": self.checkpoint,
            "device": self.device,
            "action_dim": getattr(self._policy.config, "max_action_dim", None),
            "chunk_size": getattr(self._policy.config, "chunk_size", None),
            **(extra or {}),
        }
        (path / "rde_checkpoint_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return meta

    @staticmethod
    def verify_checkpoint(path: str | Path) -> dict[str, Any]:
        """Fail loudly on a missing/partial/foreign checkpoint directory.

        A model that silently loads *something* from an incomplete directory is
        worse than one that refuses to start -- the same principle
        :meth:`load` applies to a missing dependency, applied to a missing file.
        """
        path = Path(path)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"no checkpoint directory at {path}")
        missing = [f for f in _CHECKPOINT_REQUIRED_FILES if not (path / f).exists()]
        has_weights = any(path.glob("model.safetensors")) or any(path.glob("*.safetensors"))
        if not has_weights:
            missing.append("*.safetensors")
        if missing:
            raise ValueError(
                f"checkpoint at {path} is incomplete or not one saved by "
                f"SmolVLAPolicy.save_checkpoint(); missing: {missing}"
            )
        meta = json.loads((path / "rde_checkpoint_meta.json").read_text())
        return meta

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None) -> "SmolVLAPolicy":
        """Reload a checkpoint saved by :meth:`save_checkpoint`, verified first."""
        meta = cls.verify_checkpoint(path)
        policy = cls(checkpoint=str(path), device=device or meta.get("device"))
        policy.load()
        policy._checkpoint_meta = meta
        return policy

    def train_mode(self) -> None:
        """Switch the underlying torch module to train mode (dropout/BN active).

        Exposed so a caller doing periodic validation *during* :meth:`fit`
        (e.g. the training orchestrator's checkpoint/validate loop) can flip to
        eval mode for the validation pass and back, without reaching into the
        private ``_policy`` attribute.
        """
        if self._policy is not None:
            self._policy.train()

    def eval_mode(self) -> None:
        """Switch the underlying torch module to eval mode. See :meth:`train_mode`."""
        if self._policy is not None:
            self._policy.eval()

    def describe(self) -> dict[str, Any]:
        """Checkpoint, device and whether weights are loaded."""
        return {"model": self.name, "type": "vla", "checkpoint": self.checkpoint,
                "device": self.device, "loaded": self._policy is not None}


#: Name -> (model factory, adapter class).  The registry is what lets the CLI and
#: the evaluation harness swap models without knowing their construction.
MODEL_REGISTRY: dict[str, tuple[Any, type[ModelInputAdapter]]] = {
    "mean_action": (MeanActionPolicy, StateAdapter),
    "nearest_neighbor": (NearestNeighborPolicy, StateAdapter),
    "smolvla": (SmolVLAPolicy, VisionLanguageAdapter),
}


def build_model(name: str, contract: dict[str, Any], **kwargs) -> tuple[Model, ModelInputAdapter]:
    """Instantiate a model and its paired adapter by name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")
    model_cls, adapter_cls = MODEL_REGISTRY[name]
    return model_cls(**kwargs), adapter_cls(contract)
