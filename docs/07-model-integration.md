# 07 — Model Integration

## 1. The Boundary

A prediction is produced by passing data through four stages:

```
DatasetView  →  ModelInputAdapter  →  Model  →  Prediction  →  Evaluation
```

Adapters depend on the engine, and the engine never depends on a model. None of `ingestion/`, `storage/`, `catalog/`, `validation/`, `qc/`, `transformations/`, or `versioning/` contains any reference to a model. Integrating a new model therefore requires only a new file in `rde/model_adapters/` and a corresponding registry entry.

## 2. Training Consumes a Version, Not Files

```python
from rde import DataEngine

ds = DataEngine().get_dataset("so100_pickplace", version="training/v1")
```

A dataset is always requested by version identifier, never by file path. This design keeps the physical storage layout free to change independently of training code, and it is what makes a training run reproducible from its recorded version identifier alone.

## 3. The Training Contract

The dataset builder emits a contract that adapters read, rather than rediscovering the schema themselves:

```json
{
  "observation": {
    "state.joint": {"kind": "robot_state", "shape": [6], "plane": "inline",
                    "element_names": ["main_shoulder_pan", "..."],
                    "normalization": {"scheme": "mean_std", "mean": [...], "std": [...]}},
    "rgb.top":     {"kind": "rgb", "shape": [224, 224, 3], "plane": "external",
                    "read_transforms": [{"op": "resize", "height": 224, "width": 224}]}
  },
  "action": {"key": "action.joint", "shape": [6],
             "normalization": {"scheme": "mean_std", "...": "..."}},
  "tasks": [{"task_id": 0, "description": "Pick up the cube and place it in the box."}],
  "language_conditioned": true
}
```

No adapter contains a literal field name such as `"observation.state"` or `"rgb.top"`. As a result, a bimanual robot exposing `rgb.left` and `rgb.right` instead works with the existing adapters unmodified.

### 3.1 Two Concepts Named "Contract"

The term "contract" is used for two unrelated concepts within this project, distinguished in Table 1.

**Table 1. The two meanings of "contract."**

| Contract type | Produced by | Question it answers | Consumed by |
|---|---|---|---|
| Training contract | `dataset_builder` | What shape is this data, and how should it be read? | Model adapters |
| Data health contract | `observability/contracts.py` | May this version be released at all? | The release decision |

A training contract is a description: it never blocks anything. A data health contract is a policy: it can refuse to release a version whose action distribution has drifted, even when every structural check has passed and the training contract describes the version perfectly. See [05-qc-architecture.md](05-qc-architecture.md) and [14-observability-architecture.md](14-observability-architecture.md) for further detail on each.

## 4. Lazy Images

A `Sample` carries image references rather than decoded pixel data:

```python
sample.observation["state.joint"]   # (6,) — materialised
sample.image_refs                   # {"rgb.top": (blob_id, 44.733)} — not decoded
img = sample.image("rgb.top")       # decoded now, resized per contract
```

Measured directly, 500 samples without image access take 5 ms; with image decoding, the cost is 7.1 ms per sample. A state-only policy therefore pays no cost at all for the presence of two camera streams in the dataset.

## 5. Deferred Transforms

The transform `ResizeImages` does not re-encode video. It instead records its intent in the manifest, and the model adapter applies the resize during decode. Materializing the resize immediately would cost minutes of CPU time and hundreds of megabytes of storage to save microseconds per frame at read time, and it would bake a single resolution into storage permanently — so that a second model requiring 256×256 images would need a second copy of the entire dataset. Lineage records the operator identically regardless of when it is materialized; only the point of materialization differs.

## 6. The Three Adapters

Table 2 lists the three model input adapters currently implemented.

**Table 2. Model input adapters.**

| Adapter | Consumes | Intended for |
|---|---|---|
| `StateAdapter` | Proprioception only | Baselines and state-space policies |
| `VisionStateAdapter` | Images and state | Visuomotor policies |
| `VisionLanguageAdapter` | Images, state, and instruction | Vision-language-action models |

`VisionStateAdapter` emits image tensors in NCHW layout by default, and in NHWC layout on request, since not every consuming framework is PyTorch.

## 7. The Three Models

Table 3 lists the three model adapters implemented and their measured performance.

**Table 3. Model adapters and measured performance.**

| Model | Type | Consumes | MSE | Relative to baseline |
|---|---|---|---:|---:|
| `MeanActionPolicy` | Baseline | Nothing | 0.90799 | 1.000 |
| `NearestNeighborPolicy` | Retrieval | State | 0.02385 | 0.026 |
| `SmolVLAPolicy` | Vision-language-action | Images, state, and language | Optional; not evaluated here | — |

The baseline is included as a matter of methodological necessity rather than decoration. An absolute MSE value computed on normalized actions conveys almost no information on its own; a model that cannot outperform the training-set mean has learned nothing, and it is surprisingly easy to report an MSE that sounds impressive while the mean baseline achieves the same figure. In this evaluation, the k-nearest-neighbor model reaches 2.6% of the baseline's MSE, a genuine 38-fold improvement. This result is consistent with expectation: on single-task demonstration data, the state trajectories of repeated attempts overlap heavily, so retrieval constitutes a genuinely strong baseline. That strength is precisely the point of including it — it forces a learned policy to demonstrate that it is doing more than retrieval already accomplishes.

### 7.1 Confidence Is Required, Not Optional

The field `Prediction.confidence` is mandatory because the feedback loop selects low-confidence examples for review. A model that cannot express uncertainty about its own predictions cannot participate in the feedback flywheel.

The k-nearest-neighbor model derives its confidence from retrieval distance, calibrated against the distribution of distances observed at fit time. Measured on the test split, the correlation between confidence and error is −0.79: when the model is confident, it tends to be correct, and when it is uncertain, it tends to be wrong. This correlation is what makes the flywheel's low-confidence signal meaningful rather than decorative.

`SmolVLAPolicy` exposes no calibrated uncertainty estimate. It therefore returns a constant confidence value and states this explicitly in `metadata.confidence_note`, rather than inventing a number that the feedback loop would otherwise treat as trustworthy.

## 8. SmolVLA

The SmolVLA adapter is a working adapter for the real pretrained checkpoint, implemented to remain import-light: `torch` and `lerobot` are imported only inside its `load()` method, so that the engine and its 180 tests can run without pulling in a multi-gigabyte dependency tree.

```bash
uv pip install 'torch' 'lerobot[smolvla]'     # the `vla` extra
```

If these dependencies are unavailable, the adapter raises an exception with an actionable message rather than degrading silently. A model that quietly becomes a stub is a worse failure mode than one that refuses to start, and this behavior is covered by a dedicated test.

No SmolVLA training or evaluation was performed as part of this project. What is demonstrated is narrower and more specific: integrating SmolVLA required no change to ingestion, storage, QC, versioning, or dataset management. The adapter reads the same training contract and returns the same `Prediction` type as the two baseline models.

## 9. Adding a Model

A new model is integrated by implementing the `Model` interface and registering it, as shown below.

```python
class MyPolicy(Model):
    name = "my_policy"
    adapter_class = VisionLanguageAdapter

    def fit(self, view, split="train"): ...
    def predict(self, model_input) -> Prediction: ...

MODEL_REGISTRY["my_policy"] = (MyPolicy, VisionLanguageAdapter)
```

This is the entire integration surface required. The new model can then be evaluated with the standard command:

```bash
rde model evaluate so100_pickplace training/v1 --model my_policy
```
