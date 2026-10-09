"""Demonstrate that models are interchangeable behind one data interface.

Three models with genuinely different appetites -- one ignores its input, one
retrieves from proprioceptive state, one would consume images plus language --
are driven by identical code.  Nothing in ingestion, storage, QC, versioning or
the dataset builder knows any of them exist.
"""

from __future__ import annotations

import json

from rde.engine import DataEngine
from rde.evaluation import Evaluator
from rde.model_adapters import MODEL_REGISTRY, build_model

DATASET = "so100_pickplace"
VERSION = "training/v1"


def main():
    engine = DataEngine()
    view = engine.get_dataset(DATASET, VERSION)

    print("=" * 78)
    print(f"DATA INTERFACE: get_dataset({DATASET!r}, version={VERSION!r})")
    print("=" * 78)
    contract = view.contract
    print(f"  observations : {list(contract['observation'])}")
    print(f"  action       : {contract['action']['key']} {contract['action']['shape']}")
    print(f"  language     : {contract['language_conditioned']}")
    print(f"  splits       : {view.splits()}")

    print("\n" + "=" * 78)
    print("THE SAME LOOP, DIFFERENT MODELS")
    print("=" * 78)

    ev = Evaluator()
    results = {}
    baseline = None

    for name in ("mean_action", "nearest_neighbor"):
        model, adapter = build_model(name, contract)
        model.fit(view, split="train")                    # identical
        res = ev.evaluate(view, model, adapter, split="test", persist=False)
        results[name] = res
        if baseline is None:
            baseline = res

    for name, res in results.items():
        if res is not baseline:
            ev.compare(res, baseline)

    print(f"{'model':20s} {'adapter':16s} {'consumes':22s} {'MSE':>9} {'MAE':>9} {'vs base':>8}")
    print("-" * 78)
    for name, res in results.items():
        _, adapter = build_model(name, contract)
        consumes = ",".join(adapter.describe()["consumes"]["state_keys"]) or "nothing"
        ratio = res.baseline["mse_ratio"] if res.baseline else 1.0
        print(f"{name:20s} {res.adapter_name:16s} {consumes:22s} "
              f"{res.metrics['mse']:>9.5f} {res.metrics['mae']:>9.5f} {ratio:>8.3f}")

    print("\n" + "=" * 78)
    print("A VISION-LANGUAGE MODEL USES THE SAME INTERFACE")
    print("=" * 78)
    from rde.model_adapters import VisionLanguageAdapter

    vla_adapter = VisionLanguageAdapter(contract)
    batch = []
    for s in view.iter_samples(split="test"):
        batch.append(s)
        if len(batch) >= 4:
            break
    mi = vla_adapter.adapt(batch)
    print(f"  adapter      : {vla_adapter.name}")
    print(f"  instruction  : {mi.instruction!r}")
    for key, arr in sorted(mi.tensors.items()):
        print(f"  {key:24s} {tuple(arr.shape)} {arr.dtype}")
    print("\n  SmolVLA consumes exactly this. Its registry entry:")
    model_cls, adapter_cls = MODEL_REGISTRY["smolvla"]
    print(f"    {model_cls.__name__} + {adapter_cls.__name__}")
    print("  Adding it required no change to ingestion, storage, QC or versioning.")

    view.close()


if __name__ == "__main__":
    main()
