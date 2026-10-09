"""Model adapter layer: the boundary between the data engine and any model.

Adapters depend on the engine; the engine never depends on a model.
"""

from .adapters import StateAdapter, VisionLanguageAdapter, VisionStateAdapter
from .base import Model, ModelInput, ModelInputAdapter, Prediction
from .models import (
    MODEL_REGISTRY,
    MeanActionPolicy,
    NearestNeighborPolicy,
    SmolVLAPolicy,
    build_model,
)

__all__ = [
    "MODEL_REGISTRY",
    "MeanActionPolicy",
    "Model",
    "ModelInput",
    "ModelInputAdapter",
    "NearestNeighborPolicy",
    "Prediction",
    "SmolVLAPolicy",
    "StateAdapter",
    "VisionLanguageAdapter",
    "VisionStateAdapter",
    "build_model",
]
