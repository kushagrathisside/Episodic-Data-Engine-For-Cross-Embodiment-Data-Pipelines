"""Episodic Data Engine.

A modular pipeline that turns raw multimodal robot demonstrations into validated,
synchronised, versioned, training-ready datasets -- and exposes them through one
stable interface that different models can consume without the data layer
changing.

Typical use::

    from rde import DataEngine

    engine = DataEngine()
    ds = engine.get_dataset("so100_pickplace", version="training/v1")
"""

from .config import SOFTWARE_VERSION, Settings, get_settings, set_settings
from .engine import DataEngine, DatasetView, Sample

__version__ = SOFTWARE_VERSION

__all__ = [
    "DataEngine",
    "DatasetView",
    "Sample",
    "Settings",
    "get_settings",
    "set_settings",
    "__version__",
]
