"""Dataset builder: clean version -> training-ready version with a contract."""

from .builder import BuildResult, DatasetBuilder, build_training_contract

__all__ = ["BuildResult", "DatasetBuilder", "build_training_contract"]
