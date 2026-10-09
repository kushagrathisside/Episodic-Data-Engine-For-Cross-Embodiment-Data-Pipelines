"""Evaluation: score a model against a dataset version, emit per-frame outcomes."""

from .evaluator import EvaluationResult, Evaluator, FrameOutcome

__all__ = ["EvaluationResult", "Evaluator", "FrameOutcome"]
