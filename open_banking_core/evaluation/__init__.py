"""Metrics, evaluation engine, and structured report."""

from .evaluator import Evaluator
from .metrics import compute_metrics, register_metric, registered_metrics
from .report import EvaluationReport

__all__ = ["EvaluationReport", "Evaluator", "compute_metrics", "register_metric", "registered_metrics"]
