"""Registry and built-in classification metrics."""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

Metric = Callable[[Any, Any], float]
_METRIC_REGISTRY: dict[str, Metric] = {}


def register_metric(name: str) -> Callable[[Metric], Metric]:
    """Register a metric function accepting ``(y_true, y_pred)``."""
    if not name or not name.strip():
        raise ValueError("metric name must not be empty")

    def decorator(metric: Metric) -> Metric:
        if name in _METRIC_REGISTRY:
            raise ValueError(f"metric {name!r} is already registered")
        _METRIC_REGISTRY[name] = metric
        return metric

    return decorator


@register_metric("accuracy")
def accuracy(y_true: Any, y_pred: Any) -> float:
    """Return classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


@register_metric("macro_f1")
def macro_f1(y_true: Any, y_pred: Any) -> float:
    """Return unweighted mean F1 across labels present in truth or prediction."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


@register_metric("balanced_accuracy")
def balanced_accuracy(y_true: Any, y_pred: Any) -> float:
    """Return mean per-class recall."""
    return float(balanced_accuracy_score(y_true, y_pred))


@register_metric("weighted_f1")
def weighted_f1(y_true: Any, y_pred: Any) -> float:
    """Return support-weighted F1 across labels."""
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


@register_metric("macro_precision")
def macro_precision(y_true: Any, y_pred: Any) -> float:
    """Return unweighted mean precision across labels."""
    return float(precision_score(y_true, y_pred, average="macro", zero_division=0))


@register_metric("weighted_precision")
def weighted_precision(y_true: Any, y_pred: Any) -> float:
    """Return support-weighted precision across labels."""
    return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))


@register_metric("macro_recall")
def macro_recall(y_true: Any, y_pred: Any) -> float:
    """Return unweighted mean recall across labels."""
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


@register_metric("weighted_recall")
def weighted_recall(y_true: Any, y_pred: Any) -> float:
    """Return support-weighted recall across labels."""
    return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))


def compute_metrics(names: list[str], y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute registered metrics, rejecting unknown names clearly."""
    output: dict[str, float] = {}
    for name in names:
        try:
            metric = _METRIC_REGISTRY[name]
        except KeyError as exc:
            raise ValueError(f"unknown evaluation metric {name!r}") from exc
        output[name] = metric(y_true, y_pred)
    return output


def registered_metrics() -> tuple[str, ...]:
    """Return registered metric names in registration order."""
    return tuple(_METRIC_REGISTRY)
