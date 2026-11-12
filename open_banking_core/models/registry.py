"""Decorator-based registry for scikit-learn-compatible classifiers."""

from collections.abc import Callable
from typing import Any

from sklearn.base import BaseEstimator

from open_banking_core.config import ModelSpec

ModelClass = type[BaseEstimator]
_MODEL_REGISTRY: dict[str, ModelClass] = {}


def register_model(name: str) -> Callable[[ModelClass], ModelClass]:
    """Register a classifier class under a configuration name."""
    if not name or not name.strip():
        raise ValueError("model name must not be empty")

    def decorator(model_class: ModelClass) -> ModelClass:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        _MODEL_REGISTRY[name] = model_class
        return model_class

    return decorator


def build_model(spec: ModelSpec | dict[str, Any]) -> BaseEstimator:
    """Instantiate a registered classifier from a model spec or mapping."""
    parsed = spec if isinstance(spec, ModelSpec) else ModelSpec.model_validate(spec)
    try:
        model_class = _MODEL_REGISTRY[parsed.type]
    except KeyError as exc:
        raise ValueError(f"unknown model type {parsed.type!r}") from exc
    return model_class(**parsed.params)


def registered_models() -> tuple[str, ...]:
    """Return registered model names in registration order."""
    return tuple(_MODEL_REGISTRY)
