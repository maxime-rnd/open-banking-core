"""Decorator-based registry for project and core feature transformers."""

from collections.abc import Callable
from typing import Any

from open_banking_core.config import FeatureSpec
from .base import FeatureTransformer

FeatureClass = type
_FEATURE_REGISTRY: dict[str, FeatureClass] = {}


def register_feature(name: str) -> Callable[[FeatureClass], FeatureClass]:
    """Register a transformer class under a stable configuration name."""
    if not name or not name.strip():
        raise ValueError("feature name must not be empty")

    def decorator(feature_class: FeatureClass) -> FeatureClass:
        if name in _FEATURE_REGISTRY:
            raise ValueError(f"feature {name!r} is already registered")
        _FEATURE_REGISTRY[name] = feature_class
        return feature_class

    return decorator


def build_feature(spec: FeatureSpec | dict[str, Any]) -> FeatureTransformer:
    """Instantiate a registered transformer from a feature spec or mapping."""
    parsed = spec if isinstance(spec, FeatureSpec) else FeatureSpec.model_validate(spec)
    try:
        feature_class = _FEATURE_REGISTRY[parsed.type]
    except KeyError as exc:
        raise ValueError(f"unknown feature type {parsed.type!r}") from exc
    return feature_class(**parsed.params)


def registered_features() -> tuple[str, ...]:
    """Return registered feature names in registration order."""
    return tuple(_FEATURE_REGISTRY)
