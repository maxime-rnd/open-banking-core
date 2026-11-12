"""Built-in and user-registered transaction feature transformers."""

# Import built-ins so decorators register them when this package is imported.
from .amount import AmountFeatures
from .base import FeatureTransformer
from .date import DateFeatures
from .embeddings import EmbeddingProvider, build_embedding_provider, register_embedding_provider
from .regex import RegexFeatures
from .registry import build_feature, register_feature, registered_features
from .text import TextFeature, clean_text

__all__ = [
    "AmountFeatures",
    "DateFeatures",
    "EmbeddingProvider",
    "FeatureTransformer",
    "RegexFeatures",
    "TextFeature",
    "build_embedding_provider",
    "build_feature",
    "clean_text",
    "register_embedding_provider",
    "register_feature",
    "registered_features",
]
