"""Configurable text cleaning and TF-IDF, embedding, or hybrid features."""

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import FeatureTransformer
from .embeddings import EmbeddingProvider, build_embedding_provider
from .registry import register_feature


def clean_text(value: object, config: dict[str, Any] | None = None) -> str:
    """Normalize one text value using configurable, conservative cleanup."""
    options = config or {}
    text = "" if pd.isna(value) else str(value)
    normalization = options.get("unicode_normalization", "NFKC")
    if normalization:
        text = unicodedata.normalize(normalization, text)
    if options.get("lowercase", True):
        text = text.lower()
    if options.get("strip", True):
        text = text.strip()
    if options.get("collapse_whitespace", True):
        text = re.sub(r"\s+", " ", text)
    for pattern, replacement in options.get("substitutions", {}).items():
        text = re.sub(pattern, replacement, text)
    return text


@register_feature("text")
class TextFeature(FeatureTransformer):
    """Transform a DataFrame text column with a selected representation."""

    def __init__(
        self,
        column: str = "description",
        clean: dict[str, Any] | None = None,
        representation: dict[str, Any] | None = None,
    ) -> None:
        self.column = column
        self.clean = clean
        self.representation = representation

    def _options(self) -> dict[str, Any]:
        if not self.representation or "type" not in self.representation:
            raise ValueError("text representation must specify type: tfidf, embedding, or hybrid")
        return self.representation

    def _clean_values(self, X: pd.DataFrame) -> list[str]:
        if self.column not in X:
            raise ValueError(f"text column {self.column!r} is missing")
        return [clean_text(value, self.clean) for value in X[self.column].tolist()]

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TextFeature":
        options = self._options()
        kind = options["type"]
        text = self._clean_values(X)
        self.vectorizer_ = None
        self.embedding_provider_: EmbeddingProvider | None = None
        if kind in {"tfidf", "hybrid"}:
            tfidf_params = dict(options.get("tfidf", {}))
            if isinstance(tfidf_params.get("ngram_range"), list):
                tfidf_params["ngram_range"] = tuple(tfidf_params["ngram_range"])
            self.vectorizer_ = TfidfVectorizer(**tfidf_params)
            self.vectorizer_.fit(text)
        if kind == "embedding":
            provider_options = options.get("embedding", {})
            provider_name = provider_options.get("provider")
            if not provider_name:
                raise ValueError("embedding representation must specify a registered provider")
            self.embedding_provider_ = build_embedding_provider(
                provider_name,
                **{key: value for key, value in provider_options.items() if key != "provider"},
            )
        elif kind == "hybrid":
            provider_options = options.get("embedding", {})
            provider_name = provider_options.get("provider")
            if not provider_name:
                raise ValueError("hybrid representation must specify an embedding provider")
            self.embedding_provider_ = build_embedding_provider(
                provider_name,
                **{key: value for key, value in provider_options.items() if key != "provider"},
            )
        elif kind != "tfidf":
            raise ValueError(f"unsupported text representation {kind!r}")
        return self

    def transform_components(self, X: pd.DataFrame) -> dict[str, Any]:
        """Return named text representation blocks for evaluation diagnostics."""
        self._options()
        text = self._clean_values(X)
        tfidf_matrix = self.vectorizer_.transform(text) if self.vectorizer_ is not None else None
        embedding_matrix = None
        if self.embedding_provider_ is not None:
            embedding_matrix = np.asarray(self.embedding_provider_.embed(text), dtype=np.float32)
            if embedding_matrix.ndim != 2 or embedding_matrix.shape[0] != len(text):
                raise ValueError("embedding provider must return a 2D matrix with one row per input")
            self.embedding_dimension_ = embedding_matrix.shape[1]
        components: dict[str, Any] = {}
        if tfidf_matrix is not None:
            components["tfidf"] = tfidf_matrix
        if embedding_matrix is not None:
            components["embedding"] = embedding_matrix
        return components

    def transform(self, X: pd.DataFrame) -> Any:
        components = self.transform_components(X)
        matrices = list(components.values())
        if len(matrices) == 1:
            return matrices[0]
        if not matrices:
            return np.empty((len(X), 0), dtype=np.float32)
        return sparse.hstack(
            [matrix if sparse.issparse(matrix) else sparse.csr_matrix(matrix) for matrix in matrices],
            format="csr",
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        options = self._options()
        if options["type"] == "tfidf":
            return self.vectorizer_.get_feature_names_out()
        if options["type"] == "embedding":
            return np.asarray([f"embedding_{index}" for index in range(self.embedding_dimension_)])
        tfidf_names = self.vectorizer_.get_feature_names_out()
        embedding_dim = getattr(self, "embedding_dimension_", None)
        if embedding_dim is None:
            raise ValueError("embedding provider must expose dimension_ after embedding")
        return np.concatenate((tfidf_names, [f"embedding_{i}" for i in range(embedding_dim)]))
