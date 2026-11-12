"""Pluggable local and OpenAI-compatible text embedding providers."""

import os
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider contract used by embedding text features."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a two-dimensional vector matrix, one row per text."""


_PROVIDER_REGISTRY: dict[str, type | Callable[..., EmbeddingProvider]] = {}


def register_embedding_provider(name: str) -> Callable[[type], type]:
    """Register a custom embedding provider class."""
    if not name or not name.strip():
        raise ValueError("embedding provider name must not be empty")

    def decorator(provider_class: type) -> type:
        if name in _PROVIDER_REGISTRY:
            raise ValueError(f"embedding provider {name!r} is already registered")
        _PROVIDER_REGISTRY[name] = provider_class
        return provider_class

    return decorator


def build_embedding_provider(name: str, **kwargs: Any) -> EmbeddingProvider:
    """Create an embedding provider, loading optional integrations lazily."""
    try:
        provider_class = _PROVIDER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown embedding provider {name!r}") from exc
    return provider_class(**kwargs)


@register_embedding_provider("local")
class LocalSentenceTransformerProvider:
    """Compute vectors with a locally loaded Sentence Transformers model."""

    def __init__(self, model: str, device: str | None = None, batch_size: int = 32) -> None:
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "local embeddings require the optional 'embeddings' dependencies; "
                    "install with `pip install open-banking-core[embeddings]`"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension_ = int(self._model.get_sentence_embedding_dimension())
        vectors = self._model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


class _OpenAICompatibleProvider:
    """Shared implementation for OpenAI and vLLM's OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        api_key_env: str | None = "OPENAI_API_KEY",
        base_url: str | None = None,
        batch_size: int = 32,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.batch_size = batch_size
        self.timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "remote embeddings require the optional 'remote-embeddings' dependency; "
                    "install with `pip install open-banking-core[remote-embeddings]`"
                ) from exc
            api_key = os.getenv(self.api_key_env) if self.api_key_env else "not-needed"
            if self.api_key_env and not api_key:
                raise ValueError(f"embedding API key environment variable {self.api_key_env!r} is not set")
            options: dict[str, Any] = {"api_key": api_key, "timeout": self.timeout}
            if self.base_url:
                options["base_url"] = self.base_url
            self._client = OpenAI(**options)
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        client = self._get_client()
        nonempty_positions = [index for index, text in enumerate(texts) if text.strip()]
        model_inputs = [texts[index] for index in nonempty_positions]
        if not model_inputs:
            model_inputs = ["[empty]"]
        vectors = []
        for start in range(0, len(model_inputs), self.batch_size):
            response = client.embeddings.create(model=self.model, input=model_inputs[start : start + self.batch_size])
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        vectors_array = np.asarray(vectors, dtype=np.float32)
        self.dimension_ = vectors_array.shape[1]
        if not nonempty_positions:
            return np.zeros((len(texts), self.dimension_), dtype=np.float32)
        result = np.zeros((len(texts), self.dimension_), dtype=np.float32)
        result[nonempty_positions] = vectors_array
        return result


@register_embedding_provider("openai")
class OpenAIEmbeddingProvider(_OpenAICompatibleProvider):
    """Compute vectors through the OpenAI embeddings endpoint."""


@register_embedding_provider("vllm")
class VLLMEmbeddingProvider(_OpenAICompatibleProvider):
    """Compute vectors through a vLLM OpenAI-compatible embeddings endpoint."""

    def __init__(self, model: str, api_key_env: str | None = "VLLM_API_KEY", **kwargs: Any) -> None:
        super().__init__(model=model, api_key_env=api_key_env, **kwargs)
