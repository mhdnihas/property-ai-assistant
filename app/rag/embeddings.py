"""Thin wrapper around fastembed (ONNX, CPU-only, no API key, no torch)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    """Produces L2-normalised float32 embeddings so dot product == cosine similarity."""

    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding  # imported lazily: heavy, optional at test time

        self.model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
        )

    def embed_documents(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        vecs = np.asarray(
            list(self._model.passage_embed(list(texts), batch_size=batch_size)), dtype=np.float32
        )
        return _normalise(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)
        return _normalise(vec[None, :])[0]


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms
