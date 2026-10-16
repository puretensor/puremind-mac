"""Embedder and reranker interfaces plus the sentence-transformers implementations.

Everything else in the package depends only on the two Protocols, so tests and
alternative backends can inject their own. torch is imported lazily, on first use.
"""

from __future__ import annotations

import threading
from typing import Protocol, Sequence

import numpy as np

from .config import QUERY_INSTRUCTION


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""

    def embed_query(self, query: str) -> np.ndarray:
        """Return a (dim,) float32 L2-normalised vector."""


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage; higher is better."""


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


class SentenceTransformerEmbedder:
    """Qwen3-Embedding (or any sentence-transformers model) with the query instruction."""

    def __init__(self, model: str, device: str = "auto", max_seq: int = 1024, batch_size: int = 16):
        self.name = model
        self._device = device
        self._max_seq = max_seq
        self._batch = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                m = SentenceTransformer(self.name, device=pick_device(self._device))
                m.max_seq_length = self._max_seq
                self._model = m
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = self._load().encode(
            list(texts), batch_size=self._batch, normalize_embeddings=True, show_progress_bar=False
        )
        return _normalise(out)

    def embed_query(self, query: str) -> np.ndarray:
        out = self._load().encode([QUERY_INSTRUCTION + query], normalize_embeddings=True, show_progress_bar=False)
        return _normalise(out)[0]


class CrossEncoderReranker:
    """BGE reranker via sentence_transformers.CrossEncoder (max_length 512)."""

    def __init__(self, model: str, device: str = "auto", query_chars: int = 1000):
        self.name = model
        self._device = device
        self._query_chars = query_chars
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.name, max_length=512, device=pick_device(self._device))
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        q = query[: self._query_chars]
        scores = self._load().predict([(q, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in np.asarray(scores).reshape(-1)]
