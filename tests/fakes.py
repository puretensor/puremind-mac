"""Stand-ins for the two models: deterministic, instant, no downloads.

Kept apart from conftest.py so that helper scripts run in a child process can import them
without the path set-up conftest does for the test process itself.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

DIM = 64


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeEmbedder:
    """Hashed bag of words: deterministic, no model, similar texts get similar vectors."""

    name = "fake-embedder"

    def __init__(self):
        self.doc_calls = 0
        self.texts_embedded = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for t in _tokens(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        self.doc_calls += 1
        self.texts_embedded += len(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, DIM), dtype=np.float32)

    def embed_query(self, query):
        return self._vec(query)


class FakeReranker:
    """Scores by query-token overlap; records what it was asked to score."""

    name = "fake-reranker"

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        q = set(_tokens(query))
        return [float(len(q & set(_tokens(p)))) for p in passages]
