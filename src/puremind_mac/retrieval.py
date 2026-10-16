"""MAC-16b hybrid retrieval.

BM25 (English stemmer) top-N and dense top-N, fused with reciprocal rank fusion
(k=60), then optionally re-scored by a cross-encoder over the fused candidates.
Pure functions over arrays: no database and no model loading in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .models import Embedder, Reranker


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion. Ranks are 1-based; ties keep first-seen order."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


@dataclass(frozen=True)
class Scored:
    position: int
    score: float


class HybridIndex:
    """An in-memory corpus: texts plus an (n, dim) matrix of normalised vectors."""

    def __init__(self, texts: Sequence[str], vectors: np.ndarray | None):
        self.texts = list(texts)
        self.vectors = vectors
        self._bm25 = None
        self._stemmer = None

    def __len__(self) -> int:
        return len(self.texts)

    def _tokenize(self, docs: list[str]):
        import bm25s
        import Stemmer

        if self._stemmer is None:
            self._stemmer = Stemmer.Stemmer("english")
        return bm25s.tokenize(docs, stopwords="en", stemmer=self._stemmer, show_progress=False)

    def lexical(self, query: str, k: int) -> list[int]:
        """Positions of the top-k BM25 matches with a positive score."""
        if not self.texts or k <= 0:
            return []
        import bm25s

        if self._bm25 is None:
            self._bm25 = bm25s.BM25()
            self._bm25.index(self._tokenize(self.texts), show_progress=False)
        try:
            res, scores = self._bm25.retrieve(
                self._tokenize([query]), k=min(k, len(self.texts)), show_progress=False
            )
        except (ValueError, IndexError):  # query reduced to nothing (all stopwords)
            return []
        return [int(j) for j, s in zip(res[0], scores[0]) if s > 0]

    def dense(self, qvec: np.ndarray, k: int) -> list[int]:
        if self.vectors is None or len(self.vectors) == 0 or k <= 0:
            return []
        sims = self.vectors @ qvec
        return [int(j) for j in np.argsort(-sims, kind="stable")[:k]]

    def search(
        self,
        query: str,
        embedder: Embedder,
        reranker: Reranker | None,
        limit: int = 10,
        candidates: int = 50,
        rrf_k: int = 60,
        rerank_chars: int = 800,
    ) -> list[Scored]:
        lex = self.lexical(query, candidates)
        dense = self.dense(embedder.embed_query(query), candidates) if self.vectors is not None else []
        fused = rrf_fuse([lex, dense], rrf_k)[:candidates]
        if not fused:
            return []
        if reranker is None:
            return [Scored(pos, score) for pos, score in fused[:limit]]
        passages = [self.texts[pos][:rerank_chars] for pos, _ in fused]
        scores = reranker.score(query, passages)
        ranked = sorted(zip((pos for pos, _ in fused), scores), key=lambda t: t[1], reverse=True)
        return [Scored(pos, float(s)) for pos, s in ranked[:limit]]
