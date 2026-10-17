"""Hybrid retrieval: BM25 and dense search fused with reciprocal rank fusion.

BM25 over English-stemmed tokens (character bigrams for Chinese, Japanese and Korean) and
dense cosine search each give a top-N. These are fused with reciprocal rank fusion (k=60),
then optionally re-scored by a cross-encoder over the fused candidates. Pure functions over
arrays: no database and no model loading in this module.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .models import Embedder, Reranker

# Han, Hiragana, Katakana and Hangul letters as code point ranges; punctuation such as the katakana
# middle dot is left out, so a run never contains a character the tokeniser would split on.
_CJK_RANGES = (
    (0x3005, 0x3007), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x3134F),  # Han
    (0x3041, 0x3096), (0x309D, 0x309F),  # Hiragana
    (0x30A1, 0x30FA), (0x30FC, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F),  # Katakana, half-width included
    (0x1100, 0x11FF), (0x3131, 0x318E), (0xA960, 0xA97F), (0xAC00, 0xD7A3), (0xD7B0, 0xD7FF),  # Hangul
)
_CJK = "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _CJK_RANGES)
_CJK_RUN = re.compile(f"[{_CJK}]+")
# bm25s's default token pattern, plus a lone CJK character (the default drops one-character tokens).
_TOKEN_PATTERN = rf"(?u)\b\w\w+\b|[{_CJK}]"

_UNBUILT = object()  # BM25 index not built yet; None means built, with nothing to match


def _bigrams(run: re.Match[str]) -> str:
    chars = run.group()
    return " " + " ".join([chars[i : i + 2] for i in range(len(chars) - 1)] or [chars]) + " "


def _unigrams_and_bigrams(run: re.Match[str]) -> str:
    chars = run.group()
    return " " + " ".join(chars) + _bigrams(run) if len(chars) > 1 else _bigrams(run)


def _segment_cjk(text: str, unigrams: bool = False) -> str:
    """Rewrite every run of CJK characters as space-separated overlapping bigrams.

    Chinese and Japanese are written without spaces and Korean words carry attached particles, so
    the default tokeniser turns a whole sentence into one token that no query can match. A run of n
    characters becomes its n - 1 bigrams, and a lone character stays as it is. With ``unigrams``
    (for the documents) every character is added as well, so a one-character query such as 猫 finds
    them. Text without CJK characters is returned unchanged, so English and other space-separated
    scripts tokenise as before.
    """
    if text.isascii():  # O(1)
        return text
    return _CJK_RUN.sub(_unigrams_and_bigrams if unigrams else _bigrams, text)


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
    """An in-memory corpus: texts plus an (n, dim) matrix of normalised vectors.

    Searches may overlap on several threads. The BM25 index is built on first use, by one
    thread, and is visible to the others only once complete; everything else is read-only.
    """

    def __init__(self, texts: Sequence[str], vectors: np.ndarray | None):
        self.texts = list(texts)
        self.vectors = vectors
        self._bm25: object = _UNBUILT
        self._build_lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.texts)

    def _tokenize(self, docs: list[str], query: bool = False):
        import bm25s
        import Stemmer

        # A PyStemmer instance is stateful and must not be used by two threads at once, so
        # every call makes its own (construction is free).
        return bm25s.tokenize(
            [_segment_cjk(d, unigrams=not query) for d in docs],
            token_pattern=_TOKEN_PATTERN,
            stopwords="en",
            stemmer=Stemmer.Stemmer("english"),
            show_progress=False,
        )

    def _lexicon(self):
        """The BM25 index, built once; None if no chunk contains a token."""
        bm25 = self._bm25
        if bm25 is _UNBUILT:
            with self._build_lock:
                bm25 = self._bm25
                if bm25 is _UNBUILT:
                    bm25 = self._build_bm25()
                    self._bm25 = bm25  # published only once complete
        return bm25

    def _build_bm25(self):
        import bm25s

        tokens = self._tokenize(self.texts)
        if not tokens.vocab:  # only emoji, one-letter words or stopwords: bm25s cannot index that
            return None
        bm25 = bm25s.BM25()
        bm25.index(tokens, show_progress=False)
        return bm25

    def lexical(self, query: str, k: int) -> list[int]:
        """Positions of the top-k BM25 matches with a positive score."""
        if not self.texts or k <= 0:
            return []
        bm25 = self._lexicon()
        if bm25 is None:
            return []
        try:
            res, scores = bm25.retrieve(
                self._tokenize([query], query=True), k=min(k, len(self.texts)), show_progress=False
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
