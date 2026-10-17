"""Hybrid retrieval: BM25 and dense search fused with reciprocal rank fusion.

BM25 over English-stemmed tokens (character bigrams for Chinese, Japanese and Korean) and
dense cosine search each give a top-N. These are fused with reciprocal rank fusion (k=60),
then optionally re-scored by a cross-encoder over the fused candidates. Pure functions over
arrays: no database and no model loading in this module.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable, Mapping, Sequence
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
# bm25s's defaults (its Lucene variant), with which the retrieval benchmark was run.
BM25_K1 = 1.5
BM25_B = 0.75


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


def tokenize(docs: list[str], query: bool = False):
    """bm25s tokens (a ``Tokenized``) of ``docs``: English-stemmed words without stopwords, CJK
    runs cut as described in ``_segment_cjk``."""
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


def _words(tokens) -> list[list[str]]:
    """The token strings of each document in a bm25s ``Tokenized``."""
    spelled = {i: word for word, i in tokens.vocab.items()}
    return [[spelled[i] for i in ids] for ids in tokens.ids]


class Lexicon:
    """BM25 over a fixed list of documents, scored at query time as bm25s scores it.

    Each document is given as its term ids and their counts, so making one is a few array
    operations and needs no index of its own: a snapshot of a large vault is ready at once.
    Read-only once made, so several threads may search it.
    """

    def __init__(self, vocab: Mapping[str, int], docs: Sequence[tuple[np.ndarray, np.ndarray]]):
        self.vocab = vocab  # never changed afterwards: a newer Lexicon gets a dict of its own
        self.size = len(docs)
        widths = np.array([len(terms) for terms, _ in docs], dtype=np.int64)
        self._terms = np.concatenate([terms for terms, _ in docs]) if docs else np.zeros(0, np.int32)
        counts = np.concatenate([n for _, n in docs]).astype(np.float64) if docs else np.zeros(0)
        self._docs = np.repeat(np.arange(self.size, dtype=np.int32), widths)
        lengths = np.bincount(self._docs, weights=counts, minlength=self.size)  # tokens per document
        df = np.bincount(self._terms, minlength=len(vocab)).astype(np.float64)
        self._idf = np.log(1 + (self.size - df + 0.5) / (df + 0.5))
        norm = BM25_K1 * (1 - BM25_B + BM25_B * lengths / max(lengths.mean() if self.size else 0.0, 1e-12))
        # The term-frequency part of each entry's score, in float32 as bm25s keeps its scores.
        self._tfc = (counts / (counts + norm[self._docs])).astype(np.float32)

    def __bool__(self) -> bool:
        return self._terms.size > 0

    def scores(self, words: Sequence[str]) -> np.ndarray:
        """The BM25 score of every document for the query tokens ``words``. A repeated token counts
        again and a token no document has adds nothing, as in bm25s."""
        ids = [self.vocab[w] for w in words if w in self.vocab]
        if not ids:
            return np.zeros(self.size)
        query, repeats = np.unique(np.asarray(ids, dtype=np.int32), return_counts=True)
        hit = np.flatnonzero(np.isin(self._terms, query, kind="table"))
        terms = self._terms[hit]
        weight = self._idf[terms] * repeats[np.searchsorted(query, terms)] * self._tfc[hit]
        return np.bincount(self._docs[hit], weights=weight, minlength=self.size)

    def top(self, words: Sequence[str], k: int) -> list[int]:
        """Positions of the ``k`` best documents with a positive score, best first, ties in position order."""
        if k <= 0:
            return []
        scores = self.scores(words)
        found = np.flatnonzero(scores > 0)
        return [int(j) for j in found[np.lexsort((found, -scores[found]))][:k]]


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()


class LexiconBuilder:
    """Makes the Lexicon of each new snapshot, tokenizing only text the previous one did not hold.

    It keeps the term ids and counts of the last snapshot's documents, keyed by a digest of their
    text (chunk ids are reused after a delete), and nothing of text that has left the index. Not
    thread-safe: the caller makes one Lexicon at a time.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._docs: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        self._compact_at = 4096  # vocabulary size at which terms no document uses are looked for

    def build(self, texts: Sequence[str], tokenizer: Callable | None = None) -> Lexicon | None:
        """The Lexicon of ``texts``, or None if none of them has a single token. ``tokenizer``
        defaults to ``tokenize``."""
        keys = [_digest(t) for t in texts]
        docs = {key: self._docs[key] for key in keys if key in self._docs}
        new = {key: text for key, text in zip(keys, texts) if key not in docs}
        vocab = self._vocab
        if new:
            tokens = (tokenizer or tokenize)(list(new.values()))
            if any(word not in vocab for word in tokens.vocab):
                vocab = dict(vocab)  # the Lexicon already made keeps the dict it was made with
                for word in tokens.vocab:
                    vocab.setdefault(word, len(vocab))
            ours = np.zeros(len(tokens.vocab), dtype=np.int32)
            for word, i in tokens.vocab.items():
                ours[i] = vocab[word]
            for key, ids in zip(new, tokens.ids):
                terms, counts = np.unique(ours[np.asarray(ids, dtype=np.int64)], return_counts=True)
                docs[key] = (terms.astype(np.int32), counts.astype(np.int32))
        if len(vocab) >= self._compact_at:
            vocab, docs = _compact(vocab, docs)
            self._compact_at = max(4096, 2 * len(vocab))
        self._vocab, self._docs = vocab, docs  # only now: a failed build leaves the last state intact
        lexicon = Lexicon(vocab, [docs[key] for key in keys])
        return lexicon if lexicon else None


def _compact(
    vocab: dict[str, int], docs: dict[bytes, tuple[np.ndarray, np.ndarray]]
) -> tuple[dict[str, int], dict[bytes, tuple[np.ndarray, np.ndarray]]]:
    """Drop the terms no document uses any more (words of deleted or edited text), if they are at
    least half the vocabulary, and number the rest again."""
    used = np.zeros(len(vocab), dtype=bool)
    for terms, _ in docs.values():
        used[terms] = True
    if used.sum() * 2 > len(vocab):
        return vocab, docs
    renumber = np.cumsum(used, dtype=np.int64).astype(np.int32) - 1  # keeps the order of the ids
    vocab = {word: int(renumber[i]) for word, i in vocab.items() if used[i]}
    return vocab, {key: (renumber[terms], counts) for key, (terms, counts) in docs.items()}


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

    Searches may overlap on several threads. ``lexicon`` is the BM25 index of ``texts`` when the
    caller has made it (see LexiconBuilder); otherwise it is made on first use, by one thread,
    and is visible to the others only once complete. Everything else is read-only.
    """

    def __init__(self, texts: Sequence[str], vectors: np.ndarray | None, lexicon: Lexicon | None | object = _UNBUILT):
        self.texts = list(texts)
        self.vectors = vectors
        self._lex = lexicon
        self._build_lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.texts)

    def _tokenize(self, docs: list[str], query: bool = False):
        return tokenize(docs, query)

    def _lexicon(self) -> Lexicon | None:
        """The BM25 index, made once; None if no chunk contains a token."""
        lex = self._lex
        if lex is _UNBUILT:
            with self._build_lock:
                lex = self._lex
                if lex is _UNBUILT:
                    lex = LexiconBuilder().build(self.texts, self._tokenize)
                    self._lex = lex  # published only once complete
        return lex

    def lexical(self, query: str, k: int) -> list[int]:
        """Positions of the top-k BM25 matches with a positive score."""
        if not self.texts or k <= 0:
            return []
        lex = self._lexicon()
        if lex is None:
            return []
        return lex.top(_words(self._tokenize([query], query=True))[0], min(k, len(self.texts)))

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
