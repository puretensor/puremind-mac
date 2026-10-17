"""Service: store + models + retrieval, shared by the CLI and the MCP server."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .config import Config
from .models import (
    CrossEncoderReranker,
    Embedder,
    Reranker,
    SentenceTransformerEmbedder,
    embedder_revision,
    model_key,
    model_label,
)
from .retrieval import HybridIndex
from .store import ChunkRow, Store


@dataclass(frozen=True)
class Hit:
    path: str
    heading_path: str
    chunk_index: int
    score: float
    content: str


class IndexMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class _Snapshot:
    """The index as of one store generation.

    Positions returned by ``index`` are row numbers in ``chunks``, so the two are kept together
    and never changed: a request that took a snapshot resolves its hits against that snapshot
    even if another thread has loaded a newer one in the meantime.
    """

    gen: int
    chunks: tuple[ChunkRow, ...]
    index: HybridIndex


class Service:
    """Store, models and retrieval behind one object. Searches may overlap on several threads."""

    def __init__(
        self,
        cfg: Config,
        store: Store | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ):
        self.cfg = cfg
        self.store = store or Store(cfg.db_path)
        self.embedder: Embedder = embedder or SentenceTransformerEmbedder(
            cfg.embedder, cfg.device, cfg.embed_max_seq, revision=cfg.embedder_revision
        )
        self.reranker: Reranker = reranker or CrossEncoderReranker(
            cfg.reranker, cfg.device, revision=cfg.reranker_revision
        )
        self._lock = threading.Lock()
        self._snap = _Snapshot(-1, (), HybridIndex([], None))

    def _loaded(self) -> _Snapshot:
        """The current snapshot, reloaded from the store if its generation has moved on."""
        with self._lock:
            gen = self.store.generation()  # read first: a write racing the load only costs one extra reload
            if gen != self._snap.gen:
                chunks, vectors = self.store.load_index()
                index = HybridIndex([c.content for c in chunks], vectors if chunks else None)
                self._snap = _Snapshot(gen, tuple(chunks), index)
            return self._snap

    def search(self, query: str, limit: int = 10, rerank: bool = True) -> list[Hit]:
        built_with = self.store.get_meta("embedder")
        built_rev = self.store.get_meta("embedder_revision")  # None in an index from 0.1.1, which pinned none
        revision = embedder_revision(self.embedder)
        if built_with and model_key(built_with) != model_key(self.embedder.name):
            raise IndexMismatch(
                f"the index was built with {model_label(built_with)} but {model_label(self.embedder.name)} "
                "is configured; run `puremind index --rebuild`"
            )
        if built_with and built_rev is not None and built_rev != revision:
            raise IndexMismatch(
                f"the index was built with {model_label(built_with)} at revision {built_rev or '(none)'} but "
                f"revision {revision or '(none)'} is configured; run `puremind index --rebuild`"
            )
        snap = self._loaded()
        if not len(snap.index):
            return []
        scored = snap.index.search(
            query,
            self.embedder,
            self.reranker if rerank else None,
            limit=limit,
            candidates=self.cfg.candidates,
            rrf_k=self.cfg.rrf_k,
            rerank_chars=self.cfg.rerank_chars,
        )
        return [
            Hit(c.path, c.heading_path, c.chunk_index, s.score, c.content)
            for s in scored
            for c in (snap.chunks[s.position],)
        ]

    def fts_search(self, query: str, limit: int = 10) -> list[Hit]:
        return [
            Hit(c.path, c.heading_path, c.chunk_index, score, c.content)
            for c, score in self.store.fts_hits(query, limit)
        ]
