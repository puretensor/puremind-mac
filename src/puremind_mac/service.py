"""Service: store + models + retrieval, shared by the CLI and the MCP server."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .models import CrossEncoderReranker, Embedder, Reranker, SentenceTransformerEmbedder
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


class Service:
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
            cfg.embedder, cfg.device, cfg.embed_max_seq
        )
        self.reranker: Reranker = reranker or CrossEncoderReranker(cfg.reranker, cfg.device)
        self._lock = threading.Lock()
        self._gen = -1
        self._chunks: list[ChunkRow] = []
        self._index = HybridIndex([], None)

    @property
    def vault_root(self) -> Path | None:
        if self.cfg.vault:
            return self.cfg.vault.expanduser().resolve()
        stored = self.store.get_meta("vault_root")
        return Path(stored) if stored else None

    def _loaded(self) -> HybridIndex:
        with self._lock:
            gen = self.store.generation()
            if gen != self._gen:
                self._chunks, vectors = self.store.load_index()
                self._index = HybridIndex([c.content for c in self._chunks], vectors if self._chunks else None)
                self._gen = gen
            return self._index

    def search(self, query: str, limit: int = 10, rerank: bool = True) -> list[Hit]:
        built_with = self.store.get_meta("embedder")
        if built_with and built_with != self.embedder.name:
            raise IndexMismatch(
                f"the index was built with {built_with} but {self.embedder.name} is configured; "
                "run `puremind index --rebuild`"
            )
        index = self._loaded()
        if not len(index):
            return []
        scored = index.search(
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
            for c in (self._chunks[s.position],)
        ]

    def fts_search(self, query: str, limit: int = 10) -> list[Hit]:
        rows = self.store.fts_search(query, limit)
        by_id = {c.id: c for c in self.store.get_chunks([i for i, _ in rows])}
        return [
            Hit(c.path, c.heading_path, c.chunk_index, score, c.content)
            for i, score in rows
            for c in (by_id[i],)
        ]
