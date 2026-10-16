from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puremind_mac.config import Config  # noqa: E402
from puremind_mac.service import Service  # noqa: E402
from puremind_mac.store import Store  # noqa: E402

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


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def reranker():
    return FakeReranker()


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    (v / "projects").mkdir(parents=True)
    (v / "journal").mkdir()
    (v / "projects" / "espresso.md").write_text(
        "# Espresso machine\n\nThe grinder is set to 14 for the washed Ethiopian beans. "
        "Descale the boiler every three months.\n\n## Maintenance\n\nBackflush weekly with detergent.\n"
    )
    (v / "projects" / "garden.md").write_text(
        "# Garden plan\n\nPlant tomatoes in May. The raised beds need compost and mulch before spring.\n"
    )
    (v / "journal" / "2026-10-01.md").write_text(
        "# Journal\n\nDecided to migrate the build pipeline to hermetic containers next quarter.\n"
    )
    (v / ".obsidian").mkdir()
    (v / ".obsidian" / "workspace.md").write_text("# hidden\n\nshould never be indexed\n")
    (v / "notes.txt").write_text("not markdown")
    return v


@pytest.fixture
def cfg(tmp_path, vault):
    return Config(home=tmp_path / "home", vault=vault, db=tmp_path / "home" / "index.sqlite")


@pytest.fixture
def store(cfg):
    s = Store(cfg.db_path)
    yield s
    s.close()


@pytest.fixture
def service(cfg, store, embedder, reranker):
    return Service(cfg, store=store, embedder=embedder, reranker=reranker)
