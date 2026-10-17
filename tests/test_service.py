"""Service: lexical search over the store, and serving an index that cannot be written."""

from __future__ import annotations

import os

import numpy as np
import pytest

from puremind_mac.indexer import index_vault
from puremind_mac.service import Service
from puremind_mac.store import Store

ONE = np.ones((1, 4), dtype=np.float32)


def test_fts_search_returns_text_heading_and_a_positive_ranked_score(service, vault):
    index_vault(vault, service.store, service.embedder)
    hits = service.fts_search("espresso grinder")
    assert hits[0].path == "projects/espresso.md"
    assert hits[0].heading_path == "Espresso machine" and hits[0].chunk_index == 0
    assert "grinder is set to 14" in hits[0].content
    assert all(h.score > 0 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_fts_search_edges(service, vault):
    index_vault(vault, service.store, service.embedder)
    assert service.fts_search("zzzzqqq") == []
    assert service.fts_search("   ") == []
    assert len(service.fts_search("the", limit=1)) <= 1


def test_search_finds_chinese_and_japanese_notes_by_their_words(service, vault):
    # The fake embedder gives CJK text a zero vector, so only the BM25 half can find these notes.
    (vault / "projects" / "migration.md").write_text("# 数据库迁移\n\n数据库迁移到新的服务器,需要在周末完成。\n")
    (vault / "projects" / "kaisha.md").write_text("# 会議\n\n来週までに移行作業を終わらせる必要があります。\n")
    index_vault(vault, service.store, service.embedder)
    assert service.search("服务器 迁移", limit=1, rerank=False)[0].path == "projects/migration.md"
    assert service.search("移行作業", limit=1, rerank=False)[0].path == "projects/kaisha.md"
    assert service.search("espresso grinder", limit=1, rerank=False)[0].path == "projects/espresso.md"


def test_fts_search_is_not_broken_by_a_reindex_right_after_the_match(store, service):
    # The chunk ids of a note change when it is re-indexed. The old two-step lookup (match, then
    # fetch the rows by id) raised KeyError when that happened between the two steps.
    store.replace_file("a.md", "h1", 1.0, 1, "a", [("H", "zebra stripes")], ONE)
    store.replace_file("b.md", "h1", 1.0, 1, "b", [("H", "giraffe spots")], ONE)
    fired = []
    real = store.db.execute

    def execute(sql, *a, **k):
        result = real(sql, *a, **k)
        if "MATCH" in sql and not fired:
            fired.append(sql)
            store.replace_file("a.md", "h2", 2.0, 1, "a", [("H", "zebra stripes, rewritten")], ONE)
        return result

    store.db.execute = execute
    hits = service.fts_search("zebra")
    assert fired
    assert [(h.path, h.content) for h in hits] == [("a.md", "zebra stripes")]  # the rows as they were at the match
    assert [h.content for h in service.fts_search("zebra")] == ["zebra stripes, rewritten"]


ENFORCED_MODES = os.name == "posix" and not (hasattr(os, "geteuid") and os.geteuid() == 0)


@pytest.mark.skipif(not ENFORCED_MODES, reason="needs enforced file modes")
def test_service_searches_an_index_it_cannot_write(cfg, vault, embedder, reranker, tmp_path):
    home = tmp_path / "shared"
    with Store(home / "index.sqlite") as s:
        index_vault(vault, s, embedder)
    for p in home.iterdir():
        p.chmod(0o444)
    home.chmod(0o555)
    try:
        with Store(home / "index.sqlite") as s:
            assert s.read_only
            svc = Service(cfg, store=s, embedder=embedder, reranker=reranker)
            assert svc.search("espresso grinder")[0].path == "projects/espresso.md"
            assert svc.fts_search("tomatoes")[0].path == "projects/garden.md"
            assert s.get_meta("vault_root") == str(vault.resolve())
    finally:
        home.chmod(0o755)
        for p in home.iterdir():
            p.chmod(0o644)


def test_the_server_searches_an_index_built_with_a_tilde_model_path(cfg, store, vault, embedder, reranker):
    # `puremind serve` and the `connect` entry make a local model path absolute; `index` may store it with ~.
    embedder.name = "~/models/embedder"
    index_vault(vault, store, embedder)
    embedder.name = os.path.expanduser("~/models/embedder")
    svc = Service(cfg, store=store, embedder=embedder, reranker=reranker)
    assert svc.search("espresso grinder", rerank=False)[0].path == "projects/espresso.md"


def test_an_index_built_at_another_embedder_revision_is_refused(cfg, store, vault, embedder, reranker):
    from puremind_mac.service import IndexMismatch

    embedder.revision = "rev-a"
    index_vault(vault, store, embedder)
    embedder.revision = "rev-b"
    with pytest.raises(IndexMismatch, match="revision rev-a but revision rev-b"):
        Service(cfg, store=store, embedder=embedder, reranker=reranker).search("espresso", rerank=False)


def test_a_reload_after_one_saved_note_tokenizes_only_that_note(cfg, store, vault, embedder, reranker, monkeypatch):
    from puremind_mac import retrieval

    index_vault(vault, store, embedder)
    corpus = []
    real = retrieval.tokenize
    monkeypatch.setattr(
        retrieval, "tokenize", lambda docs, query=False: (corpus.append(len(docs)) if not query else None, real(docs, query))[1]
    )
    svc = Service(cfg, store=store, embedder=embedder, reranker=reranker)
    svc.search("espresso", rerank=False)
    assert corpus == [store.stats()["chunks"]]
    (vault / "projects" / "garden.md").write_text("# Garden plan\n\nPlant zucchini in June.\n")
    index_vault(vault, store, embedder)
    hits = svc.search("zucchini", rerank=False)
    assert corpus[1:] == [1]  # the one new chunk; the others come from the previous snapshot
    assert hits[0].path == "projects/garden.md" and "zucchini" in hits[0].content
    assert not any("tomatoes" in h.content for h in svc.search("tomatoes", rerank=False))
