import pytest

from puremind_mac.indexer import discover, index_vault, title_of


def test_discover_skips_hidden_dirs_and_non_markdown(vault):
    names = sorted(p.name for p in discover(vault))
    assert names == ["2026-10-01.md", "espresso.md", "garden.md"]


def test_discover_skips_symlinks_leaving_the_vault(vault, tmp_path):
    outside = tmp_path / "secret.md"
    outside.write_text("# secret\n")
    (vault / "link.md").symlink_to(outside)
    assert "link.md" not in [p.name for p in discover(vault)]


def test_first_index_adds_everything(vault, store, embedder):
    stats = index_vault(vault, store, embedder)
    assert (stats.scanned, stats.added, stats.updated, stats.unchanged, stats.removed) == (3, 3, 0, 0, 0)
    assert stats.chunks == store.stats()["chunks"] > 0
    assert set(store.file_hashes()) == {"projects/espresso.md", "projects/garden.md", "journal/2026-10-01.md"}


def test_second_index_is_a_no_op_and_embeds_nothing(vault, store, embedder):
    index_vault(vault, store, embedder)
    embedded = embedder.texts_embedded
    stats = index_vault(vault, store, embedder)
    assert (stats.unchanged, stats.added, stats.updated) == (3, 0, 0)
    assert embedder.texts_embedded == embedded


def test_touching_without_changing_content_does_not_reindex(vault, store, embedder):
    index_vault(vault, store, embedder)
    p = vault / "projects" / "garden.md"
    p.write_text(p.read_text())
    assert index_vault(vault, store, embedder).unchanged == 3


def test_only_changed_file_is_reembedded(vault, store, embedder):
    index_vault(vault, store, embedder)
    before = embedder.texts_embedded
    p = vault / "projects" / "garden.md"
    p.write_text("# Garden plan\n\nPlant carrots in April instead.\n")
    stats = index_vault(vault, store, embedder)
    assert (stats.updated, stats.unchanged) == (1, 2)
    assert embedder.texts_embedded == before + stats.chunks
    contents = [c.content for c in store.load_index()[0]]
    assert any("carrots" in c for c in contents)
    assert not any("tomatoes" in c for c in contents)


def test_new_file_is_added(vault, store, embedder):
    index_vault(vault, store, embedder)
    (vault / "new.md").write_text("# New\n\nfresh note\n")
    stats = index_vault(vault, store, embedder)
    assert (stats.added, stats.unchanged) == (1, 3)


def test_deleted_file_is_removed(vault, store, embedder):
    index_vault(vault, store, embedder)
    (vault / "projects" / "garden.md").unlink()
    stats = index_vault(vault, store, embedder)
    assert stats.removed == 1
    assert "projects/garden.md" not in store.file_hashes()
    assert not any("tomatoes" in c.content for c in store.load_index()[0])


def test_renamed_file_counts_as_remove_plus_add(vault, store, embedder):
    index_vault(vault, store, embedder)
    (vault / "projects" / "garden.md").rename(vault / "projects" / "yard.md")
    stats = index_vault(vault, store, embedder)
    assert (stats.added, stats.removed) == (1, 1)


def test_rebuild_reembeds_everything(vault, store, embedder):
    index_vault(vault, store, embedder)
    stats = index_vault(vault, store, embedder, rebuild=True)
    assert stats.added == 3 and stats.unchanged == 0


def test_changing_the_embedder_triggers_a_rebuild(vault, store, embedder):
    index_vault(vault, store, embedder)
    embedder.name = "another-model"
    stats = index_vault(vault, store, embedder)
    assert stats.added == 3
    assert store.get_meta("embedder") == "another-model"


def test_a_different_vault_cannot_share_an_index(vault, store, embedder, tmp_path):
    index_vault(vault, store, embedder)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="different --db"):
        index_vault(other, store, embedder)


def test_missing_vault_raises(store, embedder, tmp_path):
    with pytest.raises(NotADirectoryError):
        index_vault(tmp_path / "nope", store, embedder)


def test_oversized_file_is_skipped(vault, store, embedder, monkeypatch):
    monkeypatch.setattr("puremind_mac.indexer.MAX_FILE_BYTES", 50)
    stats = index_vault(vault, store, embedder)
    assert len(stats.skipped) >= 1


def test_empty_file_is_tracked_without_chunks(vault, store, embedder):
    (vault / "empty.md").write_text("")
    index_vault(vault, store, embedder)
    assert "empty.md" in store.file_hashes()


def test_invalid_utf8_does_not_crash(vault, store, embedder):
    (vault / "bad.md").write_bytes(b"# Bad\n\ncaf\xe9 notes\n")
    index_vault(vault, store, embedder)
    assert "bad.md" in store.file_hashes()


def test_title_of():
    assert title_of("---\ntitle: x\n---\n# Real Title\n", "stem") == "Real Title"
    assert title_of("no heading", "stem") == "stem"
    assert title_of("```\n# in fence\n```\n", "stem") == "stem"


def test_checked_in_sample_vault_indexes_and_is_searchable(store, embedder, reranker, cfg):
    from pathlib import Path

    from puremind_mac.service import Service

    sample = Path(__file__).parent / "fixtures" / "vault"
    stats = index_vault(sample, store, embedder)
    assert stats.added == 4
    import dataclasses

    svc = Service(dataclasses.replace(cfg, vault=sample), store=store, embedder=embedder, reranker=reranker)
    assert svc.search("signing keys rotate", limit=1)[0].path == "journal/2026-10-08.md"
