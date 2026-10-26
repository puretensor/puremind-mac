import errno
import hashlib
import logging
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from puremind_mac import indexer
from puremind_mac.indexer import (
    EMBED_BATCH,
    MAX_PATH_CHARS,
    EmbedderError,
    ModelMismatch,
    discover,
    index_vault,
    title_of,
)
from puremind_mac.store import Store


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


def test_touching_without_changing_content_refreshes_recent_without_reembedding(vault, store, embedder, service):
    index_vault(vault, store, embedder)
    before = embedder.texts_embedded
    chunks, vectors = store.load_index()
    p = vault / "projects" / "garden.md"
    touched = time.time() + 60
    os.utime(p, (touched, touched))
    assert index_vault(vault, store, embedder).unchanged == 3
    row = store.recent(0, 1)[0]
    assert (row.path, row.mtime, row.size) == ("projects/garden.md", p.stat().st_mtime, p.stat().st_size)
    assert embedder.texts_embedded == before and store.load_index()[0] == chunks
    np.testing.assert_array_equal(store.load_index()[1], vectors)
    from puremind_mac.tools import puremind_recent

    assert puremind_recent(service).splitlines()[1].startswith("- projects/garden.md")


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
    before = embedder.texts_embedded
    stats = index_vault(vault, store, embedder, rebuild=True)
    assert (stats.updated, stats.unchanged) == (3, 0) and embedder.texts_embedded == 2 * before


def test_a_rebuild_keeps_the_old_index_searchable_until_each_note_is_replaced(vault, store, embedder, monkeypatch):
    index_vault(vault, store, embedder)
    found = []
    real = embedder.embed_documents

    def search_meanwhile(texts):  # what a search finds while the rebuild is embedding
        found.append(len(store.fts_hits("tomatoes")))
        return real(texts)

    monkeypatch.setattr(embedder, "embed_documents", search_meanwhile)
    assert index_vault(vault, store, embedder, rebuild=True).updated == 3
    assert found and all(found)


def test_notes_cut_by_an_earlier_chunker_are_cut_again_once(vault, store, embedder):
    index_vault(vault, store, embedder)
    with store.db:
        store.db.execute("UPDATE files SET chunker = 0")  # how an index from 0.1.1 looks after the upgrade
    stats = index_vault(vault, store, embedder)
    assert (stats.updated, stats.unchanged) == (3, 0)
    assert index_vault(vault, store, embedder).unchanged == 3


def test_a_new_embedder_is_refused_without_rebuild(vault, store, embedder):
    index_vault(vault, store, embedder)
    before, calls = store.stats(), embedder.doc_calls
    embedder.name = "another-model"  # a `watch` on the old config and a fresh `index` must not wipe each other
    with pytest.raises(ModelMismatch, match="built with fake-embedder but another-model is configured; nothing was"):
        index_vault(vault, store, embedder)
    assert store.stats() == before and store.get_meta("embedder") == "fake-embedder"
    assert embedder.doc_calls == calls  # refused before anything is embedded, the probe included
    stats = index_vault(vault, store, embedder, rebuild=True)
    assert stats.added == 3
    assert store.get_meta("embedder") == "another-model"


def test_an_index_with_no_notes_takes_a_new_embedder_without_rebuild(tmp_path, store, embedder):
    empty = tmp_path / "empty"
    empty.mkdir()
    index_vault(empty, store, embedder)  # records the model, holds nothing that could be lost
    (empty / "a.md").write_text("# A\n\nhello\n")
    embedder.name = "another-model"
    assert index_vault(empty, store, embedder).added == 1
    assert store.get_meta("embedder") == "another-model"


def test_a_different_vault_cannot_share_an_index(vault, store, embedder, tmp_path):
    index_vault(vault, store, embedder)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="different PUREMIND_DB") as info:
        index_vault(other, store, embedder)
    message = str(info.value)
    assert "--db" not in message and "puremind index --rebuild" in message and "config.toml" in message


def test_another_spelling_of_the_same_vault_shares_its_index(vault, store, embedder, tmp_path):
    index_vault(vault, store, embedder)
    alias = tmp_path / "alias"
    alias.symlink_to(vault)  # stands in for a differently cased path on case-insensitive APFS
    store.set_meta("vault_root", str(alias))
    stats = index_vault(vault, store, embedder)
    assert stats.unchanged == stats.scanned > 0 and store.get_meta("vault_root") == str(vault.resolve())


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


class _Wrapped:
    """The fixture embedder behind a hook that runs first (to fail, edit files or count)."""

    def __init__(self, inner, hook=None, name=None):
        self._inner, self._hook = inner, hook
        self.name = name or inner.name

    def embed_documents(self, texts):
        if self._hook:
            self._hook(list(texts))
        return self._inner.embed_documents(texts)

    def embed_query(self, query):
        return self._inner.embed_query(query)


def _boom(texts):
    raise OSError("could not load the model")


def _rows(store, path):
    return [c.content for c in store.load_index()[0] if c.path == path]


# ---- odd directory entries


def test_symlink_loops_do_not_abort_the_run(vault, store, embedder):
    (vault / "self.md").symlink_to("self.md")
    (vault / "a.md").symlink_to("b.md")
    (vault / "b.md").symlink_to("a.md")
    assert sorted(p.name for p in discover(vault)) == ["2026-10-01.md", "espresso.md", "garden.md"]
    assert index_vault(vault, store, embedder).added == 3


def test_a_fifo_named_like_a_note_does_not_hang_the_run(vault, store, embedder):
    if not hasattr(os, "mkfifo"):
        pytest.skip("no named pipes here")
    os.mkfifo(vault / "pipe.md")
    done = []
    worker = threading.Thread(target=lambda: done.append(index_vault(vault, store, embedder)), daemon=True)
    worker.start()
    worker.join(10)
    assert done and done[0].added == 3


def test_reading_a_note_never_blocks_on_a_pipe(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("no named pipes here")
    fifo = tmp_path / "swapped-in.md"  # what a note swapped for a pipe after the scan looks like
    os.mkfifo(fifo)
    outcome = []

    def read():
        try:
            indexer._read_note(fifo)
        except OSError as exc:
            outcome.append(exc)

    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    worker.join(10)
    assert outcome and "regular file" in str(outcome[0])


def test_symlinks_to_hidden_or_non_markdown_targets_are_not_indexed(vault, store, embedder):
    (vault / ".env").write_text("API_KEY=FAKE-ENV-SECRET\n")
    plugin = vault / ".obsidian" / "plugins" / "sync"
    plugin.mkdir(parents=True)
    (plugin / "data.json").write_text('{"token": "FAKE-PLUGIN-TOKEN"}')
    (vault / ".trash").mkdir()
    (vault / ".trash" / "old.md").write_text("# Old\n\nFAKE-TRASHED-NOTE\n")
    (vault / "link-env.md").symlink_to(vault / ".env")
    (vault / "link-json.md").symlink_to(plugin / "data.json")
    (vault / "link-trash.md").symlink_to(vault / ".trash" / "old.md")
    assert not [p for p in discover(vault) if p.name.startswith("link-")]
    index_vault(vault, store, embedder)
    assert not [p for p in store.file_hashes() if p.startswith("link-")]
    assert not any("FAKE" in c.content for c in store.load_index()[0])


def _write_raw(path: bytes, content: bytes) -> None:
    try:
        with open(path, "wb") as fh:
            fh.write(content)
    except OSError:
        pytest.skip("this file system rejects names that are not UTF-8")


def test_a_file_name_that_is_not_utf8_is_skipped_with_a_safe_message(vault, store, embedder):
    _write_raw(os.fsencode(vault) + b"/caf\xe9.md", b"# Latin one\n\ntext\n")
    stats = index_vault(vault, store, embedder)
    assert stats.added == 3
    assert any("caf\\xe9.md" in item and "UTF-8" in item for item in stats.skipped)


def test_a_directory_name_that_is_not_utf8_is_skipped_once(vault, store, embedder):
    bad = os.fsencode(vault) + b"/d\xe9"
    try:
        os.mkdir(bad)
    except OSError:
        pytest.skip("this file system rejects names that are not UTF-8")
    for name in (b"one.md", b"two.md"):
        _write_raw(bad + b"/" + name, b"# Inside\n\ntext\n")
    stats = index_vault(vault, store, embedder)
    assert stats.added == 3
    assert [item for item in stats.skipped if "d\\xe9" in item] == ["d\\xe9/ (directory name is not valid UTF-8)"]


# ---- one note must not stop the run


def test_a_note_that_cannot_be_chunked_is_skipped_and_keeps_its_rows(vault, store, embedder, monkeypatch):
    index_vault(vault, store, embedder)
    (vault / "projects" / "garden.md").write_text("# Garden plan\n\nPOISON text\n")
    (vault / "zz-new.md").write_text("# New\n\nfresh note\n")
    real = indexer.chunk_markdown

    def flaky(text, *args, **kwargs):
        if "POISON" in text:
            raise ValueError("cannot chunk this")
        return real(text, *args, **kwargs)

    monkeypatch.setattr(indexer, "chunk_markdown", flaky)
    stats = index_vault(vault, store, embedder)
    assert any("projects/garden.md" in item and "cannot chunk this" in item for item in stats.skipped)
    assert (stats.added, stats.removed) == (1, 0)
    assert any("tomatoes" in c for c in _rows(store, "projects/garden.md"))


def test_a_failing_index_file_stops_the_run(vault, store, embedder, monkeypatch):
    real = store.replace_file

    def full(path, *args, **kwargs):  # a full disk fails every later note too, so it is not one note's problem
        if path == "projects/garden.md":
            raise sqlite3.OperationalError("database or disk is full")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(store, "replace_file", full)
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        index_vault(vault, store, embedder)
    assert "projects/garden.md" not in store.file_hashes()


@pytest.mark.parametrize("swapped", ["note", "folder"])
def test_a_note_replaced_by_a_symlink_during_the_run_is_not_read_through_it(vault, store, embedder, monkeypatch, swapped):
    hidden = vault / ".private"  # hidden, so nothing in it is a note
    hidden.mkdir()
    (hidden / "garden.md").write_text("API_KEY=swapped-in-secret-value\n")
    note, folder = vault / "projects" / "garden.md", vault / "projects"
    real = embedder.embed_documents

    def swap_then_embed(texts):  # the scan and the first read have already accepted the note
        if swapped == "note" and not note.is_symlink():
            note.unlink()
            note.symlink_to(hidden / "garden.md")
        if swapped == "folder" and not folder.is_symlink():
            folder.rename(vault / "projects-old")
            folder.symlink_to(hidden)
        return real(texts)

    monkeypatch.setattr(embedder, "embed_documents", swap_then_embed)
    index_vault(vault, store, embedder)
    assert "projects/garden.md" not in store.file_hashes()
    assert not any("swapped-in-secret" in c.content for c in store.load_index()[0])


def test_a_note_the_model_cannot_embed_costs_only_that_note(vault, store, embedder, monkeypatch):
    for i in range(30):
        (vault / f"n{i:02}.md").write_text(f"# Note {i}\n\nordinary text {i}\n")
    (vault / "n15.md").write_text("# Poison\n\nPOISON text\n")
    real = embedder.embed_documents

    def picky(texts):  # a backend that fails on one input, inside a batch of many notes
        if any("POISON" in t for t in texts):
            raise RuntimeError("backend error on this input")
        return real(texts)

    monkeypatch.setattr(embedder, "embed_documents", picky)
    stats = index_vault(vault, store, embedder)
    assert stats.added == 32 and "n15.md" not in store.file_hashes()
    assert "n15.md (the embedding model failed on it: RuntimeError: backend error on this input)" in stats.skipped


def test_a_model_that_fails_on_every_note_still_stops_the_run(vault, store, embedder, monkeypatch):
    def broken(texts):
        raise RuntimeError("broken backend")

    monkeypatch.setattr(embedder, "embed_documents", broken)
    with pytest.raises(EmbedderError, match="broken backend"):
        index_vault(vault, store, embedder)
    assert store.stats()["files"] == 0


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS does not allow a path this long")
def test_a_vault_nested_a_thousand_folders_deep_is_indexed_down_to_the_path_limit(vault, store, embedder):
    deep = vault  # Python 3.11 walks folders recursively, and overflowed here
    for _ in range(1100):  # one by one: mkdir(parents=True) recurses too
        deep = deep / "a"
        deep.mkdir()
    (deep / "bottom.md").write_text("# bottom\n\ntext\n")
    try:
        stats = index_vault(vault, store, embedder)
    finally:
        (deep / "bottom.md").unlink()
        while deep != vault:  # bottom up, since pytest's own clean-up (rmtree) recurses as well
            deep.rmdir()
            deep = deep.parent
    assert stats.added == 3
    assert any(f"folders below it have paths longer than {MAX_PATH_CHARS} characters" in item for item in stats.skipped)


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS does not allow a path this long")
def test_a_note_whose_path_puremind_get_would_refuse_is_skipped(vault, store, embedder):
    deep = vault.joinpath(*["d" * 100] * 11)
    deep.mkdir(parents=True)
    (deep / "n.md").write_text("# deep\n\ntext\n")
    assert len((deep / "n.md").relative_to(vault).as_posix()) > MAX_PATH_CHARS
    stats = index_vault(vault, store, embedder)
    assert any(f"(path longer than {MAX_PATH_CHARS} characters)" in item for item in stats.skipped)
    assert all(len(path) <= MAX_PATH_CHARS for path in store.file_hashes())


def test_a_note_that_exists_but_cannot_be_read_keeps_its_rows(vault, store, embedder, monkeypatch):
    index_vault(vault, store, embedder)
    real = indexer._read_note

    def deny(path):
        if path.name == "garden.md":
            raise PermissionError(errno.EACCES, "Permission denied")
        return real(path)

    monkeypatch.setattr(indexer, "_read_note", deny)
    stats = index_vault(vault, store, embedder)
    assert stats.removed == 0 and "projects/garden.md" in store.file_hashes()
    assert "projects/garden.md (Permission denied)" in stats.skipped


def test_a_note_that_vanishes_during_the_scan_is_removed(vault, store, embedder, monkeypatch):
    index_vault(vault, store, embedder)
    real = indexer._read_note

    def gone(path):
        if path.name == "garden.md":
            raise FileNotFoundError(errno.ENOENT, "No such file or directory")
        return real(path)

    monkeypatch.setattr(indexer, "_read_note", gone)
    stats = index_vault(vault, store, embedder)
    assert stats.removed == 1 and "projects/garden.md" not in store.file_hashes()


def test_a_failing_embedder_stops_the_run_with_a_clear_error(vault, store, embedder):
    with pytest.raises(EmbedderError, match="fake-embedder failed on .*projects/garden.md: OSError: could not load the model") as info:
        index_vault(vault, store, _Wrapped(embedder, _boom))
    assert isinstance(info.value, RuntimeError)  # the CLI reports RuntimeError as "error: ..." with exit 2


def test_a_note_that_crashes_the_embedder_is_named_in_the_error(vault, store, embedder):
    index_vault(vault, store, embedder)
    (vault / "projects" / "garden.md").write_text("# Garden plan\n\nPOISON\n")

    def poison(texts):
        if any("POISON" in t for t in texts):
            raise RuntimeError("cannot embed this")

    with pytest.raises(EmbedderError, match=r"failed on projects/garden\.md: RuntimeError: cannot embed this"):
        index_vault(vault, store, _Wrapped(embedder, poison))
    assert store.stats()["files"] == 3


# ---- deletion is not inferred from a failed scan


def _unlistable(monkeypatch, directory):
    real_walk = os.walk

    def walk(top, onerror=None, **kwargs):
        for dirpath, dirnames, filenames in real_walk(top, onerror=onerror, **kwargs):
            if Path(dirpath).name == directory:
                dirnames[:] = []
                if onerror:
                    onerror(PermissionError(errno.EACCES, "Permission denied", dirpath))
                continue
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(indexer.os, "walk", walk)


def test_a_directory_that_cannot_be_listed_keeps_its_notes(vault, store, embedder, monkeypatch):
    index_vault(vault, store, embedder)
    _unlistable(monkeypatch, "projects")
    stats = index_vault(vault, store, embedder)
    assert stats.removed == 0 and store.stats()["files"] == 3
    assert "projects/ (cannot list directory: Permission denied)" in stats.skipped
    assert "removal skipped: 1 directory could not be read" in stats.summary()
    monkeypatch.undo()
    (vault / "projects" / "garden.md").unlink()
    assert index_vault(vault, store, embedder).removed == 1  # a clean scan removes again


def test_a_directory_that_cannot_be_listed_does_not_keep_deleted_notes_elsewhere(vault, store, embedder, monkeypatch):
    index_vault(vault, store, embedder)
    (vault / "journal" / "2026-10-01.md").unlink()
    _unlistable(monkeypatch, "projects")  # a lost+found or a protected folder that is always there
    stats = index_vault(vault, store, embedder)
    assert stats.removed == 1 and sorted(store.file_hashes()) == ["projects/espresso.md", "projects/garden.md"]
    assert "removal skipped: 1 directory could not be read" in stats.summary()


def test_a_vault_that_cannot_be_listed_keeps_every_note(vault, store, embedder, monkeypatch):
    index_vault(vault, store, embedder)
    _unlistable(monkeypatch, vault.name)
    stats = index_vault(vault, store, embedder)
    assert stats.removed == 0 and store.stats()["files"] == 3


def test_a_new_embedder_revision_embeds_every_note_again(vault, store, embedder):
    embedder.revision = "rev-a"
    index_vault(vault, store, embedder)
    assert index_vault(vault, store, embedder).unchanged == 3 and store.get_meta("embedder_revision") == "rev-a"
    embedder.revision = "rev-b"  # vectors of two revisions cannot be searched together
    with pytest.raises(ModelMismatch, match="at revision rev-a but fake-embedder at revision rev-b"):
        index_vault(vault, store, embedder)
    assert store.get_meta("embedder_revision") == "rev-a" and store.stats()["files"] == 3
    stats = index_vault(vault, store, embedder, rebuild=True)
    assert stats.added == 3 and stats.unchanged == 0 and store.get_meta("embedder_revision") == "rev-b"


def test_an_index_without_a_recorded_revision_is_kept(vault, store, embedder):
    index_vault(vault, store, embedder)
    with store.db:
        store.db.execute("DELETE FROM meta WHERE key = 'embedder_revision'")  # as 0.1.1 left it
    embedder.revision = "rev-a"
    assert index_vault(vault, store, embedder).unchanged == 3


def test_a_changed_embedding_window_requires_rebuild(vault, store, embedder):
    embedder.max_seq = 2048
    index_vault(vault, store, embedder)
    fingerprint = store.get_meta("embedding_config")
    assert '"embed_max_seq": 2048' in fingerprint
    before = embedder.texts_embedded
    embedder.max_seq = 1024
    with pytest.raises(
        ModelMismatch, match=r"embedding configuration \(embed_max_seq 2048, but embed_max_seq 1024 is configured\).*rebuild"
    ):
        index_vault(vault, store, embedder)
    assert store.get_meta("embedding_config") == fingerprint
    assert store.stats()["files"] == 3 and embedder.texts_embedded == before
    stats = index_vault(vault, store, embedder, rebuild=True)
    assert (stats.added, stats.unchanged) == (3, 0)
    assert embedder.texts_embedded == before + stats.chunks + 1  # probe before clearing
    assert index_vault(vault, store, embedder).unchanged == 3


def test_a_new_model_with_a_new_window_is_reported_as_a_model_change(vault, store, embedder):
    embedder.max_seq = 2048
    index_vault(vault, store, embedder)
    embedder.name, embedder.max_seq = "another-model", 1024
    with pytest.raises(ModelMismatch, match="built with fake-embedder but another-model is configured; nothing was"):
        index_vault(vault, store, embedder)
    assert store.get_meta("embedder") == "fake-embedder" and store.stats()["files"] == 3


def test_an_index_that_never_recorded_the_window_adopts_it(vault, store, embedder):
    embedder.max_seq = 2048
    index_vault(vault, store, embedder)
    fingerprint = store.get_meta("embedding_config")
    with store.db:
        store.db.execute("DELETE FROM meta WHERE key = 'embedding_config'")
    before = embedder.texts_embedded
    assert index_vault(vault, store, embedder).unchanged == 3  # upgrading must not force a rebuild
    assert embedder.texts_embedded == before
    assert store.get_meta("embedding_config") == fingerprint


def test_unchanged_notes_write_their_stat_only_when_it_moved(vault, store, embedder, monkeypatch):
    index_vault(vault, store, embedder)
    calls = []
    real = store.update_file_metadata
    monkeypatch.setattr(store, "update_file_metadata", lambda *a: calls.append(a[0]) or real(*a))
    assert index_vault(vault, store, embedder).unchanged == 3
    assert calls == []
    rel = min(store.file_hashes())
    note = vault / rel
    os.utime(note, (note.stat().st_atime, note.stat().st_mtime + 60))
    assert index_vault(vault, store, embedder).unchanged == 3
    assert calls == [rel]


def test_a_local_model_path_spelled_with_a_tilde_matches_its_absolute_form(vault, store, embedder):
    index_vault(vault, store, _Wrapped(embedder, name="~/models/embedder"))
    stats = index_vault(vault, store, _Wrapped(embedder, name=os.path.expanduser("~/models/embedder")))
    assert stats.unchanged == stats.scanned == 3  # the same model: nothing is cleared or embedded again


def test_a_failed_window_change_rebuild_preserves_the_index(vault, store, embedder, monkeypatch):
    embedder.max_seq = 2048
    index_vault(vault, store, embedder)
    fingerprint = store.get_meta("embedding_config")
    chunks, vectors = store.load_index()
    embedder.max_seq = 1024
    monkeypatch.setattr(embedder, "embed_documents", _boom)
    with pytest.raises(EmbedderError, match="could not load the model"):
        index_vault(vault, store, embedder, rebuild=True)
    assert store.get_meta("embedding_config") == fingerprint
    assert store.load_index()[0] == chunks
    np.testing.assert_array_equal(store.load_index()[1], vectors)


def test_an_empty_index_can_adopt_the_embedding_config(vault, store, embedder):
    store.set_meta("embedder", embedder.name)
    embedder.max_seq = 1024
    assert index_vault(vault, store, embedder).added == 3
    assert store.get_meta("embedding_config") == '{"embed_max_seq": 1024}'


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root reads everything")
def test_a_real_unreadable_directory_keeps_its_notes(vault, store, embedder):
    index_vault(vault, store, embedder)
    (vault / "projects").chmod(0)
    try:
        stats = index_vault(vault, store, embedder)
    finally:
        (vault / "projects").chmod(0o755)
    assert stats.removed == 0 and store.stats()["files"] == 3


def test_an_empty_vault_does_not_wipe_the_index(vault, store, embedder):
    index_vault(vault, store, embedder)
    for path in discover(vault):
        path.unlink()
    stats = index_vault(vault, store, embedder)
    assert stats.removed == 0 and store.stats()["files"] == 3
    assert "no Markdown files were found" in stats.removal_skipped and "removal skipped" in stats.summary()
    assert index_vault(vault, store, embedder, rebuild=True).scanned == 0 and store.stats()["files"] == 0


# ---- the guards run before anything is destroyed


def test_the_vault_guard_runs_before_a_rebuild_for_a_new_embedder(vault, store, embedder, tmp_path):
    index_vault(vault, store, embedder)
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.md").write_text("# X\n\nhello\n")
    with pytest.raises(ValueError, match="belongs to"):
        index_vault(other, store, _Wrapped(embedder, name="another-model"))
    assert store.stats()["files"] == 3 and store.get_meta("embedder") == "fake-embedder"


def test_a_model_that_cannot_run_does_not_destroy_the_index(vault, store, embedder):
    index_vault(vault, store, embedder)
    before = store.stats()
    typo = _Wrapped(embedder, _boom, name="Qwen/Qwen3-Embeding-0.6B")  # a mistyped model name
    with pytest.raises(ModelMismatch):
        index_vault(vault, store, typo)
    with pytest.raises(EmbedderError, match="could not load the model"):
        index_vault(vault, store, typo, rebuild=True)
    assert store.stats() == before and store.get_meta("embedder") == "fake-embedder"
    with pytest.raises(EmbedderError):
        index_vault(vault, store, _Wrapped(embedder, _boom), rebuild=True)
    assert store.stats() == before


def test_rebuild_repoints_an_index_at_a_moved_vault(vault, store, embedder, tmp_path):
    index_vault(vault, store, embedder)
    moved = tmp_path / "moved"
    vault.rename(moved)
    with pytest.raises(ValueError, match="--rebuild"):
        index_vault(moved, store, embedder)
    assert index_vault(moved, store, embedder, rebuild=True).updated == 3
    assert store.get_meta("vault_root") == str(moved.resolve())


def test_no_probe_is_embedded_when_nothing_is_rebuilt(vault, store, embedder):
    index_vault(vault, store, embedder)
    calls = embedder.doc_calls
    index_vault(vault, store, embedder)
    assert embedder.doc_calls == calls


# ---- titles and decoding


def test_a_utf8_bom_does_not_hide_the_first_heading(vault, store, embedder):
    (vault / "bom.md").write_bytes(b"\xef\xbb\xbf# Project\n" + b"p" * 120 + b"\n\n## Setup\n" + b"s" * 120 + b"\n")
    index_vault(vault, store, embedder)
    assert [r.title for r in store.list_files() if r.path == "bom.md"] == ["Project"]
    paths = [c.heading_path for c in store.load_index()[0] if c.path == "bom.md"]
    assert paths == ["Project", "Project > Setup"]


def test_title_of_ignores_frontmatter_comments_and_a_bom():
    text = "---\n# Config for the publisher\ntitle: Real Title\n---\n# Actual Heading\n"
    assert title_of(text, "stem") == "Actual Heading"
    assert title_of("\ufeff# My Note\n", "stem") == "My Note"
    assert title_of("---\n# only a comment\n---\n", "stem") == "stem"
    assert title_of("# " + "t" * 500, "stem") == "t" * 197 + "..."


# ---- throughput and consistency


def test_embedding_is_batched_across_notes_and_every_note_keeps_its_own_vectors(tmp_path, store, embedder):
    many = tmp_path / "many"
    many.mkdir()
    for i in range(100):
        (many / f"n{i:03d}.md").write_text(f"# Note {i}\n\nunique body text number {i}\n")
    stats = index_vault(many, store, embedder)
    assert stats.added == 100 and embedder.texts_embedded == 100
    assert embedder.doc_calls == 2 and EMBED_BATCH == 64  # a full batch of 64, then the remaining 36
    chunks, matrix = store.load_index()
    for chunk, vector in zip(chunks, matrix):
        assert np.allclose(vector, embedder.embed_query(chunk.content), atol=1e-2)


def test_a_note_edited_while_it_was_embedded_is_stored_as_it_is_on_disk(vault, store, embedder):
    note = vault / "projects" / "garden.md"
    edits = []

    def edit_once(texts):
        if not edits and any("tomatoes" in t for t in texts):
            edits.append(note.write_text("# Garden plan\n\nPlant carrots in April instead.\n"))

    stats = index_vault(vault, store, _Wrapped(embedder, edit_once))
    assert edits and stats.added == 3
    assert store.file_hashes()["projects/garden.md"] == hashlib.sha256(note.read_bytes()).hexdigest()
    rows = _rows(store, "projects/garden.md")
    assert any("carrots" in c for c in rows) and not any("tomatoes" in c for c in rows)


def test_a_note_that_keeps_changing_keeps_its_previous_version(vault, store, embedder):
    index_vault(vault, store, embedder)
    note = vault / "projects" / "garden.md"
    note.write_text("# Garden plan\n\nversion two\n")
    versions = iter(range(3, 100))

    def keep_editing(texts):
        note.write_text(f"# Garden plan\n\nversion {next(versions)}\n")

    stats = index_vault(vault, store, _Wrapped(embedder, keep_editing))
    assert any("changed while it was being indexed" in item for item in stats.skipped)
    assert (stats.updated, stats.removed) == (0, 0)
    assert any("tomatoes" in c for c in _rows(store, "projects/garden.md"))  # the previous version stays
    assert next(versions) == 6  # three attempts embedded versions 3, 4 and 5


def test_a_note_edited_back_to_the_indexed_text_needs_no_update(vault, store, embedder):
    index_vault(vault, store, embedder)
    note = vault / "projects" / "garden.md"
    original = note.read_text()
    note.write_text("# Garden plan\n\ntemporary text\n")

    def restore(texts):
        note.write_text(original)
        touched = time.time() + 60
        os.utime(note, (touched, touched))

    stats = index_vault(vault, store, _Wrapped(embedder, restore))
    assert (stats.updated, stats.unchanged) == (0, 3)
    row = store.recent(0, 1)[0]
    assert (row.path, row.mtime, row.size) == ("projects/garden.md", note.stat().st_mtime, note.stat().st_size)


# ---- bounded work per note


def test_a_note_of_many_small_sections_is_packed_instead_of_exploding(tmp_path, store, embedder):
    v = tmp_path / "v"
    v.mkdir()
    (v / "glossary.md").write_text("".join(f"# t{i}\n{'d' * 100}\n" for i in range(12000)))
    stats = index_vault(v, store, embedder)
    assert 0 < stats.chunks < indexer.MAX_CHUNKS_PER_FILE and not stats.skipped
    assert "d" * 100 in store.load_index()[0][0].content


def test_a_note_over_the_chunk_limit_is_cut_and_reported(tmp_path, store, embedder, monkeypatch):
    monkeypatch.setattr(indexer, "MAX_CHUNKS_PER_FILE", 3)
    v = tmp_path / "v"
    v.mkdir()
    (v / "big.md").write_text("".join(f"# t{i}\n{'d' * 100}\n" for i in range(100)))
    stats = index_vault(v, store, embedder)
    assert stats.chunks == 3
    assert any("big.md (only the first 3 of " in item for item in stats.skipped)


# ---- stopping between notes


def test_a_run_asked_to_stop_ends_between_notes_and_removes_nothing(vault, store, embedder):
    index_vault(vault, store, embedder)
    (vault / "projects" / "garden.md").unlink()
    asked = iter([False, True])  # let one note through, then ask to stop
    stats = index_vault(vault, store, embedder, should_stop=lambda: next(asked, True))
    assert stats.stopped and stats.scanned == 1 and stats.removed == 0 and store.stats()["files"] == 3
    assert stats.summary().endswith(", stopped early")
    assert index_vault(vault, store, embedder).removed == 1  # the next full run catches up


def test_a_run_asked_to_stop_first_does_nothing(vault, store, embedder):
    stats = index_vault(vault, store, embedder, should_stop=lambda: True)
    assert stats.stopped and stats.scanned == 0 and embedder.doc_calls == 0


# ---- what deleted and edited notes leave in the index files


def _copies(directory: Path, needle: bytes) -> int:
    return sum(f.read_bytes().count(needle) for f in directory.iterdir() if f.is_file())


@pytest.fixture
def private_index(tmp_path):
    store = Store(tmp_path / "idx" / "index.sqlite")
    yield store
    store.close()


def test_a_run_purges_deleted_and_replaced_text_from_the_files(vault, private_index, embedder):
    home = Path(private_index.path).parent
    (vault / "secret.md").write_text("# Secret\n\nthe code is zqxcanary7f3a9\n")
    (vault / "projects" / "garden.md").write_text("# Garden\n\nold plan wqvcanary2b8c\n")
    index_vault(vault, private_index, embedder)
    assert _copies(home, b"zqxcanary7f3a9") > 0  # the check can see the text while it is indexed
    (vault / "secret.md").unlink()
    (vault / "projects" / "garden.md").write_text("# Garden\n\nnew plan\n")
    stats = index_vault(vault, private_index, embedder)
    assert (stats.removed, stats.updated) == (1, 1)
    assert _copies(home, b"zqxcanary7f3a9") == 0 and _copies(home, b"wqvcanary2b8c") == 0
    assert (home / "index.sqlite-wal").stat().st_size == 0


@pytest.mark.parametrize("how", ["rebuild", "new model"])
def test_a_rebuild_leaves_nothing_of_the_old_index_in_the_files(vault, private_index, embedder, how):
    home = Path(private_index.path).parent
    (vault / "secret.md").write_text("# Secret\n\nthe code is zqxcanary7f3a9\n")
    index_vault(vault, private_index, embedder)
    (vault / "secret.md").unlink()
    if how == "new model":
        embedder.name = "another-model"
    stats = index_vault(vault, private_index, embedder, rebuild=True)
    assert stats.added + stats.updated == 3
    assert _copies(home, b"zqxcanary7f3a9") == 0


def test_a_reader_holding_the_log_gets_a_warning_instead_of_a_stall(vault, private_index, embedder, caplog):
    index_vault(vault, private_index, embedder)
    reader = sqlite3.connect(private_index.path)
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM files").fetchone()  # keeps an old snapshot open
    (vault / "projects" / "garden.md").unlink()
    try:
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="puremind.index"):
            stats = index_vault(vault, private_index, embedder)
        took = time.monotonic() - started
    finally:
        reader.close()
    assert stats.removed == 1 and took < 10  # the checkpoint does not wait out the 30 s lock timeout
    assert any("until the next run" in r.getMessage() for r in caplog.records)
    assert private_index.checkpoint()  # with the reader gone, the log is emptied


def test_a_run_can_leave_the_keyword_index_compaction_to_the_next_purging_run(vault, private_index, embedder):
    home = Path(private_index.path).parent
    (vault / "secret.md").write_text("# Secret\n\nthe code is zqxcanary7f3a9\n")
    index_vault(vault, private_index, embedder)
    (vault / "secret.md").unlink()
    stats = index_vault(vault, private_index, embedder, purge=False)
    assert stats.removed == 1 and stats.purge_deferred and private_index.purge_pending()
    assert not any("zqxcanary7f3a9" in c.content for c in private_index.load_index()[0])  # the text itself is gone
    stats = index_vault(vault, private_index, embedder)  # nothing changed, but the compaction is owed
    assert (stats.removed, stats.updated, stats.purge_deferred) == (0, 0, False)
    assert not private_index.purge_pending() and _copies(home, b"zqxcanary7f3a9") == 0
