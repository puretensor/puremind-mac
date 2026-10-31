import os
import random
import shutil
import sqlite3
import stat
import threading

import numpy as np
import pytest

from puremind_mac.store import SCHEMA, SCHEMA_VERSION, Store, StoreError


def vec(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def put(store, path="a.md", sha="h1", chunks=(("H", "alpha beta"),), vectors=None, mtime=100.0):
    if vectors is None:
        vectors = np.stack([vec(1, i + 1) for i in range(len(chunks))]) if chunks else None
    store.replace_file(path, sha, mtime, 10, path, list(chunks), vectors)


def test_schema_created_in_file(tmp_path):
    with Store(tmp_path / "x" / "db.sqlite") as s:
        assert s.stats() == {"files": 0, "chunks": 0, "vectors": 0}
    assert (tmp_path / "x" / "db.sqlite").exists()


def test_meta_roundtrip_and_upsert(store):
    assert store.get_meta("k") is None
    assert store.get_meta("k", "d") == "d"
    store.set_meta("k", "1")
    store.set_meta("k", "2")
    assert store.get_meta("k") == "2"


def test_replace_file_inserts_chunks_and_vectors(store):
    put(store, chunks=(("A", "one"), ("A > B", "two")))
    assert store.stats() == {"files": 1, "chunks": 2, "vectors": 2}
    assert store.file_hashes() == {"a.md": "h1"}


def test_replace_file_replaces_previous_chunks(store):
    put(store, chunks=(("A", "one"), ("B", "two")))
    put(store, sha="h2", chunks=(("C", "three"),))
    assert store.stats() == {"files": 1, "chunks": 1, "vectors": 1}
    assert store.file_hashes()["a.md"] == "h2"


def test_vectors_roundtrip_as_float16(store):
    v = vec(3, 4, 12)
    put(store, chunks=(("H", "x"),), vectors=v[None, :])
    _, mat = store.load_index()
    assert mat.dtype == np.float32 and mat.shape == (1, 3)
    assert np.allclose(mat[0], v, atol=1e-3)
    raw = store.db.execute("SELECT vec FROM vectors").fetchone()[0]
    assert len(raw) == 3 * 2  # float16


def test_vector_count_must_match_chunk_count(store):
    with pytest.raises(ValueError):
        store.replace_file("a.md", "h", 1.0, 1, "t", [("H", "x"), ("H", "y")], np.ones((1, 2), dtype=np.float32))


def test_failed_replace_rolls_back(store):
    put(store, chunks=(("A", "keep me"),))
    with pytest.raises(ValueError):
        store.replace_file("a.md", "h2", 1.0, 1, "t", [("H", "x"), ("H", "y")], np.ones((1, 2), dtype=np.float32))
    assert store.file_hashes() == {"a.md": "h1"}
    assert store.stats()["chunks"] == 1


def test_delete_file_cascades(store):
    put(store, chunks=(("A", "one"), ("B", "two")))
    store.delete_file("a.md")
    assert store.stats() == {"files": 0, "chunks": 0, "vectors": 0}
    assert store.fts_search("one") == []


def test_metadata_refresh_changes_only_the_matching_file_stat(store):
    path = "it's a note.md"
    put(store, path=path)
    put(store, path="other.md")
    chunks, vectors = store.load_index()
    generation = store.generation()
    store.update_file_metadata(path, "h1", 200.0, 42)
    rows = {r.path: r for r in store.list_files()}
    assert (rows[path].mtime, rows[path].size, rows[path].sha256) == (200.0, 42, "h1")
    assert (rows["other.md"].mtime, rows["other.md"].size) == (100.0, 10)
    assert store.load_index()[0] == chunks
    np.testing.assert_array_equal(store.load_index()[1], vectors)
    assert store.generation() == generation
    store.update_file_metadata(path, "stale hash", 300.0, 99)
    store.update_file_metadata("missing.md", "h1", 300.0, 99)
    assert {r.path: r for r in store.list_files()} == rows


def test_file_with_no_chunks(store):
    put(store, chunks=(), vectors=None)
    assert store.stats() == {"files": 1, "chunks": 0, "vectors": 0}
    assert store.load_index()[0] == []


def test_generation_bumps_on_every_write(store):
    g0 = store.generation()
    put(store)
    g1 = store.generation()
    store.delete_file("a.md")
    assert g0 < g1 < store.generation()


def test_fts_search_finds_stemmed_terms_and_ranks(store):
    put(store, "a.md", chunks=(("H", "gardening tips for tomatoes"),))
    put(store, "b.md", chunks=(("H", "tomatoes tomatoes tomatoes and more tomatoes"),))
    put(store, "c.md", chunks=(("H", "unrelated text about boilers"),))
    hits = store.fts_search("tomato garden")
    assert len(hits) == 2
    assert all(score > 0 for _, score in hits)
    assert {c.path for c in store.get_chunks([i for i, _ in hits])} == {"a.md", "b.md"}


def test_fts_search_survives_punctuation_and_empty_queries(store):
    put(store, chunks=(("H", "hello world"),))
    assert store.fts_search('"; DROP TABLE x; --') == []
    assert store.fts_search("   ") == []
    assert store.fts_search("hello (world)")


def test_fts_matches_heading_path(store):
    put(store, chunks=(("Quarterly Planning", "body text"),))
    assert store.fts_search("quarterly")


def test_recent_filters_and_orders(store):
    put(store, "old.md", mtime=100.0)
    put(store, "mid.md", mtime=200.0)
    put(store, "new.md", mtime=300.0)
    assert [r.path for r in store.recent(150.0, 10)] == ["new.md", "mid.md"]
    assert [r.path for r in store.recent(0.0, 1)] == ["new.md"]


def test_persists_across_connections(tmp_path):
    with Store(tmp_path / "db.sqlite") as s:
        put(s)
    with Store(tmp_path / "db.sqlite") as s:
        assert s.stats()["chunks"] == 1


def test_clear(store):
    put(store, "a.md")
    put(store, "b.md")
    store.clear()
    assert store.stats()["files"] == 0


# ---- transactions

def test_a_failing_transaction_rolls_back_and_frees_the_connection(store):
    put(store, chunks=(("A", "keep me"),))
    before = store.generation()
    with pytest.raises(sqlite3.IntegrityError), store.db:
        store.db.execute("DELETE FROM files")
        store._bump()
        store.db.execute("INSERT INTO files(path) VALUES ('no-other-columns')")
    assert store.stats() == {"files": 1, "chunks": 1, "vectors": 1} and store.generation() == before
    other = threading.Thread(target=store.stats)  # the lock was released, so another thread is not stuck
    other.start()
    other.join(10)
    assert not other.is_alive()


def test_a_transaction_block_is_not_interleaved_with_another_threads_reads(store):
    put(store, chunks=(("A", "one"),))
    inside, release = threading.Event(), threading.Event()
    seen = []

    def writer():
        with store.db:
            store.db.execute("INSERT INTO meta(key, value) VALUES ('half', 'done')")
            inside.set()
            assert release.wait(10)
            store.db.execute("INSERT INTO meta(key, value) VALUES ('other half', 'done')")

    t = threading.Thread(target=writer)
    t.start()
    assert inside.wait(10)
    reader = threading.Thread(target=lambda: seen.append((store.get_meta("half"), store.get_meta("other half"))))
    reader.start()
    reader.join(0.3)
    assert reader.is_alive()  # waits for the transaction instead of reading half of it
    release.set()
    t.join(10)
    reader.join(10)
    assert seen == [("done", "done")]


# ---- one statement per read

def statements_of(store):
    """Record the SQL text of every statement the store runs from now on."""
    seen = []
    real = store.db.execute

    def spy(sql, *a, **k):
        seen.append(sql)
        return real(sql, *a, **k)

    store.db.execute = spy
    return seen


def test_fts_hits_returns_the_chunk_rows_ranked_like_fts_search(store):
    put(store, "a.md", chunks=(("Garden", "gardening tips for tomatoes"), ("Other", "boilers")))
    put(store, "b.md", chunks=(("H", "tomatoes tomatoes tomatoes and more tomatoes"),))
    hits = store.fts_hits("tomato garden")
    assert [(c.id, s) for c, s in hits] == store.fts_search("tomato garden")
    assert store.get_chunks([c.id for c, _ in hits]) == [c for c, _ in hits]
    assert {(c.path, c.heading_path) for c, _ in hits} == {("a.md", "Garden"), ("b.md", "H")}
    assert all(s > 0 for _, s in hits)
    assert store.fts_hits("   ") == [] and store.fts_hits('"; DROP TABLE x; --') == []


def test_fts_hits_matches_and_reads_the_chunks_in_one_statement(store):
    put(store, chunks=(("H", "hello world"),))
    seen = statements_of(store)
    assert store.fts_hits("hello")
    assert len(seen) == 1


def test_stats_counts_in_one_statement(store):
    put(store, chunks=(("A", "one"), ("B", "two")))
    seen = statements_of(store)
    assert store.stats() == {"files": 1, "chunks": 2, "vectors": 2}
    assert len(seen) == 1


def test_get_chunks_keeps_the_order_asked_for_and_skips_missing_ids(store):
    put(store, chunks=tuple(("H", f"chunk number {i}") for i in range(1200)))
    ids = [c.id for c in store.get_chunks(range(1, 5000))]  # more ids than one SQL statement may bind
    assert ids == sorted(ids) and len(ids) == 1200
    asked = [ids[700], ids[3], 999_999, ids[1100], ids[3]]
    assert [c.id for c in store.get_chunks(asked)] == [ids[700], ids[3], ids[1100], ids[3]]
    assert store.get_chunks([]) == []


# ---- results are fixed when execute() returns

def test_execute_result_supports_the_cursor_calls_the_store_uses(store):
    put(store, chunks=(("A", "one"), ("B", "two"), ("C", "three")))
    res = store.db.execute("SELECT content FROM chunks ORDER BY id")
    assert res.fetchone()["content"] == "one"
    assert [r["content"] for r in res.fetchmany(1)] == ["two"]
    assert [r["content"] for r in res] == ["three"]
    assert res.fetchone() is None and res.fetchall() == []
    write = store.db.execute("INSERT INTO meta(key, value) VALUES ('x', 'y')")
    assert write.rowcount == 1 and write.lastrowid and write.fetchall() == []
    store.db.commit()


def test_executemany_and_executescript_give_finished_results_too(store):
    with store.db:
        many = store.db.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", [("a", "1"), ("b", "2")])
    assert many.rowcount == 2 and many.fetchall() == []
    script = store.db.executescript("CREATE TABLE IF NOT EXISTS scratch(x); INSERT INTO scratch VALUES (7);")
    assert script.fetchall() == []
    assert store.db.execute("SELECT x FROM scratch").fetchone()[0] == 7
    assert store.get_meta("b") == "2"


def test_rows_already_fetched_are_not_changed_by_a_later_write(store):
    # The lock used to be released before the rows were read, so a writer on another thread
    # could change what a half-read result went on to return.
    put(store, "a.md", chunks=tuple(("H", f"OLD {i}") for i in range(4)))
    put(store, "b.md", chunks=tuple(("H", f"b {i}") for i in range(4)))
    res = store.db.execute("SELECT path, content FROM chunks ORDER BY id")
    first = res.fetchone()
    new_chunks = tuple(("H", f"NEW {i}") for i in range(6))
    writer = threading.Thread(target=put, args=(store, "a.md"), kwargs={"sha": "h2", "chunks": new_chunks})
    writer.start()
    writer.join(10)
    assert not writer.is_alive()
    rows = [first, *res.fetchall()]
    assert [r["content"] for r in rows if r["path"] == "a.md"] == [f"OLD {i}" for i in range(4)]
    assert len(rows) == 8


# ---- who can read the index

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
not_root = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes")


def mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture(params=[0o022, 0o000, 0o077], ids=lambda m: f"umask{m:03o}")
def umask(request):
    old = os.umask(request.param)
    yield request.param
    os.umask(old)


@posix_only
def test_a_new_index_and_its_directory_are_private(tmp_path, umask):
    db = tmp_path / "fresh" / "home" / "index.sqlite"
    with Store(db) as s:
        put(s)
        assert mode(db.parent) == 0o700
        assert mode(db) == 0o600
        sidecars = [p for p in (db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")) if p.exists()]
        assert sidecars, "WAL mode keeps -wal and -shm while the index is open"
        assert [mode(p) for p in sidecars] == [0o600] * len(sidecars)


@posix_only
def test_an_existing_directory_keeps_its_mode(tmp_path):
    old = os.umask(0o022)
    try:
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o755)
        with Store(shared / "index.sqlite"):
            pass
    finally:
        os.umask(old)
    assert mode(shared) == 0o755
    assert mode(shared / "index.sqlite") == 0o600


@posix_only
@pytest.mark.parametrize("loose", [0o644, 0o664, 0o640, 0o604, 0o666])
def test_an_existing_open_index_is_tightened(tmp_path, loose):
    db = tmp_path / "index.sqlite"
    with Store(db) as first:  # keeps the -wal and -shm files alive
        put(first)
        files = [db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")]
        assert all(p.exists() for p in files)
        for p in files:
            p.chmod(loose)
        with Store(db) as second:
            assert second.stats()["chunks"] == 1
        assert [mode(p) for p in files] == [0o600] * 3


@posix_only
def test_tightening_only_removes_group_and_other_access(tmp_path):
    db = tmp_path / "index.sqlite"
    Store(db).close()
    db.chmod(0o444)
    Store(db).close()
    assert mode(db) == 0o400  # still read-only for its owner, no longer readable by anyone else


@posix_only
def test_an_in_memory_store_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with Store(":memory:") as s:
        put(s)
        assert s.stats()["chunks"] == 1
    assert list(tmp_path.iterdir()) == []


# ---- an index that cannot be written

def freeze(path):
    """Make the directory and everything in it read-only, as on a read-only volume."""
    for p in path.iterdir():
        p.chmod(0o444)
    path.chmod(0o555)


def thaw(path):
    path.chmod(0o755)
    for p in path.iterdir():
        p.chmod(0o644)


def populate(s):
    s.set_meta("embedder", "m")
    put(s, "a.md", chunks=(("Garden", "tomatoes need compost"), ("Other", "boilers")))
    put(s, "b.md", chunks=(("H", "espresso grinder"),), mtime=500.0)


@pytest.fixture
def frozen_copy(tmp_path):
    """A populated index copied into a directory that is then made read-only."""
    src = tmp_path / "src"
    with Store(src / "index.sqlite") as s:
        populate(s)
    copy = tmp_path / "frozen"
    shutil.copytree(src, copy)
    freeze(copy)
    yield copy / "index.sqlite"
    thaw(copy)


def reads_work(s):
    assert s.stats() == {"files": 2, "chunks": 3, "vectors": 3}
    assert s.get_meta("embedder") == "m"
    chunks, mat = s.load_index()
    assert len(chunks) == len(mat) == 3
    assert [r.path for r in s.recent(0.0, 5)] == ["b.md", "a.md"]
    assert [c.heading_path for c, _ in s.fts_hits("tomato")] == ["Garden"]
    assert s.generation() > 0


def writes_refused(s):
    for call in (
        lambda: s.set_meta("k", "v"),
        lambda: put(s, "c.md"),
        lambda: s.update_file_metadata("a.md", "h1", 200.0, 42),
        lambda: s.delete_file("a.md"),
        lambda: s.clear(),
    ):
        with pytest.raises(StoreError, match="read-only"):
            call()
    assert s.stats()["files"] == 2


@posix_only
@not_root
def test_an_index_in_a_read_only_directory_can_be_read_but_not_written(frozen_copy):
    before = sorted(os.listdir(frozen_copy.parent))
    with Store(frozen_copy) as s:
        assert s.read_only
        reads_work(s)
        writes_refused(s)
    assert sorted(os.listdir(frozen_copy.parent)) == before  # no -wal, -shm or journal was created


@posix_only
@not_root
def test_read_only_open_copes_with_awkward_characters_in_the_path(tmp_path):
    home = tmp_path / "my index #1 ? 100% \u00e9 cache"
    with Store(home / "index.sqlite") as s:
        populate(s)
    freeze(home)
    try:
        with Store(home / "index.sqlite") as s:
            assert s.read_only
            reads_work(s)
    finally:
        thaw(home)


@posix_only
@not_root
def test_a_read_only_index_file_in_a_writable_directory(tmp_path):
    db = tmp_path / "index.sqlite"
    with Store(db) as s:
        populate(s)
    db.chmod(0o444)
    try:
        with Store(db) as s:
            assert s.read_only
            reads_work(s)
            writes_refused(s)
    finally:
        db.chmod(0o644)


@posix_only
@not_root
def test_a_read_only_index_that_is_not_in_wal_mode(tmp_path):
    db = tmp_path / "sub" / "index.sqlite"
    with Store(db) as s:
        populate(s)
        s.db.execute("PRAGMA journal_mode = DELETE")
    freeze(db.parent)
    try:
        with Store(db) as s:
            assert s.read_only
            reads_work(s)
            writes_refused(s)
    finally:
        thaw(db.parent)


@posix_only
@not_root
def test_read_only_means_nothing_is_created(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    empty.chmod(0o555)
    try:
        with pytest.raises(StoreError, match=r"no index at .* cannot be created"):
            Store(empty / "index.sqlite")
        assert list(empty.iterdir()) == []
    finally:
        empty.chmod(0o755)


@posix_only
@not_root
def test_a_read_only_file_that_is_not_an_index(tmp_path):
    db = tmp_path / "index.sqlite"
    db.write_bytes(b"")
    db.chmod(0o444)
    try:
        with pytest.raises(StoreError, match="holds no puremind index"):
            Store(db)
    finally:
        db.chmod(0o644)


@posix_only
@not_root
def test_unflushed_wal_changes_are_not_hidden_behind_an_immutable_open(tmp_path):
    # A directory SQLite cannot add the -shm file to, with a -wal that still holds committed rows:
    # reading only the main file would silently show stale data, so the open is refused instead.
    live = tmp_path / "live"
    with Store(live / "index.sqlite") as s:
        put(s, "a.md")
        copy = tmp_path / "copy"
        copy.mkdir()
        for suffix in ("", "-wal"):
            shutil.copy(str(live / "index.sqlite") + suffix, copy / ("index.sqlite" + suffix))
    assert (copy / "index.sqlite-wal").stat().st_size > 0
    freeze(copy)
    try:
        with pytest.raises(StoreError, match=r"cannot open the index .*waiting in its -wal file"):
            Store(copy / "index.sqlite")
    finally:
        thaw(copy)


# ---- format version

def user_version(path) -> int:
    con = sqlite3.connect(path)
    try:
        return con.execute("PRAGMA user_version").fetchone()[0]
    finally:
        con.close()


def test_a_new_index_is_stamped_with_the_schema_version(tmp_path):
    assert SCHEMA_VERSION == 1
    with Store(tmp_path / "index.sqlite"):
        pass
    assert user_version(tmp_path / "index.sqlite") == SCHEMA_VERSION
    with Store(":memory:") as s:
        assert s.db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_an_index_from_before_the_stamp_is_accepted_and_stamped(tmp_path):
    db = tmp_path / "index.sqlite"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    con.execute("INSERT INTO meta VALUES ('embedder', 'old-model')")
    con.commit()
    con.close()
    assert user_version(db) == 0
    with Store(db) as s:
        assert s.get_meta("embedder") == "old-model"
    assert user_version(db) == SCHEMA_VERSION


def test_an_index_from_a_newer_version_is_refused_and_left_alone(tmp_path):
    db = tmp_path / "index.sqlite"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    con.execute("PRAGMA user_version = 7")
    con.commit()
    con.close()
    with pytest.raises(StoreError, match=r"format 7, newer than this puremind understands \(1\).*upgrade puremind"):
        Store(db)
    assert user_version(db) == 7
    assert not (tmp_path / "index.sqlite-wal").exists()
    assert sqlite3.connect(db).execute("PRAGMA journal_mode").fetchone()[0] == "delete"  # not switched to WAL


@posix_only
@not_root
def test_a_newer_index_is_refused_when_opened_read_only_too(tmp_path):
    db = tmp_path / "index.sqlite"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    con.execute("PRAGMA user_version = 2")
    con.commit()
    con.close()
    db.chmod(0o444)
    try:
        with pytest.raises(StoreError, match="newer"):
            Store(db)
    finally:
        db.chmod(0o644)


def _occurrences(directory, needle: bytes) -> int:
    return sum(f.read_bytes().count(needle) for f in directory.iterdir() if f.is_file())


def test_deleted_note_text_does_not_stay_in_the_files(tmp_path):
    d = tmp_path / "idx"
    store = Store(d / "index.sqlite")
    for i in range(30):
        store.replace_file(f"n{i}.md", "h", 0, 1, "t", [("", f"an ordinary note {i} " * 40)], np.ones((1, 4)))
    store.replace_file("secret.md", "h", 0, 1, "t", [("", "the key is zqxcanary7f3a9, keep it safe")], np.ones((1, 4)))
    store.replace_file("edited.md", "h", 0, 1, "t", [("", "old text with wqvcanary2b8c")], np.ones((1, 4)))
    store.replace_file("edited.md", "h2", 0, 1, "t", [("", "new text")], np.ones((1, 4)))
    store.delete_file("secret.md")
    assert _occurrences(d, b"zqxcanary7f3a9") > 0  # until the purge, FTS5 keeps the terms of deleted rows
    assert store.purge()
    assert _occurrences(d, b"zqxcanary7f3a9") == 0 and _occurrences(d, b"wqvcanary2b8c") == 0
    assert [c.content for c, _ in store.fts_hits("ordinary", 50)]  # the keyword index still works
    store.close()
    assert _occurrences(d, b"zqxcanary7f3a9") == 0



def test_an_index_from_0_1_1_gets_the_chunker_column(tmp_path):
    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)
    old_schema = SCHEMA.replace(",\n  chunker INTEGER NOT NULL DEFAULT 0", "")
    assert old_schema != SCHEMA
    con.executescript(old_schema)
    con.execute("INSERT INTO files(path, sha256, mtime, size, indexed_at) VALUES ('a.md', 'h', 0, 1, 0)")
    con.commit()
    con.close()
    with Store(db) as store:
        assert store.chunked_with(0) == {"a.md"}  # so the indexer cuts it again
        put(store, "b.md")
        assert store.chunked_with(0) == {"a.md", "b.md"}  # replace_file records the version it is given


def test_clear_leaves_none_of_the_old_text_in_the_files(tmp_path):
    d = tmp_path / "idx"
    store = Store(d / "index.sqlite")
    for i in range(30):
        store.replace_file(f"n{i}.md", "h", 0, 1, "t", [("", f"an ordinary note {i} " * 40)], np.ones((1, 4)))
    store.replace_file("secret.md", "h", 0, 1, "t", [("", "the key is zqxcanary7f3a9")], np.ones((1, 4)))
    store.clear()
    assert _occurrences(d, b"zqxcanary7f3a9") == 0 and _occurrences(d, b"ordinary") == 0
    assert store.stats() == {"files": 0, "chunks": 0, "vectors": 0}
    put(store, "b.md", chunks=(("H", "gamma delta"),))
    assert [c.path for c, _ in store.fts_hits("gamma")] == ["b.md"]
    store.close()


def test_text_deleted_by_an_earlier_version_is_purged_when_the_index_is_upgraded(tmp_path):
    d = tmp_path / "idx"
    d.mkdir()
    db = d / "index.sqlite"
    con = sqlite3.connect(db)  # written as puremind 0.1.1 did: format 0, freed pages not zeroed
    con.execute("PRAGMA secure_delete = OFF")
    con.execute("PRAGMA journal_mode = WAL").fetchone()
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    with con:
        for i, text in [*((i, f"an ordinary note {i} " * 40) for i in range(30)), (30, "the key is zqxcanary7f3a9")]:
            con.execute("INSERT INTO files(path, sha256, mtime, size, indexed_at) VALUES (?, 'h', 0, 1, 0)", (f"n{i}.md",))
            con.execute("INSERT INTO chunks(path, chunk_index, heading_path, content) VALUES (?, 0, '', ?)", (f"n{i}.md", text))
    with con:
        con.execute("DELETE FROM files WHERE path = 'n30.md'")
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    con.close()
    assert _occurrences(d, b"zqxcanary7f3a9") > 0  # what the earlier version left behind
    with Store(db) as store:
        assert _occurrences(d, b"zqxcanary7f3a9") == 0
        assert store.stats()["files"] == 30
        assert store.db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_heavy_churn_keeps_the_keyword_index_intact(tmp_path):
    # FTS5's own secure-delete option fails this on SQLite 3.42.0 to 3.46.0 (at op 200 with this seed on 3.45.1).
    rnd = random.Random(2)
    words = [f"zq{i}" for i in range(300)]
    weights = [1 / (i + 1) for i in range(300)]  # a few common words, many rare ones, as in prose
    store = Store(tmp_path / "churn" / "index.sqlite")
    for op in range(1, 601):
        path = f"n{rnd.randrange(150)}.md"
        if rnd.random() < 0.6:
            chunks = [(f"h{j}", " ".join(rnd.choices(words, weights, k=rnd.randint(5, 60)))) for j in range(rnd.randint(1, 5))]
            store.replace_file(path, f"h{op}", 0, 1, "t", chunks, np.ones((len(chunks), 4)))
        else:
            store.delete_file(path)
        if op % 50 == 0:
            assert store.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok", op
        if op % 200 == 0:
            assert store.purge()
    with store.db:
        store.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('integrity-check')")  # raises if it is not
    store.close()


def test_clear_drops_pages_an_earlier_version_freed_without_zeroing(tmp_path):
    d = tmp_path / "idx"
    d.mkdir()
    db = d / "index.sqlite"
    with Store(db):
        pass  # the current format, so opening it again does not compact it
    con = sqlite3.connect(db)
    con.execute("PRAGMA secure_delete = OFF")  # as the upgrade compaction would find it, had it failed
    with con:
        con.execute("INSERT INTO files(path, sha256, mtime, size, indexed_at) VALUES ('s.md', 'h', 0, 1, 0)")
        con.execute("INSERT INTO meta(key, value) VALUES ('blob', ?)", ("zqxcanary7f3a9 " * 2000,))
    with con:
        con.execute("DELETE FROM meta WHERE key = 'blob'")
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    con.close()
    assert _occurrences(d, b"zqxcanary7f3a9") > 0
    with Store(db) as store:
        store.clear()
        assert _occurrences(d, b"zqxcanary7f3a9") == 0


def test_an_old_index_whose_column_another_process_just_added_still_opens(tmp_path, monkeypatch):
    from puremind_mac import store as store_module

    path = tmp_path / "old.sqlite"
    Store(path).close()
    checks = iter([False])  # the first check misses the column, as when another process adds it meanwhile
    real = store_module._has_chunker_column
    monkeypatch.setattr(store_module, "_has_chunker_column", lambda conn: next(checks, None) or real(conn))
    Store(path).close()


def test_vectors_of_different_lengths_are_refused(store):
    store.replace_file("a.md", "h", 0.0, 1, "A", [("", "alpha")], np.ones((1, 4), dtype=np.float32))
    store.replace_file("b.md", "h", 0.0, 1, "B", [("", "beta")], np.ones((1, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="same length"):
        store.load_index()


def test_a_deferred_purge_is_recorded_until_a_purge_or_clear(store):
    store.replace_file("a.md", "h", 0.0, 1, "A", [("", "alpha")], np.ones((1, 4), dtype=np.float32))
    assert not store.purge_pending()
    store.defer_purge()
    store.defer_purge()
    assert store.purge_pending()
    store.purge()
    assert not store.purge_pending()
    store.defer_purge()
    store.clear()
    assert not store.purge_pending()


@pytest.mark.parametrize("operation", ["replace", "delete", "clear"])
def test_deleted_text_debt_commits_with_the_change(store, monkeypatch, operation):
    put(store)
    assert not store.purge_pending()
    monkeypatch.setattr(store, "purge", lambda: False)  # stop clear() before it pays the debt
    if operation == "replace":
        put(store, sha="h2", chunks=(("H", "new text"),))
    elif operation == "delete":
        store.delete_file("a.md")
    else:
        store.clear()
    with Store(store.path) as reopened:
        assert reopened.purge_pending()
        assert reopened.file_hashes() == ({"a.md": "h2"} if operation == "replace" else {})


@pytest.mark.parametrize("operation", ["replace", "delete", "clear"])
def test_failed_deletion_rolls_back_rows_and_debt(store, monkeypatch, operation):
    put(store)
    store.set_meta("vault_root", "/old")
    generation = store.generation()

    def fail():
        raise sqlite3.OperationalError("disk is full")

    monkeypatch.setattr(store, "_bump", fail)
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        if operation == "replace":
            put(store, sha="h2")
        elif operation == "delete":
            store.delete_file("a.md")
        else:
            store.clear({"vault_root": "/new"})
    assert store.file_hashes() == {"a.md": "h1"}
    assert not store.purge_pending() and store.generation() == generation
    assert store.get_meta("vault_root") == "/old"


def test_clear_changes_the_vault_root_and_rows_atomically(store, monkeypatch):
    put(store)
    store.set_meta("vault_root", "/old")
    reader = sqlite3.connect(store.path)
    original = store._bump

    def snapshot():
        return reader.execute(
            "SELECT (SELECT value FROM meta WHERE key = 'vault_root'), (SELECT count(*) FROM files)"
        ).fetchone()

    def observe():
        assert snapshot() == ("/old", 1)
        original()

    monkeypatch.setattr(store, "_bump", observe)
    try:
        store.clear({"vault_root": "/new"})
        assert snapshot() == ("/new", 0)
    finally:
        reader.close()


@pytest.mark.parametrize("failure", ["busy", "checkpoint error", "compaction error"])
def test_purge_keeps_debt_until_compaction_and_checkpoint_succeed(store, monkeypatch, failure):
    put(store)
    store.delete_file("a.md")
    execute = store.db.execute

    def checkpoint():
        assert store.purge_pending()
        if failure == "checkpoint error":
            raise sqlite3.OperationalError("checkpoint failed")
        return False

    def fail_compaction(sql, *args):
        if "'optimize'" in sql:
            raise sqlite3.OperationalError("compaction failed")
        return execute(sql, *args)

    with monkeypatch.context() as patch:
        if failure == "compaction error":
            patch.setattr(store.db, "execute", fail_compaction)
        else:
            patch.setattr(store, "checkpoint", checkpoint)
        if failure == "busy":
            assert not store.purge()
        else:
            with pytest.raises(sqlite3.OperationalError, match="failed"):
                store.purge()
    with Store(store.path) as reopened:
        assert reopened.purge_pending()
        assert reopened.purge()
        assert not reopened.purge_pending()


def test_clear_keeps_debt_when_its_checkpoint_is_busy(store, monkeypatch):
    put(store)
    with monkeypatch.context() as patch:
        patch.setattr(store, "checkpoint", lambda: False)
        store.clear()
    assert not store.file_hashes() and store.purge_pending()
    assert store.purge() and not store.purge_pending()


@pytest.mark.parametrize("when", ["before checkpoint", "after checkpoint"])
def test_purge_does_not_clear_a_concurrent_writers_debt(store, monkeypatch, when):
    put(store)
    store.defer_purge()
    checkpoint = store.checkpoint
    with Store(store.path) as writer:
        def write_during_purge():
            if when == "before checkpoint":
                writer.delete_file("a.md")
            assert checkpoint()
            if when == "after checkpoint":
                writer.delete_file("a.md")
            return True

        with monkeypatch.context() as patch:
            patch.setattr(store, "checkpoint", write_during_purge)
            assert store.purge()
    assert store.purge_pending()
    assert store.purge() and not store.purge_pending()
