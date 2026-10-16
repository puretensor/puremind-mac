import numpy as np
import pytest

from puremind_mac.store import Store


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
