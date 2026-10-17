"""Several threads on one HybridIndex, Service or Store, as MCP tool handlers are on mcp 2.x.

The interleavings that matter are forced with events and barriers, so those tests fail every
time on a racy implementation. The two stress tests do a fixed amount of work instead of
running for a fixed time.
"""

from __future__ import annotations

import asyncio
import collections
import threading
import time

import numpy as np
import pytest
from fakes import FakeReranker

from puremind_mac.indexer import index_vault
from puremind_mac.retrieval import HybridIndex
from puremind_mac.server import build_server
from puremind_mac.service import Service
from puremind_mac.store import Store


def capture(fn):
    """The result of ``fn()``, or the exception it raised."""
    try:
        return fn()
    except BaseException as exc:
        return exc


def run_together(calls, timeout=30):
    """Start every callable on its own thread at the same moment; return their results in order."""
    barrier = threading.Barrier(len(calls))
    out: list = [None] * len(calls)

    def go(i, call):
        barrier.wait()
        out[i] = capture(call)

    threads = [threading.Thread(target=go, args=(i, c), daemon=True) for i, c in enumerate(calls)]
    run_threads(threads, timeout)
    return out


def run_threads(threads, timeout=60):
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    assert not any(t.is_alive() for t in threads), "a thread is still running"


def slow_corpus_tokenizer(monkeypatch, delay=0.15):
    """Make building a BM25 index slow, so every other thread arrives while it is being built."""
    real = HybridIndex._tokenize
    builds = []

    def tokenize(self, docs, query=False):
        if not query:  # the corpus
            builds.append(len(docs))
            time.sleep(delay)
        return real(self, docs, query)

    monkeypatch.setattr(HybridIndex, "_tokenize", tokenize)
    return builds


# ---- the BM25 index is shared, and built lazily

DOCS = [f"note {i} about alpha beta gamma topic{i % 7} widget{i}" for i in range(60)]


def test_concurrent_first_searches_get_a_complete_bm25_index_built_once(monkeypatch):
    expected = HybridIndex(DOCS, None).lexical("alpha topic3", 5)
    assert expected
    builds = slow_corpus_tokenizer(monkeypatch)
    idx = HybridIndex(DOCS, None)
    got = run_together([lambda: idx.lexical("alpha topic3", 5)] * 8)
    assert got == [expected] * 8  # no AttributeError from a half-built index, and every thread agrees
    assert len(builds) == 1  # and the others waited for the builder instead of building their own


def test_searches_that_start_while_the_index_is_being_built_wait_for_it():
    # The first thread is inside the (slow) build when the second one starts. The old code had
    # already published the unfinished BM25 object by then.
    idx = HybridIndex(DOCS, None)
    real = idx._tokenize
    inside, finish = threading.Event(), threading.Event()

    def tokenize(docs, query=False):
        if not query:
            inside.set()
            assert finish.wait(20)
        return real(docs, query)

    idx._tokenize = tokenize
    first = threading.Thread(target=lambda: idx.lexical("alpha topic3", 5), daemon=True)
    first.start()
    assert inside.wait(20)
    second = {}
    t2 = threading.Thread(target=lambda: second.update(r=capture(lambda: idx.lexical("alpha topic3", 5))), daemon=True)
    t2.start()
    t2.join(0.3)  # long enough for the second thread to reach the index if nothing stops it
    finish.set()
    first.join(20)
    t2.join(20)
    assert second["r"] == HybridIndex(DOCS, None).lexical("alpha topic3", 5)


def test_overlapping_first_searches_on_a_service_reload_and_build_once(cfg, store, vault, embedder, monkeypatch):
    index_vault(vault, store, embedder)
    queries = ["espresso grinder", "tomatoes compost", "hermetic containers", "backflush detergent"] * 2
    reference = Service(cfg, store=store, embedder=embedder, reranker=FakeReranker())
    expected = [[(h.path, h.content) for h in reference.search(q)] for q in queries]
    assert all(expected)

    builds = slow_corpus_tokenizer(monkeypatch)
    loads = []
    real_load = store.load_index
    store.load_index = lambda: (loads.append(1), real_load())[1]
    svc = Service(cfg, store=store, embedder=embedder, reranker=FakeReranker())
    got = run_together([lambda q=q: [(h.path, h.content) for h in svc.search(q)] for q in queries])
    assert got == expected
    assert len(loads) == 1 and len(builds) == 1


def test_parallel_tool_calls_do_not_fail(service, vault, monkeypatch):
    # With mcp 2.x each call runs on a worker thread; with mcp 1.x they run one after another.
    index_vault(vault, service.store, service.embedder)
    slow_corpus_tokenizer(monkeypatch)
    server = build_server(service)
    queries = ["espresso grinder", "tomatoes compost", "hermetic containers", "backflush detergent"] * 2

    async def call_all():
        return await asyncio.gather(
            *(server.call_tool("puremind_search", {"query": q}) for q in queries), return_exceptions=True
        )

    for query, result in zip(queries, asyncio.run(call_all())):
        assert not isinstance(result, BaseException), repr(result)
        blocks = getattr(result, "content", None)
        if blocks is None:
            blocks = result[0] if isinstance(result, tuple) else result
        text = "".join(getattr(c, "text", "") for c in blocks)
        assert "result(s)" in text and "Error" not in text, (query, text[:200])


# ---- a search keeps the snapshot it started with


class GatedReranker(FakeReranker):
    """Holds a search inside the reranker until the test lets it go."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def score(self, query, passages):
        self.entered.set()
        assert self.release.wait(20), "the test never released the reranker"
        return super().score(query, passages)


def search_during(service, change, limit):
    """Start a reranked search, then run ``change`` and a second search that reloads the index while
    the first is in the reranker. Returns the first search's hits and the hits it should have."""
    query = "espresso grinder ethiopian"
    expected = as_tuples(service.search(query, limit=limit))
    assert expected and expected[0][0] == "projects/espresso.md"
    gate = GatedReranker()
    service.reranker = gate
    outcome = {}
    first = threading.Thread(
        target=lambda: outcome.update(r=capture(lambda: service.search(query, limit=limit))), daemon=True
    )
    first.start()
    assert gate.entered.wait(20)
    change()  # commits a new generation while the first search is mid-flight
    service.search("tomatoes compost", rerank=False)  # another request reloads the shared index
    gate.release.set()
    first.join(20)
    assert not first.is_alive()
    assert not isinstance(outcome["r"], BaseException), repr(outcome["r"])
    return as_tuples(outcome["r"]), expected


def as_tuples(hits):
    return [(h.path, h.heading_path, h.chunk_index, h.content) for h in hits]


@pytest.mark.parametrize("limit", [1, 10])
def test_a_reload_during_a_search_does_not_change_which_notes_it_returns(service, vault, store, limit):
    index_vault(vault, store, service.embedder)
    # Dropping the note that was indexed first moves every later chunk up one position. With
    # limit=1 the old code answered with another chunk of the note; with 10 it ran off the end.
    got, expected = search_during(service, lambda: store.delete_file("journal/2026-10-01.md"), limit)
    assert got == expected


def test_a_reload_that_empties_the_index_during_a_search_does_not_raise(service, vault, store):
    index_vault(vault, store, service.embedder)
    got, expected = search_during(service, store.clear, 10)  # every old position is now out of range
    assert got == expected


def test_searches_stay_correct_while_the_vault_is_reindexed(cfg, vault, store, embedder):
    # A note that never changes, behind a filler note that keeps leaving and coming back, so the
    # stable note moves around in the chunk list. Every search for it must find it.
    stable = vault / "projects" / "stable.md"
    stable.write_text("# Stable\n\nThe zanzibar protocol handshake needs a quartz crystal.\n")
    filler = vault / "00-filler.md"
    filler.write_text("# Filler\n\nNothing to see here, move along.\n")
    index_vault(vault, store, embedder)
    svc = Service(cfg, store=store, embedder=embedder, reranker=FakeReranker())
    done = threading.Event()
    problems = []

    def writer():
        try:
            for i in range(40):
                if filler.exists():
                    filler.unlink()
                else:
                    filler.write_text(f"# Filler\n\nNothing to see here, round {i}.\n")
                index_vault(vault, store, embedder)
        except BaseException as exc:
            problems.append(("writer", repr(exc)))
        finally:
            done.set()

    def reader():
        searches = 0
        while searches < 80 or not done.is_set():  # a minimum amount of searching, however fast the writer is
            searches += 1
            try:
                hits = svc.search("zanzibar protocol quartz", limit=3)
                if not hits or hits[0].path != "projects/stable.md" or "zanzibar" not in hits[0].content:
                    problems.append(("wrong hit", [(h.path, h.content[:30]) for h in hits[:1]]))
            except BaseException as exc:
                problems.append(("reader", repr(exc)))

    readers = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
    run_threads([threading.Thread(target=writer, daemon=True), *readers])
    assert not problems, problems[:3]


# ---- one SQLite connection shared by readers and a writer

FILES, WRITES, READERS, MIN_READS = 5, 120, 4, 200


def test_readers_never_see_torn_or_wrong_results_while_a_writer_commits(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    store.set_meta("embedder", "model-a")
    store.set_meta("vault_root", "/vault")
    done = threading.Event()
    problems: list = []

    def writer():
        try:
            for i in range(WRITES):
                n = 3 if (i // FILES) % 2 == 0 else 5  # every file alternates between 3 and 5 chunks
                chunks = [("H", f"v{i} chunk{j} alpha") for j in range(n)]
                vectors = np.ones((n, 8), dtype=np.float32)
                store.replace_file(f"f{i % FILES}.md", f"h{i}", float(i), 1, "t", chunks, vectors)
        except BaseException as exc:
            problems.append(("writer", repr(exc)))
        finally:
            done.set()

    def check_once(last_generation):
        if store.get_meta("embedder") != "model-a" or store.get_meta("vault_root") != "/vault":
            problems.append(("meta value lost", store.get_meta("embedder"), store.get_meta("vault_root")))
        gen = store.generation()
        if gen < last_generation:
            problems.append(("generation went backwards", last_generation, gen))
        chunks, vectors = store.load_index()
        if len(chunks) != len(vectors):
            problems.append(("chunks and vectors differ", len(chunks), len(vectors)))
        by_file = collections.defaultdict(list)
        for c in chunks:
            by_file[c.path].append(c)
        for path, rows in by_file.items():
            if sorted(r.chunk_index for r in rows) not in (list(range(3)), list(range(5))):
                problems.append(("partial file", path, sorted(r.chunk_index for r in rows)))
            if len({r.content.split()[0] for r in rows}) != 1:
                problems.append(("old and new text of one file", path))
        stats = store.stats()
        if stats["chunks"] != stats["vectors"]:
            problems.append(("stats disagree", stats))
        paths = [r.path for r in store.recent(0.0, 50)]
        if len(paths) != len(set(paths)):
            problems.append(("recent repeats a file", paths))
        for chunk, _ in store.fts_hits("alpha", 50):
            if "alpha" not in chunk.content:
                problems.append(("fts hit without its text", chunk.id))
        return gen

    def reader():
        last, reads = -1, 0
        while reads < MIN_READS or not done.is_set():  # keep reading after the writer is done, too
            reads += 1
            try:
                last = check_once(last)
            except BaseException as exc:
                problems.append(("reader", repr(exc)))

    try:
        readers = [threading.Thread(target=reader, daemon=True) for _ in range(READERS)]
        run_threads([threading.Thread(target=writer, daemon=True), *readers])
        assert not problems, problems[:5]
        assert store.stats()["files"] == FILES
    finally:
        store.close()


def test_threads_asking_for_the_same_row_each_get_the_right_answer(tmp_path):
    # Many threads running the same SQL text on one connection: no thread may see another's row.
    with Store(tmp_path / "index.sqlite") as store:
        for i in range(8):
            store.set_meta(f"key{i}", f"value{i}")
        calls = [
            (lambda i=i: [store.get_meta(f"key{i % 8}") for _ in range(1500)])
            for i in range(8)
        ]
        got = run_together(calls)
        for i, result in enumerate(got):
            assert result == [f"value{i % 8}"] * 1500
