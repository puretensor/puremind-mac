import numpy as np

from puremind_mac.retrieval import HybridIndex, rrf_fuse

from conftest import FakeEmbedder, FakeReranker

DOCS = [
    "tomatoes need compost and mulch in the raised garden beds",
    "the espresso grinder setting is fourteen for ethiopian beans",
    "migrate the build pipeline to hermetic containers",
    "descale the espresso boiler every three months",
    "weekly backflush keeps the espresso machine clean",
]


def make_index():
    emb = FakeEmbedder()
    return HybridIndex(DOCS, emb.embed_documents(DOCS)), emb


def test_rrf_scores_follow_formula():
    fused = dict(rrf_fuse([[10, 20], [20, 30]], k=60))
    assert fused[10] == 1 / 61
    assert abs(fused[20] - (1 / 62 + 1 / 61)) < 1e-12
    assert fused[30] == 1 / 62


def test_rrf_orders_items_found_by_both_lists_first():
    order = [i for i, _ in rrf_fuse([[1, 2, 3], [3, 2, 9]])]
    assert order[:2] == [3, 2]  # 1/63 + 1/61 edges out 2 * 1/62


def test_rrf_ties_keep_first_seen_order():
    assert [i for i, _ in rrf_fuse([[5], [7]])] == [5, 7]


def test_rrf_empty_and_single_list():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
    assert [i for i, _ in rrf_fuse([[4, 5, 6]])] == [4, 5, 6]


def test_rrf_k_changes_scores():
    assert dict(rrf_fuse([[1]], k=10))[1] == 1 / 11


def test_lexical_matches_stems_and_ignores_stopwords():
    idx, _ = make_index()
    assert idx.lexical("the tomato garden", 5) == [0]


def test_lexical_returns_nothing_for_unknown_or_stopword_only_queries():
    idx, _ = make_index()
    assert idx.lexical("zzzzqqq", 5) == []
    assert idx.lexical("the and of", 5) == []


def test_lexical_on_empty_index():
    assert HybridIndex([], None).lexical("anything", 5) == []


def test_dense_ranks_by_cosine():
    idx, emb = make_index()
    top = idx.dense(emb.embed_query("espresso boiler descale"), 2)
    assert top[0] == 3 and len(top) == 2


def test_search_without_reranker_returns_rrf_scores():
    idx, emb = make_index()
    hits = idx.search("espresso grinder", emb, None, limit=3)
    assert len(hits) == 3
    assert hits[0].position == 1
    assert hits[0].score > hits[1].score >= hits[2].score
    assert hits[0].score <= 2 / 61 + 1e-9


def test_search_with_reranker_orders_by_reranker_and_truncates_passages():
    idx, emb = make_index()
    rr = FakeReranker()
    hits = idx.search("espresso boiler descale", emb, rr, limit=2, rerank_chars=20)
    assert hits[0].position == 3
    query, passages = rr.calls[0]
    assert query == "espresso boiler descale"
    assert all(len(p) <= 20 for p in passages)


def test_reranker_sees_at_most_candidates_passages():
    texts = [f"common word number{i}" for i in range(80)]
    emb = FakeEmbedder()
    idx = HybridIndex(texts, emb.embed_documents(texts))
    rr = FakeReranker()
    idx.search("common word", emb, rr, limit=5, candidates=50)
    assert len(rr.calls[0][1]) <= 50


def test_search_limit_applies():
    idx, emb = make_index()
    assert len(idx.search("espresso", emb, None, limit=1)) == 1


def test_search_empty_index_returns_nothing():
    emb = FakeEmbedder()
    assert HybridIndex([], None).search("x", emb, None) == []


def test_lexical_only_hit_still_surfaces():
    # A document the dense side cannot see (zero vector) is still found by BM25 and fused in.
    docs = ["alpha beta", "gamma delta"]
    vecs = np.zeros((2, 64), dtype=np.float32)
    idx = HybridIndex(docs, vecs)
    emb = FakeEmbedder()
    assert idx.search("gamma", emb, None, limit=2)[0].position == 1
