import random
import re
import threading
import unicodedata

import bm25s
import numpy as np
import pytest
import Stemmer
from fakes import FakeEmbedder, FakeReranker

from puremind_mac import retrieval
from puremind_mac.retrieval import (
    _CJK,
    _CJK_RANGES,
    HybridIndex,
    LexiconBuilder,
    _segment_cjk,
    _words,
    rrf_fuse,
)

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
    # "tomato" only matches "tomatoes" through the stemmer, and "the" would match four other documents
    assert idx.lexical("the tomato", 5) == [0]


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


# ---- stemming and edge corpora


def test_lexical_stems_in_both_directions():
    idx = HybridIndex(["raised beds", "tomato plants"], None)
    assert idx.lexical("raising", 5) == [0]  # raised, raising -> rais
    assert idx.lexical("tomatoes", 5) == [1]


@pytest.mark.parametrize(
    "texts",
    [["😀😀😀", "🎉🎉"], ["a b c", "x y z"], ["the and of", "it is a"], ["", ""], ["   ", "\n"]],
    ids=["emoji", "one-letter", "stopwords", "empty", "blank"],
)
def test_corpus_without_a_single_token_has_no_lexical_candidates(texts):
    vecs = FakeEmbedder().embed_documents(["x", "y"])
    idx = HybridIndex(texts, vecs)
    assert idx.lexical("party", 5) == []
    assert idx.lexical("a", 5) == []
    emb = FakeEmbedder()
    hits = idx.search("party", emb, None, limit=5)  # the dense side still answers
    assert sorted(h.position for h in hits) == [0, 1]
    assert idx.lexical("party", 5) == []  # and asking again does not rebuild or fail


def test_a_failed_bm25_build_is_not_cached():
    idx = HybridIndex(["alpha beta", "gamma delta"], None)
    real = idx._tokenize
    calls = []

    def flaky(docs, query=False):
        calls.append(len(docs))
        if len(calls) == 1:
            raise MemoryError("simulated")
        return real(docs, query)

    idx._tokenize = flaky
    with pytest.raises(MemoryError):
        idx.lexical("alpha", 5)
    assert idx.lexical("alpha", 5) == [0]


def test_each_tokenize_call_gets_its_own_stemmer(monkeypatch):
    made = []
    real = Stemmer.Stemmer

    def factory(*a, **k):
        made.append(real(*a, **k))
        return made[-1]

    monkeypatch.setattr(Stemmer, "Stemmer", factory)
    idx = HybridIndex(["tomatoes"], None)
    threads = [threading.Thread(target=idx._tokenize, args=(["gardens"],)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    idx._tokenize(["beds"])
    assert len(made) == 5 and len({id(m) for m in made}) == 5  # PyStemmer instances must not be shared


# ---- Chinese, Japanese and Korean

CJK_DOCS = [
    "数据库迁移到新的服务器,需要在周末完成。",
    "今天天气很好,我们去公园散步。",
    "来週までに移行作業を終わらせる必要があります。",
    "서울에서 부산까지 기차로 이동했다.",
    "ジョン・スミスはサーバーを再起動した。",
    "Deploy the new Kubernetes cluster to production next week.",
    "GPU加速 and 猫 live here",
]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("迁移", [0]),  # Chinese word inside an unspaced sentence
        ("数据库迁移", [0]),
        ("天气", [1]),
        ("移行", [2]),  # Japanese kanji word
        ("作業を終わらせる", [2]),  # kanji and kana together
        ("サーバー", [4]),  # katakana with the long-vowel mark
        ("ジョン", [4]),  # the middle dot is a separator, not part of a name
        ("서울", [3]),  # Korean word that carries a particle in the text
        ("부산까지", [3]),
        ("kubernetes", [5]),  # Latin text next to CJK documents is unaffected
        ("GPU", [6]),  # Latin glued to CJK is split off
        ("加速", [6]),
        ("猫", [6]),  # a lone character matches a lone character
        ("的", [0]),  # and one inside a longer run, since documents also index every character
        ("天气 迁移", [0, 1]),  # several words: every document that has one of them
    ],
)
def test_lexical_matches_cjk_words(query, expected):
    idx = HybridIndex(CJK_DOCS, None)
    got = idx.lexical(query, 5)
    assert sorted(got) == expected


def test_cjk_query_finds_the_note_through_search():
    emb = FakeEmbedder()  # its vectors are all zero for CJK text, so only BM25 can find the match
    idx = HybridIndex(CJK_DOCS, emb.embed_documents(CJK_DOCS))
    assert idx.search("数据库迁移", emb, None, limit=3)[0].position == 0
    assert idx.search("移行作業", emb, None, limit=3)[0].position == 2


def test_cjk_runs_become_overlapping_bigrams():
    assert _segment_cjk("数据库") == " 数据 据库 "
    assert _segment_cjk("迁移") == " 迁移 "
    assert _segment_cjk("猫") == " 猫 "
    assert _segment_cjk("数据库", unigrams=True) == " 数 据 库 数据 据库 "  # how documents are indexed
    assert _segment_cjk("猫", unigrams=True) == " 猫 "
    assert _segment_cjk("a数据b") == "a 数据 b"
    assert _segment_cjk("ジョン・スミス") == " ジョ ョン ・ スミ ミス "
    assert _segment_cjk("서울에서") == " 서울 울에 에서 "
    beyond_bmp = "\U00020000\U00020001\U00020002"
    assert _segment_cjk(beyond_bmp) == " \U00020000\U00020001 \U00020001\U00020002 "
    assert _segment_cjk("你好,世界") == " 你好 , 世界 "  # punctuation ends a run


def test_cjk_class_covers_only_word_characters():
    word = re.compile(r"\w")
    for lo, hi in _CJK_RANGES:
        for cp in range(lo, hi + 1):
            if unicodedata.category(chr(cp)) != "Cn":  # unassigned code points do not matter
                assert word.match(chr(cp)), hex(cp)  # so a run never contains a character the tokeniser splits on


def test_cjk_class_leaves_out_punctuation_and_marks():
    cjk = re.compile(f"[{_CJK}]")
    for cp in (0x30FB, 0x30A0, 0x3001, 0x3002, 0x300C, 0xFF0C, 0x3099, 0x309B, 0x0041, 0x00E9):
        assert not cjk.match(chr(cp)), hex(cp)  # katakana middle dot, CJK punctuation, kana marks, Latin


def tokens_of(tok):
    inverse = {i: t for t, i in tok.vocab.items()}
    return [[inverse[i] for i in doc] for doc in tok.ids]


def legacy_tokenize(docs):
    """What HybridIndex produced before CJK handling: the stock bm25s call."""
    return bm25s.tokenize(docs, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)


def legacy_tokens(docs):
    return tokens_of(legacy_tokenize(docs))


ENGLISH = [
    "The quick brown fox's jump over the lazy dog's back; see https://example.org/a/b?id=42, v2.0-beta!",
    "It's 3:45pm - don't panic (really). Tomatoes need compost and mulch in the raised garden beds.",
    "Decided to migrate the build pipeline to hermetic containers next quarter.\n\n## Notes\n\n```py\nx = 1\n```",
    (
        "Résumé café naïve façade Zürich, Ελληνικά, Русский текст, हिन्दी, العربية, "
        "emoji 😀 and 12,345.67 EUR"
    ),
    "",
]


def test_english_and_other_non_cjk_text_tokenises_exactly_as_before():
    idx = HybridIndex([], None)
    assert tokens_of(idx._tokenize(ENGLISH)) == legacy_tokens(ENGLISH)
    assert all(_segment_cjk(t) == t for t in ENGLISH)


def test_random_non_cjk_text_tokenises_exactly_as_before():
    rng = random.Random(20261017)
    blocks = [(0x20, 0x7E), (0xA0, 0x24F), (0x370, 0x3FF), (0x400, 0x4FF), (0x590, 0x6FF), (0x900, 0x97F),
              (0xE00, 0xE7F), (0x2000, 0x206F), (0x3000, 0x3004), (0xFF01, 0xFF65), (0x1F300, 0x1F64F)]
    cjk = re.compile(f"[{_CJK}]")
    docs = []
    for _ in range(400):
        chars = (chr(rng.randint(*rng.choice(blocks))) for _ in range(rng.randint(0, 300)))
        docs.append("".join(c if rng.random() > 0.15 else " " for c in chars if not cjk.match(c)))
    assert tokens_of(HybridIndex([], None)._tokenize(docs)) == legacy_tokens(docs)


def test_english_lexical_ranking_is_unchanged():
    legacy = bm25s.BM25()
    legacy.index(legacy_tokenize(DOCS), show_progress=False)
    idx, _ = make_index()
    for q in ("espresso grinder", "tomato garden compost", "the build pipeline", "boiler"):
        res, scores = legacy.retrieve(legacy_tokenize([q]), k=len(DOCS), show_progress=False)
        ranked = sorted(zip(scores[0].tolist(), res[0].tolist()), key=lambda t: (-t[0], t[1]))  # ties by position
        assert idx.lexical(q, len(DOCS)) == [j for s, j in ranked if s > 0]


# ---- the BM25 index scores as bm25s does, and is rebuilt from what changed

WORDS = [
    "the", "a", "of", "and", "espresso", "grinder", "boiler", "tomatoes", "garden", "compost", "pipeline",
    "containers", "backflush", "weekly", "raised", "beds", "descale", "months", "beans", "burr", "setting",
    "migrate", "hermetic", "build", "kubernetes", "cluster", "deploy",
    *(f"term{i}" for i in range(60)), "数据库", "服务器", "猫", "서울", "サーバー",
]


def random_texts(rng, n):
    texts = []
    for _ in range(n):
        size = rng.choice([0, 1, 3, 12, 40, 120])
        texts.append(" ".join(rng.choice(WORDS) for _ in range(size)))
    return texts


def query_words(query):
    return _words(retrieval.tokenize([query], query=True))[0]


QUERIES = ["espresso", "espresso espresso grinder", "the of", "term3 term3 term7 boiler", "unknownword", "猫 服务器",
           "tomatoes garden compost raised beds pipeline", ""]


@pytest.mark.parametrize("seed", range(6))
def test_lexicon_scores_match_bm25s(seed):
    rng = random.Random(seed)
    texts = random_texts(rng, 200) + ["espresso grinder"]  # at least one token
    reference = bm25s.BM25()
    reference.index(retrieval.tokenize(texts), show_progress=False)
    lexicon = LexiconBuilder().build(texts)
    for query in QUERIES:
        words = query_words(query)
        ours = lexicon.scores(words)
        theirs = reference.get_scores(words) if words else np.zeros(len(texts))
        assert np.allclose(ours, theirs, rtol=1e-5, atol=1e-6), query


def test_a_lexicon_built_from_the_previous_one_scores_like_a_fresh_build():
    rng = random.Random(7)
    builder = LexiconBuilder()
    before = random_texts(rng, 150)
    builder.build(before)
    after = before[20:] + random_texts(rng, 30)  # some chunks gone, some new
    after[5] += " espresso"  # one edited
    built, fresh = builder.build(after), LexiconBuilder().build(after)
    for query in QUERIES:
        words = query_words(query)
        assert np.allclose(built.scores(words), fresh.scores(words), rtol=1e-12, atol=0)
        assert built.top(words, 20) == fresh.top(words, 20)


def test_the_builder_tokenizes_only_text_it_has_not_seen():
    seen = []

    def counting(docs, query=False):
        seen.append(list(docs))
        return retrieval.tokenize(docs, query)

    builder = LexiconBuilder()
    builder.build(["alpha one", "beta two", "alpha one"], counting)
    assert seen == [["alpha one", "beta two"]]  # a repeated text is tokenized once
    builder.build(["alpha one", "beta two", "gamma three"], counting)
    assert seen[-1] == ["gamma three"]
    builder.build(["gamma three", "alpha one"], counting)
    assert len(seen) == 2  # nothing new


def test_the_builder_keeps_nothing_of_text_that_left_the_index():
    builder = LexiconBuilder()
    builder.build(["the code is zqxcanary", "garden plan"])
    lexicon = builder.build(["garden plan"])
    assert len(builder._docs) == 1
    assert lexicon.top(query_words("zqxcanary"), 5) == []


def test_unused_words_are_dropped_and_an_older_lexicon_keeps_working():
    builder = LexiconBuilder()
    first = None
    for round_ in range(4):  # every round replaces the whole vault with 3,000 words not seen before
        texts = [" ".join(f"w{round_}x{i}y{j}" for j in range(10)) for i in range(300)]
        lexicon = builder.build(texts)
        assert lexicon.top(query_words(texts[3]), 1) == [3]
        assert len(builder._vocab) <= 6000  # words of earlier rounds do not pile up
        if first is None:
            first, first_texts = lexicon, texts
    assert first.top(query_words(first_texts[7]), 1) == [7]  # a search still holding the first snapshot


def test_a_failed_build_leaves_the_builder_as_it_was():
    builder = LexiconBuilder()
    builder.build(["alpha beta"])

    def failing(docs, query=False):
        raise MemoryError("simulated")

    with pytest.raises(MemoryError):
        builder.build(["alpha beta", "gamma delta"], failing)
    lexicon = builder.build(["alpha beta", "gamma delta"])
    assert lexicon.top(query_words("gamma"), 5) == [1] and lexicon.top(query_words("alpha"), 5) == [0]


def test_ties_are_ordered_by_position():
    lexicon = LexiconBuilder().build(["espresso machine", "espresso boiler", "espresso cups", "garden"])
    assert lexicon.top(query_words("espresso"), 3) == [0, 1, 2]
