import importlib.util
import sys
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "longmemeval", Path(__file__).resolve().parents[1] / "bench" / "longmemeval.py")
lme = importlib.util.module_from_spec(spec)
sys.modules["longmemeval"] = lme
spec.loader.exec_module(lme)


def turns(*pairs):
    out = []
    for u, a in pairs:
        out += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return out


def test_session_chunks_pair_exchanges_and_prefix_date():
    chunks = lme.session_chunks(turns(("hi there", "hello"), ("q2", "a2")), "2023/05/01")
    assert chunks[0] == "[2023/05/01]\n[user] hi there\n[assistant] hello"
    assert len(chunks) == 2


def test_session_chunks_unpaired_and_empty_turns():
    t = [{"role": "user", "content": "one"}, {"role": "user", "content": "two"},
         {"role": "assistant", "content": "  "}, {"role": "assistant", "content": "ans"}]
    assert lme.session_chunks(t, "d") == ["[d]\n[user] one", "[d]\n[user] two\n[assistant] ans"]


def test_session_chunks_byte_cap():
    (c,) = lme.session_chunks(turns(("é" * 20000, "x")), "d")
    assert len(c.encode()) <= lme.MAX_BYTES


def test_metrics_perfect_and_miss():
    assert lme.retrieval_metrics({"a"}, ["a", "b"])["ndcg10"] == 1.0
    m = lme.retrieval_metrics({"z"}, ["a", "b", "c"])
    assert (m["r5"], m["ndcg10"], m["any5"]) == (0.0, 0.0, 0.0)


def test_recall_is_fractional_over_evidence_sessions():
    m = lme.retrieval_metrics({"a", "b"}, ["x", "a", "y", "z", "w", "b"])
    assert m["r5"] == 0.5 and m["r10"] == 1.0 and m["r1"] == 0.0 and m["any5"] == 1.0


def test_ndcg_discounts_by_rank():
    m = lme.retrieval_metrics({"a"}, ["x", "a"])
    assert m["ndcg10"] == pytest.approx(1 / 1.5849625007, rel=1e-6)  # 1 / log2(3)


def test_session_rank_dedupes_keeping_first_occurrence():
    assert lme.session_rank(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_duplicate_session_ids_get_separate_slots():
    q = {"haystack_session_ids": ["s1", "s1"], "haystack_dates": ["d1", "d2"],
         "haystack_sessions": [turns(("hello world", "hi")), turns(("hello world", "hi"))]}
    assert [c.uid for c in lme.question_chunks(q)] == ["s1#0", "s1~2#0"]


def make_question(qid, evidence, others):
    sessions = {evidence: turns((f"my favourite fruit is the {evidence} mango", "noted")), **{
        o: turns((f"talking about topic {o} and weather", "ok")) for o in others}}
    ids = list(sessions)
    return {"question_id": qid, "question_type": "single-session-user", "question": f"what is my favourite fruit {evidence} mango",
            "answer_session_ids": [evidence], "haystack_session_ids": ids,
            "haystack_dates": ["2024/01/01"] * len(ids), "haystack_sessions": [sessions[i] for i in ids]}


def test_run_end_to_end_with_fakes(tmp_path, embedder, reranker):
    qs = [make_question("q1", "e1", ["o1", "o2", "o3"]), make_question("q2_abs", "e2", ["o4"]),
          make_question("q3", "e3", ["o5", "o6"])]
    out = tmp_path / "r" / "retrieval.jsonl"
    rows = lme.run(qs, embedder, reranker, out)
    assert len(rows) == 3
    by = {r["qid"]: r for r in rows}
    assert by["q1"]["ranked_sessions"][0] == "e1" and by["q1"]["metrics"]["r5"] == 1.0
    assert "metrics" not in by["q2_abs"]
    assert by["q1"]["rerank_applied"] is True
    assert len(out.read_text().splitlines()) == 3
    table = lme.summarize(rows)
    assert "Recall@5 | 1.000" in table and "2 non-abstention" in table


def test_run_resume_skips_finished_questions(tmp_path, embedder):
    qs = [make_question("q1", "e1", ["o1"]), make_question("q3", "e3", ["o5"])]
    out = tmp_path / "retrieval.jsonl"
    lme.run(qs[:1], embedder, None, out)
    n = embedder.doc_calls
    rows = lme.run(qs, embedder, None, out, resume=True)
    assert len(rows) == 2 and embedder.doc_calls == n + 1
    assert len(out.read_text().splitlines()) == 2
    assert all(json.loads(l)["rerank_applied"] is False for l in out.read_text().splitlines())
