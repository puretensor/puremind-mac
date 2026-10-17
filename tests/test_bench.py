import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("longmemeval", ROOT / "bench" / "longmemeval.py")
lme = importlib.util.module_from_spec(spec)
sys.modules["longmemeval"] = lme
spec.loader.exec_module(lme)

SETTINGS = {"device": "cpu", "max_seq": 1024, "embedder_revision": "rev-e", "reranker_revision": None}


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


# ---- LongMemEval's own metrics


def reference_metrics(ranking, evidence, corpus, k):
    """LongMemEval's eval_utils.evaluate_retrieval, restated: `ranking` holds indices into `corpus` (session ids)."""
    def dcg(gains):
        gains = np.asarray(gains[:k], dtype=float)
        return float(gains[0] + np.sum(gains[1:] / np.log2(np.arange(2, gains.size + 1)))) if gains.size else 0.0

    top = [corpus[i] for i in ranking[:k]]
    relevance = [1 if s in evidence else 0 for s in corpus]
    ideal = dcg(sorted(relevance, reverse=True))
    ndcg = dcg([relevance[i] for i in ranking[:k]]) / ideal if ideal else 0.0
    return float(any(s in top for s in evidence)), float(all(s in top for s in evidence)), ndcg


def test_official_ndcg_gives_ranks_one_and_two_the_same_weight():
    for rank, expected in {1: 1.0, 2: 1.0, 3: 1 / math.log2(3), 5: 1 / math.log2(5), 10: 1 / math.log2(10)}.items():
        ranked = [f"x{i}" for i in range(rank - 1)] + ["a"]
        assert lme.official_metrics({"a"}, ranked)["ndcg_any@10"] == pytest.approx(expected)
    assert lme.retrieval_metrics({"a"}, ["x", "a"])["ndcg10"] < 0.64  # the standard discount differs at rank 2


def test_official_recall_any_and_all_with_two_evidence_sessions():
    ranked = ["a", "x1", "x2", "x3", "x4", "x5", "x6", "b"]
    m = lme.official_metrics({"a", "b"}, ranked)
    assert (m["recall_any@5"], m["recall_all@5"], m["ndcg_any@5"]) == (1.0, 0.0, 0.5)
    assert (m["recall_any@10"], m["recall_all@10"]) == (1.0, 1.0)
    assert (m["recall_any@1"], m["recall_all@1"]) == (1.0, 0.0)
    assert lme.retrieval_metrics({"a", "b"}, ranked)["r5"] == 0.5  # the harness metric stays fractional


def test_official_metrics_match_the_reference_on_random_rankings():
    rng = random.Random(7)
    for _ in range(300):
        corpus = [f"s{i}" for i in range(rng.randint(3, 60))]
        evidence = set(rng.sample(corpus, rng.randint(1, min(6, len(corpus)))))
        order = rng.sample(range(len(corpus)), rng.randint(1, len(corpus)))  # may stop short, like the 50 candidates
        got = lme.official_metrics(evidence, [corpus[i] for i in order])
        for k in lme.KS:
            any_, all_, ndcg = reference_metrics(order, evidence, corpus, k)
            assert got[f"recall_any@{k}"] == any_ and got[f"recall_all@{k}"] == all_
            assert got[f"ndcg_any@{k}"] == pytest.approx(ndcg)


def test_official_evidence_needs_a_user_side_answer_turn():
    def session(user_flag=False, assistant_flag=False):
        return [{"role": "user", "content": "q", "has_answer": user_flag},
                {"role": "assistant", "content": "a", "has_answer": assistant_flag}]

    q = {"answer_session_ids": ["answer_1", "answer_2", "answer_3"],
         "haystack_session_ids": ["noise", "answer_1", "answer_2", "answer_3"],
         "haystack_sessions": [session(), session(user_flag=True), session(assistant_flag=True), session(True, True)]}
    assert lme.official_evidence(q) == {"answer_1", "answer_3"}
    only_assistant = {**q, "answer_session_ids": ["answer_2"]}
    assert lme.official_evidence(only_assistant) == set()  # such a question is not scored by the official script


def make_question(qid, evidence, others, user_side=True):
    ev = turns((f"my favourite fruit is the {evidence} mango", "noted"))
    ev[0]["has_answer"] = user_side
    sessions = {evidence: ev, **{o: turns((f"talking about topic {o} and weather", "ok")) for o in others}}
    ids = list(sessions)
    return {"question_id": qid, "question_type": "single-session-user",
            "question": f"what is my favourite fruit {evidence} mango",
            "answer_session_ids": [evidence], "haystack_session_ids": ids,
            "haystack_dates": ["2024/01/01"] * len(ids), "haystack_sessions": [sessions[i] for i in ids]}


def cells(table, label):
    for line in table.splitlines():
        if line.startswith(f"| {label} |"):
            return [c.strip() for c in line.strip("|").split("|")[1:]]
    raise AssertionError(f"no row {label!r} in\n{table}")


def test_run_end_to_end_with_fakes(tmp_path, embedder, reranker):
    qs = [make_question("q1", "e1", ["o1", "o2", "o3"]), make_question("q2_abs", "e2", ["o4"]),
          make_question("q3", "e3", ["o5", "o6"])]
    out = tmp_path / "r" / "retrieval.jsonl"
    rows = lme.run(qs, embedder, reranker, out)
    assert len(rows) == 3
    by = {r["qid"]: r for r in rows}
    assert by["q1"]["ranked_sessions"][0] == "e1" and by["q1"]["metrics"]["r5"] == 1.0
    assert "metrics" not in by["q2_abs"] and "official" not in by["q2_abs"]
    assert by["q1"]["rerank_applied"] is True
    assert by["q1"]["official"]["recall_all@5"] == 1.0 and by["q1"]["official"]["ndcg_any@10"] == 1.0
    assert len(out.read_text().splitlines()) == 3
    table = lme.summarize(rows)
    assert "2 non-abstention questions" in table
    assert cells(table, "Fractional recall") == ["1.000", "1.000", "1.000"]
    assert cells(table, "Any-hit") == ["", "1.000", ""]
    assert cells(table, "NDCG, standard discount") == ["", "", "1.000"]
    for name in ("recall_any", "recall_all", "ndcg_any"):
        assert cells(table, name) == ["1.000", "1.000", "1.000"]


def test_summary_scores_official_metrics_only_on_questions_with_a_user_side_target(tmp_path, embedder):
    qs = [make_question("q1", "e1", ["o1", "o2"]), make_question("q3", "e3", ["o5"], user_side=False)]
    rows = lme.run(qs, embedder, None, tmp_path / "retrieval.jsonl")
    assert "official" in rows[0] and "official" not in rows[1] and "metrics" in rows[1]
    table = lme.summarize(rows)
    assert "all 2 non-abstention questions" in table
    assert "1 questions with a user-side target turn" in table


def test_summary_without_any_user_side_target_says_so(tmp_path, embedder):
    rows = lme.run([make_question("q1", "e1", ["o1"], user_side=False)], embedder, None, tmp_path / "retrieval.jsonl")
    table = lme.summarize(rows)
    assert "not reported" in table and "recall_all" not in table
    assert cells(table, "Fractional recall")[1] == "1.000"


def test_summary_names_the_settings(tmp_path, embedder):
    rows = lme.run([make_question("q1", "e1", ["o1"])], embedder, None, tmp_path / "retrieval.jsonl", settings=SETTINGS)
    settings = json.loads((tmp_path / "retrieval.settings.json").read_text())
    first = lme.summarize(rows, settings).splitlines()
    assert any(line.startswith("Settings:") and "max_seq 1024" in line and "device cpu" in line for line in first)


# ---- resume


def test_run_resume_skips_finished_questions(tmp_path, embedder):
    qs = [make_question("q1", "e1", ["o1"]), make_question("q3", "e3", ["o5"])]
    out = tmp_path / "retrieval.jsonl"
    lme.run(qs[:1], embedder, None, out)
    n = embedder.doc_calls
    rows = lme.run(qs, embedder, None, out, resume=True)
    assert len(rows) == 2 and embedder.doc_calls == n + 1
    assert len(out.read_text().splitlines()) == 2
    assert all(json.loads(line)["rerank_applied"] is False for line in out.read_text().splitlines())


def three_questions():
    return [make_question(f"q{i}", f"e{i}", [f"o{i}a", f"o{i}b"]) for i in (1, 2, 3)]


def test_resume_drops_a_line_cut_off_by_a_crash(tmp_path, embedder):
    out = tmp_path / "retrieval.jsonl"
    lme.run(three_questions()[:2], embedder, None, out, settings=SETTINGS)
    text = out.read_text()
    out.write_text(text[: text.rindex('"ranked_sessions"')])  # killed in the middle of the second row
    rows = lme.run(three_questions(), embedder, None, out, resume=True, settings=SETTINGS)
    assert [r["qid"] for r in rows] == ["q1", "q2", "q3"]
    assert [json.loads(line)["qid"] for line in out.read_text().splitlines()] == ["q1", "q2", "q3"]


def test_resume_recomputes_a_final_row_that_lost_its_newline(tmp_path, embedder):
    out = tmp_path / "retrieval.jsonl"
    lme.run(three_questions()[:2], embedder, None, out, settings=SETTINGS)
    out.write_text(out.read_text().rstrip("\n"))  # whole row, no newline: appending would glue two rows together
    lme.run(three_questions(), embedder, None, out, resume=True, settings=SETTINGS)
    assert [json.loads(line)["qid"] for line in out.read_text().splitlines()] == ["q1", "q2", "q3"]


def test_resume_refuses_a_damaged_file_and_leaves_it_alone(tmp_path, embedder):
    out = tmp_path / "retrieval.jsonl"
    lme.run(three_questions(), embedder, None, out, settings=SETTINGS)
    lines = out.read_text().splitlines()
    out.write_text("\n".join([lines[0], lines[1][:40], lines[2]]) + "\n")
    before = out.read_bytes()
    with pytest.raises(lme.ResumeError, match="line 2"):
        lme.run(three_questions(), embedder, None, out, resume=True, settings=SETTINGS)
    assert out.read_bytes() == before


def test_run_records_its_settings_next_to_the_results(tmp_path, embedder, reranker):
    out = tmp_path / "retrieval.jsonl"
    lme.run(three_questions()[:1], embedder, reranker, out, candidates=20, rrf_k=30, rerank_chars=400,
            settings=SETTINGS)
    assert json.loads((tmp_path / "retrieval.settings.json").read_text()) == {
        "results_format": lme.RESULTS_FORMAT, "embedder": "fake-embedder", "reranker": "fake-reranker", "rerank": True,
        "candidates": 20, "rrf_k": 30, "rerank_chars": 400, **SETTINGS,
    }
    lme.run(three_questions()[:1], embedder, None, out)  # a fresh run replaces the record
    again = json.loads((tmp_path / "retrieval.settings.json").read_text())
    assert again["rerank"] is False and again["reranker"] is None and "device" not in again


@pytest.mark.parametrize("changed, message", [
    ({"candidates": 20}, "candidates"),
    ({"rrf_k": 30}, "rrf_k"),
    ({"rerank_chars": 400}, "rerank_chars"),
    ({"settings": {**SETTINGS, "max_seq": 512}}, "max_seq"),
    ({"settings": {**SETTINGS, "device": "mps"}}, "device"),
    ({"settings": {**SETTINGS, "embedder_revision": "rev-new"}}, "embedder_revision"),
])
def test_resume_refuses_different_settings(tmp_path, embedder, changed, message):
    out = tmp_path / "retrieval.jsonl"
    lme.run(three_questions()[:2], embedder, None, out, settings=SETTINGS)
    before = out.read_bytes()
    with pytest.raises(lme.ResumeError, match=message):
        lme.run(three_questions(), embedder, None, out, resume=True, **{"settings": SETTINGS, **changed})
    assert out.read_bytes() == before


def test_resume_refuses_a_different_model_or_rerank_switch(tmp_path, embedder, reranker):
    out = tmp_path / "retrieval.jsonl"
    lme.run(three_questions()[:2], embedder, None, out, settings=SETTINGS)
    with pytest.raises(lme.ResumeError, match="rerank"):
        lme.run(three_questions(), embedder, reranker, out, resume=True, settings=SETTINGS)
    embedder.name = "another-embedder"
    with pytest.raises(lme.ResumeError, match="embedder"):
        lme.run(three_questions(), embedder, None, out, resume=True, settings=SETTINGS)


def test_resume_refuses_results_without_a_settings_record(tmp_path, embedder):
    out = tmp_path / "retrieval.jsonl"
    lme.run(three_questions()[:2], embedder, None, out, settings=SETTINGS)
    (tmp_path / "retrieval.settings.json").unlink()
    with pytest.raises(lme.ResumeError, match="no retrieval.settings.json"):
        lme.run(three_questions(), embedder, None, out, resume=True, settings=SETTINGS)


def test_resume_with_nothing_to_resume_just_starts(tmp_path, embedder):
    out = tmp_path / "retrieval.jsonl"
    rows = lme.run(three_questions(), embedder, None, out, resume=True, settings=SETTINGS)  # no file yet
    assert len(rows) == 3 and (tmp_path / "retrieval.settings.json").is_file()
    out.write_text('{"qid": "q1", "ranked": [')  # only a torn first row
    (tmp_path / "retrieval.settings.json").unlink()
    assert len(lme.run(three_questions(), embedder, None, out, resume=True, settings=SETTINGS)) == 3


# ---- command line


@pytest.fixture
def cli_stubs(monkeypatch, embedder, reranker):
    """Config and the two model classes replaced by stubs that record how the harness builds them."""
    built = {}
    cfg = SimpleNamespace(embedder="org/embed", reranker="org/rank", embedder_revision="rev-e",
                          reranker_revision="rev-r", device="cpu", candidates=50, rrf_k=60, rerank_chars=800)

    def make_embedder(model, device, max_seq, *, revision):
        built["embedder"] = (model, device, max_seq, revision)
        embedder.name = model
        return embedder

    def make_reranker(model, device, *, revision):
        built["reranker"] = (model, device, revision)
        reranker.name = model
        return reranker

    monkeypatch.setattr(lme, "load_config", lambda: cfg)
    monkeypatch.setattr(lme, "SentenceTransformerEmbedder", make_embedder)
    monkeypatch.setattr(lme, "CrossEncoderReranker", make_reranker)
    return built


@pytest.fixture
def data_file(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(three_questions()))
    return path


def run_main(data_file, out, *args):
    return lme.main(["--data", str(data_file), "--out", str(out), *args])


def test_main_measures_with_the_published_sequence_length_by_default(tmp_path, data_file, cli_stubs, capsys):
    out = tmp_path / "out"
    assert run_main(data_file, out) == 0
    assert cli_stubs["embedder"] == ("org/embed", "cpu", 1024, "rev-e")
    assert cli_stubs["reranker"] == ("org/rank", "cpu", "rev-r")
    settings = json.loads((out / "retrieval.settings.json").read_text())
    assert settings["max_seq"] == 1024 and settings["device"] == "cpu" and settings["rerank"] is True
    assert (settings["embedder_revision"], settings["reranker_revision"]) == ("rev-e", "rev-r")
    summary = (out / "summary.md").read_text()
    assert summary == capsys.readouterr().out.rstrip("\n") + "\n"
    assert "3 non-abstention questions" in summary and "max_seq 1024" in summary


def test_main_max_seq_and_no_rerank(tmp_path, data_file, cli_stubs):
    out = tmp_path / "out"
    assert run_main(data_file, out, "--max-seq", "512", "--no-rerank", "--device", "cpu", "--limit", "2") == 0
    assert cli_stubs["embedder"][2] == 512 and "reranker" not in cli_stubs
    settings = json.loads((out / "retrieval.settings.json").read_text())
    assert (settings["max_seq"], settings["rerank"]) == (512, False)
    assert (settings["reranker"], settings["reranker_revision"]) == (None, None)
    assert len((out / "retrieval.jsonl").read_text().splitlines()) == 2


def test_main_records_the_device_that_auto_resolves_to(tmp_path, data_file, cli_stubs, monkeypatch):
    monkeypatch.setattr(lme, "load_config", lambda: SimpleNamespace(
        embedder="org/embed", reranker="org/rank", embedder_revision=None, reranker_revision=None,
        device="auto", candidates=50, rrf_k=60, rerank_chars=800))
    monkeypatch.setattr(lme, "pick_device", lambda requested: "mps" if requested == "auto" else requested)
    out = tmp_path / "out"
    assert run_main(data_file, out) == 0
    assert cli_stubs["embedder"][1] == "mps" and cli_stubs["reranker"][1] == "mps"
    assert json.loads((out / "retrieval.settings.json").read_text())["device"] == "mps"


def test_main_resume_continues_a_run_and_refuses_changed_settings(tmp_path, data_file, cli_stubs, capsys):
    out = tmp_path / "out"
    assert run_main(data_file, out, "--limit", "2") == 0
    assert run_main(data_file, out, "--resume") == 0
    assert len((out / "retrieval.jsonl").read_text().splitlines()) == 3
    before = (out / "retrieval.jsonl").read_bytes()
    capsys.readouterr()
    assert run_main(data_file, out, "--resume", "--max-seq", "2048") == 2
    assert "max_seq" in capsys.readouterr().err
    assert (out / "retrieval.jsonl").read_bytes() == before


# ---- the script's own documentation


def test_download_instructions_use_the_current_hf_command():
    doc = lme.__doc__
    download = "xiaowu0162/longmemeval-cleaned longmemeval_s_cleaned.json --repo-type dataset --local-dir ./data"
    assert f"hf download {download}" in doc
    assert "pip install -e ." in doc and "arXiv 2410.10813" in doc and "MIT licence" in doc
    assert doc.index("hf download") < doc.index("huggingface-cli download")  # the old command is the footnote


def test_script_text_has_no_internal_labels_or_em_dashes():
    text = (ROOT / "bench" / "longmemeval.py").read_text(encoding="utf-8")
    labels = ("MAC" + "-16b", "R1 " + "Mac")  # in pieces, so a search of the repository finds no stray copy
    assert not [label for label in labels if label in text]
    assert chr(0x2014) not in text


def test_importing_the_script_leaves_the_source_tree_off_sys_path_when_testing_the_wheel(testing_installed):
    if not testing_installed:
        pytest.skip("source-tree run")
    assert str(ROOT / "src") not in sys.path
