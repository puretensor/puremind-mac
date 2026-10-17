"""Measure pureMind's hybrid retrieval on LongMemEval_s.

Retrieval is the same code `puremind search` runs: BM25 top-50 and dense top-50 (Qwen3-Embedding-0.6B),
fused with reciprocal rank fusion (k=60), then re-scored by the bge-reranker-v2-m3 cross-encoder over the 50
fused candidates.

Install the package first, from the repository root:

    pip install -e .

The dataset is not vendored. Download the cleaned LongMemEval_s file (MIT licence, as stated on its Hugging
Face page) with the `hf` command, which needs huggingface_hub 0.34 or newer:

    hf download xiaowu0162/longmemeval-cleaned longmemeval_s_cleaned.json --repo-type dataset --local-dir ./data

Older huggingface_hub releases have `huggingface-cli download` instead, with the same arguments (version 1.0
and newer no longer run it). LongMemEval is described in Wu et al., "LongMemEval: Benchmarking Chat
Assistants on Long-Term Interactive Memory", arXiv 2410.10813. Then run, from the repository root:

    python bench/longmemeval.py --data data/longmemeval_s_cleaned.json
    python bench/longmemeval.py --data data/longmemeval_s_cleaned.json --limit 20   # smoke test
    python bench/longmemeval.py --data data/longmemeval_s_cleaned.json --no-rerank
    python bench/longmemeval.py --data data/longmemeval_s_cleaned.json --resume     # after an interruption

Each question is one simulated user. Its haystack sessions become that user's memory: one chunk per
user/assistant exchange, prefixed with the session date, capped at 12,000 bytes. Chunk rankings collapse to
session rankings (first occurrence wins) and are scored against `answer_session_ids`. The ranking holds only
the sessions that the 50 reranked candidates come from, so Recall@10 is understated when they span fewer than
ten sessions.

The model settings come from `puremind`'s own configuration, except that the embedder sequence length is
`--max-seq` (1024 by default; the published benchmark read whole chunks, so pass 8192 to match it). Every setting that
affects the rankings is recorded in `retrieval.settings.json` next to the results, and `--resume` refuses to
continue a run whose settings differ.

Metrics (session level, summary.md labels each one):
  Harness metrics, over the non-abstention questions (ids not ending in `_abs`):
    Fractional recall@k  evidence sessions in the top k / evidence sessions, averaged (k = 1, 5, 10)
    Any-hit@5            1 when an evidence session is in the top 5
    NDCG@10              binary session relevance, standard discount 1 / log2(rank + 1)
  LongMemEval's own metrics (src/retrieval/eval_utils.py), on the same rankings:
    recall_any@k, recall_all@k, ndcg_any@k   (k = 1, 5, 10)
  Its script scores only questions with a user-side target turn and treats an answer session as evidence only
  when one of its user turns carries `has_answer`, so these are reported for that subset of questions.
Output: bench/results/retrieval.jsonl (one row per question), retrieval.settings.json and summary.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import puremind_mac  # noqa: F401  (installed: pip install -e . or a wheel)
except ImportError:  # run straight from a clone without installing it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puremind_mac.config import load_config
from puremind_mac.models import (
    CrossEncoderReranker,
    Embedder,
    Reranker,
    SentenceTransformerEmbedder,
    pick_device,
)
from puremind_mac.retrieval import HybridIndex

MIN_CHARS = 3
MAX_BYTES = 12_000
MAX_SEQ = 1024  # embedder sequence length; 0.5% of chunks are longer, and the published run read them whole (8192)
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS_FORMAT = 2  # bumped whenever the layout of a result row or the settings record changes
KS = (1, 5, 10)


class ResumeError(RuntimeError):
    """The existing results cannot be continued."""


@dataclass(frozen=True)
class BenchChunk:
    uid: str
    session_id: str
    text: str


def session_chunks(turns: Sequence[dict], date: str) -> list[str]:
    """One chunk per user/assistant exchange, date-prefixed."""
    out: list[str] = []

    def emit(user: str | None, assistant: str | None) -> None:
        parts = [f"[{date}]"]
        if user:
            parts.append(f"[user] {user}")
        if assistant:
            parts.append(f"[assistant] {assistant}")
        if len(parts) == 1:
            return
        text = "\n".join(parts)
        if len(text.strip()) < MIN_CHARS:
            return
        raw = text.encode("utf-8")
        if len(raw) > MAX_BYTES:
            text = raw[:MAX_BYTES].decode("utf-8", errors="ignore")
        out.append(text)

    pending: str | None = None
    for turn in turns:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if turn.get("role") == "user":
            if pending is not None:
                emit(pending, None)
            pending = content
        elif turn.get("role") == "assistant":
            emit(pending, content)
            pending = None
    if pending is not None:
        emit(pending, None)
    return out


def question_chunks(q: dict) -> list[BenchChunk]:
    seen: dict[str, int] = defaultdict(int)
    chunks: list[BenchChunk] = []
    for sid, date, turns in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
        seen[sid] += 1
        slot = sid if seen[sid] == 1 else f"{sid}~{seen[sid]}"  # a session listed twice gets its own slot
        for i, text in enumerate(session_chunks(turns, date)):
            chunks.append(BenchChunk(f"{slot}#{i}", sid, text))
    return chunks


def session_rank(session_ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in session_ids:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def retrieval_metrics(evidence: set[str], ranked_sessions: Sequence[str]) -> dict[str, float]:
    """The harness metrics: fractional recall@k, any-hit@5 and NDCG@10 with the standard discount."""
    m = {f"r{k}": len(evidence & set(ranked_sessions[:k])) / len(evidence) for k in KS}
    m["any5"] = float(bool(evidence & set(ranked_sessions[:5])))
    dcg = sum(1 / math.log2(i + 1) for i, s in enumerate(ranked_sessions[:10], 1) if s in evidence)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(evidence), 10) + 1))
    m["ndcg10"] = dcg / idcg
    return m


def _official_dcg(gains: Sequence[float], k: int) -> float:
    """DCG@k as LongMemEval's eval_utils.dcg computes it: ranks 1 and 2 both have discount 1."""
    gains = list(gains)[:k]
    if not gains:
        return 0.0
    return gains[0] + sum(g / math.log2(rank) for rank, g in enumerate(gains[1:], 2))


def official_metrics(evidence: set[str], ranked_sessions: Sequence[str]) -> dict[str, float]:
    """recall_any@k, recall_all@k and ndcg_any@k as LongMemEval's eval_utils.evaluate_retrieval defines them."""
    out: dict[str, float] = {}
    for k in KS:
        top = ranked_sessions[:k]
        found = evidence & set(top)
        ideal = _official_dcg([1.0] * len(evidence), k)
        actual = _official_dcg([float(s in evidence) for s in top], k)
        out[f"recall_any@{k}"] = float(bool(found))
        out[f"recall_all@{k}"] = float(found == evidence)
        out[f"ndcg_any@{k}"] = actual / ideal if ideal else 0.0
    return out


def official_evidence(q: dict) -> set[str]:
    """The evidence sessions as LongMemEval's own retrieval script labels them.

    Its session index holds the user turns only, so an answer session counts as evidence only when one of its
    user turns carries `has_answer`. A question left without evidence this way is not scored by that script.
    """
    answers = set(q["answer_session_ids"])
    return {
        sid
        for sid, turns in zip(q["haystack_session_ids"], q["haystack_sessions"])
        if sid in answers and any(t.get("has_answer") for t in turns if t.get("role") == "user")
    }


def settings_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + ".settings.json")


def run_settings(
    embedder: Embedder,
    reranker: Reranker | None,
    candidates: int,
    rrf_k: int,
    rerank_chars: int,
    extra: Mapping[str, object] | None = None,
) -> dict:
    """Everything that decides the rankings, as a JSON-ready dict. `extra` carries what the models do not expose."""
    record = {
        **(extra or {}),
        "results_format": RESULTS_FORMAT,
        "embedder": embedder.name,
        "reranker": reranker.name if reranker is not None else None,
        "rerank": reranker is not None,
        "candidates": candidates,
        "rrf_k": rrf_k,
        "rerank_chars": rerank_chars,
    }
    return json.loads(json.dumps(record))


def load_rows(path: Path) -> tuple[dict[str, dict], int]:
    """The rows of an earlier run by question id, and the byte length of the intact part of the file.

    A final line without its newline was cut off mid-write (the process was killed) and is left out. A broken
    line anywhere else means the file was damaged, which resuming must not paper over.
    """
    raw = path.read_bytes()
    *lines, tail = raw.split(b"\n")
    rows: dict[str, dict] = {}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rows[row["qid"]] = row
        except (ValueError, KeyError, TypeError) as exc:
            raise ResumeError(
                f"{path} line {number} is not a result row ({exc!r}); delete the file to start again"
            ) from exc
    return rows, len(raw) - len(tail)


def check_resume_settings(sidecar: Path, current: dict) -> None:
    """Refuse to continue results that were produced under other settings."""
    if not sidecar.is_file():
        raise ResumeError(
            f"cannot resume: no {sidecar.name} records the settings of the existing results; run without --resume"
        )
    try:
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ResumeError(f"cannot resume: {sidecar.name} is unreadable ({exc}); run without --resume") from exc
    if recorded != current:
        keys = sorted(recorded.keys() | current.keys())
        diff = "; ".join(
            f"{k} was {recorded.get(k)!r}, now {current.get(k)!r}" for k in keys if recorded.get(k) != current.get(k)
        )
        raise ResumeError(f"cannot resume with different settings ({diff}); run without --resume to start again")


def write_settings(sidecar: Path, record: dict) -> None:
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, sidecar)


def run(
    questions: Sequence[dict],
    embedder: Embedder,
    reranker: Reranker | None,
    out_path: Path,
    resume: bool = False,
    candidates: int = 50,
    rrf_k: int = 60,
    rerank_chars: int = 800,
    settings: Mapping[str, object] | None = None,
) -> list[dict]:
    """Retrieve for every question and append one scored row each to `out_path`.

    `settings` adds what the models do not expose (device, sequence length, revisions) to the record written
    to the settings file next to `out_path`. With `resume`, questions already in `out_path` are skipped, but
    only if that record equals this run's.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = settings_path(out_path)
    record = run_settings(embedder, reranker, candidates, rrf_k, rerank_chars, settings)
    done: dict[str, dict] = {}
    if resume and out_path.exists():
        done, intact = load_rows(out_path)
        if done:
            check_resume_settings(sidecar, record)
            if intact < out_path.stat().st_size:
                with out_path.open("r+b") as fh:
                    fh.truncate(intact)  # drop the line a crash cut off
    if not done:
        write_settings(sidecar, record)
    cache: dict[str, np.ndarray] = {}  # chunk text -> float16 vector; sessions repeat across haystacks
    t_all = time.time()
    with out_path.open("a" if done else "w") as fh:
        for n, q in enumerate(questions, 1):
            qid = q["question_id"]
            if qid in done:
                continue
            chunks = question_chunks(q)
            texts = [c.text for c in chunks]
            missing = sorted({t for t in texts if t not in cache})
            if missing:
                for t, v in zip(missing, embedder.embed_documents(missing)):
                    cache[t] = v.astype(np.float16)
            vecs = np.stack([cache[t] for t in texts]).astype(np.float32) if texts else None
            t0 = time.perf_counter()
            index = HybridIndex(texts, vecs)
            scored = index.search(q["question"], embedder, reranker, limit=candidates,
                                  candidates=candidates, rrf_k=rrf_k, rerank_chars=rerank_chars)
            ms = (time.perf_counter() - t0) * 1000
            ranked = [chunks[s.position] for s in scored]
            row = {"qid": qid, "question_type": q["question_type"],
                   "ranked": [c.uid for c in ranked], "ranked_sessions": session_rank([c.session_id for c in ranked]),
                   "rerank_applied": reranker is not None and bool(scored), "latency_ms": round(ms, 1)}
            if not qid.endswith("_abs"):
                row["metrics"] = retrieval_metrics(set(q["answer_session_ids"]), row["ranked_sessions"])
                target = official_evidence(q)
                if target:
                    row["official"] = official_metrics(target, row["ranked_sessions"])
            done[qid] = row
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(questions)} ({time.time() - t_all:.0f}s)", flush=True)
    return list(done.values())


def describe_settings(settings: Mapping[str, object]) -> str:
    """One line naming the settings of a run, as recorded in the settings file."""

    def model(name_key: str, revision_key: str) -> str:
        name, revision = settings.get(name_key), settings.get(revision_key)
        return f"{name} (revision {revision})" if revision else f"{name}"

    parts = [f"embedder {model('embedder', 'embedder_revision')}"]
    parts.append(f"reranker {model('reranker', 'reranker_revision')}" if settings.get("rerank") else "reranker off")
    parts += [f"{k} {settings[k]}" for k in ("device", "max_seq", "candidates", "rrf_k", "rerank_chars")
              if k in settings]
    return "Settings: " + ", ".join(parts)


def summarize(rows: Sequence[dict], settings: Mapping[str, object] | None = None) -> str:
    """The markdown report. Every metric is named with its definition, so no figure is ambiguous."""
    scored = [r for r in rows if "metrics" in r]
    if not scored:
        return "No scored questions.\n"
    official = [r for r in scored if "official" in r]

    def mean(group: str, key: str, rs: Sequence[dict]) -> float:
        return statistics.fmean(r[group][key] for r in rs)

    def line(label: str, group: str, keys: Sequence[str | None], rs: Sequence[dict]) -> str:
        return f"| {label} | " + " | ".join("" if k is None else f"{mean(group, k, rs):.3f}" for k in keys) + " |"

    lat = sorted(r["latency_ms"] for r in rows)
    lines = [
        (
            f"LongMemEval_s retrieval, {len(scored)} non-abstention questions "
            f"({sum(r['rerank_applied'] for r in rows)}/{len(rows)} reranked)"
        ),
        "",
    ]
    if settings:
        lines += [describe_settings(settings), ""]
    lines += [
        f"Harness metrics, all {len(scored)} non-abstention questions, session level:",
        "",
        "| Metric | @1 | @5 | @10 |",
        "|---|---|---|---|",
        line("Fractional recall", "metrics", ["r1", "r5", "r10"], scored),
        line("Any-hit", "metrics", [None, "any5", None], scored),
        line("NDCG, standard discount", "metrics", [None, None, "ndcg10"], scored),
        "",
        (
            "- Fractional recall: evidence sessions in the top k divided by all evidence sessions, averaged over "
            "questions."
        ),
        (
            "- Any-hit: 1 when at least one evidence session is in the top 5 (the definition of recall_any@5 "
            "below, over every non-abstention question)."
        ),
        "- NDCG, standard discount: binary session relevance, gain divided by log2(rank + 1).",
        "",
    ]
    if official:
        lines += [
            (
                f"LongMemEval's own metrics (src/retrieval/eval_utils.py), {len(official)} questions with a "
                "user-side target turn, same rankings:"
            ),
            "",
            "| Metric | @1 | @5 | @10 |",
            "|---|---|---|---|",
            line("recall_any", "official", [f"recall_any@{k}" for k in KS], official),
            line("recall_all", "official", [f"recall_all@{k}" for k in KS], official),
            line("ndcg_any", "official", [f"ndcg_any@{k}" for k in KS], official),
            "",
            "- recall_any: 1 when at least one evidence session is in the top k. recall_all: 1 when every one is.",
            (
                "- ndcg_any: the official DCG, in which ranks 1 and 2 share the discount 1 and rank i is divided "
                "by log2(i) from rank 3 on."
            ),
            (
                "- The official script indexes user turns only and counts an answer session as evidence only when "
                "one of its user turns has `has_answer`. This harness indexes user/assistant exchanges, so the "
                "scores describe a different retriever from the published baselines."
            ),
            "",
        ]
    else:
        lines += ["LongMemEval's own metrics are not reported: no question has a user-side `has_answer` turn.", ""]
    lines += [
        "| Latency | ms |",
        "|---|---|",
        f"| p50 | {lat[len(lat) // 2]:.0f} |",
        f"| p95 | {lat[min(len(lat) - 1, int(len(lat) * 0.95))]:.0f} |",
        "",
        "| Question type | n | Fractional recall@5 | NDCG@10, standard discount |",
        "|---|---|---|---|",
    ]
    for t in sorted({r["question_type"] for r in scored}):
        sub = [r for r in scored if r["question_type"] == t]
        lines.append(f"| {t} | {len(sub)} | {mean('metrics', 'r5', sub):.3f} | {mean('metrics', 'ndcg10', sub):.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, type=Path, help="longmemeval_s_cleaned.json")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip questions already in the output file; refused if the settings differ")
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--device", default=None, help="auto, mps, cuda or cpu")
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ,
                    help="embedder sequence length in tokens (default 1024); the published benchmark read whole "
                         "chunks, so pass 8192 to match it")
    args = ap.parse_args(argv)

    cfg = load_config()
    device = pick_device(args.device or cfg.device)
    questions = json.loads(args.data.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    embedder = SentenceTransformerEmbedder(cfg.embedder, device, args.max_seq, revision=cfg.embedder_revision)
    reranker = None if args.no_rerank else CrossEncoderReranker(cfg.reranker, device, revision=cfg.reranker_revision)
    extra = {"device": device, "max_seq": args.max_seq, "embedder_revision": cfg.embedder_revision,
             "reranker_revision": None if reranker is None else cfg.reranker_revision}
    results = args.out / "retrieval.jsonl"
    try:
        rows = run(questions, embedder, reranker, results, resume=args.resume, candidates=cfg.candidates,
                   rrf_k=cfg.rrf_k, rerank_chars=cfg.rerank_chars, settings=extra)
    except ResumeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    table = summarize(rows, json.loads(settings_path(results).read_text(encoding="utf-8")))
    (args.out / "summary.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
