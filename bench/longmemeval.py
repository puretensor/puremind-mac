#!/usr/bin/env python3
"""Reproduce the R1 Mac arm (MAC-16b retrieval) on LongMemEval_s.

The dataset is not vendored. Download the cleaned LongMemEval_s file yourself:

    huggingface-cli download xiaowu0162/longmemeval-cleaned longmemeval_s_cleaned.json \
        --repo-type dataset --local-dir ./data

then run, from the repository root:

    python bench/longmemeval.py --data data/longmemeval_s_cleaned.json
    python bench/longmemeval.py --data data/longmemeval_s_cleaned.json --limit 20   # smoke test
    python bench/longmemeval.py --data data/longmemeval_s_cleaned.json --no-rerank

Each question is one simulated user. Its haystack sessions become that user's memory:
one chunk per user/assistant exchange, prefixed with the session date, capped at 12,000
bytes. Retrieval is the same code `puremind search` runs (BM25 top-50 + dense top-50,
RRF k=60, cross-encoder over the 50 fused candidates). Chunk rankings collapse to session
rankings (first occurrence wins) and are scored against `answer_session_ids`.

Metrics, over the non-abstention questions (ids not ending in `_abs`):
  Recall@k  = |evidence sessions in the top k| / |evidence sessions|   (k = 1, 5, 10)
  NDCG@10   = DCG@10 / ideal DCG@10 with binary session relevance and log2(rank + 1) discounts
Output: bench/results/retrieval.jsonl (one row per question) and bench/results/summary.md.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puremind_mac.config import load_config  # noqa: E402
from puremind_mac.models import CrossEncoderReranker, Embedder, Reranker, SentenceTransformerEmbedder  # noqa: E402
from puremind_mac.retrieval import HybridIndex  # noqa: E402

MIN_CHARS = 3
MAX_BYTES = 12_000
RESULTS = Path(__file__).resolve().parent / "results"


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
    m = {f"r{k}": len(evidence & set(ranked_sessions[:k])) / len(evidence) for k in (1, 5, 10)}
    m["any5"] = float(bool(evidence & set(ranked_sessions[:5])))
    dcg = sum(1 / math.log2(i + 1) for i, s in enumerate(ranked_sessions[:10], 1) if s in evidence)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(evidence), 10) + 1))
    m["ndcg10"] = dcg / idcg
    return m


def run(
    questions: Sequence[dict],
    embedder: Embedder,
    reranker: Reranker | None,
    out_path: Path,
    resume: bool = False,
    candidates: int = 50,
    rrf_k: int = 60,
    rerank_chars: int = 800,
) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if resume and out_path.exists():
        done = {r["qid"]: r for r in map(json.loads, out_path.read_text().splitlines()) if r}
    cache: dict[str, np.ndarray] = {}  # chunk text -> float16 vector; sessions repeat across haystacks
    t_all = time.time()
    with out_path.open("a" if resume else "w") as fh:
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
            done[qid] = row
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(questions)} ({time.time() - t_all:.0f}s)", flush=True)
    return list(done.values())


def summarize(rows: Sequence[dict]) -> str:
    scored = [r for r in rows if "metrics" in r]
    if not scored:
        return "No scored questions.\n"
    mean = lambda key, rs: statistics.fmean(r["metrics"][key] for r in rs)  # noqa: E731
    lat = sorted(r["latency_ms"] for r in rows)
    lines = [
        f"LongMemEval_s retrieval, {len(scored)} non-abstention questions "
        f"({sum(r['rerank_applied'] for r in rows)}/{len(rows)} reranked)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Recall@1 | {mean('r1', scored):.3f} |",
        f"| Recall@5 | {mean('r5', scored):.3f} |",
        f"| Recall@10 | {mean('r10', scored):.3f} |",
        f"| Any-hit@5 | {mean('any5', scored):.3f} |",
        f"| NDCG@10 | {mean('ndcg10', scored):.3f} |",
        f"| Latency p50 (ms) | {lat[len(lat) // 2]:.0f} |",
        f"| Latency p95 (ms) | {lat[min(len(lat) - 1, int(len(lat) * 0.95))]:.0f} |",
        "",
        "| Question type | n | Recall@5 | NDCG@10 |",
        "|---|---|---|---|",
    ]
    for t in sorted({r["question_type"] for r in scored}):
        sub = [r for r in scored if r["question_type"] == t]
        lines.append(f"| {t} | {len(sub)} | {mean('r5', sub):.3f} | {mean('ndcg10', sub):.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, type=Path, help="longmemeval_s_cleaned.json")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip questions already in the output file")
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--device", default=None, help="auto, mps, cuda or cpu")
    args = ap.parse_args(argv)

    cfg = load_config()
    device = args.device or cfg.device
    questions = json.loads(args.data.read_text())
    if args.limit:
        questions = questions[: args.limit]
    embedder = SentenceTransformerEmbedder(cfg.embedder, device, cfg.embed_max_seq)
    reranker = None if args.no_rerank else CrossEncoderReranker(cfg.reranker, device)
    rows = run(questions, embedder, reranker, args.out / "retrieval.jsonl", resume=args.resume,
               candidates=cfg.candidates, rrf_k=cfg.rrf_k, rerank_chars=cfg.rerank_chars)
    table = summarize(rows)
    (args.out / "summary.md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
