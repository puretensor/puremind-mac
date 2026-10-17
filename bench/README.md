# Benchmarks

`longmemeval.py` measures this package's hybrid retrieval on LongMemEval_s (Wu et al., arXiv 2410.10813), with the same code `puremind search` runs. The dataset is not included; install the package and download the file as the script's docstring shows:

```bash
pip install -e .
hf download xiaowu0162/longmemeval-cleaned longmemeval_s_cleaned.json --repo-type dataset --local-dir ./data
python bench/longmemeval.py --data data/longmemeval_s_cleaned.json --limit 20   # smoke test
python bench/longmemeval.py --data data/longmemeval_s_cleaned.json             # all 500 questions
```

Results land in `bench/results/` (git-ignored): `retrieval.jsonl` with one row per question, `retrieval.settings.json` with every setting that affects the rankings, and `summary.md`. `--resume` continues an interrupted run, and refuses to mix results produced with different settings.

On a CPU a question takes a few minutes, since it embeds about 250 exchanges and reranks 50 candidates, so the full run takes roughly a day; a GPU is much faster.

The embedder reads at most `--max-seq` tokens per chunk, 1024 by default. A chunk here is a whole chat exchange of up to 12,000 bytes, and 0.5% of them (640 of 124,362) are longer than 1,024 tokens. The README's figures were measured with an embedder that read whole chunks (up to 8,192 tokens), so the default can differ slightly from them; pass `--max-seq 8192` to use the same setting, which is slower on a CPU.

Scope: this measures retrieval only. `summary.md` reports fractional session Recall@k, Any-hit@5 and NDCG@10 over the 470 non-abstention questions, plus LongMemEval's own recall_any, recall_all and ndcg_any on the subset of questions its scoring script covers. The script does not run a reader model or a judge, so it does not reproduce the QA accuracy in the main README, and it does not run claude-mem.
