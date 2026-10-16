# Benchmarks

`longmemeval.py` reproduces the retrieval half of the R1 Mac arm on LongMemEval_s. The dataset is not
included; the download command is in the script's docstring.

Results land in `bench/results/` (`retrieval.jsonl`, `summary.md`), which is git-ignored.

Scope: this measures retrieval only (session Recall@k and NDCG@10 against the dataset's evidence
session ids). It does not run a reader model or a judge, so it does not reproduce the QA accuracy
figure in the main README.
