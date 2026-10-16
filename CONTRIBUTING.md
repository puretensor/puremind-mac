# Contributing

Issues and pull requests are welcome. v0.1 is deliberately small, so open an issue before starting anything large.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
```

- Tests must run without network access and without downloading models. Use the fake embedder and reranker in `tests/conftest.py`.
- The MCP server stays read-only. The write path is planned for v0.2 and will arrive with its own design discussion.
- Keep the retrieval pipeline (BM25 top 50, dense top 50, RRF k=60, rerank over 50) unchanged unless a change is backed by a benchmark run.
- Add a line to `CHANGELOG.md` for user-visible changes.

By contributing you agree that your work is licensed under Apache-2.0.
