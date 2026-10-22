# Contributing

Issues and pull requests are welcome. v0.1 is deliberately small, so open an issue before starting anything large.

## Set up and test

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
```

The suite is hermetic and needs no network or models. For every test it removes all `PUREMIND_*` variables, sets `HF_HUB_OFFLINE=1`, and points `HOME` and `PUREMIND_HOME` at a temporary directory, so a `PUREMIND_DB` or `PUREMIND_VAULT` exported in your shell cannot touch your own index or vault. Tests use the fake embedder and reranker in `tests/conftest.py`.

Optional runs:

- `PUREMIND_REAL_MODEL_TESTS=1 python -m pytest -q` adds a smoke test with the real models. It uses your Hugging Face cache and downloads them there (about 3.5 GB) if they are missing. CI runs it weekly.
- `PUREMIND_TEST_INSTALLED=1` tests an installed copy of the package instead of `src/`, as CI does with the built wheel. It needs a regular install, not the editable one above, so build the wheel and install it into a separate environment:

  ```bash
  python -m build --wheel && python -m venv ../wheel-env
  ../wheel-env/bin/pip install "$(ls dist/*.whl)[dev]"
  PUREMIND_TEST_INSTALLED=1 ../wheel-env/bin/python -m pytest -q
  ```

## The other CI checks

```bash
pip install ruff==0.17.1 && ruff check src tests bench        # lint, ruff's default rules
pip install build twine && python -m build && twine check --strict dist/*
uv pip install --resolution lowest-direct --torch-backend cpu -e '.[dev]'   # in a fresh venv: lowest allowed versions
gitleaks git --config .gitleaks.toml --redact .               # secrets in the history
```

CI also runs pip-audit on the dependencies a new install resolves, and the suite against the mcp 1.x SDK.

## Rules

- The MCP server never writes to the vault. An agent can only leave a proposal in the inbox (`puremind_propose`), and only `puremind inbox approve`, run by the user, creates a note. A change that gives an MCP tool any other way to write needs a design discussion first.
- Keep the retrieval pipeline (BM25 top 50, dense top 50, RRF k=60, rerank over 50 with an 800-character window) unchanged unless a change is backed by a benchmark run.
- A new third-party import in `src/` needs a dependency in `pyproject.toml`, with a lower bound you have actually run; a test checks this.
- The version lives only in `src/puremind_mac/__init__.py`. A release bumps it and adds the matching heading to `CHANGELOG.md` in the same commit; a test checks that too.
- Add a line to `CHANGELOG.md` for every user-visible change.

By contributing you agree that your work is licensed under Apache-2.0.
