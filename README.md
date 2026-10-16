# pureMind for Mac

One memory for Claude Code, Codex and Cursor, kept in plain Markdown files you own, searched and served entirely on your own machine.

pureMind indexes a directory of Markdown notes (an Obsidian vault works as is), builds a local search index, and exposes it to your coding agents over MCP. Every agent reads the same notes, so what you write down once is available to all of them. Nothing leaves your computer except the one-time download of the embedding and reranking models from Hugging Face.

Version 0.1.0. Python 3.11 or newer. macOS (Apple Silicon) and Linux.

## Install

```bash
pipx install .          # from a clone of this repository
puremind --help
```

The first search downloads two models into the Hugging Face cache, about 3.4 GB in total: `Qwen/Qwen3-Embedding-0.6B` (1.2 GB) and `BAAI/bge-reranker-v2-m3` (2.2 GB). Both are Apache-2.0. On Apple Silicon they run on the GPU through MPS; on Linux they use CUDA if present, else the CPU.

## 60-second quickstart

```bash
puremind init ~/Notes        # writes ~/.puremind/config.toml
puremind index               # chunks, embeds and stores every note (incremental)
puremind search "how did I set up the espresso grinder"
puremind connect claude-code # prints the config to add to your agent
```

Re-run `puremind index` after editing notes, or run `puremind watch` to re-index automatically a couple of seconds after files change. Only files whose SHA-256 changed are re-embedded.

## Connect your agents

`puremind connect <agent>` prints the exact snippet and tells you where it goes. It never edits your agent's config files in v0.1.

| Agent | Command | Where the snippet goes |
|---|---|---|
| Claude Code | `puremind connect claude-code` | `claude mcp add --scope user puremind -- puremind serve`, or `.mcp.json` |
| Codex | `puremind connect codex` | `~/.codex/config.toml` |
| Cursor | `puremind connect cursor` | `~/.cursor/mcp.json` |

The snippet uses the absolute path of the `puremind` executable, because GUI apps such as Cursor do not inherit your shell's `PATH`.

The MCP server is read-only and offers exactly three tools:

| Tool | Purpose |
|---|---|
| `puremind_search(query, limit=10)` | Hybrid search. Returns the matching passages with file path, heading path and score. |
| `puremind_get(path)` | Reads one note in full. Paths are confined to the vault. |
| `puremind_recent(days=7, limit=20)` | Lists recently modified notes. |

## How it works

- **Vault.** A directory of `.md` files. Hidden directories (such as `.obsidian`) and symlinks that leave the vault are skipped.
- **Chunker.** Heading-aware: splits on `#` to `####`, keeps a breadcrumb such as `Project > Setup`, never splits inside a fenced code block or YAML frontmatter, merges tiny sections, and cuts sections over 2,048 characters into windows with 20% overlap.
- **Store.** One SQLite file, `~/.puremind/index.sqlite`: files and hashes, chunks, an FTS5 index (used by `puremind search --lexical-only`), and float16 vectors. Vectors are loaded into a numpy matrix for dense search.
- **Retrieval (MAC-16b).** BM25 with an English stemmer (top 50) and Qwen3-Embedding-0.6B cosine search (top 50, 1024 dimensions, with the query instruction prefix) are fused with reciprocal rank fusion (k=60). A BGE cross-encoder then re-scores the 50 fused candidates. `--no-rerank` skips the last step.

## Configuration

`~/.puremind/config.toml` is written by `puremind init`. Environment variables override it.

| Setting | Environment variable | Default |
|---|---|---|
| Vault directory | `PUREMIND_VAULT` | set by `init` |
| Index location | `PUREMIND_DB` | `~/.puremind/index.sqlite` |
| Embedding model | `PUREMIND_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` |
| Reranker model | `PUREMIND_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` |
| Device | `PUREMIND_DEVICE` | `auto` (mps, then cuda, then cpu) |
| Config directory | `PUREMIND_HOME` | `~/.puremind` |

Changing the embedding model makes the next `puremind index` rebuild the index, since vectors from different models are not comparable.

## Benchmark

The retrieval design was measured on LongMemEval_s (500 questions, about 50 chat sessions of history each) in a pre-registered comparison against claude-mem. The "Mac stack" arm is the design in this repository: BM25 plus Qwen3-Embedding-0.6B plus bge-reranker-v2-m3.

| System | Session Recall@5 | Judged QA accuracy |
|---|---|---|
| pureMind Mac stack (this design) | 0.971 | 90.4% |
| claude-mem | 0.928 | 78.2% |

QA used the 10 retrieved exchanges as context. The reader model was `claude-sonnet-5-5` and the judge was `claude-opus-5-5`, using the official LongMemEval per-type judge prompts. A second judge agreed with the first on 99 of 100 sampled verdicts.

Read these numbers with the following limits:

- The Mac stack numbers come from the design measured with an equivalent harness, not from running this package. `bench/longmemeval.py` reproduces the retrieval half with this package's code. It measures Recall and NDCG only and does not run the reader or judge.
- The claim covers retrieval stacks fed identical text. claude-mem's capture pipeline, which uses an LLM observer, was not exercised. Its 256-token embedder sees only the opening of 78% of the raw exchanges it was given. The chunker and the answer prompt were pureMind's.
- The pre-registered significance tests were run on the production pureMind server recipe against claude-mem. The Mac stack arm was exploratory. It was statistically indistinguishable from the production recipe, which is what supports quoting it next to the claude-mem figures.
- The benchmark covers retrieval and question answering over chat history. It says nothing about claude-mem's other features or about your own notes.

To run the retrieval benchmark yourself, see `bench/README.md` and the docstring of `bench/longmemeval.py`.

## What it is not

- **Not a cloud service.** There is no account, no sync and no telemetry. Models are downloaded once from Hugging Face; after that it works offline (set `HF_HUB_OFFLINE=1` to enforce it).
- **No write path yet.** The MCP server cannot create or edit notes. You write the notes; agents read them.
- **No app yet.** v0.1 is a command line tool and a stdio server. There is no menu-bar app, no installer and no auto-start.
- **Not a capture tool.** It does not record agent sessions. It searches the notes you already have.

## Roadmap

- **v0.2:** a write path through a human-approved inbox. Agents propose memories, you approve them, and approved memories become Markdown files in your vault.
- **v0.3:** a menu-bar app that runs the watcher and server, with first-run setup for the supported agents.

## Development

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q          # no network, no models
```

## Licence

Apache-2.0. Copyright PureTensor, Inc. See `LICENSE` and `NOTICE`.
