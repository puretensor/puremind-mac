# pureMind for Mac

One memory for Claude Code, Codex and Cursor, kept in plain Markdown files you own, searched and served entirely on your own machine.

pureMind indexes a directory of Markdown notes (an Obsidian vault works as is), builds a local search index, and exposes it to your coding agents over MCP. Every agent reads the same notes, so what you write down once is available to all of them. Once the two models are downloaded, nothing leaves your computer.

Python 3.11 or newer, on macOS (Apple Silicon) or Linux.

## Install

pureMind for Mac is not published on PyPI. Install it from this repository:

```bash
pipx install git+https://github.com/puretensor/puremind-mac
puremind --help
```

On Linux without an NVIDIA GPU, add `--pip-args='--extra-index-url https://download.pytorch.org/whl/cpu'` to that `pipx install` to get the CPU build of PyTorch, which is a fraction of the size of the default CUDA build. Intel Macs are not supported: the PyTorch releases pureMind needs have no Intel macOS builds. With Python 3.14 on a Mac, pip builds `watchdog` from source, which needs the Xcode command line tools.

Two models are downloaded into the Hugging Face cache the first time they are needed, about 3.5 GB in total: `Qwen/Qwen3-Embedding-0.6B` (1.2 GB, fetched by the first `puremind index`) and `BAAI/bge-reranker-v2-m3` (2.3 GB, fetched by the first search). Both are Apache-2.0. On Apple Silicon they run on the GPU through MPS; on Linux they use CUDA if present, else the CPU.

Homebrew users: there is no tap yet, but a formula is kept in this repository (`packaging/homebrew/puremind.rb`). To use it, put it in a tap of your own:

```bash
brew tap-new "$USER/puremind"
cp packaging/homebrew/puremind.rb "$(brew --repository "$USER/puremind")/Formula/"
brew install "$USER/puremind/puremind"
```

It builds the latest released tag in a Python 3.13 environment, the same as `pipx` does, and needs an Apple Silicon Mac.

## Quickstart

```bash
puremind init ~/Notes        # writes ~/.puremind/config.toml
puremind index               # chunks, embeds and stores every note (incremental)
puremind search "how did I set up the espresso grinder"
puremind connect claude-code # prints the command that registers pureMind with your agent
```

The first `puremind index` embeds every note, which takes from seconds to many minutes depending on the size of the vault and the machine. Run one search from the terminal before you connect an agent, so the reranker download does not happen inside an agent's first tool call.

Re-run `puremind index` after editing notes, or run `puremind watch` to re-index automatically once the vault has been quiet for a couple of seconds. Only files whose SHA-256 changed are re-embedded.

## Connect your agents

`puremind connect <agent>` prints the exact command or snippet and tells you where it goes. It never edits your agent's config files.

| Agent | Command | What it prints |
|---|---|---|
| Claude Code | `puremind connect claude-code` | a `claude mcp add --scope user puremind ...` command |
| Codex | `puremind connect codex` | a `codex mcp add puremind ...` command, or the entry for `~/.codex/config.toml` |
| Cursor | `puremind connect cursor` | the entry for `~/.cursor/mcp.json` |

The snippet uses the absolute path of the `puremind` executable and carries any `PUREMIND_*`, `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `HF_ENDPOINT` variables set in your shell, because GUI apps such as Cursor do not inherit your shell's environment.

The MCP server offers four tools. Three only read your vault and are marked read-only for the client; the fourth, `puremind_propose`, cannot touch your notes (see Inbox below).

| Tool | Purpose |
|---|---|
| `puremind_search(query, limit=10)` | Hybrid search. Returns up to 50 matching passages, best first, each with its note path, heading path and score. A query is at most 4,096 characters. |
| `puremind_get(path)` | Reads one note by the path search returned. It serves only the notes the indexer reads (see Vault below) and cuts text beyond 200,000 characters, saying so. |
| `puremind_recent(days=7, limit=20)` | Lists the notes modified in the last `days` days, newest first, as of the last index run. |
| `puremind_propose(title, content)` | Proposes a memory. It is saved in an inbox outside the vault and does nothing else until you approve it. |

Each agent session starts its own server process. Tool calls may overlap: every search is answered from one consistent snapshot of the index, even while `puremind watch` updates it.

**Scores.** With the reranker, a score is the cross-encoder's probability from 0 to 1. Without it (`--no-rerank`, or `rerank = false`), it is a reciprocal rank fusion value, at most about 0.033. `puremind search --lexical-only` prints BM25 keyword scores. Compare scores only within one result list. There is no relevance cut-off, so a search always returns up to `limit` passages, and weak ones are still listed.

## Inbox: agents propose, you approve

An agent can suggest that something is worth remembering, but it cannot write to your notes. `puremind_propose` saves the suggestion as a small file in `~/.puremind/inbox/` (mode 0600, outside the vault). A proposal is not indexed, not searchable and not readable through `puremind_get` until you approve it.

```bash
puremind inbox                      # list the proposals waiting
puremind inbox show 20261022-a1b2   # read one (an id, or the start of one)
puremind inbox approve 20261022-a1b2
puremind inbox reject 20261022-a1b2
```

`approve` prints the whole proposal and the note it would create, and asks before writing. The note goes to `agent-memories/<date>-<title>.md` in your vault (change the folder with `[inbox] folder` in `config.toml`, or pass `--path projects/grinder.md` for one note) and starts with the proposal's title as a `#` heading. It only ever creates a file: it never overwrites or edits an existing note (a name that is taken gets `-2`, `-3`, and so on, and an explicit `--path` that exists is an error), it refuses `..`, hidden folders and absolute paths, and it will not write through a symlink at any level. Run `puremind index` afterwards, or let `puremind watch` do it.

Limits: a title of at most 200 characters, content of at most 20,000, and 100 proposals waiting at a time; an identical proposal is kept once.

- **This is a review step, not a sandbox.** Approval asks you at a terminal, and without one needs `--yes`. An agent that can run shell commands as you could pass `--yes` itself. If your agent has a shell, treat the inbox as a safeguard against mistakes and prompt injection through the MCP server, not as a boundary against the agent, and read what you approve.
- **Read before you approve.** A proposal is text an agent wrote, perhaps from a web page or a file it read. Once approved it is indexed, and any agent may later be shown it as a search result.
- **Turn it off.** `puremind serve --read-only` (add it to the arguments of the server entry), or `enabled = false` under `[inbox]` in `config.toml`, leaves the server with its three read-only tools.

## How it works

- **Vault.** The indexer reads regular `.md` and `.markdown` files of at most 2 MiB. It skips hidden files and folders (such as `.obsidian` and `.git`), `node_modules` and `__pycache__` (in any letter case), symlinks that resolve outside the vault or to a skipped file, pipes, names that are not valid UTF-8, and paths inside the vault longer than 1,024 characters. A note that is replaced by such a symlink during a run is not read through it. A note it cannot read keeps its previous index entry. Notes below a folder that cannot be listed are kept in the index, nothing is removed when the vault looks empty while the index is not, and the run says why.
- **Chunker.** Heading-aware: splits on `#` to `####` and keeps a breadcrumb such as `Project > Setup`. Headings inside fenced code blocks and YAML frontmatter do not count. A heading with no text of its own joins the subsection under it, and most sections under 100 characters merge with the one before. A section longer than 2,048 characters is cut into near-even windows with about 20% overlap, preferably at blank lines. A code block or frontmatter that fits in a window is usually kept whole, but one that is longer, or that starts late in a long section, can be cut; every part of a cut block keeps its own opening and closing fence lines. Frontmatter of 100 characters or more is a chunk of its own.
- **Store.** One SQLite file, `~/.puremind/index.sqlite`, readable only by you: files and hashes, chunks, an FTS5 keyword index (used by `puremind search --lexical-only`), and float16 vectors. Vectors are loaded into a numpy matrix for dense search.
- **Retrieval.** BM25 over English-stemmed words (top 50) and Qwen3-Embedding-0.6B cosine search (top 50, 1024 dimensions, with the query instruction prefix) are fused with reciprocal rank fusion (k=60). The bge-reranker-v2-m3 cross-encoder then re-scores the 50 fused candidates, reading the first 800 characters of each. `--no-rerank` skips that last step.
- **Languages.** The embedder is multilingual, so search by meaning works in most languages. Keyword matching uses an English stemmer. In hybrid search, Chinese, Japanese and Korean text is matched by single characters and overlapping pairs of characters; `search --lexical-only` does not do this, and treats an unspaced run of those characters as one word. Thai, Lao, Khmer and Burmese are not split into words for keyword matching. In scripts that need more than one token per character, such as Telugu, Burmese, Khmer and Ethiopic, a full-length chunk is longer than the 2,048 tokens the embedder reads, so search by meaning misses its end (keyword search does not); raise `embed_max_seq` to about 3,200 if most of your notes are in such a script, then run `puremind index --rebuild`.
- **Watcher.** `puremind watch` indexes once, then again each time the vault has been quiet for `--debounce` seconds (default 2) after a change to a visible note or folder. Reads, hidden folders and other file types never trigger a run, and runs never overlap. It follows a vault directory that is replaced or made again (for example by a remount or a sync tool) or that cannot be read for a while, and if nothing has happened for 10 minutes it scans anyway, in case a change was missed. A run that fails is reported on one line and watching goes on, except when no later run could succeed (a vault symlink that now names another directory than the index's, or a system limit such as the inotify watch limit): then `puremind watch` stops with an error.

## Memory and speed

Each `puremind serve` process loads the models on its first search, which takes several seconds. Measured on Linux on 8 CPU cores, with the CPU build of PyTorch and current releases of the libraries:

| | With the reranker | With `serve --no-rerank` |
|---|---|---|
| Resident memory per server | about 3 to 3.5 GB, up to about 3.8 GB while reranking | about 1.5 GB |
| A search after the first | 10 to 20 seconds | about 0.3 seconds |

The reranker costs about 0.2 seconds per candidate on a CPU, so with the default 50 candidates it takes most of the search time. On a machine without a GPU, use `--no-rerank`, or lower `candidates`, if that is too slow. On Apple Silicon both models run on the GPU (MPS). We have only a first check, on an Apple M5 Pro with a 4-note vault, and it is not a benchmark: the first search, which loads both models, took about 5 seconds and the next ones 0.2 seconds. Memory use on a Mac has not been measured. An idle server that has not searched yet uses under 100 MB. Older transformers 4.x releases load the models as 32-bit floats and need about twice the memory.

## Configuration

`~/.puremind/config.toml` is written by `puremind init` (and rewritten, keeping your settings, by `puremind init --force`). Environment variables override it.

| Setting | In config.toml | Environment variable | Default |
|---|---|---|---|
| Vault directory | `vault` | `PUREMIND_VAULT` | set by `init` |
| Index file | `db` | `PUREMIND_DB` | `~/.puremind/index.sqlite` |
| Embedding model | `[models] embedder` | `PUREMIND_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` |
| Embedding model revision | `[models] embedder_revision` | `PUREMIND_EMBED_REVISION` | a pinned commit, see below |
| Reranker model | `[models] reranker` | `PUREMIND_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` |
| Reranker revision | `[models] reranker_revision` | `PUREMIND_RERANK_REVISION` | a pinned commit, see below |
| Device | `[models] device` | `PUREMIND_DEVICE` | `auto` (mps, then cuda, then cpu) |
| Tokens of each chunk the embedder reads | `[models] embed_max_seq` | | `2048` |
| Candidates from each retriever | `[retrieval] candidates` | | `50` |
| Reciprocal rank fusion constant | `[retrieval] rrf_k` | | `60` |
| Rerank the candidates | `[retrieval] rerank` | | `true` |
| Offer `puremind_propose` | `[inbox] enabled` | | `true` |
| Folder for approved proposals | `[inbox] folder` | | `agent-memories` |
| Characters of each passage the reranker reads | `[retrieval] rerank_chars` | | `800` |
| Config directory | | `PUREMIND_HOME` | `~/.puremind` |

- **Model revisions.** The default models are pinned to `Qwen/Qwen3-Embedding-0.6B` at `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` and `BAAI/bge-reranker-v2-m3` at `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Another model uses its repository's default branch as it was at the first download, and is not refreshed afterwards unless you set a revision or clear the cache.
- **Model names.** A name is a Hugging Face model id unless it starts with `/`, `./`, `../` or `~`, which mark a local directory. A directory in your working directory is never picked up by accident. Models load with `trust_remote_code` off.
- **Changing the embedding model**, `embedder_revision` or `embed_max_seq` takes `puremind index --rebuild`, since vectors from different models are not comparable, and a new window changes the vectors of long chunks. Until then `puremind index` and `puremind watch` stop with an error and change nothing, and searches are refused. A `puremind watch` keeps the model and window it started with, so restart it after the change; one left running stops instead of wiping the rebuilt index. A running `puremind serve` keeps them too, and refuses searches until it is restarted. With `--rebuild`, the new model must first load and embed a test passage, so a mistyped name leaves the existing index untouched. The index is then emptied, and searches find nothing until the run finishes. `puremind index --rebuild` with the same model, by contrast, replaces one note at a time, so searches keep working.
- **The reranker window.** 800 characters is the setting the benchmark below was measured with. Raising `rerank_chars` lets the reranker see more of each passage, at the cost of slower searches.
- **Indexes from 0.1.1 or earlier** are brought up to date by the next `puremind index` or `puremind watch` run, which cuts and embeds every note again. It takes about as long as the first index did, and searches keep working meanwhile.

## Network and privacy

Once both models are in the Hugging Face cache, pureMind makes no network requests: models load from the cache at their pinned revisions, and Hugging Face telemetry is turned off unless you have set `HF_HUB_DISABLE_TELEMETRY` yourself. The only network use is the one-time download, which sends your Hugging Face token if you have one configured. Set `HF_HUB_OFFLINE=1` to forbid downloads altogether; a model that is not downloaded then fails at once with a clear message. A cached copy that lacks a file the models need (after an interrupted download, or one made by another tool) is completed on the next run with network access, and reported by name when offline.

Proposals waiting in `~/.puremind/inbox/` are a copy of the text an agent proposed, kept until you approve or reject them; a rejected or approved one is deleted from the inbox.

The index is a second copy of your notes' text. It is created readable only by you (mode 0600 in a 0700 directory), and an index, `~/.puremind` directory or `config.toml` that an earlier version left more open is tightened when pureMind next uses it. When a note is deleted or edited, the index run that notices it removes the old text from the index file and its `-wal` file; if another process is reading the index at that moment, the `-wal` file is emptied by a later run. The keyword index keeps the old words a little longer under `puremind watch`: it is compacted once the vault has been quiet for a minute, and at the latest five minutes after the change, rather than after every save. A watch that stops first compacts on its way out, and one that crashes leaves a note in the index so the next `puremind index` or `puremind watch` does it. `puremind index` compacts at the end of every run. Text deleted while you used 0.1.1 or earlier is removed when 0.1.2 first opens the index. Copies of the index in backups or snapshots still hold it; to remove everything, delete `~/.puremind/index.sqlite` and the `-wal` and `-shm` files beside it, then run `puremind index` again.

## Command line

| Command | What it does |
|---|---|
| `puremind init <vault_dir> [--force]` | Writes `config.toml`. |
| `puremind index [vault_dir] [--rebuild]` | Indexes the vault incrementally; `--rebuild` embeds every note again, and is needed after changing the embedding model or `embed_max_seq`. |
| `puremind watch [vault_dir] [--debounce SECONDS]` | Indexes, then re-indexes when notes change. |
| `puremind search <query> [--limit N] [--no-rerank] [--lexical-only]` | Searches from the terminal; `--limit` is 1 to 50. |
| `puremind serve [--no-rerank] [--read-only]` | Runs the MCP server on stdio. Your agent starts this for you. `--read-only` leaves out `puremind_propose`. |
| `puremind inbox [list\|show ID\|approve ID [--path P] [--yes]\|reject ID]` | Reviews the memories agents have proposed. |
| `puremind connect <claude-code\|codex\|cursor>` | Prints the setup snippet for an agent. |

Exit status: 0 on success, 1 when a search finds nothing, 2 for an error such as a missing vault, a broken config file, a model that cannot load or an index file that cannot be written (reported on an `error:` line, sometimes followed by a hint), and 130 after Ctrl-C.

## What it is not

- **Not a cloud service.** There is no account, no sync and no telemetry.
- **No agent write access.** The MCP server cannot create or edit notes. Agents can only propose, and you approve (see Inbox). You write the notes; agents read them.
- **No app yet.** v0.1 is a command line tool and a stdio server. There is no menu-bar app, no installer and no auto-start.
- **Not a capture tool.** It does not record agent sessions. It searches the notes you already have.

## Benchmark

We measured the retrieval design in this repository on LongMemEval_s (500 questions, each with about 50 sessions of chat history) against claude-mem v13.39.0. pureMind is not affiliated with claude-mem. The numbers come from our own harness; read them with the limits listed under the table.

| System | Session Recall@5 | Judged QA accuracy |
|---|---|---|
| This design: BM25, Qwen3-Embedding-0.6B, bge-reranker-v2-m3 | 0.971 | 90.4% |
| claude-mem v13.39.0, storage and search engine only (see below) | 0.928 | 78.2% |

- **Session Recall@5** is the share of a question's evidence sessions that appear in the top 5 results, averaged over the 470 questions that have evidence. The other 30 are abstention questions with nothing to find.
- **Judged QA accuracy** covers all 500 questions. A reader model (`claude-sonnet-5-5`) answered from the top 10 retrieved exchanges, and a judge model (`claude-opus-5-5`) graded each answer with the official LongMemEval per-type prompts. A second judge agreed on 99 of 100 sampled verdicts. The QA difference is +12.2 percentage points (95% confidence interval +9.0 to +15.6).

### Results by question type

LongMemEval_s groups its questions into six types. The same run of this design (BM25, Qwen3-Embedding-0.6B, bge-reranker-v2-m3), split by type. The totals are the figures in the table above.

| Question type | Recall@5 (questions with evidence) | Judged QA accuracy |
|---|---|---|
| knowledge-update | 1.000 (72) | 93.6% (73 of 78) |
| multi-session | 0.951 (121) | 81.2% (108 of 133) |
| single-session-assistant | 1.000 (56) | 98.2% (55 of 56) |
| single-session-preference | 0.967 (30) | 93.3% (28 of 30) |
| single-session-user | 1.000 (64) | 98.6% (69 of 70) |
| temporal-reasoning | 0.946 (127) | 89.5% (119 of 133) |
| **All** | **0.971 (470)** | **90.4% (452 of 500)** |

Multi-session questions, which need evidence from several past conversations, and temporal-reasoning questions are the weakest types. Each type has between 30 and 133 questions, so differences of a few points between types are within noise. Recall@5 counts only the questions that have evidence sessions; the 30 abstention questions are part of the QA column, spread over the types they belong to. These are aggregates of the same exploratory run as above, with the same limits.

Limits:

- **Not this package on a Mac.** The design ran in our benchmark harness, with the models served from an NVIDIA GPU (the embedder in bfloat16, reading whole exchanges). No Mac hardware was used. `bench/longmemeval.py` reproduces the retrieval half with this package's code; it does not run the reader or the judge.
- **claude-mem was not used the way its users use it.** Both systems received the same chat exchanges, one chunk per user and assistant exchange, cut by the benchmark harness (not by this package's Markdown chunker). claude-mem got them through its import API, one observation per exchange, so its LLM observer, which normally writes short summaries and facts, was bypassed. Searches passed an explicit full date range, because the default 90-day window drops older memories to keyword search only (Recall@5 0.896 and QA 79.0% in that configuration). This compares retrieval engines fed identical text, not the two memory products.
- **No reranker in claude-mem.** Part of the gap is the cross-encoder. pureMind's server stack scored Recall@5 0.942 without its reranker, against claude-mem's 0.928; the reranker added 0.035.
- **Embedding window.** claude-mem's default embedder reads only the first 256 tokens of a document. Re-running it with every exchange split into facts of at most 256 tokens left its Recall@5 at 0.921 and its QA at 76.0%.
- **Home field.** The harness's exchange chunking and answer prompt were developed in part against LongMemEval_s in August 2026, before this comparison.
- **Patched dependencies.** To let ingestion finish in time, two performance patches were applied to claude-mem's Chroma dependencies during the run (a cached ONNX session and a narrower duplicate-id check). They left its vectors bit-identical and were reverted afterwards.
- **Exploratory arm.** The study protocol was written before any scored run but is not published. Its primary endpoints were set on pureMind's larger server stack (Qwen3-Embedding-4B), which scored Recall@5 0.977 and QA 90.4% against claude-mem's 0.928 and 78.2% (Holm-corrected p = 0.0002 and p < 0.0001). The design in this repository was an exploratory arm of the same study, and its QA run was added after the primary QA result was known. It was statistically indistinguishable from the server stack.
- **Scope.** The benchmark covers retrieval and question answering over chat history. It says nothing about claude-mem's other features, or about how either system does on your own notes.

To run the retrieval benchmark yourself, see `bench/README.md`.

## Roadmap

- **v0.2 (done, unreleased):** a write path through a human-approved inbox, and a Homebrew formula in `packaging/homebrew/`.
- **v0.3:** a menu-bar app that runs the watcher and server, with first-run setup for the supported agents.

## Development

CI runs the test suite on macOS (Apple Silicon) and Linux with Python 3.11, 3.12 and 3.13, and on Linux with 3.14. The tests use stand-in models, and a weekly job runs a smoke test with the real models on Linux. The real models have been run by hand on an Apple M5 Pro (index and search), but CI does not run them on a Mac. See `CONTRIBUTING.md` to run the tests, the linter and the other CI checks yourself.

## Licence

Apache-2.0. Copyright PureTensor, Inc. See `LICENSE` and `NOTICE`.

Claude Code, Codex, Cursor and claude-mem are trademarks of their owners. pureMind is not affiliated with or endorsed by them.
