# Changelog

## 0.1.4 (unreleased)

- A new embedding model, or a new `embedder_revision`, now needs `puremind index --rebuild`. `puremind index` without it stops with an error and changes nothing; 0.1.3 and earlier emptied the index and embedded every note again. A `puremind watch` still running on the old configuration, and a fresh `puremind index` on the new one, used to wipe each other's index in turn, with a full re-embed each time. Now the stale watch stops with an error and a hint to restart it, and the index stays as it is. An index that holds no notes takes a new model without `--rebuild`.

## 0.1.3 (2026-10-17)

Fixes from a second review of 0.1.2. Upgrading from 0.1.2 embeds nothing again. The `v0.1.2` tag was made before these fixes, so an install from `main` between the two releases reports 0.1.2 but already has them.

### Indexing

- A local model path is compared in absolute form, so an index built with a path as typed (`~/models/x`) no longer makes every search from `puremind serve` fail with a model mismatch that `--rebuild` could not clear.
- The index records the model's revision. A changed `embedder_revision` now causes a full re-embed: the next index run empties the index and embeds every note again, and searches are refused until then. An index from an earlier release, which recorded no revision, is kept as it is.
- When a folder cannot be listed (a root-owned `lost+found`, for example), only the notes below it are kept in the index. Notes deleted elsewhere in the vault are removed; 0.1.2 removed none while any folder was unreadable.
- Two processes upgrading an index from 0.1.1 at the same time no longer fail with "duplicate column".

### Watcher

- `puremind watch` stops with an error when a system limit keeps it from watching the vault (the inotify watch limit, or no free file descriptors or memory). 0.1.2 took this for a missing vault and retried four times a second without end, leaking a file descriptor each time.
- Changes are no longer missed when the vault path and the paths in file events differ only in letter case or Unicode normalisation.
- `puremind watch` stops when a vault symlink is pointed at a directory the index does not belong to, instead of failing every run.

### Models

- Each model runs one encode or rerank call at a time. Overlapping tool calls shared a model without a lock, and its fast tokenizer could fail with "Already borrowed".
- When a cached model copy has what the models need but completing its missing files fails, the cached copy is used instead of the load failing.
- The offline error names the variable that is set (`HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE`).

### Chunker

- A tiny section merged into the chunk before it is labelled with the headings the sections share, so the breadcrumb covers all of its text. Notes already indexed keep their breadcrumbs until they change or `puremind index --rebuild` runs.

### MCP tools

- `puremind_get` clears non-blocking mode before reading, so a note that has to be fetched first (a cloud placeholder) no longer fails with `TypeError`.
- Tool errors name a local model by its last path component, so no absolute path reaches the agent. The command line still prints the full path.

### Command line

- Bidirectional control characters (U+202E, for example) are removed from printed text, so a note can no longer reorder the lines `search` prints.
- `puremind connect` also carries `TRANSFORMERS_OFFLINE` and `HF_ENDPOINT`.

### Benchmark

- `bench/longmemeval.py` follows `[retrieval] rerank`.

## 0.1.2 (2026-10-17)

Fixes from a full review of 0.1.1. The first `puremind index` or `watch` run after upgrading cuts and embeds every note again, so it takes about as long as the first index did; searches keep working meanwhile.

### Concurrency

- Overlapping tool calls no longer fail or return the wrong note. mcp 2.x runs each tool on a worker thread, and overlapping first searches could fail with `AttributeError: 'BM25' object has no attribute ...`, while a search running during a reload could return another note's text or raise `IndexError`. Each search now answers from one snapshot of the index, and the keyword index is built once, before any search uses it.
- SQLite reads are now fetched while the connection lock is held, and every read is a single statement. The lock added in 0.1.1 was released before rows were fetched, so it did not make the shared connection safe on Python 3.12 and newer, and readers could see wrong or missing values. (0.1.1 also said the watcher shared the server's connection; it does not, `puremind watch` is a separate process.)
- The three tools are marked read-only, idempotent and closed-world for MCP clients.

### Watcher

- `puremind watch` on Linux no longer re-triggers itself: the indexer's own reads of notes started the next run, in an endless loop that could exhaust memory with the real models. Reads, directory metadata events, hidden folders and other file types are now ignored.
- Runs happen one at a time on the main thread, and a change during a run schedules exactly one more. A note edited while it was being embedded is read again before it is stored, so an older version can no longer overwrite a newer one.
- Ctrl-C exits with status 130 and no traceback, without leaving the index half-written.
- The watcher follows a vault directory that is replaced (a remount, or a sync tool swapping the folder), made again under the same inode number (ext4 does this), or unreadable for a while, and scans anyway after 10 quiet minutes. A vault that disappears again while it is being watched no longer ends `puremind watch`, and nothing runs while the vault is missing.
- A run that fails while watching is reported on one line, without a traceback, and watching goes on.

### Indexing

- The vault-ownership check runs before anything is cleared, and a new embedding model must embed a test passage before the old index is discarded. A mistyped model name, or a test run with `PUREMIND_DB` exported, can no longer wipe an index.
- Nothing is removed from the index when a folder cannot be listed or the vault looks empty while the index is not (an unmounted drive, for example). A note that exists but cannot be read keeps its entry.
- One bad note no longer stops a run: symlink loops, pipes, names that are not valid UTF-8, and notes that fail to read or chunk are skipped (the last three are also reported). Visible symlinks to hidden or non-Markdown files are skipped, and a note replaced by such a symlink while a run is in progress is not read through it.
- A failing index file (a full disk, or another process holding the lock too long) stops the run with an error and exit status 2, instead of skipping every remaining note.
- Notes whose path inside the vault is longer than 1,024 characters are skipped and reported, since `puremind_get` could not open them.
- Embedding is batched across notes (64 chunks per call), two to four times faster for vaults of small notes in our measurements. When the model fails on a batch, its notes are embedded one at a time, so a note the model cannot handle is skipped and named instead of stopping the run; a model that fails on every note still stops it.
- `--rebuild` embeds every note again in place, so searches keep finding the old version of each note until it is replaced. It used to empty the index first. (A new embedding model still empties it, since vectors from two models cannot be searched together.)
- Each note records the version of the chunker that cut it, so a release that changes chunking cuts every note again by itself. 0.1.2 does this once for indexes from earlier versions.
- Folders nested so deep that no note below them could be opened are not entered. On Python 3.11 a vault nested about 1,000 folders deep used to stop the run.
- An index whose vault path is spelled differently (APFS ignores letter case) is recognised as the same vault.

### Chunker

- Long sections are cut into near-even windows that begin at a line or word boundary and end at blank lines where possible. A fenced code block or frontmatter that fits in a window is usually kept whole, and every part of one that is cut keeps its own fence lines.
- A heading with no text of its own joins the subsection under it, instead of becoming a chunk of just the heading.
- A byte-order mark no longer hides the first heading, and headings inside frontmatter are not taken as the note's title.
- Heading text in breadcrumbs is clipped to 200 characters, and a note yields at most 10,000 chunks, so a hostile note cannot blow up the index.

### Retrieval

- Chinese, Japanese and Korean words now match in hybrid search: runs of those characters are indexed as single characters and overlapping character pairs, so one-character words such as 猫 match too. English and other text tokenises exactly as before.
- `embed_max_seq` (tokens of each chunk the embedder reads) defaults to 2048 and is now read from config.toml. At 1024, the tail of long chunks in many non-Latin scripts was silently dropped. Scripts that need more than one token per character (Telugu, Burmese, Khmer and Ethiopic, for example) still exceed 2048 in a full chunk; raise the setting for notes in those scripts.
- A vault in which no chunk has a keyword token (emoji only, for example) no longer makes the first search fail.
- `[retrieval] rerank = false` and `puremind serve --no-rerank` run without the reranker, using about 1.5 GB less memory.

### Models and network

- Models load from the Hugging Face cache at pinned revisions, and Hugging Face telemetry is off. Once the models are cached, pureMind makes no network requests. 0.1.1 contacted the Hub on every run, with dozens of requests, and stalled for minutes when the Hub did not answer unless `HF_HUB_OFFLINE=1` was set.
- A cached copy that lacks a file the models need is completed by a download, or reported by name when that is not possible. An interrupted download no longer leaves a copy that never heals, and a copy made by transformers alone (without the sentence-transformers files) can no longer make the embedder pool its output differently.
- When offline, a model that is cached only at another revision is reported with the setting that uses it.
- A model name is looked up in the working directory only when it starts with `/`, `./`, `../` or `~`. A bare id such as `Qwen/Qwen3-Embedding-0.6B` could load a directory of that name from wherever the agent started the server. Models load with `trust_remote_code` off, and `puremind serve` changes into the pureMind home directory.
- A model that is not downloaded fails at once with one clear line when the Hub cannot be reached.

### MCP tools

- `puremind_get` serves exactly the notes the indexer reads: no hidden folders, `node_modules`, non-Markdown files or notes over 2 MiB, and it reads a bounded amount from disk. Long or malformed paths are refused at once. Names that begin with `~`, a space or a line break now open, a path is tried exactly as given before stray white space is removed, and harmless spellings such as `./notes/a.md` are accepted.
- Refusals are reported as tool errors (`isError`) with short messages. Those of `puremind_get` never contain absolute paths.
- `puremind_get` resolves paths against the vault the index was built from, and search and get say so when the configured vault differs.
- Parameter bounds are part of the tool schemas (search limit 1 to 50, recent days 1 to 3650 and limit 1 to 200), and the descriptions say what scores mean.
- `puremind_recent` leaves out notes dated in the future instead of letting them fill the list.
- The server reports pureMind's version to the client on mcp 1.x too.

### Command line

- `puremind init` writes TOML correctly for any vault path (an emoji in a path made every later command fail), keeps a configured `db` with `--force`, never copies environment overrides into the file, and creates `~/.puremind` as 0700 and config.toml as 0600. A config.toml that is a symlink keeps its target's mode. A malformed config.toml is reported with its path, and `init --force` repairs it.
- `puremind connect` prints an absolute, quoted command and carries the `PUREMIND_*` and Hugging Face cache variables set in your shell. The Cursor entry has `type: stdio`.
- Exit status is 0 on success, 1 when a search finds nothing, 2 on errors (an `error:` line, sometimes followed by a hint or cause line, and no traceback for expected failures) and 130 on Ctrl-C. `puremind serve` now stops on Ctrl-C.
- `index` and `watch` check the vault directory before opening the index, so a mistyped path no longer creates an empty index, and the error names the path.
- `puremind serve` no longer writes a library's debug lines to its log.
- `--limit` must be 1 to 50, an empty query is refused, and `search` without an index says so instead of creating one.
- Terminal control characters in note text and file names are removed before printing.
- Scores print with three significant digits, under a header that names the scale.
- `python -m puremind_mac` works.

### Index file

- The index and its `-wal` and `-shm` files are created readable only by you (0600, directory 0700), and existing ones you own are tightened when opened, as are an existing `~/.puremind` and config.toml.
- Deleted and edited note text is removed from the index file and its `-wal` file at the end of the index run that notices it, on every SQLite version: freed pages are zeroed, and the keyword index is compacted so it drops the words of deleted text. Text that 0.1.1 or earlier left behind is removed when 0.1.2 first opens the index. Emptying the `-wal` file waits at most a second for other readers instead of stalling every writer.
- An index on a read-only volume, or in a directory you cannot write, opens read-only, so `search` and `serve` still work.
- The index carries a format version, and an index from a newer release is refused instead of being changed.

### Packaging and CI

- Dependency floors are versions that were actually installed and tested: `sentence-transformers>=5.6.0` (CVE-2026-68770: a local model directory could run code), `torch>=2.6`, `transformers>=4.51.1`, `mcp>=1.21.1`, `pydantic>=2.11`, `watchdog>=5.0.3` (earlier releases can hang on exit after watching a directory they cannot read), and a non-yanked `bm25s`. Large libraries are capped below their next major release.
- The version has one source, `puremind_mac.__version__`.
- CI runs on macOS and Linux with Python 3.11 to 3.13 (and 3.14 on Linux), plus jobs for the mcp 1.x SDK, the lowest allowed dependencies, the built wheel, ruff, pip-audit and gitleaks, and a weekly smoke test with the real models. Actions are pinned to commit SHAs with a read-only token.
- The test suite is hermetic: it ignores `PUREMIND_*` variables in your shell, runs offline, and never touches your index or vault. Against an installed wheel, its server and command-line subprocesses run the wheel too.
- `bench/longmemeval.py` uses the current Hugging Face download command, adds LongMemEval's own metrics, records its settings, and resumes safely after a crash.

## 0.1.1 (2026-10-16)

- Support the mcp 2.x SDK (server class renamed to MCPServer) alongside 1.x; dependency bound `mcp>=1.2,<3`.
- Serialise SQLite access behind a re-entrant lock for tool handlers running on MCP worker threads. (This was incomplete; see 0.1.2.)
- Explicit stdio transport in `puremind serve`.
- Tests pass on both SDK versions.

## 0.1.0 (2026-10-16)

Initial release.

- Heading-aware Markdown chunker that keeps fenced code blocks and YAML frontmatter intact (long blocks were split without their fence lines until 0.1.2).
- SQLite store with an FTS5 lexical index and float16 vectors.
- Hybrid retrieval: BM25 and Qwen3-Embedding-0.6B fused with reciprocal rank fusion, re-scored by bge-reranker-v2-m3. Model names are configurable.
- Incremental indexing by SHA-256 per file, `puremind index` and a debounced `puremind watch`.
- Read-only MCP server over stdio with `puremind_search`, `puremind_get` and `puremind_recent`.
- CLI: `init`, `index`, `watch`, `search`, `serve`, `connect`.
- `bench/longmemeval.py` for the LongMemEval_s retrieval benchmark.
