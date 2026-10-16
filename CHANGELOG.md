# Changelog

## 0.1.1 (2026-10-16)

- Support the mcp 2.x SDK (server class renamed to MCPServer) alongside 1.x; dependency bound `mcp>=1.2,<3`.
- Serialise SQLite access behind a re-entrant lock so tool handlers running on MCP worker threads and the file watcher share one connection safely.
- Explicit stdio transport in `puremind serve`.
- Tests pass on both SDK versions.

## 0.1.0

Initial release.

- Heading-aware Markdown chunker that preserves fenced code blocks and YAML frontmatter.
- SQLite store with an FTS5 lexical index and float16 vectors.
- MAC-16b retrieval: BM25 and Qwen3-Embedding-0.6B fused with reciprocal rank fusion, re-scored by bge-reranker-v2-m3. Model names are configurable.
- Incremental indexing by SHA-256 per file, `puremind index` and a debounced `puremind watch`.
- Read-only MCP server over stdio with `puremind_search`, `puremind_get` and `puremind_recent`.
- CLI: `init`, `index`, `watch`, `search`, `serve`, `connect`.
- `bench/longmemeval.py` for the LongMemEval_s retrieval benchmark.
