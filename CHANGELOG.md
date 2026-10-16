# Changelog

## 0.1.0

Initial release.

- Heading-aware Markdown chunker that preserves fenced code blocks and YAML frontmatter.
- SQLite store with an FTS5 lexical index and float16 vectors.
- MAC-16b retrieval: BM25 and Qwen3-Embedding-0.6B fused with reciprocal rank fusion, re-scored by bge-reranker-v2-m3. Model names are configurable.
- Incremental indexing by SHA-256 per file, `puremind index` and a debounced `puremind watch`.
- Read-only MCP server over stdio with `puremind_search`, `puremind_get` and `puremind_recent`.
- CLI: `init`, `index`, `watch`, `search`, `serve`, `connect`.
- `bench/longmemeval.py` for the LongMemEval_s retrieval benchmark.
