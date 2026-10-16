"""Handlers for the three read-only MCP tools. Each returns plain text."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from .service import IndexMismatch, Service

MAX_LIMIT = 50
MAX_GET_CHARS = 200_000
READABLE = (".md", ".markdown")


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def puremind_search(service: Service, query: str, limit: int = 10) -> str:
    query = (query or "").strip()
    if not query:
        return "Error: query is empty."
    limit = _clamp(limit, 1, MAX_LIMIT)
    try:
        hits = service.search(query, limit=limit)
    except IndexMismatch as exc:
        return f"Error: {exc}."
    if not hits:
        return f'No results for "{query}". The index may be empty; run `puremind index`.'
    blocks = [f'{len(hits)} result(s) for "{query}":']
    for n, h in enumerate(hits, 1):
        blocks.append(
            f"[{n}] {h.path}\n"
            f"    heading: {h.heading_path or '(top of file)'}\n"
            f"    score: {h.score:.4f}\n\n"
            f"{h.content}"
        )
    return "\n\n---\n\n".join(blocks)


def puremind_get(service: Service, path: str) -> str:
    root = service.vault_root
    if root is None:
        return "Error: no vault is configured. Run `puremind init <vault_dir>`."
    rel = (path or "").strip()
    if not rel or rel.startswith(("/", "~")):
        return "Error: give a path relative to the vault, as returned by puremind_search."
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return "Error: path is outside the vault."
    if target.suffix.lower() not in READABLE or not target.is_file():
        return f"Error: no Markdown file at {rel}."
    text = target.read_text(encoding="utf-8", errors="replace")
    note = ""
    if len(text) > MAX_GET_CHARS:
        text, note = text[:MAX_GET_CHARS], f"\n\n[truncated at {MAX_GET_CHARS} characters]"
    return f"path: {target.relative_to(root).as_posix()}\n\n{text}{note}"


def puremind_recent(service: Service, days: int = 7, limit: int = 20) -> str:
    days = _clamp(days, 1, 3650)
    limit = _clamp(limit, 1, 200)
    rows = service.store.recent(time.time() - days * 86400, limit)
    if not rows:
        return f"No indexed notes were modified in the last {days} day(s)."
    lines = [f"{len(rows)} note(s) modified in the last {days} day(s), newest first:"]
    for r in rows:
        when = datetime.fromtimestamp(r.mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"- {r.path}  ({when})  {r.title}")
    return "\n".join(lines)
