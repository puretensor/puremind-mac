"""Handlers for the MCP tools: three that only read, and puremind_propose, which only leaves a proposal.

Each returns plain text. A request that cannot be served raises ToolRefused, whose message is
short and names no absolute path; the server turns it into a protocol error (isError=true).
"""

from __future__ import annotations

import logging
import os
import stat
import time
from datetime import datetime
from pathlib import Path

from . import inbox
from .indexer import (
    MARKDOWN_SUFFIXES,
    MAX_FILE_BYTES,
    MAX_PATH_CHARS,
    _excluded_part,
    _same_directory,
    is_indexable,
    open_resolved,
)
from .models import ModelLoadError
from .service import IndexMismatch, Service
from .store import Store

log = logging.getLogger("puremind.tools")

MAX_LIMIT = 50
MAX_GET_CHARS = 200_000
MAX_RECENT_LIMIT = 200
MAX_RECENT_DAYS = 3650
# A modification time further ahead than this is clock skew or a corrupt value, not a recent edit.
FUTURE_SLACK_SECONDS = 86_400


class ToolRefused(Exception):
    """The request cannot be served. The message is short and safe to show to a model."""


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def format_score(score: float) -> str:
    """Three significant digits, so small reranker probabilities stay distinguishable."""
    return f"{score:.3g}"


def score_scale(rerank: bool) -> str:
    """What the score of a search result means, for the tool output and descriptions."""
    if rerank:
        return "reranker probability from 0 to 1"
    return "reciprocal rank fusion value, about 0.01 to 0.03"


def _resolved(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError, ValueError):  # RuntimeError: symlink loop on 3.11 and 3.12
        return path.expanduser().absolute()


def vault_roots(service: Service) -> tuple[Path | None, Path | None]:
    """(vault the index was built from, vault the configuration names), each None when unknown."""
    stored = service.store.get_meta("vault_root")
    configured = service.cfg.vault
    return (_resolved(Path(stored)) if stored else None, _resolved(configured) if configured else None)


def vault_mismatch(service: Service) -> tuple[Path, Path] | None:
    """(indexed vault, configured vault) when they are different directories, else None."""
    indexed, configured = vault_roots(service)
    if indexed and configured and not _same_directory(str(indexed), configured):  # APFS ignores case
        return indexed, configured
    return None


_MISMATCH_NOTE = (
    "the configured vault is not the vault this index was built from; paths refer to the indexed vault "
    "(run `puremind index --rebuild` to index the configured one)"
)


def puremind_search(service: Service, query: str, limit: int = 10) -> str:
    query = (query or "").strip()
    if not query:
        raise ToolRefused("query is empty")
    limit = _clamp(limit, 1, MAX_LIMIT)
    rerank = service.cfg.rerank
    try:
        hits = service.search(query, limit=limit, rerank=rerank)
    except IndexMismatch as exc:
        raise ToolRefused(str(exc)) from None
    except ModelLoadError as exc:
        log.warning("%s: %s", exc, exc.__cause__)
        raise ToolRefused(str(exc)) from None
    if not hits:
        return f'No results for "{query}". The index may be empty; run `puremind index`.'
    head = (
        f'{len(hits)} result(s) for "{query}" '
        f"(score is a {score_scale(rerank)}; compare scores only within this list):"
    )
    if vault_mismatch(service):
        head += f"\nNote: {_MISMATCH_NOTE}."
    blocks = [head]
    for n, h in enumerate(hits, 1):
        blocks.append(
            f"[{n}] {h.path}\n"
            f"    heading: {h.heading_path or '(top of file)'}\n"
            f"    score: {format_score(h.score)}\n\n"
            f"{h.content}"
        )
    return "\n\n---\n\n".join(blocks)


def _relative_path(rel: str) -> str:
    """``rel`` if it passes the checks that need no file system access, else ToolRefused."""
    if not rel.strip():
        raise ToolRefused("give a path relative to the vault, as returned by puremind_search")
    if "\x00" in rel:
        raise ToolRefused("the path contains a NUL character")
    if len(rel) > MAX_PATH_CHARS:
        raise ToolRefused(f"the path is longer than {MAX_PATH_CHARS} characters")
    if rel.startswith("/") or os.path.isabs(rel):  # "~" is an ordinary character here: nothing expands it
        raise ToolRefused("give a path relative to the vault, as returned by puremind_search")
    parts = [part for part in rel.split("/") if part not in ("", ".")]  # "./a.md" and "a//b.md" are fine
    if any(_excluded_part(part) for part in parts):  # hidden names, "..", node_modules, ...
        raise ToolRefused("hidden files and folders, '..' and skipped folders are not part of the vault")
    if not parts or not parts[-1].lower().endswith(MARKDOWN_SUFFIXES):
        raise ToolRefused("only Markdown notes (.md and .markdown files) are served")
    return rel


def _spellings(path: str) -> list[str]:
    """The ways to read ``path`` that pass the checks, most literal first.

    A note's own name may begin with spaces or even a line break, so the path is tried exactly
    as given, then without line breaks around it (a client may add one), then without any
    surrounding white space. Each spelling is checked on its own. If none passes, the refusal
    for the path as given is raised.
    """
    path = path or ""
    found: list[str] = []
    first: ToolRefused | None = None
    for rel in dict.fromkeys((path, path.strip("\r\n"), path.strip())):
        try:
            found.append(_relative_path(rel))
        except ToolRefused as exc:
            first = first or exc
    if not found:
        raise first or ToolRefused("give a path relative to the vault, as returned by puremind_search")
    return found


def _read_note(path: Path) -> tuple[str, bool]:
    """Up to 4 * MAX_GET_CHARS bytes of the note as text, and whether the file continued beyond them."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(open_resolved(path, flags), "rb") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise ToolRefused("not a regular Markdown note")
        if st.st_size > MAX_FILE_BYTES:
            raise ToolRefused(
                f"the note is over {MAX_FILE_BYTES // (1024 * 1024)} MiB; "
                "pureMind neither indexes nor serves files that large"
            )
        os.set_blocking(fh.fileno(), True)  # as the indexer does: a cloud placeholder may need to download
        limit = 4 * MAX_GET_CHARS  # UTF-8 uses at most four bytes per character
        data = fh.read(limit + 1)
    return data[:limit].decode("utf-8", errors="replace"), len(data) > limit


def puremind_get(service: Service, path: str) -> str:
    indexed, configured = vault_roots(service)
    root = indexed or configured
    if root is None:
        raise ToolRefused("no vault is configured; run `puremind init <vault_dir>`")
    spellings = _spellings(path)
    try:
        if not root.is_dir():
            if indexed:
                raise ToolRefused(
                    "the vault directory this index was built from no longer exists; if it moved, "
                    "run `puremind init <new_dir> --force` and then `puremind index --rebuild`"
                )
            raise ToolRefused("the configured vault directory does not exist; run `puremind init <vault_dir>`")
        real = next((r for rel in spellings if (r := is_indexable(root, root / rel)) is not None), None)
        if real is None:
            raise ToolRefused(
                "no such note in the vault; give a path relative to the vault, as returned by puremind_search"
            )
        text, more = _read_note(real)
        shown = real.relative_to(root).as_posix()
    except (OSError, ValueError, RuntimeError):  # unreadable, vanished, or a path the system rejects
        raise ToolRefused("the note cannot be read") from None
    note = ""
    if more or len(text) > MAX_GET_CHARS:
        text, note = text[:MAX_GET_CHARS], f"\n\n[truncated at {MAX_GET_CHARS} characters]"
    header = f"path: {shown}"
    if vault_mismatch(service):
        header += f"\nnote: {_MISMATCH_NOTE}"
    return f"{header}\n\n{text}{note}"


def _when(mtime: float) -> str:
    try:
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return "unknown"


def _recent_rows(store: Store, since: float, horizon: float, limit: int):
    """(up to ``limit`` rows newest first, count of rows dated after ``horizon``).

    Rows dated in the future sort first, so the query is repeated with a larger limit until
    enough present-day rows have been seen or the table is exhausted.
    """
    fetch = limit
    while True:
        rows = store.recent(since, fetch)
        usable = [r for r in rows if r.mtime <= horizon]
        if len(usable) >= limit or len(rows) < fetch:
            return usable[:limit], len(rows) - len(usable)
        fetch *= 4


def puremind_recent(service: Service, days: int = 7, limit: int = 20) -> str:
    days = _clamp(days, 1, MAX_RECENT_DAYS)
    limit = _clamp(limit, 1, MAX_RECENT_LIMIT)
    now = time.time()
    rows, future = _recent_rows(service.store, now - days * 86400, now + FUTURE_SLACK_SECONDS, limit)
    skipped = f"\n({future} note(s) dated in the future were left out.)" if future else ""
    if not rows:
        return f"No indexed notes were modified in the last {days} day(s)." + skipped
    lines = [f"{len(rows)} note(s) modified in the last {days} day(s), newest first:"]
    for r in rows:
        lines.append(f"- {r.path}  ({_when(r.mtime)})  {r.title}")
    return "\n".join(lines) + skipped


def puremind_propose(service: Service, title: str, content: str) -> str:
    """Leave a proposed memory in the inbox. It changes nothing the user has until they approve it."""
    try:
        proposal, new = inbox.propose(service.cfg.home, title, content)
    except inbox.InboxError as exc:
        raise ToolRefused(str(exc)) from None
    except OSError:
        log.warning("puremind_propose could not save the proposal", exc_info=True)
        raise ToolRefused("the proposal could not be saved") from None
    state = "saved" if new else "was already waiting"
    return (
        f"Proposal {proposal.id} {state}. It is not in the user's notes: the user has to review and approve it "
        "with `puremind inbox`, and may reject it. Do not treat it as remembered, and do not propose it again."
    )
