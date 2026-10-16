"""Heading-aware Markdown chunker.

Sections are split on ATX headings (levels 1 to 4) and carry a breadcrumb
("Parent > Child"). Headings inside fenced code blocks and inside a leading YAML
frontmatter block never split a section. Tiny sections merge into their previous
sibling. Sections longer than ``max_chars`` are cut into overlapping windows that
prefer to end on a line boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHARS = 2048
MIN_CHARS = 100
OVERLAP = 0.2

_HEADING = re.compile(r"^(#{1,4})\s+(.+)$")


@dataclass(frozen=True)
class Chunk:
    index: int
    heading_path: str
    content: str


def _fence(line: str) -> tuple[str, int, str] | None:
    """Parse a CommonMark fence line into (char, run length, info string)."""
    raw = line.rstrip("\r\n")
    indent = len(raw) - len(raw.lstrip(" "))
    if indent > 3:
        return None
    body = raw[indent:]
    if not body.startswith(("```", "~~~")):
        return None
    char = body[0]
    run = len(body) - len(body.lstrip(char))
    info = body[run:].strip()
    if char == "`" and "`" in info:
        return None
    return char, run, info


def fenced_lines(lines: list[str], start: int = 0) -> set[int]:
    """Indexes of lines inside a closed fenced block, fence lines included.

    An opener without a matching closer is ordinary text, so a stray fence cannot
    swallow the rest of a note. Lines before ``start`` (frontmatter) are ignored.
    """
    n = len(lines)
    infos = [_fence(line) if i >= start else None for i, line in enumerate(lines)]
    # longest closing-capable fence at or after each line, per fence character
    best = {"`": [0] * (n + 1), "~": [0] * (n + 1)}
    for i in range(n - 1, -1, -1):
        for c in best:
            best[c][i] = best[c][i + 1]
        info = infos[i]
        if info and info[2] == "":
            best[info[0]][i] = max(best[info[0]][i], info[1])
    inside: set[int] = set()
    i = 0
    while i < n:
        info = infos[i]
        if info is None or best[info[0]][i + 1] < info[1]:
            i += 1
            continue
        char, run, _ = info
        j = i + 1
        while j < n:
            cand = infos[j]
            if cand and cand[0] == char and cand[1] >= run and cand[2] == "":
                break
            j += 1
        inside.update(range(i, j + 1))
        i = j + 1
    return inside


def frontmatter_end(lines: list[str]) -> int | None:
    """Line index of the closing delimiter of a leading YAML block, if any."""
    if not lines or lines[0].lstrip("﻿").rstrip() != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i].rstrip() in ("---", "..."):
            return i
    return None


def _sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines(keepends=True)
    fm = frontmatter_end(lines)
    protected = fenced_lines(lines, 0 if fm is None else fm + 1)
    stack: dict[int, str] = {}
    path = ""
    buf: list[str] = []
    out: list[tuple[str, str]] = []

    def flush() -> None:
        body = "".join(buf).strip()
        if body:
            out.append((path, body))

    for i, line in enumerate(lines):
        in_fm = fm is not None and i <= fm
        m = None if (in_fm or i in protected) else _HEADING.match(line.rstrip())
        if m is None:
            buf.append(line)
            continue
        flush()
        level = len(m.group(1))
        stack[level] = m.group(2).strip()
        for deeper in [k for k in stack if k > level]:
            del stack[deeper]
        path = " > ".join(stack[k] for k in sorted(stack))
        buf = [line]
    flush()
    return out


def _parent(path: str) -> str:
    return path.rsplit(" > ", 1)[0] if " > " in path else ""


def _depth(path: str) -> int:
    return path.count(" > ") + 1 if path else 0


def _merge_small(sections: list[tuple[str, str]], min_chars: int) -> list[tuple[str, list[str]]]:
    merged: list[tuple[str, list[str]]] = []
    for path, body in sections:
        if merged and len(body) < min_chars and _parent(path) == _parent(merged[-1][0]):
            prev_path, parts = merged[-1]
            parts.append(body)
            if _depth(path) > _depth(prev_path):
                merged[-1] = (path, parts)
        else:
            merged.append((path, [body]))
    return merged


def _windows(text: str, max_chars: int, overlap: float) -> list[str]:
    ov = int(max_chars * overlap)
    step = max_chars - ov
    out: list[str] = []
    pos = 0
    while pos < len(text):
        end = pos + max_chars
        if end < len(text):
            cut = text.rfind("\n", pos + step, end)
            if cut != -1:
                end = cut + 1
        piece = text[pos:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        pos = end - ov
    return out


def chunk_markdown(
    text: str,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    overlap: float = OVERLAP,
) -> list[Chunk]:
    """Split Markdown into heading-aware chunks of at most ``max_chars``."""
    sections = _sections(text)
    if len(sections) > 1 and not sections[0][0] and len(sections[0][1]) < min_chars:
        # a tiny preamble (usually frontmatter) travels with the first real section
        (_, pre), (path, body) = sections[0], sections[1]
        sections = [(path, f"{pre}\n\n{body}")] + sections[2:]
    chunks: list[Chunk] = []
    for path, parts in _merge_small(sections, min_chars):
        body = "\n\n".join(parts)
        pieces = _windows(body, max_chars, overlap) if len(body) > max_chars else [body]
        for piece in pieces:
            chunks.append(Chunk(len(chunks), path, piece))
    return chunks
