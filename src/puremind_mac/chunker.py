"""Heading-aware Markdown chunker.

Sections are split on ATX headings (levels 1 to 4) and carry a breadcrumb
("Parent > Child"). Headings inside fenced code blocks and inside a leading YAML
frontmatter block never start a section. A heading with no body of its own travels
with the child section that follows it, and tiny sections merge into their previous
sibling, or into a previous chunk that is itself tiny. Sections longer than
``max_chars`` are cut into near-even windows with about 20% overlap. A window ends
at a blank line or a line break outside fenced blocks and frontmatter when it can.
A block longer than a window is cut anyway, and each part then carries its own
opening and closing fence line.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from dataclasses import dataclass

VERSION = 1  # bumped whenever chunk boundaries change, so the indexer re-chunks every note once
MAX_CHARS = 2048
MIN_CHARS = 100
OVERLAP = 0.2
MAX_HEADING_CHARS = 200

_HEADING = re.compile(r"^(#{1,4})\s+(.+)$")
_BOM = "\ufeff"


@dataclass(frozen=True)
class Chunk:
    index: int
    heading_path: str
    content: str


def clip_heading(text: str, limit: int = MAX_HEADING_CHARS) -> str:
    """Heading text as used in breadcrumbs and titles; a longer line is cut with an ellipsis."""
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


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


def fenced_blocks(lines: list[str], start: int = 0) -> list[tuple[int, int]]:
    """(opening, closing) line indexes of every closed fenced block, fence lines included.

    An opener without a matching closer is ordinary text, so a stray fence cannot
    swallow the rest of a note. Lines before ``start`` (frontmatter) are ignored.
    """
    n = len(lines)
    infos = [_fence(line) if i >= start else None for i, line in enumerate(lines)]
    # longest closing-capable fence at or after each line, per fence character
    best = {"`": [0] * (n + 1), "~": [0] * (n + 1)}
    for i in range(n - 1, -1, -1):
        for column in best.values():
            column[i] = column[i + 1]
        info = infos[i]
        if info and info[2] == "":
            best[info[0]][i] = max(best[info[0]][i], info[1])
    blocks: list[tuple[int, int]] = []
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
        blocks.append((i, j))
        i = j + 1
    return blocks


def fenced_lines(lines: list[str], start: int = 0) -> set[int]:
    """Indexes of lines inside a closed fenced block, fence lines included."""
    inside: set[int] = set()
    for first, last in fenced_blocks(lines, start):
        inside.update(range(first, last + 1))
    return inside


def frontmatter_end(lines: list[str]) -> int | None:
    """Line index of the closing delimiter of a leading YAML block, if any."""
    if not lines or lines[0].lstrip(_BOM).rstrip() != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i].rstrip() in ("---", "..."):
            return i
    return None


@dataclass
class _Section:
    path: str
    level: int  # heading level, 0 for the text before the first heading
    body: str
    bare: bool  # nothing but the heading line


def _sections(text: str) -> list[_Section]:
    lines = text.removeprefix(_BOM).splitlines(keepends=True)
    fm = frontmatter_end(lines)
    protected = fenced_lines(lines, 0 if fm is None else fm + 1)
    stack: dict[int, str] = {}
    path, level = "", 0
    buf: list[str] = []
    out: list[_Section] = []

    def flush() -> None:
        body = "".join(buf).strip()
        if body:
            out.append(_Section(path, level, body, level > 0 and not any(x.strip() for x in buf[1:])))

    for i, line in enumerate(lines):
        in_fm = fm is not None and i <= fm
        m = None if (in_fm or i in protected) else _HEADING.match(line.rstrip())
        if m is None:
            buf.append(line)
            continue
        flush()
        level = len(m.group(1))
        stack[level] = clip_heading(m.group(2).strip())
        for deeper in [k for k in stack if k > level]:
            del stack[deeper]
        path = " > ".join(stack[k] for k in sorted(stack))
        buf = [line]
    flush()
    return out


def _fold_bare(sections: list[_Section]) -> list[_Section]:
    """Fold a heading that has no body into the deeper section that follows it."""
    out: list[_Section] = []
    carry = ""
    for i, sec in enumerate(sections):
        following = sections[i + 1] if i + 1 < len(sections) else None
        if sec.bare and following is not None and following.level > sec.level:
            carry = f"{carry}\n\n{sec.body}" if carry else sec.body
            continue
        if carry:
            sec = _Section(sec.path, sec.level, f"{carry}\n\n{sec.body}", sec.bare)
            carry = ""
        out.append(sec)
    return out


def _parent(path: str) -> str:
    return path.rsplit(" > ", 1)[0] if " > " in path else ""


def _crumbs(path: str) -> list[str]:
    return path.split(" > ") if path else []


def _inside(outer: str, inner: str) -> bool:
    """True if the section ``inner`` lies inside the section ``outer`` (or is it)."""
    a, b = _crumbs(outer), _crumbs(inner)
    return b[: len(a)] == a


def _shared(a: str, b: str) -> str:
    """The headings two breadcrumbs have in common."""
    pa, pb = _crumbs(a), _crumbs(b)
    n = 0
    while n < min(len(pa), len(pb)) and pa[n] == pb[n]:
        n += 1
    return " > ".join(pa[:n])


def _merge_small(sections: list[_Section], min_chars: int) -> list[tuple[str, list[str]]]:
    """Merge tiny sections into the previous sibling, or into a previous chunk that is tiny too.

    The second case bounds the chunk count of outlines made of one-word headings,
    which share no parent and would otherwise become one chunk each. Such a chunk
    carries the headings its sections share, not those of the first one alone.
    """
    merged: list[tuple[str, list[str]]] = []
    size = 0  # characters in the last group
    chain = True  # each section of the last group lies inside the one before, so the deepest path names them all
    for sec in sections:
        sibling = bool(merged) and _parent(sec.path) == _parent(merged[-1][0])
        if merged and len(sec.body) < min_chars and (sibling or size < min_chars):
            prev_path, parts = merged[-1]
            parts.append(sec.body)
            size += len(sec.body) + 2
            if sibling:
                chain = False  # the group keeps its first section's path, as siblings always have
            elif chain and _inside(prev_path, sec.path):
                merged[-1] = (sec.path, parts)
            else:
                merged[-1] = (_shared(prev_path, sec.path), parts)
                chain = False
        else:
            merged.append((sec.path, [sec.body]))
            size = len(sec.body)
            chain = True
    return merged


class _Layout:
    """Line starts and protected blocks (fences, frontmatter) of one oversized body."""

    def __init__(self, text: str, max_chars: int):
        self.text = text
        lines = text.splitlines(keepends=True)
        self.starts = [0]
        for line in lines:
            self.starts.append(self.starts[-1] + len(line))
        self.blank = [not line.strip() for line in lines]
        fm = frontmatter_end(lines)
        spans = [] if fm is None else [(0, fm)]
        spans += fenced_blocks(lines, 0 if fm is None else fm + 1)
        self.block_of = [-1] * len(lines)
        self.first: list[int] = []
        self.last: list[int] = []
        self.opener: list[str] = []  # "" when the block is too wide to repeat the fences
        self.closer: list[str] = []
        for b, (first, last) in enumerate(spans):
            opener, closer = lines[first].strip("\r\n"), lines[last].strip()
            if len(opener) + len(closer) + 2 > max_chars // 4:
                opener = closer = ""
            self.first.append(first)
            self.last.append(last)
            self.opener.append(opener)
            self.closer.append(closer)
            for i in range(first, last + 1):
                self.block_of[i] = b
        self.max_reserve = max((len(c) + 1 for c in self.closer if c), default=0)

    def _line(self, offset: int) -> int:
        return bisect_right(self.starts, offset) - 1

    def ends_in(self, end: int) -> int:
        """The block a window ending at ``end`` leaves open, or -1."""
        i = self._line(end - 1)
        b = self.block_of[i] if i < len(self.block_of) else -1
        return b if b >= 0 and i < self.last[b] else -1

    def starts_in(self, start: int) -> int:
        """The block a window starting at ``start`` begins inside, or -1."""
        i = self._line(start)
        b = self.block_of[i] if i < len(self.block_of) else -1
        return b if b >= 0 and i > self.first[b] else -1

    def breaks(self, lo: int, hi: int, t: int, begin: int, room: int) -> dict[int, int]:
        """Best (nearest to ``t``) line boundary per rank within [lo, hi].

        Rank 0 is a blank line and rank 1 any line break, both outside blocks; ranks 2
        and 3 are the same inside a block, where the closing fence must still fit.
        """
        best: dict[int, int] = {}
        j = max(1, bisect_left(self.starts, lo))
        while j < len(self.blank) and self.starts[j] <= hi:
            off = self.starts[j]
            b = self.block_of[j]
            inside = b >= 0 and self.block_of[j - 1] == b
            rank = (0 if self.blank[j - 1] or self.blank[j] else 1) + (2 if inside else 0)
            fits = not inside or off - begin + len(self.closer[b]) + (1 if self.closer[b] else 0) <= room
            if fits and (rank not in best or abs(off - t) <= abs(best[rank] - t)):
                best[rank] = off
            j += 1
        return best

    def pick_end(self, begin: int, t: int, room: int, ov: int, slack: int) -> int:
        """Where the window starting at ``begin`` ends, aiming for ``t``."""
        limit = min(len(self.text), begin + room)
        wide_lo = begin + max(ov + 1, (t - begin) // 2)
        near = self.breaks(max(wide_lo, t - slack), min(limit, t + slack), t, begin, room)
        for rank in (0, 1):
            if rank in near:
                return near[rank]
        wide = self.breaks(wide_lo, limit, t, begin, room)
        for rank in (0, 1, 2, 3):  # ranks 2 and 3: every block in reach is longer than a window
            if rank in wide:
                return wide[rank]
        up = limit - self.max_reserve
        if up >= wide_lo:
            for off in _spiral(wide_lo, up, t):
                if self.text[off].isspace():
                    return off
        return max(wide_lo, min(t, up))

    def pick_start(self, p: int, end: int, ov: int) -> int:
        """First place at or after ``p``, within half the overlap, where a line or a word begins.

        A start inside a block that the previous window ended after is moved past that
        block, which is already whole in the previous window.
        """
        text, hi = self.text, min(end - 1, p + ov // 2)
        j = bisect_left(self.starts, p)
        if j < len(self.blank) and self.starts[j] <= hi:
            start = self.starts[j]
        else:
            q = p
            if q > 0 and not text[q - 1].isspace():
                while q <= hi and not text[q].isspace():
                    q += 1
            while q <= hi and text[q].isspace():
                q += 1
            start = q if q <= hi else p
        i = self._line(start)
        b = self.block_of[i] if i < len(self.block_of) else -1
        if b >= 0 and self.closer[b] and i > self.first[b] and self.ends_in(end) != b:
            after = self.starts[self.last[b] + 1]
            if after <= end:
                start = after
        return start


def _spiral(lo: int, hi: int, centre: int) -> Iterator[int]:
    """Offsets of [lo, hi] by increasing distance from ``centre``."""
    centre = max(lo, min(hi, centre))
    yield centre
    for d in range(1, hi - lo + 1):
        if centre + d <= hi:
            yield centre + d
        if centre - d >= lo:
            yield centre - d


def _trim(piece: str) -> str:
    """Drop leading blank lines and trailing whitespace, keeping the first line's indentation."""
    body = piece.rstrip()
    text = body.lstrip()
    return body[: len(body) - len(text)].rsplit("\n", 1)[-1] + text


def _windows(text: str, max_chars: int, overlap: float) -> list[str]:
    """Cut ``text`` into near-even windows of at most ``max_chars`` that overlap by ``overlap``."""
    n = len(text)
    ov = min(int(max_chars * overlap), max_chars // 3)
    slack = max_chars // 16
    lay = _Layout(text, max_chars)
    out: list[str] = []
    start = 0
    while True:
        b = lay.starts_in(start)
        head = lay.opener[b] + "\n" if b >= 0 and lay.opener[b] else ""
        room = max_chars - len(head)
        if n - start <= room:
            end = n
        else:
            cap = room - slack
            windows_left = max(2, -(-(n - start - ov) // (cap - ov)))
            size = -(-(n - start + (windows_left - 1) * ov) // windows_left)
            end = lay.pick_end(start, start + size, room, ov, slack)
        core = _trim(text[start:end])
        e = lay.ends_in(end) if end < n else -1
        tail = "\n" + lay.closer[e] if e >= 0 and lay.closer[e] else ""
        if core.strip():
            out.append(head + core + tail)
        if end >= n:
            return out
        nxt = lay.pick_start(end - ov, end, ov)
        start = nxt if nxt > start else end


def chunk_markdown(
    text: str,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    overlap: float = OVERLAP,
) -> list[Chunk]:
    """Split Markdown into heading-aware chunks of at most ``max_chars``."""
    sections = _fold_bare(_sections(text))
    if len(sections) > 1 and not sections[0].path and len(sections[0].body) < min_chars:
        # a tiny preamble (usually frontmatter) travels with the first real section
        pre, first = sections[0], sections[1]
        sections = [_Section(first.path, first.level, f"{pre.body}\n\n{first.body}", first.bare)] + sections[2:]
    chunks: list[Chunk] = []
    for path, parts in _merge_small(sections, min_chars):
        body = "\n\n".join(parts)
        pieces = _windows(body, max_chars, overlap) if len(body) > max_chars else [body]
        for piece in pieces:
            chunks.append(Chunk(len(chunks), path, piece))
    return chunks
