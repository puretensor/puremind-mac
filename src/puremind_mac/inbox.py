"""The inbox: how an agent proposes a memory and how you approve it.

An agent can leave a proposal here and nothing else. A proposal is a small JSON file in
``<home>/inbox/``, outside the vault, so it is neither indexed nor served by ``puremind_get``
until you approve it. ``puremind inbox approve`` is the only code that writes a note, and it only
creates: it never overwrites or edits an existing file, and it refuses to follow a symlink.
"""

from __future__ import annotations

import errno
import json
import os
import re
import secrets
import stat
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .indexer import MARKDOWN_SUFFIXES, MAX_PATH_CHARS, _excluded_part

INBOX_DIRNAME = "inbox"
DEFAULT_FOLDER = "agent-memories"
MAX_TITLE_CHARS = 200
MAX_CONTENT_CHARS = 20_000
MAX_PENDING = 100
SLUG_CHARS = 60
# Longest proposal file read back: the content cap at four bytes per character, plus the JSON around it.
_MAX_PROPOSAL_BYTES = 6 * MAX_CONTENT_CHARS + 4096
_ID = re.compile(r"^[0-9]{8}-[0-9a-f]{6}$")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f  ]")  # all but tab and newline


class InboxError(Exception):
    """The request cannot be served. The message is short, safe to show to a model, and names no absolute path."""


@dataclass(frozen=True)
class Proposal:
    id: str
    created: str  # ISO 8601, UTC
    title: str
    content: str


def inbox_dir(home: Path) -> Path:
    return home / INBOX_DIRNAME


def _private_dir(path: Path, create: bool) -> None:
    """The inbox directory exists as a real directory (never a symlink), readable only by this user."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        if not create:
            return
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise InboxError("the inbox is not a directory")
    if st.st_uid == os.geteuid() and st.st_mode & 0o077:
        os.chmod(path, stat.S_IMODE(st.st_mode) & ~0o077)


def _check_text(title: str, content: str) -> tuple[str, str]:
    title = _CONTROL.sub(" ", title or "")
    title = " ".join(title.split())
    content = (content or "").replace("\r\n", "\n").replace("\r", "\n")
    if not title:
        raise InboxError("give the memory a short title")
    if not content.strip():
        raise InboxError("the memory has no content")
    if len(title) > MAX_TITLE_CHARS:
        raise InboxError(f"the title is longer than {MAX_TITLE_CHARS} characters")
    if len(content) > MAX_CONTENT_CHARS:
        raise InboxError(f"the content is longer than {MAX_CONTENT_CHARS:,} characters; propose a shorter note")
    if "\x00" in content:
        raise InboxError("the content contains a NUL character")
    for text in (title, content):
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            raise InboxError("the text is not valid Unicode") from None
    return title, content.strip("\n")


def _read_proposal(path: Path) -> Proposal:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            raise ValueError("not a regular file")
        data = fh.read(_MAX_PROPOSAL_BYTES + 1)
    if len(data) > _MAX_PROPOSAL_BYTES:
        raise ValueError("too large")
    raw = json.loads(data.decode("utf-8"))
    if not isinstance(raw, dict) or not all(isinstance(raw.get(k), str) for k in ("id", "created", "title", "content")):
        raise ValueError("unexpected fields")
    if raw["id"] != path.stem or not _ID.match(raw["id"]):
        raise ValueError("the id does not match the file name")
    return Proposal(raw["id"], raw["created"], raw["title"], raw["content"])


def list_proposals(home: Path) -> tuple[list[Proposal], list[str]]:
    """(pending proposals oldest first, names of files in the inbox that could not be read)."""
    root = inbox_dir(home)
    found: list[Proposal] = []
    unreadable: list[str] = []
    try:
        names = sorted(os.listdir(root))
    except FileNotFoundError:
        return [], []
    for name in names:
        if name.startswith(".") or not name.endswith(".json"):
            continue  # a temporary file of a write in progress, or something that is not ours
        try:
            found.append(_read_proposal(root / name))
        except (OSError, ValueError):  # json.JSONDecodeError and UnicodeDecodeError are ValueErrors
            unreadable.append(name)
    found.sort(key=lambda p: (p.created, p.id))
    return found, unreadable


def propose(home: Path, title: str, content: str) -> tuple[Proposal, bool]:
    """Save a proposal and return it, with False when an identical one was already waiting."""
    title, content = _check_text(title, content)
    root = inbox_dir(home)
    _private_dir(home, create=True)
    _private_dir(root, create=True)
    pending, unreadable = list_proposals(home)
    for p in pending:
        if p.title == title and p.content == content:
            return p, False
    if len(pending) + len(unreadable) >= MAX_PENDING:
        raise InboxError(
            f"the inbox already holds {MAX_PENDING} proposals; ask the user to review them with `puremind inbox`"
        )
    now = datetime.now(UTC)
    for _ in range(8):
        pid = f"{now:%Y%m%d}-{secrets.token_hex(3)}"
        proposal = Proposal(pid, now.strftime("%Y-%m-%dT%H:%M:%SZ"), title, content)
        try:
            fd = os.open(root / f"{pid}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except FileExistsError:
            continue
        # Written in place: a reader that meets the half-written file counts it as unreadable, and
        # the writer is a short-lived server process that holds the whole text already.
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(proposal.__dict__, fh)
        except BaseException:
            try:
                os.unlink(root / f"{pid}.json")
            except OSError:
                pass
            raise
        return proposal, True
    raise InboxError("could not save the proposal; try again")


def find(home: Path, ref: str) -> Proposal:
    """The pending proposal whose id is ``ref`` or begins with it. Raises InboxError if there is not exactly one."""
    ref = (ref or "").strip()
    if not ref:
        raise InboxError("give a proposal id; `puremind inbox` lists them")
    pending, _ = list_proposals(home)
    matches = [p for p in pending if p.id == ref] or [p for p in pending if p.id.startswith(ref)]
    if not matches:
        raise InboxError(f"no pending proposal matches {ref[:40]!r}; `puremind inbox` lists them")
    if len(matches) > 1:
        raise InboxError(f"{ref[:40]!r} matches {len(matches)} proposals; give more of the id")
    return matches[0]


def discard(home: Path, proposal: Proposal) -> None:
    """Remove the proposal file. One already gone (approved from another terminal) is not an error."""
    try:
        os.unlink(inbox_dir(home) / f"{proposal.id}.json")
    except FileNotFoundError:
        pass


def slugify(title: str) -> str:
    slug = re.sub(r"[\W_]+", "-", title.casefold()).strip("-")[:SLUG_CHARS].strip("-")
    return slug or "note"


def note_text(proposal: Proposal) -> str:
    return f"# {proposal.title}\n\n{proposal.content}\n"


def check_folder(folder: str) -> str:
    """``folder`` as a vault-relative directory name the indexer would read, else InboxError."""
    parts = [p for p in (folder or "").replace("\\", "/").split("/") if p not in ("", ".")]
    if (
        not parts
        or os.path.isabs(folder)
        or "\x00" in folder
        or any(_excluded_part(p) or _CONTROL.search(p) for p in parts)
    ):
        raise InboxError(f"{folder!r} is not a folder name inside the vault (no '..', hidden names or absolute paths)")
    joined = "/".join(parts)
    if len(joined) > MAX_PATH_CHARS - 200:
        raise InboxError("the folder name is too long")
    return joined


def check_note_path(rel: str) -> str:
    """``rel`` as a vault-relative Markdown note path the indexer would read, else InboxError."""
    rel = rel or ""
    parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or os.path.isabs(rel) or "\x00" in rel or any(_excluded_part(p) or _CONTROL.search(p) for p in parts):
        raise InboxError(f"{rel!r} is not a note path inside the vault (no '..', hidden names or absolute paths)")
    if not parts[-1].lower().endswith(MARKDOWN_SUFFIXES):
        raise InboxError("a note path ends in .md or .markdown")
    joined = "/".join(parts)
    if len(joined) > MAX_PATH_CHARS:
        raise InboxError(f"the path is longer than {MAX_PATH_CHARS} characters")
    return joined


def default_path(proposal: Proposal, folder: str = DEFAULT_FOLDER) -> str:
    return f"{check_folder(folder)}/{proposal.created[:10]}-{slugify(proposal.title)}.md"


def _create_exclusive(root: Path, rel: str, data: bytes) -> None:
    """Create the note ``rel`` below ``root``, and fail with FileExistsError if it exists.

    The walk goes one directory at a time through descriptors with O_NOFOLLOW, so a symlink at
    any level of ``rel`` (the vault itself is not checked: it is yours) is refused rather than
    followed out of the vault. Folders that are missing are made.
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    dir_flags = os.O_RDONLY | nofollow | getattr(os, "O_DIRECTORY", 0)
    *folders, name = rel.split("/")
    fd = os.open(root.resolve(), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        for part in folders:
            try:
                nxt = os.open(part, dir_flags, dir_fd=fd)
            except FileNotFoundError:
                os.mkdir(part, 0o777, dir_fd=fd)
                nxt = os.open(part, dir_flags, dir_fd=fd)
            os.close(fd)
            fd = nxt
        out = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o666, dir_fd=fd)
        try:
            with os.fdopen(out, "wb") as fh:
                fh.write(data)
        except BaseException:
            try:
                os.unlink(name, dir_fd=fd)
            except OSError:
                pass
            raise
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise InboxError("part of that path is a symlink or a file, not a folder; pureMind will not write through it") from None
        raise
    finally:
        os.close(fd)


def approve(home: Path, vault: Path, proposal: Proposal, path: str | None = None, folder: str = DEFAULT_FOLDER) -> str:
    """Write the proposal into the vault as a new note and drop it from the inbox. Returns the vault-relative path.

    With ``path`` the note goes exactly there, and an existing note of that name is an error. Without
    it, the name comes from the date and title, and a name taken gets -2, -3 and so on.
    """
    data = note_text(proposal).encode("utf-8")
    if path is not None:
        rel = check_note_path(path)
        candidates = [rel]
    else:
        base = default_path(proposal, folder)
        stem, suffix = base[:-3], ".md"
        candidates = [base] + [f"{stem}-{n}{suffix}" for n in range(2, 100)]
    for rel in candidates:
        try:
            _create_exclusive(vault, rel, data)
        except FileExistsError:
            if path is not None:
                raise InboxError(f"{rel} already exists; pureMind never overwrites a note") from None
            continue
        discard(home, proposal)
        return rel
    raise InboxError("too many notes with that name; give one with --path")


def age(proposal: Proposal, now: float | None = None) -> str:
    """``proposal`` age as a short phrase: '5 min', '3 h', '2 d'."""
    try:
        made = datetime.strptime(proposal.created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return "unknown age"
    seconds = max(0.0, (time.time() if now is None else now) - made)
    if seconds < 3600:
        return f"{int(seconds // 60)} min"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h"
    return f"{int(seconds // 86400)} d"
