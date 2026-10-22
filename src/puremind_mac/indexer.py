"""Incremental vault indexing: SHA-256 per file, re-embed only what changed."""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import re
import sqlite3
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import (
    MAX_CHARS,
    Chunk,
    chunk_markdown,
    clip_heading,
    fenced_lines,
    frontmatter_end,
)
from .chunker import VERSION as CHUNKER_VERSION
from .models import Embedder, ModelLoadError, embedder_revision, model_key, model_label
from .store import Store, StoreError

log = logging.getLogger("puremind.index")

MARKDOWN_SUFFIXES = (".md", ".markdown")
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_CHUNKS_PER_FILE = 10_000
MAX_PATH_CHARS = 1024  # longest note path, relative to the vault, that is indexed and that puremind_get accepts
EMBED_BATCH = 64  # chunks per embedder call, gathered across files
RECHECK_ATTEMPTS = 3  # reads of one note when it keeps changing underneath the indexer
SKIP_DIRS = {"node_modules", "__pycache__"}
PROBE_TEXT = "pureMind embedder check"


class EmbedderError(RuntimeError):
    """The embedding model could not be loaded or failed. The index is left as it was."""


class VaultMismatch(ValueError):
    """The index belongs to another vault directory. Nothing was changed."""


class ModelMismatch(ValueError):
    """The index was built with another embedding model or revision, and no rebuild was asked for.
    Nothing was changed."""


@dataclass
class IndexStats:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    removal_skipped: str = ""  # why notes missing from the scan were not removed from the index
    stopped: bool = False  # the run was asked to stop before it finished
    purge_deferred: bool = False  # the keyword index still holds words of text this run removed

    def summary(self) -> str:
        s = (
            f"scanned {self.scanned}: {self.added} added, {self.updated} updated, "
            f"{self.unchanged} unchanged, {self.removed} removed ({self.chunks} chunks written)"
        )
        if self.removal_skipped:
            s += f", removal skipped: {self.removal_skipped}"
        if self.stopped:
            s += ", stopped early"
        return s + (f", {len(self.skipped)} skipped" if self.skipped else "")


def _excluded_part(part: str) -> bool:
    low = part.lower()  # APFS is case-insensitive by default
    return low.startswith(".") or low in SKIP_DIRS


def is_indexable(root: Path, path: Path) -> Path | None:
    """The resolved file if ``path`` is a note the indexer reads, else None.

    A note is a regular .md or .markdown file that, after resolving symlinks, lies
    inside ``root`` (already resolved) with no hidden or skipped component in its
    path relative to the vault. discover() and puremind_get both use this, so the
    tool never serves a file the indexer would exclude. Size is not checked here.
    """
    try:
        real = path.resolve()
        rel = real.relative_to(root)
        st = real.stat()
    except (OSError, RuntimeError, ValueError):  # RuntimeError: symlink loop on 3.11/3.12
        return None
    if not stat.S_ISREG(st.st_mode) or real.suffix.lower() not in MARKDOWN_SUFFIXES:
        return None
    if any(_excluded_part(p) for p in rel.parts):
        return None
    return real


def open_resolved(path: Path, flags: int) -> int:
    """Open the already-resolved ``path``, refusing a symlink at ANY component.

    O_NOFOLLOW alone covers only the last component, so an ancestor directory swapped for a
    symlink after is_indexable() would redirect the open outside the vault. Walk from the
    file system root one directory descriptor at a time instead.
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    dir_flags = os.O_RDONLY | nofollow | getattr(os, "O_DIRECTORY", 0)
    parts = os.fsencode(path).split(b"/")
    names = [p for p in parts if p]
    if not os.path.isabs(path) or not names:
        raise OSError(errno.EINVAL, "not an absolute file path")
    dir_fd = os.open("/", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        for name in names[:-1]:
            nxt = os.open(name, dir_flags, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = nxt
        return os.open(names[-1], flags | nofollow, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def _printable(name: str | os.PathLike[str]) -> str:
    """A file name safe to print: bytes that are not valid UTF-8 appear as \\xNN escapes."""
    return os.fsencode(name).decode("utf-8", "backslashreplace")


def _encodable(name: str) -> bool:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:  # a file name that is not valid UTF-8 comes back with surrogates
        return False
    return True


@dataclass
class _Scan:
    paths: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # entries ignored because of their name
    errors: list[str] = field(default_factory=list)  # directories that could not be listed
    unlisted: list[str] = field(default_factory=list)  # those directories relative to the vault, "" for the vault itself

    def unlisted_holds(self, rel: str) -> bool:
        """True if the note ``rel`` lies below a directory that could not be listed."""
        return any(d == "" or rel.startswith(d + "/") for d in self.unlisted)


def _scan(vault: Path) -> _Scan:
    root = vault.resolve()
    scan = _Scan()

    def unreadable(exc: OSError) -> None:
        where = Path(os.fsdecode(exc.filename)) if exc.filename else root
        try:
            where = where.relative_to(root)
            scan.unlisted.append("" if where == Path(".") else where.as_posix())
        except ValueError:
            scan.unlisted.append("")  # cannot tell where it is: keep every note
        scan.errors.append(f"{_printable(where)}/ (cannot list directory: {exc.strerror})")

    for dirpath, dirnames, filenames in os.walk(root, onerror=unreadable):
        here = Path(dirpath).relative_to(root).as_posix()
        if dirnames and len(here) >= MAX_PATH_CHARS:  # no note below could be opened; on Python 3.11 os.walk
            scan.skipped.append(f"{here[:100]}.../ (folders below it have paths longer than {MAX_PATH_CHARS} characters)")
            dirnames[:] = []  # also recurses once per level, so this bounds how deep it goes
        kept = []
        for d in sorted(dirnames):
            if _excluded_part(d):
                continue
            if _encodable(d):
                kept.append(d)
            else:
                shown = _printable(Path(dirpath, d).relative_to(root))
                scan.skipped.append(f"{shown}/ (directory name is not valid UTF-8)")
        dirnames[:] = kept
        for name in sorted(filenames):
            if name.startswith(".") or not name.lower().endswith(MARKDOWN_SUFFIXES):
                continue
            p = Path(dirpath, name)
            if is_indexable(root, p) is None:
                continue
            if not _encodable(name):  # directories with such names were left out above
                scan.skipped.append(f"{_printable(p.relative_to(root))} (file name is not valid UTF-8)")
                continue
            rel = p.relative_to(root).as_posix()
            if len(rel) > MAX_PATH_CHARS:  # puremind_get could not open it
                scan.skipped.append(f"{rel[:100]}... (path longer than {MAX_PATH_CHARS} characters)")
                continue
            scan.paths.append(p)
    return scan


def discover(vault: Path) -> list[Path]:
    """Markdown files under ``vault``, skipping hidden directories and links that leave it."""
    return _scan(vault).paths


def title_of(text: str, fallback: str) -> str:
    """The first level-1 heading outside frontmatter and code fences, else ``fallback``."""
    lines = text.removeprefix("\ufeff").splitlines()
    fm = frontmatter_end(lines)
    first = 0 if fm is None else fm + 1
    inside = fenced_lines(lines, first)
    for i in range(first, len(lines)):
        if i not in inside:
            m = re.match(r"#\s+(.+)$", lines[i].rstrip())
            if m:
                return clip_heading(m.group(1).strip())
    return fallback


class _TooLarge(Exception):
    pass


def _read_note(path: Path) -> tuple[bytes, os.stat_result]:
    """Bytes and stat of a regular file, read without blocking on a pipe, following a symlink or
    passing the size cap. ``path`` is already resolved (see is_indexable)."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)  # a FIFO swapped in must not hang us
    fd = open_resolved(path, flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(errno.EINVAL, "not a regular file")
        if st.st_size > MAX_FILE_BYTES:
            raise _TooLarge
        os.set_blocking(fd, True)  # a regular file reads as usual (a cloud placeholder may need to download)
        parts, room = [], MAX_FILE_BYTES + 1  # one byte more than allowed shows that the file grew
        while room > 0 and (part := os.read(fd, min(room, 1 << 20))):
            parts.append(part)
            room -= len(part)
    finally:
        os.close(fd)
    raw = b"".join(parts)
    if len(raw) > MAX_FILE_BYTES:
        raise _TooLarge
    return raw, st


def _one_line(exc: BaseException, limit: int = 300) -> str:
    """``exc`` as one printable line: control and other unprintable characters become spaces."""
    text = " ".join("".join(ch if ch.isprintable() else " " for ch in str(exc)).split())
    return f"{type(exc).__name__}: {text}"[:limit]


def _embed(embedder: Embedder, texts: Sequence[str], notes: Sequence[str] = ()):
    """Vectors for ``texts``; any failure of the model is an EmbedderError naming (some of) ``notes``."""
    where = ""
    if notes:
        where = f" on {', '.join(notes[:3])}" + (f" and {len(notes) - 3} more" if len(notes) > 3 else "")
    try:
        out = embedder.embed_documents(list(texts))
    except Exception as exc:
        raise EmbedderError(f"the embedding model {embedder.name} failed{where}: {_one_line(exc)}") from exc
    if len(out) != len(texts):
        raise EmbedderError(f"the embedding model {embedder.name} returned {len(out)} vectors for {len(texts)} texts")
    return out


@dataclass
class _Job:
    rel: str
    path: Path
    digest: str
    mtime: float
    size: int
    title: str
    chunks: list[Chunk]
    attempt: int = 1


class _Run:
    """One pass over the vault. Notes are read and chunked one by one, embedded in batches
    across notes, and committed one by one, each after a check that it still matches the file."""

    def __init__(self, vault: Path, store: Store, embedder: Embedder, stats: IndexStats, every: bool = False):
        self.vault, self.store, self.embedder, self.stats = vault, store, embedder, stats
        self.old = store.file_hashes()
        # Rows that may stay as they are when the note is unchanged: none when ``every`` note is to be
        # embedded again, and none cut by an earlier chunker.
        current = set() if every else store.chunked_with(CHUNKER_VERSION)
        self.keep = {rel: digest for rel, digest in self.old.items() if rel in current}
        self.seen: set[str] = set()
        self.queue: list[_Job] = []
        self.queued = 0

    def fail(self, rel: str, why: str) -> None:
        self.stats.skipped.append(f"{rel} ({why})")

    def read(self, path: Path) -> tuple[bytes, os.stat_result]:
        """The note's bytes, if ``path`` is still a note the indexer reads.

        It is checked again here because the vault can change after the scan: a note replaced
        by a symlink to a hidden file must not be read through it. One that no longer
        qualifies counts as gone.
        """
        real = is_indexable(self.vault, path)
        if real is None:
            os.lstat(path)  # raises if it is gone, or if it cannot be checked (then its entry is kept)
            raise FileNotFoundError(errno.ENOENT, "no longer a note of the vault", str(path))
        return _read_note(real)

    def prepare(self, rel: str, path: Path, raw: bytes, st: os.stat_result, digest: str, attempt: int = 1) -> _Job:
        text = raw.decode("utf-8-sig", errors="replace")
        chunks = chunk_markdown(text)
        if len(chunks) > MAX_CHUNKS_PER_FILE:
            chunks = chunk_markdown(text, min_chars=MAX_CHARS)  # pack the small sections instead
        if len(chunks) > MAX_CHUNKS_PER_FILE:
            self.fail(rel, f"only the first {MAX_CHUNKS_PER_FILE} of {len(chunks)} chunks were indexed")
            chunks = chunks[:MAX_CHUNKS_PER_FILE]
        return _Job(rel, path, digest, st.st_mtime, st.st_size, title_of(text, path.stem), chunks, attempt)

    def visit(self, path: Path) -> None:
        rel = path.relative_to(self.vault).as_posix()
        self.stats.scanned += 1
        try:
            raw, st = self.read(path)
        except _TooLarge:
            self.stats.skipped.append(f"{rel} (over {MAX_FILE_BYTES} bytes)")
            return
        except (FileNotFoundError, NotADirectoryError):
            return  # gone since the scan: not seen, so it is removed from the index
        except OSError as exc:
            self.seen.add(rel)  # present but unreadable: keep what is indexed
            self.fail(rel, exc.strerror or _one_line(exc))
            return
        self.seen.add(rel)
        digest = hashlib.sha256(raw).hexdigest()
        if self.keep.get(rel) == digest:
            self.stats.unchanged += 1
            return
        try:
            job = self.prepare(rel, path, raw, st, digest)
        except Exception as exc:  # noqa: BLE001 - chunking one odd note must not stop the run
            self.fail(rel, _one_line(exc))
            return
        self.queue.append(job)
        self.queued += len(job.chunks)
        if self.queued >= EMBED_BATCH:
            self.flush()

    def flush(self) -> None:
        jobs, self.queue, self.queued = self.queue, [], 0
        texts = [c.content for job in jobs for c in job.chunks]
        notes = [job.rel for job in jobs if job.chunks]
        try:
            vectors = _embed(self.embedder, texts, notes) if texts else None
        except EmbedderError as exc:
            if len(notes) < 2 or isinstance(exc.__cause__, ModelLoadError):
                raise
            self.one_by_one(jobs, exc)
            return
        at = 0
        for job in jobs:
            n = len(job.chunks)
            self.commit(job, vectors[at : at + n] if n else None)
            at += n

    def one_by_one(self, jobs: list[_Job], batch_error: EmbedderError) -> None:
        """The model failed on a batch: embed its notes one at a time, so that a note the model
        cannot handle costs only that note, which is named. If every note fails, the model itself
        is broken and the batch's error is raised."""
        failed: list[tuple[_Job, EmbedderError]] = []
        embedded = False
        for job in jobs:
            if not job.chunks:
                self.commit(job, None)
                continue
            try:
                vectors = _embed(self.embedder, [c.content for c in job.chunks], [job.rel])
            except EmbedderError as exc:
                failed.append((job, exc))
                continue
            embedded = True
            self.commit(job, vectors)
        if failed and not embedded:
            raise batch_error
        for job, exc in failed:  # still seen, so whatever was indexed for it stays
            self.fail(job.rel, f"the embedding model failed on it: {_one_line(exc.__cause__ or exc)}")

    def commit(self, job: _Job, vectors) -> None:
        """Store one note, but only if it is still the version that was embedded.

        Another process (a manual `puremind index` beside `puremind watch`) or an
        editor may have changed the file while the model was busy. It is read again
        just before the commit; a changed file is chunked and embedded again, a few
        times at most, so what is stored always matches a version that was on disk.
        """
        try:
            while True:
                raw, st = self.read(job.path)
                digest = hashlib.sha256(raw).hexdigest()
                if digest == job.digest:
                    break
                if self.keep.get(job.rel) == digest:
                    self.stats.unchanged += 1  # edited back to the indexed text
                    return
                if job.attempt >= RECHECK_ATTEMPTS:
                    self.fail(job.rel, "changed while it was being indexed; the previous version was kept")
                    return
                job = self.prepare(job.rel, job.path, raw, st, digest, job.attempt + 1)
                vectors = _embed(self.embedder, [c.content for c in job.chunks], [job.rel]) if job.chunks else None
            self.store.replace_file(
                job.rel, job.digest, st.st_mtime, st.st_size, job.title,
                [(c.heading_path, c.content) for c in job.chunks], vectors, chunker=CHUNKER_VERSION,
            )
        except (EmbedderError, StoreError, sqlite3.Error):
            raise  # the model or the index failed, not this note: every later note would fail the same way
        except (FileNotFoundError, NotADirectoryError):
            self.seen.discard(job.rel)  # deleted meanwhile: the removal pass drops it
            return
        except _TooLarge:
            self.seen.discard(job.rel)
            self.stats.skipped.append(f"{job.rel} (over {MAX_FILE_BYTES} bytes)")
            return
        except Exception as exc:  # noqa: BLE001 - reading or chunking one note must not stop the run
            self.fail(job.rel, _one_line(exc))
            return
        self.stats.chunks += len(job.chunks)
        if job.rel in self.old:
            self.stats.updated += 1
        else:
            self.stats.added += 1


def _same_directory(known: str, vault: Path) -> bool:
    """True if ``known`` names ``vault``, also when spelled differently (APFS ignores case)."""
    if known == str(vault):
        return True
    try:
        return os.path.samefile(known, vault)
    except OSError:
        return False


def index_vault(
    vault: Path,
    store: Store,
    embedder: Embedder,
    rebuild: bool = False,
    *,
    should_stop: Callable[[], bool] | None = None,
    purge: bool = True,
) -> IndexStats:
    """Bring the index in line with the vault.

    ``rebuild`` embeds every note again, replacing each one's rows in turn, so searches keep
    finding the old version meanwhile. A new embedding model, or a new revision of it, is refused
    with ``ModelMismatch`` unless ``rebuild`` is set, and then discards the index first, since
    vectors from two models cannot be searched together. Notes cut by an earlier chunker are cut
    and embedded again. ``should_stop`` is polled between notes; once it returns True the run
    ends early without removing anything.

    Text a run removes or replaces leaves the tables at once. Its words leave the keyword index
    when the index is compacted (``Store.purge``), which takes about half a second at 50,000
    chunks. With ``purge`` the run compacts at its end, also paying a compaction an earlier run
    put off; without it, the run only records that one is owed (``stats.purge_deferred``) and the
    caller pays it later, as ``puremind watch`` does.
    """
    vault = vault.expanduser().resolve()
    if not vault.is_dir():
        raise NotADirectoryError(f"vault directory not found: {vault}")
    stats = IndexStats()

    known_root = store.get_meta("vault_root")
    if not rebuild and known_root and not _same_directory(known_root, vault) and store.stats()["files"]:
        raise VaultMismatch(
            f"this index belongs to {known_root}, not {vault}; use a different PUREMIND_DB "
            "(or db = in config.toml) for this vault, or run `puremind index --rebuild` if the vault moved"
        )
    known_model = store.get_meta("embedder")
    known_rev = store.get_meta("embedder_revision")  # None in an index from 0.1.1, which pinned none
    revision = embedder_revision(embedder)
    changed = known_model is not None and (
        model_key(known_model) != model_key(embedder.name) or (known_rev is not None and known_rev != revision)
    )
    if changed and store.stats()["files"]:
        if not rebuild:  # else a `watch` left on the old config and a fresh `index` wipe each other in turn
            built = model_label(known_model) + (f" at revision {known_rev}" if known_rev else "")
            wanted = model_label(embedder.name) + (f" at revision {revision}" if revision else "")
            raise ModelMismatch(f"the index was built with {built} but {wanted} is configured; nothing was changed")
        _embed(embedder, [PROBE_TEXT])  # a model that cannot run must not cost the user an index
    if changed:
        log.warning(
            "embedder changed (%s%s -> %s%s); rebuilding the index",
            known_model, f" at {known_rev}" if known_rev else "", embedder.name, f" at {revision}" if revision else "",
        )
        store.clear()
    store.set_meta("embedder", model_key(embedder.name))
    store.set_meta("embedder_revision", revision)
    store.set_meta("vault_root", str(vault))

    scan = _scan(vault)
    stats.skipped.extend(scan.skipped)
    stats.skipped.extend(scan.errors)
    run = _Run(vault, store, embedder, stats, every=rebuild)
    for path in scan.paths:
        if should_stop is not None and should_stop():
            stats.stopped = True
            break
        run.visit(path)
    if not stats.stopped:
        run.flush()
        gone = sorted(set(run.old) - run.seen)
        if scan.errors:  # notes below a directory that could not be listed stay; the others are really gone
            many = len(scan.errors)
            stats.removal_skipped = f"{many} director{'y' if many == 1 else 'ies'} could not be read"
            gone = [rel for rel in gone if not scan.unlisted_holds(rel)]
        elif not scan.paths and run.old and not rebuild:  # a rebuild of an empty vault empties the index
            stats.removal_skipped = (
                f"no Markdown files were found but the index holds {len(run.old)} "
                "(is the vault mounted? run `puremind index --rebuild` if it is really empty)"
            )
            gone = []
        for rel in gone:
            store.delete_file(rel)
            stats.removed += 1
    emptied = True
    if stats.removed or stats.updated:  # clear() purges by itself
        if purge:
            emptied = store.purge()
        else:
            emptied = store.defer_purge()
            stats.purge_deferred = True
    elif purge and store.purge_pending():  # owed by a `puremind watch` that stopped before paying it
        emptied = store.purge()
    if not emptied:
        log.warning(
            "another process was reading the index, so text deleted or replaced in this run stays in "
            "%s-wal until the next run", store.path,
        )
    return stats
