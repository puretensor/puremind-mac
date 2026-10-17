"""Vault watcher: re-run the incremental indexer once the vault has been quiet for a while.

The file system observer only records that something changed. The indexer runs on the
thread that called ``watch``, one run at a time, so runs never overlap and Ctrl-C
interrupts the one place where work happens.
"""

from __future__ import annotations

import errno
import logging
import os
import threading
import time
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path

from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirMovedEvent,
    FileClosedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from .indexer import MARKDOWN_SUFFIXES, _excluded_part, _one_line

log = logging.getLogger("puremind.watch")

# Event types that can change the set of notes. Plain reads (watchdog reports them as
# "opened" and "closed_no_write" on Linux) are absent on purpose: the indexer reads every
# note on every run, and those reads must never schedule the next run.
CHANGES = frozenset({"created", "modified", "deleted", "moved", "closed"})
# Asked of the observer as well: on Linux this narrows the inotify mask, so the kernel does not
# even report reads (tens of thousands of events per run on a big vault). The handler checks again.
EVENT_FILTER = [
    FileCreatedEvent, FileModifiedEvent, FileDeletedEvent, FileMovedEvent, FileClosedEvent,
    DirCreatedEvent, DirDeletedEvent, DirMovedEvent,
]
TICK = 0.25  # how often the loop looks at the clock, at ``stop`` and at the vault directory
RESCAN = 600.0  # seconds without a run after which the vault is scanned anyway, in case an event was missed
# Errors of observer.schedule() that retrying cannot fix: the watch or instance limit, or no file descriptors left.
_LIMITS = frozenset({errno.ENOSPC, errno.EMFILE, errno.ENFILE, errno.ENOMEM})


class Settle:
    """Thread-safe note that the vault changed, and when it counts as quiet again."""

    def __init__(self, delay: float, clock: Callable[[], float] = time.monotonic):
        self._delay = delay
        self._clock = clock
        self._lock = threading.Lock()
        self._due: float | None = None

    def poke(self) -> None:
        """Record a change: the vault is quiet again ``delay`` seconds from now."""
        with self._lock:
            self._due = self._clock() + self._delay

    def remaining(self) -> float | None:
        """Seconds until a pending change has settled (0 if it has), None if nothing is pending."""
        with self._lock:
            return None if self._due is None else max(0.0, self._due - self._clock())

    def take(self) -> bool:
        """Consume a settled change. False if nothing is pending or the vault is still busy."""
        with self._lock:
            if self._due is None or self._due > self._clock():
                return False
            self._due = None
            return True


def _fold(name: str) -> str:
    return unicodedata.normalize("NFD", name).casefold()


def _below(p: Path, root: Path) -> tuple[str, ...] | None:
    """The parts of ``p`` below ``root``, or None if it is not inside it.

    The vault may be spelled with another letter case or Unicode normalisation than the one on
    disk (APFS ignores both, and FSEvents reports the spelling on disk), so that still counts as inside.
    """
    try:
        return p.relative_to(root).parts
    except ValueError:
        pass
    n = len(root.parts)
    if len(p.parts) >= n and all(_fold(a) == _fold(b) for a, b in zip(p.parts, root.parts)):
        return p.parts[n:]
    return None


def relevant(path: str, root: Path | None = None, *, directory: bool = False) -> bool:
    """Whether a change at ``path`` can alter what discover() finds.

    With ``root``, the same rules as discover() apply to the path inside the vault: no
    hidden or skipped component, and a Markdown suffix unless it is a directory. Without
    it only the file name is judged.
    """
    p = Path(path)
    if root is None:
        parts: Sequence[str] | None = (p.name,)
    else:
        parts = _below(p, root)
        if parts is None:
            return False
    if any(_excluded_part(part) for part in parts):
        return False
    return directory or p.suffix.lower() in MARKDOWN_SUFFIXES


class _Handler(FileSystemEventHandler):
    def __init__(self, vault: Path, settle: Settle):
        vault = Path(vault).expanduser().absolute()
        self._roots = [vault] if vault.resolve() == vault else [vault, vault.resolve()]
        self._settle = settle

    def _relevant(self, path: str, directory: bool) -> bool:
        return any(relevant(path, root, directory=directory) for root in self._roots)

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type not in CHANGES:
            return
        if event.is_directory and event.event_type not in ("created", "deleted", "moved"):
            return  # a directory "modified" only mirrors a change to one of its files
        paths = [os.fsdecode(event.src_path)]
        dest = getattr(event, "dest_path", "")
        if dest:
            paths.append(os.fsdecode(dest))
        if any(self._relevant(p, event.is_directory) for p in paths):
            self._settle.poke()


def _identity(path: Path) -> tuple[int, int] | None:
    """Device and inode of the directory ``path``, or None if it is gone or cannot be listed.

    An observer started on a directory it cannot read sees nothing, and with watchdog before
    5.0.3 it also cannot be stopped, so such a directory is treated as not there.
    """
    try:
        st = os.stat(path)
        with os.scandir(path):
            pass
    except OSError:
        return None
    return st.st_dev, st.st_ino


def watch(
    vault: Path,
    run_index: Callable[[], None],
    delay: float = 2.0,
    stop: threading.Event | None = None,
    rescan: float = RESCAN,
) -> None:
    """Run ``run_index`` once, then again each time the vault has been quiet for ``delay`` seconds.

    Runs happen on the calling thread, one at a time. A change that arrives during a run
    schedules exactly one more run after it. A failing run is logged and watching goes on,
    except the first, whose error propagates. If the vault directory is replaced or made
    again (a remount, or a sync tool swapping it), the new directory is watched and indexed;
    while it is missing or cannot be read, nothing runs. After ``rescan`` seconds without a
    run the vault is indexed anyway, in case the observer missed an event. Returns when
    ``stop`` is set. Ctrl-C raises KeyboardInterrupt here after the observer has been shut down.
    """
    stop = stop or threading.Event()
    settle = Settle(delay)
    root = Path(vault).expanduser().absolute()  # events carry the path given here, so make it absolute
    observer = Observer()

    def arm() -> bool:
        """Watch ``root`` with a new handler, since a symlinked vault may now resolve somewhere else.
        False if the directory went away again before it could be watched.

        A system limit (the inotify watch or instance limit, too many open files) is raised: it does
        not pass by itself, and every failed attempt leaks an inotify instance inside watchdog.
        """
        observer.unschedule_all()
        try:
            observer.schedule(_Handler(root, settle), str(root), recursive=True, event_filter=EVENT_FILTER)
        except OSError as exc:
            if exc.errno in _LIMITS:
                raise
            return False
        return True

    def deaf() -> bool:
        """True once the observer has stopped watching. inotify ends a watch when its directory is
        deleted, and a directory made again at once can get the same inode number (ext4 does this)."""
        return not any(emitter.is_alive() for emitter in observer.emitters)

    identity = _identity(root)
    armed = identity is not None and arm()
    observer.start()
    try:
        if not stop.is_set():
            run_index()
        last_run = time.monotonic()
        while not stop.is_set():
            current = _identity(root)
            recreated = current is not None and current == identity and armed and deaf()
            if current != identity or recreated or (current is not None and not armed):
                before, identity = identity, current
                if current is None:
                    if before is not None:
                        log.warning("%s is gone or cannot be read; watching resumes when it is back", root)
                    armed = False
                elif arm():
                    armed = True
                    what = "is back" if before is None else "was made again" if recreated else "was replaced"
                    log.warning("%s %s; watching it", root, what)
                    settle.poke()
                else:
                    armed, identity = False, None  # gone again already: try on the next tick
            if identity is None:  # nothing to index until the vault is back; a change waits for it
                stop.wait(TICK)
                continue
            wait = settle.remaining()
            if wait is not None and wait > 0:
                stop.wait(min(wait, TICK))
                continue
            if wait is None:
                if time.monotonic() - last_run < rescan:
                    stop.wait(TICK)
                    continue
            elif not settle.take():
                continue
            try:
                run_index()
            except Exception as exc:  # keep watching after a failed run
                log.error("index run failed: %s", _one_line(exc))
                log.debug("index run failed", exc_info=True)
            last_run = time.monotonic()
    finally:
        observer.stop()
        observer.join(timeout=5)
