"""Debounced vault watcher: re-run the incremental indexer after edits settle."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

log = logging.getLogger("puremind.watch")


class Debouncer:
    """Calls ``fn`` once, ``delay`` seconds after the last ``poke``."""

    def __init__(self, fn: Callable[[], None], delay: float = 2.0):
        self._fn = fn
        self._delay = delay
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def poke(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        try:
            self._fn()
        except Exception:  # keep watching after a failed run
            log.exception("index run failed")

    def cancel(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()


def relevant(path: str) -> bool:
    p = Path(path)
    return p.suffix.lower() in (".md", ".markdown") and not p.name.startswith(".")


class _Handler(FileSystemEventHandler):
    def __init__(self, debouncer: Debouncer):
        self._d = debouncer

    def on_any_event(self, event: FileSystemEvent) -> None:
        paths = [str(event.src_path)]
        dest = getattr(event, "dest_path", "")
        if dest:
            paths.append(str(dest))
        if event.is_directory or any(relevant(p) for p in paths):
            self._d.poke()


def watch(vault: Path, run_index: Callable[[], None], delay: float = 2.0, stop: threading.Event | None = None) -> None:
    """Index once, then re-index after changes until ``stop`` is set (or Ctrl-C)."""
    stop = stop or threading.Event()
    debouncer = Debouncer(run_index, delay)
    observer = Observer()
    observer.schedule(_Handler(debouncer), str(vault), recursive=True)
    observer.start()
    try:
        run_index()
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        debouncer.cancel()
        observer.stop()
        observer.join()
