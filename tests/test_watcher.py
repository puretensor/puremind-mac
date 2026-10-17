"""The watcher against a real file system observer (inotify on Linux, FSEvents on macOS).

Positive waits use generous timeouts and return as soon as they can. To show that
something did NOT happen, a test first waits until the observer has delivered the
events in question, then lets the debounce expire a few times over.
"""

from __future__ import annotations

import errno
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from puremind_mac import watcher
from puremind_mac.indexer import index_vault
from puremind_mac.store import Store
from puremind_mac.watcher import Settle, relevant, watch

STARTUP_HISTORY = sys.platform == "darwin"  # FSEvents can replay changes made just before the stream started
DELAY = 0.6 if STARTUP_HISTORY else 0.2  # FSEvents coalesces and delays events, so give it more room
PATIENT = 20.0  # upper bound for anything that should happen


def wait_for(condition, timeout=PATIENT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class Watching:
    """``watch`` on its own thread, with a run function that records every run."""

    def __init__(self, vault, monkeypatch, work=None, first_run=True):
        self.vault, self.work, self.first_run = vault, work, first_run
        self.runs: list[tuple[float, float, str]] = []  # (start, end, thread name)
        self.events: list[tuple[str, str, bool]] = []  # what the handler was given
        self.active = self.most_active = 0
        self.stop = threading.Event()
        self.failure: BaseException | None = None
        # A daemon, so that a failing assertion in __enter__ (which skips __exit__) cannot hang the run.
        self.thread = threading.Thread(target=self._main, name="watch-thread", daemon=True)
        self._lock = threading.Lock()
        original = watcher._Handler.on_any_event
        events = self.events

        def spy(handler, event):
            events.append((event.event_type, Path(os.fsdecode(event.src_path)).name, event.is_directory))
            return original(handler, event)

        monkeypatch.setattr(watcher._Handler, "on_any_event", spy)

    def _main(self):
        try:
            watch(self.vault, self._run, delay=DELAY, stop=self.stop)
        except BaseException as exc:  # reported by __exit__
            self.failure = exc

    def _run(self):
        with self._lock:
            self.active += 1
            self.most_active = max(self.most_active, self.active)
        started = time.monotonic()
        try:
            if self.work:
                self.work(len(self.runs) + 1)
        finally:
            with self._lock:
                self.active -= 1
                self.runs.append((started, time.monotonic(), threading.current_thread().name))

    def __enter__(self):
        self.thread.start()
        if self.first_run:
            assert wait_for(lambda: self.runs), "the first run never happened"
            self.settle(quiet=3 * DELAY if STARTUP_HISTORY else 0)
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(PATIENT)
        assert not self.thread.is_alive(), "watch() did not return after stop was set"
        if self.failure is not None and exc[0] is None:
            raise self.failure

    def settle(self, quiet=2 * DELAY):
        """Wait until no run is active and none has ended for ``quiet`` seconds."""
        def calm():
            return self.active == 0 and bool(self.runs) and time.monotonic() - self.runs[-1][1] >= quiet
        assert wait_for(calm), "runs kept coming"

    def delivered(self, *names):
        """Wait until the observer has handed the handler an event for each of ``names``."""
        assert wait_for(lambda: all(any(n == name for _, n, _ in self.events) for name in names)), (names, self.events)

    def quiet_period(self):
        time.sleep(4 * DELAY)


@pytest.fixture
def watching(vault, monkeypatch):
    return lambda **kw: Watching(vault, monkeypatch, **kw)


# ---- reads must never schedule a run


@pytest.mark.parametrize("narrowed", [True, False], ids=["event-filter", "every-event"])
def test_reading_notes_never_schedules_a_run(watching, vault, monkeypatch, narrowed):
    if not narrowed:
        monkeypatch.setattr(watcher, "EVENT_FILTER", None)  # let the reads reach the handler: it must ignore them
    linux = sys.platform.startswith("linux")
    with watching() as w:
        before = len(w.runs)
        for note in vault.rglob("*.md"):
            note.read_bytes()
            with open(note) as fh:
                fh.read()
        if linux and not narrowed:  # inotify reports plain reads; make sure they reached the handler
            assert wait_for(lambda: any(kind == "opened" for kind, _, _ in w.events))
        w.quiet_period()
        assert len(w.runs) == before
        if linux and narrowed:  # the observer was asked not to hear about reads at all
            assert not [kind for kind, _, _ in w.events if kind in ("opened", "closed_no_write")]


def test_an_idle_watch_runs_the_indexer_once(watching, vault, tmp_path, embedder):
    store = Store(tmp_path / "idle.sqlite")
    try:
        with watching(work=lambda n: index_vault(vault, store, embedder)) as w:  # the indexer reads every note
            before = len(w.runs)
            time.sleep(6 * DELAY)
            assert len(w.runs) == before
        assert store.stats()["files"] == 3
    finally:
        store.close()


# ---- bursts, noise and the changes that matter


def test_a_vault_given_as_a_relative_path_is_watched(vault, monkeypatch):
    monkeypatch.chdir(vault.parent)
    runs = []
    stop = threading.Event()
    thread = threading.Thread(target=watch, args=(Path(vault.name), lambda: runs.append(1)), kwargs={"delay": DELAY, "stop": stop})
    thread.start()
    try:
        assert wait_for(lambda: runs)
        time.sleep(STARTUP_HISTORY * 3 * DELAY)
        before = len(runs)
        (vault / "relative.md").write_text("# Relative\n\ntext\n")
        assert wait_for(lambda: len(runs) == before + 1)
    finally:
        stop.set()
        thread.join(PATIENT)


def test_a_burst_of_writes_gives_one_run(watching, vault):
    with watching() as w:
        before = len(w.runs)
        for i in range(8):
            (vault / "projects" / "garden.md").write_text(f"# Garden\n\nedit {i}\n")
            (vault / f"extra{i}.md").write_text("# Extra\n\nnote\n")
        assert wait_for(lambda: len(w.runs) == before + 1)
        w.settle()
        assert len(w.runs) == before + 1


def test_changes_that_cannot_alter_the_note_set_are_ignored(watching, vault):
    with watching() as w:
        before = len(w.runs)
        (vault / "notes.txt").write_text("changed")
        (vault / ".DS_Store").write_bytes(b"\0")
        (vault / "image.png").write_bytes(b"\x89PNG")
        (vault / ".obsidian" / "workspace.json").write_text("{}")
        (vault / ".obsidian" / "workspace.md").write_text("# edited hidden note\n")
        (vault / ".obsidian" / "plugins").mkdir()
        (vault / "node_modules" / "pkg").mkdir(parents=True)
        (vault / "node_modules" / "pkg" / "README.md").write_text("# a dependency\n")
        w.delivered("notes.txt", ".DS_Store", "image.png", "workspace.json", "workspace.md", "README.md")
        w.quiet_period()
        assert len(w.runs) == before


@pytest.mark.parametrize("action", ["new", "edit", "delete", "rename", "atomic-save", "folder-in", "folder-out", "into-hidden"])
def test_changes_to_visible_notes_and_folders_schedule_exactly_one_run(watching, vault, tmp_path, action):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "inbox").mkdir()
    (outside / "inbox" / "from-outside.md").write_text("# Moved in\n\ntext\n")
    garden = vault / "projects" / "garden.md"
    with watching() as w:
        before = len(w.runs)
        if action == "new":
            (vault / "brand-new.md").write_text("# New\n\ntext\n")
        elif action == "edit":
            garden.write_text("# Garden plan\n\nedited\n")
        elif action == "delete":
            garden.unlink()
        elif action == "rename":
            garden.rename(vault / "projects" / "yard.md")
        elif action == "atomic-save":  # what most editors do: write a temporary file, rename it over the note
            tmp = vault / "projects" / ".garden.md.tmp"
            tmp.write_text("# Garden plan\n\nsaved atomically\n")
            os.replace(tmp, garden)
        elif action == "folder-in":
            os.rename(outside / "inbox", vault / "inbox")
        elif action == "folder-out":
            os.rename(vault / "projects", outside / "projects")
        elif action == "into-hidden":
            os.rename(garden, vault / ".obsidian" / "garden.md")
        assert wait_for(lambda: len(w.runs) == before + 1), (action, w.events)
        w.settle()
        assert len(w.runs) == before + 1


# ---- runs are serialised


def test_runs_never_overlap_and_all_happen_on_the_calling_thread(watching, vault):
    def slow(n):
        time.sleep(0.4 if n > 1 else 0)

    with watching(work=slow) as w:
        for i in range(3):
            (vault / f"burst{i}.md").write_text("# Burst\n\ntext\n")
            time.sleep(DELAY * 2.5)  # shorter than a run, longer than the debounce
        w.settle(quiet=1.0)
    assert len(w.runs) >= 2 and w.most_active == 1
    assert {name for _, _, name in w.runs} == {"watch-thread"}
    for (_, previous_end, _), (start, _, _) in zip(w.runs, w.runs[1:]):
        assert start >= previous_end


def test_an_edit_during_a_run_gives_exactly_one_follow_up_run(watching, vault):
    gate = threading.Event()
    started = threading.Event()

    def work(n):
        if n == 1:
            started.set()
            assert gate.wait(PATIENT)

    with watching(work=work, first_run=False) as w:
        assert started.wait(PATIENT)
        for i in range(3):
            (vault / "projects" / "garden.md").write_text(f"# Garden\n\nedited during the run {i}\n")
        w.delivered("garden.md")
        time.sleep(3 * DELAY)  # long enough for a debounce timer to have fired during the run
        gate.set()
        assert wait_for(lambda: len(w.runs) == 2)
        w.settle()
        assert len(w.runs) == 2
        assert w.runs[1][0] >= w.runs[0][1]


# ---- failures and shutdown


def test_a_failing_run_is_logged_and_watching_goes_on(watching, vault, caplog):
    armed = []

    def work(n):
        if armed:
            armed.clear()
            raise RuntimeError("model fell over")

    with caplog.at_level(logging.ERROR, logger="puremind.watch"), watching(work=work) as w:
        before = len(w.runs)
        armed.append(1)
        (vault / "one.md").write_text("# One\n\ntext\n")
        assert wait_for(lambda: len(w.runs) == before + 1)
        w.settle()
        (vault / "two.md").write_text("# Two\n\ntext\n")
        assert wait_for(lambda: len(w.runs) == before + 2)
    assert "index run failed" in caplog.text and "model fell over" in caplog.text


def test_a_failing_first_run_propagates_and_stops_the_observer(vault):
    def boom():
        raise ValueError("this index belongs elsewhere")

    before = threading.active_count()
    with pytest.raises(ValueError, match="belongs elsewhere"):
        watch(vault, boom, delay=DELAY)
    assert wait_for(lambda: threading.active_count() <= before)


def test_ctrl_c_in_a_run_propagates_after_the_observer_is_stopped(vault):
    def interrupted():
        raise KeyboardInterrupt

    before = threading.active_count()
    with pytest.raises(KeyboardInterrupt):
        watch(vault, interrupted, delay=DELAY)
    assert wait_for(lambda: threading.active_count() <= before)


def test_nothing_runs_when_stop_is_already_set(vault):
    stop = threading.Event()
    stop.set()
    runs = []
    watch(vault, lambda: runs.append(1), delay=DELAY, stop=stop)
    assert runs == []


# ---- a replaced vault directory, and the periodic rescan


def _watch_in_thread(vault, run, **kw):
    stop = threading.Event()
    thread = threading.Thread(target=watch, args=(vault, run), kwargs={"delay": DELAY, "stop": stop, **kw})
    thread.start()
    return stop, thread


def test_a_replaced_vault_directory_is_followed(tmp_path, caplog):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# a\n")
    seen = []
    stop, thread = _watch_in_thread(vault, lambda: seen.append(sorted(p.name for p in vault.glob("*.md"))))
    try:
        assert wait_for(lambda: seen)
        with caplog.at_level(logging.WARNING, logger="puremind.watch"):
            vault.rename(tmp_path / "vault.old")  # what a remount or a sync tool swapping the folder looks like
            vault.mkdir()
            (vault / "b.md").write_text("# b\n")
            assert wait_for(lambda: seen[-1] == ["b.md"])
            (vault / "c.md").write_text("# c\n")  # only a watch on the new directory sees this
            assert wait_for(lambda: seen[-1] == ["b.md", "c.md"])
        assert any("was replaced; watching it" in r.getMessage() for r in caplog.records)
    finally:
        stop.set()
        thread.join(PATIENT)
    assert not thread.is_alive()


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a directory this user cannot read")
def test_an_unreadable_replacement_is_watched_once_it_can_be_read(tmp_path, caplog):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# a\n")
    seen = []
    stop, thread = _watch_in_thread(vault, lambda: seen.append(sorted(p.name for p in vault.glob("*.md"))))
    try:
        assert wait_for(lambda: seen)
        with caplog.at_level(logging.WARNING, logger="puremind.watch"):
            vault.rename(tmp_path / "vault.old")
            vault.mkdir(mode=0)
            assert wait_for(lambda: "cannot be read" in caplog.text)
            vault.chmod(0o755)
            (vault / "b.md").write_text("# b\n")
            assert wait_for(lambda: seen[-1] == ["b.md"])
            (vault / "c.md").write_text("# c\n")  # only an observer armed on the readable directory sees this
            assert wait_for(lambda: seen[-1] == ["b.md", "c.md"])
    finally:
        vault.chmod(0o755)
        stop.set()
        thread.join(PATIENT)
    assert not thread.is_alive()


def test_a_failed_later_run_is_reported_on_one_printable_line(vault, caplog):
    runs = []

    def run():
        runs.append(1)
        if len(runs) == 2:
            raise OSError("cannot read evil\x1b]0;TITLE\x07.md\nsecond line")

    with caplog.at_level(logging.DEBUG, logger="puremind.watch"):
        stop, thread = _watch_in_thread(vault, run, rescan=0.3)
        try:
            assert wait_for(lambda: len(runs) >= 3)  # watching went on
        finally:
            stop.set()
            thread.join(PATIENT)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and errors[0].exc_info is None
    assert errors[0].getMessage() == "index run failed: OSError: cannot read evil ]0;TITLE .md second line"
    assert any(r.levelno == logging.DEBUG and r.exc_info for r in caplog.records)  # the traceback, for debugging


def test_an_idle_vault_is_scanned_again_after_the_rescan_interval(vault):
    runs = []
    stop, thread = _watch_in_thread(vault, lambda: runs.append(time.monotonic()), rescan=0.4)
    try:
        assert wait_for(lambda: len(runs) >= 3)  # nothing changed, yet it looks again
    finally:
        stop.set()
        thread.join(PATIENT)
    assert not thread.is_alive()


def test_puremind_watch_reports_a_model_that_cannot_run_and_exits_2(vault, tmp_path, monkeypatch, capsys):
    from puremind_mac import cli

    class Broken:
        name = "broken-stub"

        def __init__(self, *args, **kwargs):
            pass

        def embed_documents(self, texts):
            raise OSError("model files not found")

    monkeypatch.setenv("PUREMIND_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PUREMIND_DB", raising=False)
    monkeypatch.setattr("puremind_mac.service.SentenceTransformerEmbedder", Broken)
    assert cli.main(["watch", str(vault), "--debounce", "0.1"]) == 2
    out = capsys.readouterr()
    assert "error: the embedding model broken-stub failed" in out.err and "model files not found" in out.err


def test_puremind_watch_prints_what_it_skipped(vault, tmp_path, monkeypatch, capsys):
    import numpy as np

    from puremind_mac import cli

    (vault / "toolong.md").write_text("x" * 100)
    monkeypatch.setenv("PUREMIND_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PUREMIND_DB", raising=False)
    monkeypatch.setattr("puremind_mac.indexer.MAX_FILE_BYTES", 50)
    stub = SimpleNamespace(name="stub", embed_documents=lambda texts: np.ones((len(texts), 4), dtype="float32"))
    monkeypatch.setattr("puremind_mac.service.SentenceTransformerEmbedder", lambda *a, **k: stub)
    monkeypatch.setattr("puremind_mac.watcher.watch", lambda vault, run_index, delay, stop=None: run_index())  # one run
    (vault / "evil\x1b]0;TITLE\x07.md").write_text("x" * 100)
    assert cli.main(["watch", str(vault), "--debounce", "0.1"]) == 0
    out = capsys.readouterr()
    assert "scanned" in out.out and "skipped: toolong.md (over 50 bytes)" in out.err
    assert "skipped: evil]0;TITLE.md (over 50 bytes)" in out.err  # no escape sequence reaches the terminal


def test_puremind_watch_stops_when_a_symlinked_vault_names_another_directory(tmp_path, monkeypatch, capsys):
    import numpy as np

    from puremind_mac import cli

    first, second = tmp_path / "first", tmp_path / "second"
    for folder in (first, second):
        folder.mkdir()
        (folder / "a.md").write_text(f"# {folder.name}\n")
    link = tmp_path / "vault"
    link.symlink_to(first)
    monkeypatch.setenv("PUREMIND_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PUREMIND_DB", raising=False)
    stub = SimpleNamespace(name="stub", embed_documents=lambda texts: np.ones((len(texts), 4), dtype="float32"))
    monkeypatch.setattr("puremind_mac.service.SentenceTransformerEmbedder", lambda *a, **k: stub)

    def two_runs(vault, run_index, delay, stop):
        run_index()
        link.unlink()
        link.symlink_to(second)  # the index belongs to the first directory: every later run is refused
        run_index()
        assert stop.is_set()

    monkeypatch.setattr("puremind_mac.watcher.watch", two_runs)
    assert cli.main(["watch", str(link)]) == 2
    assert "error: this index belongs to" in capsys.readouterr().err


def test_a_stale_puremind_watch_stops_instead_of_wiping_an_index_rebuilt_for_another_model(
    vault, tmp_path, monkeypatch, capsys
):
    import numpy as np

    from puremind_mac import cli
    from puremind_mac.indexer import index_vault
    from puremind_mac.store import Store

    home = tmp_path / "home"
    monkeypatch.setenv("PUREMIND_HOME", str(home))
    monkeypatch.delenv("PUREMIND_DB", raising=False)

    def stub(name):
        return SimpleNamespace(name=name, embed_documents=lambda texts: np.ones((len(texts), 4), dtype="float32"))

    monkeypatch.setattr("puremind_mac.service.SentenceTransformerEmbedder", lambda *a, **k: stub("old-model"))

    def two_runs(vault, run_index, delay, stop):
        run_index()
        store = Store(home / "index.sqlite")
        try:  # `puremind index --rebuild` with the new config, while this watch still has the old one
            index_vault(vault, store, stub("new-model"), rebuild=True)
        finally:
            store.close()
        run_index()
        assert stop.is_set()

    monkeypatch.setattr("puremind_mac.watcher.watch", two_runs)
    assert cli.main(["watch", str(vault)]) == 2
    err = capsys.readouterr().err
    assert "error: the index was built with new-model but old-model is configured; nothing was changed" in err
    assert "hint: `puremind watch` keeps the model it started with: restart it if the model has changed" in err
    store = Store(home / "index.sqlite")
    try:
        assert store.get_meta("embedder") == "new-model" and store.stats()["files"] == 3
    finally:
        store.close()


def test_puremind_watch_names_a_missing_vault(tmp_path, monkeypatch, capsys):
    from puremind_mac import cli

    monkeypatch.setenv("PUREMIND_HOME", str(tmp_path / "home"))
    for command in ("watch", "index"):
        assert cli.main([command, str(tmp_path / "nowhere")]) == 2
        assert f"error: vault directory not found: {tmp_path / 'nowhere'}" in capsys.readouterr().err
    assert not (tmp_path / "home" / "index.sqlite").exists()  # a mistyped path creates no empty index


# ---- the handler's filter, without a file system


def _event(kind, src, dest="", directory=False):
    return SimpleNamespace(event_type=kind, src_path=src, dest_path=dest, is_directory=directory)


# (event type, source, destination, is a directory, expected to schedule a run); {v} is the vault
HANDLER_CASES = [
    ("modified", "{v}/a.md", "", False, True),
    ("created", "{v}/sub/a.MARKDOWN", "", False, True),
    ("deleted", "{v}/a.md", "", False, True),
    ("closed", "{v}/a.md", "", False, True),
    ("moved", "{v}/a.md.tmp", "{v}/a.md", False, True),
    ("moved", "{v}/a.md", "{v}/.obsidian/a.md", False, True),
    ("moved", "{v}/.trash/a.md", "{v}/a.md", False, True),
    ("created", "{v}/folder", "", True, True),
    ("deleted", "{v}/folder", "", True, True),
    ("moved", "{v}/folder", "{v}/renamed", True, True),
    ("moved", "{v}/folder", "{v}/.hidden", True, True),
    ("opened", "{v}/a.md", "", False, False),
    ("closed_no_write", "{v}/a.md", "", False, False),
    ("modified", "{v}", "", True, False),
    ("modified", "{v}/sub", "", True, False),
    ("modified", "{v}/a.txt", "", False, False),
    ("created", "{v}/.DS_Store", "", False, False),
    ("modified", "{v}/.hidden.md", "", False, False),
    ("modified", "{v}/.obsidian/workspace.md", "", False, False),
    ("created", "{v}/node_modules/pkg/README.md", "", False, False),
    ("created", "{v}/__pycache__/x.md", "", False, False),
    ("created", "{v}/Node_Modules/x.md", "", False, False),
    ("created", "{v}/.obsidian", "", True, False),
    ("created", "{v}/sub/.git", "", True, False),
    ("modified", "/somewhere/else/a.md", "", False, False),
]


@pytest.mark.parametrize("kind, src, dest, directory, poked", HANDLER_CASES)
def test_the_handler_pokes_only_for_changes_that_can_alter_the_note_set(tmp_path, kind, src, dest, directory, poked):
    pokes = []
    vault = tmp_path / "vault"
    vault.mkdir()
    handler = watcher._Handler(vault, SimpleNamespace(poke=lambda: pokes.append(1)))
    handler.on_any_event(_event(kind, src.replace("{v}", str(vault)), dest.replace("{v}", str(vault)), directory))
    assert len(pokes) == (1 if poked else 0)


def test_the_handler_follows_a_vault_given_through_a_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    pokes = []
    handler = watcher._Handler(link, SimpleNamespace(poke=lambda: pokes.append(1)))
    handler.on_any_event(_event("modified", str(real / "a.md")))  # FSEvents reports resolved paths
    handler.on_any_event(_event("modified", str(link / "b.md")))
    handler.on_any_event(_event("modified", str(real / ".obsidian" / "c.md")))
    assert len(pokes) == 2


def test_a_symlinked_vault_pointed_elsewhere_is_watched_at_its_new_target(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    link = tmp_path / "vault"
    link.symlink_to(first)
    scheduled = []

    class Recorder:  # the observer, reduced to what watch() asks of it
        emitters = (SimpleNamespace(is_alive=lambda: True),)

        def schedule(self, handler, path, **kwargs):
            scheduled.append(handler)

        def unschedule_all(self):
            pass

        start = stop = join = lambda self, *a, **k: None

    monkeypatch.setattr(watcher, "Observer", Recorder)
    runs = []
    stop, thread = _watch_in_thread(link, lambda: runs.append(1))
    try:
        assert wait_for(lambda: runs)
        link.unlink()
        link.symlink_to(second)  # the same path now names another directory
        assert wait_for(lambda: len(scheduled) == 2)
    finally:
        stop.set()
        thread.join(PATIENT)
    pokes = []
    scheduled[-1]._settle = SimpleNamespace(poke=lambda: pokes.append(1))
    scheduled[-1].on_any_event(_event("modified", str(second.resolve() / "a.md")))  # FSEvents reports resolved paths
    assert pokes == [1]


class FlakyObserver:
    """An observer whose watch can end, and whose next schedule() calls can fail, on demand."""

    def __init__(self):
        self.alive, self.failures, self.scheduled = True, 0, 0

    @property
    def emitters(self):
        return (SimpleNamespace(is_alive=lambda: self.alive),)

    def schedule(self, handler, path, **kwargs):
        if self.failures:
            self.failures -= 1
            raise FileNotFoundError(2, "No such file or directory")
        self.scheduled += 1
        self.alive = True

    def unschedule_all(self):
        pass

    start = stop = join = lambda self, *a, **k: None


def test_a_vault_made_again_under_the_same_inode_is_watched_again(tmp_path, monkeypatch, caplog):
    vault = tmp_path / "vault"
    vault.mkdir()
    observer = FlakyObserver()
    monkeypatch.setattr(watcher, "Observer", lambda: observer)
    runs = []
    with caplog.at_level(logging.WARNING, logger="puremind.watch"):
        stop, thread = _watch_in_thread(vault, lambda: runs.append(1))
        try:
            assert wait_for(lambda: runs)
            observer.alive = False  # what inotify does once the watched directory is deleted (ext4 then reuses the inode)
            assert wait_for(lambda: observer.scheduled == 2 and len(runs) == 2)
        finally:
            stop.set()
            thread.join(PATIENT)
    assert "was made again; watching it" in caplog.text


def test_a_vault_that_flaps_while_it_is_being_watched_again_does_not_end_watching(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    observer = FlakyObserver()
    monkeypatch.setattr(watcher, "Observer", lambda: observer)
    runs = []
    stop, thread = _watch_in_thread(vault, lambda: runs.append(1))
    try:
        assert wait_for(lambda: runs)
        observer.failures, observer.alive = 3, False  # gone again each time it is about to be watched
        assert wait_for(lambda: observer.scheduled == 2)
        assert thread.is_alive() and wait_for(lambda: len(runs) == 2)
    finally:
        stop.set()
        thread.join(PATIENT)
    assert not thread.is_alive()


def test_relevant_judges_names_alone_without_a_root_and_paths_with_one(tmp_path):
    assert relevant("/v/a.md") and relevant("/v/b.MARKDOWN")
    assert not relevant("/v/a.txt") and not relevant("/v/.hidden.md")
    assert relevant(str(tmp_path / "x" / "a.md"), tmp_path)
    assert not relevant(str(tmp_path / ".x" / "a.md"), tmp_path)
    assert relevant(str(tmp_path / "x"), tmp_path, directory=True)
    assert not relevant(str(tmp_path.parent / "a.md"), tmp_path)


def test_settle_waits_for_a_quiet_period_and_hands_out_each_change_once():
    now = [100.0]
    settle = Settle(2.0, clock=lambda: now[0])
    assert settle.remaining() is None and not settle.take()
    settle.poke()
    assert settle.remaining() == 2.0 and not settle.take()
    now[0] += 1.5
    settle.poke()  # still busy: the quiet period starts over
    now[0] += 1.5
    assert settle.remaining() == 0.5 and not settle.take()
    now[0] += 0.5
    assert settle.remaining() == 0.0 and settle.take()
    assert settle.remaining() is None and not settle.take()


# ---- Ctrl-C while the model is busy

CHILD = """
import sys, threading, time
import numpy as np
import puremind_mac.service as service
from puremind_mac import cli

calls = []
first_done = threading.Event()


class SlowModel:
    name = "slow-stub"

    def __init__(self, *args, **kwargs):
        pass

    def embed_documents(self, texts):
        calls.append(len(texts))
        if len(calls) >= {slow_from}:
            print("EMBEDDING", flush=True)
            time.sleep(120)  # a model that takes its time; the parent interrupts it
        first_done.set()
        return np.ones((len(texts), 4), dtype=np.float32)

    def embed_query(self, query):
        return np.ones(4, dtype=np.float32)


class Reranker:
    name = "stub-reranker"

    def __init__(self, *args, **kwargs):
        pass


def edit_a_note_later():
    first_done.wait(60)
    time.sleep(0.8)
    with open({note!r}, "a") as fh:
        fh.write("\\nA new line that makes the debounced run embed again.\\n")


service.SentenceTransformerEmbedder = SlowModel
service.CrossEncoderReranker = Reranker
if {slow_from} > 1:
    threading.Thread(target=edit_a_note_later, daemon=True).start()
sys.exit(cli.main(["watch", {vault!r}, "--debounce", "0.2"]))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT delivery differs on Windows")
@pytest.mark.parametrize("slow_from", [1, 2], ids=["first-run", "debounced-run"])
def test_ctrl_c_while_the_model_is_busy_exits_130_cleanly(tmp_path, vault, slow_from):
    script = tmp_path / "child.py"
    script.write_text(CHILD.format(vault=str(vault), note=str(vault / "projects" / "garden.md"), slow_from=slow_from))
    env = {k: v for k, v in os.environ.items() if not k.startswith("PUREMIND_")}
    env["PUREMIND_HOME"] = str(tmp_path / "home")
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()
    try:
        seen = ""
        deadline = time.monotonic() + 60
        while "EMBEDDING" not in seen and time.monotonic() < deadline:
            try:
                seen += lines.get(timeout=0.5)
            except queue.Empty:
                if proc.poll() is not None:
                    break
        assert "EMBEDDING" in seen, "the child never reached the slow model: " + proc.stderr.read()
        proc.send_signal(signal.SIGINT)
        code = proc.wait(timeout=30)
        stderr = proc.stderr.read()
    finally:
        if proc.poll() is None:
            proc.kill()
    assert code == 130, stderr
    assert "Traceback" not in stderr and "terminate called" not in stderr


def test_events_under_another_spelling_of_the_vault_still_count(tmp_path):
    vault = tmp_path / "Notes"
    vault.mkdir()
    pokes = []
    handler = watcher._Handler(tmp_path / "notes", SimpleNamespace(poke=lambda: pokes.append(1)))
    handler.on_any_event(_event("modified", str(vault / "a.md")))  # FSEvents reports the spelling on disk
    handler.on_any_event(_event("modified", str(vault / ".obsidian" / "b.md")))
    assert len(pokes) == 1
    assert relevant(str(tmp_path / "Café" / "a.md"), tmp_path / "Café")  # NFD on disk, NFC as typed


def test_a_system_limit_while_watching_again_ends_watching(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    observer = FlakyObserver()

    def schedule(handler, path, **kwargs):
        if observer.scheduled:
            raise OSError(errno.ENOSPC, "inotify watch limit reached")
        observer.scheduled += 1

    observer.schedule = schedule
    monkeypatch.setattr(watcher, "Observer", lambda: observer)
    failure = []

    def main():
        try:
            watch(vault, lambda: None, delay=DELAY)
        except OSError as exc:
            failure.append(exc)

    thread = threading.Thread(target=main, daemon=True)  # a regression must fail the test, not hang the run
    thread.start()
    try:
        assert wait_for(lambda: observer.scheduled == 1)
        observer.alive = False  # the watch ended; re-arming now hits the limit, which retrying cannot fix
        thread.join(PATIENT)
    finally:
        assert not thread.is_alive()
    assert failure and failure[0].errno == errno.ENOSPC
