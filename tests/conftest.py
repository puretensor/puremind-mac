from __future__ import annotations

import fcntl
import os
import site
import sys
import threading
from pathlib import Path

import pytest

# PUREMIND_TEST_INSTALLED=1 runs the suite against the installed package (the wheel job in CI) instead of the
# source tree. It is read here, once, because the autouse fixture below removes every PUREMIND_* variable.
TESTING_INSTALLED = os.environ.get("PUREMIND_TEST_INSTALLED") == "1"
SRC = Path(__file__).resolve().parents[1] / "src"
if not TESTING_INSTALLED:
    sys.path.insert(0, str(SRC))

from fakes import FakeEmbedder, FakeReranker

from puremind_mac.config import Config
from puremind_mac.indexer import run_lock_path
from puremind_mac.service import Service
from puremind_mac.store import Store

USER_BASE = site.getuserbase()  # where this interpreter finds user-site packages, before HOME is redirected


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def reranker():
    return FakeReranker()


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    (v / "projects").mkdir(parents=True)
    (v / "journal").mkdir()
    (v / "projects" / "espresso.md").write_text(
        "# Espresso machine\n\nThe grinder is set to 14 for the washed Ethiopian beans. "
        "Descale the boiler every three months.\n\n## Maintenance\n\nBackflush weekly with detergent.\n"
    )
    (v / "projects" / "garden.md").write_text(
        "# Garden plan\n\nPlant tomatoes in May. The raised beds need compost and mulch before spring.\n"
    )
    (v / "journal" / "2026-10-01.md").write_text(
        "# Journal\n\nDecided to migrate the build pipeline to hermetic containers next quarter.\n"
    )
    (v / ".obsidian").mkdir()
    (v / ".obsidian" / "workspace.md").write_text("# hidden\n\nshould never be indexed\n")
    (v / "notes.txt").write_text("not markdown")
    return v


@pytest.fixture
def cfg(tmp_path, vault):
    return Config(home=tmp_path / "home", vault=vault, db=tmp_path / "home" / "index.sqlite")


@pytest.fixture
def store(cfg):
    s = Store(cfg.db_path)
    yield s
    s.close()


@pytest.fixture
def service(cfg, store, embedder, reranker):
    return Service(cfg, store=store, embedder=embedder, reranker=reranker)


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch, tmp_path_factory):
    """Keep every test away from the developer's real settings, index and model cache.

    All PUREMIND_* variables are removed, the Hugging Face Hub is forced offline, and HOME and PUREMIND_HOME
    point at a throwaway directory. A test can then neither read ~/.puremind nor write to an index that
    PUREMIND_DB names in the shell running pytest. Tests that need a variable set it themselves.
    """
    for name in [n for n in os.environ if n.startswith("PUREMIND_")]:
        monkeypatch.delenv(name)
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PYTHONUSERBASE", USER_BASE)  # a child Python must still find the same packages
    if not TESTING_INSTALLED:  # and import the source tree under test, not some other installed copy
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH")])))
    monkeypatch.setenv("PUREMIND_HOME", str(home / ".puremind"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="session")
def testing_installed() -> bool:
    """True when the suite runs against an installed package (PUREMIND_TEST_INSTALLED=1)."""
    return TESTING_INSTALLED


class HeldRunLock:
    """A database's run lock, held on a descriptor of its own as another process's index run holds it."""

    def __init__(self, db_path):
        self.fd = os.open(run_lock_path(db_path), os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        self.timer: threading.Timer | None = None

    def release_in(self, seconds: float) -> None:
        self.timer = threading.Timer(seconds, fcntl.flock, (self.fd, fcntl.LOCK_UN))
        self.timer.start()

    def close(self) -> None:
        if self.timer is not None:
            self.timer.join(5)  # before the descriptor is closed, so a late unlock cannot reach a reused one
        os.close(self.fd)


@pytest.fixture
def another_run():
    """``another_run(db_path)`` holds that database's run lock until the test ends, or until ``release_in``."""
    held: list[HeldRunLock] = []

    def hold(db_path) -> HeldRunLock:
        held.append(HeldRunLock(db_path))
        return held[-1]

    yield hold
    for lock in held:
        lock.close()
