"""SQLite store: files, chunks, FTS5 lexical index, and float16 vectors."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import stat
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

log = logging.getLogger("puremind.store")

SCHEMA_VERSION = 1  # stamped in PRAGMA user_version; a database with a higher number is refused
BUSY_TIMEOUT_MS = 30_000  # how long a statement waits for another connection's lock
CHECKPOINT_WAIT_MS = 1_000  # how long emptying the -wal file waits for readers; writers wait meanwhile

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  mtime REAL NOT NULL,
  size INTEGER NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  indexed_at REAL NOT NULL,
  chunker INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  heading_path TEXT NOT NULL,
  content TEXT NOT NULL,
  UNIQUE (path, chunk_index)
);
CREATE TABLE IF NOT EXISTS vectors (
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  content, heading_path, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, content, heading_path) VALUES (new.id, new.content, new.heading_path);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, content, heading_path)
  VALUES ('delete', old.id, old.content, old.heading_path);
END;
"""


@dataclass(frozen=True)
class ChunkRow:
    id: int
    path: str
    chunk_index: int
    heading_path: str
    content: str


@dataclass(frozen=True)
class FileRow:
    path: str
    sha256: str
    mtime: float
    size: int
    title: str


class StoreError(RuntimeError):
    """The index cannot be used as asked: it is read-only, or was written by a newer version."""


class _Result:
    """The rows of one statement, fetched while the connection lock was held.

    Offers the parts of a cursor the store uses: fetchone, fetchmany, fetchall, iteration,
    lastrowid, rowcount and description.
    """

    __slots__ = ("_pos", "_rows", "description", "lastrowid", "rowcount")

    def __init__(self, cur: sqlite3.Cursor):
        self._rows = cur.fetchall()
        self._pos = 0
        self.lastrowid = cur.lastrowid
        self.rowcount = cur.rowcount
        self.description = cur.description

    def fetchone(self):
        if self._pos >= len(self._rows):
            return None
        self._pos += 1
        return self._rows[self._pos - 1]

    def fetchmany(self, size: int = 1) -> list:
        rows = self._rows[self._pos : self._pos + size]
        self._pos += len(rows)
        return rows

    def fetchall(self) -> list:
        rows = self._rows if self._pos == 0 else self._rows[self._pos :]
        self._pos = len(self._rows)
        return rows

    def __iter__(self):
        return iter(self.fetchall())


class _LockedConnection:
    """One sqlite3 connection shared by several threads (MCP worker threads, a watcher).

    Guaranteed:
    * Each statement runs under one re-entrant lock and its rows are fetched before the lock is
      released. ``execute`` returns that finished result, never a live cursor, so no cursor is
      stepped by one thread while another thread runs statements on the connection.
    * ``with conn:`` holds the lock for the whole block and commits, or rolls back on an error,
      so a transaction is never interleaved with another thread's statements.

    Not guaranteed: statements outside one ``with`` block do not form a snapshot. Another thread
    or process may commit between them, so a read that must be consistent is a single statement.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.RLock()

    def execute(self, *a, **k) -> _Result:
        with self._lock:
            return _Result(self._conn.execute(*a, **k))

    def executemany(self, *a, **k) -> _Result:
        with self._lock:
            return _Result(self._conn.executemany(*a, **k))

    def executescript(self, *a, **k) -> _Result:
        with self._lock:
            return _Result(self._conn.executescript(*a, **k))

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self):
        self._lock.acquire()
        self._conn.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            return self._conn.__exit__(*exc)
        finally:
            self._lock.release()

    @property
    def row_factory(self):
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._conn.row_factory = value


# SQLite result codes that mean "this database cannot be written here".
_UNWRITABLE = (sqlite3.SQLITE_READONLY, sqlite3.SQLITE_CANTOPEN)


def _prepare_files(path: str) -> None:
    """Keep the index private: it holds the full text of every note.

    A missing parent directory is created with mode 0700 and a missing database file with 0600;
    SQLite gives the -wal and -shm files the mode of the main file. An existing database, -wal or
    -shm owned by this user loses its group and other permission bits.
    """
    Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "posix":
        return
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except OSError:  # already there, or the directory is not writable (then it is opened read-only)
        pass
    for p in (path, path + "-wal", path + "-shm"):
        try:
            st = os.stat(p)
            if stat.S_ISREG(st.st_mode) and st.st_uid == os.geteuid() and st.st_mode & 0o077:
                os.chmod(p, stat.S_IMODE(st.st_mode) & ~0o077)
                log.info("removed group and other access from %s", p)
        except OSError:
            pass


def _check_version(conn: sqlite3.Connection, path: str) -> int:
    """The database's schema version; refuses one written by a newer puremind."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise StoreError(
            f"the index at {path} uses format {version}, newer than this puremind understands "
            f"({SCHEMA_VERSION}); upgrade puremind, or point PUREMIND_DB (or db = in config.toml) at another file"
        )
    return version


def _has_chunker_column(conn: sqlite3.Connection) -> bool:
    return "chunker" in {row[1] for row in conn.execute("PRAGMA table_info(files)")}


def _checkpoint(conn) -> bool:
    """Copy the -wal file into the database and empty it; False if readers kept it from being emptied.

    The checkpoint holds the write lock while it waits for readers, so it waits only briefly.
    """
    conn.execute(f"PRAGMA busy_timeout = {CHECKPOINT_WAIT_MS}")
    try:
        return conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    finally:
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")


def _upgrade(conn: sqlite3.Connection, path: str) -> None:
    """Purge what an index written by puremind 0.1.1 or earlier still holds of deleted notes.

    Those versions left a deleted note's terms in the keyword index and its text in pages that
    SQLite freed without zeroing. ``optimize`` drops the terms and VACUUM rewrites the file
    without free pages. A failure (a full disk, a lock held elsewhere) does not stop the upgrade.
    """
    try:
        with conn:
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        conn.execute("VACUUM")
        _checkpoint(conn)
    except sqlite3.OperationalError as exc:
        log.warning("could not compact %s (%s); text deleted by an earlier version may remain in it", path, exc)


def _fts_match(query: str) -> str | None:
    terms = re.findall(r"\w+", query)
    return " OR ".join(f'"{t}"' for t in terms) if terms else None


class Store:
    """The index: SQLite file with files, chunks, FTS5 and vectors. Shareable between threads.

    The files stay private to their owner (see ``_prepare_files``). ``read_only`` is True when the
    database cannot be written (its file or directory is not writable, or it is on a read-only
    volume). It is then opened read-only, without touching its schema or settings, and every write
    method raises StoreError.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.read_only = False
        memory = self.path == ":memory:"
        if not memory:
            _prepare_files(self.path)
        conn = None
        # A file this user cannot write goes straight to read-only: SQLite would open it that way silently.
        if memory or not os.path.exists(self.path) or os.access(self.path, os.W_OK):
            try:
                conn = self._open_writable()
            except sqlite3.OperationalError as exc:
                if memory or exc.sqlite_errorcode & 0xFF not in _UNWRITABLE:
                    raise
                log.info("cannot open %s for writing (%s); trying read-only", self.path, exc)
        if conn is None:
            conn = self._open_read_only()
            self.read_only = True
        self.db = _LockedConnection(conn)
        self.db.row_factory = sqlite3.Row

    def _open_writable(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_MS / 1000, check_same_thread=False)
        try:
            version = _check_version(conn, self.path)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA secure_delete = ON")  # freed pages are zeroed, so deleted text does not linger
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL").fetchone()  # fetched so the statement is finished
            conn.executescript(SCHEMA)
            if not _has_chunker_column(conn):  # 0.1.1 and earlier
                try:
                    conn.execute("ALTER TABLE files ADD COLUMN chunker INTEGER NOT NULL DEFAULT 0")
                except sqlite3.OperationalError:
                    if not _has_chunker_column(conn):  # else another process added it meanwhile
                        raise
            if version != SCHEMA_VERSION:
                if self.path != ":memory:":
                    _upgrade(conn, self.path)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except BaseException:
            conn.close()
            raise
        return conn

    def _open_read_only(self) -> sqlite3.Connection:
        uri = Path(self.path).absolute().as_uri()
        wal = self.path + "-wal"
        # SQLite has to create the -shm file to read a WAL database, which a read-only directory
        # forbids. Such a database can still be read as immutable if no writer left changes in
        # its -wal file; it then shows the file as it is now and misses later changes.
        pending = os.path.exists(wal) and os.path.getsize(wal) > 0
        error: sqlite3.OperationalError | None = None
        for query in ("mode=ro",) if pending else ("mode=ro", "mode=ro&immutable=1"):
            try:
                conn = self._connect_read_only(f"{uri}?{query}")
            except sqlite3.OperationalError as exc:
                if exc.sqlite_errorcode & 0xFF not in _UNWRITABLE:
                    raise
                error = exc
                continue
            if "immutable" in query:
                log.warning("%s is read-only here, opened as immutable: later changes will not be seen", self.path)
            return conn
        if not os.path.exists(self.path):
            raise StoreError(f"no index at {self.path}, and it cannot be created there (the directory is not writable)")
        raise StoreError(
            f"cannot open the index at {self.path} read-only: {error}"
            + (" (it has changes waiting in its -wal file; open it once where it can be written)" if pending else "")
        )

    def _connect_read_only(self, uri: str) -> sqlite3.Connection:
        conn = sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT_MS / 1000, check_same_thread=False)
        try:
            _check_version(conn, self.path)
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'meta'").fetchone() is None:
                raise StoreError(f"{self.path} holds no puremind index and is read-only, so one cannot be created")
            conn.execute("PRAGMA foreign_keys = ON")
        except BaseException:
            conn.close()
            raise
        return conn

    def _check_writable(self) -> None:
        if self.read_only:
            raise StoreError(
                f"cannot change the index at {self.path}: it is read-only "
                "(the file or its directory is not writable by this user, or it is on a read-only volume)"
            )

    def close(self) -> None:
        self.db.close()

    def checkpoint(self) -> bool:
        """Copy the -wal file into the database and empty it, so text that was just deleted or
        replaced is not left behind there.

        Waits at most CHECKPOINT_WAIT_MS for other connections that are reading. Returns False
        if they kept the file from being emptied; a later checkpoint empties it.
        """
        if self.read_only or self.path == ":memory:":
            return True
        with self.db:  # holds the lock, so no other thread runs a statement under the short timeout
            return _checkpoint(self.db)

    def purge(self) -> bool:
        """Remove what deleted and replaced chunks left in the file, then empty the -wal file.

        FTS5 only marks a deleted row's terms as deleted and drops them when it merges the parts
        of its index, so the parts are merged now; secure_delete zeroes the pages this frees.
        Returns what ``checkpoint`` returns. Clearing the debt afterwards may leave metadata
        in the -wal file; a concurrent write keeps the debt for the next purge.
        """
        self._check_writable()
        with self.db:
            self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
            generation = self.generation()
        if not self.checkpoint():
            return False
        with self.db:
            # A writer after compaction owes another purge, even if the checkpoint included its pages.
            self.db.execute(
                "DELETE FROM meta WHERE key = 'purge_pending' "
                "AND COALESCE((SELECT value FROM meta WHERE key = 'generation'), '0') = ?",
                (str(generation),),
            )
        return True

    def _owe_purge(self) -> None:
        self.db.execute("INSERT INTO meta(key, value) VALUES ('purge_pending', '1') ON CONFLICT(key) DO NOTHING")

    def defer_purge(self) -> bool:
        """Note in the index that ``purge`` is still owed, so that a process that stops before
        paying it (a crash, a power cut) leaves the debt to the next one, and empty the -wal file.
        The deleted chunks' text is already gone from the tables; only the keyword index still
        holds their words. Returns what ``checkpoint`` returns."""
        self._check_writable()
        with self.db:
            self._owe_purge()
        return self.checkpoint()

    def purge_pending(self) -> bool:
        return self.get_meta("purge_pending") is not None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- meta
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self._check_writable()
        with self.db:
            self.db.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def generation(self) -> int:
        """Counter bumped by every write, so long-lived readers know to reload."""
        return int(self.get_meta("generation", "0") or 0)

    def _bump(self) -> None:
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES ('generation', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
        )

    # ---- files
    def file_hashes(self) -> dict[str, str]:
        return {r["path"]: r["sha256"] for r in self.db.execute("SELECT path, sha256 FROM files")}

    def chunked_with(self, version: int) -> set[str]:
        """Paths of the files whose chunks were cut by chunker ``version`` (0: by puremind 0.1.1 or earlier)."""
        return {r["path"] for r in self.db.execute("SELECT path FROM files WHERE chunker = ?", (version,))}

    def list_files(self) -> list[FileRow]:
        rows = self.db.execute("SELECT path, sha256, mtime, size, title FROM files ORDER BY path")
        return [FileRow(**dict(r)) for r in rows]

    def replace_file(
        self,
        path: str,
        sha256: str,
        mtime: float,
        size: int,
        title: str,
        chunks: Sequence[tuple[str, str]],
        vectors: np.ndarray | None,
        chunker: int = 0,
    ) -> None:
        """Atomically replace one file's row, chunks and vectors.

        ``chunks`` is a sequence of (heading_path, content); ``vectors`` has one row per chunk.
        ``chunker`` is the version of the chunker that cut them.
        """
        self._check_writable()
        if chunks and (vectors is None or len(vectors) != len(chunks)):
            raise ValueError("vectors must have one row per chunk")
        with self.db:
            if self.db.execute("DELETE FROM files WHERE path = ?", (path,)).rowcount:
                self._owe_purge()
            self.db.execute(
                "INSERT INTO files(path, sha256, mtime, size, title, indexed_at, chunker) VALUES (?,?,?,?,?,?,?)",
                (path, sha256, mtime, size, title, time.time(), chunker),
            )
            for i, (heading, content) in enumerate(chunks):
                cur = self.db.execute(
                    "INSERT INTO chunks(path, chunk_index, heading_path, content) VALUES (?,?,?,?)",
                    (path, i, heading, content),
                )
                vec = np.asarray(vectors[i], dtype=np.float16)
                self.db.execute(
                    "INSERT INTO vectors(chunk_id, dim, vec) VALUES (?,?,?)",
                    (cur.lastrowid, vec.shape[0], vec.tobytes()),
                )
            self._bump()

    def update_file_metadata(self, path: str, sha256: str, mtime: float, size: int) -> None:
        """Refresh an unchanged note's stat only while its indexed content still matches."""
        self._check_writable()
        with self.db:
            self.db.execute(
                "UPDATE files SET mtime = ?, size = ? WHERE path = ? AND sha256 = ?",
                (mtime, size, path, sha256),
            )

    def delete_file(self, path: str) -> None:
        self._check_writable()
        with self.db:
            if self.db.execute("DELETE FROM files WHERE path = ?", (path,)).rowcount:
                self._owe_purge()
            self._bump()

    def clear(self, *, vault_root: str | None = None) -> None:
        """Remove every note and adopt ``vault_root`` in one transaction, then purge their text."""
        self._check_writable()
        with self.db:
            self.db.execute("DELETE FROM files")
            self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")  # empties the keyword index outright
            self._owe_purge()
            if vault_root is not None:
                self.db.execute(
                    "INSERT INTO meta(key, value) VALUES ('vault_root', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (vault_root,),
                )
            self._bump()
        if self.path != ":memory:":
            with self.db:  # no transaction is open here, which VACUUM needs
                self.db.execute("VACUUM")  # also drops pages an earlier version freed without zeroing
        self.purge()

    def recent(self, since: float, limit: int) -> list[FileRow]:
        rows = self.db.execute(
            "SELECT path, sha256, mtime, size, title FROM files WHERE mtime >= ? "
            "ORDER BY mtime DESC, path LIMIT ?",
            (since, limit),
        )
        return [FileRow(**dict(r)) for r in rows]

    # ---- chunks and vectors
    def load_index(self) -> tuple[list[ChunkRow], np.ndarray]:
        """All chunks (ordered by id) and their vectors as an (n, dim) float32 matrix."""
        rows = self.db.execute(
            "SELECT c.id, c.path, c.chunk_index, c.heading_path, c.content, v.dim, v.vec "
            "FROM chunks c JOIN vectors v ON v.chunk_id = c.id ORDER BY c.id"
        ).fetchall()
        chunks = [ChunkRow(r["id"], r["path"], r["chunk_index"], r["heading_path"], r["content"]) for r in rows]
        if not rows:
            return chunks, np.zeros((0, 0), dtype=np.float32)
        dims = {r["dim"] for r in rows}
        if len(dims) != 1 or any(len(r["vec"]) != 2 * r["dim"] for r in rows):
            raise ValueError("the vectors in the index do not all have the same length; run `puremind index --rebuild`")
        mat = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float16).reshape(len(rows), dims.pop())
        return chunks, mat.astype(np.float32)

    def fts_search(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        """FTS5 lexical search. Returns (chunk_id, bm25 score) with larger meaning better."""
        return [(chunk.id, score) for chunk, score in self.fts_hits(query, limit)]

    def fts_hits(self, query: str, limit: int = 50) -> list[tuple[ChunkRow, float]]:
        """Like ``fts_search`` but returns the chunk rows, read by the same statement as the match."""
        match = _fts_match(query)
        if match is None:
            return []
        rows = self.db.execute(
            "SELECT c.id, c.path, c.chunk_index, c.heading_path, c.content, bm25(chunks_fts) AS s "
            "FROM chunks_fts JOIN chunks AS c ON c.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?",
            (match, limit),
        )
        return [
            (ChunkRow(r["id"], r["path"], r["chunk_index"], r["heading_path"], r["content"]), -float(r["s"]))
            for r in rows
        ]

    def get_chunks(self, ids: Sequence[int]) -> list[ChunkRow]:
        """The chunks with these ids, in the order given; ids that no longer exist are skipped."""
        ids = list(ids)
        found: dict[int, ChunkRow] = {}
        for start in range(0, len(ids), 500):  # stay under SQLite's variable limit
            batch = ids[start : start + 500]
            rows = self.db.execute(
                "SELECT id, path, chunk_index, heading_path, content FROM chunks "
                f"WHERE id IN ({','.join('?' * len(batch))})",
                batch,
            )
            found.update((r["id"], ChunkRow(**dict(r))) for r in rows)
        return [found[i] for i in ids if i in found]

    def stats(self) -> dict[str, int]:
        row = self.db.execute(  # one statement, so the three counts come from one snapshot
            "SELECT (SELECT COUNT(*) FROM files), (SELECT COUNT(*) FROM chunks), (SELECT COUNT(*) FROM vectors)"
        ).fetchone()
        return {"files": row[0], "chunks": row[1], "vectors": row[2]}
