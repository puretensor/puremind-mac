"""SQLite store: files, chunks, FTS5 lexical index, and float16 vectors."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  mtime REAL NOT NULL,
  size INTEGER NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  indexed_at REAL NOT NULL
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


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- meta
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
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
    ) -> None:
        """Atomically replace one file's row, chunks and vectors.

        ``chunks`` is a sequence of (heading_path, content); ``vectors`` has one row per chunk.
        """
        if chunks and (vectors is None or len(vectors) != len(chunks)):
            raise ValueError("vectors must have one row per chunk")
        with self.db:
            self.db.execute("DELETE FROM files WHERE path = ?", (path,))
            self.db.execute(
                "INSERT INTO files(path, sha256, mtime, size, title, indexed_at) VALUES (?,?,?,?,?,?)",
                (path, sha256, mtime, size, title, time.time()),
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

    def delete_file(self, path: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM files WHERE path = ?", (path,))
            self._bump()

    def clear(self) -> None:
        with self.db:
            self.db.execute("DELETE FROM files")
            self._bump()

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
        mat = np.stack([np.frombuffer(r["vec"], dtype=np.float16) for r in rows]).astype(np.float32)
        return chunks, mat

    def fts_search(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        """FTS5 lexical search. Returns (chunk_id, bm25 score) with larger meaning better."""
        terms = re.findall(r"\w+", query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        rows = self.db.execute(
            "SELECT rowid, bm25(chunks_fts) AS s FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?",
            (match, limit),
        )
        return [(r["rowid"], -float(r["s"])) for r in rows]

    def get_chunks(self, ids: Sequence[int]) -> list[ChunkRow]:
        out = []
        for i in ids:
            r = self.db.execute(
                "SELECT id, path, chunk_index, heading_path, content FROM chunks WHERE id = ?", (i,)
            ).fetchone()
            if r:
                out.append(ChunkRow(**dict(r)))
        return out

    def stats(self) -> dict[str, int]:
        one = lambda q: self.db.execute(q).fetchone()[0]  # noqa: E731
        return {
            "files": one("SELECT COUNT(*) FROM files"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "vectors": one("SELECT COUNT(*) FROM vectors"),
        }
