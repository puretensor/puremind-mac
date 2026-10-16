"""Incremental vault indexing: SHA-256 per file, re-embed only what changed."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_markdown, fenced_lines
from .models import Embedder
from .store import Store

log = logging.getLogger("puremind.index")

MARKDOWN_SUFFIXES = (".md", ".markdown")
MAX_FILE_BYTES = 2 * 1024 * 1024
SKIP_DIRS = {"node_modules", "__pycache__"}


@dataclass
class IndexStats:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        s = (
            f"scanned {self.scanned}: {self.added} added, {self.updated} updated, "
            f"{self.unchanged} unchanged, {self.removed} removed ({self.chunks} chunks written)"
        )
        return s + (f", {len(self.skipped)} skipped" if self.skipped else "")


def discover(vault: Path) -> list[Path]:
    """Markdown files under ``vault``, skipping hidden directories and links that leave it."""
    root = vault.resolve()
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.startswith(".") or not name.lower().endswith(MARKDOWN_SUFFIXES):
                continue
            p = Path(dirpath, name)
            try:
                p.resolve().relative_to(root)
            except ValueError:
                continue
            found.append(p)
    return found


def title_of(text: str, fallback: str) -> str:
    lines = text.splitlines()
    inside = fenced_lines(lines)
    for i, line in enumerate(lines):
        if i not in inside:
            m = re.match(r"#\s+(.+)$", line.rstrip())
            if m:
                return m.group(1).strip()
    return fallback


def index_vault(vault: Path, store: Store, embedder: Embedder, rebuild: bool = False) -> IndexStats:
    vault = vault.expanduser().resolve()
    if not vault.is_dir():
        raise NotADirectoryError(f"vault directory not found: {vault}")
    stats = IndexStats()

    known_model = store.get_meta("embedder")
    if rebuild or (known_model is not None and known_model != embedder.name):
        if known_model and known_model != embedder.name:
            log.warning("embedder changed (%s -> %s); rebuilding the index", known_model, embedder.name)
        store.clear()
    known_root = store.get_meta("vault_root")
    if known_root and known_root != str(vault) and store.stats()["files"]:
        raise ValueError(
            f"this index belongs to {known_root}; use a different --db (or PUREMIND_DB) for {vault}"
        )
    store.set_meta("embedder", embedder.name)
    store.set_meta("vault_root", str(vault))

    old = store.file_hashes()
    seen: set[str] = set()
    for path in discover(vault):
        rel = path.relative_to(vault).as_posix()
        stats.scanned += 1
        try:
            st = path.stat()
            if st.st_size > MAX_FILE_BYTES:
                stats.skipped.append(f"{rel} (over {MAX_FILE_BYTES} bytes)")
                continue
            raw = path.read_bytes()
        except OSError as exc:
            stats.skipped.append(f"{rel} ({exc.strerror})")
            continue
        seen.add(rel)
        digest = hashlib.sha256(raw).hexdigest()
        if old.get(rel) == digest:
            stats.unchanged += 1
            continue
        text = raw.decode("utf-8", errors="replace")
        chunks = chunk_markdown(text)
        vectors = embedder.embed_documents([c.content for c in chunks]) if chunks else None
        store.replace_file(
            rel, digest, st.st_mtime, st.st_size, title_of(text, path.stem),
            [(c.heading_path, c.content) for c in chunks], vectors,
        )
        stats.chunks += len(chunks)
        if rel in old:
            stats.updated += 1
        else:
            stats.added += 1
    for rel in set(old) - seen:
        store.delete_file(rel)
        stats.removed += 1
    return stats
