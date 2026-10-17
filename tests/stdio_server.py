"""Run the real stdio MCP server over an index built with the fake models.

Used by test_server_stdio.py as ``python stdio_server.py <home> <vault> [--no-rerank]``. The
index is built here, on first start, so the parent test needs no models and no shared state.
``python stdio_server.py --where`` prints which puremind_mac it imports and exits. The package
comes from the environment the test gives it: the source tree, or the installed copy under test.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # the fakes

from fakes import FakeEmbedder, FakeReranker

import puremind_mac
from puremind_mac.config import Config
from puremind_mac.indexer import index_vault
from puremind_mac.server import build_server
from puremind_mac.service import Service
from puremind_mac.store import Store


def main(argv: list[str]) -> None:
    if argv == ["--where"]:
        print(puremind_mac.__file__)
        return
    home, vault = Path(argv[0]), Path(argv[1])
    cfg = Config(home=home, vault=vault, db=home / "index.sqlite")
    if "--no-rerank" in argv:
        cfg = dataclasses.replace(cfg, rerank=False)
    store = Store(cfg.db_path)
    embedder = FakeEmbedder()
    index_vault(vault, store, embedder)
    build_server(Service(cfg, store=store, embedder=embedder, reranker=FakeReranker())).run(transport="stdio")


if __name__ == "__main__":
    main(sys.argv[1:])
