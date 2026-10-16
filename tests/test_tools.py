"""MCP tool handlers and the server surface, with fake models."""

import asyncio
import os
import time

from puremind_mac import tools
from puremind_mac.indexer import index_vault
from puremind_mac.server import build_server
from puremind_mac.service import Service


def indexed(service, vault):
    index_vault(vault, service.store, service.embedder)
    return service


def test_server_exposes_exactly_three_read_only_tools(service):
    names = {t.name for t in asyncio.run(build_server(service).list_tools())}
    assert names == {"puremind_search", "puremind_get", "puremind_recent"}


def test_tool_schemas_and_defaults(service):
    tools_ = {t.name: t for t in asyncio.run(build_server(service).list_tools())}
    s = tools_["puremind_search"].inputSchema
    assert s["required"] == ["query"] and s["properties"]["limit"]["default"] == 10
    r = tools_["puremind_recent"].inputSchema["properties"]
    assert r["days"]["default"] == 7 and r["limit"]["default"] == 20
    assert tools_["puremind_get"].inputSchema["required"] == ["path"]


def test_server_calls_through_to_handlers(service, vault):
    indexed(service, vault)
    result = asyncio.run(build_server(service).call_tool("puremind_search", {"query": "espresso grinder"}))
    text = "".join(c.text for c in (result[0] if isinstance(result, tuple) else result))
    assert "projects/espresso.md" in text


def test_search_returns_path_heading_and_score(service, vault):
    indexed(service, vault)
    out = tools.puremind_search(service, "espresso grinder ethiopian", 3)
    assert "projects/espresso.md" in out
    assert "heading: Espresso machine" in out
    assert "score:" in out
    assert "Ethiopian" in out


def test_search_respects_limit_and_clamps(service, vault):
    indexed(service, vault)
    assert tools.puremind_search(service, "the", 1).count("score:") <= 1
    assert tools.puremind_search(service, "espresso", 0).count("score:") == 1
    assert tools.puremind_search(service, "espresso", 10_000).count("score:") <= tools.MAX_LIMIT


def test_search_uses_reranker_scores(service, vault, reranker):
    indexed(service, vault)
    tools.puremind_search(service, "espresso backflush")
    assert reranker.calls and reranker.calls[0][0] == "espresso backflush"


def test_search_empty_query_and_empty_index(service):
    assert tools.puremind_search(service, "  ").startswith("Error")
    assert "No results" in tools.puremind_search(service, "anything")


def test_search_sees_new_content_after_reindex(service, vault):
    indexed(service, vault)
    assert "zoo.md" not in tools.puremind_search(service, "zebra enclosure")
    (vault / "zoo.md").write_text("# Zoo\n\nThe zebra enclosure opens in spring.\n")
    index_vault(vault, service.store, service.embedder)
    assert "zoo.md" in tools.puremind_search(service, "zebra enclosure")


def test_search_reports_embedder_mismatch(service, vault):
    indexed(service, vault)
    service.embedder.name = "other"
    assert "puremind index --rebuild" in tools.puremind_search(service, "espresso")


def test_get_returns_file_text(service, vault):
    out = tools.puremind_get(service, "projects/garden.md")
    assert out.startswith("path: projects/garden.md")
    assert "Plant tomatoes in May" in out


def test_get_rejects_traversal_absolute_and_non_markdown(service, vault, tmp_path):
    (tmp_path / "outside.md").write_text("secret")
    assert "outside the vault" in tools.puremind_get(service, "../outside.md")
    assert tools.puremind_get(service, "/etc/passwd").startswith("Error")
    assert tools.puremind_get(service, "~/x.md").startswith("Error")
    assert tools.puremind_get(service, "notes.txt").startswith("Error")
    assert tools.puremind_get(service, "missing.md").startswith("Error")
    assert tools.puremind_get(service, "").startswith("Error")


def test_get_rejects_symlink_escape(service, vault, tmp_path):
    (tmp_path / "outside.md").write_text("secret")
    (vault / "sneaky.md").symlink_to(tmp_path / "outside.md")
    assert tools.puremind_get(service, "sneaky.md").startswith("Error")


def test_get_truncates_huge_files(service, vault, monkeypatch):
    monkeypatch.setattr(tools, "MAX_GET_CHARS", 30)
    (vault / "big.md").write_text("x" * 500)
    assert "[truncated at 30 characters]" in tools.puremind_get(service, "big.md")


def test_get_without_a_vault(cfg, store, embedder, reranker):
    import dataclasses

    svc = Service(dataclasses.replace(cfg, vault=None), store=store, embedder=embedder, reranker=reranker)
    assert "puremind init" in tools.puremind_get(svc, "a.md")


def test_recent_lists_newest_first_within_window(service, vault):
    now = time.time()
    os.utime(vault / "projects" / "garden.md", (now - 40 * 86400,) * 2)
    os.utime(vault / "projects" / "espresso.md", (now - 3600,) * 2)
    os.utime(vault / "journal" / "2026-10-01.md", (now - 86400,) * 2)
    indexed(service, vault)
    out = tools.puremind_recent(service, days=7)
    assert out.index("projects/espresso.md") < out.index("journal/2026-10-01.md")
    assert "garden.md" not in out
    assert "Espresso machine" in out


def test_recent_limit_and_empty(service, vault):
    indexed(service, vault)
    assert tools.puremind_recent(service, 7, 1).count("\n- ") == 1

def test_recent_empty_window(service, vault):
    old = (1, 1)
    for p in vault.rglob("*.md"):
        os.utime(p, old)
    indexed(service, vault)
    assert "No indexed notes" in tools.puremind_recent(service, 1, 20)


def test_tools_never_modify_the_vault(service, vault):
    before = {p: p.read_bytes() for p in vault.rglob("*.md")}
    indexed(service, vault)
    tools.puremind_search(service, "espresso")
    tools.puremind_get(service, "projects/garden.md")
    tools.puremind_recent(service)
    assert before == {p: p.read_bytes() for p in vault.rglob("*.md")}
