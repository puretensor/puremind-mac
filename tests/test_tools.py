"""MCP tool handlers and the server surface, with fake models."""

import asyncio
import dataclasses
import os
import re
import time

import pytest

from puremind_mac import tools
from puremind_mac.indexer import MAX_FILE_BYTES, index_vault
from puremind_mac.models import ModelLoadError
from puremind_mac.server import ToolError, build_server
from puremind_mac.service import Service
from puremind_mac.tools import ToolRefused


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    for name in [n for n in os.environ if n.startswith("PUREMIND_")]:
        monkeypatch.delenv(name)


def indexed(service, vault):
    index_vault(vault, service.store, service.embedder)
    return service


def refusal(fn, *args) -> str:
    """The message of the ToolRefused that ``fn(*args)`` raises."""
    with pytest.raises(ToolRefused) as excinfo:
        fn(*args)
    return str(excinfo.value)


def _result_text(result) -> str:
    """call_tool returns a list of content blocks (mcp 1.x), a (blocks, structured)
    tuple, or a CallToolResult with .content (mcp 2.x)."""
    blocks = getattr(result, "content", None)
    if blocks is None:
        blocks = result[0] if isinstance(result, tuple) else result
    return "".join(getattr(c, "text", "") for c in blocks)


def _schema(tool):
    """mcp 1.x exposes inputSchema, mcp 2.x input_schema."""
    return getattr(tool, "inputSchema", None) or tool.input_schema


def _hint(tool, camel):
    """A tool annotation by its camelCase name (mcp 1.x) or its snake_case name (mcp 2.x)."""
    snake = re.sub("([A-Z])", lambda m: "_" + m.group(1).lower(), camel)
    return getattr(tool.annotations, camel) if hasattr(tool.annotations, camel) else getattr(tool.annotations, snake)


def _listed(service):
    return {t.name: t for t in asyncio.run(build_server(service).list_tools())}


# ---- the server surface


READ_TOOLS = {"puremind_search", "puremind_get", "puremind_recent"}


def test_server_exposes_three_read_only_tools_and_the_proposal_tool(service):
    assert set(_listed(service)) == READ_TOOLS | {"puremind_propose"}


def test_the_proposal_tool_can_be_switched_off(service):
    off = Service(dataclasses.replace(service.cfg, inbox_enabled=False), service.store, service.embedder)
    assert set(_listed(off)) == READ_TOOLS
    assert "propos" not in build_server(off).instructions


def test_read_tools_are_annotated_read_only(service):
    for name in READ_TOOLS:
        tool = _listed(service)[name]
        assert _hint(tool, "readOnlyHint") is True
        assert _hint(tool, "destructiveHint") is False
        assert _hint(tool, "idempotentHint") is True
        assert _hint(tool, "openWorldHint") is False


def test_the_proposal_tool_is_annotated_as_writing_only_to_the_inbox(service):
    tool = _listed(service)["puremind_propose"]
    assert _hint(tool, "readOnlyHint") is False
    assert _hint(tool, "destructiveHint") is False
    assert _hint(tool, "openWorldHint") is False
    schema = _schema(tool)
    assert schema["required"] == ["title", "content"]
    assert schema["properties"]["title"]["maxLength"] == 200
    assert schema["properties"]["content"]["maxLength"] == 20_000
    assert "does not write to their notes" in tool.description


def test_tool_schemas_defaults_and_bounds(service):
    tools_ = _listed(service)
    s = _schema(tools_["puremind_search"])
    assert s["required"] == ["query"] and s["properties"]["limit"]["default"] == 10
    assert (s["properties"]["limit"]["minimum"], s["properties"]["limit"]["maximum"]) == (1, tools.MAX_LIMIT)
    r = _schema(tools_["puremind_recent"])["properties"]
    assert r["days"]["default"] == 7 and r["limit"]["default"] == 20
    assert (r["days"]["minimum"], r["days"]["maximum"]) == (1, tools.MAX_RECENT_DAYS)
    assert (r["limit"]["minimum"], r["limit"]["maximum"]) == (1, tools.MAX_RECENT_LIMIT)
    g = _schema(tools_["puremind_get"])
    assert g["required"] == ["path"] and g["properties"]["path"]["maxLength"] == tools.MAX_PATH_CHARS


def test_tools_return_text_only_without_a_duplicate_structured_payload(service):
    for tool in _listed(service).values():
        assert not (getattr(tool, "outputSchema", None) or getattr(tool, "output_schema", None))


def test_descriptions_explain_the_score_the_caps_and_the_truncation(service):
    listed = _listed(service)
    search = listed["puremind_search"].description
    assert "reranker probability from 0 to 1" in search and "no relevance cut-off" in search
    assert f"At most {tools.MAX_LIMIT}" in search
    assert f"{tools.MAX_GET_CHARS:,} characters" in listed["puremind_get"].description
    no_rerank = _listed(Service(dataclasses.replace(service.cfg, rerank=False), service.store, service.embedder))
    assert "reciprocal rank fusion" in no_rerank["puremind_search"].description


def test_server_carries_instructions(service):
    text = build_server(service).instructions
    assert "read-only" in text
    assert "nothing becomes a note until the user approves it" in text


def test_server_calls_through_to_handlers(service, vault):
    indexed(service, vault)
    result = asyncio.run(build_server(service).call_tool("puremind_search", {"query": "espresso grinder"}))
    assert "projects/espresso.md" in _result_text(result)


def test_server_turns_a_refusal_into_a_tool_error(service, vault):
    indexed(service, vault)
    mcp = build_server(service)
    with pytest.raises(ToolError, match="query is empty"):
        asyncio.run(mcp.call_tool("puremind_search", {"query": "   "}))
    with pytest.raises(ToolError, match="hidden"):
        asyncio.run(mcp.call_tool("puremind_get", {"path": ".obsidian/workspace.md"}))
    with pytest.raises(ToolError):  # out of the declared bounds: rejected before the handler
        asyncio.run(mcp.call_tool("puremind_search", {"query": "espresso", "limit": 500}))


# ---- search


def test_search_returns_path_heading_and_score(service, vault):
    indexed(service, vault)
    out = tools.puremind_search(service, "espresso grinder ethiopian", 3)
    assert "projects/espresso.md" in out
    assert "heading: Espresso machine" in out
    assert "score:" in out
    assert "Ethiopian" in out


def test_search_names_the_score_scale(service, vault):
    indexed(service, vault)
    assert "reranker probability from 0 to 1" in tools.puremind_search(service, "espresso")
    plain = Service(dataclasses.replace(service.cfg, rerank=False), service.store, service.embedder, service.reranker)
    assert "reciprocal rank fusion" in tools.puremind_search(plain, "espresso")


def test_search_prints_small_scores_distinguishably(service, vault):
    class Tiny:
        name = "tiny"

        def score(self, query, passages):
            return [1e-3 / (i + 1) ** 3 for i in range(len(passages))]

    indexed(service, vault)
    svc = Service(service.cfg, service.store, service.embedder, Tiny())
    scores = [line.split("score: ")[1] for line in tools.puremind_search(svc, "the", 3).splitlines() if "score: " in line]
    assert scores == ["0.001", "0.000125", "3.7e-05"]
    assert tools.format_score(0.9876) == "0.988" and tools.format_score(13.2149) == "13.2"


def test_search_respects_limit_and_clamps(service, vault):
    indexed(service, vault)
    assert tools.puremind_search(service, "the", 1).count("score:") <= 1
    assert tools.puremind_search(service, "espresso", 0).count("score:") == 1
    assert tools.puremind_search(service, "espresso", 10_000).count("score:") <= tools.MAX_LIMIT


def test_search_uses_reranker_scores(service, vault, reranker):
    indexed(service, vault)
    tools.puremind_search(service, "espresso backflush")
    assert reranker.calls and reranker.calls[0][0] == "espresso backflush"


def test_search_skips_the_reranker_when_it_is_switched_off(service, vault, reranker):
    indexed(service, vault)
    svc = Service(dataclasses.replace(service.cfg, rerank=False), service.store, service.embedder, reranker)
    assert "projects/espresso.md" in tools.puremind_search(svc, "espresso backflush")
    assert reranker.calls == []


def test_search_refuses_an_empty_query(service):
    assert "query is empty" in refusal(tools.puremind_search, service, "  ")
    assert "query is empty" in refusal(tools.puremind_search, service, None)


def test_search_on_an_empty_index_is_not_an_error(service):
    assert "No results" in tools.puremind_search(service, "anything")


def test_search_sees_new_content_after_reindex(service, vault):
    indexed(service, vault)
    assert "zoo.md" not in tools.puremind_search(service, "zebra enclosure")
    (vault / "zoo.md").write_text("# Zoo\n\nThe zebra enclosure opens in spring.\n")
    index_vault(vault, service.store, service.embedder)
    assert "zoo.md" in tools.puremind_search(service, "zebra enclosure")


def test_search_refuses_when_the_embedder_changed(service, vault):
    indexed(service, vault)
    service.embedder.name = "other"
    assert "puremind index --rebuild" in refusal(tools.puremind_search, service, "espresso")


def test_search_refuses_with_a_short_message_when_the_model_cannot_be_loaded(service, vault, tmp_path):
    indexed(service, vault)

    def fail(query):
        raise ModelLoadError("model m is not downloaded yet")

    service.embedder.embed_query = fail
    message = refusal(tools.puremind_search, service, "espresso")
    assert message == "model m is not downloaded yet" and str(tmp_path) not in message


# ---- get


def test_get_returns_file_text(service, vault):
    out = tools.puremind_get(service, "projects/garden.md")
    assert out.startswith("path: projects/garden.md")
    assert "Plant tomatoes in May" in out


@pytest.fixture
def hostile_vault(vault, tmp_path):
    """The conftest vault plus the awkward entries a real vault can hold."""
    (vault / "projects" / ".draft.md").write_text("# draft\n\nprivate\n")
    (vault / ".private").mkdir()
    (vault / ".private" / "diary.md").write_text("# diary\n\nprivate\n")
    (vault / "node_modules" / "pkg").mkdir(parents=True)
    (vault / "node_modules" / "pkg" / "readme.md").write_text("# pkg\n")
    (vault / "Node_Modules").mkdir(exist_ok=True)
    (vault / "Node_Modules" / "x.md").write_text("# x\n")
    (tmp_path / "outside.md").write_text("secret")
    (vault / "projects" / "dir.md").mkdir()
    return vault


LEXICAL_REFUSALS = [
    "",
    "   ",
    ".obsidian/workspace.md",  # a hidden directory
    ".private/diary.md",
    "projects/.draft.md",  # a dotfile
    "node_modules/pkg/readme.md",
    "Node_Modules/x.md",  # the skip list is case-insensitive, as on APFS
    "../outside.md",
    "projects/../../outside.md",
    "projects/../projects/garden.md",  # ".." is refused even when it stays inside the vault
    "/etc/passwd",
    "~/x.md",
    "notes.txt",
    "missing.md",
    "projects",
    "projects/dir.md",
    "a\x00b.md",
    "projects/" + "a/" * 600 + "x.md",
    "a" * 300 + ".md",  # one component longer than the file system allows
    "bad\ud800name.md",  # a lone surrogate, which the file system layer cannot encode
]


@pytest.mark.parametrize("path", LEXICAL_REFUSALS, ids=lambda p: repr(p[:40]))
def test_get_refuses_what_the_indexer_would_not_serve(service, hostile_vault, tmp_path, path):
    message = refusal(tools.puremind_get, service, path)
    assert len(message) < 250
    assert str(tmp_path) not in message and str(hostile_vault) not in message


def test_get_refuses_files_that_resolve_to_something_else(service, hostile_vault, tmp_path):
    (hostile_vault / "sneaky.md").symlink_to(tmp_path / "outside.md")  # leaves the vault
    (hostile_vault / "to_hidden.md").symlink_to(hostile_vault / ".private" / "diary.md")
    (hostile_vault / "to_text.md").symlink_to(hostile_vault / "notes.txt")
    (hostile_vault / "loop.md").symlink_to("loop.md")
    (hostile_vault / "a.md").symlink_to("b.md")
    (hostile_vault / "b.md").symlink_to("a.md")
    (hostile_vault / "shortcut").symlink_to(hostile_vault / ".private")  # a visible name for a hidden folder
    (hostile_vault / "outside").symlink_to(tmp_path)  # and one for a folder beyond the vault
    os.mkfifo(hostile_vault / "pipe.md")
    for path in ("sneaky.md", "to_hidden.md", "to_text.md", "loop.md", "a.md", "pipe.md", "shortcut/diary.md", "outside/outside.md"):
        message = refusal(tools.puremind_get, service, path)
        assert str(tmp_path) not in message, path


def test_get_opens_every_path_that_search_returns(service, vault):
    names = ["~tilde.md", "~dir/inside.md", "~$word-temp.md", " leading space.md", " spaced dir/inside.md", "sub dir/n.md"]
    for name in names:
        (vault / name).parent.mkdir(parents=True, exist_ok=True)
        (vault / name).write_text(f"# {name.strip()}\n\nbody of {name}\n")
    index_vault(vault, service.store, service.embedder)
    indexed = {f.path for f in service.store.list_files()}
    assert set(names) <= indexed
    for path in indexed:
        assert tools.puremind_get(service, path).startswith(f"path: {path}\n"), path
    assert tools.puremind_get(service, "  projects/garden.md ").startswith("path: projects/garden.md")
    for path in ("~/../../outside.md", "~/x.md", "sub dir/../../outside.md"):
        refusal(tools.puremind_get, service, path)


def test_get_tries_the_name_exactly_as_given_first(service, vault):
    names = ["\nleading newline.md", "\rcarriage return.md", " space.md"]
    for name in names:
        (vault / name).write_text(f"# {name.strip()}\n\nbody\n")
    (vault / "space.md").write_text("# another note\n\nbody\n")
    index_vault(vault, service.store, service.embedder)
    for name in names:
        assert tools.puremind_get(service, name).startswith(f"path: {name}\n"), repr(name)
    assert tools.puremind_get(service, "projects/garden.md\n").startswith("path: projects/garden.md\n")
    assert tools.puremind_get(service, "space.md").startswith("path: space.md\n")


@pytest.mark.parametrize("path", ["./projects/garden.md", "projects/./garden.md", "projects//garden.md", " ./projects/garden.md"])
def test_get_accepts_harmless_spellings_of_a_note_path(service, vault, path):
    assert tools.puremind_get(service, path).startswith("path: projects/garden.md\n")


def test_get_never_follows_an_absolute_path_behind_white_space(service, vault, tmp_path):
    (tmp_path / "outside.md").write_text("# outside\n\nsecret\n")
    for target in (tmp_path / "outside.md", vault / "projects" / "garden.md"):
        for path in (f" {target}", f"\t{target}", f"{target}\n", f" \n{target}"):
            message = refusal(tools.puremind_get, service, path)
            assert str(tmp_path) not in message, repr(path)


def test_get_says_when_a_file_is_not_markdown(service, vault):
    assert "only Markdown notes" in refusal(tools.puremind_get, service, "notes.txt")
    assert "only Markdown notes" in refusal(tools.puremind_get, service, "projects/espresso")
    (vault / "Upper.MARKDOWN").write_text("# Upper\n")
    assert "# Upper" in tools.puremind_get(service, "Upper.MARKDOWN")


def test_get_serves_a_symlink_to_a_visible_note(service, vault):
    (vault / "alias.md").symlink_to(vault / "projects" / "garden.md")
    out = tools.puremind_get(service, "alias.md")
    assert out.startswith("path: projects/garden.md") and "Plant tomatoes" in out


def test_get_refuses_files_over_the_indexing_cap(service, vault):
    (vault / "big.md").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    assert "2 MiB" in refusal(tools.puremind_get, service, "big.md")
    (vault / "edge.md").write_bytes(b"y" * MAX_FILE_BYTES)
    assert tools.puremind_get(service, "edge.md").endswith("[truncated at 200000 characters]")


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_get_reports_an_unreadable_note_without_the_system_message(service, vault, tmp_path):
    locked = vault / "locked.md"
    locked.write_text("# locked\n")
    locked.chmod(0)
    try:
        message = refusal(tools.puremind_get, service, "locked.md")
    finally:
        locked.chmod(0o644)
    assert str(tmp_path) not in message and "Permission" not in message


def test_get_does_not_resolve_an_over_long_path(service, monkeypatch):
    def boom(*args):
        raise AssertionError("the file system was touched")

    monkeypatch.setattr(tools, "is_indexable", boom)
    for path in ("a/" * 50_000 + "x.md", "x" * 100_000, "a\x00.md", "../x.md", ".hidden/x.md", "/abs.md"):
        refusal(tools.puremind_get, service, path)


def test_get_reads_at_most_four_bytes_per_character_of_the_limit(service, vault, monkeypatch):
    monkeypatch.setattr(tools, "MAX_GET_CHARS", 25)
    (vault / "long.md").write_text("é" * 5000)  # 10,000 bytes
    reads = []
    real_fdopen = os.fdopen

    class Spy:
        def __init__(self, handle):
            self.handle = handle

        def read(self, n=-1):
            reads.append(n)
            return self.handle.read(n)

        def fileno(self):
            return self.handle.fileno()

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *exc):
            return self.handle.__exit__(*exc)

    monkeypatch.setattr(tools.os, "fdopen", lambda *a, **k: Spy(real_fdopen(*a, **k)))
    out = tools.puremind_get(service, "long.md")
    assert reads == [4 * 25 + 1]
    assert out.endswith("é" * 25 + "\n\n[truncated at 25 characters]")


def test_get_does_not_flag_a_file_that_just_fits(service, vault, monkeypatch):
    monkeypatch.setattr(tools, "MAX_GET_CHARS", 30)
    (vault / "fits.md").write_text("é" * 30)
    assert "truncated" not in tools.puremind_get(service, "fits.md")
    (vault / "over.md").write_text("é" * 31)
    assert "[truncated at 30 characters]" in tools.puremind_get(service, "over.md")


def test_get_replaces_undecodable_bytes(service, vault):
    (vault / "latin.md").write_bytes(b"# caf\xe9\n")
    assert "caf�" in tools.puremind_get(service, "latin.md")


def test_get_truncates_huge_files(service, vault, monkeypatch):
    monkeypatch.setattr(tools, "MAX_GET_CHARS", 30)
    (vault / "big.md").write_text("x" * 500)
    assert "[truncated at 30 characters]" in tools.puremind_get(service, "big.md")


def test_get_without_a_vault(cfg, store, embedder, reranker):
    svc = Service(dataclasses.replace(cfg, vault=None), store=store, embedder=embedder, reranker=reranker)
    assert "puremind init" in refusal(tools.puremind_get, svc, "a.md")


def test_get_follows_the_vault_the_index_was_built_from(service, vault, tmp_path):
    indexed(service, vault)
    other = tmp_path / "other"
    (other / "projects").mkdir(parents=True)
    (other / "projects" / "garden.md").write_text("# Garden plan\n\nAnother tree entirely.\n")
    svc = Service(dataclasses.replace(service.cfg, vault=other), service.store, service.embedder, service.reranker)
    out = tools.puremind_get(svc, "projects/garden.md")
    assert "Plant tomatoes in May" in out and "Another tree" not in out
    assert "not the vault this index was built from" in out
    assert str(tmp_path) not in out
    assert "not the vault this index was built from" in tools.puremind_search(svc, "tomatoes")
    assert tools.vault_mismatch(svc) == (vault.resolve(), other.resolve())


def test_get_uses_the_configured_vault_when_the_index_has_none(cfg, store, embedder, reranker, vault):
    svc = Service(cfg, store=store, embedder=embedder, reranker=reranker)
    assert store.get_meta("vault_root") is None
    assert "Plant tomatoes in May" in tools.puremind_get(svc, "projects/garden.md")
    assert tools.vault_mismatch(svc) is None


def test_get_uses_the_indexed_vault_when_none_is_configured(service, vault):
    indexed(service, vault)
    svc = Service(dataclasses.replace(service.cfg, vault=None), service.store, service.embedder, service.reranker)
    assert "Plant tomatoes in May" in tools.puremind_get(svc, "projects/garden.md")


def test_get_says_when_the_indexed_vault_has_gone(service, vault, tmp_path):
    indexed(service, vault)
    moved = tmp_path / "moved"
    vault.rename(moved)
    message = refusal(tools.puremind_get, service, "projects/garden.md")
    assert "no longer exists" in message and "puremind index --rebuild" in message and "puremind init" in message
    assert str(tmp_path) not in message


def test_get_says_when_the_configured_vault_does_not_exist(cfg, store, embedder, reranker, tmp_path):
    svc = Service(dataclasses.replace(cfg, vault=tmp_path / "nowhere"), store=store, embedder=embedder, reranker=reranker)
    message = refusal(tools.puremind_get, svc, "a.md")
    assert "does not exist" in message and "puremind init" in message and str(tmp_path) not in message


def test_get_follows_a_vault_reached_through_a_symlink(service, vault, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(vault)
    indexed(service, vault)
    svc = Service(dataclasses.replace(service.cfg, vault=link), service.store, service.embedder, service.reranker)
    assert tools.vault_mismatch(svc) is None
    assert tools.puremind_get(svc, "projects/garden.md").startswith("path: projects/garden.md\n\n")


def test_a_vault_spelled_in_another_case_is_the_same_vault_where_the_file_system_ignores_case(service, vault):
    other = vault.with_name(vault.name.upper())
    if not other.exists():
        pytest.skip("this file system tells letter case apart")
    indexed(service, vault)
    svc = Service(dataclasses.replace(service.cfg, vault=other), service.store, service.embedder, service.reranker)
    assert tools.vault_mismatch(svc) is None


# ---- recent


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


@pytest.mark.parametrize("mtime", [3.5e11, 1e300, float("inf"), time.time() + 10 * 86400])
def test_recent_leaves_out_future_dated_notes_and_survives_absurd_times(service, vault, mtime):
    indexed(service, vault)
    for i in range(5):  # five of them sort ahead of every real note
        service.store.replace_file(f"future/{i}.md", "x", mtime, 1, f"Future {i}", [], None)
    out = tools.puremind_recent(service, days=3650, limit=2)
    assert "future/" not in out and "2 note(s) modified" in out
    assert "(5 note(s) dated in the future were left out.)" in out


def test_recent_reports_only_future_notes_as_an_empty_window(service):
    service.store.replace_file("future.md", "x", 3.5e11, 1, "Future", [], None)
    out = tools.puremind_recent(service)
    assert out.startswith("No indexed notes") and "1 note(s) dated in the future" in out


def test_recent_shows_unknown_for_a_time_the_platform_cannot_format():
    assert tools._when(3.5e11) == "unknown" and tools._when(float("nan")) == "unknown"
    assert tools._when(0) != "unknown"


def test_tools_never_modify_the_vault(service, vault):
    before = {p: p.read_bytes() for p in vault.rglob("*.md")}
    indexed(service, vault)
    tools.puremind_search(service, "espresso")
    tools.puremind_get(service, "projects/garden.md")
    tools.puremind_recent(service)
    assert before == {p: p.read_bytes() for p in vault.rglob("*.md")}


def test_get_rejects_ancestor_directory_swapped_after_validation(service, vault, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "garden.md").write_text("OUTSIDE_VAULT_CANARY")
    (vault / "projects").mkdir(exist_ok=True)
    (vault / "projects" / "garden.md").write_text("inside")
    original = tools._read_note

    def swap_then_read(path):
        (vault / "projects").rename(vault / "original-projects")
        (vault / "projects").symlink_to(outside, target_is_directory=True)
        return original(path)

    monkeypatch.setattr(tools, "_read_note", swap_then_read)
    with pytest.raises(tools.ToolRefused):
        result = tools.puremind_get(service, "projects/garden.md")
        assert "OUTSIDE_VAULT_CANARY" not in result
