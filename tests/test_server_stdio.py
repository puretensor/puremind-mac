"""The real MCP server in a subprocess, spoken to over stdio.

Two clients are used. A raw JSON-RPC client proves what is on the wire (every stdout line is a
JSON-RPC message, errors carry isError=true, the metadata is as declared). The official mcp client
checks that real SDK traffic works, including overlapping tool calls. The index is built with the
fake models by tests/stdio_server.py, so no model is loaded and no network is used.
"""

import asyncio
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from puremind_mac import __version__

TESTS = Path(__file__).resolve().parent
HELPER = TESTS / "stdio_server.py"
MCP_MAJOR = int(importlib.metadata.version("mcp").split(".")[0])
TIMEOUT = 60


def _environment(home: Path) -> dict[str, str]:
    # PYTHONPATH comes from conftest.py: the source tree, or nothing when the installed package is under test.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PUREMIND_", "HF_"))}
    env.update(PUREMIND_HOME=str(home), HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")  # numpy would start one busy thread per core per process
    return env


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    (v / "projects").mkdir(parents=True)
    (v / "journal").mkdir()
    (v / "projects" / "espresso.md").write_text(
        "# Espresso machine\n\nThe grinder is set to 14 for the washed Ethiopian beans. "
        "Descale the boiler every three months.\n"
    )
    (v / "projects" / "garden.md").write_text("# Garden plan\n\nPlant tomatoes in May. The raised beds need compost.\n")
    (v / "journal" / "2026-10-01.md").write_text("# Journal\n\nMigrate the build pipeline to hermetic containers.\n")
    (v / ".private").mkdir()
    (v / ".private" / "diary.md").write_text("# Diary\n\nnever served\n")
    return v


class RawClient:
    """Speaks JSON-RPC to a server process by hand and keeps every line the server writes to stdout."""

    def __init__(self, argv, env, cwd):
        self.stderr = open(Path(env["PUREMIND_HOME"]).parent / "server.stderr", "wb")  # noqa: SIM115 - closed in close()
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, env=env, cwd=cwd
        )
        self.lines: list[bytes] = []
        self.messages: dict[int, dict] = {}
        self.bad: list[bytes] = []
        self._ids = iter(range(1, 10**6))
        self._cond = threading.Condition()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line)
            try:
                message = json.loads(line)
                assert isinstance(message, dict) and message.get("jsonrpc") == "2.0"
            except (ValueError, AssertionError):
                self.bad.append(line)
                continue
            with self._cond:
                if "id" in message and ("result" in message or "error" in message):
                    self.messages[message["id"]] = message
                self._cond.notify_all()

    def send(self, message: dict) -> None:
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        self.proc.stdin.flush()

    def start(self) -> dict:
        init = self.request(
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}},
        )
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        return init

    def request_nowait(self, method: str, params: dict) -> int:
        ident = next(self._ids)
        self.send({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})
        return ident

    def wait(self, *idents: int) -> list[dict]:
        with self._cond:
            done = self._cond.wait_for(lambda: all(i in self.messages for i in idents), timeout=TIMEOUT)
        assert done, f"no answer to {idents}; server exit {self.proc.poll()}"
        return [self.messages[i] for i in idents]

    def request(self, method: str, params: dict | None = None) -> dict:
        return self.wait(self.request_nowait(method, params or {}))[0]

    def call(self, name: str, **arguments) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})["result"]

    def close(self) -> int:
        """End the session the way a client does, by closing the server's stdin; return its exit status."""
        self.proc.stdin.close()
        try:
            return self.proc.wait(timeout=TIMEOUT)
        finally:
            self._reader.join(timeout=10)
            self.proc.stdout.close()
            self.stderr.close()


def text_of(result: dict) -> str:
    return "".join(block.get("text", "") for block in result["content"])


@pytest.fixture
def server(tmp_path, vault):
    clients = []

    def start(*extra: str) -> RawClient:
        home = tmp_path / "home"
        client = RawClient(
            [sys.executable, str(HELPER), str(home), str(vault), *extra], _environment(home), cwd=tmp_path
        )
        clients.append(client)
        return client

    yield start
    for client in clients:
        if client.proc.poll() is None:
            client.proc.kill()
            client.proc.wait()


def test_the_wire_carries_only_json_rpc_and_declares_a_read_only_server(server):
    client = server()
    init = client.start()["result"]
    assert init["serverInfo"]["name"] == "puremind"
    assert "read-only" in init["instructions"] and "puremind_search" in init["instructions"]
    assert "tools" in init["capabilities"]
    assert init["serverInfo"]["version"] == __version__

    listed = {t["name"]: t for t in client.request("tools/list")["result"]["tools"]}
    assert set(listed) == {"puremind_search", "puremind_get", "puremind_recent", "puremind_propose"}
    assert listed["puremind_propose"]["annotations"] == {
        "readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False,
    }
    for name in ("puremind_search", "puremind_get", "puremind_recent"):
        tool = listed[name]
        assert tool["annotations"] == {
            "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False,
        }
        assert not tool.get("outputSchema")
    search = listed["puremind_search"]["inputSchema"]["properties"]
    assert search["query"]["maxLength"] == 4096
    assert (search["limit"]["minimum"], search["limit"]["maximum"], search["limit"]["default"]) == (1, 50, 10)
    recent = listed["puremind_recent"]["inputSchema"]["properties"]
    assert (recent["days"]["minimum"], recent["days"]["maximum"]) == (1, 3650)
    assert (recent["limit"]["minimum"], recent["limit"]["maximum"]) == (1, 200)
    assert listed["puremind_get"]["inputSchema"]["properties"]["path"]["maxLength"] == 1024
    assert "reranker probability" in listed["puremind_search"]["description"]
    assert "200,000 characters" in listed["puremind_get"]["description"]

    for name, arguments, expected in (
        ("puremind_search", {"query": "espresso grinder"}, "projects/espresso.md"),
        ("puremind_search", {"query": "tomatoes compost", "limit": 1}, "projects/garden.md"),
        ("puremind_get", {"path": "projects/garden.md"}, "Plant tomatoes in May"),
        ("puremind_recent", {}, "journal/2026-10-01.md"),
    ):
        result = client.call(name, **arguments)
        assert not result.get("isError"), result
        assert expected in text_of(result)
        assert result.get("structuredContent") is None  # no duplicate payload next to the text
    assert client.close() == 0
    assert client.lines and client.bad == [], client.bad


def test_a_proposal_over_the_wire_lands_in_the_inbox_and_never_in_the_vault(server, tmp_path, vault):
    before = sorted(p.relative_to(vault).as_posix() for p in vault.rglob("*"))
    client = server()
    client.start()
    result = client.call("puremind_propose", title="Grinder", content="Set the grinder to 14.")
    assert not result.get("isError"), result
    assert "not in the user's notes" in text_of(result)
    files = list((tmp_path / "home" / "inbox").glob("*.json"))
    assert len(files) == 1 and "Set the grinder to 14." in files[0].read_text()
    assert sorted(p.relative_to(vault).as_posix() for p in vault.rglob("*")) == before
    again = client.call("puremind_propose", title="Grinder", content="Set the grinder to 14.")
    assert "already waiting" in text_of(again) and len(list((tmp_path / "home" / "inbox").glob("*.json"))) == 1
    for arguments, expected in (({"title": "", "content": "x"}, "at least 1"), ({"title": "t", "content": "x" * 20_001}, "at most 20000")):
        refused = client.call("puremind_propose", **arguments)
        assert refused["isError"] and expected in text_of(refused)
    assert client.close() == 0
    assert client.bad == []


def test_the_switch_that_turns_reranking_off_reaches_the_tool_text(server):
    client = server("--no-rerank")
    client.start()
    search = {t["name"]: t for t in client.request("tools/list")["result"]["tools"]}["puremind_search"]
    assert "reciprocal rank fusion" in search["description"] and "reranker probability" not in search["description"]
    assert "reciprocal rank fusion" in text_of(client.call("puremind_search", query="espresso grinder"))
    client.close()


REFUSED = [
    ("puremind_search", {"query": "   "}, "query is empty"),
    ("puremind_search", {"query": "x" * 4097}, "4096"),
    ("puremind_get", {"path": "../outside.md"}, "hidden"),
    ("puremind_get", {"path": ".private/diary.md"}, "hidden"),
    ("puremind_get", {"path": "/etc/passwd"}, "relative to the vault"),
    ("puremind_get", {"path": "projects/missing.md"}, "no such note"),
    ("puremind_get", {"path": "a\u0000b.md"}, "NUL"),
    ("puremind_get", {"path": "x/" * 600}, "1024"),
    ("puremind_search", {"query": "espresso", "limit": 100}, "less than or equal to 50"),
    ("puremind_recent", {"days": 0}, "greater than or equal to 1"),
]


def test_refusals_are_protocol_errors_without_absolute_paths(server, tmp_path):
    client = server()
    client.start()
    for name, arguments, expected in REFUSED:
        result = client.call(name, **arguments)
        message = text_of(result)
        assert result["isError"] is True, (name, arguments, result)
        assert expected in message, (name, arguments, message)
        assert str(tmp_path) not in message and "Traceback" not in message, message
        assert "x" * 4097 not in message and len(message) < 2000
    assert client.close() == 0


def _is_error(result) -> bool:
    return bool(getattr(result, "isError", None) or getattr(result, "is_error", None))


def _stray_output_recorder():
    """A message handler for ClientSession. The SDK hands it an Exception for every line on the
    server's stdout that is not a JSON-RPC message, so an empty list means stdout was clean."""
    strays: list[Exception] = []

    async def handler(message):
        if isinstance(message, Exception):
            strays.append(message)

    return strays, handler


QUERIES = {
    "espresso grinder ethiopian": "projects/espresso.md",
    "tomatoes compost raised beds": "projects/garden.md",
    "hermetic containers build pipeline": "journal/2026-10-01.md",
}


def test_overlapping_calls_all_succeed_with_the_mcp_client(tmp_path, vault):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(HELPER), str(tmp_path / "home"), str(vault)],
        env=_environment(tmp_path / "home"),
        cwd=tmp_path,
    )

    strays, handler = _stray_output_recorder()

    async def run():
        async with stdio_client(params) as (read, write), ClientSession(read, write, message_handler=handler) as session:
            await session.initialize()
            searches = [(q, expected) for q, expected in QUERIES.items() for _ in range(4)]
            calls = [session.call_tool("puremind_search", {"query": q, "limit": 3}) for q, _ in searches]
            calls += [session.call_tool("puremind_get", {"path": "projects/garden.md"}) for _ in range(3)]
            calls += [session.call_tool("puremind_recent", {"days": 3650}) for _ in range(3)]
            return searches, await asyncio.wait_for(asyncio.gather(*calls), TIMEOUT)

    searches, results = asyncio.run(run())
    assert strays == []  # nothing but JSON-RPC reached the client on stdout
    assert len(results) == len(searches) + 6
    for result in results:
        assert not _is_error(result), result.content
    for (query, expected), result in zip(searches, results):
        first_hit = result.content[0].text.split("[1] ", 1)[1].splitlines()[0]
        assert first_hit == expected, (query, result.content[0].text[:300])
    assert "Plant tomatoes in May" in results[len(searches)].content[0].text


def test_the_mcp_client_sees_tools_annotations_and_a_tool_error(tmp_path, vault):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(HELPER), str(tmp_path / "home"), str(vault)],
        env=_environment(tmp_path / "home"),
        cwd=tmp_path,
    )

    strays, handler = _stray_output_recorder()

    async def run():
        async with stdio_client(params) as (read, write), ClientSession(read, write, message_handler=handler) as session:
            await session.initialize()
            listed = await session.list_tools()
            refused = await session.call_tool("puremind_get", {"path": ".private/diary.md"})
            served = await session.call_tool("puremind_get", {"path": "projects/espresso.md"})
            return listed, refused, served

    listed, refused, served = asyncio.run(run())
    assert strays == []
    assert {t.name for t in listed.tools} == {"puremind_search", "puremind_get", "puremind_recent", "puremind_propose"}
    assert all(t.annotations is not None for t in listed.tools)
    assert _is_error(refused) and "hidden" in refused.content[0].text
    assert not _is_error(served) and "Ethiopian" in served.content[0].text


# ---- `puremind serve` itself: how it ends


def _serve(tmp_path) -> subprocess.Popen:
    home = tmp_path / "serve-home"
    proc = subprocess.Popen(
        [sys.executable, "-m", "puremind_mac", "serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env=_environment(home), cwd=tmp_path,
    )
    proc.stdin.write(
        (json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}},
        }) + "\n").encode()
    )
    proc.stdin.flush()
    assert json.loads(proc.stdout.readline())["id"] == 1  # the server is up and its signal handlers are installed
    proc.stdin.write(b'{"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}\n')
    proc.stdin.flush()
    time.sleep(0.2)
    return proc


def test_ctrl_c_ends_serve_at_once_with_status_130(tmp_path):
    proc = _serve(tmp_path)
    try:
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=10) == 130  # without the handler it waits for a line on stdin
    finally:
        proc.kill()
        proc.wait()


def test_clients_still_end_serve_by_closing_stdin_or_with_sigterm(tmp_path):
    proc = _serve(tmp_path)
    proc.stdin.close()
    assert proc.wait(timeout=10) == 0
    proc = _serve(tmp_path)
    try:
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == -signal.SIGTERM
    finally:
        proc.kill()
        proc.wait()


def test_serve_leaves_the_working_directory_for_the_home(tmp_path):
    """The agent's project directory must not be the server's working directory."""
    home = tmp_path / "serve-home"
    project = tmp_path / "project"
    project.mkdir()
    (project / "marker").write_text("x")
    probe = (
        "import os, sys\n"
        "from puremind_mac import cli, server\n"
        "server.build_server = lambda svc: type('S', (), {'run': lambda self, transport: print(os.getcwd())})()\n"
        "sys.exit(cli.main(['serve']))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], env=_environment(home), cwd=project, capture_output=True, text=True, timeout=TIMEOUT,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()).resolve() == home.resolve()


def test_the_server_process_runs_the_package_under_test(tmp_path):
    import puremind_mac

    done = subprocess.run(
        [sys.executable, str(HELPER), "--where"], env=_environment(tmp_path / "home"),
        capture_output=True, text=True, timeout=TIMEOUT, check=True,
    )
    assert Path(done.stdout.strip()) == Path(puremind_mac.__file__)  # the source tree, or the installed wheel
