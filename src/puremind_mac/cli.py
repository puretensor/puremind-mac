"""The `puremind` command line.

Exit status: 0 on success, 1 when a search finds nothing (as grep does), 2 for every error
(including bad arguments), 130 after Ctrl-C.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import shlex
import shutil
import signal
import sqlite3
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import __version__
from .config import (
    Config,
    ConfigError,
    load_config,
    load_file_config,
    private_home,
    render_config,
    toml_key,
    toml_string,
)
from .models import ModelLoadError, is_local_path

EXIT_NO_RESULTS = 1
EXIT_ERROR = 2
EXIT_INTERRUPTED = 130
SEARCH_LIMIT_MAX = 50  # the same cap the MCP tool applies (tools.MAX_LIMIT)

# C0 and C1 control characters except tab and newline: a terminal acts on these (escape sequences).
# Also the bidirectional controls, which can make a printed line show its text out of order.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u200e\u200f\u202a-\u202e\u2066-\u2069]")


class CliError(Exception):
    """A failure the user can act on; main() prints it as one line and exits with status 2."""


def _clean(text: str) -> str:
    """``text`` without control characters, so note content cannot drive the terminal."""
    return _CONTROL.sub("", text)


def _one_line(text: str) -> str:
    return " ".join(_clean(text).split())


def _service(cfg: Config):
    from .service import Service

    private_home(cfg.home, create=False)  # one made by an earlier version may still be readable by others
    return Service(cfg)


def _vault(cfg: Config, given: str | None) -> Path:
    vault = Path(given).expanduser() if given else cfg.vault
    if vault is None:
        raise CliError("no vault set. Run `puremind init <vault_dir>` or pass the directory.")
    if not vault.is_dir():  # before the index is opened, which would create an empty one
        raise CliError(f"vault directory not found: {vault.absolute()}")
    return vault


def cmd_init(args, cfg: Config) -> int:
    vault = Path(args.vault_dir).expanduser().resolve()
    if not vault.is_dir():
        raise CliError(f"not a directory: {vault}")
    private_home(cfg.home)  # the index in it holds the text of every note
    if cfg.config_path.exists() and not args.force:
        raise CliError(f"{cfg.config_path} exists; pass --force to overwrite it.")
    # Start from the defaults plus what the file being replaced says, never from the environment.
    base = Config(home=cfg.home)
    if cfg.config_path.exists():
        try:
            base = load_file_config(cfg.home)
        except ConfigError as exc:
            print(f"warning: replacing a config file that could not be read: {exc}", file=sys.stderr)
    text = render_config(vault, base)
    fd = os.open(cfg.config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    private_home(cfg.home, create=False)  # a file being replaced keeps its old mode otherwise
    print(f"wrote {cfg.config_path}")
    print(f"vault: {vault}")
    print("next: `puremind index`, then `puremind connect claude-code` (or codex, cursor).")
    return 0


def cmd_index(args, cfg: Config) -> int:
    from .indexer import ModelMismatch, index_vault

    vault = _vault(cfg, args.vault_dir)
    cfg = dataclasses.replace(cfg, vault=vault)
    svc = _service(cfg)
    try:
        stats = index_vault(vault, svc.store, svc.embedder, rebuild=args.rebuild)
    except ModelMismatch as exc:
        return _fail(str(exc), "run `puremind index --rebuild` to embed every note again with the configured model")
    print(stats.summary())
    for item in stats.skipped:
        print(f"  skipped: {_one_line(item)}", file=sys.stderr)
    return 0


# Under `puremind watch` the keyword index is compacted (Store.purge: about half a second at 50,000
# chunks) once the vault has been quiet this long after a run, not after every save...
PURGE_WHEN_QUIET = 60.0
# ...but at the latest this long after the first run that left words of removed text in it.
PURGE_AT_LEAST_EVERY = 300.0


def cmd_watch(args, cfg: Config) -> int:
    from .indexer import ModelMismatch, VaultMismatch, index_vault
    from .watcher import watch

    vault = _vault(cfg, args.vault_dir)
    svc = _service(dataclasses.replace(cfg, vault=vault))
    stop = threading.Event()
    refused: list[ValueError] = []
    owed_since: list[float | None] = [None]  # when a run first left the keyword index to compact
    first = [True]

    def run() -> None:  # always called on this thread, one run at a time
        # The first run compacts, and so pays what a watch that crashed left owing.
        now = time.monotonic()
        purge = first[0] or (owed_since[0] is not None and now - owed_since[0] >= PURGE_AT_LEAST_EVERY)
        first[0] = False
        try:
            stats = index_vault(vault, svc.store, svc.embedder, purge=purge)
        except (VaultMismatch, ModelMismatch) as exc:
            # A symlinked vault that now names another directory, or an index rebuilt for another
            # model since this watch read its config: every later run would fail the same way.
            refused.append(exc)
            stop.set()
            return
        if purge:
            owed_since[0] = None
        elif stats.purge_deferred and owed_since[0] is None:
            owed_since[0] = now
        print(stats.summary(), flush=True)
        for item in stats.skipped:
            print(f"  skipped: {_one_line(item)}", file=sys.stderr, flush=True)

    def compact() -> None:
        if svc.store.purge_pending():
            svc.store.purge()
        owed_since[0] = None

    print(f"watching {_one_line(str(vault.resolve()))} (Ctrl-C to stop)", flush=True)
    try:
        watch(vault, run, delay=args.debounce, stop=stop, quiet=compact, quiet_after=PURGE_WHEN_QUIET)
    finally:
        try:
            compact()  # skipped by a crash or power cut: the record in the index carries it to the next run
        except Exception as exc:  # noqa: BLE001 - must not hide why watching ended
            print(f"warning: the keyword index was not compacted ({_one_line(exc)}); the next run does it",
                  file=sys.stderr)
        finally:
            svc.store.close()
    if refused:
        if isinstance(refused[0], ModelMismatch):
            return _fail(
                str(refused[0]),
                "`puremind watch` keeps the model it started with: restart it if the model has changed "
                "since, or else run `puremind index --rebuild`",
            )
        raise refused[0]
    return 0


def cmd_search(args, cfg: Config) -> int:
    from .tools import format_score, score_scale, vault_mismatch

    query = args.query.strip()
    if not query:
        raise CliError("the query is empty")
    if not cfg.db_path.is_file():  # opening the store would create an empty index
        raise CliError(f"no index at {cfg.db_path}; run `puremind index`")
    svc = _service(cfg)
    rerank = cfg.rerank and not args.no_rerank
    if args.lexical_only:
        hits, scale = svc.fts_search(query, args.limit), "BM25 keyword score"
    else:
        hits = svc.search(query, args.limit, rerank=rerank)
        scale = score_scale(rerank)
    if not hits:
        empty = svc.store.stats()["chunks"] == 0
        print("no results" + ("; the index is empty, run `puremind index`" if empty else ""))
        return EXIT_NO_RESULTS
    print(f"{len(hits)} result(s); score is a {scale}, comparable only within this list")
    for n, h in enumerate(hits, 1):
        snippet = _one_line(h.content)[:240]
        where = _one_line(h.heading_path) or "top of file"
        print(f"[{n}] {_one_line(h.path)}  ({where})  score={format_score(h.score)}\n    {snippet}")
    if vault_mismatch(svc):
        print(
            "note: config.toml names a different vault than the one this index was built from; "
            "run `puremind index --rebuild` to index the configured vault",
            file=sys.stderr,
        )
    return 0


def _absolute(cfg: Config) -> Config:
    """``cfg`` with every relative path made absolute against the current directory."""

    def path(p: Path | None) -> Path | None:
        return Path(os.path.abspath(p)) if p is not None else None

    def model(name: str) -> str:
        return os.path.abspath(os.path.expanduser(name)) if is_local_path(name) else name

    return dataclasses.replace(
        cfg,
        home=path(cfg.home),
        vault=path(cfg.vault),
        db=path(cfg.db),
        embedder=model(cfg.embedder),
        reranker=model(cfg.reranker),
    )


def _exit_on_sigint() -> None:
    """Make Ctrl-C end `serve` at once.

    The SDK's stdio transport keeps a thread blocked reading stdin, and interpreter shutdown waits
    for it, so on a terminal the process would stay up until a line arrived. A client that closes
    stdin or sends SIGTERM is unaffected. A parent that ignores SIGINT is respected.
    """
    if signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
        return

    def leave(signum, frame) -> None:
        sys.stderr.flush()
        os._exit(EXIT_INTERRUPTED)

    try:
        signal.signal(signal.SIGINT, leave)
    except ValueError:  # not the main thread (puremind used as a library): nothing to install
        pass


def _stderr_handler() -> logging.Handler:
    """Warnings and errors to stderr, also from libraries that set their own logger to DEBUG (bm25s does)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    return handler


def cmd_serve(args, cfg: Config) -> int:
    cfg = _absolute(cfg)
    if args.no_rerank:
        cfg = dataclasses.replace(cfg, rerank=False)
    # The agent starts the server in its own project directory, which may hold anything: leave it
    # before any model or library is loaded, so nothing is looked up there.
    private_home(cfg.home)
    os.chdir(cfg.home)
    from .server import build_server

    logging.basicConfig(level=logging.WARNING, handlers=[_stderr_handler()])
    _exit_on_sigint()
    build_server(_service(cfg)).run(transport="stdio")
    return 0


# Variables a GUI-launched agent cannot see in your shell but the server needs to behave the same.
CARRIED_NAMES = ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_ENDPOINT")  # what models.py reads
_PATH_VARIABLES = ("PUREMIND_HOME", "PUREMIND_VAULT", "PUREMIND_DB", "HF_HOME", "HF_HUB_CACHE")
_MODEL_VARIABLES = ("PUREMIND_EMBED_MODEL", "PUREMIND_RERANK_MODEL")


def carried_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The PUREMIND_* variables and the Hugging Face cache and offline variables that are set.

    Paths are made absolute: the agent starts the server from some other directory.
    """
    environ = os.environ if environ is None else environ
    names = sorted(k for k in environ if k.startswith("PUREMIND_")) + [n for n in CARRIED_NAMES if n in environ]
    out: dict[str, str] = {}
    for name in names:
        value = environ[name]
        if value and (name in _PATH_VARIABLES or (name in _MODEL_VARIABLES and is_local_path(value))):
            value = os.path.abspath(os.path.expanduser(value))
        if value:
            out[name] = value
    return out


def server_command(argv0: str | None = None, executable: str | None = None) -> list[str]:
    """The absolute command that starts the `puremind` being run now, without the subcommand.

    The console script this process was started from; else the `puremind` script beside the
    interpreter; else the absolute interpreter with ``-m puremind_mac`` (``-P`` keeps the working
    directory off sys.path). The bare name is the last resort when nothing is found.
    """
    argv0 = sys.argv[0] if argv0 is None else argv0
    executable = sys.executable if executable is None else executable
    if argv0 and Path(argv0).name in ("puremind", "puremind.exe"):
        script = Path(os.path.abspath(argv0))
        if not script.is_file() and (found := shutil.which(argv0)):
            script = Path(os.path.abspath(found))
        if script.is_file():
            return [str(script)]
    if executable:
        beside = Path(os.path.abspath(executable)).parent / "puremind"
        if beside.is_file():
            return [str(beside)]
        return [os.path.abspath(executable), "-P", "-m", "puremind_mac"]
    return [os.path.abspath(found) if (found := shutil.which("puremind")) else "puremind"]


def _env_note(env: Mapping[str, str]) -> str:
    if not env:
        return ""
    return (
        "\n\nThese variables are set in your shell, which a GUI app does not inherit, so the entry "
        f"carries them: {', '.join(env)}."
    )


def connect_snippet(agent: str, command: str | Sequence[str], env: Mapping[str, str] | None = None) -> str:
    """The text `puremind connect <agent>` prints: where the entry goes and what it says.

    ``command`` is the server command line without the ``serve`` subcommand; ``env`` is added to
    the entry as environment variables.
    """
    argv = [command] if isinstance(command, str) else list(command)
    env = dict(env or {})
    args = [*argv[1:], "serve"]
    note = _env_note(env)
    if agent == "claude-code":
        add = ["claude", "mcp", "add", "--scope", "user", "puremind"]
        for key, value in env.items():
            add += ["-e", f"{key}={value}"]
        shared = {"mcpServers": {"puremind": {"type": "stdio", "command": "puremind", "args": ["serve"]}}}
        return (
            "Run this once. It registers pureMind for every Claude Code session on this machine (user scope):\n\n"
            f"    {shlex.join([*add, '--', *argv, 'serve'])}{note}\n\n"
            "Do not put that absolute path in a project's .mcp.json: the file is committed and shared with everyone "
            "who uses the repository, and the path exists only on this machine. A shared file can carry this "
            "machine-independent entry instead, which relies on `puremind` being on each person's PATH and on "
            "their own ~/.puremind/config.toml:\n\n"
            + json.dumps(shared, indent=2)
            + "\n\nThen restart Claude Code and run /mcp to confirm puremind is connected."
        )
    if agent == "codex":
        add = ["codex", "mcp", "add", "puremind"]
        for key, value in env.items():
            add += ["--env", f"{key}={value}"]
        toml = [
            "[mcp_servers.puremind]",
            f"command = {toml_string(argv[0])}",
            f"args = [{', '.join(toml_string(a) for a in args)}]",
        ]
        if env:
            toml += ["", "[mcp_servers.puremind.env]", *(f"{toml_key(k)} = {toml_string(v)}" for k, v in env.items())]
        return (
            "Run this once. It adds pureMind to ~/.codex/config.toml:\n\n"
            f"    {shlex.join([*add, '--', *argv, 'serve'])}{note}\n\n"
            "Or add this to ~/.codex/config.toml yourself:\n\n"
            + "\n".join(toml)
            + "\n\nThen restart Codex. `codex mcp list` (or /mcp in the terminal UI) shows whether it is connected."
        )
    if agent == "cursor":
        server: dict[str, object] = {"type": "stdio", "command": argv[0], "args": args}
        if env:
            server["env"] = env
        return (
            "Add this to ~/.cursor/mcp.json (all projects). Keep the absolute path out of a project's "
            ".cursor/mcp.json, which is shared when it is committed:\n\n"
            + json.dumps({"mcpServers": {"puremind": server}}, indent=2)
            + f"{note}\n\nThen restart Cursor and enable puremind from Customize in the sidebar. "
            'If it does not start, open the Output panel (Cmd+Shift+U) and choose "MCP Logs".'
        )
    raise ValueError(agent)


def cmd_connect(args, cfg: Config) -> int:
    print(connect_snippet(args.agent, server_command(), carried_env()))
    print(
        "\npureMind does not edit your agent's config files. The server is read-only: it can search "
        "and read your vault but cannot write to it."
    )
    return 0


def _limit(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if not 1 <= number <= SEARCH_LIMIT_MAX:
        raise argparse.ArgumentTypeError(f"must be a whole number from 1 to {SEARCH_LIMIT_MAX}")
    return number


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="puremind", description="Local memory for AI coding agents.")
    p.add_argument("--version", action="version", version=f"puremind {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    init_help = "write config.toml for a vault, in the directory PUREMIND_HOME names (default ~/.puremind)"
    s = sub.add_parser("init", help=init_help, description=init_help)
    s.add_argument("vault_dir")
    s.add_argument("--force", action="store_true", help="overwrite an existing config, keeping the settings it holds")
    s.set_defaults(fn=cmd_init, reads_config=False)  # it replaces the file, so a broken one must not stop it

    s = sub.add_parser("index", help="index the vault (incremental)")
    s.add_argument("vault_dir", nargs="?")
    s.add_argument(
        "--rebuild", action="store_true", help="embed every note again; needed after changing the embedding model"
    )
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("watch", help="index, then re-index when files change")
    s.add_argument("vault_dir", nargs="?")
    s.add_argument("--debounce", type=float, default=2.0, help="seconds of quiet before re-indexing (default 2)")
    s.set_defaults(fn=cmd_watch)

    s = sub.add_parser("search", help="search the index from the terminal")
    s.add_argument("query")
    s.add_argument(
        "--limit", type=_limit, default=10, help=f"number of results, 1 to {SEARCH_LIMIT_MAX} (default 10)"
    )
    s.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder (faster, less precise)")
    s.add_argument("--lexical-only", action="store_true", help="FTS5 keyword search only; loads no models")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("serve", help="run the read-only MCP server on stdio")
    s.add_argument(
        "--no-rerank", action="store_true",
        help="never load the cross-encoder reranker: about 1.5 GB less memory and faster searches, "
        "less precise ranking",
    )
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("connect", help="print the MCP config snippet for an agent")
    s.add_argument("agent", choices=["claude-code", "codex", "cursor"])
    s.set_defaults(fn=cmd_connect, reads_config=False)
    return p


def _fail(message: str, hint: str | None = None, cause: BaseException | None = None) -> int:
    print(f"error: {_clean(message)}", file=sys.stderr)
    if cause is not None and str(cause):
        print(f"  cause: {_one_line(str(cause))}", file=sys.stderr)
    if hint:
        print(f"hint: {hint}", file=sys.stderr)
    return EXIT_ERROR


def _describe(exc: OSError) -> str:
    return f"{exc.strerror}: {exc.filename}" if exc.strerror and exc.filename else str(exc)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg: Config | None = None
    try:
        cfg = load_config() if getattr(args, "reads_config", True) else Config()
        return args.fn(args, cfg)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except BrokenPipeError:  # the reader went away, as in `puremind search x | head -1`
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())  # so the exit flush cannot fail too
        except (OSError, ValueError):
            pass
        return 0
    except ConfigError as exc:
        return _fail(str(exc), "fix the file, or run `puremind init <vault_dir> --force` to write a new one")
    except ModelLoadError as exc:
        return _fail(str(exc), cause=exc.__cause__)
    except (CliError, ValueError, RuntimeError) as exc:
        return _fail(str(exc))
    except sqlite3.Error as exc:
        where = f" at {cfg.db_path}" if cfg else ""
        return _fail(
            f"cannot use the index{where}: {exc}",
            "check that the file is writable, or remove it and run `puremind index`",
        )
    except OSError as exc:
        return _fail(_describe(exc), "check the path and its permissions")


if __name__ == "__main__":
    sys.exit(main())
