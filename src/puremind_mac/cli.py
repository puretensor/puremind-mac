"""The `puremind` command line."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import sys
from pathlib import Path

from . import __version__
from .config import Config, load_config, render_config


def _service(cfg: Config):
    from .service import Service

    return Service(cfg)


def _vault(cfg: Config, given: str | None) -> Path:
    vault = Path(given).expanduser() if given else cfg.vault
    if vault is None:
        sys.exit("error: no vault set. Run `puremind init <vault_dir>` or pass the directory.")
    return vault


def cmd_init(args, cfg: Config) -> int:
    vault = Path(args.vault_dir).expanduser().resolve()
    if not vault.is_dir():
        sys.exit(f"error: not a directory: {vault}")
    cfg.home.mkdir(parents=True, exist_ok=True)
    if cfg.config_path.exists() and not args.force:
        sys.exit(f"error: {cfg.config_path} exists; pass --force to overwrite it.")
    cfg.config_path.write_text(render_config(vault, cfg), encoding="utf-8")
    print(f"wrote {cfg.config_path}")
    print(f"vault: {vault}")
    print("next: `puremind index`, then `puremind connect claude-code` (or codex, cursor).")
    return 0


def cmd_index(args, cfg: Config) -> int:
    from .indexer import index_vault

    vault = _vault(cfg, args.vault_dir)
    cfg = dataclasses.replace(cfg, vault=vault)
    svc = _service(cfg)
    stats = index_vault(vault, svc.store, svc.embedder, rebuild=args.rebuild)
    print(stats.summary())
    for item in stats.skipped:
        print(f"  skipped: {item}", file=sys.stderr)
    return 0


def cmd_watch(args, cfg: Config) -> int:
    from .indexer import index_vault
    from .watcher import watch

    vault = _vault(cfg, args.vault_dir)
    svc = _service(dataclasses.replace(cfg, vault=vault))

    def run() -> None:
        from .store import Store

        store = Store(cfg.db_path)  # a fresh connection: the timer runs on its own thread
        try:
            print(index_vault(vault, store, svc.embedder).summary(), flush=True)
        finally:
            store.close()

    print(f"watching {vault.resolve()} (Ctrl-C to stop)", flush=True)
    watch(vault, run, delay=args.debounce)
    return 0


def cmd_search(args, cfg: Config) -> int:
    svc = _service(cfg)
    if args.lexical_only:
        hits = svc.fts_search(args.query, args.limit)
    else:
        hits = svc.search(args.query, args.limit, rerank=not args.no_rerank)
    if not hits:
        print("no results")
        return 1
    for n, h in enumerate(hits, 1):
        snippet = " ".join(h.content.split())[:240]
        print(f"[{n}] {h.path}  ({h.heading_path or 'top of file'})  score={h.score:.4f}\n    {snippet}")
    return 0


def cmd_serve(args, cfg: Config) -> int:
    from .server import build_server

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    build_server(_service(cfg)).run(transport="stdio")
    return 0


def _exe() -> str:
    return shutil.which("puremind") or "puremind"


def connect_snippet(agent: str, exe: str) -> str:
    server = {"command": exe, "args": ["serve"]}
    if agent == "claude-code":
        return (
            "Run this once. It registers pureMind for every Claude Code session:\n\n"
            f"    claude mcp add --scope user puremind -- {exe} serve\n\n"
            "Or add this to a project's .mcp.json (shared with that repo) or to ~/.claude.json:\n\n"
            + json.dumps({"mcpServers": {"puremind": server}}, indent=2)
            + "\n\nThen restart Claude Code and run /mcp to confirm puremind is connected."
        )
    if agent == "codex":
        return (
            "Add this to ~/.codex/config.toml:\n\n"
            "[mcp_servers.puremind]\n"
            f"command = {json.dumps(exe)}\n"
            'args = ["serve"]\n\n'
            "Then restart Codex."
        )
    if agent == "cursor":
        return (
            "Add this to ~/.cursor/mcp.json (all projects) or .cursor/mcp.json (one project):\n\n"
            + json.dumps({"mcpServers": {"puremind": server}}, indent=2)
            + "\n\nThen restart Cursor and enable puremind under Settings > MCP."
        )
    raise ValueError(agent)


def cmd_connect(args, cfg: Config) -> int:
    print(connect_snippet(args.agent, _exe()))
    print(
        "\npureMind does not edit your agent's config files. The server is read-only: it can search "
        "and read your vault but cannot write to it."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="puremind", description="Local memory for AI coding agents.")
    p.add_argument("--version", action="version", version=f"puremind {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="write ~/.puremind/config.toml for a vault")
    s.add_argument("vault_dir")
    s.add_argument("--force", action="store_true", help="overwrite an existing config")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("index", help="index the vault (incremental)")
    s.add_argument("vault_dir", nargs="?")
    s.add_argument("--rebuild", action="store_true", help="discard the index and re-embed everything")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("watch", help="index, then re-index when files change")
    s.add_argument("vault_dir", nargs="?")
    s.add_argument("--debounce", type=float, default=2.0, help="seconds of quiet before re-indexing")
    s.set_defaults(fn=cmd_watch)

    s = sub.add_parser("search", help="search the index from the terminal")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder (faster, less precise)")
    s.add_argument("--lexical-only", action="store_true", help="FTS5 keyword search only; loads no models")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("serve", help="run the read-only MCP server on stdio")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("connect", help="print the MCP config snippet for an agent")
    s.add_argument("agent", choices=["claude-code", "codex", "cursor"])
    s.set_defaults(fn=cmd_connect)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args, load_config())
    except KeyboardInterrupt:
        return 130
    except (NotADirectoryError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
