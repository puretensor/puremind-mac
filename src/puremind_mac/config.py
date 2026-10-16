"""Configuration: defaults, then ~/.puremind/config.toml, then environment."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
QUERY_INSTRUCTION = "Instruct: Given a search query, retrieve relevant passages that answer it\nQuery: "


def home_dir() -> Path:
    return Path(os.environ.get("PUREMIND_HOME") or Path.home() / ".puremind").expanduser()


@dataclass(frozen=True)
class Config:
    home: Path = field(default_factory=home_dir)
    vault: Path | None = None
    db: Path | None = None
    embedder: str = DEFAULT_EMBEDDER
    reranker: str = DEFAULT_RERANKER
    device: str = "auto"
    candidates: int = 50
    rrf_k: int = 60
    rerank_chars: int = 800
    embed_max_seq: int = 1024

    @property
    def db_path(self) -> Path:
        return self.db or self.home / "index.sqlite"

    @property
    def config_path(self) -> Path:
        return self.home / "config.toml"


def load_config(env: dict[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    home = Path(env.get("PUREMIND_HOME") or Path.home() / ".puremind").expanduser()
    cfg = Config(home=home)
    path = home / "config.toml"
    if path.is_file():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        models = data.get("models", {})
        retr = data.get("retrieval", {})
        cfg = replace(
            cfg,
            vault=Path(data["vault"]).expanduser() if data.get("vault") else None,
            db=Path(data["db"]).expanduser() if data.get("db") else None,
            embedder=models.get("embedder", cfg.embedder),
            reranker=models.get("reranker", cfg.reranker),
            device=models.get("device", cfg.device),
            candidates=int(retr.get("candidates", cfg.candidates)),
            rrf_k=int(retr.get("rrf_k", cfg.rrf_k)),
            rerank_chars=int(retr.get("rerank_chars", cfg.rerank_chars)),
        )
    overrides = {
        "PUREMIND_VAULT": ("vault", lambda v: Path(v).expanduser()),
        "PUREMIND_DB": ("db", lambda v: Path(v).expanduser()),
        "PUREMIND_EMBED_MODEL": ("embedder", str),
        "PUREMIND_RERANK_MODEL": ("reranker", str),
        "PUREMIND_DEVICE": ("device", str),
    }
    for var, (attr, conv) in overrides.items():
        if env.get(var):
            cfg = replace(cfg, **{attr: conv(env[var])})
    return cfg


def render_config(vault: Path, cfg: Config) -> str:
    """TOML text for ``puremind init``. JSON strings are valid TOML basic strings."""
    return (
        "# pureMind for Mac configuration. Environment variables override these values:\n"
        "# PUREMIND_VAULT, PUREMIND_DB, PUREMIND_EMBED_MODEL, PUREMIND_RERANK_MODEL, PUREMIND_DEVICE.\n"
        f"vault = {json.dumps(str(vault))}\n\n"
        "[models]\n"
        f"embedder = {json.dumps(cfg.embedder)}\n"
        f"reranker = {json.dumps(cfg.reranker)}\n"
        '# "auto" picks mps on Apple Silicon, cuda if present, else cpu.\n'
        f"device = {json.dumps(cfg.device)}\n\n"
        "[retrieval]\n"
        f"candidates = {cfg.candidates}\n"
        f"rrf_k = {cfg.rrf_k}\n"
        f"rerank_chars = {cfg.rerank_chars}\n"
    )
