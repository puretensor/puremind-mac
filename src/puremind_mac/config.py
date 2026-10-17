"""Configuration: defaults, then <home>/config.toml, then environment."""

from __future__ import annotations

import os
import stat
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
# Hub commits of the two default models that this release was tested with. A model that is not
# one of the defaults has no pinned revision unless the user sets one.
DEFAULT_EMBEDDER_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DEFAULT_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
# A 2,048-character chunk is up to about 1,900 tokens in Hindi, Greek, Korean, Japanese and Chinese.
DEFAULT_EMBED_MAX_SEQ = 2048
QUERY_INSTRUCTION = "Instruct: Given a search query, retrieve relevant passages that answer it\nQuery: "


class ConfigError(ValueError):
    """The configuration file is unreadable or invalid; the message names the file."""


def home_dir(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("PUREMIND_HOME") or Path.home() / ".puremind").expanduser()


def _make_private(path: Path) -> None:
    """Remove group and other access from ``path`` if this user owns it. A symlink is left alone."""
    try:
        st = os.lstat(path)
        if not stat.S_ISLNK(st.st_mode) and st.st_uid == os.geteuid() and st.st_mode & 0o077:
            os.chmod(path, stat.S_IMODE(st.st_mode) & ~0o077)
    except OSError:
        pass


def private_home(home: Path, create: bool = True) -> None:
    """Create the pureMind home directory with mode 0700, and keep it and its config.toml private.

    The index in it holds the text of every note. An existing ~/.puremind and an existing
    config.toml lose their group and other access if this user owns them. A directory chosen
    with PUREMIND_HOME keeps its mode, and a config.toml that is a symlink keeps its target's.
    """
    if create:
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "posix":
        return
    if home.absolute() == (Path.home() / ".puremind").absolute():
        _make_private(home)
    _make_private(home / "config.toml")


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
    embed_max_seq: int = DEFAULT_EMBED_MAX_SEQ
    # Hub revision (commit, tag or branch) to load; None means the repository's default branch.
    # For the two default models a None is replaced by the pinned commit when the model is loaded.
    embedder_revision: str | None = DEFAULT_EMBEDDER_REVISION
    reranker_revision: str | None = DEFAULT_RERANKER_REVISION
    rerank: bool = True  # False never loads the reranker: searches then rank by rank fusion alone

    @property
    def db_path(self) -> Path:
        return self.db or self.home / "index.sqlite"

    @property
    def config_path(self) -> Path:
        return self.home / "config.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    except UnicodeDecodeError:
        raise ConfigError(f"{path}: the file is not valid UTF-8") from None


def _model_with_revision(
    cfg_model: str,
    cfg_revision: str | None,
    model: str | None,
    revision: str | None,
    default_model: str,
    default_revision: str,
) -> tuple[str, str | None]:
    """A layer's model and revision settings merged over the current ones.

    A new model drops the current revision (it belongs to the old model): the layer's own
    revision applies, else the pinned one for a default model, else None (the default branch).
    """
    if revision:
        return model or cfg_model, revision
    if model is None or model == cfg_model:
        return cfg_model, cfg_revision
    return model, default_revision if model == default_model else None


def _from_file(cfg: Config, data: dict[str, Any], path: Path) -> Config:
    def table(name: str) -> dict[str, Any]:
        value = data.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: [{name}] must be a table")
        return value

    def text(tbl: dict[str, Any], key: str, where: str) -> str | None:
        value = tbl.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ConfigError(f"{path}: {where}{key} must be a string")
        return value or None

    def number(tbl: dict[str, Any], key: str, default: int, where: str) -> int:
        value = tbl.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"{path}: {where}{key} must be a whole number of at least 1")
        return value

    models, retr = table("models"), table("retrieval")
    flag = retr.get("rerank", cfg.rerank)
    if not isinstance(flag, bool):
        raise ConfigError(f"{path}: retrieval.rerank must be true or false")
    vault, db = text(data, "vault", ""), text(data, "db", "")
    embedder, embedder_rev = _model_with_revision(
        cfg.embedder, cfg.embedder_revision,
        text(models, "embedder", "models."), text(models, "embedder_revision", "models."),
        DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION,
    )
    reranker, reranker_rev = _model_with_revision(
        cfg.reranker, cfg.reranker_revision,
        text(models, "reranker", "models."), text(models, "reranker_revision", "models."),
        DEFAULT_RERANKER, DEFAULT_RERANKER_REVISION,
    )
    return replace(
        cfg,
        vault=Path(vault).expanduser() if vault else None,
        db=Path(db).expanduser() if db else None,
        embedder=embedder,
        embedder_revision=embedder_rev,
        reranker=reranker,
        reranker_revision=reranker_rev,
        device=text(models, "device", "models.") or cfg.device,
        embed_max_seq=number(models, "embed_max_seq", cfg.embed_max_seq, "models."),
        candidates=number(retr, "candidates", cfg.candidates, "retrieval."),
        rrf_k=number(retr, "rrf_k", cfg.rrf_k, "retrieval."),
        rerank=flag,
        rerank_chars=number(retr, "rerank_chars", cfg.rerank_chars, "retrieval."),
    )


def load_file_config(home: Path) -> Config:
    """Defaults plus the values in ``<home>/config.toml``, without environment overrides.

    Raises ConfigError, naming the file, when the file exists but cannot be parsed.
    """
    cfg = Config(home=home)
    path = home / "config.toml"
    if path.is_file():
        cfg = _from_file(cfg, _read_toml(path), path)
    return cfg


def load_config(env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    cfg = load_file_config(home_dir(env))
    embedder, embedder_rev = _model_with_revision(
        cfg.embedder, cfg.embedder_revision,
        env.get("PUREMIND_EMBED_MODEL") or None, env.get("PUREMIND_EMBED_REVISION") or None,
        DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION,
    )
    reranker, reranker_rev = _model_with_revision(
        cfg.reranker, cfg.reranker_revision,
        env.get("PUREMIND_RERANK_MODEL") or None, env.get("PUREMIND_RERANK_REVISION") or None,
        DEFAULT_RERANKER, DEFAULT_RERANKER_REVISION,
    )
    cfg = replace(
        cfg,
        embedder=embedder,
        embedder_revision=embedder_rev,
        reranker=reranker,
        reranker_revision=reranker_rev,
    )
    for var, attr, conv in (
        ("PUREMIND_VAULT", "vault", lambda v: Path(v).expanduser()),
        ("PUREMIND_DB", "db", lambda v: Path(v).expanduser()),
        ("PUREMIND_DEVICE", "device", str),
    ):
        if env.get(var):
            cfg = replace(cfg, **{attr: conv(env[var])})
    return cfg


_TOML_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}


def toml_string(value: str) -> str:
    """``value`` as a TOML basic string.

    Backslash, the double quote and control characters (including DEL) are escaped; every other
    character is written as it is, so the text must be saved as UTF-8. json.dumps is not a
    substitute: it writes characters above U+FFFF as surrogate-pair escapes, which TOML forbids.
    """
    out = ['"']
    for ch in value:
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            raise ValueError("a path or name containing undecodable bytes cannot be stored in config.toml")
        if ch in _TOML_ESCAPES:
            out.append(_TOML_ESCAPES[ch])
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def toml_key(name: str) -> str:
    """``name`` as a TOML key: bare when it only uses letters, digits, ``_`` and ``-``."""
    bare = name and all(c.isascii() and (c.isalnum() or c in "_-") for c in name)
    return name if bare else toml_string(name)


def _revision_line(
    key: str, model: str, revision: str | None, default_model: str, default_revision: str
) -> list[str]:
    """The revision setting, written only when it is not what the model gets by default."""
    default = default_revision if model == default_model else None
    return [f"{key} = {toml_string(revision)}"] if revision and revision != default else []


def render_config(vault: Path, cfg: Config) -> str:
    """TOML text for ``puremind init``: ``vault`` plus the settings in ``cfg``."""
    lines = [
        "# pureMind for Mac configuration. Environment variables override these values:",
        "# PUREMIND_VAULT, PUREMIND_DB, PUREMIND_EMBED_MODEL, PUREMIND_RERANK_MODEL, PUREMIND_DEVICE,",
        "# PUREMIND_EMBED_REVISION, PUREMIND_RERANK_REVISION.",
        f"vault = {toml_string(str(vault))}",
    ]
    if cfg.db:
        lines.append(f"db = {toml_string(str(cfg.db))}")
    lines += [
        "",
        "[models]",
        f"embedder = {toml_string(cfg.embedder)}",
        *_revision_line(
            "embedder_revision", cfg.embedder, cfg.embedder_revision, DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION
        ),
        f"reranker = {toml_string(cfg.reranker)}",
        *_revision_line(
            "reranker_revision", cfg.reranker, cfg.reranker_revision, DEFAULT_RERANKER, DEFAULT_RERANKER_REVISION
        ),
        '# "auto" picks mps on Apple Silicon, cuda if present, else cpu.',
        f"device = {toml_string(cfg.device)}",
        "# Tokens per chunk the embedder reads; a chunk is at most 2,048 characters.",
        f"embed_max_seq = {cfg.embed_max_seq}",
        "",
        "[retrieval]",
        f"candidates = {cfg.candidates}",
        f"rrf_k = {cfg.rrf_k}",
        "# false skips the cross-encoder: less memory and faster searches, less precise ranking.",
        f"rerank = {'true' if cfg.rerank else 'false'}",
        f"rerank_chars = {cfg.rerank_chars}",
    ]
    return "\n".join(lines) + "\n"
