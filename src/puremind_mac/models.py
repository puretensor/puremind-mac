"""Embedder and reranker interfaces plus the sentence-transformers implementations.

Everything else in the package depends only on the two Protocols, so tests and
alternative backends can inject their own. numpy and torch are imported lazily, on first use,
so the command line starts without them.

A model is never loaded by its bare Hub id, because sentence-transformers would first look for a
directory of that name in the working directory (an MCP client starts the server in the user's
project, which may hold anything). ``resolve_model`` turns the configured name into an absolute
directory instead: the cached Hub snapshot, downloaded once if it is missing, or a local path.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .config import (
    DEFAULT_EMBED_MAX_SEQ,
    DEFAULT_EMBEDDER,
    DEFAULT_EMBEDDER_REVISION,
    DEFAULT_RERANKER,
    DEFAULT_RERANKER_REVISION,
    QUERY_INSTRUCTION,
)

if TYPE_CHECKING:
    import numpy as np

# huggingface_hub reads this when it is first imported. With it set, its requests no longer carry
# library versions or the name of the calling agent, and it stops fetching the agent registry.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

log = logging.getLogger("puremind.models")

# Revision used when none is configured for a default model. A new embedder revision here makes
# `index` and `watch` refuse every existing index until `puremind index --rebuild`, so a release
# that changes it must say so.
PINNED_REVISIONS = {DEFAULT_EMBEDDER: DEFAULT_EMBEDDER_REVISION, DEFAULT_RERANKER: DEFAULT_RERANKER_REVISION}
# What the loaders read: configuration, tokenizer files and safetensors weights (never pickle files).
# A download stays at the weights the models need even when a repository also ships other formats.
HUB_FILE_PATTERNS = ("*.json", "*.txt", "*.model", "*.safetensors", "*.tiktoken")
# Files the loaders need from the two default models, beyond the configuration and weights every
# model needs. A cached copy made by another tool (transformers alone) lacks the sentence-transformers
# files, and without modules.json the embedder would load and silently pool its output differently.
REQUIRED_FILES = {
    DEFAULT_EMBEDDER: ("modules.json", "1_Pooling/config.json", "tokenizer.json"),
    DEFAULT_RERANKER: ("tokenizer.json",),
}


class ModelLoadError(RuntimeError):
    """A model could not be found, downloaded or loaded.

    The message is one line and names no local path, so it is safe to show to an agent;
    the underlying exception is chained as ``__cause__``.
    """


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""

    def embed_query(self, query: str) -> np.ndarray:
        """Return a (dim,) float32 L2-normalised vector."""


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage; higher is better."""


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_local_path(name: str) -> bool:
    """True when ``name`` is a path: absolute, or starting with ``./``, ``../`` or ``~``.

    Anything else is a Hub id, even if a directory of that name happens to exist.
    """
    return os.path.isabs(name) or name.startswith(("./", "../", "~")) or name in (".", "..")


def model_key(name: str) -> str:
    """``name`` as the index records and compares it: a local path made absolute, a Hub id as it is.

    `puremind serve` and the `connect` entry make local model paths absolute, while `index` may get
    the same directory as ``~/models/e``; both spellings must name the same model.
    """
    return os.path.abspath(os.path.expanduser(name)) if is_local_path(name) else name


def model_label(name: str) -> str:
    """``name`` as messages show it: a local directory by its last component, so that no message
    names a local path (they reach the agent). The CLI prints the full path from the cause."""
    if not is_local_path(name):
        return name
    return f"{os.path.basename(os.path.normpath(os.path.expanduser(name))) or name} (a local directory)"


def embedder_revision(embedder: object) -> str:
    """The Hub revision whose weights ``embedder`` computes, "" when it names none (a local
    directory, a stand-in). Vectors of two revisions are as incompatible as those of two models."""
    return getattr(embedder, "revision", None) or ""


def _offline_switch() -> str | None:
    """The Hugging Face offline variable that is set, if any."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name, "").strip().lower() in ("1", "on", "yes", "true"):
            return name
    return None


def offline() -> bool:
    """True when the Hugging Face offline switch is set, so nothing may be downloaded."""
    return _offline_switch() is not None


def _unreachable(timeout: float = 5.0) -> str | None:
    """Why the Hub cannot be reached right now, or None when it can.

    The download client waits more than two minutes for a host that silently drops packets, so a
    first download checks with a short TCP connection first. The check is skipped when a proxy is
    configured, because then the connection to the Hub is not made directly.
    """
    import socket
    import urllib.parse
    import urllib.request

    url = urllib.parse.urlsplit(os.environ.get("HF_ENDPOINT") or "https://huggingface.co")
    host = url.hostname
    if not host:
        return None
    proxies = urllib.request.getproxies()
    if (url.scheme in proxies or "all" in proxies) and not urllib.request.proxy_bypass(host):
        return None
    try:
        socket.create_connection((host, url.port or (443 if url.scheme == "https" else 80)), timeout).close()
    except OSError as exc:
        return f"{host}: {exc.strerror or 'timed out'}"
    return None


def missing_file(folder: str, name: str) -> str | None:
    """The first file the loaders need that the cached copy ``folder`` of model ``name`` lacks.

    Every model needs config.json and its safetensors weights (all shards, when an index lists
    them); an interrupted download leaves the small files but no weights. The default models
    also need REQUIRED_FILES. None when nothing is missing.
    """
    root = Path(folder)
    for needed in ("config.json", *REQUIRED_FILES.get(name, ())):
        if not (root / needed).is_file():
            return needed
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return None if any(root.glob("*.safetensors")) else "model.safetensors"
    try:
        shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return index.name
    return next((shard for shard in shards if not (root / shard).is_file()), None)


def _cached_at_another_revision(repo: str) -> bool:
    """True when the Hugging Face cache holds some snapshot of ``repo``."""
    try:
        from huggingface_hub import constants

        snapshots = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots"
        return any(snapshots.iterdir())
    except Exception:  # noqa: BLE001 - only used to word an error message
        return False


def resolve_model(name: str, revision: str | None = None, *, namespace: str | None = None) -> str:
    """The absolute directory to load ``name`` from.

    A path (see ``is_local_path``) is made absolute and must exist. A Hub id is looked up in the
    local Hugging Face cache at ``revision`` (the pinned commit for the two default models, else
    the default branch), which needs no network. Only when it is not cached, or the cached copy
    lacks a file the loaders need (see ``missing_file``), and offline mode is off and the Hub
    answers within a few seconds, is it downloaded, which also completes a partial copy. A copy
    that huggingface_hub alone knows to be incomplete is used as it is when it cannot be completed.
    ``namespace`` is added to a Hub id that has none, as sentence-transformers does for its own
    models ("all-MiniLM-L6-v2"). Raises ModelLoadError, never a bare library exception.
    """
    if is_local_path(name):
        path = Path(name).expanduser()
        if not path.is_dir():
            raise ModelLoadError(f"model directory not found: {model_label(name)}") from FileNotFoundError(
                errno.ENOENT, "no such directory", os.path.abspath(path)
            )
        return os.path.abspath(path)
    revision = revision or PINNED_REVISIONS.get(name)
    repo = f"{namespace}/{name}" if namespace and "/" not in name else name
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelLoadError("the huggingface_hub package is not installed") from exc
    cached_error: Exception | None = None
    try:  # the same patterns as the download below, so a complete cache is recognised as such
        partial = snapshot_download(repo, revision=revision, local_files_only=True, allow_patterns=list(HUB_FILE_PATTERNS))
    except Exception as exc:  # noqa: BLE001 - not cached, or the cached copy is incomplete
        cached_error = exc
        # huggingface_hub 1.x names the folder of a cached copy it knows to be incomplete (for
        # instance one filled by sentence-transformers, which skips files such as generation_config.json).
        partial = getattr(exc, "snapshot_path", None)
    lacking = missing_file(partial, name) if partial else None
    if partial and cached_error is None and lacking is None:
        return os.path.abspath(partial)
    if lacking:
        problem = f"the cached copy of model {name} is incomplete ({lacking} is missing)"
    elif partial:
        problem = None  # only huggingface_hub misses something; the loaders have what they need
    elif _cached_at_another_revision(repo):
        problem = (
            f"model {name} is cached, but not at the revision this setup asks for "
            "(set embedder_revision or reranker_revision under [models] in config.toml to use the cached one)"
        )
    else:
        problem = f"model {name} is not downloaded yet"
    if (switch := _offline_switch()) is not None:
        blocked = f"offline mode ({switch}) is on; unset it and run the command again with network access to download it"
    elif (why := _unreachable()) is not None:
        blocked = f"the Hugging Face Hub cannot be reached ({why}); connect to the network and run the command again"
    else:
        blocked = None
    if blocked is not None:
        if problem is None:
            return os.path.abspath(partial)
        raise ModelLoadError(f"{problem}, and {blocked}") from cached_error
    log.warning(
        "downloading %s%s from the Hugging Face Hub (one time only)", repo, f" at {revision}" if revision else ""
    )
    try:
        path = snapshot_download(repo, revision=revision, allow_patterns=list(HUB_FILE_PATTERNS))
    except Exception as exc:
        if problem is None:  # completing the copy failed, but the loaders have what they need
            log.warning("could not complete the cached copy of %s (%s); using it as it is", repo, type(exc).__name__)
            return os.path.abspath(partial)
        raise ModelLoadError(
            f"could not download model {name} ({type(exc).__name__}); check the model name and the network, "
            "then run the command again. To load a local directory, give an absolute path or one starting "
            "with ./ or ~"
        ) from exc
    if (lacking := missing_file(path, name)) is not None:
        raise ModelLoadError(
            f"model {name} has no {lacking} at that revision; pureMind loads safetensors weights only"
            if lacking.endswith(".safetensors") else f"model {name} has no {lacking} at that revision"
        )
    return os.path.abspath(path)


def _normalise(v: np.ndarray) -> np.ndarray:
    import numpy as np

    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


# Attention memory grows with batch x sequence length squared. Chunks longer than this many
# characters (about 1,000 tokens even in Hindi, the densest common script) are embedded in smaller
# batches, which keeps batch x length squared at or below the former default of 16 x 1,024 tokens
# although up to 2,048 tokens are now read. Shorter chunks, the usual case, keep the full batch.
SHORT_CHUNK_CHARS = 1024
_ATTENTION_BUDGET = 16 * 1024 * 1024


class SentenceTransformerEmbedder:
    """Qwen3-Embedding (or any sentence-transformers model) with the query instruction.

    ``batch_size`` is an upper bound; long chunks use fewer per batch (4 at the default 2,048 tokens).
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        max_seq: int = DEFAULT_EMBED_MAX_SEQ,
        batch_size: int = 16,
        *,
        revision: str | None = None,
    ):
        self.name = model
        self._device = device
        self._max_seq = max_seq
        self._batch = batch_size
        self._long_batch = max(1, min(batch_size, _ATTENTION_BUDGET // (max_seq * max_seq)))
        self._revision = revision
        # What resolve_model loads: the pinned commit of a default model when none is given.
        self.revision = None if is_local_path(model) else revision or PINNED_REVISIONS.get(model)
        self._model = None
        self._lock = threading.Lock()
        # Searches overlap on MCP worker threads, but one model must not run two calls at once: its
        # fast tokenizer raises "Already borrowed" when two threads use it, and devices such as mps
        # are not safe for concurrent forward passes on one model.
        self._busy = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                path = resolve_model(self.name, self._revision, namespace="sentence-transformers")
                try:
                    from sentence_transformers import SentenceTransformer

                    m = SentenceTransformer(
                        path, device=pick_device(self._device), trust_remote_code=False, local_files_only=True
                    )
                    m.max_seq_length = self._max_seq
                except Exception as exc:
                    raise ModelLoadError(f"could not load model {model_label(self.name)} ({type(exc).__name__})") from exc
                self._model = m
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        import numpy as np

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        texts = list(texts)
        model = self._load()
        short = [i for i, t in enumerate(texts) if len(t) <= SHORT_CHUNK_CHARS]
        long = [i for i, t in enumerate(texts) if len(t) > SHORT_CHUNK_CHARS]
        out = None
        for rows, batch in ((short, self._batch), (long, self._long_batch)):
            if rows:
                with self._busy:
                    vectors = np.asarray(
                        model.encode(
                            [texts[i] for i in rows],
                            batch_size=batch,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        )
                    )
                if out is None:
                    out = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
                out[rows] = vectors
        return _normalise(out)

    def embed_query(self, query: str) -> np.ndarray:
        model = self._load()
        with self._busy:
            out = model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True, show_progress_bar=False)
        return _normalise(out)[0]


class CrossEncoderReranker:
    """BGE reranker via sentence_transformers.CrossEncoder (max_length 512)."""

    def __init__(self, model: str, device: str = "auto", query_chars: int = 1000, *, revision: str | None = None):
        self.name = model
        self._device = device
        self._query_chars = query_chars
        self._revision = revision
        self._model = None
        self._lock = threading.Lock()
        self._busy = threading.Lock()  # one call at a time, as for the embedder

    def _load(self):
        with self._lock:
            if self._model is None:
                path = resolve_model(self.name, self._revision)
                try:
                    from sentence_transformers import CrossEncoder

                    m = CrossEncoder(
                        path, max_length=512, device=pick_device(self._device),
                        trust_remote_code=False, local_files_only=True,
                    )
                except Exception as exc:
                    raise ModelLoadError(f"could not load model {model_label(self.name)} ({type(exc).__name__})") from exc
                self._model = m
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        import numpy as np

        if not passages:
            return []
        q = query[: self._query_chars]
        model = self._load()
        with self._busy:
            scores = model.predict([(q, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in np.asarray(scores).reshape(-1)]
