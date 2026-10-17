"""The model layer, against stub sentence_transformers, torch and huggingface_hub modules.

Nothing here loads a model or touches the network. The stubs record how the loaders are called,
which is what decides which files are read and what the query and document texts look like.
"""

import os
import socket
import subprocess
import sys
import types
import urllib.request
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from puremind_mac import models
from puremind_mac.config import (
    DEFAULT_EMBEDDER,
    DEFAULT_EMBEDDER_REVISION,
    DEFAULT_RERANKER,
    DEFAULT_RERANKER_REVISION,
    QUERY_INSTRUCTION,
)
from puremind_mac.models import (
    CrossEncoderReranker,
    ModelLoadError,
    SentenceTransformerEmbedder,
    pick_device,
)

# The real-model test reads the user's own Hugging Face cache, found here before the hermetic fixture
# in conftest.py points HOME at a throwaway directory.
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME") or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "huggingface"),
    "hub",
)


class StubSentenceTransformer:
    instances: ClassVar[list] = []

    def __init__(self, path, **kwargs):
        self.path, self.kwargs = path, kwargs
        self.max_seq_length = None
        self.encoded: list[list[str]] = []
        StubSentenceTransformer.instances.append(self)

    def encode(self, sentences, **kwargs):
        self.encoded.append(list(sentences))
        self.encode_kwargs = kwargs
        return np.array([[3.0, 4.0, 0.0]] * len(sentences), dtype=np.float32)  # deliberately not unit length


class StubCrossEncoder:
    instances: ClassVar[list] = []

    def __init__(self, path, **kwargs):
        self.path, self.kwargs = path, kwargs
        self.pairs: list = []
        StubCrossEncoder.instances.append(self)

    def predict(self, pairs, **kwargs):
        self.pairs = list(pairs)
        return np.array([0.25 * (i + 1) for i in range(len(self.pairs))])


class Hub:
    """What the stub huggingface_hub knows: which (repo, revision) pairs are cached, and the calls made."""

    def __init__(self, root: Path):
        self.root = root
        self.cached: set[tuple[str, str | None]] = CachedSet(self)
        self.calls: list[dict] = []
        self.download_error: Exception | None = None
        self.real_unreachable = models._unreachable

    def snapshot_path(self, repo: str, revision: str | None) -> str:
        return str(self.root / f"models--{repo.replace('/', '--')}" / "snapshots" / (revision or "main"))

    def snapshot_download(self, repo, **kwargs):
        self.calls.append({"repo": repo, **kwargs})
        if kwargs.get("local_files_only"):
            if (repo, kwargs.get("revision")) not in self.cached:
                raise FileNotFoundError("not in the cache")
            return self.snapshot_path(repo, kwargs.get("revision"))
        if self.download_error is not None:
            raise self.download_error
        self.cached.add((repo, kwargs.get("revision")))
        return fill(self.snapshot_path(repo, kwargs.get("revision")))  # a download completes the folder


SNAPSHOT_FILES = ("config.json", "modules.json", "1_Pooling/config.json", "tokenizer.json", "model.safetensors")


def fill(folder: str, files=SNAPSHOT_FILES) -> str:
    """Make ``folder`` hold the files the loaders look for, as a complete download would."""
    for name in files:
        (Path(folder) / name).parent.mkdir(parents=True, exist_ok=True)
        (Path(folder) / name).touch()
    return folder


class CachedSet(set):
    """The (repo, revision) pairs in the stub cache; adding one creates its complete folder."""

    def __init__(self, hub):
        super().__init__()
        self.hub = hub

    def add(self, item):
        super().add(item)
        fill(self.hub.snapshot_path(*item))

    def __ior__(self, items):
        for item in items:
            self.add(item)
        return self


@pytest.fixture(autouse=True)
def stubs(monkeypatch, tmp_path):
    for name in [n for n in os.environ if n.startswith("PUREMIND_")]:
        monkeypatch.delenv(name)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_ENDPOINT", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.delenv(name, raising=False)
    StubSentenceTransformer.instances.clear()
    StubCrossEncoder.instances.clear()
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer, st.CrossEncoder = StubSentenceTransformer, StubCrossEncoder
    hub = Hub(tmp_path / "hub")
    hf = types.ModuleType("huggingface_hub")
    hf.snapshot_download = hub.snapshot_download
    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    for name, module in (("sentence_transformers", st), ("huggingface_hub", hf), ("torch", torch)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(models, "_unreachable", lambda timeout=5.0: None)
    hub.torch = torch
    return hub


# ---- the wiring the README describes


def test_queries_get_the_instruction_and_documents_go_in_bare(stubs):
    stubs.cached.add((DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION))
    embedder = SentenceTransformerEmbedder(DEFAULT_EMBEDDER, "cpu")
    embedder.embed_documents(["first note", "second note"])
    embedder.embed_query("where is the grinder")
    docs, query = StubSentenceTransformer.instances[0].encoded
    assert docs == ["first note", "second note"]
    assert query == [QUERY_INSTRUCTION + "where is the grinder"]
    assert QUERY_INSTRUCTION.startswith("Instruct: ") and QUERY_INSTRUCTION.endswith("\nQuery: ")


def test_embeddings_are_normalised_even_if_the_model_returns_longer_vectors(stubs):
    stubs.cached.add((DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION))
    embedder = SentenceTransformerEmbedder(DEFAULT_EMBEDDER, "cpu")
    docs = embedder.embed_documents(["a", "b"])
    query = embedder.embed_query("q")
    assert docs.dtype == np.float32 and docs.shape == (2, 3)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0) and np.allclose(docs[0], [0.6, 0.8, 0.0])
    assert query.shape == (3,) and np.isclose(np.linalg.norm(query), 1.0)
    assert StubSentenceTransformer.instances[0].encode_kwargs["normalize_embeddings"] is True


def test_no_documents_means_no_model_load(stubs):
    assert SentenceTransformerEmbedder(DEFAULT_EMBEDDER).embed_documents([]).shape == (0, 0)
    assert StubSentenceTransformer.instances == [] and stubs.calls == []


def test_the_embedder_sets_its_sequence_length_to_2048_by_default(stubs):
    stubs.cached.add((DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION))
    SentenceTransformerEmbedder(DEFAULT_EMBEDDER, "cpu").embed_query("q")
    assert StubSentenceTransformer.instances[0].max_seq_length == 2048
    SentenceTransformerEmbedder(DEFAULT_EMBEDDER, "cpu", max_seq=512).embed_query("q")
    assert StubSentenceTransformer.instances[1].max_seq_length == 512


@pytest.mark.parametrize(
    ("max_seq", "batch_size", "expected"),
    [(2048, 16, 4), (1024, 16, 16), (512, 16, 16), (4096, 16, 1), (8192, 16, 1), (2048, 2, 2), (128, 16, 16)],
)
def test_long_chunks_get_smaller_batches_so_memory_stays_bounded(stubs, max_seq, batch_size, expected):
    stubs.cached.add((DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION))
    embedder = SentenceTransformerEmbedder(DEFAULT_EMBEDDER, "cpu", max_seq=max_seq, batch_size=batch_size)
    embedder.embed_documents(["a" * 1500])
    assert StubSentenceTransformer.instances[-1].encode_kwargs["batch_size"] == expected


def test_short_chunks_keep_the_full_batch_and_the_original_order(stubs, monkeypatch):
    stubs.cached.add((DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION))
    seen = []

    def encode(self, sentences, **kwargs):
        seen.append((list(sentences), kwargs["batch_size"]))
        return np.array([[float(len(t)), 1.0] for t in sentences], dtype=np.float32)

    monkeypatch.setattr(StubSentenceTransformer, "encode", encode)
    long_a, short_b, long_c, short_d = "a" * 1500, "b" * 10, "c" * 2000, "d" * models.SHORT_CHUNK_CHARS
    out = SentenceTransformerEmbedder(DEFAULT_EMBEDDER, "cpu").embed_documents([long_a, short_b, long_c, short_d])
    assert seen == [([short_b, short_d], 16), ([long_a, long_c], 4)]
    assert [round(float(v[0] / v[1])) for v in out] == [1500, 10, 2000, 1024]  # row i is still text i
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_the_reranker_sets_its_length_and_truncates_the_query(stubs):
    stubs.cached.add((DEFAULT_RERANKER, DEFAULT_RERANKER_REVISION))
    reranker = CrossEncoderReranker(DEFAULT_RERANKER, "cpu", query_chars=10)
    assert reranker.score("q" * 50, ["one", "two"]) == [0.25, 0.5]
    stub = StubCrossEncoder.instances[0]
    assert stub.kwargs["max_length"] == 512
    assert stub.pairs == [("q" * 10, "one"), ("q" * 10, "two")]
    assert reranker.score("q", []) == [] and len(StubCrossEncoder.instances) == 1


def test_both_models_are_loaded_from_the_absolute_snapshot_with_remote_code_off(stubs):
    stubs.cached |= {(DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION), (DEFAULT_RERANKER, DEFAULT_RERANKER_REVISION)}
    SentenceTransformerEmbedder(DEFAULT_EMBEDDER, "cpu").embed_query("q")
    CrossEncoderReranker(DEFAULT_RERANKER, "cpu").score("q", ["p"])
    for stub, repo, revision in (
        (StubSentenceTransformer.instances[0], DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION),
        (StubCrossEncoder.instances[0], DEFAULT_RERANKER, DEFAULT_RERANKER_REVISION),
    ):
        assert stub.path == stubs.snapshot_path(repo, revision) and os.path.isabs(stub.path)
        assert stub.kwargs["trust_remote_code"] is False and stub.kwargs["local_files_only"] is True
        assert stub.kwargs["device"] == "cpu"
    assert all(c["local_files_only"] is True for c in stubs.calls)  # found in the cache: nothing downloaded


def test_the_embedder_name_stays_the_configured_id():
    assert SentenceTransformerEmbedder("org/model").name == "org/model"
    assert CrossEncoderReranker("org/rank", revision="abc").name == "org/rank"


def test_a_bare_id_is_not_taken_from_a_directory_in_the_working_directory(stubs, tmp_path, monkeypatch):
    planted = tmp_path / "project" / "Qwen" / "Qwen3-Embedding-0.6B"
    planted.mkdir(parents=True)
    (planted / "modules.json").write_text("[]")
    monkeypatch.chdir(tmp_path / "project")
    stubs.cached.add((DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION))
    SentenceTransformerEmbedder(DEFAULT_EMBEDDER, "cpu").embed_query("q")
    assert StubSentenceTransformer.instances[0].path == stubs.snapshot_path(DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION)
    assert Path(StubSentenceTransformer.instances[0].path) != planted.resolve()
    assert models.resolve_model(DEFAULT_EMBEDDER) == stubs.snapshot_path(DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION)


def test_a_relative_directory_is_only_used_when_written_as_a_path(stubs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mymodel" / "sub").mkdir(parents=True)
    stubs.download_error = RuntimeError("repository not found")
    # no ./ prefix: a Hub id, so it goes to the Hub client even though the directory exists
    with pytest.raises(ModelLoadError):
        models.resolve_model("mymodel/sub")
    assert stubs.calls and all(c["repo"] == "mymodel/sub" for c in stubs.calls)
    expected = (tmp_path / "mymodel").resolve()
    assert Path(models.resolve_model("./mymodel")).resolve() == expected
    assert Path(models.resolve_model("../" + tmp_path.name + "/mymodel")).resolve() == expected
    assert Path(models.resolve_model(".")).resolve() == tmp_path.resolve()
    assert all(os.path.isabs(models.resolve_model(n)) for n in ("./mymodel", ".", "../" + tmp_path.name))


def test_absolute_and_home_paths_are_used_as_they_are(stubs, tmp_path, monkeypatch):
    (tmp_path / "m").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    assert models.resolve_model(str(tmp_path / "m")) == str(tmp_path / "m")
    assert models.resolve_model("~/m") == str(tmp_path / "m")
    with pytest.raises(ModelLoadError, match="model directory not found"):
        models.resolve_model(str(tmp_path / "missing"))
    assert stubs.calls == []


def test_a_local_model_directory_reaches_the_loader(stubs, tmp_path):
    (tmp_path / "m").mkdir()
    SentenceTransformerEmbedder(str(tmp_path / "m"), "cpu").embed_query("q")
    assert StubSentenceTransformer.instances[0].path == str(tmp_path / "m") and stubs.calls == []


@pytest.mark.parametrize("name", ["/abs/x", "./x", "../x", "~/x", "~", ".", ".."])
def test_what_counts_as_a_path(name):
    assert models.is_local_path(name)


@pytest.mark.parametrize("name", [DEFAULT_EMBEDDER, "bert-base-uncased", "x/y", "dir/", ".hidden/x", "x"])
def test_what_counts_as_a_hub_id(name):
    assert not models.is_local_path(name)


# ---- revisions, the cache and the download


def test_default_models_use_the_pinned_revisions_and_others_follow_the_hub_default(stubs):
    stubs.cached |= {
        (DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION),
        (DEFAULT_RERANKER, DEFAULT_RERANKER_REVISION),
        ("org/other", None),
    }
    for name in (DEFAULT_EMBEDDER, DEFAULT_RERANKER, "org/other"):
        models.resolve_model(name)
    assert [c["revision"] for c in stubs.calls] == [DEFAULT_EMBEDDER_REVISION, DEFAULT_RERANKER_REVISION, None]
    assert DEFAULT_EMBEDDER_REVISION == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    assert DEFAULT_RERANKER_REVISION == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def test_the_embedder_keeps_the_sentence_transformers_shorthand_for_its_own_models(stubs):
    stubs.cached.add(("sentence-transformers/all-MiniLM-L6-v2", None))
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2", "cpu")
    embedder.embed_query("q")
    assert embedder.name == "all-MiniLM-L6-v2"  # the index still records the configured name
    assert stubs.calls[0]["repo"] == "sentence-transformers/all-MiniLM-L6-v2"
    stubs.cached.add(("org/model", None))
    SentenceTransformerEmbedder("org/model", "cpu").embed_query("q")
    assert stubs.calls[-1]["repo"] == "org/model"  # a name with an organisation is left alone
    stubs.cached.add(("bert-base-uncased", None))
    CrossEncoderReranker("bert-base-uncased", "cpu").score("q", ["p"])
    assert stubs.calls[-1]["repo"] == "bert-base-uncased"  # the reranker never adds a namespace


def test_a_configured_revision_wins_over_the_pin(stubs):
    stubs.cached.add((DEFAULT_EMBEDDER, "main"))
    SentenceTransformerEmbedder(DEFAULT_EMBEDDER, "cpu", revision="main").embed_query("q")
    assert stubs.calls[0]["revision"] == "main"
    stubs.cached.add(("org/other", "v2"))
    CrossEncoderReranker("org/other", "cpu", revision="v2").score("q", ["p"])
    assert stubs.calls[-1]["revision"] == "v2"


def test_a_cached_model_is_found_without_the_network(stubs, monkeypatch):
    stubs.cached.add((DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION))
    monkeypatch.setattr(models, "_unreachable", lambda timeout=5.0: pytest.fail("the network was probed"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("the network was used"))
    models.resolve_model(DEFAULT_EMBEDDER)
    assert [c.get("local_files_only") for c in stubs.calls] == [True]


def test_a_cache_miss_downloads_once_only_the_files_the_loaders_need(stubs, caplog):
    path = models.resolve_model(DEFAULT_EMBEDDER)
    local, download = stubs.calls
    assert local["local_files_only"] is True and "local_files_only" not in download
    assert download["revision"] == DEFAULT_EMBEDDER_REVISION
    assert download["allow_patterns"] == local["allow_patterns"]
    patterns = download["allow_patterns"]
    assert "*.safetensors" in patterns and "*.json" in patterns
    assert not any(p in ("*", "*.bin", "*.pt", "*.pth", "*.onnx", "*.gguf", "*.py") for p in patterns)
    assert path == stubs.snapshot_path(DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION)
    assert "downloading" in caplog.text
    models.resolve_model(DEFAULT_EMBEDDER)  # now cached
    assert len(stubs.calls) == 3 and stubs.calls[-1]["local_files_only"] is True


@pytest.mark.parametrize("switch", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
@pytest.mark.parametrize("value", ["1", "true", "ON", "yes"])
def test_offline_mode_never_downloads(stubs, monkeypatch, switch, value, tmp_path):
    monkeypatch.setenv(switch, value)
    with pytest.raises(ModelLoadError) as excinfo:
        models.resolve_model(DEFAULT_EMBEDDER)
    message = str(excinfo.value)
    assert "not downloaded yet" in message and "offline" in message and "run the command again" in message
    assert "\n" not in message and str(tmp_path) not in message
    assert [c.get("local_files_only") for c in stubs.calls] == [True]


@pytest.mark.parametrize("lost", ["modules.json", "1_Pooling/config.json", "model.safetensors", "config.json"])
def test_a_cached_copy_without_a_file_the_loaders_need_is_completed_online(stubs, lost):
    stubs.cached.add((DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION))
    folder = Path(stubs.snapshot_path(DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION))
    (folder / lost).unlink()  # another tool filled the cache, or the download was interrupted
    assert models.resolve_model(DEFAULT_EMBEDDER) == str(folder)
    assert [c.get("local_files_only") for c in stubs.calls] == [True, None]
    assert (folder / lost).is_file()


def test_an_incomplete_cached_copy_is_named_when_it_cannot_be_completed(stubs, monkeypatch, tmp_path):
    stubs.cached.add((DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION))
    (Path(stubs.snapshot_path(DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION)) / "modules.json").unlink()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ModelLoadError) as excinfo:
        models.resolve_model(DEFAULT_EMBEDDER)
    message = str(excinfo.value)
    assert "incomplete (modules.json is missing)" in message and "offline" in message
    assert str(tmp_path) not in message
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.setattr(models, "_unreachable", lambda timeout=5.0: "huggingface.co: timed out")
    with pytest.raises(ModelLoadError, match=r"incomplete \(modules.json is missing\).*cannot be reached"):
        models.resolve_model(DEFAULT_EMBEDDER)


def test_sharded_weights_must_all_be_there(tmp_path):
    folder = tmp_path / "m"
    fill(str(folder), ("config.json", "model-00001-of-00002.safetensors"))
    (folder / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}'
    )
    assert models.missing_file(str(folder), "org/other") == "model-00002-of-00002.safetensors"
    fill(str(folder), ("model-00002-of-00002.safetensors",))
    assert models.missing_file(str(folder), "org/other") is None
    assert models.missing_file(str(folder), DEFAULT_RERANKER) == "tokenizer.json"


def test_a_model_without_safetensors_weights_is_reported_after_the_download(stubs, monkeypatch):
    real = stubs.snapshot_download

    def bin_only(repo, **kwargs):
        path = real(repo, **kwargs)
        if not kwargs.get("local_files_only"):
            (Path(path) / "model.safetensors").unlink()  # the repository has only pickle weights
        return path

    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", bin_only)
    with pytest.raises(ModelLoadError, match="no model.safetensors at that revision; pureMind loads safetensors"):
        models.resolve_model("org/old-model")


def test_offline_with_the_model_cached_at_another_revision_says_how_to_use_it(stubs, monkeypatch, tmp_path):
    cache = tmp_path / "hf"
    (cache / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots" / ("0" * 40)).mkdir(parents=True)
    sys.modules["huggingface_hub"].constants = types.SimpleNamespace(HF_HUB_CACHE=str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ModelLoadError) as excinfo:
        models.resolve_model(DEFAULT_EMBEDDER)
    message = str(excinfo.value)
    assert "not at the revision this setup asks for" in message and "embedder_revision" in message
    assert str(tmp_path) not in message


class IncompleteCache(FileNotFoundError):
    """What huggingface_hub 1.x raises for a cached snapshot that lacks some listed files."""

    def __init__(self, snapshot_path):
        super().__init__("the cached snapshot is incomplete")
        self.snapshot_path = snapshot_path


def test_an_incomplete_cached_snapshot_is_completed_online_and_used_as_it_is_offline(stubs, monkeypatch, tmp_path):
    partial = str(tmp_path / "partial")
    real = stubs.snapshot_download
    calls = []

    def download(repo, **kwargs):
        calls.append(kwargs.get("local_files_only"))
        if kwargs.get("local_files_only") and not stubs.completed:
            raise IncompleteCache(partial)
        stubs.completed = True
        return real(repo, **kwargs)

    stubs.completed = False
    fill(partial)  # what the loaders need is there; huggingface_hub misses some file they do not read
    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert models.resolve_model(DEFAULT_EMBEDDER) == partial  # offline: used as it is
    assert calls == [True]
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(models, "_unreachable", lambda timeout=5.0: "huggingface.co: timed out")
    assert models.resolve_model(DEFAULT_EMBEDDER) == partial  # unreachable: the same
    monkeypatch.setattr(models, "_unreachable", lambda timeout=5.0: None)
    assert models.resolve_model(DEFAULT_EMBEDDER) == stubs.snapshot_path(DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION)
    assert calls[-2:] == [True, None]  # online: the download completed the snapshot
    assert models.resolve_model(DEFAULT_EMBEDDER) == stubs.snapshot_path(DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION)


def test_offline_switch_set_to_zero_does_not_block_the_download(stubs, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    models.resolve_model(DEFAULT_EMBEDDER)
    assert len(stubs.calls) == 2


def test_a_download_failure_is_one_clear_line(stubs, tmp_path):
    stubs.download_error = ConnectionError(f"boom at {tmp_path}")
    with pytest.raises(ModelLoadError) as excinfo:
        models.resolve_model(DEFAULT_EMBEDDER)
    message = str(excinfo.value)
    assert "could not download" in message and "ConnectionError" in message and "\n" not in message
    assert str(tmp_path) not in message
    assert isinstance(excinfo.value.__cause__, ConnectionError)


def test_a_load_failure_is_a_model_load_error_not_a_library_exception(stubs, monkeypatch):
    stubs.cached.add((DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION))

    def broken(path, **kwargs):
        raise OSError(f"cannot read {path}")

    monkeypatch.setattr(StubSentenceTransformer, "__init__", lambda self, path, **kw: broken(path))
    with pytest.raises(ModelLoadError) as excinfo:
        SentenceTransformerEmbedder(DEFAULT_EMBEDDER, "cpu").embed_query("q")
    assert "could not load model" in str(excinfo.value) and "OSError" in str(excinfo.value)
    assert str(stubs.root) not in str(excinfo.value)


def test_an_unreachable_hub_fails_fast_instead_of_waiting_for_the_client(stubs, monkeypatch):
    monkeypatch.setattr(models, "_unreachable", lambda timeout=5.0: "huggingface.co: timed out")
    with pytest.raises(ModelLoadError, match="cannot be reached"):
        models.resolve_model(DEFAULT_EMBEDDER)
    assert [c.get("local_files_only") for c in stubs.calls] == [True]  # the download was never attempted


def test_the_reachability_check(stubs, monkeypatch):
    check = stubs.real_unreachable
    monkeypatch.setattr(urllib.request, "getproxies", dict)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{listener.getsockname()[1]}")
        assert check(1.0) is None
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{port}")
    assert "127.0.0.1" in check(1.0)
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"http": "http://proxy.invalid:3128"})
    assert check(1.0) is None  # a proxy is in the way: the client is left to try
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("no direct connection expected"))
    assert check(1.0) is None


def test_service_hands_the_configured_revisions_to_the_loaders(tmp_path):
    from puremind_mac.config import Config
    from puremind_mac.service import Service
    from puremind_mac.store import Store

    cfg = Config(home=tmp_path, embedder_revision="rev-embed", reranker_revision="rev-rank")
    with Store(":memory:") as store:
        svc = Service(cfg, store=store)
    assert svc.embedder._revision == "rev-embed" and svc.reranker._revision == "rev-rank"


# ---- devices


@pytest.mark.parametrize(
    ("mps", "cuda", "expected"),
    [(True, True, "mps"), (True, False, "mps"), (False, True, "cuda"), (False, False, "cpu")],
)
def test_pick_device_prefers_mps_then_cuda_then_cpu(stubs, mps, cuda, expected):
    stubs.torch.backends.mps.is_available = lambda: mps
    stubs.torch.cuda.is_available = lambda: cuda
    assert pick_device() == pick_device("auto") == expected


def test_an_explicit_device_is_used_without_importing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # an import would raise
    assert pick_device("cpu") == "cpu" and pick_device("cuda:1") == "cuda:1"


def test_the_loaders_get_the_picked_device(stubs):
    stubs.torch.backends.mps.is_available = lambda: True
    stubs.cached |= {(DEFAULT_EMBEDDER, DEFAULT_EMBEDDER_REVISION), (DEFAULT_RERANKER, DEFAULT_RERANKER_REVISION)}
    SentenceTransformerEmbedder(DEFAULT_EMBEDDER).embed_query("q")
    CrossEncoderReranker(DEFAULT_RERANKER).score("q", ["p"])
    assert StubSentenceTransformer.instances[0].kwargs["device"] == "mps"
    assert StubCrossEncoder.instances[0].kwargs["device"] == "mps"


# ---- the Hub client's telemetry


def _env_after_import(extra: dict[str, str]) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "DISABLE_TELEMETRY", "DO_NOT_TRACK"))}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", **extra)  # PYTHONPATH from conftest.py
    code = "import os, puremind_mac.models; print(os.environ.get('HF_HUB_DISABLE_TELEMETRY'))"
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip()


def test_importing_the_models_disables_hub_telemetry_unless_the_user_chose():
    assert _env_after_import({}) == "1"
    assert _env_after_import({"HF_HUB_DISABLE_TELEMETRY": "0"}) == "0"


@pytest.mark.skipif(os.environ.get("PUREMIND_REAL_MODEL_TESTS") != "1", reason="set PUREMIND_REAL_MODEL_TESTS=1 (loads the real models)")
def test_real_models_rank_the_relevant_passage_first(monkeypatch):
    """Optional smoke test with the real models (needs them cached, or network for the first download)."""
    for name in ("sentence_transformers", "huggingface_hub", "torch"):
        monkeypatch.delitem(sys.modules, name)  # drop the stubs for this test
    monkeypatch.setenv("HF_HUB_CACHE", HF_HUB_CACHE)
    embedder = SentenceTransformerEmbedder(DEFAULT_EMBEDDER, os.environ.get("PUREMIND_DEVICE", "cpu"))
    docs = embedder.embed_documents(["Descale the espresso boiler every three months.", "Plant tomatoes in May."])
    query = embedder.embed_query("how often do I descale the espresso machine")
    assert docs.shape[0] == 2 and docs.shape[1] == query.shape[0] and np.isclose(np.linalg.norm(query), 1.0)
    assert float(docs[0] @ query) > float(docs[1] @ query)
    reranker = CrossEncoderReranker(DEFAULT_RERANKER, os.environ.get("PUREMIND_DEVICE", "cpu"))
    good, bad = reranker.score("how often do I descale the espresso machine", ["Descale the boiler every three months.", "Plant tomatoes in May."])
    assert 0.0 <= bad < good <= 1.0
