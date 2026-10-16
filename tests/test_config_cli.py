import json
import tomllib

import pytest

from puremind_mac import cli
from puremind_mac.config import DEFAULT_EMBEDDER, DEFAULT_RERANKER, Config, load_config, render_config
from puremind_mac.watcher import Debouncer, relevant


def test_defaults(tmp_path):
    cfg = load_config({"PUREMIND_HOME": str(tmp_path)})
    assert cfg.embedder == DEFAULT_EMBEDDER == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg.reranker == DEFAULT_RERANKER == "BAAI/bge-reranker-v2-m3"
    assert (cfg.candidates, cfg.rrf_k, cfg.rerank_chars) == (50, 60, 800)
    assert cfg.db_path == tmp_path / "index.sqlite" and cfg.vault is None


def test_config_file_then_env_override(tmp_path):
    (tmp_path / "config.toml").write_text(
        'vault = "/data/notes"\n[models]\nembedder = "org/embed"\nreranker = "org/rank"\ndevice = "cpu"\n'
        "[retrieval]\ncandidates = 20\n"
    )
    cfg = load_config({"PUREMIND_HOME": str(tmp_path)})
    assert (str(cfg.vault), cfg.embedder, cfg.reranker, cfg.device, cfg.candidates) == (
        "/data/notes", "org/embed", "org/rank", "cpu", 20)
    cfg = load_config({"PUREMIND_HOME": str(tmp_path), "PUREMIND_EMBED_MODEL": "env/embed",
                       "PUREMIND_RERANK_MODEL": "env/rank", "PUREMIND_DEVICE": "mps",
                       "PUREMIND_VAULT": "/other", "PUREMIND_DB": "/tmp/x.db"})
    assert (cfg.embedder, cfg.reranker, cfg.device, str(cfg.vault), str(cfg.db)) == (
        "env/embed", "env/rank", "mps", "/other", "/tmp/x.db")


def test_rendered_config_is_valid_toml_and_roundtrips(tmp_path):
    vault = tmp_path / 'we "ird" dir'
    data = tomllib.loads(render_config(vault, Config(home=tmp_path)))
    assert data["vault"] == str(vault)
    assert data["models"]["embedder"] == DEFAULT_EMBEDDER


def test_init_writes_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PUREMIND_HOME", str(tmp_path / "home"))
    vault = tmp_path / "v"
    vault.mkdir()
    assert cli.main(["init", str(vault)]) == 0
    assert load_config().vault == vault.resolve()
    with pytest.raises(SystemExit):
        cli.main(["init", str(vault)])  # refuses to overwrite
    assert cli.main(["init", str(vault), "--force"]) == 0


def test_init_rejects_missing_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMIND_HOME", str(tmp_path / "home"))
    with pytest.raises(SystemExit):
        cli.main(["init", str(tmp_path / "nope")])


def test_connect_claude_code():
    out = cli.connect_snippet("claude-code", "/usr/local/bin/puremind")
    assert "claude mcp add --scope user puremind -- /usr/local/bin/puremind serve" in out
    snippet = out[out.index("{"): out.rindex("}") + 1]
    assert json.loads(snippet) == {"mcpServers": {"puremind": {"command": "/usr/local/bin/puremind", "args": ["serve"]}}}


def test_connect_cursor_is_valid_json():
    out = cli.connect_snippet("cursor", "puremind")
    assert "~/.cursor/mcp.json" in out
    snippet = out[out.index("{"): out.rindex("}") + 1]
    assert json.loads(snippet)["mcpServers"]["puremind"]["args"] == ["serve"]


def test_connect_codex_is_valid_toml():
    out = cli.connect_snippet("codex", "/opt/bin/puremind")
    block = out[out.index("[mcp_servers"): out.index("Then restart")]
    assert tomllib.loads(block) == {"mcp_servers": {"puremind": {"command": "/opt/bin/puremind", "args": ["serve"]}}}


def test_connect_prints_and_does_not_touch_agent_configs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PUREMIND_HOME", str(tmp_path / "pm"))
    for agent in ("claude-code", "codex", "cursor"):
        assert cli.main(["connect", agent]) == 0
    assert "does not edit your agent's config files" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_connect_rejects_unknown_agent():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["connect", "emacs"])


def test_parser_has_no_write_commands():
    sub = next(a for a in cli.build_parser()._actions if a.dest == "command")
    assert set(sub.choices) == {"init", "index", "watch", "search", "serve", "connect"}


def test_cli_index_and_lexical_search_without_models(tmp_path, monkeypatch, vault, capsys, embedder):
    monkeypatch.setenv("PUREMIND_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("puremind_mac.service.SentenceTransformerEmbedder", lambda *a, **k: embedder)
    assert cli.main(["index", str(vault)]) == 0
    assert "3 added" in capsys.readouterr().out
    assert cli.main(["search", "--lexical-only", "espresso"]) == 0
    assert "projects/espresso.md" in capsys.readouterr().out
    assert cli.main(["search", "--lexical-only", "qqqqzzzz"]) == 1


def test_cli_search_no_rerank_flag(tmp_path, monkeypatch, vault, capsys, embedder, reranker):
    monkeypatch.setenv("PUREMIND_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("puremind_mac.service.SentenceTransformerEmbedder", lambda *a, **k: embedder)
    monkeypatch.setattr("puremind_mac.service.CrossEncoderReranker", lambda *a, **k: reranker)
    cli.main(["index", str(vault)])
    capsys.readouterr()
    cli.main(["search", "--no-rerank", "espresso grinder"])
    assert reranker.calls == []
    cli.main(["search", "espresso grinder"])
    assert len(reranker.calls) == 1


def test_cli_index_without_a_vault_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMIND_HOME", str(tmp_path / "home"))
    with pytest.raises(SystemExit):
        cli.main(["index"])


def test_debouncer_collapses_bursts():
    import threading

    calls = []
    done = threading.Event()
    d = Debouncer(lambda: (calls.append(1), done.set()), delay=0.15)
    for _ in range(5):
        d.poke()
    assert done.wait(2)
    done.clear()
    assert not done.wait(0.4)
    assert calls == [1]


def test_debouncer_survives_a_failing_job():
    import threading

    ok = threading.Event()
    state = {"n": 0}

    def job():
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("boom")
        ok.set()

    d = Debouncer(job, delay=0.05)
    d.poke()
    threading.Event().wait(0.3)
    d.poke()
    assert ok.wait(2)


def test_watcher_relevance():
    assert relevant("/v/a.md") and relevant("/v/b.MARKDOWN")
    assert not relevant("/v/a.txt") and not relevant("/v/.hidden.md")
