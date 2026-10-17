import json
import logging
import os
import shlex
import sqlite3
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from puremind_mac import __version__, cli, tools
from puremind_mac.config import (
    DEFAULT_EMBEDDER,
    DEFAULT_EMBEDDER_REVISION,
    DEFAULT_RERANKER,
    DEFAULT_RERANKER_REVISION,
    Config,
    ConfigError,
    load_config,
    load_file_config,
    private_home,
    render_config,
    toml_string,
)
from puremind_mac.models import ModelLoadError
from puremind_mac.watcher import relevant

VARIABLES = ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE")


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    for name in [n for n in os.environ if n.startswith("PUREMIND_")] + list(VARIABLES):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / "home"
    monkeypatch.setenv("PUREMIND_HOME", str(path))
    return path


def run(*argv):
    return cli.main(list(argv))


# ---- configuration


def test_defaults(tmp_path):
    cfg = load_config({"PUREMIND_HOME": str(tmp_path)})
    assert cfg.embedder == DEFAULT_EMBEDDER == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg.reranker == DEFAULT_RERANKER == "BAAI/bge-reranker-v2-m3"
    assert (cfg.candidates, cfg.rrf_k, cfg.rerank_chars) == (50, 60, 800)
    assert cfg.embed_max_seq == 2048 and cfg.rerank is True
    assert (cfg.embedder_revision, cfg.reranker_revision) == (DEFAULT_EMBEDDER_REVISION, DEFAULT_RERANKER_REVISION)
    assert DEFAULT_EMBEDDER_REVISION == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    assert DEFAULT_RERANKER_REVISION == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
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


def test_embedding_length_and_rerank_switch_come_from_the_file(tmp_path):
    (tmp_path / "config.toml").write_text("[models]\nembed_max_seq = 512\n[retrieval]\nrerank = false\n")
    cfg = load_config({"PUREMIND_HOME": str(tmp_path)})
    assert cfg.embed_max_seq == 512 and cfg.rerank is False


def test_model_revisions_follow_the_model(tmp_path):
    home = {"PUREMIND_HOME": str(tmp_path)}
    cfg = tmp_path / "config.toml"
    cfg.write_text('[models]\nembedder = "org/embed"\nreranker = "org/rank"\n')
    loaded = load_config(home)  # a model that is not one of the defaults has no pinned revision
    assert (loaded.embedder_revision, loaded.reranker_revision) == (None, None)
    cfg.write_text('[models]\nembedder = "org/embed"\nembedder_revision = "v2"\nreranker_revision = "abc123"\n')
    loaded = load_config(home)  # a revision with the default reranker replaces its pin
    assert (loaded.embedder_revision, loaded.reranker_revision) == ("v2", "abc123")
    # the environment changing the model discards the file's revision for the old one
    loaded = load_config({**home, "PUREMIND_EMBED_MODEL": "other/embed"})
    assert (loaded.embedder, loaded.embedder_revision) == ("other/embed", None)
    loaded = load_config({**home, "PUREMIND_EMBED_MODEL": "other/embed", "PUREMIND_EMBED_REVISION": "r9"})
    assert loaded.embedder_revision == "r9"
    loaded = load_config({**home, "PUREMIND_EMBED_MODEL": "org/embed"})  # the same model keeps its revision
    assert loaded.embedder_revision == "v2"
    loaded = load_config({**home, "PUREMIND_EMBED_MODEL": DEFAULT_EMBEDDER})
    assert loaded.embedder_revision == DEFAULT_EMBEDDER_REVISION
    loaded = load_config({"PUREMIND_HOME": str(tmp_path / "none"), "PUREMIND_RERANK_REVISION": "main"})
    assert loaded.reranker == DEFAULT_RERANKER and loaded.reranker_revision == "main"


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ('vault = 12\n', "vault must be a string"),
        ('[models]\nembedder = 3\n', "models.embedder must be a string"),
        ('[models]\nembed_max_seq = 0\n', "models.embed_max_seq must be a whole number"),
        ('[retrieval]\ncandidates = "many"\n', "retrieval.candidates must be a whole number"),
        ('[retrieval]\nrerank = "yes"\n', "retrieval.rerank must be true or false"),
        ('models = 5\n', "[models] must be a table"),
        ('vault = "unterminated\n', "config.toml"),
    ],
)
def test_a_bad_config_value_names_the_file(tmp_path, text, needle):
    (tmp_path / "config.toml").write_text(text)
    with pytest.raises(ConfigError) as excinfo:
        load_config({"PUREMIND_HOME": str(tmp_path)})
    assert str(tmp_path / "config.toml") in str(excinfo.value) and needle in str(excinfo.value)


def test_a_config_that_is_not_utf8_names_the_file(tmp_path):
    (tmp_path / "config.toml").write_bytes(b'vault = "\xff\xfe"\n')
    with pytest.raises(ConfigError, match="not valid UTF-8"):
        load_config({"PUREMIND_HOME": str(tmp_path)})


# ---- TOML strings


HOSTILE = [
    "plain",
    "/Users/me/Notes \U0001f4d3",  # an emoji above U+FFFF: json.dumps writes surrogate escapes TOML rejects
    "\U0001f600\U0001f9ea\U0001f468‍\U0001f469‍\U0001f467",
    "DEL\x7fchar",
    "tab\there",
    "line\nbreak\r\nand\rreturn",
    'say "hi" and \'bye\'',
    "back\\slash \\n not a newline \\u0041",
    "笔记/ノート/한글/हिन्दी/ελληνικά",
    "\x00nul\x01\x02\x08\x0b\x0c\x1b[2J\x1f",
    "sep arators  and bom﻿",
    "café über",
    "",
    '"' * 5 + "\\" * 5,
]


@pytest.mark.parametrize("value", HOSTILE, ids=lambda v: ascii(v)[:30])
def test_toml_string_round_trips(value):
    encoded = toml_string(value)
    assert tomllib.loads(f"x = {encoded}")["x"] == value
    assert "\n" not in encoded and "\x7f" not in encoded and "\ud800" not in encoded
    assert not any(ord(c) < 0x20 for c in encoded)


def test_toml_string_does_not_use_surrogate_pairs():
    assert "\\ud83d" not in toml_string("\U0001f4d3").lower()
    assert toml_string("\U0001f4d3") == '"\U0001f4d3"'
    assert toml_string("\x7f") == '"\\u007F"'


def test_toml_string_refuses_what_toml_cannot_hold():
    with pytest.raises(ValueError, match="undecodable"):
        toml_string("bad\udcff")


def test_rendered_config_is_valid_toml_and_roundtrips(tmp_path):
    vault = tmp_path / 'we "ird" dir'
    data = tomllib.loads(render_config(vault, Config(home=tmp_path)))
    assert data["vault"] == str(vault)
    assert data["models"]["embedder"] == DEFAULT_EMBEDDER
    assert data["models"]["embed_max_seq"] == 2048 and data["retrieval"]["rerank"] is True
    assert "embedder_revision" not in data["models"] and "db" not in data


@pytest.mark.parametrize("value", HOSTILE[1:11], ids=lambda v: ascii(v)[:30])
def test_config_with_hostile_strings_loads_back(tmp_path, value):
    cfg = Config(home=tmp_path, db=Path(tmp_path, value + ".sqlite"), embedder="org/" + value.replace("\x00", ""),
                 embedder_revision=value, device="cpu")
    text = render_config(Path("/vault") / value, cfg)
    (tmp_path / "config.toml").write_text(text, encoding="utf-8")
    loaded = load_config({"PUREMIND_HOME": str(tmp_path)})
    assert loaded.vault == Path("/vault") / value and loaded.db == cfg.db
    assert loaded.embedder == cfg.embedder and loaded.embedder_revision == value


def test_rendered_config_keeps_non_default_revisions_and_the_db_only(tmp_path):
    cfg = Config(home=tmp_path, embedder="org/embed", embedder_revision="abc", reranker_revision="main", rerank=False)
    data = tomllib.loads(render_config(tmp_path / "v", cfg))
    assert data["models"]["embedder_revision"] == "abc" and data["models"]["reranker_revision"] == "main"
    assert data["retrieval"]["rerank"] is False


# ---- init


def test_init_writes_config(home, tmp_path, capsys):
    vault = tmp_path / "v"
    vault.mkdir()
    assert run("init", str(vault)) == 0
    assert load_config().vault == vault.resolve()
    assert run("init", str(vault)) == 2  # refuses to overwrite
    assert "pass --force" in capsys.readouterr().err
    assert run("init", str(vault), "--force") == 0


def test_init_creates_the_home_directory_private(home, tmp_path):
    vault = tmp_path / "v"
    vault.mkdir()
    old = os.umask(0o022)
    try:
        assert run("init", str(vault)) == 0
    finally:
        os.umask(old)
    assert stat.S_IMODE(home.stat().st_mode) == 0o700  # the index in it holds the text of every note
    assert stat.S_IMODE((home / "config.toml").stat().st_mode) == 0o600
    (home / "config.toml").chmod(0o644)
    assert run("init", str(vault), "--force") == 0
    assert stat.S_IMODE((home / "config.toml").stat().st_mode) == 0o600


def test_init_force_leaves_a_symlinked_config_and_its_target_alone(home, tmp_path):
    vault = tmp_path / "v"
    vault.mkdir()
    shared = tmp_path / "dotfiles" / "puremind.toml"
    shared.parent.mkdir()
    shared.write_text('vault = "/elsewhere"\n')
    shared.chmod(0o664)
    home.mkdir(mode=0o700)
    (home / "config.toml").symlink_to(shared)
    assert run("init", str(vault), "--force") == 0
    assert (home / "config.toml").is_symlink() and load_config().vault == vault.resolve()
    assert stat.S_IMODE(shared.stat().st_mode) == 0o664  # the dotfiles repository decides its mode


def test_a_home_made_by_an_earlier_version_is_made_private(tmp_path, monkeypatch):
    default = Path.home() / ".puremind"  # HOME is a throwaway directory in the tests
    monkeypatch.setenv("PUREMIND_HOME", str(default))
    default.mkdir(mode=0o755)
    default.chmod(0o755)
    (default / "config.toml").write_text(render_config(tmp_path, Config(home=default)))
    (default / "config.toml").chmod(0o644)
    private_home(default, create=False)
    assert stat.S_IMODE(default.stat().st_mode) == 0o700
    assert stat.S_IMODE((default / "config.toml").stat().st_mode) == 0o600


def test_a_home_chosen_with_puremind_home_keeps_its_mode(tmp_path):
    chosen = tmp_path / "shared"
    chosen.mkdir(mode=0o755)
    chosen.chmod(0o755)
    (chosen / "config.toml").write_text("")
    (chosen / "config.toml").chmod(0o644)
    private_home(chosen, create=False)
    assert stat.S_IMODE(chosen.stat().st_mode) == 0o755
    assert stat.S_IMODE((chosen / "config.toml").stat().st_mode) == 0o600  # the file is pureMind's own


def test_init_rejects_missing_directory(home, tmp_path, capsys):
    assert run("init", str(tmp_path / "nope")) == 2
    assert "not a directory" in capsys.readouterr().err
    assert not (home / "config.toml").exists()


def test_init_works_for_a_vault_with_an_emoji_in_its_path(home, tmp_path):
    vault = tmp_path / "Notes \U0001f4d3 笔记"
    vault.mkdir()
    assert run("init", str(vault)) == 0
    assert load_config().vault == vault.resolve()
    assert run("init", str(vault), "--force") == 0


def test_init_force_repairs_a_config_that_cannot_be_read(home, tmp_path, capsys):
    vault = tmp_path / "v"
    vault.mkdir()
    home.mkdir()
    (home / "config.toml").write_text('vault = "\\ud83d\\udcd3"\n')  # what json.dumps wrote for an emoji path
    assert run("search", "anything") == 2
    err = capsys.readouterr().err
    assert str(home / "config.toml") in err and "init <vault_dir> --force" in err
    assert run("init", str(vault)) == 2 and "exists; pass --force" in capsys.readouterr().err
    assert run("init", str(vault), "--force") == 0
    assert "could not be read" in capsys.readouterr().err
    assert load_config().vault == vault.resolve()


def test_connect_works_even_when_the_config_file_is_broken(home, capsys):
    home.mkdir()
    (home / "config.toml").write_text("this is [not toml")
    assert run("connect", "cursor") == 0
    assert '"command"' in capsys.readouterr().out


def test_init_force_keeps_what_the_file_holds_and_ignores_the_environment(home, tmp_path, monkeypatch):
    vault = tmp_path / "v"
    vault.mkdir()
    home.mkdir()
    (home / "config.toml").write_text(
        f'vault = "/old"\ndb = "{tmp_path / "my index.sqlite"}"\n[models]\nembedder = "org/embed"\n'
        'embedder_revision = "v2"\ndevice = "cpu"\nembed_max_seq = 512\n[retrieval]\ncandidates = 20\nrerank = false\n'
    )
    for name, value in (("PUREMIND_DEVICE", "mps"), ("PUREMIND_EMBED_MODEL", "env/embed"),
                        ("PUREMIND_RERANK_MODEL", "env/rank"), ("PUREMIND_DB", "/env.sqlite"),
                        ("PUREMIND_VAULT", "/env-vault")):
        monkeypatch.setenv(name, value)
    assert run("init", str(vault), "--force") == 0
    data = tomllib.loads((home / "config.toml").read_text())
    assert data["vault"] == str(vault.resolve())
    assert data["db"] == str(tmp_path / "my index.sqlite")
    assert data["models"] == {
        "embedder": "org/embed", "embedder_revision": "v2", "reranker": DEFAULT_RERANKER, "device": "cpu",
        "embed_max_seq": 512,
    }
    assert data["retrieval"]["candidates"] == 20 and data["retrieval"]["rerank"] is False


def test_init_does_not_bake_one_off_environment_overrides_into_the_file(home, tmp_path, monkeypatch):
    vault = tmp_path / "v"
    vault.mkdir()
    monkeypatch.setenv("PUREMIND_DEVICE", "cpu")
    monkeypatch.setenv("PUREMIND_EMBED_MODEL", "org/other-embed")
    monkeypatch.setenv("PUREMIND_RERANK_MODEL", "org/other-rank")
    monkeypatch.setenv("PUREMIND_DB", "/somewhere/else.sqlite")
    assert run("init", str(vault)) == 0
    data = tomllib.loads((home / "config.toml").read_text())
    assert data["models"]["device"] == "auto" and data["models"]["embedder"] == DEFAULT_EMBEDDER
    assert data["models"]["reranker"] == DEFAULT_RERANKER and "db" not in data
    assert load_file_config(home).device == "auto"


# ---- the embedding length


def test_a_full_chunk_fits_the_embedding_length_in_scripts_with_few_characters_per_token():
    """A chunk is at most 2,048 characters; with the old limit of 1,024 tokens the tail of such a
    window was silently dropped from the vector (Korean, Greek, Hindi, Japanese and Chinese)."""
    tokenizers = pytest.importorskip("tokenizers")
    try:
        from huggingface_hub import snapshot_download

        folder = snapshot_download(
            DEFAULT_EMBEDDER, revision=DEFAULT_EMBEDDER_REVISION, local_files_only=True, allow_patterns=["tokenizer.json"]
        )
        tokenizer = tokenizers.Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
    except Exception:
        pytest.skip("the Qwen3 tokenizer is not in the local Hugging Face cache")
    paragraphs = {
        "chinese": "今天我们讨论了项目的整体进度和下一步的工作计划。团队决定在下个季度之前完成数据迁移，并且对现有的服务进行全面的性能测试。所有的文档都需要更新，以便新成员可以快速了解系统的架构和部署流程。",
        "japanese": "今日はプロジェクト全体の進捗と今後の作業計画について話し合いました。チームは来四半期までにデータ移行を完了し、既存のサービスに対して包括的な性能試験を行うことを決めました。すべての文書を更新して、新しいメンバーがシステムの構成と配備の手順をすぐに理解できるようにする必要があります。",
        "korean": "오늘은 프로젝트의 전체 진행 상황과 앞으로의 작업 계획에 대해 논의했습니다. 팀은 다음 분기까지 데이터 이전을 완료하고 기존 서비스에 대해 전반적인 성능 시험을 실시하기로 결정했습니다. 모든 문서를 갱신하여 새로운 구성원이 시스템 구조와 배포 절차를 빠르게 이해할 수 있도록 해야 합니다.",
        "greek": "Σήμερα συζητήσαμε τη συνολική πρόοδο του έργου και το πρόγραμμα εργασιών για το επόμενο στάδιο. Η ομάδα αποφάσισε να ολοκληρώσει τη μεταφορά των δεδομένων πριν από το επόμενο τρίμηνο και να εκτελέσει πλήρεις δοκιμές απόδοσης στις υπάρχουσες υπηρεσίες.",
        "hindi": "आज हमने परियोजना की समग्र प्रगति और अगले चरण की कार्ययोजना पर चर्चा की। टीम ने तय किया कि अगली तिमाही से पहले डेटा स्थानांतरण पूरा कर लिया जाएगा और मौजूदा सेवाओं का पूरा प्रदर्शन परीक्षण किया जाएगा। सभी दस्तावेज़ों को अद्यतन करना होगा ताकि नए सदस्य प्रणाली की संरचना और परिनियोजन की प्रक्रिया को जल्दी समझ सकें।",
    }
    limit = Config().embed_max_seq
    for script, paragraph in paragraphs.items():
        window = (paragraph * (2048 // len(paragraph) + 1))[:2048]
        assert len(window) == 2048
        tokens = len(tokenizer.encode(window).ids) + 1  # the end-of-text token the model appends
        assert tokens <= limit, f"{script}: a 2,048-character chunk is {tokens} tokens, over {limit}"


# ---- connect


def test_server_command_is_the_console_script_that_is_running(tmp_path, monkeypatch):
    script = tmp_path / "bin" / "puremind"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    assert cli.server_command(str(script), "/usr/bin/python3") == [str(script)]
    monkeypatch.chdir(script.parent)
    assert cli.server_command("./puremind", "/usr/bin/python3") == [str(script.resolve())]  # relative in, absolute out
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    assert cli.server_command("puremind", "/usr/bin/python3") == [str(script.resolve())]  # found on PATH


def test_server_command_falls_back_to_the_script_beside_the_interpreter(tmp_path):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "puremind").write_text("#!/bin/sh\n")
    assert cli.server_command("/x/puremind_mac/cli.py", str(bin_dir / "python")) == [str(bin_dir / "puremind")]
    assert cli.server_command("-c", str(bin_dir / "python")) == [str(bin_dir / "puremind")]


def test_server_command_ends_with_the_absolute_interpreter_and_module(tmp_path, monkeypatch):
    python = tmp_path / "bin" / "python3"
    python.parent.mkdir()
    python.write_text("")
    assert cli.server_command("/x/cli.py", str(python)) == [str(python), "-P", "-m", "puremind_mac"]
    monkeypatch.chdir(tmp_path)
    assert cli.server_command("/x/cli.py", "bin/python3") == [str(python), "-P", "-m", "puremind_mac"]
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.server_command("", "") == ["puremind"]  # nothing known at all
    assert all(os.path.isabs(part) for part in cli.server_command()[:1])


def test_carried_environment_is_the_puremind_and_hugging_face_cache_variables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = {
        "PUREMIND_HOME": "pm", "PUREMIND_DEVICE": "cpu", "PUREMIND_EMBED_MODEL": "org/e", "PUREMIND_RERANK_MODEL": "./local",
        "HF_HOME": "~/hf", "HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": "", "HF_TOKEN": "secret", "PATH": "/bin", "EMPTY": "",
    }
    carried = cli.carried_env(env)
    assert carried == {
        "PUREMIND_DEVICE": "cpu",
        "PUREMIND_EMBED_MODEL": "org/e",
        "PUREMIND_HOME": str(tmp_path / "pm"),
        "PUREMIND_RERANK_MODEL": str(tmp_path / "local"),
        "HF_HOME": os.path.expanduser("~/hf"),
        "HF_HUB_OFFLINE": "1",
    }
    assert "HF_TOKEN" not in carried and list(carried)[:4] == sorted(k for k in carried if k.startswith("PUREMIND_"))
    assert cli.carried_env({}) == {}


def _json_block(text: str) -> dict:
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def test_connect_claude_code():
    out = cli.connect_snippet("claude-code", ["/usr/local/bin/puremind"])
    assert "claude mcp add --scope user puremind -- /usr/local/bin/puremind serve" in out
    assert cli.connect_snippet("claude-code", "/usr/local/bin/puremind") == out  # a plain string works too
    assert "/mcp" in out


def test_connect_claude_code_explains_the_shared_project_file():
    out = cli.connect_snippet("claude-code", ["/opt/me/bin/puremind"], {"PUREMIND_HOME": "/data/pm"})
    assert "Do not put that absolute path in a project's .mcp.json" in out and "shared" in out
    shared = _json_block(out)["mcpServers"]["puremind"]
    assert shared == {"type": "stdio", "command": "puremind", "args": ["serve"]}  # no path, no environment
    assert "/opt/me" not in out[out.index("Do not put"):]


def test_connect_claude_code_quotes_the_path_and_carries_the_environment():
    command = ["/Users/me/My Tools/puremind"]
    env = {"PUREMIND_HOME": "/data/my pm", "HF_HUB_OFFLINE": "1"}
    out = cli.connect_snippet("claude-code", command, env)
    line = next(ln for ln in out.splitlines() if "claude mcp add" in ln)
    assert shlex.split(line) == [
        "claude", "mcp", "add", "--scope", "user", "puremind",
        "-e", "PUREMIND_HOME=/data/my pm", "-e", "HF_HUB_OFFLINE=1",
        "--", "/Users/me/My Tools/puremind", "serve",
    ]
    assert "PUREMIND_HOME, HF_HUB_OFFLINE" in out


def test_connect_with_the_interpreter_fallback_keeps_the_module_arguments():
    command = ["/opt/py/bin/python3", "-P", "-m", "puremind_mac"]
    line = next(ln for ln in cli.connect_snippet("claude-code", command).splitlines() if "claude mcp add" in ln)
    assert shlex.split(line)[-6:] == ["--", "/opt/py/bin/python3", "-P", "-m", "puremind_mac", "serve"]
    cursor = _json_block(cli.connect_snippet("cursor", command))["mcpServers"]["puremind"]
    assert (cursor["command"], cursor["args"]) == ("/opt/py/bin/python3", ["-P", "-m", "puremind_mac", "serve"])
    block = cli.connect_snippet("codex", command)
    block = block[block.index("[mcp_servers"): block.index("Then restart")]
    server = tomllib.loads(block)["mcp_servers"]["puremind"]
    assert (server["command"], server["args"]) == ("/opt/py/bin/python3", ["-P", "-m", "puremind_mac", "serve"])


def test_connect_cursor_is_valid_json_with_the_documented_fields():
    out = cli.connect_snippet("cursor", ["/opt/bin/puremind"], {"PUREMIND_HOME": "/data/pm"})
    assert "~/.cursor/mcp.json" in out and "Customize" in out and "Settings > MCP" not in out
    assert _json_block(out) == {
        "mcpServers": {
            "puremind": {
                "type": "stdio", "command": "/opt/bin/puremind", "args": ["serve"], "env": {"PUREMIND_HOME": "/data/pm"},
            }
        }
    }
    assert "env" not in _json_block(cli.connect_snippet("cursor", ["/opt/bin/puremind"]))["mcpServers"]["puremind"]


def test_connect_codex_is_valid_toml_and_has_the_environment_table():
    out = cli.connect_snippet("codex", ["/opt/bin/puremind"])
    block = out[out.index("[mcp_servers"): out.index("Then restart")]
    assert tomllib.loads(block) == {"mcp_servers": {"puremind": {"command": "/opt/bin/puremind", "args": ["serve"]}}}
    env = {"PUREMIND_HOME": "/data/päth \U0001f4d3", "HF_HUB_OFFLINE": "1"}
    out = cli.connect_snippet("codex", ["/opt/my bin/puremind"], env)
    block = out[out.index("[mcp_servers"): out.index("Then restart")]
    assert tomllib.loads(block) == {
        "mcp_servers": {"puremind": {"command": "/opt/my bin/puremind", "args": ["serve"], "env": env}}
    }
    line = next(ln for ln in out.splitlines() if "codex mcp add" in ln)
    assert shlex.split(line) == [
        "codex", "mcp", "add", "puremind", "--env", f"PUREMIND_HOME={env['PUREMIND_HOME']}", "--env",
        "HF_HUB_OFFLINE=1", "--", "/opt/my bin/puremind", "serve",
    ]


def test_connect_prints_an_absolute_command_and_does_not_touch_agent_configs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PUREMIND_HOME", str(tmp_path / "pm"))
    for agent in ("claude-code", "codex", "cursor"):
        assert run("connect", agent) == 0
        out = capsys.readouterr().out
        assert "does not edit your agent's config files" in out
        assert str(tmp_path / "pm") in out  # the variable set in this shell is carried into the entry
    assert list(tmp_path.iterdir()) == []
    assert run("connect", "claude-code") == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "claude mcp add" in ln)
    command = shlex.split(line)[shlex.split(line).index("--") + 1]
    assert os.path.isabs(command)


def test_serve_logs_warnings_but_not_the_debug_lines_of_chatty_libraries(capsys):
    logger = logging.getLogger("puremind-test-chatty")  # bm25s sets its own logger to DEBUG like this
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = cli._stderr_handler()
    logger.addHandler(handler)
    try:
        logger.debug("Building index from IDs objects")
        logger.warning("something to know")
    finally:
        logger.removeHandler(handler)
    err = capsys.readouterr().err
    assert "Building index" not in err and "something to know" in err


def test_connect_rejects_unknown_agent():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["connect", "emacs"])


def test_python_dash_m_runs_the_command_line():
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}  # PYTHONPATH from conftest.py picks the copy under test
    done = subprocess.run([sys.executable, "-m", "puremind_mac", "--version"], env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0 and done.stdout.strip() == f"puremind {__version__}"
    done = subprocess.run(
        [sys.executable, "-m", "puremind_mac", "connect", "cursor"], env=env, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0
    server = _json_block(done.stdout)["mcpServers"]["puremind"]
    assert os.path.isabs(server["command"])


# ---- the command line


def test_parser_has_no_write_commands():
    sub = next(a for a in cli.build_parser()._actions if a.dest == "command")
    assert set(sub.choices) == {"init", "index", "watch", "search", "serve", "connect"}


def test_help_texts_name_the_variable_that_moves_the_home_directory_and_the_limit_range():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.dest == "command").choices
    assert "PUREMIND_HOME" in " ".join(sub["init"].format_help().split())
    assert "1 to 50" in sub["search"].format_help()
    assert cli.SEARCH_LIMIT_MAX == tools.MAX_LIMIT


@pytest.fixture
def fake_models(monkeypatch, embedder, reranker):
    monkeypatch.setattr("puremind_mac.service.SentenceTransformerEmbedder", lambda *a, **k: embedder)
    monkeypatch.setattr("puremind_mac.service.CrossEncoderReranker", lambda *a, **k: reranker)


def test_cli_index_and_lexical_search_without_models(home, vault, capsys, fake_models):
    assert run("index", str(vault)) == 0
    assert "3 added" in capsys.readouterr().out
    assert run("search", "--lexical-only", "espresso") == 0
    out = capsys.readouterr().out
    assert "projects/espresso.md" in out and "BM25 keyword score" in out
    assert run("search", "--lexical-only", "qqqqzzzz") == 1  # nothing found: 1, as grep does


def test_cli_search_no_rerank_flag(home, vault, capsys, fake_models, reranker):
    run("index", str(vault))
    capsys.readouterr()
    run("search", "--no-rerank", "espresso grinder")
    assert reranker.calls == [] and "reciprocal rank fusion" in capsys.readouterr().out
    run("search", "espresso grinder")
    assert len(reranker.calls) == 1 and "reranker probability from 0 to 1" in capsys.readouterr().out


def test_cli_search_honours_the_config_switch_for_reranking(home, vault, capsys, fake_models, reranker):
    run("index", str(vault))
    home.joinpath("config.toml").write_text("[retrieval]\nrerank = false\n")
    capsys.readouterr()
    assert run("search", "espresso grinder") == 0
    assert reranker.calls == []


def test_cli_search_prints_small_scores_with_three_significant_digits(home, vault, capsys, fake_models, reranker):
    run("index", str(vault))
    reranker.score = lambda query, passages: [1e-3 / 2**i for i in range(len(passages))]
    capsys.readouterr()
    run("search", "--limit", "3", "espresso")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[")]
    assert [ln.rsplit("score=", 1)[1] for ln in lines] == ["0.001", "0.0005", "0.00025"]


def test_cli_exit_codes(home, vault, tmp_path, capsys, fake_models):
    assert run("index") == 2  # no vault configured
    assert "no vault set" in capsys.readouterr().err
    assert run("index", str(tmp_path / "missing")) == 2
    assert "error:" in capsys.readouterr().err
    assert run("index", str(vault)) == 0
    assert run("search", "--lexical-only", "qqqqzzzz") == 1
    for argv in (["search"], ["search", "--bogus", "x"], ["nosuchcommand"], []):
        with pytest.raises(SystemExit) as excinfo:
            run(*argv)
        assert excinfo.value.code == 2


def test_cli_index_refuses_a_new_model_until_rebuild_is_given(home, vault, capsys, fake_models, embedder):
    assert run("index", str(vault)) == 0
    embedder.name = "another-model"
    capsys.readouterr()
    assert run("index", str(vault)) == 2
    err = capsys.readouterr().err
    assert "error: the index was built with fake-embedder but another-model is configured; nothing was changed" in err
    assert "hint: run `puremind index --rebuild`" in err
    assert run("index", "--rebuild", str(vault)) == 0
    assert "3 added" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "-1", "51", "1000", "x", "1.5", ""])
def test_search_limit_must_be_a_whole_number_from_1_to_50(home, capsys, value):
    with pytest.raises(SystemExit) as excinfo:
        run("search", "--limit", value, "query")
    assert excinfo.value.code == 2
    assert "1 to 50" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["1", "50", "7"])
def test_search_limit_accepts_the_valid_range(home, vault, fake_models, value, capsys):
    run("index", str(vault))
    capsys.readouterr()
    assert run("search", "--limit", value, "--lexical-only", "the") in (0, 1)


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_search_rejects_an_empty_query(home, vault, fake_models, capsys, query):
    run("index", str(vault))
    capsys.readouterr()
    assert run("search", query) == 2
    assert "query is empty" in capsys.readouterr().err


def test_search_without_an_index_says_so_and_creates_nothing(home, capsys):
    assert run("search", "espresso") == 2
    err = capsys.readouterr().err
    assert f"no index at {home / 'index.sqlite'}; run `puremind index`" in err
    assert not (home / "index.sqlite").exists()
    assert run("search", "--lexical-only", "espresso") == 2
    assert not (home / "index.sqlite").exists()


def test_search_on_an_empty_index_hints_at_indexing(home, capsys, fake_models):
    from puremind_mac.store import Store

    Store(home / "index.sqlite").close()
    assert run("search", "espresso") == 1
    assert "index is empty" in capsys.readouterr().out


def test_search_says_when_the_configured_vault_is_not_the_indexed_one(home, vault, tmp_path, capsys, fake_models):
    run("index", str(vault))
    other = tmp_path / "other"
    other.mkdir()
    home.joinpath("config.toml").write_text(f'vault = "{other}"\n')
    capsys.readouterr()
    assert run("search", "--lexical-only", "espresso") == 0
    assert "different vault" in capsys.readouterr().err


def test_terminal_control_characters_in_notes_are_not_printed(home, vault, capsys, fake_models):
    (vault / "evil.md").write_text(
        "# Evil \x1b[31mred\x1b[0m\n\nclip \x1b]52;c;ZXZpbA==\x07 title \x1b]0;pwned\x07 clear \x1b[2J end \x9b31m \x08 ok\n"
    )
    run("index", str(vault))
    capsys.readouterr()
    assert run("search", "--lexical-only", "clip") == 0
    out = capsys.readouterr().out
    assert "clip" in out and "end" in out
    assert not any(ch in out for ch in "\x1b\x07\x08\x00\x9b")


def test_control_characters_in_file_names_and_skipped_files_are_not_printed(home, vault, capsys, fake_models):
    name = "pwn\x1b]0;x\x07ed.md"
    (vault / name).write_text("# Pwned\n\nlexicalmarker text\n")
    (vault / "big\x1b[2J.md").write_bytes(b"x" * (2 * 1024 * 1024 + 1))  # skipped: over the size cap
    assert run("index", str(vault)) == 0
    captured = capsys.readouterr()
    assert "skipped:" in captured.err and "big" in captured.err
    assert "\x1b" not in captured.err and "\x07" not in captured.err
    assert run("search", "--lexical-only", "lexicalmarker") == 0
    out = capsys.readouterr().out
    assert "pwn" in out and "\x1b" not in out and "\x07" not in out


def test_clean_keeps_newline_and_tab_only():
    assert cli._clean("a\tb\nc\x1bd\x7fe\x85f\x9fg\rh") == "a\tb\ncdefgh"
    assert cli._one_line("a \x1b[0m  b\n\tc") == "a [0m b c"
    assert cli._one_line("notes/\u202etxt.exe\u202c.md \u2066x\u2069\u200f") == "notes/txt.exe.md x"  # bidi controls


@pytest.mark.parametrize(
    ("error", "needle", "hint"),
    [
        (ModelLoadError("model m is not downloaded yet"), "model m is not downloaded yet", None),
        (sqlite3.OperationalError("unable to open database file"), "cannot use the index", "puremind index"),
        (sqlite3.DatabaseError("file is not a database"), "file is not a database", "puremind index"),
        (PermissionError(13, "Permission denied", "/x/y"), "Permission denied: /x/y", "permissions"),
        (FileNotFoundError(2, "No such file or directory", "/x/y"), "No such file or directory: /x/y", "permissions"),
        (NotADirectoryError("vault directory not found: /v"), "vault directory not found: /v", "permissions"),
        (ValueError("this index belongs to /a"), "this index belongs to /a", None),
        (RuntimeError("Symlink loop from /v/l.md"), "Symlink loop", None),
    ],
)
def test_failures_are_one_line_with_a_hint_and_status_2(home, monkeypatch, capsys, error, needle, hint):
    def boom(cfg):
        raise error

    monkeypatch.setattr(cli, "_service", boom)
    (home).mkdir()
    (home / "index.sqlite").write_bytes(b"")
    assert run("search", "anything") == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err and "error: " in err and needle in err
    if hint:
        assert "hint: " in err and hint in err


def test_a_model_load_failure_shows_its_cause(home, monkeypatch, capsys):
    def boom(cfg):
        raise ModelLoadError("could not download model m (ConnectError)") from OSError("Connection refused")

    monkeypatch.setattr(cli, "_service", boom)
    home.mkdir()
    (home / "index.sqlite").write_bytes(b"")
    assert run("search", "anything") == 2
    err = capsys.readouterr().err
    assert "could not download model m" in err and "cause: Connection refused" in err and "Traceback" not in err


def test_ctrl_c_returns_130(home, monkeypatch):
    def interrupted(cfg):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_service", interrupted)
    home.mkdir()
    (home / "index.sqlite").write_bytes(b"")
    assert run("search", "anything") == 130


def test_a_closed_pipe_ends_quietly(home, monkeypatch, capsys):  # capsys: stdout has no file descriptor to redirect
    def broken(cfg):
        raise BrokenPipeError

    monkeypatch.setattr(cli, "_service", broken)
    home.mkdir()
    (home / "index.sqlite").write_bytes(b"")
    assert run("search", "anything") == 0


def test_cli_index_without_a_vault_errors(home):
    assert run("index") == 2


# ---- serve


def test_serve_turns_reranking_off_and_leaves_the_project_directory(home, tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    seen = {}

    class Recorder:
        def run(self, transport):
            seen["cwd"], seen["transport"] = Path.cwd(), transport

    def build(service):
        seen["rerank"], seen["vault"], seen["db"] = service.cfg.rerank, service.cfg.vault, service.cfg.db
        return Recorder()

    monkeypatch.setattr("puremind_mac.server.build_server", build)
    monkeypatch.setattr(cli, "_exit_on_sigint", lambda: seen.setdefault("sigint", True))
    monkeypatch.setenv("PUREMIND_VAULT", "relative-vault")
    monkeypatch.setenv("PUREMIND_DB", "relative.sqlite")
    assert run("serve", "--no-rerank") == 0
    assert seen["rerank"] is False and seen["transport"] == "stdio" and seen["sigint"] is True
    assert seen["cwd"].resolve() == home.resolve()
    # relative settings were made absolute against the directory the user started in, before leaving it
    assert seen["vault"].resolve() == (project / "relative-vault").resolve()
    assert seen["db"].resolve() == (project / "relative.sqlite").resolve()


def test_serve_keeps_reranking_on_by_default(home, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # serve changes into its home; monkeypatch puts the directory back afterwards
    seen = []

    class Silent:
        def run(self, transport):
            pass

    def build(service):
        seen.append(service.cfg.rerank)
        return Silent()

    monkeypatch.setattr("puremind_mac.server.build_server", build)
    monkeypatch.setattr(cli, "_exit_on_sigint", lambda: None)
    assert run("serve") == 0 and seen == [True]


def test_absolute_makes_relative_settings_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = cli._absolute(Config(home=Path("h"), vault=Path("v"), db=Path("d.sqlite"), embedder="./models/e", reranker="org/r"))
    assert cfg.home == tmp_path / "h" and cfg.vault == tmp_path / "v" and cfg.db == tmp_path / "d.sqlite"
    assert cfg.embedder == str(tmp_path / "models" / "e") and cfg.reranker == "org/r"


# ---- the file watcher


def test_watcher_relevance():
    assert relevant("/v/a.md") and relevant("/v/b.MARKDOWN")
    assert not relevant("/v/a.txt") and not relevant("/v/.hidden.md")
