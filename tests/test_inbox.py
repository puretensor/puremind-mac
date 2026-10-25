"""The inbox: proposals from an agent, and the approval that turns one into a note."""

import dataclasses
import fcntl
import io
import json
import os
import select
import stat
import subprocess
import sys

import pytest

from puremind_mac import cli, inbox, tools
from puremind_mac.config import Config, ConfigError, load_file_config, render_config
from puremind_mac.tools import ToolRefused


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def made(home, title="Grinder", content="The grinder is set to 14."):
    return inbox.propose(home, title, content)[0]


# ---- proposing


def test_a_proposal_is_a_private_json_file_in_a_private_directory_outside_the_vault(home, vault):
    p, new = inbox.propose(home, "  Grinder\nsetting ", "The grinder is set to 14.\r\n\r\n")
    assert new and p.title == "Grinder setting" and p.content == "The grinder is set to 14."
    path = home / "inbox" / f"{p.id}.json"
    assert json.loads(path.read_text())["title"] == "Grinder setting"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((home / "inbox").stat().st_mode) == 0o700
    assert vault not in (home / "inbox").parents


def test_the_same_proposal_twice_is_kept_once(home):
    first, new_first = inbox.propose(home, "Grinder", "Set to 14.")
    second, new_second = inbox.propose(home, "Grinder", "Set to 14.")
    assert (new_first, new_second) == (True, False) and first == second
    assert len(inbox.list_proposals(home)[0]) == 1


@pytest.mark.parametrize("same", [False, True])
def test_concurrent_processes_deduplicate_and_obey_the_cap(home, same):
    home.mkdir()
    (home / "inbox").mkdir()
    script = """
import sys, time
from pathlib import Path
from puremind_mac import inbox
inbox.MAX_PENDING = 1
original = inbox.list_proposals
def slow_list(home):
    result = original(home)
    time.sleep(0.1)
    return result
inbox.list_proposals = slow_list
print('ready', flush=True)
sys.stdin.readline()
try:
    proposal, new = inbox.propose(Path(sys.argv[1]), sys.argv[2], 'content')
    print(proposal.id, new, flush=True)
except inbox.InboxError:
    print('full', flush=True)
"""
    children = []
    try:
        for n in range(4):
            children.append(subprocess.Popen(
                [sys.executable, "-c", script, str(home), "same" if same else f"note {n}"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
        for child in children:
            assert select.select([child.stdout], [], [], 10)[0]
            assert child.stdout.readline().strip() == "ready"
        for child in children:
            child.stdin.write("go\n")
            child.stdin.flush()
        results = []
        for child in children:
            out, err = child.communicate(timeout=10)
            assert child.returncode == 0, err
            results.append(out.strip())
        pending, unreadable = inbox.list_proposals(home)
        assert len(pending) == 1 and not unreadable
        assert results.count(f"{pending[0].id} True") == 1
        assert results.count(f"{pending[0].id} False" if same else "full") == 3
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)


def test_proposing_waits_for_the_sidecar_lock_before_checking_the_inbox(home):
    made(home)
    with (home / "inbox" / ".propose.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        script = """
import sys
from pathlib import Path
from puremind_mac import inbox
original = inbox.list_proposals
def checked(home):
    print('checked', flush=True)
    return original(home)
inbox.list_proposals = checked
print('ready', flush=True)
inbox.propose(Path(sys.argv[1]), 'other', 'content')
"""
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(home)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            assert select.select([child.stdout], [], [], 10)[0]
            assert child.stdout.readline().strip() == "ready"
            assert not select.select([child.stdout], [], [], 0.2)[0]
            fcntl.flock(lock, fcntl.LOCK_UN)
            out, err = child.communicate(timeout=10)
            assert child.returncode == 0 and out.strip() == "checked", err
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)


def test_failed_creation_releases_the_lock(home, monkeypatch):
    def broken(*args):
        raise OSError("write failed")

    with monkeypatch.context() as patch:
        patch.setattr(inbox.json, "dump", broken)
        with pytest.raises(OSError, match="write failed"):
            made(home)
    with (home / "inbox" / ".propose.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert inbox.list_proposals(home) == ([], [])
    assert made(home)


@pytest.mark.parametrize(
    ("title", "content", "message"),
    [
        ("", "text", "title"),
        (" \n ", "text", "title"),
        ("t", "  \n", "no content"),
        ("t" * 201, "text", "title is longer"),
        ("t", "x" * 20_001, "content is longer"),
        ("t", "a\x00b", "NUL"),
        ("t", "lone \ud800 surrogate", "Unicode"),
        ("lone \udc00", "text", "Unicode"),
    ],
)
def test_bad_proposals_are_refused_and_leave_nothing_behind(home, title, content, message):
    with pytest.raises(inbox.InboxError, match=message):
        inbox.propose(home, title, content)
    assert inbox.list_proposals(home) == ([], [])


def test_control_characters_in_a_title_become_spaces(home):
    assert made(home, "a\x1b[31mred\x07\tb").title == "a [31mred b"


def test_the_inbox_holds_at_most_a_hundred_proposals(home):
    for n in range(inbox.MAX_PENDING):
        inbox.propose(home, f"note {n}", f"text {n}")
    with pytest.raises(inbox.InboxError, match="already holds 100"):
        inbox.propose(home, "one more", "text")
    inbox.discard(home, inbox.list_proposals(home)[0][0])
    inbox.propose(home, "one more", "text")


def test_files_that_cannot_be_read_are_reported_not_fatal_and_count_against_the_cap(home):
    good = made(home)
    root = home / "inbox"
    (root / "20261022-aaaaaa.json").write_text("not json")
    (root / "20261022-bbbbbb.json").write_text(json.dumps({"id": "20261022-cccccc", "created": "x", "title": "t", "content": "c"}))
    (root / "20261022-dddddd.json").write_text(json.dumps({"id": "20261022-dddddd", "title": 5}))
    (root / "notes.txt").write_text("ignored")
    (root / ".tmp").write_text("ignored")
    pending, unreadable = inbox.list_proposals(home)
    assert pending == [good]
    assert unreadable == ["20261022-aaaaaa.json", "20261022-bbbbbb.json", "20261022-dddddd.json"]


def test_an_inbox_that_is_a_symlink_or_a_file_is_refused(home, tmp_path):
    home.mkdir()
    (tmp_path / "elsewhere").mkdir()
    (home / "inbox").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(inbox.InboxError, match="not a directory"):
        inbox.propose(home, "t", "c")
    assert list((tmp_path / "elsewhere").iterdir()) == []


def test_a_symlink_in_place_of_a_proposal_is_not_read(home, tmp_path):
    made(home)
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps({"id": "20261022-ffffff", "created": "c", "title": "t", "content": "stolen"}))
    (home / "inbox" / "20261022-ffffff.json").symlink_to(secret)
    pending, unreadable = inbox.list_proposals(home)
    assert [p.content for p in pending] == ["The grinder is set to 14."] and unreadable == ["20261022-ffffff.json"]


def test_a_looser_inbox_directory_is_tightened(home):
    made(home)
    os.chmod(home / "inbox", 0o755)
    made(home, "other", "text")
    assert stat.S_IMODE((home / "inbox").stat().st_mode) == 0o700


# ---- finding


def test_a_proposal_is_found_by_the_start_of_its_id(home):
    p = made(home)
    assert inbox.find(home, p.id) == p and inbox.find(home, p.id[:10]) == p
    with pytest.raises(inbox.InboxError, match="no pending proposal"):
        inbox.find(home, "99999999")
    with pytest.raises(inbox.InboxError, match="give a proposal id"):
        inbox.find(home, " ")


def test_an_ambiguous_start_of_an_id_is_refused(home, monkeypatch):
    tokens = iter(["aaaaaa", "aaaabb"])
    monkeypatch.setattr(inbox.secrets, "token_hex", lambda n: next(tokens))
    a, b = made(home, "a", "1"), made(home, "b", "2")
    with pytest.raises(inbox.InboxError, match="matches 2"):
        inbox.find(home, a.id[:12])
    assert inbox.find(home, b.id) == b


def test_an_id_with_path_characters_matches_nothing(home):
    made(home)
    for ref in ("../../etc/passwd", "*", "/"):
        with pytest.raises(inbox.InboxError, match="no pending proposal"):
            inbox.find(home, ref)


# ---- approving


def test_approving_writes_a_new_note_and_drops_the_proposal(home, vault):
    p = made(home, "Grinder setting: 14!", "The grinder is set to 14.")
    rel = inbox.approve(home, vault, p)
    assert rel == f"agent-memories/{p.created[:10]}-grinder-setting-14.md"
    assert (vault / rel).read_text() == "# Grinder setting: 14!\n\nThe grinder is set to 14.\n"
    assert inbox.list_proposals(home) == ([], [])


def test_an_approved_note_is_one_the_indexer_reads_and_search_finds(home, vault, service):
    from puremind_mac.indexer import discover, index_vault

    rel = inbox.approve(home, vault, made(home, "Quokka handshake", "The quokka handshake uses port 4242."))
    assert rel in [p.relative_to(vault.resolve()).as_posix() for p in discover(vault)]
    index_vault(vault, service.store, service.embedder)
    assert [h.path for h in service.fts_search("quokka")] == [rel]


def test_a_name_that_is_taken_gets_a_number_and_nothing_is_overwritten(home, vault):
    first = inbox.approve(home, vault, made(home, "Same", "one"))
    second = inbox.approve(home, vault, made(home, "Same", "two"))
    third = inbox.approve(home, vault, made(home, "Same", "three"))
    assert [first.endswith("-same.md"), second.endswith("-same-2.md"), third.endswith("-same-3.md")] == [True] * 3
    assert (vault / first).read_text().endswith("one\n") and (vault / second).read_text().endswith("two\n")


def test_an_explicit_path_is_used_and_never_overwrites(home, vault):
    p = made(home)
    (vault / "projects" / "espresso.md").write_text("mine\n")
    with pytest.raises(inbox.InboxError, match="already exists"):
        inbox.approve(home, vault, p, "projects/espresso.md")
    assert (vault / "projects" / "espresso.md").read_text() == "mine\n"
    assert inbox.list_proposals(home)[0] == [p]  # still waiting
    assert inbox.approve(home, vault, p, "./projects//new/grinder.md") == "projects/new/grinder.md"
    assert (vault / "projects/new/grinder.md").is_file()


@pytest.mark.parametrize(
    "path",
    ["../escape.md", "/etc/x.md", "a/../../x.md", ".hidden/x.md", "a/.x.md", "node_modules/x.md", "x.txt", "", "a/\x00.md", "a\\..\\x.md", "dir/"],
)
def test_paths_the_indexer_would_not_read_are_refused(home, vault, path):
    p = made(home)
    with pytest.raises(inbox.InboxError):
        inbox.approve(home, vault, p, path)
    assert inbox.list_proposals(home)[0] == [p]


def test_approval_does_not_write_through_a_symlinked_folder(home, vault, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (vault / "agent-memories").symlink_to(outside)
    p = made(home)
    with pytest.raises(inbox.InboxError, match="symlink"):
        inbox.approve(home, vault, p)
    with pytest.raises(inbox.InboxError, match="symlink"):
        inbox.approve(home, vault, p, "agent-memories/deeper/x.md")
    assert list(outside.iterdir()) == [] and inbox.list_proposals(home)[0] == [p]


def test_approval_does_not_write_through_a_symlinked_note(home, vault, tmp_path):
    target = tmp_path / "target.md"
    (vault / "agent-memories").mkdir()
    (vault / "agent-memories" / "x.md").symlink_to(target)
    with pytest.raises(inbox.InboxError, match="already exists"):
        inbox.approve(home, vault, made(home), "agent-memories/x.md")
    assert not target.exists()


def test_a_file_where_a_folder_is_needed_is_refused(home, vault):
    (vault / "agent-memories").write_text("a file")
    with pytest.raises(inbox.InboxError, match="not a folder"):
        inbox.approve(home, vault, made(home))


def test_the_folder_comes_from_the_setting_and_is_checked(home, vault):
    p = made(home)
    assert inbox.approve(home, vault, p, folder="from agents/2026").startswith("from agents/2026/")
    for folder in ("", "..", "../x", "/abs", ".hidden", "a/../b"):
        with pytest.raises(inbox.InboxError):
            inbox.approve(home, vault, made(home, folder or "e", "c"), folder=folder)


def test_titles_of_every_kind_make_usable_file_names():
    for title, slug in (("Straße & Café", "strasse-café"), ("日本語のメモ", "日本語のメモ"), ("!!!", "note"), ("a" * 300, "a" * 60)):
        assert inbox.slugify(title) == slug


def test_ages_are_short_phrases():
    p = inbox.Proposal("20261022-aaaaaa", "2026-10-22T10:00:00Z", "t", "c")
    base = 1_792_663_200.0  # 2026-10-22T10:00:00Z
    assert inbox.age(p, base + 120) == "2 min" and inbox.age(p, base + 7200) == "2 h" and inbox.age(p, base + 3 * 86400) == "3 d"
    assert inbox.age(dataclasses.replace(p, created="garbage")) == "unknown age"


# ---- the tool


def test_the_tool_saves_a_proposal_and_says_it_is_not_remembered_yet(service):
    out = tools.puremind_propose(service, "Grinder", "Set to 14.")
    assert "saved" in out and "not in the user's notes" in out
    again = tools.puremind_propose(service, "Grinder", "Set to 14.")
    assert "already waiting" in again and again.split()[1] == out.split()[1]
    assert len(inbox.list_proposals(service.cfg.home)[0]) == 1


def test_the_tool_refuses_with_a_short_message_that_names_no_path(service):
    for title, content in (("", "x"), ("t", ""), ("t", "x" * 20_001)):
        with pytest.raises(ToolRefused) as excinfo:
            tools.puremind_propose(service, title, content)
        assert str(service.cfg.home) not in str(excinfo.value)


def test_the_tool_reports_a_disk_error_without_a_path(service, monkeypatch):
    def broken(*args):
        raise PermissionError(13, "Permission denied", str(service.cfg.home))

    monkeypatch.setattr(inbox, "propose", broken)
    with pytest.raises(ToolRefused) as excinfo:
        tools.puremind_propose(service, "t", "c")
    assert str(excinfo.value) == "the proposal could not be saved"


def test_a_proposal_is_not_searchable_or_readable_until_approved(service, vault):
    from puremind_mac.indexer import index_vault

    tools.puremind_propose(service, "Zebra protocol", "Quokka handshake details.")
    index_vault(vault, service.store, service.embedder)
    assert service.fts_search("quokka") == []
    with pytest.raises(ToolRefused):
        tools.puremind_get(service, "inbox/anything.json")


# ---- the command line


@pytest.fixture
def cli_home(tmp_path, monkeypatch, vault):
    path = tmp_path / "home"
    monkeypatch.setenv("PUREMIND_HOME", str(path))
    monkeypatch.setenv("PUREMIND_VAULT", str(vault))
    return path


def test_inbox_lists_shows_and_rejects(cli_home, capsys):
    assert cli.main(["inbox"]) == 0
    assert "inbox is empty" in capsys.readouterr().out
    p = made(cli_home, "Grinder\x1b[2J", "Line one.\x1b]0;evil\x07 Line two.")
    assert cli.main(["inbox", "list"]) == 0
    out = capsys.readouterr().out
    assert p.id in out and "1 proposal(s) waiting" in out and "\x1b" not in out
    assert cli.main(["inbox", "show", p.id[:11]]) == 0
    shown = capsys.readouterr().out
    assert "Line one." in shown and "\x1b" not in shown and "\x07" not in shown
    assert cli.main(["inbox", "reject", p.id]) == 0
    assert "rejected" in capsys.readouterr().out and inbox.list_proposals(cli_home) == ([], [])
    assert cli.main(["inbox", "show", p.id]) == 2
    assert "no pending proposal" in capsys.readouterr().err


def test_inbox_list_warns_about_files_it_cannot_read(cli_home, capsys):
    made(cli_home)
    (cli_home / "inbox" / "20261022-aaaaaa.json").write_text("{")
    assert cli.main(["inbox"]) == 0
    assert "could not be read" in capsys.readouterr().err


def test_approving_without_a_terminal_needs_yes(cli_home, vault, capsys, monkeypatch):
    p = made(cli_home)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # not a terminal
    assert cli.main(["inbox", "approve", p.id]) == 2
    assert "pass --yes" in capsys.readouterr().err
    assert not (vault / "agent-memories").exists() and inbox.list_proposals(cli_home)[0] == [p]
    assert cli.main(["inbox", "approve", p.id, "--yes"]) == 0
    out = capsys.readouterr().out
    assert "wrote agent-memories/" in out and "puremind index" in out
    assert inbox.list_proposals(cli_home) == ([], [])


class _Terminal:
    def isatty(self):
        return True


@pytest.mark.parametrize(("answer", "written"), [("y", True), ("yes", True), ("", False), ("n", False)])
def test_approving_at_a_terminal_asks_first(cli_home, vault, capsys, monkeypatch, answer, written):
    p = made(cli_home)
    monkeypatch.setattr("sys.stdin", _Terminal())
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    assert cli.main(["inbox", "approve", p.id]) == 0
    assert (vault / "agent-memories").exists() is written
    assert (inbox.list_proposals(cli_home)[0] == []) is written
    assert ("not written" in capsys.readouterr().out) is (not written)


def test_approving_at_a_closed_terminal_writes_nothing(cli_home, vault, monkeypatch):
    p = made(cli_home)
    monkeypatch.setattr("sys.stdin", _Terminal())

    def eof(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert cli.main(["inbox", "approve", p.id]) == 0
    assert not (vault / "agent-memories").exists()


def test_approve_takes_a_path_and_reports_problems_as_errors(cli_home, vault, capsys):
    p = made(cli_home)
    assert cli.main(["inbox", "approve", p.id, "--yes", "--path", "../x.md"]) == 2
    assert "not a note path" in capsys.readouterr().err
    assert cli.main(["inbox", "approve", p.id, "--yes", "--path", "mine/grinder.md"]) == 0
    assert (vault / "mine" / "grinder.md").is_file()
    q = made(cli_home, "Other", "text")
    assert cli.main(["inbox", "approve", q.id, "--yes", "--path", "mine/grinder.md"]) == 2
    assert "already exists" in capsys.readouterr().err


def test_approve_uses_the_configured_folder(cli_home, vault, capsys):
    cli_home.mkdir(parents=True, exist_ok=True)
    (cli_home / "config.toml").write_text('[inbox]\nfolder = "from-agents"\n')
    assert cli.main(["inbox", "approve", made(cli_home).id, "--yes"]) == 0
    assert (vault / "from-agents").is_dir()


def test_approve_needs_a_vault(cli_home, monkeypatch, capsys):
    monkeypatch.delenv("PUREMIND_VAULT")
    assert cli.main(["inbox", "approve", made(cli_home).id, "--yes"]) == 2
    assert "no vault set" in capsys.readouterr().err


def test_serve_has_a_read_only_switch():
    assert cli.build_parser().parse_args(["serve", "--read-only"]).read_only is True
    assert cli.build_parser().parse_args(["serve"]).read_only is False


# ---- configuration


def test_the_inbox_settings_default_to_on_and_round_trip(tmp_path):
    assert Config().inbox_enabled is True and Config().inbox_folder == "agent-memories"
    cfg = dataclasses.replace(Config(home=tmp_path), inbox_enabled=False, inbox_folder='my "agents"')
    (tmp_path / "config.toml").write_text(render_config(tmp_path / "vault", cfg))
    loaded = load_file_config(tmp_path)
    assert loaded.inbox_enabled is False and loaded.inbox_folder == 'my "agents"'


@pytest.mark.parametrize("text", ["[inbox]\nenabled = 1\n", "[inbox]\nfolder = 5\n", "inbox = 3\n"])
def test_bad_inbox_settings_name_the_file(tmp_path, text):
    (tmp_path / "config.toml").write_text(text)
    with pytest.raises(ConfigError, match="config.toml"):
        load_file_config(tmp_path)


def test_a_config_file_from_before_the_inbox_existed_turns_it_on(tmp_path):
    (tmp_path / "config.toml").write_text('vault = "/x"\n[retrieval]\nrerank = false\n')
    cfg = load_file_config(tmp_path)
    assert cfg.inbox_enabled is True and cfg.inbox_folder == "agent-memories"
