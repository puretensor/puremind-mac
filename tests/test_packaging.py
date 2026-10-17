"""Packaging, release and test-environment invariants.

These tests guard claims the project makes outside its code: the version, the licence files, the dependency
bounds, the Python versions CI covers, and the promise that the suite never touches a developer's real index.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

import puremind_mac
from puremind_mac import cli
from puremind_mac.config import load_config
from puremind_mac.indexer import index_vault
from puremind_mac.store import Store

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]
PROBE_FLAG = "PMM_HERMETIC_PROBE"  # not PUREMIND_-prefixed: the autouse fixture removes those

# Top-level import name -> distribution name, for everything the package imports that is not standard library.
IMPORT_TO_DISTRIBUTION = {
    "bm25s": "bm25s",
    "huggingface_hub": "huggingface-hub",
    "mcp": "mcp",
    "numpy": "numpy",
    "pydantic": "pydantic",
    "sentence_transformers": "sentence-transformers",
    "Stemmer": "PyStemmer",
    "torch": "torch",
    "transformers": "transformers",
    "watchdog": "watchdog",
}
# Large, fast-moving libraries are capped at the next major release; Dependabot widens a cap once CI passes.
# transformers is declared although nothing imports it: sentence-transformers alone allows releases that cannot
# load the default embedder (Qwen3 needs 4.51.1 or newer; 4.51.0 fails without accelerate).
CAPPED_AT_NEXT_MAJOR = {
    "huggingface-hub": 2, "mcp": 3, "pydantic": 3, "sentence-transformers": 7, "torch": 3, "transformers": 6,
}


def requirements() -> dict[str, Requirement]:
    reqs = [Requirement(r) for r in PROJECT["dependencies"]]
    return {canonicalize_name(r.name): r for r in reqs}


# ---- version


def test_version_is_single_sourced_from_the_package():
    assert "version" in PROJECT["dynamic"] and "version" not in PROJECT
    source = ROOT / PYPROJECT["tool"]["hatch"]["version"]["path"]
    found = re.search(r'^__version__ = "([^"]+)"$', source.read_text(encoding="utf-8"), re.MULTILINE)
    assert found and found.group(1) == puremind_mac.__version__
    Version(puremind_mac.__version__)  # raises if it is not a valid PEP 440 version


def test_cli_reports_the_package_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"puremind {puremind_mac.__version__}"


def test_changelog_newest_entry_is_the_current_version():
    headings = re.findall(r"^## (.+)$", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.MULTILINE)
    released = [h for h in headings if not h.strip("[] ").lower().startswith("unreleased")]
    assert released, "CHANGELOG.md has no version entry"
    newest = re.match(r"\[?v?(\d+\.\d+\.\d+\S*?)\]?(\s|$)", released[0])
    assert newest, f"cannot read a version from the newest CHANGELOG heading {released[0]!r}"
    assert newest.group(1) == puremind_mac.__version__, (
        f"CHANGELOG.md's newest entry is {newest.group(1)} but the package is {puremind_mac.__version__}: "
        "add the entry for this version"
    )


def test_readme_does_not_state_another_version():
    version = r"v?(\d+\.\d+\.\d+(?:[.-]?(?:a|b|rc|dev|post)\d+)?)"
    statements = [
        re.compile(rf"(?im)^\W*version\W{{0,3}}\s*{version}\b"),  # "Version 0.1.0.", "**Version:** 0.1.0"
        re.compile(rf"(?i)\bpuremind[-_ ]mac\s*(?:==|\s)\s*{version}\b"),  # "puremind-mac 0.1.0", "puremind-mac==0.1.0"
        re.compile(rf"(?i)\bpureMind for Mac\s+{version}\b"),
        re.compile(rf"(?i)\bpuremind\s+{version}\b"),  # the output of `puremind --version`
    ]
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    stated = {m.group(1) for pattern in statements for m in pattern.finditer(text)}
    assert stated <= {puremind_mac.__version__}, (
        f"README.md states {sorted(stated - {puremind_mac.__version__})} but the package is "
        f"{puremind_mac.__version__}; the version lives in src/puremind_mac/__init__.py only"
    )


# ---- metadata


def test_licence_metadata_uses_the_spdx_expression_and_ships_notice():
    assert PROJECT["license"] == "Apache-2.0"
    assert not [c for c in PROJECT["classifiers"] if c.startswith("License ::")]  # superseded by PEP 639
    assert set(PROJECT["license-files"]) == {"LICENSE", "NOTICE"}
    assert all((ROOT / name).is_file() for name in PROJECT["license-files"])


def test_build_backend_understands_pep_639():
    (requirement,) = (Requirement(r) for r in PYPROJECT["build-system"]["requires"])
    assert canonicalize_name(requirement.name) == "hatchling"
    assert requirement.specifier.contains("1.27.0") and not requirement.specifier.contains("1.26.3")


def test_project_links_and_description():
    repo = "https://github.com/puretensor/puremind-mac"
    urls = PROJECT["urls"]
    assert {"Homepage", "Source", "Issues", "Changelog"} <= set(urls)
    assert all(url.startswith(repo) for url in urls.values())
    assert "daemon" not in PROJECT["description"].lower()  # there is no background service


def test_python_classifiers_match_the_ci_matrix():
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    matrix = re.search(r"^\s*python:\s*\[([^\]]*)\]", workflow, re.MULTILINE)
    assert matrix, "no 'python: [...]' matrix in test.yml"
    tested = {v.strip().strip("\"'") for v in matrix.group(1).split(",")}
    classified = {
        c.rsplit("::", 1)[1].strip()
        for c in PROJECT["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    }
    assert classified == tested
    assert PROJECT["requires-python"] == f">={min(tested, key=Version)}"


# ---- dependencies


def imported_third_party_modules() -> set[str]:
    modules: set[str] = set()
    for path in (ROOT / "src" / "puremind_mac").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules - set(sys.stdlib_module_names) - {"puremind_mac"}


def test_every_third_party_import_is_a_declared_dependency():
    imported = imported_third_party_modules()
    unknown = imported - set(IMPORT_TO_DISTRIBUTION)
    assert not unknown, f"add {sorted(unknown)} to IMPORT_TO_DISTRIBUTION and to pyproject.toml"
    needed = {canonicalize_name(IMPORT_TO_DISTRIBUTION[m]) for m in imported}
    assert needed <= set(requirements()), f"imported but not declared: {sorted(needed - set(requirements()))}"


def test_dependency_bounds():
    reqs = requirements()
    for name, req in reqs.items():
        assert any(spec.operator in (">=", "==", "~=") for spec in req.specifier), f"{name} has no lower bound"
    for name, major in CAPPED_AT_NEXT_MAJOR.items():
        assert reqs[name].specifier.contains(f"{major - 1}.99"), name
        assert not reqs[name].specifier.contains(f"{major}.0"), f"{name} is not capped below {major}"
    # Releases the bounds must keep out: a code-execution flaw in the model loader, an import failure with
    # current pydantic, sdist-only builds (one does not build), a yanked release, and pickle-based loading.
    assert not reqs["sentence-transformers"].specifier.contains("5.5.1")
    assert not reqs["mcp"].specifier.contains("1.21.0")
    assert not reqs["pystemmer"].specifier.contains("2.2.0.1")
    assert not reqs["bm25s"].specifier.contains("0.2.7")
    assert not reqs["torch"].specifier.contains("2.5.1")
    assert not reqs["transformers"].specifier.contains("4.51.0")


# ---- the suite does not touch the developer's machine


def test_environment_is_hermetic(tmp_path_factory):
    ambient = {k for k in os.environ if k.startswith("PUREMIND_")}
    assert ambient == {"PUREMIND_HOME"}
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    home = Path(os.environ["HOME"])
    assert Path.home() == home and home.is_relative_to(tmp_path_factory.getbasetemp())
    assert Path(os.environ["PUREMIND_HOME"]) == home / ".puremind"
    cfg = load_config()
    assert (cfg.home, cfg.vault, cfg.db, cfg.embedder) == (home / ".puremind", None, None, "Qwen/Qwen3-Embedding-0.6B")


def test_child_processes_import_the_package_under_test():
    shown = subprocess.run(
        [sys.executable, "-c", "import puremind_mac; print(puremind_mac.__file__)"],
        capture_output=True, text=True, check=True,
    )
    assert Path(shown.stdout.strip()).resolve() == Path(puremind_mac.__file__).resolve()


def test_ambient_puremind_settings_cannot_reach_a_real_index(tmp_path, testing_installed):
    """PUREMIND_DB, _VAULT, _HOME and the model settings exported before pytest starts must be ignored.

    A child pytest process runs the probe below with every variable pointing at stand-ins for a user's real
    index, vault and home. Before the autouse fixture existed the probe replaced the index with the fixture
    notes embedded by the fake embedder.
    """
    real_home = tmp_path / "real-home"
    real_vault = tmp_path / "real-vault"
    real_vault.mkdir(parents=True)
    (real_vault / "private.md").write_text("# Private\n\nmy real note\n", encoding="utf-8")
    real_home.mkdir()
    (real_home / "config.toml").write_text(f'vault = "{real_vault.as_posix()}"\n', encoding="utf-8")
    real_db = tmp_path / "real-index.sqlite"
    with Store(real_db) as store:
        store.set_meta("embedder", "real/embedder")
        store.set_meta("vault_root", str(real_vault))

    def snapshot() -> dict[str, str]:
        return {
            str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(tmp_path.rglob("*"))
            if p.is_file()
        }

    before = snapshot()
    env = {
        **os.environ,
        "PUREMIND_DB": str(real_db),
        "PUREMIND_VAULT": str(real_vault),
        "PUREMIND_HOME": str(real_home),
        "PUREMIND_EMBED_MODEL": "real/embedder",
        "PUREMIND_RERANK_MODEL": "real/reranker",
        "PUREMIND_DEVICE": "mps",
        "HF_HUB_OFFLINE": "0",
        "HOME": str(tmp_path / "real-user-home"),
        PROBE_FLAG: "1",
    }
    if testing_installed:
        env["PUREMIND_TEST_INSTALLED"] = "1"
    probe = f"{Path(__file__).as_posix()}::test_probe_index_goes_to_the_temporary_home"
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", probe],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1 passed" in done.stdout, done.stdout  # a skipped probe would prove nothing
    assert snapshot() == before


@pytest.mark.skipif(os.environ.get(PROBE_FLAG) != "1", reason="runs only as the child of the test above")
def test_probe_index_goes_to_the_temporary_home(vault, embedder):
    assert os.environ["HF_HUB_OFFLINE"] == "1" and Path.home().name != "real-user-home"
    cfg = load_config()
    with Store(cfg.db_path) as store:
        index_vault(vault, store, embedder)
    assert cfg.db_path == Path(os.environ["PUREMIND_HOME"]) / "index.sqlite" and cfg.db_path.is_file()


# ---- the wheel job: the suite must be testing the installed distribution


def test_installed_mode_keeps_the_source_tree_off_sys_path():
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('probe_conftest', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "try:\n"
        "    spec.loader.exec_module(module)\n"
        "except ImportError:\n"
        "    pass  # nothing is installed in this interpreter: only sys.path matters here\n"
        "print(sys.path[0] == sys.argv[2])\n"  # conftest inserts at index 0; an editable install's .pth does not
    )
    src = str(ROOT / "src")
    for flag, expected in (("1", "False"), ("0", "True")):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # PYTHONPATH would put src on the path
        env["PUREMIND_TEST_INSTALLED"] = flag
        shown = subprocess.run(
            [sys.executable, "-c", code, str(Path(__file__).with_name("conftest.py")), src],
            capture_output=True, text=True, check=True, env=env,
        )
        assert shown.stdout.strip() == expected, f"PUREMIND_TEST_INSTALLED={flag}: {shown.stdout}{shown.stderr}"


def test_installed_distribution_is_what_is_under_test(testing_installed):
    if not testing_installed:
        pytest.skip("source-tree run: set PUREMIND_TEST_INSTALLED=1 to test an installed wheel")
    assert not Path(puremind_mac.__file__).resolve().is_relative_to(ROOT / "src")
    assert metadata.version("puremind-mac") == puremind_mac.__version__
    names = {Path(f.as_posix()).name for f in metadata.files("puremind-mac") if "licenses" in f.as_posix()}
    assert {"LICENSE", "NOTICE"} <= names
    shown = subprocess.run(
        [sys.executable, "-m", "puremind_mac.cli", "--version"], capture_output=True, text=True, check=True
    )
    assert shown.stdout.strip() == f"puremind {puremind_mac.__version__}"
