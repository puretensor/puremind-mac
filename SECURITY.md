# Security policy

## Reporting a vulnerability

Report it privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability**. You can also email security@puretensor.ai. Please do not open a public issue for a security problem.

Include the version (`puremind --version`), your operating system, and steps to reproduce.

This policy covers the code in this repository. Report a flaw in a dependency (the MCP SDK, sentence-transformers, SQLite and so on) or in the models to its own maintainers; tell us as well if pureMind's use of it makes the flaw reachable.

We aim to acknowledge a report within three working days and to tell you within ten working days whether we accept it. For an accepted report we agree a disclosure date with you, normally within 90 days, and credit you in the release notes unless you would rather we did not.

## Supported versions

Security fixes go into the latest 0.1.x release. Upgrade to it before reporting.

## Design notes

- **No network service.** The MCP server is read-only and speaks over stdio only. It opens no network port and has no write path to your notes.
- **What `puremind_get` serves.** Only the notes the indexer reads: regular `.md` and `.markdown` files of at most 2 MiB that, after resolving symlinks, lie inside the vault, outside hidden folders, `node_modules` and `__pycache__`. Absolute paths, `..`, NUL characters and paths over 1,024 characters are refused before the file system is touched, and refusals never contain absolute paths. A path is tried as given and then without surrounding white space, and each spelling must pass every check.
- **Network use.** The only network access is the one-time download of the two models from Hugging Face, at pinned revisions, which sends your Hugging Face token if one is configured. Once they are cached, models load from the cache and nothing contacts the network. pureMind turns Hugging Face telemetry off unless you have set `HF_HUB_DISABLE_TELEMETRY` yourself. Set `HF_HUB_OFFLINE=1` to forbid downloads altogether. A cached copy that lacks a file the models need is completed by a download, or refused when offline, so a partial copy is never loaded as if it were complete.
- **Model loading.** A model name is a Hugging Face id unless it starts with `/`, `./`, `../` or `~`, so a directory in the working directory is never loaded by accident. Models load with `trust_remote_code` off, pureMind downloads only safetensors weights (no pickle files), and `puremind serve` runs from the pureMind home directory. `sentence-transformers` 5.6.0 or newer is required because older releases could run code from a local model directory (CVE-2026-68770).
- **The index holds your notes' text.** `~/.puremind/index.sqlite` is a second copy of every note's text. It and its `-wal` and `-shm` files are created mode 0600 in a 0700 directory, and existing files you own, including `~/.puremind` and `config.toml`, are tightened when pureMind uses them. The index run that notices a deleted or edited note removes the old text from the file and its `-wal` file (SQLite zeroes the freed pages). The keyword index is compacted so it drops the deleted words at the end of that run under `puremind index`; under `puremind watch` this waits until the vault has been quiet for a minute, five minutes at most, or until the watch stops, and a watch that crashes leaves a note in the index so the next `index` or `watch` run does it. If another process is reading the index at that moment, the `-wal` file is emptied by a later run. Text deleted under 0.1.1 or earlier is removed when 0.1.2 first opens the index. Backups and snapshots of the index keep whatever they captured. To purge everything, delete the three files and re-index.
- **Hostile vault content.** The indexer only reads the vault. Symlink loops, pipes, undecodable names, oversized files, a cap of 10,000 chunks per note and a 200-character cap on each heading in a breadcrumb bound the work a vault can cause. A note is checked again just before it is read, so a symlink swapped in during a run is not followed.
- **Notes are untrusted input to an agent.** Note text returned to an agent is data, not instructions, but a model may still follow instructions written in a note. Keep untrusted material out of the vault you connect to your agents.
- **MCP SDK advisories.** The published advisories for the `mcp` package releases this project allows concern its HTTP, SSE and WebSocket transports, or its experimental task handlers (CVE-2026-52870, mcp 1.23.0 to 1.27.1). pureMind uses stdio only and never enables tasks, so none of them apply.

## How the repository is checked

CI scans the full history with gitleaks and audits the dependencies a new install resolves with pip-audit, on every push to main, every pull request and weekly. Workflow actions are pinned to commit SHAs and run with a read-only token. GitHub secret scanning with push protection and Dependabot security updates are enabled.
