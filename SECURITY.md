# Security policy

Report vulnerabilities to security@puretensor.ai. Please do not open a public issue for a security problem.

Include the version (`puremind --version`), your operating system, and steps to reproduce. We aim to acknowledge reports within three working days.

## Scope and design notes

- The MCP server is read-only and speaks over stdio only. It opens no network port.
- `puremind_get` resolves paths and refuses anything outside the vault, including symlinks that point out of it.
- The only network access is the model download from Hugging Face on first use. Set `HF_HUB_OFFLINE=1` after the models are cached to forbid it.
- Note contents returned to an agent are untrusted text from that agent's point of view. A note can contain instructions that a model may follow. Keep untrusted material out of the vault you connect to your agents.
