# Homebrew formula for pureMind for Mac.
#
# There is no tap yet: this file is kept in the repository so that it is reviewed and tested with the
# code. Install it from a tap of your own (see "Homebrew" in the README). Bump `url` and `sha256` at
# every release: the tarball of the release tag, `shasum -a 256` of the file GitHub serves.
class Puremind < Formula
  include Language::Python::Virtualenv

  desc "Local Markdown memory for AI coding agents, served over MCP"
  homepage "https://github.com/puretensor/puremind-mac"
  url "https://github.com/puretensor/puremind-mac/archive/refs/tags/v0.1.5.tar.gz"
  sha256 "2515a2acb7fea942049157b99371daec9612a14ea84b3d381bd845b64801539c"
  license "Apache-2.0"
  head "https://github.com/puretensor/puremind-mac.git", branch: "main"

  # The PyTorch releases pureMind needs have no Intel macOS builds.
  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.13"

  def install
    # Dependencies (PyTorch, sentence-transformers, the MCP SDK and the rest) are installed from PyPI at
    # their pinned bounds in pyproject.toml, as for `pipx install`: they are too large and too fast-moving
    # to list as resources. The two models are not part of the formula; they are downloaded into the
    # Hugging Face cache by the first `puremind index` and the first search.
    virtualenv_create(libexec, "python3.13")
    system libexec/"bin/python", "-m", "pip", "install", "--no-cache-dir", buildpath
    bin.install_symlink libexec/"bin/puremind"
  end

  def caveats
    <<~EOS
      Two models (about 3.5 GB) are downloaded the first time they are needed:
        puremind init ~/Notes && puremind index && puremind search "a question"
      Then print the setup for your agent with:
        puremind connect claude-code   # or codex, cursor
    EOS
  end

  test do
    ENV["PUREMIND_HOME"] = testpath/".puremind"
    assert_match version.to_s, shell_output("#{bin}/puremind --version")
    (testpath/"notes").mkpath
    (testpath/"notes/a.md").write("# A note\n")
    system bin/"puremind", "init", testpath/"notes"
    assert_path_exists testpath/".puremind/config.toml"
    assert_match "claude mcp add", shell_output("#{bin}/puremind connect claude-code")
  end
end
