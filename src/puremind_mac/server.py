"""Read-only MCP server over stdio. Exposes exactly three tools; there is no write path."""

from __future__ import annotations

try:  # mcp 2.x renamed FastMCP to MCPServer; the decorator and run() are unchanged
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from . import tools
from .service import Service


def build_server(service: Service) -> FastMCP:
    mcp = FastMCP("puremind")

    @mcp.tool()
    def puremind_search(query: str, limit: int = 10) -> str:
        """Search the user's Markdown vault. Returns matching passages with file path, heading path and score."""
        return tools.puremind_search(service, query, limit)

    @mcp.tool()
    def puremind_get(path: str) -> str:
        """Read one note in full. `path` is relative to the vault, as returned by puremind_search."""
        return tools.puremind_get(service, path)

    @mcp.tool()
    def puremind_recent(days: int = 7, limit: int = 20) -> str:
        """List notes modified in the last `days` days, newest first."""
        return tools.puremind_recent(service, days, limit)

    return mcp
