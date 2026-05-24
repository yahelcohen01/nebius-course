"""FastMCP server exposing dataset tools over the MCP protocol."""
from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("bitext-data-analyst")

# TODO: register at least 3 tools from agent.tools as @mcp.tool functions.


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
