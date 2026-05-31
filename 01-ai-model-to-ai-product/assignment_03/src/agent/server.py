"""FastMCP server exposing the Bitext dataset tools over the MCP protocol.

The tools registered here mirror the agent's own LangChain tools defined in
``agent.tools.data_tools``: they are the same plain Python functions, so an MCP
client sees the identical schema (derived from type hints + docstrings) and
behaviour as the in-process ReAct agent. This lets the dataset be inspected,
counted, sampled, and searched either through the graph or through any MCP
client.
"""
from __future__ import annotations

from fastmcp import FastMCP

from agent.tools.data_tools import (
    category_distribution,
    count_rows,
    intent_distribution,
    list_categories,
    list_intents,
    sample_examples,
    search_examples,
)

mcp = FastMCP("bitext-data-analyst")

# Register the dataset tools. FastMCP derives each tool's schema and
# description from the function's type hints and docstring.
for _tool in (
    list_categories,
    list_intents,
    count_rows,
    sample_examples,
    intent_distribution,
    category_distribution,
    search_examples,
):
    mcp.tool(_tool)


def main() -> None:
    """Run the FastMCP server over its default (stdio) transport."""
    mcp.run()


if __name__ == "__main__":
    main()
