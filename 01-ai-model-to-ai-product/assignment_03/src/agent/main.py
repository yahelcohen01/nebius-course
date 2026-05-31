"""Interactive REPL for the customer-service data-analyst agent.

Run with:

    PYTHONPATH=src .venv/bin/python -m agent.main --session my_session --user alice

The ``--session`` value is used as the LangGraph ``thread_id``: restarting the
CLI with the same ``--session`` resumes the prior conversation from the SQLite
checkpointer. The ``--user`` value is threaded into ``state["user_id"]`` so the
profile node can persist durable, per-user facts.

Unlike a bare chat loop, this REPL streams the agent's *reasoning*: every tool
call the agent decides to make, the (truncated) observation each tool returns,
and finally the assistant's prose answer.
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

try:  # Pretty output if rich is available; degrade gracefully otherwise.
    from rich.console import Console

    _console: Console | None = Console()
except Exception:  # pragma: no cover - rich is a declared dependency.
    _console = None

from agent.config.settings import load_settings
from agent.graph.main_graph import build_graph

# Maximum characters of a tool observation to echo before truncating.
_OBS_LIMIT = 300


def _emit(text: str, *, style: str | None = None) -> None:
    """Print ``text`` via rich (with optional ``style``) or plain ``print``.

    Args:
        text: The line to display.
        style: An optional rich style name (e.g. ``"bold cyan"``). Ignored when
            rich is unavailable.
    """
    if _console is not None:
        _console.print(text, style=style, highlight=False)
    else:
        print(text)


def _format_tool_call(call: dict[str, Any]) -> str:
    """Render a single tool call as ``name(arg=value, ...)``.

    Args:
        call: A LangChain tool-call dict with ``name`` and ``args`` keys.

    Returns:
        A compact human-readable representation of the call.
    """
    name = call.get("name", "<unknown>")
    args = call.get("args", {}) or {}
    rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"{name}({rendered})"


def _truncate(text: str, limit: int = _OBS_LIMIT) -> str:
    """Collapse whitespace and clip ``text`` to ``limit`` characters."""
    flat = " ".join(text.split())
    if len(flat) > limit:
        return flat[:limit] + "..."
    return flat


def _print_node_update(_node: str, update: dict[str, Any]) -> str:
    """Print the reasoning emitted by one node and return any final answer.

    Inspects the ``messages`` carried in a single ``stream_mode="updates"``
    event for one node. Agent tool calls and tool observations are echoed as
    intermediate reasoning; the latest non-empty assistant text is captured and
    returned so the caller can show it as the final answer.

    Args:
        node: The name of the node that emitted this update.
        update: The partial state update produced by that node.

    Returns:
        The final assistant text found in this update, or ``""`` if none.
    """
    if not isinstance(update, dict):
        return ""

    messages = update.get("messages") or []
    final_text = ""

    for message in messages:
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            for call in tool_calls:
                _emit(f"  -> tool: {_format_tool_call(call)}", style="yellow")

            content = message.content
            text = content if isinstance(content, str) else str(content)
            if text.strip():
                final_text = text

        elif isinstance(message, ToolMessage):
            content = message.content
            text = content if isinstance(content, str) else str(content)
            _emit(f"     obs: {_truncate(text)}", style="dim")

    return final_text


def _handle_turn(app: Any, text: str, user_id: str, config: dict[str, Any]) -> None:
    """Stream one user turn through the graph, printing reasoning and answer.

    Args:
        app: The compiled LangGraph application.
        text: The user's message for this turn.
        user_id: The active user id, threaded into ``state["user_id"]``.
        config: The LangGraph run config carrying the ``thread_id``.
    """
    final_answer = ""
    for event in app.stream(
        {"messages": [HumanMessage(text)], "user_id": user_id},
        config,
        stream_mode="updates",
    ):
        for node, update in event.items():
            answer = _print_node_update(node, update)
            if answer:
                final_answer = answer

    if final_answer:
        _emit(f"agent> {final_answer}", style="bold green")
    else:
        _emit("agent> (no response)", style="bold green")


def main() -> None:
    """Parse arguments, build the graph, and run the interactive REPL."""
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent (interactive REPL)."
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session id, used as the LangGraph thread_id (resumes on restart).",
    )
    parser.add_argument(
        "--user",
        default="default",
        help="User id, threaded into state['user_id'] for profile memory.",
    )
    args = parser.parse_args()

    settings = load_settings()
    app = build_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": args.session}}

    _emit("Customer Service Data Analyst Agent", style="bold cyan")
    _emit(f"  session = {args.session!r}  (same --session resumes after restart)")
    _emit(f"  user    = {args.user!r}")
    _emit(f"  router  = {settings.model_router}   agent = {settings.model_agent}")
    _emit("Type 'exit' or 'quit' (or Ctrl-D / Ctrl-C) to leave.")
    _emit("")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            _emit("\nGoodbye.", style="bold cyan")
            break

        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            _emit("Goodbye.", style="bold cyan")
            break

        try:
            _handle_turn(app, text, args.user, config)
        except Exception as exc:  # Keep the REPL alive on any single-turn error.
            _emit(f"[error] {type(exc).__name__}: {exc}", style="bold red")


if __name__ == "__main__":
    main()
