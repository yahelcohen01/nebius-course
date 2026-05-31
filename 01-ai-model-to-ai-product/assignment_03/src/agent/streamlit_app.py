"""Streamlit chat UI for the LangGraph data-analyst agent.

A thin web wrapper around the compiled agent graph
(:func:`agent.graph.main_graph.build_graph`). The sidebar exposes a
*Session ID* (used as the checkpointer ``thread_id``, so switching it
resumes a separate persisted conversation) and a *User ID* (used to load
and update the per-user semantic profile). The agent's reasoning -- tool
calls and their observations -- is surfaced in collapsible panels so the
ReAct loop is visible, and the final answer is rendered as a normal chat
message.

Run with::

    PYTHONPATH=src uv run streamlit run src/agent/streamlit_app.py
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph.main_graph import build_graph


@st.cache_resource
def get_app():
    """Build the compiled agent graph once and cache it across reruns.

    Streamlit re-executes the whole script on every interaction; the
    ``@st.cache_resource`` decorator ensures the (expensive) graph
    compilation and SQLite checkpointer setup happen a single time per
    process rather than on each rerun.

    Returns:
        The compiled LangGraph application.
    """
    return build_graph()


def _format_args(args: Any) -> str:
    """Render tool-call arguments compactly for display.

    Args:
        args: The ``args`` payload from a tool call (usually a dict).

    Returns:
        A short, single-line string representation of the arguments.
    """
    try:
        return json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(args)


def _render_tool_calls(message: AIMessage) -> None:
    """Render any tool calls requested by an AI message as status panels.

    Args:
        message: An :class:`AIMessage` that may carry ``tool_calls``.
    """
    for call in getattr(message, "tool_calls", None) or []:
        name = call.get("name", "tool")
        args = _format_args(call.get("args", {}))
        with st.status(f"tool: {name}({args})", state="complete"):
            st.caption("Calling tool...")


def _render_tool_observation(message: ToolMessage) -> None:
    """Render a tool observation (its returned content) in an expander.

    Args:
        message: A :class:`ToolMessage` produced by the tools node.
    """
    name = getattr(message, "name", None) or "tool"
    content = message.content
    text = content if isinstance(content, str) else str(content)
    with st.expander(f"observation: {name}"):
        st.code(text)


def _render_assistant_text(text: str) -> None:
    """Render a non-empty final assistant message in a chat bubble.

    Args:
        text: The assistant's answer text.
    """
    if text.strip():
        with st.chat_message("assistant"):
            st.markdown(text)


def _replay_history(history: list[dict[str, str]]) -> None:
    """Re-render stored chat history on a rerun.

    Args:
        history: A list of ``{"role", "content"}`` dicts for the active
            session.
    """
    for entry in history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])


def main() -> None:
    """Run the Streamlit chat application."""
    st.set_page_config(page_title="Bitext Data-Analyst Agent", page_icon=":bar_chart:")
    st.title("Bitext Data-Analyst Agent")

    with st.sidebar:
        st.header("Conversation")
        session_id = st.text_input("Session ID", value="default")
        user_id = st.text_input("User ID", value="default")
        st.caption(
            "Session ID is the conversation thread -- change it to start or "
            "resume a separate chat. User ID selects the saved profile."
        )

    app = get_app()

    # Per-session display history, keyed by session id so switching the
    # Session ID swaps the visible transcript.
    if "histories" not in st.session_state:
        st.session_state.histories = {}
    history: list[dict[str, str]] = st.session_state.histories.setdefault(
        session_id, []
    )

    _replay_history(history)

    prompt = st.chat_input("Ask about the Bitext customer-service dataset...")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    history.append({"role": "user", "content": prompt})

    final_answer = ""
    for update in app.stream(
        {"messages": [HumanMessage(prompt)], "user_id": user_id},
        {"configurable": {"thread_id": session_id}},
        stream_mode="updates",
    ):
        for node_output in update.values():
            if not isinstance(node_output, dict):
                continue
            for message in node_output.get("messages", []) or []:
                if isinstance(message, AIMessage):
                    _render_tool_calls(message)
                    content = message.content
                    text = content if isinstance(content, str) else str(content)
                    if text.strip():
                        final_answer = text
                        _render_assistant_text(text)
                elif isinstance(message, ToolMessage):
                    _render_tool_observation(message)

    if final_answer.strip():
        history.append({"role": "assistant", "content": final_answer})


if __name__ == "__main__":
    main()
