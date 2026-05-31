"""LangGraph data-analyst agent: router -> agent <-> tools, with checkpointed memory.

This module assembles the keystone :class:`StateGraph` for the customer-service
data-analyst agent over the Bitext dataset:

  * a **router node** classifies the latest turn (structured / unstructured /
    out_of_scope) using a recent slice of the conversation;
  * out-of-scope turns are politely **declined** without answering from general
    knowledge;
  * in-scope turns enter a ReAct **agent <-> tools** loop bounded by
    ``MAX_ITER`` to guarantee termination;
  * exceeding the loop budget routes to a **fallback** message;
  * a best-effort **profile node** extracts durable facts about the user after
    each answered turn and persists them to semantic memory.

A module-level ``graph`` (the compiled app) is exposed so LangGraph Studio can
discover it, alongside :func:`build_graph` for programmatic construction.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from agent.agents.base_agent import build_chat_model
from agent.agents.router import create_router
from agent.config.settings import load_settings
from agent.graph.memory import (
    build_checkpointer,
    format_profile,
    load_profile,
    merge_profile,
    save_profile,
)
from agent.tools.data_tools import ALL_TOOLS

# --------------------------------------------------------------------------- #
# Module-level configuration and shared clients.
# --------------------------------------------------------------------------- #
settings = load_settings()
router = create_router(
    settings.model_router, settings.nebius_api_key, settings.nebius_base_url
)
MAX_ITER: int = settings.max_iterations

# How many trailing messages to hand the router as classification context.
_ROUTER_WINDOW = 6

# The 11 top-level categories, surfaced verbatim in the agent system prompt.
_CATEGORIES = (
    "ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, "
    "PAYMENT, REFUND, SHIPPING, SUBSCRIPTION"
)


class GraphState(MessagesState):
    """Conversation state threaded through the graph.

    Extends :class:`MessagesState` (which supplies the accumulating
    ``messages`` list) with routing metadata and a step counter used to bound
    the ReAct loop.
    """

    route: str
    router_reason: str
    user_id: str
    steps: int


class ProfileUpdate(BaseModel):
    """Durable, user-specific facts extracted from a single turn.

    All fields default to empty so the extractor can signal "nothing new"
    cleanly. Only NEW, durable information about the *user* should populate
    these fields -- never transient query content or dataset facts.
    """

    name: str | None = Field(
        default=None, description="The user's name if they stated it, else null."
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Recurring dataset topics the user cares about "
        "(e.g. 'REFUND', 'shipping delays'). Empty if none new.",
    )
    preferences: list[str] = Field(
        default_factory=list,
        description="Stated preferences about how answers should be given "
        "(e.g. 'wants concise answers'). Empty if none new.",
    )
    new_facts: list[str] = Field(
        default_factory=list,
        description="Other durable facts about the user worth remembering. "
        "Empty if none new.",
    )


# --------------------------------------------------------------------------- #
# Nodes.
# --------------------------------------------------------------------------- #
def router_node(state: GraphState) -> dict:
    """Classify the current turn into a route using recent conversation context.

    Passes up to the last ``_ROUTER_WINDOW`` messages to the structured-output
    router so short follow-ups inherit the prior topic. Resets the per-turn
    step counter.

    Args:
        state: The current graph state.

    Returns:
        A partial state update with ``route``, ``router_reason``, and a reset
        ``steps`` counter.
    """
    recent = state["messages"][-_ROUTER_WINDOW:]
    decision = router.invoke(recent)
    return {"route": decision.route, "router_reason": decision.reason, "steps": 0}


def decline_node(state: GraphState) -> dict:
    """Politely decline an out-of-scope request without answering it.

    Args:
        state: The current graph state (unused; signature kept uniform).

    Returns:
        A partial state update appending a single declining ``AIMessage``.
    """
    message = AIMessage(
        content=(
            "That's outside what I can help with -- I only answer questions "
            "about the Bitext customer-service dataset (categories, intents, "
            "counts, examples, summaries). Try asking about a category like "
            "REFUND or SHIPPING."
        )
    )
    return {"messages": [message]}


def _build_agent_system_prompt(route: str, profile_text: str) -> SystemMessage:
    """Construct the route-aware system prompt for the agent node.

    Args:
        route: The classification from the router (``"structured"`` or
            ``"unstructured"``; other values fall back to structured guidance).
        profile_text: Rendered user-profile block from
            :func:`agent.graph.memory.format_profile`.

    Returns:
        A :class:`SystemMessage` for the agent LLM.
    """
    if route == "unstructured":
        route_guidance = (
            "ROUTE = unstructured: this is an open-ended / summarization "
            "question. Call sample_examples with n between 15 and 25 (filtered "
            "by the relevant category or intent) and then synthesize the "
            "patterns you see into clear prose. Describe themes and phrasing; "
            "do not just dump rows."
        )
    else:
        route_guidance = (
            "ROUTE = structured: this is a precise, data-driven question. Use "
            "the counting / listing / distribution tools and back every number "
            "with a tool call. Never fabricate or estimate numbers."
        )

    content = (
        "You are a data analyst for the Bitext customer-service support "
        "dataset. You help users explore the data by calling tools; you do not "
        "answer from outside knowledge.\n\n"
        "SCHEMA: each row has columns instruction (the customer message), "
        "category, intent, and response (the agent reply). There are 11 "
        f"top-level categories: {_CATEGORIES}. There are about 27 fine-grained "
        "intents (lowercase snake_case, e.g. cancel_order, get_refund, "
        "track_order). User phrasing rarely matches an exact intent label, so "
        "call list_intents (optionally scoped to a category) to map the user's "
        "wording to the exact intent label BEFORE counting or filtering.\n\n"
        f"{route_guidance}\n\n"
        "MEMORY: questions about the user themselves, about what you remember "
        "of them, or about earlier turns in this conversation should be answered "
        "directly from the USER PROFILE below and the conversation history -- no "
        "tools are needed for those.\n\n"
        "RECOMMENDER: if the user asks what they should query next, or for a "
        "suggestion / recommendation, propose exactly ONE relevant follow-up "
        "query grounded in the conversation so far and the user profile below, "
        "then ASK the user to confirm. Do NOT call any tool for a suggestion "
        "until the user explicitly confirms.\n\n"
        "USER PROFILE:\n"
        f"{profile_text}\n\n"
        "Be concise and always show the actual numbers."
    )
    return SystemMessage(content=content)


def agent_node(state: GraphState) -> dict:
    """Run one agent step: think and either call tools or produce an answer.

    Builds a tool-bound agent model, injects a route-aware system prompt plus
    the user's profile, and invokes it over the conversation. Increments the
    step counter so the ReAct loop stays bounded.

    Args:
        state: The current graph state.

    Returns:
        A partial state update appending the model response and bumping
        ``steps``.
    """
    model = build_chat_model(
        settings.model_agent, settings.nebius_api_key, settings.nebius_base_url
    ).bind_tools(ALL_TOOLS)

    profile = load_profile(state.get("user_id", "default"), settings.profile_dir)
    profile_text = format_profile(profile)
    system = _build_agent_system_prompt(state.get("route", "structured"), profile_text)

    response = model.invoke([system] + state["messages"])
    return {"messages": [response], "steps": state.get("steps", 0) + 1}


tools_node = ToolNode(ALL_TOOLS)


def fallback_node(state: GraphState) -> dict:
    """Emit a graceful message when the step budget is exhausted.

    Args:
        state: The current graph state (unused; signature kept uniform).

    Returns:
        A partial state update appending a single ``AIMessage``.
    """
    message = AIMessage(
        content=(
            "I wasn't able to finish this within the step limit. Could you "
            "narrow the question?"
        )
    )
    return {"messages": [message]}


def _last_human_message(state: GraphState) -> str:
    """Return the text of the most recent human message, or ``""`` if none."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _last_ai_text(state: GraphState) -> str:
    """Return the text of the most recent AI answer, or ``""`` if none."""
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def profile_node(state: GraphState) -> dict:
    """Best-effort extraction and persistence of durable user facts.

    Uses the reliable structured-output router model to extract only NEW,
    durable facts about the user from the latest human message and final AI
    answer, then merges and saves them. Any failure is swallowed so a profile
    hiccup never breaks the conversation.

    Args:
        state: The current graph state.

    Returns:
        An empty dict (this node never mutates conversation state).
    """
    try:
        uid = state.get("user_id", "default")
        extractor = build_chat_model(
            settings.model_router,
            settings.nebius_api_key,
            settings.nebius_base_url,
        ).with_structured_output(ProfileUpdate, method="json_schema")

        human = _last_human_message(state)
        answer = _last_ai_text(state)
        instruction = SystemMessage(
            content=(
                "Extract ONLY new, durable facts about the USER from the "
                "exchange below: their name (if stated), recurring topics they "
                "care about, and stated preferences about how they want "
                "answers. Do NOT record dataset facts, query results, or "
                "anything transient. Return empty lists / null when there is "
                "nothing genuinely new about the user."
            )
        )
        exchange = HumanMessage(
            content=f"User said:\n{human}\n\nAssistant answered:\n{answer}"
        )
        # The gpt-oss extractor is occasionally non-deterministic and emits
        # Harmony tags that fail structured-output parsing; retry a few times so
        # a single glitch never silently drops a freshly stated fact (e.g. a name).
        update: ProfileUpdate | None = None
        for _ in range(4):
            try:
                update = extractor.invoke([instruction, exchange])
                break
            except Exception:  # noqa: BLE001 - retry transient parse/transport errors
                continue
        if update is None:
            return {}

        merged = merge_profile(
            load_profile(uid, settings.profile_dir),
            {
                "name": update.name,
                "topics": update.topics,
                "preferences": update.preferences,
                "facts": update.new_facts,
            },
        )
        save_profile(uid, settings.profile_dir, merged)
    except Exception:
        return {}
    return {}


# --------------------------------------------------------------------------- #
# Conditional edges.
# --------------------------------------------------------------------------- #
def route_after_router(state: GraphState) -> Literal["decline", "agent"]:
    """Send out-of-scope turns to ``decline``, everything else to ``agent``."""
    if state.get("route") == "out_of_scope":
        return "decline"
    return "agent"


def route_after_agent(state: GraphState) -> Literal["tools", "fallback", "profile"]:
    """Branch the ReAct loop based on tool calls and the step budget.

    Continues to ``tools`` while the agent requests them and the budget holds;
    diverts to ``fallback`` once the budget is exhausted; otherwise the answer
    is final and we proceed to ``profile``.
    """
    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    steps = state.get("steps", 0)

    if has_tool_calls and steps < MAX_ITER:
        return "tools"
    if steps >= MAX_ITER:
        return "fallback"
    return "profile"


# --------------------------------------------------------------------------- #
# Graph assembly.
# --------------------------------------------------------------------------- #
def build_graph():
    """Build and compile the data-analyst agent graph.

    Wires the router, decline, agent, tools, fallback, and profile nodes with
    their conditional edges and attaches a SQLite checkpointer for episodic
    memory.

    Returns:
        The compiled LangGraph application.
    """
    builder = StateGraph(GraphState)

    builder.add_node("router", router_node)
    builder.add_node("decline", decline_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("profile", profile_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {"decline": "decline", "agent": "agent"},
    )
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "fallback": "fallback", "profile": "profile"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("decline", END)
    builder.add_edge("fallback", END)
    builder.add_edge("profile", END)

    return builder.compile(checkpointer=build_checkpointer(settings.checkpoint_db))


graph = build_graph()
