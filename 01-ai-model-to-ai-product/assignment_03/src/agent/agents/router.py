"""Query router: classifies an incoming query as structured / unstructured / out_of_scope."""

from __future__ import annotations
from typing import Any, Sequence
from typing import Literal
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["structured", "unstructured", "out_of_scope"] = Field(
        description=(
            "Which domain handles the question. "
            "structured -  questions with concrete, data-driven answers"
            "unstructured - open-ended questions requiring summarization"
            "Out-of-scope - questions unrelated to the dataset"
        ),
    )
    reason: str = Field(description="One sentence explaining the choice")


system_prompt = """You route questions about the Bitext customer-service support dataset.

The dataset contains customer questions and the agent responses to them, organized
into categories such as REFUND, SHIPPING, ACCOUNT, ORDER, PAYMENT, and CANCEL.

Classify the LATEST user message into exactly one route. Use the prior conversation
as context: short follow-ups like "show 3 more", "what about refunds?", or "and
shipping?" inherit the topic of the previous turn, so they are usually structured
or unstructured, NOT out_of_scope.

Routes:
- structured: concrete, data-driven answers from the dataset, such as counts, lists,
  filtering, distributions, or showing specific examples
  (e.g. "how many REFUND rows?", "list the intents", "show 5 SHIPPING examples").
- unstructured: open-ended questions about the dataset that require summarization or
  qualitative description across many rows
  (e.g. "summarize the FEEDBACK category", "how do agents respond to complaints?").
- out_of_scope: anything that cannot be answered from this customer-service dataset,
  including general knowledge, world facts (sports, history, politics), creative
  writing, math puzzles, and unrelated software questions. Sports results, celebrity
  trivia, and "who won X" questions are ALWAYS out_of_scope.

IN-SCOPE meta questions: questions about THIS conversation, about the user, or about
what you remember are NOT out_of_scope -- the agent can answer them from the
conversation history and the stored user profile. Classify them as `structured`.
Examples: "what do you remember about me?", "what's my name?", "what was the first
thing I asked?", "what have we discussed?", "what should I query next?".

You MUST respond by populating the RouteDecision structure: a `route` field (one of
structured / unstructured / out_of_scope) and a one-sentence `reason`. Do not reply
with free-form prose."""


class _Router:
    """Callable wrapper that injects the routing system prompt before classifying.

    Wrapping the structured-output model lets callers pass only the recent
    conversation messages to :meth:`invoke`; the routing instructions are
    prepended automatically so the model reliably emits a
    :class:`RouteDecision` rather than conversational prose.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def invoke(self, messages: Sequence[BaseMessage]) -> RouteDecision:
        """Classify the latest turn given recent conversation ``messages``.

        The router model occasionally emits Harmony ``<tool_call>`` tags instead
        of clean JSON, which fails structured-output parsing. We retry a few
        times and, as a last resort, fall back to a conservative in-scope
        ``structured`` route so a transient formatting glitch never aborts the
        whole turn.

        Args:
            messages: Recent conversation messages used as routing context.

        Returns:
            The model's :class:`RouteDecision`.
        """
        prompt = [SystemMessage(content=system_prompt), *messages]
        last_error: Exception | None = None
        for _ in range(5):
            try:
                return self._model.invoke(prompt)
            except Exception as exc:  # noqa: BLE001 - retry any parse/transport error
                last_error = exc
        return RouteDecision(
            route="structured",
            reason=(
                "Router structured-output parsing failed after retries; "
                f"defaulting to in-scope structured. ({type(last_error).__name__})"
            ),
        )


def create_router(modelName: str, api_key: str, base_url: str) -> _Router:
    """Build a structured-output query router.

    Args:
        modelName: Router model id (a small, fast model that supports
            structured output, e.g. ``"openai/gpt-oss-120b-fast"``).
        api_key: Nebius Token Factory API key.
        base_url: Nebius Token Factory base URL.

    Returns:
        A router whose ``.invoke(messages)`` returns a :class:`RouteDecision`.
    """
    model = ChatOpenAI(
        base_url=base_url,
        model=modelName,
        api_key=api_key,
        temperature=0,
    )
    # Pin method="json_schema": the gpt-oss router otherwise defaults to
    # function-calling and emits Harmony <tool_call> tags that fail to parse.
    return _Router(model.with_structured_output(RouteDecision, method="json_schema"))
