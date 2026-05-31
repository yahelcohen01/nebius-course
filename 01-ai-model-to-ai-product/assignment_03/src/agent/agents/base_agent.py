"""Factory for the Nebius-backed chat model used across the agent graph.

Provides a single helper, :func:`build_chat_model`, that constructs a
``ChatOpenAI`` client pointed at the Nebius Token Factory endpoint. Callers
are responsible for binding tools to the returned model as needed.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI


def build_chat_model(
    model: str, api_key: str, base_url: str, temperature: float = 0.0
) -> ChatOpenAI:
    """Construct a Nebius-backed chat model. Caller binds tools as needed.

    Args:
        model: The model identifier to serve (e.g. ``"moonshotai/Kimi-K2.6"``).
        api_key: Nebius Token Factory API key.
        base_url: Nebius Token Factory base URL.
        temperature: Sampling temperature; defaults to deterministic ``0.0``.

    Returns:
        A configured :class:`~langchain_openai.ChatOpenAI` client.
    """
    return ChatOpenAI(
        model=model, api_key=api_key, base_url=base_url, temperature=temperature
    )
