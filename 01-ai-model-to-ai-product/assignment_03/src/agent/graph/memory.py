"""Episodic (SqliteSaver) + per-user profile persistence.

This module provides two complementary kinds of memory for the agent:

1. **Episodic / conversational memory** via a LangGraph ``SqliteSaver``
   checkpointer. This stores the full message history per thread so the
   graph can resume a conversation across turns and process restarts.

2. **Semantic / profile memory** via small per-user JSON files. A profile
   captures durable facts about a user (their name, the topics they ask
   about, stated preferences, and miscellaneous known facts) that we want
   to surface in the system prompt on every turn, independent of the
   current conversation window.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

# Keys that make up a user profile, in display order.
_PROFILE_LIST_KEYS: tuple[str, ...] = ("topics", "preferences", "facts")


def build_checkpointer(db_path: str) -> SqliteSaver:
    """Create a long-lived SQLite checkpointer for the graph.

    The parent directory of ``db_path`` is created if needed. A SQLite
    connection is opened with ``check_same_thread=False`` so the saver can
    be shared across threads (the CLI re-enters the graph on each turn).

    We deliberately avoid :meth:`SqliteSaver.from_conn_string`, which returns
    a context manager that closes the connection on exit; we need the saver
    to outlive any single ``with`` block.

    Args:
        db_path: Filesystem path to the SQLite database file. Created if it
            does not already exist.

    Returns:
        A :class:`SqliteSaver` backed by a persistent connection.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


def _empty_profile() -> dict[str, Any]:
    """Return a fresh, empty profile with all expected keys.

    Returns:
        A dict with ``name`` (``None``) and empty ``topics``,
        ``preferences``, and ``facts`` lists.
    """
    return {"name": None, "topics": [], "preferences": [], "facts": []}


def _normalize_profile(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce an arbitrary dict into a well-formed profile.

    Missing keys are filled from :func:`_empty_profile`, ``name`` is kept as
    ``None`` unless it is a non-empty value, and the list-valued keys are
    forced to be lists of strings.

    Args:
        data: A (possibly partial or malformed) profile-like mapping.

    Returns:
        A new dict containing exactly the four profile keys.
    """
    profile = _empty_profile()

    name = data.get("name")
    profile["name"] = name if name else None

    for key in _PROFILE_LIST_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            profile[key] = [str(item) for item in value]
        elif value:
            # Tolerate a single scalar where a list was expected.
            profile[key] = [str(value)]
        else:
            profile[key] = []

    return profile


def _profile_path(user_id: str, profile_dir: str) -> Path:
    """Return the JSON file path for ``user_id`` under ``profile_dir``."""
    return Path(profile_dir) / f"{user_id}.json"


def load_profile(user_id: str, profile_dir: str) -> dict[str, Any]:
    """Load a user's profile from disk.

    Reads ``<profile_dir>/<user_id>.json``. If the file is missing or its
    contents cannot be parsed as a JSON object, an empty profile is
    returned. The result always contains all four profile keys.

    Args:
        user_id: Identifier used as the JSON file's base name.
        profile_dir: Directory containing per-user profile files.

    Returns:
        A normalized profile dict.
    """
    path = _profile_path(user_id, profile_dir)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return _empty_profile()

    if not isinstance(data, dict):
        return _empty_profile()

    return _normalize_profile(data)


def save_profile(user_id: str, profile_dir: str, profile: dict[str, Any]) -> None:
    """Persist a user's profile to disk as pretty-printed JSON.

    The ``profile_dir`` is created if needed. The profile is normalized
    before writing so the on-disk shape is always consistent.

    Args:
        user_id: Identifier used as the JSON file's base name.
        profile_dir: Directory in which to store the profile file.
        profile: The profile dict to write.
    """
    directory = Path(profile_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = _profile_path(user_id, profile_dir)
    normalized = _normalize_profile(profile)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, ensure_ascii=False)


def _merge_lists(old: list[str], new: list[str]) -> list[str]:
    """Union two string lists, preserving order and de-duping case-insensitively.

    Items from ``old`` come first (in their original order), followed by any
    items from ``new`` whose lowercased form has not already been seen.

    Args:
        old: Existing list of values.
        new: Incoming list of values to merge in.

    Returns:
        A new merged list.
    """
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*old, *new]:
        text = str(item)
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(text)
    return merged


def merge_profile(old: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into ``old``, returning a new profile dict.

    The ``name`` is taken from ``updates`` when truthy, otherwise the old
    name is kept. The ``topics``, ``preferences``, and ``facts`` lists are
    unioned with order preserved and duplicates removed case-insensitively.
    Neither input dict is mutated.

    Args:
        old: The existing profile (or anything profile-like).
        updates: New information to fold in.

    Returns:
        A freshly allocated, normalized, merged profile dict.
    """
    base = _normalize_profile(old)
    incoming = _normalize_profile(updates)

    merged = _empty_profile()
    merged["name"] = incoming["name"] if incoming["name"] else base["name"]
    for key in _PROFILE_LIST_KEYS:
        merged[key] = _merge_lists(base[key], incoming[key])

    return merged


def format_profile(profile: dict[str, Any]) -> str:
    """Render a profile as a human-readable, multi-line summary.

    Suitable for injection into a system prompt. Empty sections are omitted.
    If the profile carries no information at all, a fixed placeholder string
    is returned.

    Args:
        profile: The profile dict to summarize.

    Returns:
        A multi-line string, or ``"No profile information yet."`` when empty.
    """
    normalized = _normalize_profile(profile)

    lines: list[str] = []
    if normalized["name"]:
        lines.append(f"Name: {normalized['name']}")
    if normalized["topics"]:
        lines.append(f"Frequently asks about: {', '.join(normalized['topics'])}")
    if normalized["preferences"]:
        lines.append(f"Preferences: {', '.join(normalized['preferences'])}")
    if normalized["facts"]:
        lines.append(f"Known facts: {', '.join(normalized['facts'])}")

    if not lines:
        return "No profile information yet."

    return "\n".join(lines)
