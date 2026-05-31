"""Data-analysis tools over the Bitext customer-service dataset.

These plain functions inspect, count, and sample rows of the cached Bitext
DataFrame so a ReAct agent can answer questions about customer-support data.
Each function's docstring is surfaced to the LLM as the tool description, so it
states WHEN to use the tool and WHAT it returns.

Conventions:
  * Category filters are case-insensitive and normalized to UPPERCASE.
  * Intent filters are case-insensitive and normalized to lowercase.
  * Keyword filters are case-insensitive substring matches on the customer
    'instruction' column.
  * Unknown filters simply yield 0 rows / an empty list (never raise).
  * Sampling is deterministic (head of the filtered frame) for reproducible,
    easily graded output.

Exports:
  RAW_TOOLS  -- list of plain undecorated callables (used by the MCP server).
  ALL_TOOLS  -- list of LangChain StructuredTools (used by the graph).
  plus each plain function importable by name.
"""
from __future__ import annotations

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.tools.dataset import load_bitext

# Maximum number of rows any sampling tool will return, to keep outputs small.
MAX_SAMPLE = 50

# Keys returned for each example row.
_EXAMPLE_KEYS = ("category", "intent", "instruction", "response")


def _apply_filters(
    df: pd.DataFrame,
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
) -> pd.DataFrame:
    """Return the subset of ``df`` matching the given filters.

    Category is matched case-insensitively (normalized UPPERCASE), intent
    case-insensitively (normalized lowercase), and keyword as a
    case-insensitive substring of the 'instruction' column. Filters left as
    ``None`` are ignored. Unknown values yield an empty frame rather than
    raising.
    """
    result = df
    if category is not None:
        result = result[result["category"].str.upper() == category.strip().upper()]
    if intent is not None:
        result = result[result["intent"].str.lower() == intent.strip().lower()]
    if keyword is not None:
        result = result[
            result["instruction"].str.contains(keyword.strip(), case=False, na=False)
        ]
    return result


def _rows_to_dicts(df: pd.DataFrame) -> list[dict]:
    """Convert a filtered frame into a list of example dicts."""
    return [
        {key: row[key] for key in _EXAMPLE_KEYS}
        for _, row in df.iterrows()
    ]


def list_categories() -> list[str]:
    """List the 11 top-level support categories (e.g. REFUND, ORDER, PAYMENT).

    Use this FIRST to discover the available top-level categories before
    filtering or counting. Returns a sorted list of unique category labels.
    """
    df = load_bitext()
    return sorted(df["category"].str.upper().unique().tolist())


def list_intents(category: str | None = None) -> list[str]:
    """List the fine-grained intent labels, optionally within one category.

    Use this to map a user's phrasing (e.g. 'refunds') to the exact intent
    label (e.g. 'get_refund') before counting or filtering. Pass ``category``
    to restrict the intents to a single top-level category. Returns a sorted
    list of unique intent labels (empty if the category is unknown).
    """
    df = _apply_filters(load_bitext(), category=category)
    return sorted(df["intent"].str.lower().unique().tolist())


def count_rows(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
) -> int:
    """Count rows matching the given category / intent / keyword filters.

    Use this to answer "how many" questions, e.g. how many examples are in the
    REFUND category, have intent 'get_refund', or whose customer message
    contains a keyword. The keyword is a case-insensitive substring of the
    customer 'instruction'. Returns the integer row count (0 if nothing
    matches).
    """
    return int(len(_apply_filters(load_bitext(), category, intent, keyword)))


def sample_examples(
    n: int = 3,
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
) -> list[dict]:
    """Return up to ``n`` example rows matching the given filters.

    Use this to inspect real customer messages and their responses. For
    summarization questions pass a larger ``n`` (15-25) to give the model more
    context. Sampling is deterministic (first matching rows). Each row is a
    dict with keys: category, intent, instruction, response. Returns an empty
    list if nothing matches. ``n`` is capped at 50.
    """
    capped = max(0, min(int(n), MAX_SAMPLE))
    df = _apply_filters(load_bitext(), category, intent, keyword)
    return _rows_to_dicts(df.head(capped))


def intent_distribution(category: str | None = None) -> dict[str, int]:
    """Return row counts per intent, sorted from most to least common.

    Use this to see which intents dominate the data, optionally within one
    category. Returns a dict mapping intent label -> row count, ordered
    descending by count (empty if the category is unknown).
    """
    df = _apply_filters(load_bitext(), category=category)
    counts = df["intent"].str.lower().value_counts()
    return {intent: int(count) for intent, count in counts.items()}


def category_distribution() -> dict[str, int]:
    """Return row counts per top-level category, sorted descending.

    Use this for an at-a-glance overview of how many examples each of the 11
    categories has. Returns a dict mapping category label -> row count, ordered
    from most to least common.
    """
    df = load_bitext()
    counts = df["category"].str.upper().value_counts()
    return {category: int(count) for category, count in counts.items()}


def search_examples(keyword: str, n: int = 5) -> list[dict]:
    """Return up to ``n`` rows whose customer message contains ``keyword``.

    Use this for fuzzy phrasing like 'people wanting their money back' -- pass
    a representative keyword ('money back', 'refund') to find matching customer
    messages. Matching is a case-insensitive substring of the 'instruction'.
    Each row is a dict with keys: category, intent, instruction, response.
    Returns an empty list if nothing matches. ``n`` is capped at 50.
    """
    capped = max(0, min(int(n), MAX_SAMPLE))
    df = _apply_filters(load_bitext(), keyword=keyword)
    return _rows_to_dicts(df.head(capped))


# --------------------------------------------------------------------------- #
# Pydantic v2 input schemas (one per tool that takes parameters).
# --------------------------------------------------------------------------- #
class ListIntentsInput(BaseModel):
    """Arguments for list_intents."""

    category: str | None = Field(
        default=None,
        description="Optional top-level category to restrict intents to "
        "(case-insensitive, e.g. 'REFUND'). Omit to list all intents.",
    )


class CountRowsInput(BaseModel):
    """Arguments for count_rows."""

    category: str | None = Field(
        default=None,
        description="Optional top-level category filter (case-insensitive, "
        "e.g. 'REFUND').",
    )
    intent: str | None = Field(
        default=None,
        description="Optional intent filter (case-insensitive, e.g. "
        "'get_refund').",
    )
    keyword: str | None = Field(
        default=None,
        description="Optional case-insensitive substring to match in the "
        "customer 'instruction' message.",
    )


class SampleExamplesInput(BaseModel):
    """Arguments for sample_examples."""

    n: int = Field(
        default=3,
        description="Number of examples to return (capped at 50). Use 15-25 "
        "for summarization questions.",
    )
    category: str | None = Field(
        default=None,
        description="Optional top-level category filter (case-insensitive).",
    )
    intent: str | None = Field(
        default=None,
        description="Optional intent filter (case-insensitive).",
    )
    keyword: str | None = Field(
        default=None,
        description="Optional case-insensitive substring to match in the "
        "customer 'instruction' message.",
    )


class IntentDistributionInput(BaseModel):
    """Arguments for intent_distribution."""

    category: str | None = Field(
        default=None,
        description="Optional top-level category to restrict the distribution "
        "to (case-insensitive).",
    )


class SearchExamplesInput(BaseModel):
    """Arguments for search_examples."""

    keyword: str = Field(
        description="Case-insensitive substring to match in the customer "
        "'instruction' message (e.g. 'money back').",
    )
    n: int = Field(
        default=5,
        description="Number of matching examples to return (capped at 50).",
    )


# --------------------------------------------------------------------------- #
# Tool registries.
# --------------------------------------------------------------------------- #
RAW_TOOLS: list = [
    list_categories,
    list_intents,
    count_rows,
    sample_examples,
    intent_distribution,
    category_distribution,
    search_examples,
]

ALL_TOOLS: list[StructuredTool] = [
    StructuredTool.from_function(list_categories),
    StructuredTool.from_function(
        func=list_intents,
        name="list_intents",
        description=list_intents.__doc__,
        args_schema=ListIntentsInput,
    ),
    StructuredTool.from_function(
        func=count_rows,
        name="count_rows",
        description=count_rows.__doc__,
        args_schema=CountRowsInput,
    ),
    StructuredTool.from_function(
        func=sample_examples,
        name="sample_examples",
        description=sample_examples.__doc__,
        args_schema=SampleExamplesInput,
    ),
    StructuredTool.from_function(
        func=intent_distribution,
        name="intent_distribution",
        description=intent_distribution.__doc__,
        args_schema=IntentDistributionInput,
    ),
    StructuredTool.from_function(category_distribution),
    StructuredTool.from_function(
        func=search_examples,
        name="search_examples",
        description=search_examples.__doc__,
        args_schema=SearchExamplesInput,
    ),
]
