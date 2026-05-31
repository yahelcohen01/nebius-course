# Customer Service Data Analyst Agent

A LangGraph ReAct agent that answers questions about the **Bitext Customer Service** dataset, with persistent memory and an optional FastMCP server exposing its tools.

> Assignment 3 — *From AI Model to AI Agent* (Nebius). See `From-AI-Model-to-AI-Agent-Assignment 3.pdf`.

## Quickstart (5 minutes)

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
# 1. install dependencies into a local .venv
uv sync

# 2. configure secrets
cp .env.example .env
# then edit .env and set NEBIUS_API_KEY (the only value you must fill in)

# 3. talk to the agent
uv run agent --session my_session --user gabi
```

`NEBIUS_API_KEY` is **required**. The defaults in `.env.example` already point at the
working Nebius **Token Factory** endpoint and the verified models, so you should not
need to change anything else.

## Run the CLI

```bash
uv run agent --session my_session --user gabi
```

- `--session` is used as the LangGraph **`thread_id`**: it selects the episodic
  conversation. Restarting the CLI with the **same `--session`** resumes the prior
  conversation from the SQLite checkpointer at `data/checkpoints.sqlite` — the agent
  still "remembers" earlier turns after a full process restart.
- `--user` is threaded into `state["user_id"]` so the profile node can persist
  durable, per-user semantic facts to `data/profiles/<user>.json`.

The CLI streams the agent's reasoning (each tool call + a truncated observation) and
then prints the final answer.

### Example test queries

```
How many refund requests did we get?          -> structured: returns a concrete count (e.g. 2,992 REFUND rows)
Show me 2 examples of that category.           -> follow-up: reuses REFUND from the prior turn (episodic memory)
List the intents in the SHIPPING category.     -> structured: lists intent labels
Summarize how agents respond to complaints.    -> unstructured: samples rows and synthesizes themes
What should I look at next?                     -> recommender: proposes ONE follow-up query and asks you to confirm
Who won the 2024 Champions League?              -> out_of_scope: politely declined, never answered from world knowledge
```

## Run the MCP server

The same dataset tools are exposed over MCP (FastMCP 3.x, stdio transport):

```bash
uv run agent-mcp
```

Call a tool from a Python client (verified working with `fastmcp` 3.3.1):

```python
from fastmcp import Client
import asyncio

async def go():
    # Point the client at the server module; it spawns it over stdio.
    async with Client("src/agent/server.py") as c:
        print([t.name for t in await c.list_tools()])
        res = await c.call_tool("list_categories", {})
        print(res.data)   # -> ['ACCOUNT', 'CANCEL', ..., 'SUBSCRIPTION']

asyncio.run(go())
```

Run it with `PYTHONPATH=src uv run python that_snippet.py`. The seven exposed tools are
`list_categories`, `list_intents`, `count_rows`, `sample_examples`,
`intent_distribution`, `category_distribution`, and `search_examples`.

## Bonuses

### A. Streamlit chat UI

```bash
PYTHONPATH=src uv run streamlit run src/agent/streamlit_app.py
```

A web wrapper around the same compiled graph. The sidebar exposes the **Session ID**
(checkpointer `thread_id`) and **User ID** (profile), and the agent's tool calls and
observations are shown in collapsible panels so the ReAct loop is visible.

### B. Query recommender

When you ask for a suggestion ("what should I look at next?"), the agent proposes
exactly **one** follow-up query grounded in the conversation so far and your stored
profile, then asks you to confirm before running any tool.

## Debug in VS Code

`.vscode/launch.json` ships two configurations that use the local `.venv` interpreter and set `PYTHONPATH=src` so module imports resolve.

1. Open this folder (`assignment_03/`) directly in VS Code so `${workspaceFolder}` resolves correctly.
2. `Cmd+Shift+P` → **Python: Select Interpreter** → pick `./.venv/bin/python`.
3. Set a breakpoint, then open **Run & Debug** (`Cmd+Shift+D`) and pick a configuration:
   - **Debug agent CLI** — runs `python -m agent.main --session debug_session --user gabi`. Edit the `args` array in `launch.json` to pass different CLI flags.
   - **Debug current file** — runs whichever `.py` is in focus (handy for poking at `dataset.py` or `data_tools.py` standalone).
4. Press **F5** to launch.

Both configs set `"justMyCode": false` so you can step into `datasets`, `langgraph`, and other library internals.

## Architecture

```
START -> router -> (out_of_scope) -> decline -> END
                \-> agent <-> tools   (ReAct loop, bounded by MAX_ITERATIONS)
                       |  \-> fallback -> END   (loop budget exhausted)
                       \-> profile -> END       (durable user-fact extraction)
```

- **Router node** — a small, fast structured-output classifier labels each query
  `structured`, `unstructured`, or `out_of_scope`, using a window of recent messages
  so short follow-ups inherit the prior topic. Out-of-scope turns short-circuit to a
  polite refusal and never reach the data tools.
- **Agent node** — the capable tool-calling model with the pandas-backed dataset tools
  bound. It runs a ReAct loop (think → call tool → observe → repeat) capped at
  `MAX_ITERATIONS`; if the budget is exhausted it diverts to a graceful **fallback**
  message instead of looping forever.
- **Profile node** — after each answered turn, the reliable structured-output model
  extracts only *new, durable* facts about the user (name, recurring topics,
  preferences) and merges them into the per-user JSON profile, which is injected into
  the system prompt on every turn.
- **Memory** — `SqliteSaver` checkpointer (`data/checkpoints.sqlite`) for **episodic**
  per-thread memory; per-user JSON files (`data/profiles/`) for **semantic** memory.

### Models

Set in `.env`. Both run on the Nebius **Token Factory** endpoint
(`https://api.tokenfactory.us-central1.nebius.com/v1/`):

| Role   | Env var        | Default                       | Why this model                                                        |
|--------|----------------|-------------------------------|-----------------------------------------------------------------------|
| Router | `MODEL_ROUTER` | `openai/gpt-oss-120b-fast`    | Cheap and fast; reliably supports structured-output JSON-schema routing. Also reused for profile-fact extraction. |
| Agent  | `MODEL_AGENT`  | `moonshotai/Kimi-K2.6`        | Strong tool-calling and reasoning for the ReAct loop over the dataset. |

**Why two models?** Routing is a tiny, high-frequency classification task — a small,
cheap model with dependable structured output is the right tool, and it keeps every
turn cheap. The actual data analysis needs a capable model that can chain tool calls
and synthesize results, so the heavier `Kimi-K2.6` is reserved for the agent node.
The router is pinned to `with_structured_output(..., method="json_schema")` because the
gpt-oss model otherwise emits Harmony `<tool_call>` tags that fail to parse; the router
also retries a few times and falls back to a conservative in-scope route so a transient
formatting glitch never aborts a turn.

## Layout

Organized by concern, following the structure recommended in `langgraph-python-zero-to-master.md` §3.

```
src/agent/
  config/
    settings.py        settings loaded from .env
  tools/
    dataset.py         Bitext loader (HuggingFace, cached)
    data_tools.py      RAW_TOOLS (plain fns for MCP) + ALL_TOOLS (LangChain StructuredTools)
  agents/
    router.py          structured / unstructured / out_of_scope classifier (json_schema)
    base_agent.py      Nebius-backed ChatOpenAI factory
  graph/
    main_graph.py      LangGraph StateGraph wiring (router/agent/tools/fallback/profile)
    memory.py          SqliteSaver + per-user profile store
  main.py              interactive CLI entry point (`uv run agent`)
  server.py            FastMCP server entry point (`uv run agent-mcp`)
  streamlit_app.py     Streamlit chat UI (bonus A)
```
