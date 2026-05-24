# Customer Service Data Analyst Agent

A LangGraph ReAct agent that answers questions about the **Bitext Customer Service** dataset, with persistent memory and an optional FastMCP server exposing its tools.

> Assignment 3 — *From AI Model to AI Agent* (Nebius). See `From-AI-Model-to-AI-Agent-Assignment 3.pdf`.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
# 1. install dependencies into a local .venv
uv sync

# 2. configure secrets
cp .env.example .env
# then edit .env and set NEBIUS_API_KEY
```

## Run the CLI

```bash
uv run agent --session my_session --user gabi
```

The CLI streams the agent's reasoning (tool calls + observations) and prompts for the next message. The same `--session` ID resumes the conversation after a restart (SQLite checkpointer).

## Run the MCP server

```bash
uv run agent-mcp
```

Connecting a client (example with the FastMCP CLI):

```bash
uv run fastmcp client stdio "uv run agent-mcp"
# then call e.g.  tools/call name=list_categories arguments={}
```

## Debug in VS Code

`.vscode/launch.json` ships two configurations that use the local `.venv` interpreter and set `PYTHONPATH=src` so module imports resolve.

1. Open this folder (`assignment_03/`) directly in VS Code so `${workspaceFolder}` resolves correctly.
2. `Cmd+Shift+P` → **Python: Select Interpreter** → pick `./.venv/bin/python`.
3. Set a breakpoint, then open **Run & Debug** (`Cmd+Shift+D`) and pick a configuration:
   - **Debug agent CLI** — runs `python -m agent.main --session debug_session --user gabi`. Edit the `args` array in `launch.json` to pass different CLI flags.
   - **Debug current file** — runs whichever `.py` is in focus (handy for poking at `dataset.py` or `tools.py` standalone).
4. Press **F5** to launch.

Both configs set `"justMyCode": false` so you can step into `datasets`, `langgraph`, and other library internals.

## Architecture

- **Router node** — classifies each query as `structured`, `unstructured`, or `out_of_scope` and short-circuits the last case with a polite refusal.
- **Agent node** — Nebius Token Factory chat model with bound tools (ReAct loop, capped at `MAX_ITERATIONS`).
- **Tools** — pandas-backed dataset operations exposed both to the agent and (a subset of) via FastMCP.
- **Memory** — `SqliteSaver` checkpointer for episodic memory; a per-user JSON profile for distilled facts.

### Models

Set in `.env`:

| Role   | Env var        | Default                                       |
|--------|----------------|-----------------------------------------------|
| Router | `MODEL_ROUTER` | `meta-llama/Meta-Llama-3.1-8B-Instruct`       |
| Agent  | `MODEL_AGENT`  | `meta-llama/Meta-Llama-3.1-70B-Instruct`      |

The small model handles cheap classification; the larger one handles reasoning and tool use.

## Layout

Organized by concern, following the structure recommended in `langgraph-python-zero-to-master.md` §3.

```
src/agent/
  config/
    settings.py      settings loaded from .env
  tools/
    dataset.py       Bitext loader (HuggingFace, cached)
    data_tools.py    LangChain @tool functions + Pydantic schemas
  agents/
    router.py        structured / unstructured / out_of_scope classifier
  graph/
    main_graph.py    LangGraph StateGraph wiring
    memory.py        SqliteSaver + per-user profile store
  main.py            interactive CLI entry point (`uv run agent`)
  server.py          FastMCP server entry point (`uv run agent-mcp`)
```
