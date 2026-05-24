# LangGraph (Python) — From Zero to Master

> A complete, offline-friendly guide to building stateful LLM applications with LangGraph in Python.
> Every concept is explained with runnable code. The final section is a full production-style project.
> All code uses **LangGraph v1+ and LangChain v1+** APIs. No deprecated patterns.

---

## Table of Contents

1. [What is LangGraph and why use it](#1-what-is-langgraph-and-why-use-it)
2. [Mental model: graphs, state, and supersteps](#2-mental-model-graphs-state-and-supersteps)
3. [Installation and project setup](#3-installation-and-project-setup)
4. [Your first graph: Hello World](#4-your-first-graph-hello-world)
5. [State and reducers in depth](#5-state-and-reducers-in-depth)
6. [Nodes: the units of work](#6-nodes-the-units-of-work)
7. [Edges and conditional routing](#7-edges-and-conditional-routing)
8. [Working with messages and LLMs](#8-working-with-messages-and-llms)
9. [Tools and the ReAct pattern](#9-tools-and-the-react-pattern)
10. [Prebuilt agents: `create_agent`](#10-prebuilt-agents-create_agent)
11. [Memory and persistence with checkpointers](#11-memory-and-persistence-with-checkpointers)
12. [Human-in-the-loop with `interrupt` and `Command`](#12-human-in-the-loop-with-interrupt-and-command)
13. [Streaming responses](#13-streaming-responses)
14. [Subgraphs and composition](#14-subgraphs-and-composition)
15. [Multi-agent patterns](#15-multi-agent-patterns)
16. [Error handling, retries, and best practices](#16-error-handling-retries-and-best-practices)
17. [Debugging with LangSmith](#17-debugging-with-langsmith)
18. [Production checklist](#18-production-checklist)
19. [Complete project: Customer Support Assistant](#19-complete-project-customer-support-assistant)
20. [Appendix: common pitfalls and API cheatsheet](#20-appendix-common-pitfalls-and-api-cheatsheet)

---

## 1. What is LangGraph and why use it

**LangGraph** is a low-level orchestration framework for building **stateful, multi-step LLM applications**. You model your application as a directed graph of nodes (functions) connected by edges. A shared state object flows between nodes, and you decide which node runs next based on that state.

### Why not just chain function calls?

For simple prompts, you don't need LangGraph. A single LLM call is fine. You need LangGraph when any of these are true:

- Your app has **branching logic** — "if the user asked about billing, do X; if about tech support, do Y".
- You need **loops** — "keep calling tools until the agent has enough information".
- You want **persistence** — "remember this user's conversation across sessions".
- You need **human-in-the-loop** — "pause before executing a sensitive action and wait for approval".
- You have **multiple agents** that collaborate — a supervisor routing to specialists.
- You need **streaming** of intermediate results.
- You want **deterministic control** over how your LLM application behaves, instead of trusting a black-box agent loop.

### LangGraph vs LangChain

LangChain provides building blocks (models, tools, prompts, output parsers). LangGraph provides the **runtime** for orchestrating those blocks into reliable, stateful workflows. You use them together: LangChain for components, LangGraph for the control flow.

### LangGraph vs `AgentExecutor` (legacy)

The old LangChain `AgentExecutor` was a single opaque loop: "reason, act, observe, repeat". You had little control over what happened inside. LangGraph replaces that with an explicit graph you design yourself. You can still get a ready-made ReAct loop via `create_agent` from the `langchain` package (covered in section 10), but you can also break it apart and customize every step.

---

## 2. Mental model: graphs, state, and supersteps

### State

A Python dictionary (defined via `TypedDict` or `Pydantic BaseModel`) that represents everything your application currently knows. Each field has a type and optionally a **reducer** (a function that merges updates). Reducers matter because several nodes may write to the same field — the reducer decides how to combine their writes.

### Nodes

Functions that take the current state as input and return a **partial update**. Whatever a node returns is merged into the state via the reducers. A node never returns the whole state — only the fields it wants to change.

```python
def my_node(state):
    new_value = do_something(state["input"])
    return {"output": new_value}  # only the `output` field is updated
```

### Edges

Connections between nodes. Two kinds:

- **Normal edges**: "after node A, always go to node B".
- **Conditional edges**: "after node A, call this routing function and go wherever it says".

There are two special nodes: `START` (entry point) and `END` (exit). You connect them like any other node.

### The execution model: supersteps

LangGraph is inspired by Google's Pregel. Execution proceeds in **supersteps**: in each superstep, all nodes that received an input run (potentially in parallel), produce their updates, the reducers merge everything, and then the next superstep begins.

### The whole lifecycle

```
 invoke({ "input": ... })
        │
        ▼
      START
        │
        ▼
    ┌──────┐    read state  ┌──────┐
    │ nodeA│ ─────────────▶ │ LLM  │
    └──┬───┘    return     └──────┘
       │     {partial update}
       ▼
  reducers merge
       │
       ▼
  conditional edge picks next node
       │
       ▼
    ┌──────┐
    │ nodeB│
    └──┬───┘
       │
       ▼
      END
        │
        ▼
 returns final state
```

---

## 3. Installation and project setup

### Prerequisites

- Python 3.10 or newer (LangGraph v1 dropped 3.9)
- An API key for your LLM provider (OpenAI, Anthropic, etc.)

### Install the core packages

```bash
pip install langchain langgraph langchain-openai
# or for Anthropic:
pip install langchain langgraph langchain-anthropic
```

- `langchain` — provides `create_agent`, `tool`, middleware helpers.
- `langgraph` — the graph runtime (`StateGraph`, `MessagesState`, `interrupt`, `Command`, `MemorySaver`).
- A provider package — `langchain-openai`, `langchain-anthropic`, etc.

For persistent checkpointers (covered later):

```bash
pip install langgraph-checkpoint-sqlite
# or for Postgres in production:
pip install langgraph-checkpoint-postgres
```

### Environment variables

Create a `.env` file and load with `python-dotenv`, or export directly:

```bash
export OPENAI_API_KEY="sk-..."
# Optional, for tracing:
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="ls-..."
export LANGSMITH_PROJECT="my-first-graph"
```

### Recommended project layout

```
my_langgraph_app/
├── config/
│   └── llm.py
├── tools/
│   ├── orders.py
│   └── refunds.py
├── agents/
│   └── agents.py
├── graph/
│   └── main_graph.py
├── main.py
└── requirements.txt
```

---

## 4. Your first graph: Hello World

Let's build the smallest possible graph: one node that greets a name.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define the shape of the state
class GraphState(TypedDict):
    name: str
    greeting: str


# 2. Define a node
def greet_node(state: GraphState) -> dict:
    return {"greeting": f"Hello, {state['name']}!"}


# 3. Build the graph
graph = (
    StateGraph(GraphState)
    .add_node("greet", greet_node)
    .add_edge(START, "greet")
    .add_edge("greet", END)
    .compile()
)

# 4. Run it
result = graph.invoke({"name": "Gabriel"})
print(result)
# {'name': 'Gabriel', 'greeting': 'Hello, Gabriel!'}
```

What happened:

1. `TypedDict` defines a typed state schema with two string fields.
2. `greet_node` reads `state["name"]` and returns a partial update containing only `greeting`.
3. `StateGraph` is built with one node and two edges (`START → greet → END`).
4. `.compile()` produces a runnable graph.
5. `.invoke()` runs the graph with an initial state and returns the final merged state.

### Two-node version

```python
def greet_node(state: GraphState) -> dict:
    return {"greeting": f"Hello, {state['name']}!"}


def shout_node(state: GraphState) -> dict:
    return {"greeting": state["greeting"].upper() + "!!!"}


graph = (
    StateGraph(GraphState)
    .add_node("greet", greet_node)
    .add_node("shout", shout_node)
    .add_edge(START, "greet")
    .add_edge("greet", "shout")
    .add_edge("shout", END)
    .compile()
)

result = graph.invoke({"name": "world"})
# {'name': 'world', 'greeting': 'HELLO, WORLD!!!!'}
```

`shout_node` reads the `greeting` that `greet_node` just wrote — that's the shared state in action.

---

## 5. State and reducers in depth

State is defined using `TypedDict` (simple) or Pydantic `BaseModel` (for validation). For reducers, use `Annotated` with a reducer function.

### Default behavior: last write wins

```python
class GraphState(TypedDict):
    count: int  # no reducer → last write wins
```

If two nodes both return `{"count": 5}` and `{"count": 10}` in the same superstep, the final value is whichever was merged last.

### Custom reducers with `Annotated`

A reducer is a function `(previous, next) -> merged`. Use it when you want to accumulate values.

**Append to a list:**

```python
from typing import Annotated
from operator import add


class GraphState(TypedDict):
    log: Annotated[list[str], add]  # `add` is operator.add, which concatenates lists
```

Now any node can return `{"log": ["some entry"]}` and it will be appended, not overwritten.

**Merge dictionaries:**

```python
def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class GraphState(TypedDict):
    results: Annotated[dict[str, str], merge_dicts]
```

This is critical when multiple nodes run in parallel and each writes to a different key.

### `MessagesState`

For LLM chat state, LangGraph ships with `MessagesState` — a pre-made `TypedDict` whose `messages` field uses the `add_messages` reducer (appends new messages, handles updates by ID).

```python
from langgraph.graph import MessagesState


# Use it directly:
graph = StateGraph(MessagesState)

# Or extend it with your own fields:
class MyState(MessagesState):
    user_id: str
    route: str
```

`MessagesState` is equivalent to:

```python
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

### Pydantic state (for validation)

```python
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    name: str = Field(default="")
    count: int = Field(default=0)
```

Pydantic gives you automatic validation, defaults, and serialization. Works interchangeably with `TypedDict` in `StateGraph`.

---

## 6. Nodes: the units of work

A node is a function that takes the state and returns a **partial update dictionary**. That's the entire contract.

### Rules for nodes

1. **Never mutate the state in place.** Always return a new dict with just the fields you want to change.
2. **Return only the fields you changed.** Unchanged fields should not appear in the return value.
3. **Nodes should be small and focused.** One node = one responsibility.
4. **Nodes can be sync or async.** Use `async def` if you're doing I/O.
5. **Nodes receive a second `config` argument** with runtime configuration (thread id, callbacks, etc.).

### Example: a node that calls an API

```python
import httpx
from langchain_core.runnables import RunnableConfig


def fetch_user_node(state: GraphState, config: RunnableConfig) -> dict:
    response = httpx.get(f"https://api.example.com/users/{state['user_id']}")
    user = response.json()
    return {"user_name": user["name"]}
```

### Example: a node that calls an LLM

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def chat_node(state: MessagesState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}  # appended via add_messages reducer
```

Note: we return `{"messages": [response]}` — a list with one new message. The `add_messages` reducer appends it to the existing list.

### Async nodes

```python
async def async_chat_node(state: MessagesState) -> dict:
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}
```

---

## 7. Edges and conditional routing

Edges determine what runs after a node completes.

### Normal edges

```python
graph_builder = (
    StateGraph(GraphState)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
)
```

Linear and predictable.

### Conditional edges

A conditional edge calls a routing function after a node finishes. The function receives the state and returns the **name of the next node**.

```python
from typing import Literal


class GraphState(TypedDict):
    input: str
    category: str


def classify_node(state: GraphState) -> dict:
    text = state["input"].lower()
    if "charge" in text:
        return {"category": "billing"}
    if "error" in text:
        return {"category": "tech"}
    return {"category": "general"}


def route_question(state: GraphState) -> Literal["billing", "tech", "general"]:
    return state["category"]


graph = (
    StateGraph(GraphState)
    .add_node("classify", classify_node)
    .add_node("billing", billing_node)
    .add_node("tech", tech_node)
    .add_node("general", general_node)
    .add_edge(START, "classify")
    .add_conditional_edges(
        "classify",
        route_question,
        {"billing": "billing", "tech": "tech", "general": "general"},
    )
    .add_edge("billing", END)
    .add_edge("tech", END)
    .add_edge("general", END)
    .compile()
)

result = graph.invoke({"input": "I see an error on checkout"})
# Routes to tech_node
```

The routing function (`route_question`) is the heart of the decision. Its return value is mapped to a node name.

### Returning multiple nodes (fan-out)

If the routing function returns a list, all those nodes run in parallel:

```python
def route_multi(state: GraphState) -> list[str]:
    active = []
    if state["needs_a"]:
        active.append("nodeA")
    if state["needs_b"]:
        active.append("nodeB")
    return active
```

### Loops

Edges can form cycles. A conditional edge that returns to an earlier node creates a loop — this is how ReAct agents work.

```python
def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


graph_builder = (
    StateGraph(MessagesState)
    .add_node("agent", agent_node)
    .add_node("tools", tool_node)
    .add_edge(START, "agent")
    .add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    .add_edge("tools", "agent")  # loop back
)
```

---

## 8. Working with messages and LLMs

Most LangGraph apps revolve around a conversation: a list of messages flowing through the graph.

### Message types

From `langchain_core.messages`:

- `HumanMessage` — from the user
- `AIMessage` — from the model (may contain tool calls)
- `SystemMessage` — instructions to the model
- `ToolMessage` — the result of a tool call

### A chat graph

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")


def chat_node(state: MessagesState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


graph = (
    StateGraph(MessagesState)
    .add_node("chat", chat_node)
    .add_edge(START, "chat")
    .add_edge("chat", END)
    .compile()
)

result = graph.invoke({"messages": [HumanMessage("What is the capital of France?")]})
print(result["messages"][-1].content)
# "The capital of France is Paris."
```

### Adding a system prompt

```python
from langchain_core.messages import SystemMessage


def chat_node(state: MessagesState) -> dict:
    system = SystemMessage("You are a pirate. Answer in pirate speak.")
    response = model.invoke([system] + state["messages"])
    return {"messages": [response]}
```

Don't return the `SystemMessage` — just use it for the call.

### Structured output

Use `with_structured_output` with a Pydantic model to force typed responses:

```python
from pydantic import BaseModel


class Classification(BaseModel):
    category: str  # "billing", "tech", or "general"
    confidence: float
    reason: str


classifier = model.with_structured_output(Classification)


def classify_node(state: MessagesState) -> dict:
    result = classifier.invoke(state["messages"])
    print(result)
    # Classification(category='tech', confidence=0.92, reason='mentions error')
    return {}
```

This is how you build reliable routers — never parse free text, always use structured output.

---

## 9. Tools and the ReAct pattern

A **tool** is a function the LLM can call. LangChain provides the `@tool` decorator.

### Defining a tool

```python
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use when the user asks about weather."""
    # Pretend we fetched from an API
    return f"The weather in {city} is 22°C and sunny."
```

Three things the LLM reads when deciding whether to call it:
1. The **function name** (`get_weather`) — short and descriptive.
2. The **docstring** — acts as the description. Include when to use it.
3. The **type hints** — the LLM reads these to know what to pass.

### Binding tools to a model

```python
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke([HumanMessage("What's the weather in Paris?")])
print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'city': 'Paris'}, 'id': 'call_abc123'}]
```

The model doesn't execute the tool — it returns a `tool_calls` list telling you what it wants to call.

### Building the ReAct loop by hand

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

tools = [get_weather]
tool_node = ToolNode(tools)  # built-in node that executes tool calls
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def agent_node(state: MessagesState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


graph = (
    StateGraph(MessagesState)
    .add_node("agent", agent_node)
    .add_node("tools", tool_node)
    .add_edge(START, "agent")
    .add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    .add_edge("tools", "agent")  # after tool runs, back to agent
    .compile()
)

result = graph.invoke({"messages": [HumanMessage("What's the weather in Paris?")]})
print(result["messages"][-1].content)
```

**What happens step by step:**

1. `agent` → model decides to call `get_weather` → returns `AIMessage` with `tool_calls`
2. `should_continue` sees tool calls → routes to `"tools"`
3. `tools` → `ToolNode` executes `get_weather("Paris")` → appends a `ToolMessage`
4. Edge `tools → agent` fires → model reads the result → produces a final answer
5. `should_continue` sees no tool calls → routes to `END`

**`model.bind_tools`** gives the model the **schemas** (so it knows what to order). **`ToolNode`** has the **implementations** (the kitchen that cooks). The model decides, the ToolNode executes. Always pass the same tool list to both.

---

## 10. Prebuilt agents: `create_agent`

Writing the ReAct loop by hand is good for learning, but in practice you'll use the prebuilt helper. In LangGraph v1+, use **`create_agent`** from the `langchain` package.

> **Migration note:** The older `create_react_agent` from `langgraph.prebuilt` is **deprecated** in LangGraph v1. It still works for now, but new code should use `create_agent`. The API is similar but has important differences noted below.

### Your first `create_agent`

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant. Always respond concisely.",
)

result = agent.invoke({"messages": [HumanMessage("What's the weather in Tel Aviv?")]})
print(result["messages"][-1].content)
```

`create_agent` returns a compiled LangGraph graph. You can `invoke` it, `stream` from it, attach a checkpointer, and compose it into a bigger graph.

### Key parameters

```python
create_agent(
    model=...,              # chat model instance OR string like "openai:gpt-4o"
    tools=...,              # list of tools
    system_prompt=...,      # string or SystemMessage
    checkpointer=...,       # optional — for persistence (see section 11)
    state_schema=...,       # optional — extend state with custom fields
    context_schema=...,     # optional — runtime context type
    response_format=...,    # optional — Pydantic model for structured final output
    middleware=...,         # optional — list of middleware
    name=...,               # optional — helpful when composing multiple agents
)
```

### What changed from `create_react_agent`

| Old (`create_react_agent`) | New (`create_agent`) |
|---|---|
| Import from `langgraph.prebuilt` | Import from `langchain.agents` |
| `model` parameter (instance only) | `model` parameter (instance OR string) |
| `prompt` parameter | `system_prompt` parameter |
| `checkpointer` | `checkpointer` (same) |
| No native middleware | `middleware` system |
| No native structured responses | `response_format` with Pydantic |

### Specifying the model

```python
from langchain_openai import ChatOpenAI

# As a model instance (full control over options)
create_agent(model=ChatOpenAI(model="gpt-4o", temperature=0), tools=tools)

# As a string (convenience form — picks up API keys from env vars)
create_agent(model="openai:gpt-4o", tools=tools)

# Other supported prefixes:
# "anthropic:claude-sonnet-4-5"
# "google:gemini-2.0-flash"
```

### Structured responses

`create_agent` can return a typed final object in addition to messages:

```python
from pydantic import BaseModel


class TravelPlan(BaseModel):
    destination: str
    days: int
    activities: list[str]


agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_flights, search_hotels],
    system_prompt="You are a travel planner.",
    response_format=TravelPlan,
)

result = agent.invoke({"messages": [HumanMessage("Plan a 3-day trip to Tokyo")]})
print(result["structured_response"])
# TravelPlan(destination='Tokyo', days=3, activities=[...])
```

### Middleware

Middleware is the big new idea in `create_agent`. Instead of hooks, you pass a list of middleware that can intercept any stage: before/after the model, before/after tools, on errors.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=tools,
    system_prompt="You are a helpful assistant.",
    middleware=[
        # Auto-summarize long conversations
        SummarizationMiddleware(
            model="anthropic:claude-sonnet-4-5",
            trigger={"tokens": 4000},
        ),
        # Require human approval for sensitive tool calls
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": {"allowed_decisions": ["approve", "reject"]}},
        ),
    ],
)
```

### When to use `create_agent` vs a hand-built graph

Use **`create_agent`** when:
- You want a standard ReAct loop (reason → act → observe → repeat)
- Middleware covers your customization needs

Use **a hand-built `StateGraph`** when:
- You have non-ReAct control flow (multi-agent routing, complex branching, parallel fan-out)
- You need custom state fields and reducers
- You want explicit control over every step

The two compose well: a hand-built supervisor graph can use `create_agent` to build its sub-agents.

---

## 11. Memory and persistence with checkpointers

By default, a graph forgets everything between invocations. Pass a **checkpointer** at compile time and it remembers.

### How it works

- After every node runs, LangGraph writes the current state to the checkpointer.
- Each conversation is identified by a `thread_id` you pass in `configurable`.
- On the next invocation with the same `thread_id`, LangGraph loads the previous state, merges your new input via the reducers, and resumes.
- The caller only sends the **new** message — never the full history.

### In-memory checkpointer (for dev)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

graph = (
    StateGraph(MessagesState)
    .add_node("chat", chat_node)
    .add_edge(START, "chat")
    .add_edge("chat", END)
    .compile(checkpointer=checkpointer)
)
```

### Using a thread

```python
config = {"configurable": {"thread_id": "user-42"}}

graph.invoke({"messages": [HumanMessage("My name is Gabriel.")]}, config)
graph.invoke({"messages": [HumanMessage("What's my name?")]}, config)
# The model will see both messages and answer "Gabriel".
```

### SQLite checkpointer (for local persistence)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("./conversations.db")
```

State survives process restarts.

### PostgresSaver (for production)

```python
from langgraph.checkpoint.postgres import PostgresSaver
import os

checkpointer = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
checkpointer.setup()  # creates tables on first run
```

### Reading the current state

```python
snapshot = graph.get_state(config)
print(snapshot.values)   # current state
print(snapshot.next)     # which node(s) will run next (empty if finished)
```

### Listing state history

```python
for snap in graph.get_state_history(config):
    print(snap.values)
```

Useful for time-travel debugging — rewind to an earlier checkpoint and branch.

---

## 12. Human-in-the-loop with `interrupt` and `Command`

For sensitive operations — sending emails, executing trades, deleting records — you want a human to approve first.

### Requirements

- A **checkpointer** (interrupts rely on persistence).
- A `thread_id` in the config.

### Basic interrupt

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage


def approval_node(state: MessagesState) -> dict:
    # Pause and surface a payload to the caller
    approved = interrupt({
        "question": "Approve sending this email?",
        "draft": state["messages"][-1].content,
    })

    if approved == "yes":
        return {"messages": [AIMessage("Email sent successfully.")]}
    else:
        return {"messages": [AIMessage("Email cancelled.")]}


graph = (
    StateGraph(MessagesState)
    .add_node("approval", approval_node)
    .add_edge(START, "approval")
    .add_edge("approval", END)
    .compile(checkpointer=MemorySaver())
)
```

### Running an interruptible graph

```python
config = {"configurable": {"thread_id": "thread-1"}}

# First invocation — runs until the interrupt
result = graph.invoke(
    {"messages": [HumanMessage("Send email to team")]},
    config,
)

# Check if we're interrupted
print(result["__interrupt__"])
# [Interrupt(value={'question': 'Approve sending this email?', 'draft': '...'})]

# Resume with the human's answer
graph.invoke(Command(resume="yes"), config)
```

Key points:
- `interrupt()` is called **inside** a node. When hit, LangGraph saves state and returns.
- The value passed to `interrupt()` surfaces in `result["__interrupt__"]`.
- To resume, invoke with `Command(resume=...)`.
- The node **restarts from the beginning** when resumed.

### Streaming is more reliable for interrupts

```python
for chunk in graph.stream(
    {"messages": [HumanMessage("Send email")]},
    config,
    stream_mode="updates",
):
    if "__interrupt__" in chunk:
        print("Waiting for approval:", chunk["__interrupt__"])
        break
    print(chunk)

# Resume
for chunk in graph.stream(Command(resume="yes"), config, stream_mode="updates"):
    print(chunk)
```

---

## 13. Streaming responses

### Stream modes

| Mode | What you get |
|------|---|
| `"values"` | The full state after each superstep |
| `"updates"` | Only the fields that changed, per node |
| `"messages"` | LLM tokens as they stream |
| `"debug"` | Detailed execution info |

### Stream updates

```python
for chunk in graph.stream(
    {"messages": [HumanMessage("What's the weather in Paris?")]},
    stream_mode="updates",
):
    for node_name, update in chunk.items():
        print(f"[{node_name}]", update)
```

### Stream LLM tokens

```python
for msg, metadata in graph.stream(
    {"messages": [HumanMessage("Tell me a story")]},
    stream_mode="messages",
):
    print(msg.content, end="", flush=True)
```

### Multiple stream modes

```python
for mode, data in graph.stream(
    input_data,
    stream_mode=["updates", "messages"],
):
    if mode == "updates":
        # handle node updates
        pass
    elif mode == "messages":
        # handle token chunks
        pass
```

### Async streaming

```python
async for chunk in graph.astream(
    {"messages": [HumanMessage("hello")]},
    stream_mode="updates",
):
    print(chunk)
```

---

## 14. Subgraphs and composition

You can use a compiled graph as a node inside another graph.

```python
# Inner graph
inner_graph = (
    StateGraph(MessagesState)
    .add_node("step1", step1)
    .add_node("step2", step2)
    .add_edge(START, "step1")
    .add_edge("step1", "step2")
    .add_edge("step2", END)
    .compile()
)

# Outer graph — uses inner as a node
outer_graph = (
    StateGraph(MessagesState)
    .add_node("prepare", prepare_node)
    .add_node("process", inner_graph)  # ← the compiled subgraph
    .add_node("finalize", finalize_node)
    .add_edge(START, "prepare")
    .add_edge("prepare", "process")
    .add_edge("process", "finalize")
    .add_edge("finalize", END)
    .compile()
)
```

If the inner and outer graphs share the same state schema, state flows through transparently.

---

## 15. Multi-agent patterns

### Pattern 1: Supervisor

A central supervisor routes to specialized sub-agents.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
worker_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Three specialist agents
math_agent = create_agent(model=worker_llm, tools=[calculator])
weather_agent = create_agent(model=worker_llm, tools=[get_weather])
news_agent = create_agent(model=worker_llm, tools=[get_news])


class RouteDecision(BaseModel):
    next: str  # "math", "weather", "news", or "end"


router = router_llm.with_structured_output(RouteDecision)


class SupervisorState(MessagesState):
    next: str
    domain_results: Annotated[dict[str, str], merge_dicts]


def supervisor_node(state: SupervisorState) -> dict:
    decision = router.invoke([
        SystemMessage("Route: math, weather, news, or end."),
        *state["messages"],
    ])
    return {"next": decision.next}


def run_math(state: SupervisorState) -> dict:
    result = math_agent.invoke({"messages": state["messages"]})
    return {"domain_results": {"math": result["messages"][-1].content}}


# ... similar for run_weather, run_news


graph = (
    StateGraph(SupervisorState)
    .add_node("supervisor", supervisor_node)
    .add_node("math", run_math)
    .add_node("weather", run_weather)
    .add_node("news", run_news)
    .add_edge(START, "supervisor")
    .add_conditional_edges("supervisor", lambda s: s["next"], {
        "math": "math", "weather": "weather", "news": "news", "end": END,
    })
    .add_edge("math", "supervisor")
    .add_edge("weather", "supervisor")
    .add_edge("news", "supervisor")
    .compile()
)
```

### Pattern 2: Swarm (handoffs)

Agents pass control to each other using `Command(goto=...)`:

```python
from langgraph.types import Command


def math_agent_node(state):
    response = math_model.invoke(state["messages"])
    if "weather" in response.content:
        return Command(goto="weather_agent", update={"messages": [response]})
    return {"messages": [response]}
```

### Pattern 3: Hierarchical

A supervisor of supervisors — nest supervisor subgraphs inside a top-level supervisor.

### Key principle

Each agent sees **only its own tools and a focused prompt**. If your specialist has 10 tools, you haven't solved the original problem.

---

## 16. Error handling, retries, and best practices

### Try/except inside nodes

```python
def fetch_node(state):
    try:
        data = fetch_from_api(state["input"])
        return {"data": data, "error": None}
    except Exception as e:
        return {"data": None, "error": str(e)}
```

Then route on the error:

```python
.add_conditional_edges("fetch", lambda s: "error_handler" if s["error"] else "success")
```

### Retry with backoff

```python
import time


def with_retry(fn, attempts=3):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i * 0.1)
```

### Best practices

1. **Small nodes, single responsibility.**
2. **Structured output over free text.** Use Pydantic with `with_structured_output`.
3. **Short, specific tool docstrings.** Include "Use when..." and "Do not use when..." clauses.
4. **Limit tool payloads.** Return `{"summary": ..., "count": ..., "data": [...]}` with sliced data.
5. **Low temperature for routing and tools.** `temperature=0` for any decision node.
6. **Small model for routing, big model for reasoning.**
7. **Attach a checkpointer early.** Even in dev.
8. **Trace with LangSmith.** See section 17.
9. **Type everything.** `TypedDict` or Pydantic for all state.
10. **Never mutate state.** Return a new dict every time.

---

## 17. Debugging with LangSmith

LangSmith records every LLM call, tool invocation, state transition, token count, and latency number.

### Setup

```bash
pip install langsmith
```

Set environment variables:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=ls-...
export LANGSMITH_PROJECT=my-project
```

No code changes needed. LangChain and LangGraph automatically send traces.

### What you get

- Timeline view of every run
- Input/output for every node and LLM call
- Token counts and cost estimates
- Latency per step
- Tool call arguments and results
- Datasets for regression testing

### Local debugging

```python
for event in graph.stream(input_data, stream_mode="debug"):
    print(event)
```

---

## 18. Production checklist

- [ ] **Persistent checkpointer** — PostgresSaver, not MemorySaver.
- [ ] **Thread ID strategy** — per user/session, never global.
- [ ] **LangSmith tracing enabled** with meaningful project name.
- [ ] **Cost budgets** — track tokens per invocation; reject runaway loops.
- [ ] **Timeout per node** — don't let a single LLM call block forever.
- [ ] **Max iterations for ReAct loops** — prevent infinite tool-calling.
- [ ] **Input validation** — Pydantic validate user input before invoking.
- [ ] **Tool security** — read-only DB roles, SQL parsers, allow-lists.
- [ ] **Prompt injection defenses** — don't concatenate untrusted strings into system prompts.
- [ ] **Error handling** — every external call wrapped; graceful fallback nodes.
- [ ] **Rate limiting** — per user and per key.
- [ ] **Sensitive data handling** — don't log PII; mask tool results.
- [ ] **Observability alerts** — spikes in latency, tokens, or errors.
- [ ] **Graceful shutdown** — drain in-flight runs on SIGTERM.

---

## 19. Complete project: Customer Support Assistant

### Scenario

An e-commerce customer support assistant handling three kinds of questions:

1. **Orders** — lookup order status, list recent orders.
2. **Refunds** — check refund eligibility, initiate refunds (requires human approval).
3. **Knowledge** — general questions from a static FAQ.

### Architecture

```
          ┌────────────┐
 user ──▶ │ supervisor │
          └─────┬──────┘
                │
   ┌────────────┼────────────┐
   ▼            ▼            ▼
 orders      refunds     knowledge
 agent       agent       agent
   │            │            │
   │            ▼            │
   │      (approval          │
   │       interrupt)        │
   │            │            │
   └────────────┴────────────┘
                │
                ▼
          ┌───────────┐
          │ finalizer │
          └─────┬─────┘
                ▼
               END
```

### `config/llm.py`

```python
from langchain_openai import ChatOpenAI

router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
worker_llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

### `tools/orders.py`

```python
from langchain_core.tools import tool
import json

# Fake database
FAKE_ORDERS = [
    {"id": "ORD-1001", "user_id": "user-42", "status": "shipped", "total": 129.99},
    {"id": "ORD-1002", "user_id": "user-42", "status": "delivered", "total": 59.50},
    {"id": "ORD-1003", "user_id": "user-42", "status": "processing", "total": 249.00},
]


@tool
def get_order_by_id(order_id: str) -> str:
    """Fetch a single order by its id.
    Use when: the user asks about a specific order.
    Do NOT use when: the user wants a list of orders."""
    order = next((o for o in FAKE_ORDERS if o["id"] == order_id), None)
    if not order:
        return json.dumps({"summary": "Order not found", "data": None})
    return json.dumps({"summary": f"Order {order['id']} is {order['status']}", "data": order})


@tool
def list_recent_orders(user_id: str) -> str:
    """List the most recent orders for a user.
    Use when: the user asks 'what are my recent orders'.
    Do NOT use when: the user asks about one specific order."""
    orders = [o for o in FAKE_ORDERS if o["user_id"] == user_id]
    return json.dumps({"summary": f"Found {len(orders)} orders", "count": len(orders), "data": orders[:10]})


orders_tools = [get_order_by_id, list_recent_orders]
```

### `tools/refunds.py`

```python
from langchain_core.tools import tool
import json


@tool
def check_refund_eligibility(order_id: str) -> str:
    """Check whether an order can be refunded.
    Use when: the user asks about refunding an order."""
    eligible = order_id != "ORD-1003"  # ORD-1003 still processing
    return json.dumps({
        "summary": f"Order {order_id} {'is' if eligible else 'is not'} eligible for refund",
        "data": {"eligible": eligible},
    })


@tool
def initiate_refund(order_id: str, amount: float) -> str:
    """Execute a refund. THIS IS SENSITIVE — only call after eligibility is confirmed."""
    import time
    return json.dumps({
        "summary": f"Refund of ${amount} initiated for {order_id}",
        "data": {"refund_id": f"REF-{int(time.time())}"},
    })


refunds_tools = [check_refund_eligibility, initiate_refund]
```

### `tools/knowledge.py`

```python
from langchain_core.tools import tool
import json

FAQ = {
    "shipping": "Standard shipping takes 3-5 business days. Express is 1-2 days.",
    "returns": "You can return most items within 30 days for a full refund.",
    "payment": "We accept Visa, Mastercard, American Express, and PayPal.",
}


@tool
def search_faq(topic: str) -> str:
    """Search the FAQ for general questions about shipping, returns, or payment."""
    key = topic.lower()
    entry = next(((k, v) for k, v in FAQ.items() if k in key), None)
    if entry:
        return json.dumps({"summary": f"Found FAQ for '{entry[0]}'", "data": {"topic": entry[0], "answer": entry[1]}})
    return json.dumps({"summary": "No FAQ found", "data": None})


knowledge_tools = [search_faq]
```

### `agents/agents.py`

```python
from langchain.agents import create_agent
from config.llm import worker_llm
from tools.orders import orders_tools
from tools.refunds import refunds_tools
from tools.knowledge import knowledge_tools

orders_agent = create_agent(
    model=worker_llm,
    tools=orders_tools,
    system_prompt=(
        "You are the Orders specialist. You can look up orders and list them.\n"
        "If the question is NOT about orders, respond with exactly: OUT_OF_SCOPE.\n"
        "Never invent order data."
    ),
)

refunds_agent = create_agent(
    model=worker_llm,
    tools=refunds_tools,
    system_prompt=(
        "You are the Refunds specialist. Check eligibility first.\n"
        "NEVER initiate a refund without an amount.\n"
        "If the question is NOT about refunds, respond with exactly: OUT_OF_SCOPE."
    ),
)

knowledge_agent = create_agent(
    model=worker_llm,
    tools=knowledge_tools,
    system_prompt=(
        "You are the Knowledge specialist. Answer general FAQ questions.\n"
        "If the question is NOT a general FAQ question, respond with exactly: OUT_OF_SCOPE."
    ),
)
```

### `graph/main_graph.py`

```python
from typing import Annotated, Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from config.llm import router_llm, worker_llm
from agents.agents import orders_agent, refunds_agent, knowledge_agent


# ──────────────────── STATE ────────────────────
def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class GraphState(MessagesState):
    route: str
    user_id: str
    domain_results: Annotated[dict[str, str], merge_dicts]
    pending_refund: dict | None


# ──────────────────── SUPERVISOR ────────────────────
class RouteDecision(BaseModel):
    next: Literal["orders", "refunds", "knowledge", "end"]
    reason: str


router = router_llm.with_structured_output(RouteDecision)


def supervisor_node(state: GraphState) -> dict:
    decision = router.invoke([
        SystemMessage(
            "Route the user's request:\n"
            "- orders: looking up an order or listing orders\n"
            "- refunds: asking about a refund\n"
            "- knowledge: general FAQ (shipping, returns, payment)\n"
            "- end: user is done or question is out of scope"
        ),
        *state["messages"],
    ])
    return {"route": decision.next}


# ──────────────────── AGENT NODES ────────────────────
def orders_node(state: GraphState) -> dict:
    result = orders_agent.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


def knowledge_node(state: GraphState) -> dict:
    result = knowledge_agent.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


def refunds_node(state: GraphState) -> dict:
    result = refunds_agent.invoke({"messages": state["messages"]})
    final_message = result["messages"][-1]

    # Check if the agent tried to call initiate_refund
    if hasattr(final_message, "tool_calls"):
        refund_call = next(
            (c for c in final_message.tool_calls if c["name"] == "initiate_refund"),
            None,
        )
        if refund_call:
            order_id = refund_call["args"]["order_id"]
            amount = refund_call["args"]["amount"]

            # PAUSE for human approval
            approved = interrupt({
                "question": f"Approve refund of ${amount} for order {order_id}?",
                "order_id": order_id,
                "amount": amount,
            })

            if approved != "yes":
                return {"messages": [AIMessage(f"Refund cancelled. {approved or ''}")]}

    return {"messages": [final_message]}


# ──────────────────── GRAPH WIRING ────────────────────
support_graph = (
    StateGraph(GraphState)
    .add_node("supervisor", supervisor_node)
    .add_node("orders", orders_node)
    .add_node("refunds", refunds_node)
    .add_node("knowledge", knowledge_node)
    .add_edge(START, "supervisor")
    .add_conditional_edges("supervisor", lambda s: s["route"], {
        "orders": "orders",
        "refunds": "refunds",
        "knowledge": "knowledge",
        "end": END,
    })
    .add_edge("orders", END)
    .add_edge("refunds", END)
    .add_edge("knowledge", END)
    .compile(checkpointer=MemorySaver())
)
```

### `main.py`

```python
import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from graph.main_graph import support_graph


def main():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n=== Thread {thread_id} ===\n")

    # Turn 1: simple order lookup
    run_turn("What's the status of order ORD-1001?", config)

    # Turn 2: refund request — will trigger interrupt
    result = run_turn("I want a refund of $129.99 for order ORD-1001", config)

    # If interrupted, simulate human approving
    if result and "__interrupt__" in result:
        print("\n[HITL] Interrupt received:", result["__interrupt__"])
        print("[HITL] Simulating human approval...\n")
        for chunk in support_graph.stream(Command(resume="yes"), config, stream_mode="updates"):
            log_chunk(chunk)

    # Turn 3: general FAQ
    run_turn("How long does shipping take?", config)


def run_turn(question: str, config: dict):
    print(f"\n> {question}")
    last_result = None
    for chunk in support_graph.stream(
        {"messages": [HumanMessage(question)]},
        config,
        stream_mode="updates",
    ):
        log_chunk(chunk)
        last_result = chunk
    return last_result


def log_chunk(chunk: dict):
    for node_name, update in chunk.items():
        if node_name == "__interrupt__":
            print(f"  [INTERRUPT] {update}")
            continue
        messages = update.get("messages", [])
        if messages:
            msg = messages[-1]
            content = getattr(msg, "content", str(msg))
            print(f"  [{node_name}] {str(content)[:200]}")
        else:
            print(f"  [{node_name}] {update}")


if __name__ == "__main__":
    main()
```

### Running it

```bash
pip install langchain langgraph langchain-openai
OPENAI_API_KEY=sk-... python main.py
```

---

## 20. Appendix: common pitfalls and API cheatsheet

### Common pitfalls

**Mutating state instead of returning a partial.**

```python
# WRONG
def bad(state):
    state["count"] += 1       # never mutate
    return state               # never return the whole state

# RIGHT
def good(state):
    return {"count": state["count"] + 1}
```

**Forgetting a reducer on a shared field.**

If two parallel nodes write to `results` without a merging reducer, one overwrites the other. Use `Annotated[dict, merge_dicts]`.

**Passing full history instead of using a checkpointer.**

```python
# WRONG
history = []
def ask(q):
    history.append(HumanMessage(q))
    result = graph.invoke({"messages": history})
    history.append(result["messages"][-1])

# RIGHT
config = {"configurable": {"thread_id": "session-1"}}
def ask(q):
    return graph.invoke({"messages": [HumanMessage(q)]}, config)
```

**Overloading one agent with all tools.** Split into supervisor + specialists.

**Large tool responses.** Never return raw DB rows. Wrap in `{"summary": ..., "count": ..., "data": [...]}` and slice.

**Using `set_entry_point` / `set_finish_point`.** These are deprecated v0.1 APIs. Use `add_edge(START, ...)` and `add_edge(..., END)`.

### API cheatsheet

**State definition**

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import MessagesState


class State(TypedDict):
    count: int
    log: Annotated[list[str], add]  # append reducer


# Or extend MessagesState:
class MyState(MessagesState):
    route: str
```

**Graph construction**

```python
from langgraph.graph import StateGraph, START, END

graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_conditional_edges("a", lambda s: s["route"], {"x": "b", "y": END})
    .add_edge("b", END)
    .compile(checkpointer=checkpointer)
)
```

**Invocation**

```python
result = graph.invoke(
    {"field": "hello"},
    {"configurable": {"thread_id": "t-1"}},
)
```

**Streaming**

```python
for chunk in graph.stream(input_data, config, stream_mode="updates"):
    print(chunk)

# Async
async for chunk in graph.astream(input_data, config, stream_mode="updates"):
    print(chunk)
```

**State introspection**

```python
snap = graph.get_state(config)
print(snap.values)  # current state
print(snap.next)    # next nodes

for s in graph.get_state_history(config):
    print(s)
```

**Interrupt and resume**

```python
from langgraph.types import interrupt, Command

# Inside a node:
answer = interrupt({"question": "OK to proceed?"})

# From the runner:
graph.invoke(Command(resume="yes"), config)
```

**Checkpointers**

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

mem = MemorySaver()
sqlite = SqliteSaver.from_conn_string("./db.sqlite")
pg = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
pg.setup()
```

**Prebuilt agent (`create_agent`)**

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",         # instance OR provider string
    tools=[tool1, tool2],
    system_prompt="You are a helpful assistant.",
    checkpointer=checkpointer,     # optional
    response_format=MyPydanticModel,  # optional
    middleware=[],                  # optional
)
```

> Note: `create_react_agent` from `langgraph.prebuilt` is deprecated in v1. Migrate to `create_agent`: rename `prompt` → `system_prompt`, import from `langchain.agents`.

**Tool definition**

```python
from langchain_core.tools import tool


@tool
def my_tool(query: str) -> str:
    """What it does. Use when: ... Do NOT use when: ..."""
    return json.dumps({"summary": "...", "data": []})
```

**Structured output**

```python
from pydantic import BaseModel


class MySchema(BaseModel):
    category: str
    confidence: float


structured = llm.with_structured_output(MySchema)
result = structured.invoke(messages)
# result is MySchema(category='...', confidence=0.92)
```

---

## Where to go next

You now know enough to build any LangGraph application. The things that take practice:

1. **Designing the state schema.** Think about what each node needs to read and write.
2. **Writing tool docstrings.** This is prompt engineering. Rewrite them after failures.
3. **Choosing when to split agents.** Confused agent? Too many tools. Split it.
4. **Balancing model sizes.** Small fast model for routing, big smart model for reasoning.
5. **Observability first.** Turn on LangSmith before you write the third node.

Build something. Break it. Look at the traces. Fix it. That's the whole loop.
