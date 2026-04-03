# Single Agents

This folder contains 9 agents, each implemented twice — once in **LangGraph** and once in **MAF** (Multi-Agent Framework). Reading both versions side by side is the fastest way to understand how the two frameworks differ.

---

## Framework Comparison: LangGraph vs MAF

| | LangGraph | MAF |
|---|---|---|
| **Style** | Explicit graph: nodes + edges wired manually | Declarative: `create_agent(instructions, tools)` |
| **State** | `TypedDict` with `add_messages` reducer | Managed internally by the framework |
| **Tools** | `@tool` decorator + `ToolNode` + `bind_tools()` | Plain Python functions passed to `tools=[]` |
| **Execution** | `graph.stream()` or `graph.invoke()` | `await agent.run(user_input)` |
| **Memory** | `MemorySaver()` passed as `checkpointer` | No built-in memory — stateless per call |
| **Async** | Optional (sync by default) | Required — all `agent.run()` calls are async |
| **Routing** | `add_conditional_edges()` with router functions | Model decides internally, no explicit routing |
| **Visibility** | Full control — you see every node execution | Black box — framework handles the loop |
| **Best for** | Complex flows, HITL, custom logic, debugging | Fast prototyping, simple agents, clean code |

> **Rule of thumb:** Use LangGraph when you need to control the flow. Use MAF when you just need it to work.

---

## Agents

---

### `basic_react_agent.py`

The entry point for both frameworks. A minimal agent with two hardcoded tools — `get_weather` and `simple_search` — that returns fixed responses.

**LangGraph** (`langgraph/basic_react_agent.py`)
Manually wires `StateGraph` → `agent node` → `ToolNode` → back to agent. Uses `MemorySaver` for in-session memory. The routing is explicit: a `tool_router` function checks for `tool_calls` on the last message.

**MAF** (`maf/basic_react_agent.py`)
One `create_agent()` call with the same two tools passed as plain Python functions. No graph, no state, no routing — just `await agent.run(user_input)`. Memory is not preserved between calls.

> **Key difference:** LangGraph requires ~40 lines to wire the graph. MAF does the same in ~10. LangGraph gives you visibility into every step; MAF gives you none.

---

### `react_agent_websearch.py`

Connects the agent to the **Tavily web search API** for live internet results.

**LangGraph** (`langgraph/react_agent_websearch.py`)
Uses `TavilySearchResults` from LangChain, bound via `bind_tools()`. Includes verbose debug logging at every node — prints message types, tool calls, and routing decisions. Good for understanding the message flow.

**MAF** (`maf/react_agent_websearch.py`)
Uses `TavilyClient` directly (not the LangChain wrapper). The `web_search` function is a plain Python function passed to the agent. Cleaner but no step-by-step visibility.

> **Key difference:** LangGraph version logs every router decision and LLM output. MAF version is ~30 lines with no debug output — you only see the final answer.

---

### `rag_react_agent.py`

Static RAG agent — loads a **pre-built FAISS index** from `utility/faiss_index/` and always retrieves before answering.

**LangGraph** (`langgraph/rag_react_agent.py`)
Uses a dedicated `retrieve` node in the graph that always runs before the `agent` node. The flow is fixed: `START → retrieve → agent → END`. No conditional routing — retrieval is mandatory.

**MAF** (`maf/rag_react_agent.py`)
Exposes retrieval as a `rag_search` tool. The agent is instructed to always call it. The model decides when to call the tool rather than the framework enforcing it.

> **Key difference:** In LangGraph, retrieval is structurally guaranteed by the graph. In MAF, it depends on the model following instructions.

---

### `in-memory-rag.py`

Dynamic RAG — accepts file uploads at **runtime** and builds a FAISS index in memory on the fly.

**LangGraph** (`langgraph/in-memory-rag.py`)
Uses a `/load <path>` CLI command to call `load_document()` which chunks, embeds, and stores in an in-memory FAISS. The `search_documents` tool is a `@tool`-decorated function bound to the agent.

**MAF** (`maf/in-memory-rag.py`)
Identical document loading logic but `search_documents` is a plain function. Same `/load` CLI command. The agent is configured via instructions to always call the tool before answering.

> **Key difference:** Logic is almost identical. LangGraph uses `@tool` + `bind_tools()`; MAF uses a plain function in `tools=[]`. The in-memory FAISS setup is shared.

---

### `code-interpreter.py`

The agent writes Python code at runtime and executes it in a **sandboxed subprocess** with a 15-second timeout.

**LangGraph** (`langgraph/code-interpreter.py`)
`execute_python` is a `@tool` decorated function that writes code to a temp file and runs it via `subprocess.run()`. Wired into a standard agent-tools graph with `MemorySaver`.

**MAF** (`maf/code-interpreter.py`)
`execute_python` is a plain function with the exact same subprocess sandboxing logic. Passed directly to the MAF agent. No memory between sessions.

> **Key difference:** The execution tool is functionally identical. LangGraph retains conversation history via `MemorySaver`; MAF starts fresh each session.

---

### `CSV-File-Analyzer.py`

A data analyst agent with 4 tools: `load_csv`, `get_csv_info`, `query_csv` (pandas eval), and `visualize_csv` (matplotlib).

**LangGraph** (`langgraph/CSV-File-Analyzer.py`)
Tools are `@tool`-decorated with full docstrings (LangChain uses these as tool descriptions for the LLM). Uses `ToolNode` and a `should_continue` conditional edge.

**MAF** (`maf/CSV-File-Analyzer.py`)
Same 4 tools as plain Python functions. The MAF agent is instructed to call `get_csv_info()` at the start of every message to check if a CSV is loaded.

> **Key difference:** LangGraph's `@tool` docstrings are parsed as tool descriptions automatically. In MAF, tool descriptions come from the function's `__doc__` string — same approach, different decorator.

---

### `mcp_skills_langgraph.py` / `mcp_skills_maf.py`

Connects the agent to a **local MCP server** (`utility/mcp-server.py`) running at `http://127.0.0.1:8787/sse`, loading tools dynamically at startup.

**LangGraph** (`langgraph/mcp_skills_langgraph.py`)
Uses `load_mcp_tools(session)` from `langchain-mcp-adapters` — automatically converts MCP tools to LangChain tools. The graph is built dynamically after tools are loaded. Fully async.

**MAF** (`maf/mcp_skills_maf.py`)
Manually converts each MCP tool into a Python function using `make_mcp_tool_func()`, which reconstructs the function signature using `inspect.Parameter` so MAF can build the correct tool schema. More boilerplate but no LangChain dependency needed.

> **Key difference:** LangGraph benefits from `langchain-mcp-adapters` which does the conversion automatically. MAF requires manual signature reconstruction — more code, but you see exactly how tool schemas are built.

---

### `Reflection-Agent.py`

Generates an essay, critiques it, then refines it — **3 loops** before returning the final version.

**LangGraph** (`langgraph/Reflection-Agent.py`)
Two nodes: `generate` and `reflect`. The key trick: message roles are flipped before reflection — AI messages become Human messages so the reflector sees the essay as a "student submission". A `should_continue` function counts messages and stops after 6 (= 3 full loops).

**MAF** (`maf/Reflection-Agent.py`)
Two separate named agents: `generator_agent` and `reflector_agent`. The orchestration loop is explicit Python: generate → reflect → generate, repeated `max_loops` times. No role-flipping needed because each agent has its own persistent instructions and context.

> **Key difference:** In LangGraph, one LLM plays both roles via role-flipping in a single graph. In MAF, two separate agents each have their own identity — simpler to reason about, but uses more API calls per turn.

---

### `Reflexion_Agent.py`

An advanced self-improving agent that stores **past mistakes in FAISS** and retrieves them before each new answer to avoid repeating errors.

**LangGraph** (`langgraph/Reflexion_Agent.py`)
Uses `StructuredTool`, `PydanticToolsParser`, and a `ResponderWithRetries` class. The graph has 3 nodes: `draft → tools → revise` with a conditional edge that loops up to `MAX_ITERATIONS = 5`. Reflections are stored to FAISS after each valid response.

**MAF** (`maf/Reflexion_Agent.py`)
No graph — the retry logic is a plain `run_reflexion_turn()` async function with a `for attempt in range(max_attempts)` loop. The agent is instructed to return structured JSON. Parsing is done manually with Pydantic. If the reflection indicates missing info, a new prompt is built and the agent is called again.

> **Key difference:** LangGraph models the iteration as a typed graph with explicit state transitions. MAF models it as a Python loop — less framework overhead, same behaviour. Both use the same FAISS memory store for cross-session persistence.

