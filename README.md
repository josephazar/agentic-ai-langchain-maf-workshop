# Agentic AI — LangChain & MAF Workshop

A hands-on workshop that teaches agentic AI patterns using two frameworks side-by-side: **LangGraph** (LangChain's graph-based agent framework) and **MAF** (Microsoft's Agent Framework, built on Azure OpenAI). All examples are backed by Azure OpenAI models.

---

## What's inside

```
agentic-ai-langchain-maf-workshop-main/
├── 01-prompting-techniques/   # Foundational prompting demos
├── single-agents/
│   ├── langgraph/             # Single-agent examples with LangGraph
│   └── maf/                   # Same examples rewritten in MAF
└── agent-patterns/
    ├── Langgraph/             # Multi-agent router system (LangGraph)
    └── MAF/                   # Multi-agent router system (MAF)
```

### 01 — Prompting Techniques

Self-contained scripts that demonstrate different prompting strategies, each paired with a JSON prompt file in `prompts/`. Good starting point if you're new to the workshop.

| Script | What it shows |
|---|---|
| `zero_shot.py` | Plain instructions with no examples |
| `few_shot.py` | In-context examples to lock in format/behavior |
| `role_prompting.py` | System-message personas (Developer, Analyst, Teacher) |
| `chain_of_thought.py` | Step-by-step reasoning |
| `chain_prompting.py` | Chaining multiple prompts together |
| `react_prompting.py` | ReAct pattern (Reason + Act loops) |
| `self_consistency.py` | Running the same prompt multiple times and voting |
| `tree_of_thought.py` | Branching reasoning paths |
| `structured_output.py` | Forcing JSON / structured responses |

### Single Agents

The same set of agents implemented in both frameworks so you can compare them directly:

- **basic_react_agent** — Minimal ReAct agent with two dummy tools (weather + search)
- **Agentic_Rag** — Retrieval-augmented generation agent
- **CSV-File-Analyzer** — Agent that reads and reasons over CSV files
- **Reflection-Agent** — Agent that critiques and improves its own output
- **Reflexion_Agent** — Extended reflection with memory (uses FAISS for storage)
- **code-interpreter** — Agent that writes and executes code
- **in-memory-rag** — RAG with an in-memory vector store
- **multiple-tools-agent** — Agent juggling several tools at once
- **react_agent_websearch** — ReAct agent with live web search (Tavily)
- **sql_react_agent** — Agent that queries a SQL database
- **mcp_skills_langgraph** — LangGraph agent using MCP (Model Context Protocol) tools

### Agent Patterns — Multi-Agent Router

The capstone of the workshop. A supervisor/router architecture where a top-level router classifies incoming messages and delegates to specialized sub-agents:

- **Router** — Classifies user intent and dispatches to the right agent
- **chat_agent** — General conversation
- **knowledge_agent** — RAG + web search (Tavily) for factual questions
- **code_agent** — Code generation and execution with human-in-the-loop approval
- **engineering_agent** — Technical/engineering queries
- **memory_store** — Shared session memory across agents

The `Langgraph/` version uses LangGraph's `StateGraph` with SQLite checkpointing. The `MAF/` version uses `agent_framework.azure` with JSON-file-based memory.

---

## Prerequisites

- Python 3.10+
- An Azure OpenAI resource with a deployed model (the code references `gpt-5-mini` and `gpt-4.1`)
- A Tavily API key (for web-search agents)

---

## Setup

**1. Clone / unzip the repo**

```bash
cd agentic-ai-langchain-maf-workshop-main
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Create a `.env` file** in the project root with your credentials:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
TAVILY_API_KEY=<your-tavily-key>   # only needed for web-search agents
```

---

## Running things

### Prompting techniques

Each script is self-contained — just run it directly:

```bash
cd 01-prompting-techniques
python zero_shot.py
python few_shot.py
# etc.
```

### Single agents

```bash
# LangGraph version
cd single-agents/langgraph
python basic_react_agent.py

# MAF version
cd single-agents/maf
python basic_react_agent.py
```

### Multi-agent router (LangGraph)

```bash
cd agent-patterns/Langgraph
python Router.py
```

### Multi-agent router (MAF)

```bash
cd agent-patterns/MAF
python Router.py
```

Both routers start an interactive chat loop in the terminal. Type your message and the router will pick the right sub-agent automatically. The LangGraph version saves a debug log to `router_log.txt`.

---

## Framework comparison at a glance

| | LangGraph | MAF |
|---|---|---|
| Core abstraction | `StateGraph` with typed state | `AzureOpenAIResponsesClient` + agents |
| Memory | SQLite checkpointer | JSON files |
| Async | Optional | Native (`asyncio`) |
| Debugging | Built-in LangSmith tracing | Log files |
| Best for | Complex graph flows, human-in-the-loop | Simpler agent composition on Azure |

---

## Reference materials

- `Langgraph_MAF.pdf` — Slide deck for the workshop
- `Untitled Diagram.drawio.png` — Architecture diagram of the multi-agent router
- `agent-patterns/Langgraph/router_graph.png` — LangGraph state graph visualization
