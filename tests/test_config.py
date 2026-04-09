"""
Test registry: defines every testable file, its category, test input, and task description.
"""

import os
import socket

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Pre-check helpers
# ---------------------------------------------------------------------------

def check_faiss_index():
    path = os.path.join(REPO_ROOT, "utility", "faiss_index", "index.faiss")
    if not os.path.exists(path):
        return False, "FAISS index not found at utility/faiss_index/"
    return True, ""


def check_chinook_db():
    path = os.path.join(REPO_ROOT, "utility", "chinook.db")
    if not os.path.exists(path):
        return False, "chinook.db not found at utility/"
    return True, ""


def check_mcp_server():
    try:
        s = socket.create_connection(("127.0.0.1", 8787), timeout=2)
        s.close()
        return True, ""
    except Exception:
        return False, "MCP server not reachable on localhost:8787"


PRE_CHECKS = {
    "check_faiss_index": check_faiss_index,
    "check_chinook_db": check_chinook_db,
    "check_mcp_server": check_mcp_server,
}


# ---------------------------------------------------------------------------
# Test entries
# ---------------------------------------------------------------------------

def get_test_entries():
    """Return the full list of testable files with metadata."""

    pt = os.path.join(REPO_ROOT, "01-prompting-techniques")
    lg = os.path.join(REPO_ROOT, "single-agents", "langgraph")
    maf = os.path.join(REPO_ROOT, "single-agents", "maf")
    nb = os.path.join(REPO_ROOT, "langchain-patterns")

    entries = [
        # ── Category A: non-interactive prompting scripts ─────────────────
        {
            "file_path": os.path.join(pt, "zero_shot.py"),
            "category": "A",
            "test_input": None,
            "task_description": "Demonstrates zero-shot prompting for classification, summarization, and extraction. Should show LLM responses for each task.",
            "timeout": 120,
            "cwd": pt,
        },
        {
            "file_path": os.path.join(pt, "chain_of_thought.py"),
            "category": "A",
            "test_input": None,
            "task_description": "Demonstrates chain-of-thought prompting: compares direct prompting vs 'let's think step by step' on a logic puzzle and billing complaint.",
            "timeout": 120,
            "cwd": pt,
        },
        {
            "file_path": os.path.join(pt, "role_prompting.py"),
            "category": "A",
            "test_input": None,
            "task_description": "Demonstrates role prompting with three personas (Developer, Data Analyst, Teacher) answering the same question differently.",
            "timeout": 120,
            "cwd": pt,
        },
        {
            "file_path": os.path.join(pt, "self_consistency.py"),
            "category": "A",
            "test_input": None,
            "task_description": "Demonstrates self-consistency: runs the same prompt 5 times and uses majority voting. Should show individual runs plus vote counts.",
            "timeout": 180,
            "cwd": pt,
        },
        {
            "file_path": os.path.join(pt, "tree_of_thought.py"),
            "category": "A",
            "test_input": None,
            "task_description": "Demonstrates tree-of-thought: generates branches, evaluates them, then solves the best branch for a water jug puzzle.",
            "timeout": 120,
            "cwd": pt,
        },
        {
            "file_path": os.path.join(pt, "react_prompting.py"),
            "category": "A",
            "test_input": None,
            "task_description": "Demonstrates a manual ReAct loop with calculate and convert_currency tools. Should show Thought/Action/Observation steps.",
            "timeout": 120,
            "cwd": pt,
        },
        {
            "file_path": os.path.join(pt, "structured_output.py"),
            "category": "A",
            "test_input": None,
            "task_description": "Demonstrates structured JSON output extraction with retry logic. Should show parsed JSON fields (name, age, sentiment, etc.).",
            "timeout": 120,
            "cwd": pt,
        },
        {
            "file_path": os.path.join(pt, "prompt_templates.py"),
            "category": "A",
            "test_input": None,
            "task_description": "Compares raw API call vs LangChain prompt template pipeline for review sentiment analysis. Should show parsed sentiment/score output.",
            "timeout": 120,
            "cwd": pt,
        },

        # ── Category B: interactive menu-based scripts ────────────────────
        {
            "file_path": os.path.join(pt, "few_shot.py"),
            "category": "B",
            "test_input": "1\n0\n",
            "task_description": "Demonstrates few-shot prompting for fallacy detection. Menu choice 1 runs the zero-shot fallacy test. Should show LLM detecting logical fallacies.",
            "timeout": 120,
            "cwd": pt,
        },
        {
            "file_path": os.path.join(pt, "chain_prompting.py"),
            "category": "B",
            "test_input": "1\n\n0\n",
            "task_description": "Demonstrates chain prompting: article -> summary -> key points -> email. Menu choice 1 runs the manual chain. Should show step-by-step outputs.",
            "timeout": 120,
            "cwd": pt,
        },

        # ── Category C: LangGraph agents (REPL) ──────────────────────────
        {
            "file_path": os.path.join(lg, "basic_react_agent.py"),
            "category": "C",
            "test_input": "What is the weather in San Francisco?\nexit\n",
            "task_description": "LangGraph ReAct agent with weather and search tools. Should return weather info for San Francisco.",
            "timeout": 120,
            "cwd": lg,
        },
        {
            "file_path": os.path.join(lg, "react_agent_websearch.py"),
            "category": "C",
            "test_input": "What is the latest news about AI?\nexit\n",
            "task_description": "LangGraph ReAct agent with Tavily web search. Should return recent AI news from the web.",
            "timeout": 120,
            "cwd": lg,
        },
        {
            "file_path": os.path.join(lg, "code-interpreter.py"),
            "category": "C",
            "test_input": "What is 123 * 456?\nexit\n",
            "task_description": "LangGraph code interpreter agent. Should execute Python to compute 123*456=56088.",
            "timeout": 120,
            "cwd": lg,
        },
        {
            "file_path": os.path.join(lg, "Reflection-Agent.py"),
            "category": "C",
            "test_input": "Write a short paragraph about climate change\nexit\n",
            "task_description": "LangGraph reflection agent: generates an essay, critiques it, and refines it. Should show generation and critique steps.",
            "timeout": 300,
            "cwd": lg,
        },
        {
            "file_path": os.path.join(lg, "Reflexion_Agent.py"),
            "category": "C",
            "test_input": "What are the latest developments in quantum computing?\nexit\n",
            "task_description": "LangGraph reflexion agent with vector memory and Tavily search. Should search and reflect on quantum computing developments.",
            "timeout": 300,
            "cwd": lg,
        },
        {
            "file_path": os.path.join(lg, "rag_react_agent.py"),
            "category": "C",
            "test_input": "What is this document about?\nexit\n",
            "task_description": "LangGraph RAG agent using FAISS vector index. Should retrieve and answer from indexed documents.",
            "timeout": 120,
            "cwd": lg,
            "pre_check": "check_faiss_index",
        },
        {
            "file_path": os.path.join(lg, "Agentic_Rag.py"),
            "category": "C",
            "test_input": "What is this document about?\nexit\n",
            "task_description": "LangGraph agentic RAG with router, grader, and rewrite loop. Should retrieve, grade relevance, and answer.",
            "timeout": 180,
            "cwd": lg,
            "pre_check": "check_faiss_index",
        },
        {
            "file_path": os.path.join(lg, "sql_react_agent.py"),
            "category": "C",
            "test_input": "List all tables in the database\nexit\n",
            "task_description": "LangGraph SQL ReAct agent querying chinook SQLite database. Should list database tables.",
            "timeout": 120,
            "cwd": lg,
            "pre_check": "check_chinook_db",
        },
        {
            "file_path": os.path.join(lg, "CSV-File-Analyzer.py"),
            "category": "C",
            "test_input": "exit\n",
            "task_description": "LangGraph CSV analyzer agent. Smoke test: should start up and exit cleanly.",
            "timeout": 60,
            "cwd": lg,
        },
        {
            "file_path": os.path.join(lg, "in-memory-rag.py"),
            "category": "C",
            "test_input": "exit\n",
            "task_description": "LangGraph in-memory RAG agent. Smoke test: should start up and exit cleanly.",
            "timeout": 60,
            "cwd": lg,
        },
        {
            "file_path": os.path.join(lg, "multiple-tools-agent-langgraph.py"),
            "category": "C",
            "test_input": "What is the weather in Dubai?\nexit\n",
            "task_description": "LangGraph prebuilt multi-tool agent. Should respond with weather information for Dubai.",
            "timeout": 120,
            "cwd": lg,
        },
        {
            "file_path": os.path.join(lg, "multiple-tools-graph.py"),
            "category": "C",
            "test_input": "What is the weather in Dubai?\nexit\n",
            "task_description": "LangGraph multi-tool agent with error tracking and retry logic. Should respond with weather information.",
            "timeout": 120,
            "cwd": lg,
        },
        {
            "file_path": os.path.join(lg, "mcp_skills_langgraph.py"),
            "category": "C",
            "test_input": "What is the weather in New York?\nexit\n",
            "task_description": "LangGraph MCP skills agent connecting to a local MCP server. Should use the weather tool and return weather for New York.",
            "timeout": 120,
            "cwd": lg,
            "needs_mcp": True,
        },

        # ── Category D: MAF agents (async REPL) ──────────────────────────
        {
            "file_path": os.path.join(maf, "basic_react_agent.py"),
            "category": "D",
            "test_input": "What is the weather in San Francisco?\nexit\n",
            "task_description": "MAF basic ReAct agent with weather and search tools. Should return weather info for San Francisco.",
            "timeout": 120,
            "cwd": maf,
        },
        {
            "file_path": os.path.join(maf, "react_agent_websearch.py"),
            "category": "D",
            "test_input": "What is the latest news about AI?\nexit\n",
            "task_description": "MAF agent with Tavily web search. Should return recent AI news.",
            "timeout": 120,
            "cwd": maf,
        },
        {
            "file_path": os.path.join(maf, "code-interpreter.py"),
            "category": "D",
            "test_input": "What is 123 * 456?\nexit\n",
            "task_description": "MAF code interpreter agent. Should execute Python to compute 123*456=56088.",
            "timeout": 120,
            "cwd": maf,
        },
        {
            "file_path": os.path.join(maf, "Reflection-Agent.py"),
            "category": "D",
            "test_input": "Write a short paragraph about climate change\nexit\n",
            "task_description": "MAF reflection agent: generates, critiques, and refines text. Should show generation and critique steps.",
            "timeout": 300,
            "cwd": maf,
        },
        {
            "file_path": os.path.join(maf, "Reflexion_Agent.py"),
            "category": "D",
            "test_input": "What are the latest developments in quantum computing?\nexit\n",
            "task_description": "MAF reflexion agent with memory and search. Should search and reflect on quantum computing.",
            "timeout": 300,
            "cwd": maf,
        },
        {
            "file_path": os.path.join(maf, "rag_react_agent.py"),
            "category": "D",
            "test_input": "What is this document about?\nexit\n",
            "task_description": "MAF RAG agent using FAISS vector index. Should retrieve and answer from indexed documents.",
            "timeout": 120,
            "cwd": maf,
            "pre_check": "check_faiss_index",
        },
        {
            "file_path": os.path.join(maf, "Agentic_Rag.py"),
            "category": "D",
            "test_input": "What is this document about?\nexit\n",
            "task_description": "MAF agentic RAG with router/grader/rewrite. Should retrieve, grade, and answer.",
            "timeout": 180,
            "cwd": maf,
            "pre_check": "check_faiss_index",
        },
        {
            "file_path": os.path.join(maf, "sql_react_agent.py"),
            "category": "D",
            "test_input": "List all tables in the database\nexit\n",
            "task_description": "MAF SQL agent querying chinook SQLite database. Should list database tables.",
            "timeout": 120,
            "cwd": maf,
            "pre_check": "check_chinook_db",
        },
        {
            "file_path": os.path.join(maf, "CSV-File-Analyzer.py"),
            "category": "D",
            "test_input": "exit\n",
            "task_description": "MAF CSV analyzer agent. Smoke test: should start up and exit cleanly.",
            "timeout": 60,
            "cwd": maf,
        },
        {
            "file_path": os.path.join(maf, "in-memory-rag.py"),
            "category": "D",
            "test_input": "exit\n",
            "task_description": "MAF in-memory RAG agent. Smoke test: should start up and exit cleanly.",
            "timeout": 60,
            "cwd": maf,
        },
        {
            "file_path": os.path.join(maf, "multiple-tools-agent.py"),
            "category": "D",
            "test_input": "What is the weather in Dubai?\nexit\n",
            "task_description": "MAF multi-tool agent with retry/fallback logic. Should respond with weather information.",
            "timeout": 120,
            "cwd": maf,
        },
        {
            "file_path": os.path.join(maf, "mcp_skills_maf.py"),
            "category": "D",
            "test_input": "What is the weather in New York?\nexit\n",
            "task_description": "MAF MCP skills agent connecting to a local MCP server. Should use the weather tool and return weather for New York.",
            "timeout": 120,
            "cwd": maf,
            "needs_mcp": True,
        },

        # ── Category E: Jupyter notebooks ─────────────────────────────────
        {
            "file_path": os.path.join(nb, "01_ReAct_Agent.ipynb"),
            "category": "E",
            "test_input": None,
            "task_description": "ReAct Agent notebook: builds a financial analyst agent with LangGraph that searches Wikipedia.",
            "timeout": 300,
            "cwd": nb,
        },
        {
            "file_path": os.path.join(nb, "02_Persistent_Memory_Agent.ipynb"),
            "category": "E",
            "test_input": None,
            "task_description": "Persistent Memory Agent notebook: dual-track memory agent that retrieves and updates client profiles.",
            "timeout": 300,
            "cwd": nb,
        },
        {
            "file_path": os.path.join(nb, "03_Prompt_Chaining.ipynb"),
            "category": "E",
            "test_input": None,
            "task_description": "Prompt Chaining notebook: LCEL chain for earnings report analysis (sentiment, metrics, summary).",
            "timeout": 300,
            "cwd": nb,
        },
        {
            "file_path": os.path.join(nb, "04_Evaluator_Optimiser.ipynb"),
            "category": "E",
            "test_input": None,
            "task_description": "Evaluator-Optimiser notebook: generator + evaluator feedback loop for investment memo compliance.",
            "timeout": 300,
            "cwd": nb,
        },
        {
            "file_path": os.path.join(nb, "05_Parallelisation.ipynb"),
            "category": "E",
            "test_input": None,
            "task_description": "Parallelisation notebook: RunnableParallel for concurrent news sentiment, entity, and summary analysis.",
            "timeout": 300,
            "cwd": nb,
        },
        {
            "file_path": os.path.join(nb, "06_Orchestrator_Worker.ipynb"),
            "category": "E",
            "test_input": None,
            "task_description": "Orchestrator-Worker notebook: dynamic task planning with specialized financial workers.",
            "timeout": 300,
            "cwd": nb,
        },
        {
            "file_path": os.path.join(nb, "07_MS_Prompt_Engineering_Guide.ipynb"),
            "category": "E",
            "test_input": None,
            "task_description": "Prompt Engineering Guide notebook: comprehensive demo of 16 prompt engineering techniques.",
            "timeout": 600,
            "cwd": nb,
        },
        {
            "file_path": os.path.join(nb, "08_Advanced_RAG_MultiSource.ipynb"),
            "category": "E",
            "test_input": None,
            "task_description": "Advanced RAG notebook: multi-source RAG with Wikipedia, arXiv papers, citation-enforced synthesis, and LLM-as-Judge evaluation.",
            "timeout": 600,
            "cwd": nb,
        },
    ]

    return entries
