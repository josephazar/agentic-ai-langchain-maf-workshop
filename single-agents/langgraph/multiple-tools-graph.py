"""
LangGraph Multi-Tool Agent
Combines: Web Search + RAG + Code Interpreter + Custom Tools
- Agent autonomously decides which tool to use for each step
- Handles tool errors gracefully with real retry logic (max 3 attempts)
- Uses LangGraph's built-in handle_tool_errors + a custom retry node

Fixes applied:
  1. handle_tool_error now has signature (e: Exception) -> str as required
     by newer LangGraph versions — state-based handlers are no longer accepted.
     Retry counting is moved into a separate 'error_tracker' graph node that
     runs after ToolNode, keeping all state logic inside the graph.
  2. web_search now handles both dict and plain-string results from Tavily,
     fixing the AttributeError: 'str' object has no attribute 'get' crash.
"""

import os
import sys
import time
import subprocess
import tempfile
from typing import Annotated, Union
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# RAG dependencies
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_RETRIES = 3      # maximum tool retry attempts per tool_call_id
RETRY_DELAY = 1.0    # seconds to wait between retries
MAX_ERROR_LOOPS = 6  # safety cap: max total error recovery cycles


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_retry_counts: dict  # { tool_call_id: int }
    error_loop_count: int    # global safety counter


# ── In-memory vector store ────────────────────────────────────────────────────

embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
    model="text-embedding-3-large",
)
vector_store: FAISS | None = None


def load_document(file_path: str) -> str:
    """Load a PDF, TXT, or CSV file into the in-memory vector store."""
    global vector_store
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    else:
        return f"Unsupported file type: {ext}. Please upload PDF, TXT, or CSV."

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        vector_store.add_documents(chunks)

    return f"Loaded '{os.path.basename(file_path)}' — {len(chunks)} chunks indexed."


# ── TOOL 1: Web Search ────────────────────────────────────────────────────────

_tavily = TavilySearchResults(max_results=5)


@tool
def web_search(query: str) -> str:
    """
    Search the internet for current or factual information.
    Use for recent news, facts, or anything not in uploaded documents.
    Falls back to search_documents if the web search API is unavailable.
    """
    try:
        results = _tavily.invoke(query)
        if not results:
            return "No results found on the web."

    
        parts = []
        for r in results:
            if isinstance(r, dict):
                parts.append(f"[{r.get('url', 'unknown')}]\n{r.get('content', '')}")
            else:
                parts.append(str(r))
        return "\n\n---\n\n".join(parts)

    except Exception as e:
        # Hard fallback: try local documents before propagating the error
        fallback = search_documents.invoke(query)  # type: ignore[attr-defined]
        if "No documents loaded" in fallback or "No relevant content" in fallback:
            raise  # re-raise so the retry node handles it
        return (
            f"Web search unavailable ({e}). "
            f"Falling back to uploaded documents:\n\n{fallback}"
        )


# ── TOOL 2: RAG Document Search ───────────────────────────────────────────────

@tool
def search_documents(query: str) -> str:
    """
    Search the uploaded documents for relevant information.
    Use this when the user asks questions about files they have uploaded
    (PDFs, TXTs, CSVs). Always prefer this over web search for document questions.
    """
    if vector_store is None:
        return "No documents loaded. Ask the user to upload a file first using /load <path>."

    try:
        results = vector_store.similarity_search(query, k=4)
        if not results:
            return "No relevant content found in the uploaded documents."
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in results
        )
    except Exception as e:
        return f"Document search failed: {str(e)}"


# ── TOOL 3: Code Interpreter ──────────────────────────────────────────────────

@tool
def execute_python(code: str) -> str:
    """
    Write and execute Python code to solve problems.
    Use this for: math calculations, data analysis, data transformations,
    generating/saving plots, sorting, or any logic requiring computation.
    Always use this instead of computing answers manually.
    For plots, use plt.savefig('output.png') instead of plt.show().
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        preamble = "import matplotlib\nmatplotlib.use('Agg')\n"
        f.write(preamble + code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            raise RuntimeError(f"Python error:\n{stderr}")

        output = stdout if stdout else "Code executed successfully (no output)."
        if "savefig" in code:
            output += "\nPlot saved. Check the working directory for the image file."
        return output

    except subprocess.TimeoutExpired:
        raise TimeoutError("Execution timed out (>15 seconds). Simplify the code.")
    finally:
        os.unlink(tmp_path)


# ── TOOL 4: Weather ───────────────────────────────────────────────────────────

@tool
def get_weather(location: str = "Beirut") -> str:
    """
    Get the current weather for a location.
    Use this when the user asks about weather conditions in any city.
    Defaults to Beirut if no location is given.
    """
    mock_data = {
        "beirut":   "Beirut: 24C, partly cloudy, humidity 65%.",
        "london":   "London: 12C, rainy, humidity 85%.",
        "new york": "New York: 18C, overcast, humidity 70%.",
        "dubai":    "Dubai: 38C, sunny, humidity 40%.",
    }
    return mock_data.get(location.lower(), f"{location}: 22C, clear skies.")


# ── LLM + Tools ───────────────────────────────────────────────────────────────

tools = [web_search, search_documents, execute_python, get_weather]

llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
)
llm_with_tools = llm.bind_tools(tools)


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a powerful multi-tool agent. You have access to four tools:\n\n"
    "1. web_search       — search the internet for current or factual information\n"
    "2. search_documents — search uploaded files (PDF, TXT, CSV)\n"
    "3. execute_python   — write and run Python code for math, data, or plots\n"
    "4. get_weather      — get current weather for a city\n\n"
    "RULES:\n"
    "- Always pick the most appropriate tool for the query.\n"
    "- For math or computation: use execute_python — never calculate manually.\n"
    "- For questions about uploaded files: use search_documents.\n"
    "- For recent news or internet facts: use web_search.\n"
    "- For weather: use get_weather.\n"
    "- If a tool returns an error message, the system will automatically retry "
    "  up to 3 times. If it still fails, explain what went wrong to the user.\n"
    "- After using a tool successfully, explain the result clearly to the user.\n"
))


# ── Agent Node ────────────────────────────────────────────────────────────────

def agent(state: State):
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "tool_retry_counts": state.get("tool_retry_counts", {}),
        "error_loop_count": state.get("error_loop_count", 0),
    }



def handle_tool_error(e: Union[Exception, KeyboardInterrupt]) -> str:
    """
    Called by ToolNode on any unhandled tool exception.
    Returns a plain string — LangGraph wraps it in a ToolMessage automatically.
    """
    return (
        f"Tool failed with error: {repr(e)}\n"
        "The system will retry automatically if attempts remain, "
        "or inform you if the maximum retries have been reached."
    )


tool_node = ToolNode(tools, handle_tool_errors=handle_tool_error)


# ── Error Tracker Node ────────────────────────────────────────────────────────
# Runs after every ToolNode execution. Scans ToolMessages for errors,
# increments per-call retry counters in state, and appends a status message
# so the agent knows whether to retry or give up.

def error_tracker(state: State) -> dict:
    retry_counts: dict = dict(state.get("tool_retry_counts") or {})
    error_loop_count: int = state.get("error_loop_count", 0)

    # Find the last AIMessage to get the original tool calls
    last_ai: AIMessage | None = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai = msg
            break

    if last_ai is None or not getattr(last_ai, "tool_calls", None):
        return {
            "tool_retry_counts": retry_counts,
            "error_loop_count": error_loop_count,
        }

    # Collect ToolMessages from this round that contain error text
    latest_tool_msgs = [
        m for m in state["messages"]
        if isinstance(m, ToolMessage)
        and any(tc["id"] == m.tool_call_id for tc in last_ai.tool_calls)
        and "Tool failed with error" in (m.content or "")
    ]

    if not latest_tool_msgs:
        # No errors this round — reset the error loop counter
        return {
            "tool_retry_counts": retry_counts,
            "error_loop_count": 0,
        }

    # Increment counters and produce status ToolMessages
    status_messages = []
    for tm in latest_tool_msgs:
        call_id = tm.tool_call_id
        tool_name = next(
            (tc["name"] for tc in last_ai.tool_calls if tc["id"] == call_id),
            "unknown",
        )
        current = retry_counts.get(call_id, 0)
        retry_counts[call_id] = current + 1

        time.sleep(RETRY_DELAY)

        if current + 1 < MAX_RETRIES:
            status = (
                f"[Retry {current + 1}/{MAX_RETRIES}] Tool '{tool_name}' failed. "
                f"Retrying automatically..."
            )
        else:
            status = (
                f"[FINAL] Tool '{tool_name}' failed after {MAX_RETRIES} attempts. "
                f"Please try a different approach or inform the user."
            )

        status_messages.append(ToolMessage(content=status, tool_call_id=call_id))

    return {
        "messages": status_messages,
        "tool_retry_counts": retry_counts,
        "error_loop_count": error_loop_count + 1,
    }


# ── Routers ───────────────────────────────────────────────────────────────────

def tool_router(state: State) -> str:
    """Route from agent: invoke tools if requested, else finish."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def retry_router(state: State) -> str:
    """
    After error_tracker runs, decide whether to loop back to the agent
    (retry) or terminate.
    """
    retry_counts: dict = state.get("tool_retry_counts") or {}
    error_loop_count: int = state.get("error_loop_count", 0)

    # Hard safety cap
    if error_loop_count >= MAX_ERROR_LOOPS:
        return END

    last_ai: AIMessage | None = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai = msg
            break

    if last_ai is None or not getattr(last_ai, "tool_calls", None):
        return END

    all_exhausted = all(
        retry_counts.get(tc["id"], 0) >= MAX_RETRIES
        for tc in last_ai.tool_calls
    )
    return END if all_exhausted else "agent"


# ── Graph ─────────────────────────────────────────────────────────────────────

memory = MemorySaver()

builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)
builder.add_node("error_tracker", error_tracker)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tool_router, ["tools", END])

# tools -> error_tracker always (success or failure)
builder.add_edge("tools", "error_tracker")

# error_tracker -> agent (retry) or END (exhausted / success)
builder.add_conditional_edges("error_tracker", retry_router, ["agent", END])

graph = builder.compile(checkpointer=memory)


# ── CLI ───────────────────────────────────────────────────────────────────────

def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {
            "messages": [("user", user_input)],
            "tool_retry_counts": {},
            "error_loop_count": 0,
        },
        {"configurable": {"thread_id": "1"}},
    ):
        for value in event.values():
            if "messages" not in value or not value["messages"]:
                continue

            last_msg = value["messages"][-1]

            if isinstance(last_msg, AIMessage) and last_msg.content:
                print("Assistant:", last_msg.content)

if __name__ == "__main__":
    print("LangGraph Multi-Tool Agent  (retries: up to 3 per tool call)")
    print("Tools: Web Search | RAG | Code Interpreter | Weather")
    print("Commands:")
    print("  /load <file_path>  — load a PDF, TXT, or CSV into RAG")
    print("  quit / exit        — exit\n")
    print("Example prompts:")
    print("  - What is the latest news about AI?")
    print("  - What is sqrt(256) * 3.14?")
    print("  - What's the weather in Dubai?")
    print("  - /load report.pdf  then  summarize the document\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if user_input.startswith("/load "):
                file_path = user_input[6:].strip().strip('"').strip("'")
                print(f"System: {load_document(file_path)}\n")
                continue
            stream_graph_updates(user_input)
        except Exception:
            import traceback
            traceback.print_exc()
            break