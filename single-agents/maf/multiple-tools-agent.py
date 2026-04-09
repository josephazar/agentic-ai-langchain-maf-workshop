import asyncio
import os
import sys
import subprocess
import tempfile
import functools
import time
from typing import Optional

from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader
)
from langchain_community.tools.tavily_search import TavilySearchResults  # ← added

load_dotenv()


# ─────────────────────────────────────────────
# RETRY DECORATOR
# ─────────────────────────────────────────────

def with_retries(max_attempts=3, delay=1.0):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    result = fn(*args, **kwargs)
                    if isinstance(result, str) and result.startswith("[TOOL ERROR]"):
                        raise RuntimeError(result)
                    return result
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts:
                        time.sleep(delay)
            return (
                f"[TOOL ERROR] '{fn.__name__}' failed after {max_attempts} attempts. "
                f"Reason: {repr(last_error)}. "
                f"Please use a fallback tool."
            )
        return wrapper
    return decorator


# ─────────────────────────────────────────────
# VECTOR STORE (RAG)
# ─────────────────────────────────────────────

embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
    model="text-embedding-3-large",
)

vector_store: Optional[FAISS] = None


def load_document(file_path: str):
    global vector_store
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    else:
        return f"Unsupported file type: {ext}"

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        vector_store.add_documents(chunks)

    return f"Loaded {len(chunks)} chunks from {os.path.basename(file_path)}"


# ─────────────────────────────────────────────
# TAVILY (same as LangGraph version)
# ─────────────────────────────────────────────

_tavily = TavilySearchResults(max_results=5)


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@with_retries(max_attempts=3, delay=1.0)
def web_search(query: str) -> str:
    """Search the internet for current or factual information."""
    try:
        results = _tavily.invoke(query)
        if not results:
            return "[TOOL ERROR] web_search: no results found."

        # same handling as LangGraph — handles both dict and plain string results
        parts = []
        for r in results:
            if isinstance(r, dict):
                parts.append(f"[{r.get('url', 'unknown')}]\n{r.get('content', '')}")
            else:
                parts.append(str(r))
        return "\n\n---\n\n".join(parts)

    except Exception as e:
        # hard fallback to documents before retrying — same as LangGraph
        fallback = search_documents(query)
        if "[TOOL ERROR]" in fallback:
            raise  # re-raise so with_retries handles it
        return (
            f"Web search unavailable ({e}). "
            f"Falling back to uploaded documents:\n\n{fallback}"
        )


def simple_search(query: str) -> str:
    """Fallback general knowledge search when web_search fails."""
    query = query.lower()
    if "ai" in query:
        return "AI stands for Artificial Intelligence."
    elif "python" in query:
        return "Python is used for AI, web, and automation."
    elif "langgraph" in query:
        return "LangGraph is a framework for building AI agents."
    return "[TOOL ERROR] simple_search: no matching knowledge found."


@with_retries(max_attempts=2, delay=0.5)
def search_documents(query: str) -> str:
    """Search uploaded documents (PDF, TXT, CSV)."""
    if vector_store is None:
        return "[TOOL ERROR] search_documents: no documents loaded. Ask the user to upload a file."
    try:
        results = vector_store.similarity_search(query, k=3)
        if not results:
            return "[TOOL ERROR] search_documents: no relevant content found in documents."
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        raise RuntimeError(f"Vector store search failed: {repr(e)}")


@with_retries(max_attempts=2, delay=0.5)
def execute_python(code: str) -> str:
    """Execute Python code for math, calculations, and data tasks."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Python error: {result.stderr.strip()}")
        return result.stdout.strip() or "Executed successfully with no output."
    except subprocess.TimeoutExpired:
        raise TimeoutError("Code execution timed out after 10 seconds.")
    finally:
        os.remove(path)


def get_weather(location: str = "Beirut") -> str:
    """Get current weather for a location."""
    try:
        mock_data = {
            "beirut":    "Beirut: 24C, partly cloudy, humidity 65%.",
            "london":    "London: 12C, rainy, humidity 85%.",
            "new york":  "New York: 18C, overcast, humidity 70%.",
            "dubai":     "Dubai: 38C, sunny, humidity 40%.",
        }
        return mock_data.get(location.lower(), f"{location}: 22C, clear skies.")
    except Exception as e:
        return f"[TOOL ERROR] get_weather failed: {repr(e)}. Try web_search for weather instead."


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    agent = OpenAIChatClient().as_agent(
        name="MAF Multi Tool Agent",
        description="Agent with multiple tools (RAG, Python, Weather, Search)",
        instructions="""
You are a powerful AI agent with access to multiple tools.

TOOLS:
- web_search       → internet queries and current information
- simple_search    → general knowledge (AI, Python, LangGraph)
- search_documents → questions about uploaded documents
- execute_python   → calculations, code, math
- get_weather      → weather queries

RULES:
- ALWAYS use a tool if one is relevant.
- NEVER answer from your own knowledge.
- Do NOT respond to the user until ALL required tools have returned valid results.

FALLBACK CHAIN (follow in order if a tool returns [TOOL ERROR]):
- web_search fails     → try simple_search
- simple_search fails  → tell user no information was found
- search_documents fails with "no documents loaded"
                       → tell user to upload a file first using /load
- search_documents fails with "no relevant content"
                       → try web_search on the same query
- execute_python fails → rewrite the code and retry once,
                         if it still fails tell user what went wrong
- get_weather fails    → try web_search for weather instead

ERROR HANDLING:
- A tool has failed if its result starts with [TOOL ERROR].
- When a tool fails, do NOT return the error to the user directly.
- Follow the fallback chain above silently, then answer with the final result.
- Only tell the user about a failure if ALL fallbacks are also exhausted.
""",
        tools=[
            web_search,
            simple_search,
            search_documents,
            execute_python,
            get_weather,
        ],
    )

    print("MAF Multi-Tool Agent running...\n")
    print("Commands:")
    print("  /load <file_path>  — load a PDF, TXT, or CSV")
    print("  exit\n")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if user_input.startswith("/load "):
            path = user_input.replace("/load ", "").strip()
            print(load_document(path))
            continue

        response = await agent.run(user_input)
        print("Assistant:", response.text)


if __name__ == "__main__":
    asyncio.run(main())