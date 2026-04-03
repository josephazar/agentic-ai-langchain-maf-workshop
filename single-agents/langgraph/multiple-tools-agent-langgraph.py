"""
LangGraph / LangChain prebuilt multi-tool agent
Replaces manual StateGraph + ToolNode wiring with create_agent.

Tools:
- Web search
- RAG over uploaded/local documents
- Python execution
- Weather
"""

import os
import sys
import subprocess
import tempfile
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ── In-memory vector store ─────────────────────────────────────────────────────

embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
    model="text-embedding-3-large",
)

vector_store: Optional[FAISS] = None


def load_document(file_path: str) -> str:
    """Load a PDF, TXT, or CSV file into the in-memory vector store."""
    global vector_store

    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
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

    return f" Loaded '{os.path.basename(file_path)}' — {len(chunks)} chunks indexed."


# ── Tools ──────────────────────────────────────────────────────────────────────

web_search = TavilySearchResults(max_results=5)


@tool
def search_documents(query: str) -> str:
    """
    Search the uploaded/local documents for relevant information.
    Use this when the user asks about files that were loaded with /load.
    """
    global vector_store

    if vector_store is None:
        return "No documents loaded. Ask the user to load a file first using /load <path>."

    try:
        results = vector_store.similarity_search(query, k=4)
        if not results:
            return "No relevant content found in the loaded documents."

        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in results
        )
    except Exception as e:
        return f" Document search failed: {str(e)}"


@tool
def execute_python(code: str) -> str:
    """
    Execute Python code for calculations, data analysis, transformations, and plots.
    For plots, use plt.savefig('output.png') instead of plt.show().
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
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
            return f" Error:\n{stderr}\nFix the code and try again."

        output = stdout if stdout else " Code executed successfully (no output)."

        if "savefig" in code:
            output += "\n Plot saved. Check the working directory for the image file."

        return output

    except subprocess.TimeoutExpired:
        return " Execution timed out (>15 seconds). Simplify the code."
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@tool
def get_weather(location: str = "Beirut") -> str:
    """
    Get the current weather for a location.
    Replace this mock with a real weather API later if needed.
    """
    mock_data = {
        "beirut": "🌤 Beirut: 24°C, partly cloudy, humidity 65%.",
        "london": "🌧 London: 12°C, rainy, humidity 85%.",
        "new york": "🌥 New York: 18°C, overcast, humidity 70%.",
        "dubai": "☀ Dubai: 38°C, sunny, humidity 40%.",
    }
    return mock_data.get(location.lower(), f"☀ {location}: 22°C, clear skies.")


tools = [
    web_search,
    search_documents,
    execute_python,
    get_weather,
]


# ── Model ──────────────────────────────────────────────────────────────────────

model = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"),
)


# ── Prebuilt agent ─────────────────────────────────────────────────────────────

memory = MemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=(
        "You are a powerful multi-tool agent. "
        "You have access to web search, document search, Python execution, and weather.\n\n"
        "Rules:\n"
        "- Use execute_python for math, calculations, transformations, and plotting.\n"
        "- Use search_documents for loaded PDF/TXT/CSV file questions.\n"
        "- Use web_search for internet/current facts.\n"
        "- Use get_weather for weather questions.\n"
        "- If a tool fails, retry once if appropriate.\n"
        "- After tool use, explain the result clearly."
    ),
    checkpointer=memory,
)


# ── CLI helpers ────────────────────────────────────────────────────────────────

def stream_agent(user_input: str, thread_id: str = "1"):
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    ):
        messages = chunk.get("messages", [])
        if not messages:
            continue

        last = messages[-1]

        # Final assistant text
        if getattr(last, "content", None):
            if isinstance(last.content, str) and last.content.strip():
                print("Assistant:", last.content)


if __name__ == "__main__":
    print("Prebuilt Multi-Tool Agent")
    print("Tools: Web Search | RAG | Code Interpreter | Weather")
    print("Commands:")
    print("  /load <file_path>  — load a PDF, TXT, or CSV into RAG")
    print("  quit / exit        — exit\n")

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

            stream_agent(user_input)

        except Exception:
            import traceback
            traceback.print_exc()
            break