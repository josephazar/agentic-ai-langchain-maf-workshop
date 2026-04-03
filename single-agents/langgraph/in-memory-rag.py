"""
ReAct Agent with In-Memory RAG
- Accepts PDF, TXT, CSV file uploads at runtime
- Chunks, embeds, and stores in an ephemeral in-memory vector store
- Agent answers questions from uploaded documents only
- Built with LangGraph (same structure as your existing agent)
"""

import os
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# --- Document loading & RAG dependencies ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # in-memory, no server needed

# ── State ────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ── In-memory vector store (ephemeral per session) ───────────────────────────

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="text-embedding-3-large",   # change to your deployment name
    api_version="2024-12-01-preview",
)

# Global in-memory store – reset each session (or call load_document to add more)
vector_store: FAISS | None = None

def load_document(file_path: str) -> str:
    """Load a PDF, TXT, or CSV file and add it to the in-memory vector store."""
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
        vector_store.add_documents(chunks)   # add to existing store

    return f" Loaded '{os.path.basename(file_path)}' — {len(chunks)} chunks indexed."

# ── RAG Tool ─────────────────────────────────────────────────────────────────

@tool
def search_documents(query: str) -> str:
    """
    Search the uploaded documents for information relevant to the query.
    Always use this tool when the user asks any question — answers must come
    from the uploaded files only.
    """
    if vector_store is None:
        return "No documents have been loaded yet. Please upload a PDF, TXT, or CSV file first."

    results = vector_store.similarity_search(query, k=4)
    if not results:
        return "No relevant content found in the uploaded documents."

    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in results
    )
    return context

# ── Agent setup ───────────────────────────────────────────────────────────────

tools = [search_documents]
tool_node = ToolNode(tools)

llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="gpt-5-mini",
)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a document assistant. You have access to one tool:\n"
    "- search_documents: searches the uploaded documents for relevant content.\n\n"
    "Rules:\n"
    "1. ALWAYS call search_documents before answering any question.\n"
    "2. Base your answer ONLY on the tool's results — do not use outside knowledge.\n"
    "3. If the documents don't contain the answer, say so clearly.\n"
    "4. Do not ask the user for clarification before trying the tool."
))

def agent(state: State):
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_router(state: State):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END

# ── Graph ─────────────────────────────────────────────────────────────────────

memory = MemorySaver()

builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tool_router, ["tools", END])
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=memory)

# ── Streaming helper ──────────────────────────────────────────────────────────

def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [("user", user_input)]},
        {"configurable": {"thread_id": "1"}},
    ):
        for value in event.values():
            last_msg = value["messages"][-1]
            if last_msg.content:
                print("Assistant:", last_msg.content)

# ── CLI loop ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ReAct Agent with In-Memory RAG")
    print("Commands:")
    print("  /load <file_path>  — load a PDF, TXT, or CSV file")
    print("  quit / exit / q    — exit\n")

    while True:
        try:
            user_input = input("User: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Special command to load documents at runtime
            if user_input.startswith("/load "):
                file_path = user_input[6:].strip()
                result = load_document(file_path)
                print(f"System: {result}\n")
                continue

            stream_graph_updates(user_input)

        except Exception:
            import traceback
            traceback.print_exc()
            break