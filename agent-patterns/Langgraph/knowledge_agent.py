import os
import asyncio
from dotenv import load_dotenv
from datetime import datetime

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from pydantic import BaseModel, Field
import sqlite3

# ========================
# SETUP
# ========================
load_dotenv()

llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="gpt-5-mini"
)

# ========================
# LOGGER
# ========================
LOG_FILE = "knowledge_supervisor_log.txt"

def _write(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def init_log():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"  KNOWLEDGE SUPERVISOR LOG  session started {ts}\n")
        f.write("=" * 80 + "\n")

def log_prompt(node: str, prompt: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _write(f"\n  +-- PROMPT SENT TO LLM [{node.upper()}]  ({ts})")
    _write("  |")
    for line in prompt.strip().splitlines():
        _write(f"  |  {line}")
    _write("  +" + "-" * 60)

def log_section(label: str, content: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _write(f"\n  +-- [{label.upper()}]  ({ts})")
    _write("  |")
    for line in content.strip().splitlines():
        _write(f"  |  {line}")
    _write("  +" + "-" * 60)

init_log()

# ========================
# VECTOR STORE
# ========================
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
    model="text-embedding-3-large"
)

vectorstore = FAISS.load_local(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "utility",
            "faiss_index"
        )
    ),
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever()

# ========================
# TAVILY SEARCH
# ========================
tavily = TavilySearchResults(max_results=5)

# ========================
# STRUCTURED OUTPUT MODELS
# ========================
class AgentTask(BaseModel):
    task: Optional[str] = Field(
        default=None,
        description="The specific task for this agent. Set to null to skip."
    )


class KnowledgePlan(BaseModel):
    rag: AgentTask = Field(description="RAG agent task.")
    web: AgentTask = Field(description="Web search agent task.")


structured_llm = llm.with_structured_output(KnowledgePlan)

# ========================
# STATE
# ========================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    raw_messages: Annotated[list, add_messages]
    plan: dict
    rag_result: str
    web_result: str
    final_answer: str


# ========================
# HELPERS
# ========================
def get_last_user_message(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ========================
# SUPERVISOR NODE
# ========================
def supervisor(state: State):
    last_message = get_last_user_message(state["messages"])

    full_history = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state["messages"]
        if isinstance(m, (HumanMessage, AIMessage))
    ])

    decision_prompt = f"""

You are a Knowledge Supervisor deciding which agents to use.

Available agents:

- rag → retrieves from internal vector store

- web → searches the internet via Tavily

Full conversation history:

{full_history}

Latest user message:

{last_message}

Rules:

- Use rag for internal docs, blogs, or curated content

- Use web for recent/current/external information

- Use both when useful

- Return null for unused agents

- The task field must be a SHORT search query (max 10 words), not a paragraph

"""

    log_prompt("supervisor", decision_prompt)

    plan: KnowledgePlan = structured_llm.invoke(decision_prompt)

    log_section("supervisor_plan", f"RAG task: {plan.rag.task}\nWEB task: {plan.web.task}")

    return {
        "plan": plan.model_dump(),
    }

# ========================
# RAG AGENT
# ========================
async def rag_agent(task: str, question: str) -> str:
    query = task if task else question
    docs = await asyncio.to_thread(retriever.invoke, query)
    if not docs:
        log_section("rag_agent", f"Query: {query}\nResult: NO_RESULTS")
        return "NO_RESULTS"

    output = "\n\n".join([d.page_content for d in docs])
    log_section("rag_agent", f"Query: {query}\n\n--- RAW RAG CHUNKS ---\n{output}")
    return output


# ========================
# WEB AGENT
# ========================
async def web_agent(task: str, question: str) -> str:
    query = task if task else question
    results = await asyncio.to_thread(tavily.invoke, {"query": query})
    if not results:
        log_section("web_agent", f"Query: {query}\nResult: NO_RESULTS")
        return "NO_RESULTS"

    formatted_parts = []
    for r in results:
        if isinstance(r, dict):
            formatted_parts.append(
                f"Source: {r.get('url', 'unknown')}\n{r.get('content', '')}"
            )
        else:
            formatted_parts.append(str(r))

    output = "\n\n".join(formatted_parts)
    log_section("web_agent", f"Query: {query}\n\n--- RAW TAVILY RESULTS ---\n{output}")
    return output


# ========================
# PARALLEL AGENT EXECUTION
# ========================
async def run_agents_async(state: State) -> dict:
    plan = state["plan"]
    question = get_last_user_message(state["messages"])

    rag_task = plan.get("rag", {}).get("task")
    web_task = plan.get("web", {}).get("task")

    coroutines = {}
    if rag_task:
        coroutines["rag"] = rag_agent(rag_task, question)
    if web_task:
        coroutines["web"] = web_agent(web_task, question)

    if not coroutines:
        log_section("parallel_agents", "No agents called — both tasks were null.")
        return {"rag_result": "NO_RESULTS", "web_result": "NO_RESULTS"}

    keys = list(coroutines.keys())
    results = await asyncio.gather(*coroutines.values())
    result_map = dict(zip(keys, results))

    return {
        "rag_result": result_map.get("rag", "NOT_CALLED"),
        "web_result": result_map.get("web", "NOT_CALLED"),
    }


def parallel_agents_node(state: State):
    return asyncio.run(run_agents_async(state))


# ========================
# MERGER NODE
# ========================
def merger_node(state: State):
    question = get_last_user_message(state["messages"])

    rag_result = state.get("rag_result", "NOT_CALLED")
    web_result = state.get("web_result", "NOT_CALLED")

    context_parts = []
    if rag_result not in ["NOT_CALLED", "NO_RESULTS"]:
        context_parts.append(f"=== Internal Knowledge ===\n{rag_result}")
    if web_result not in ["NOT_CALLED", "NO_RESULTS"]:
        context_parts.append(f"=== Web Results ===\n{web_result}")

    context_section = "\n\n".join(context_parts)

    merge_prompt = f"""
You are a knowledge assistant.

User question:
{question}

Context:
{context_section}

Answer clearly and naturally.
"""

    log_prompt("merger", merge_prompt)

    response = llm.invoke([HumanMessage(content=merge_prompt)])

    log_section("merger_response", response.content)

    return {
        "messages": [AIMessage(content=response.content)],
        "raw_messages": [AIMessage(content=response.content)],
        "final_answer": response.content,
    }


# ========================
# GRAPH
# ========================

#  SqliteSaver is the single source of truth — no separate custom table needed
sqlite_conn = sqlite3.connect("knowledge_checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(sqlite_conn)

workflow = StateGraph(State)

workflow.add_node("supervisor", supervisor)
workflow.add_node("parallel_agents", parallel_agents_node)
workflow.add_node("merger", merger_node)

workflow.add_edge(START, "supervisor")
workflow.add_edge("supervisor", "parallel_agents")
workflow.add_edge("parallel_agents", "merger")
workflow.add_edge("merger", END)

knowledge_graph = workflow.compile(checkpointer=checkpointer)


# ========================
# ROUTER ENTRY POINT
#
# session_id is passed as thread_id — SqliteSaver handles everything.
# conversation_summary is automatically restored from the checkpointer.
# No manual load/save, no custom SQLite table.
# ========================
def invoke_knowledge_supervisor(user_message: str, session_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": session_id}}

    result = knowledge_graph.invoke(
        {
            "messages":     [HumanMessage(content=user_message)],
            "raw_messages": [HumanMessage(content=user_message)],
            "plan":         {},
            "rag_result":   "",
            "web_result":   "",
            "final_answer": "",
        },
        config=config,
    )

    return {
        "answer": result.get("final_answer", ""),
        "thread_id": session_id,
    }