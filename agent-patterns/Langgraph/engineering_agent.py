from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
from typing import Optional, List
import os
import sqlite3
from datetime import datetime

from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# loads AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY from .env
load_dotenv()

# ----------- logger -----------
LOG_FILE = "engineering_supervisor_log.txt"

def _write(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def init_log():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"  ENGINEERING SUPERVISOR LOG  session started {ts}\n")
        f.write("=" * 80 + "\n")

def log_prompt(node: str, prompt: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _write(f"\n  +-- PROMPT SENT TO LLM [{node.upper()}]  ({ts})")
    _write("  |")
    for line in prompt.strip().splitlines():
        _write(f"  |  {line}")
    _write("  +" + "-" * 60)

# ----------- state -----------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    raw_messages: Annotated[list, add_messages]
    conversation_summary: str
    research: str
    code: str
    comparison: str

    plan: dict

    research_memory: list
    code_memory: list
    comparison_memory: list
# ----------- schema -----------
class AgentTask(BaseModel):
    needed: bool
    task: Optional[str] = None

class SupervisorPlan(BaseModel):
    research: AgentTask
    code: AgentTask
    comparison: AgentTask

# ----------- llm -----------
llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
)

structured_llm = llm.with_structured_output(SupervisorPlan)

# ----------- helpers -----------
def get_last_user_message(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""

def format_full_conversation(messages: list) -> str:
    if not messages:
        return "No previous conversation."

    lines = []

    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")

    return "\n".join(lines)

def format_full_memory(memory: list) -> str:
    if not memory:
        return "None"
    return "\n\n---\n\n".join(memory)

# ----------- supervisor -----------
def supervisor(state: State):
    
    last_message = get_last_user_message(state["messages"])
    conversation_summary = state.get("conversation_summary", "")
    decision_prompt = f"""
You are a supervisor deciding which agents to call.

Agents:
- research -> explanations
- code -> coding, debugging, logic
- comparison -> comparisons


Conversation summary:
{conversation_summary}

Latest user message:
{last_message}

Rules:
- Use the conversation summary for context if it is provided
- Use the latest user message to determine current intent
- If the user is asking a follow-up question, infer what they mean from the prior conversation
- Avoid calling all agents unnecessarily
- Coding/debugging/programming questions should usually go to the code agent
- Explanations should go to the research agent
- Side-by-side comparisons should go to the comparison agent

Return structured output only.
"""

    log_prompt("supervisor", decision_prompt)

    plan: SupervisorPlan = structured_llm.invoke(decision_prompt)

    return {
        "plan": plan.model_dump(),
    }

# ----------- routing -----------
def route_from_supervisor(state: State) -> List[str]:
    plan = state["plan"]
    routes = []

    if plan["research"]["needed"]:
        routes.append("research")

    if plan["code"]["needed"]:
        routes.append("code")

    if plan["comparison"]["needed"]:
        routes.append("compare")

    if not routes:
        routes.append("merge")

    return routes

# ----------- research node -----------
def research_node(state: State):
    plan = state["plan"]

    if not plan["research"]["needed"]:
        return {}

    memory = state.get("research_memory", [])
    
    full_memory = format_full_memory(memory)

    prompt = f"""
You are a research agent.


Your previous research outputs this session:
{full_memory}

Task:
{plan['research']['task']}

Rules:
- Do not repeat information already given in previous research outputs
- Stay consistent with your previous answers
- No code
- Be concise
"""

    log_prompt("research", prompt)

    response = llm.invoke(prompt)

    return {
        "research": response.content,
        "research_memory": memory + [response.content],
    }

# ----------- code node -----------
def code_node(state: State):
    plan = state["plan"]

    if not plan["code"]["needed"]:
        return {}

    memory = state.get("code_memory", [])
    
    full_memory = format_full_memory(memory)

    prompt = f"""
You are a code agent.


Your previous code outputs this session:
{full_memory}

Task:
{plan['code']['task']}

Rules:
- Stay consistent with previous code outputs
- Extend or update previous logic when appropriate
- Do not rewrite everything unless necessary
- Be concise
"""

    log_prompt("code", prompt)

    response = llm.invoke(prompt)

    return {
        "code": response.content,
        "code_memory": memory + [response.content],
    }

# ----------- compare node -----------
def compare_node(state: State):
    plan = state["plan"]

    if not plan["comparison"]["needed"]:
        return {}

    memory = state.get("comparison_memory", [])
    
    full_memory = format_full_memory(memory)

    prompt = f"""
You are a comparison agent.


Your previous comparison outputs this session:
{full_memory}

Task:
{plan['comparison']['task']}

Rules:
- Do not repeat previous comparisons
- Stay consistent with prior comparison outputs
- Be structured and concise
"""

    log_prompt("compare", prompt)

    response = llm.invoke(prompt)

    return {
        "comparison": response.content,
        "comparison_memory": memory + [response.content],
    }

# ----------- merge node -----------
def merge_node(state: State):
    parts = []

    if state.get("research"):
        parts.append(f"Explanation:\n{state['research']}")

    if state.get("code"):
        parts.append(f"Code:\n{state['code']}")

    if state.get("comparison"):
        parts.append(f"Comparison:\n{state['comparison']}")

    final = "\n\n".join(parts) if parts else "No results"

    new_ai_message = AIMessage(content=final)

    return {
        "messages": [new_ai_message],
        "raw_messages": [new_ai_message]
    }

# ----------- graph -----------
sqlite_conn = sqlite3.connect(
    "engineering_checkpoints.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(sqlite_conn)

def build_engineering_supervisor_graph():
    builder = StateGraph(State)

    builder.add_node("supervisor", supervisor)
    builder.add_node("research", research_node)
    builder.add_node("code", code_node)
    builder.add_node("compare", compare_node)
    builder.add_node("merge", merge_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "research": "research",
            "code": "code",
            "compare": "compare",
            "merge": "merge",
        }
    )

    builder.add_edge("research", "merge")
    builder.add_edge("code", "merge")
    builder.add_edge("compare", "merge")
    builder.add_edge("merge", END)

    return builder.compile(checkpointer=checkpointer)

engineering_graph = build_engineering_supervisor_graph()

# ----------- entry point -----------
def invoke_engineering_supervisor(
    user_message: str,
    session_id: str,
    conversation_summary: str = "",   
) -> dict:
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    result = engineering_graph.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "raw_messages": [HumanMessage(content=user_message)],
            "conversation_summary": conversation_summary,
            "research": "",
            "code": "",
            "comparison": "",
            "plan": {},
            # Do NOT pass research_memory/code_memory/comparison_memory
            # SqliteSaver restores them automatically from the checkpoint
        },
        config=config,
    )

    final_answer = (
        result["messages"][-1].content
        if result.get("messages")
        else ""
    )

    return {
        "answer": final_answer
    }
