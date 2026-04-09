import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
)

# ----------- logger -----------
LOG_FILE = "chat_supervisor_log.txt"

def _write(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_prompt(prompt: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _write(f"\n  +-- PROMPT SENT TO LLM  ({ts})")
    _write("  |")
    for line in prompt.strip().splitlines():
        _write(f"  |  {line}")
    _write("  +" + "-" * 60)

# ----------- state -----------
# no checkpointer, no session_id — router owns all persistence
# conversation_history is passed in from the router each call (trimmed text)
# messages uses add_messages so the graph can append the AI reply cleanly
class ChatState(TypedDict):
    messages:             Annotated[list, add_messages]
    user_message:         str
    conversation_history: str   # trimmed history string from router
    final_answer:         str

# ----------- chat node -----------
def chat_node(state: ChatState) -> dict:
    user_message         = state["user_message"]
    conversation_history = state.get("conversation_history", "").strip()
    history_block        = conversation_history if conversation_history else "No previous conversation."

    prompt = f"""You are a conversational AI assistant.

Your role:
- Friendly conversation
- Brainstorming
- Follow-up discussion
- Casual support
- Open-ended thinking
- Helping the user continue previous topics naturally

Full conversation history:
{history_block}

Current user message:
{user_message}

Rules:
- Be natural and conversational
- Continue previous context when useful
- Avoid sounding robotic
- Keep responses moderately concise
- If the user asks a follow-up, assume they refer to the most recent relevant topic
"""

    log_prompt(prompt)

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_message),
    ])

    answer         = response.content
    new_ai_message = AIMessage(content=answer)

    return {
        "final_answer": answer,
        "messages":     [new_ai_message],
    }

# ----------- graph -----------
# no checkpointer — router owns state persistence via its own SQLite
# stateless per-call: each invoke is a fresh run with history injected via conversation_history
def build_chat_graph():
    workflow = StateGraph(ChatState)
    workflow.add_node("chat", chat_node)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    return workflow.compile()   # no checkpointer

chat_graph = build_chat_graph()

# ----------- entry point -----------
# conversation_history: trimmed conversation text built by the router
# returns answer string + the updated messages list so the router can merge them
def invoke_chat_supervisor(
    user_message:         str,
    conversation_history: str = "",
) -> dict:
    result = chat_graph.invoke(
        {
            "messages":             [HumanMessage(content=user_message)],
            "user_message":         user_message,
            "conversation_history": conversation_history,
            "final_answer":         "",
        }
    )

    return {
        "answer":   result["final_answer"],
        "messages": result["messages"],   # list of HumanMessage + AIMessage from this turn
    }