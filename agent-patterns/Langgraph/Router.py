import uuid
import os
import sqlite3
from dotenv import load_dotenv
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from pydantic import BaseModel, Field

# ----------- import supervisors -----------
from knowledge_agent import invoke_knowledge_supervisor
from code_agent import (
    invoke_code_supervisor,
    resume_code_supervisor,
)
from chat_agent import invoke_chat_supervisor
from engineering_agent import invoke_engineering_supervisor

# ----------- setup -----------
load_dotenv()

llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="gpt-5-mini"
)

# ----------- logger -----------
LOG_FILE = "router_log.txt"
_turn_counter = 0
_step_counter = 0

def _write(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def init_log():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"  ROUTER DEBUG LOG  session started {ts}\n")
        f.write("=" * 80 + "\n")
    print(f"Router log -> {os.path.abspath(LOG_FILE)}")

def start_turn(user_input: str):
    global _turn_counter, _step_counter
    _turn_counter += 1
    _step_counter = 0
    ts = datetime.now().strftime("%H:%M:%S")
    _write("\n")
    _write("╔" + "═" * 78 + "╗")
    _write(f"║  TURN {_turn_counter:<3}  [{ts}]" + " " * (62 - len(str(_turn_counter))) + "║")
    _write(f"║  USER: {user_input[:70]:<70}  ║")
    _write("╚" + "═" * 78 + "╝")

def log_step(node: str, label: str, content: str):
    global _step_counter
    _step_counter += 1
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _write(f"\n  +-- [{_step_counter:02d}] {node.upper()} > {label}  ({ts})")
    _write("  |")
    for line in str(content).strip().splitlines():
        _write(f"  |  {line}")
    _write("  +" + "-" * 60)

# ----------- routing model -----------
class RoutingDecision(BaseModel):
    supervisor: str = Field(
        description="Exactly one of: 'knowledge', 'code', 'chat', 'engineering'."
    )
    reason: str = Field(description="One sentence explaining the choice.")

routing_llm = llm.with_structured_output(RoutingDecision)

VALID_SUPERVISORS = {"knowledge", "code", "chat", "engineering"}

# ----------- router state -----------
class RouterState(TypedDict):
    messages: Annotated[list, add_messages]
    raw_messages: Annotated[list, add_messages]
    active_supervisor: str
    routing_reason: str
    final_answer: str
    session_id: str
    pending_code_approval: bool
    pending_code_message: str
    conversation_summary: str
# ----------- node 1: router -----------
def router_node(state: RouterState) -> dict:
    messages = state.get("messages", [])
    if len(messages) > 10:
        old_messages = messages[:-3]
        recent_messages = messages[-3:]

        summary_parts = []
        for msg in old_messages:
            if isinstance(msg, HumanMessage):
                summary_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                summary_parts.append(f"Assistant: {msg.content}")

        summary_text = "\n".join(summary_parts)

        summarized_messages = [
            AIMessage(content=f"Conversation summary of previous discussion:\n{summary_text}")
        ] + recent_messages

        state["messages"] = summarized_messages
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    conversation_history = []
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            conversation_history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            conversation_history.append(f"Assistant: {msg.content}")

    conversation_text = "\n".join(conversation_history)

    routing_prompt = f"""You are a router directing user queries to the right AI supervisor.

Supervisors:
- knowledge    -> factual questions, research, document lookup, web search
- code         -> write/run/fix Python code
- engineering  -> explanations, architecture, tradeoffs, best practices, system design,
                  memory flow, framework comparisons, or technical reasoning
                  (output is mainly explanation, not code)
- chat         -> conversation, brainstorming, follow-ups, anything else

Recent conversation:
{conversation_text if conversation_text else "No previous conversation."}

Latest user message:
{user_message}

Important routing rules:
- If the user refers to "that", "it", "this", "the previous code", or asks a follow-up,
  use the recent conversation to understand what they mean.
- If the user is asking for explanation of code, architecture, memory, routing, LangGraph,
  agents, or system design, prefer engineering.
- If the user wants code written, fixed, debugged, or executed, prefer code.
- If the user wants factual information or research, prefer knowledge.
- Otherwise prefer chat.

Return structured output only.
"""

    log_step("router", "USER MESSAGE", user_message)
    log_step("router", "RECENT SHARED HISTORY", conversation_text if conversation_text else "No previous conversation.")
    log_step("router", "ROUTING PROMPT", routing_prompt)

    decision: RoutingDecision = routing_llm.invoke(routing_prompt)

    if decision.supervisor not in VALID_SUPERVISORS:
        decision.supervisor = "chat"

    log_step("router", "DECISION", f"{decision.supervisor.upper()} - {decision.reason}")

    return {
        "active_supervisor": decision.supervisor,
        "routing_reason":    decision.reason,
        "conversation_summary": conversation_text,
    }

# ----------- node 2: routing gate (HITL 1) -----------
def routing_gate_node(state: RouterState) -> dict:
    proposed = state["active_supervisor"]
    reason   = state.get("routing_reason", "")

    human_response: str = interrupt({
        "type":                "routing_approval",
        "proposed_supervisor": proposed,
        "reason":              reason,
        "message": (
            f"Routing to: {proposed.upper()}\n"
            f"Reason: {reason}\n"
            "Press Enter to confirm, or type a supervisor name to override "
            "(knowledge / code / engineering / chat)."
        ),
    })

    human_response = human_response.strip().lower()
    log_step("routing_gate", "HUMAN RESPONSE", human_response)

    if not human_response or human_response in ("ok", "yes", "y", proposed):
        log_step("routing_gate", "CONFIRMED", proposed)
        return {}

    if human_response in VALID_SUPERVISORS:
        log_step("routing_gate", "OVERRIDE", f"{proposed} -> {human_response}")
        return {"active_supervisor": human_response}

    log_step("routing_gate", "UNRECOGNISED - keeping original", proposed)
    return {}

# ----------- node 3: dispatch -----------
def dispatch_node(state: RouterState) -> dict:
    supervisor = state["active_supervisor"]
    session_id = state.get("session_id", "default")

    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    log_step("dispatch", "SUPERVISOR", supervisor)

    # ---- knowledge ----
    if supervisor == "knowledge":
        result = invoke_knowledge_supervisor(
            user_message=user_message,
            session_id=session_id
        )

        new_ai_message = AIMessage(content=result["answer"])

        return {
            "messages":     [new_ai_message],
            "raw_messages": [new_ai_message],
            "final_answer": result["answer"],
        }

    # ---- code ----
    elif supervisor == "code":

        # resume path - user already reviewed the code
        if state.get("pending_code_approval"):
            human_decision = state.get("pending_code_message", "").strip()
            log_step("dispatch", "RESUMING CODE APPROVAL", human_decision)

            phase2 = resume_code_supervisor(
                human_input=human_decision,
                session_id=session_id,
                conversation_summary=state.get("conversation_summary", "")
            )

            new_ai_message = AIMessage(content=phase2["answer"])

            return {
                "messages":              [new_ai_message],
                "raw_messages":          [new_ai_message],
                "final_answer":          phase2["answer"],
                "pending_code_approval": False,
                "pending_code_message":  "",
            }

        # first pass - generate code and wait for approval
        phase1 = invoke_code_supervisor(
            user_message=user_message,
            session_id=session_id,
            conversation_summary=state.get("conversation_summary", "")
        )

        if phase1.get("status") == "pending_approval":
            return {
                "pending_code_approval": True,
                "pending_code_message":  "",
            }

        new_ai_message = AIMessage(content=phase1["answer"])

        return {
            "messages":              [new_ai_message],
            "raw_messages":          [new_ai_message],
            "final_answer":          phase1["answer"],
            "pending_code_approval": False,
        }

    # ---- engineering ----
    elif supervisor == "engineering":
        result = invoke_engineering_supervisor(
            user_message=user_message,
            session_id=session_id,
            conversation_summary=state.get("conversation_summary", "")
        )

        new_ai_message = AIMessage(content=result["answer"])

        return {
            "messages":     [new_ai_message],
            "raw_messages": [new_ai_message],
            "final_answer": result["answer"],
        }

    # ---- chat ----
    elif supervisor == "chat":
        result = invoke_chat_supervisor(
            user_message=user_message,
            conversation_history=state.get("conversation_summary", ""),
        )

        return {
            "messages":     result["messages"],  #  router merges chat's messages directly
            "raw_messages": result["messages"],
            "final_answer": result["answer"],
        }

    # ---- fallback ----
    fallback = "I couldn't determine how to handle that request. Please try rephrasing."
    new_ai_message = AIMessage(content=fallback)

    return {
        "messages":     [new_ai_message],
        "raw_messages": [new_ai_message],
        "final_answer": fallback,
    }

# ----------- node 4: approval gate (HITL 2, code only) -----------
def approval_gate_node(state: RouterState) -> dict:
    human_response: str = interrupt({
        "type":    "code_approval",
        "message": "Respond with 'approve', 'cancel', or paste edited code.",
    })
    log_step("approval_gate", "HUMAN RESPONSE", human_response.strip())
    return {"pending_code_message": human_response.strip()}

# ----------- routing helpers -----------
def route_after_dispatch(state: RouterState) -> str:
    if state.get("pending_code_approval"):
        return "approval_gate"
    return END

def route_after_approval_gate(state: RouterState) -> str:
    return "dispatch"

# ----------- graph -----------
sqlite_conn  = sqlite3.connect("router_checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(sqlite_conn)

def build_router_graph():
    workflow = StateGraph(RouterState)

    workflow.add_node("router",        router_node)
    workflow.add_node("routing_gate",  routing_gate_node)
    workflow.add_node("dispatch",      dispatch_node)
    workflow.add_node("approval_gate", approval_gate_node)

    workflow.add_edge(START, "router")
    workflow.add_edge("router", "routing_gate")
    workflow.add_edge("routing_gate", "dispatch")

    workflow.add_conditional_edges(
        "dispatch",
        route_after_dispatch,
        {
            "approval_gate": "approval_gate",
            END:             END,
        },
    )
    workflow.add_conditional_edges(
        "approval_gate",
        route_after_approval_gate,
        {
            "dispatch": "dispatch",
        },
    )

    return workflow.compile(checkpointer=checkpointer)

router_graph = build_router_graph()

# ----------- display helper -----------
def _print_routing_prompt(supervisor: str, reason: str):
    print("\n" + "-" * 60)
    print("ROUTING DECISION")
    print("-" * 60)
    print(f"  Supervisor : {supervisor.upper()}")
    print(f"  Reason     : {reason}")
    print("-" * 60)
    print("  Enter / ok / yes  -> confirm")
    print("  knowledge / code / engineering / chat  -> override")
    print("-" * 60)

# ----------- main loop -----------
if __name__ == "__main__":
    init_log()

    print("\nMulti-Supervisor Router (2x HITL) ready.")
    print("  HITL 1 -> confirm or override routing decision before any work starts.")
    print("  HITL 2 -> review generated code before execution (code requests only).")
    print("  Press Enter to start a new session, or paste an existing session ID to resume.")
    print("  Type 'exit' to quit.\n")

    raw        = input("Session ID (blank = new): ").strip()
    SESSION_ID = raw if raw else str(uuid.uuid4())

    print(f"  Session: {SESSION_ID}\n")

    config = {"configurable": {"thread_id": SESSION_ID}}

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print(f"\nLog: {os.path.abspath(LOG_FILE)}")
            break

        if not user_input:
            continue

        start_turn(user_input)

        print(f"\n{'='*60}")
        print(f"INPUT: {user_input[:80]}")
        print(f"{'='*60}")

        human_message = HumanMessage(content=user_input)

        result = router_graph.invoke(
            {
                "messages":              [human_message],
                "raw_messages":          [human_message],
                "active_supervisor":     "",
                "routing_reason":        "",
                "final_answer":          "",
                "session_id":            SESSION_ID,
                "pending_code_approval": False,
                "pending_code_message":  "",
            },
            config=config,
        )

        graph_state    = router_graph.get_state(config)
        is_interrupted = bool(graph_state.tasks)

        if is_interrupted:
            interrupt_payload = graph_state.tasks[0].interrupts[0].value
            interrupt_type    = interrupt_payload.get("type", "")

            # HITL 1 - routing approval
            if interrupt_type == "routing_approval":
                _print_routing_prompt(
                    interrupt_payload["proposed_supervisor"],
                    interrupt_payload["reason"],
                )

                routing_decision = input("\nYour decision: ").strip()
                if not routing_decision:
                    routing_decision = "ok"

                result = router_graph.invoke(
                    Command(resume=routing_decision),
                    config=config,
                )

                graph_state    = router_graph.get_state(config)
                is_interrupted = bool(graph_state.tasks)

            # HITL 2 - code approval
            if is_interrupted:
                interrupt_payload = graph_state.tasks[0].interrupts[0].value

                if interrupt_payload.get("type") == "code_approval":
                    print("\n" + "-" * 60)
                    print("CODE REVIEW REQUIRED")
                    print("-" * 60)
                    print("  approve      -> run as-is")
                    print("  cancel       -> abort")
                    print("  (paste code) -> run your edited version")
                    print("-" * 60)

                    code_decision = input("\nYour decision: ").strip()

                    result = router_graph.invoke(
                        Command(resume=code_decision),
                        config=config,
                    )

        supervisor = result.get("active_supervisor", "").upper()

        print(f"\n{'='*60}")
        print(f"ANSWER [{supervisor}]:")
        print(result.get("final_answer", ""))
        print(f"{'='*60}\n")