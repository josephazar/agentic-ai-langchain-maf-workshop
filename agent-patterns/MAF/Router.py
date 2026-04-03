import asyncio
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv
from agent_framework.azure import AzureOpenAIResponsesClient
from memory_store import load_session, save_session

# ========================
# IMPORT SUPERVISORS
# ========================
from knowledge_agent import invoke_knowledge_supervisor
from code_agent import invoke_code_supervisor, resume_code_supervisor
from chat_agent import invoke_chat_supervisor
from engineering_agent import invoke_engineering_supervisor

load_dotenv()

ROUTER_MEMORY_FILE  = "router_memory.json"
VALID_SUPERVISORS   = {"knowledge", "code", "chat", "engineering"}
SUMMARY_THRESHOLD   = 10   # summarize when history exceeds this
RECENT_KEEP         = 3    # always keep last N turns raw


# ========================
# HELPERS
# ========================

def format_history_for_prompt(conversation_history: list) -> str:
    if not conversation_history:
        return ""
    lines = []
    for turn in conversation_history:
        role  = "User" if turn["role"] == "user" else "Assistant"
        agent = f" [{turn['agent'].upper()}]" if turn.get("agent") else ""
        lines.append(f"{role}{agent}: {turn['content']}")
    return "\n".join(lines)


def build_trimmed_history(conversation_history: list, summary: str) -> str:
    """
    Returns a string combining:
    - the rolling summary (if any)
    - the last RECENT_KEEP raw turns
    This is what gets passed down to agents.
    """
    recent = conversation_history[-RECENT_KEEP:]
    recent_text = format_history_for_prompt(recent)

    if summary:
        return f"Summary of earlier conversation:\n{summary}\n\nRecent messages:\n{recent_text}"
    return recent_text


# ========================
# SUMMARIZER
# ========================

async def summarize_history(old_summary: str, turns_to_summarize: list) -> str:
    turns_text = format_history_for_prompt(turns_to_summarize)

    prompt = f"""You maintain a running summary of a conversation.

Previous summary:
{old_summary if old_summary else "No summary yet."}

New messages to incorporate:
{turns_text}

Update the summary with important context, topics discussed, decisions made,
and any details that may matter later. Be concise.

Respond with ONLY the updated summary text, nothing else."""

    agent = AzureOpenAIResponsesClient().create_agent(
        name="Summary Agent",
        description="Maintains a running conversation summary.",
        instructions="You are a summarizer. Respond with ONLY the updated summary text.",
    )
    response = await agent.run(prompt)
    return response.text.strip()


async def maybe_summarize(
    conversation_history: list,
    current_summary: str,
) -> str:
    """
    If history exceeds SUMMARY_THRESHOLD, summarize everything except the last RECENT_KEEP turns.
    Returns updated summary (or unchanged if threshold not met).
    """
    if len(conversation_history) <= SUMMARY_THRESHOLD:
        return current_summary

    turns_to_summarize = conversation_history[:-RECENT_KEEP]
    updated_summary = await summarize_history(current_summary, turns_to_summarize)
    return updated_summary


# ========================
# AGENT FACTORY
# ========================

def create_router_agent(trimmed_history_text: str):
    memory_context = f"\n\nConversation so far:\n{trimmed_history_text}" if trimmed_history_text else ""

    return AzureOpenAIResponsesClient().create_agent(
        name="Router",
        description="Routes user queries to the correct supervisor.",
        instructions=f"""You are a router directing user queries to the right AI supervisor.

Supervisors:
- knowledge    -> factual questions, research, document lookup, web search
- code         -> write/run/fix Python code
- engineering  -> explanations, architecture, tradeoffs, best practices, system design,
                  memory flow, framework comparisons, technical reasoning
                  (output is mainly explanation, not code)
- chat         -> conversation, brainstorming, follow-ups, anything else
{memory_context}

Respond ONLY in this exact JSON format, nothing else:
{{
  "supervisor": "knowledge" | "code" | "chat" | "engineering",
  "reason": "one sentence explaining the choice"
}}""",
    )


# ========================
# STEP 1 — ROUTE
# ========================

async def route(
    user_message: str,
    conversation_history: list,
    current_summary: str,
) -> dict:
    print("  Router deciding supervisor...")

    trimmed = build_trimmed_history(conversation_history, current_summary)

    agent    = create_router_agent(trimmed)
    response = await agent.run(user_message)

    try:
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        decision   = json.loads(raw.strip())
        supervisor = decision.get("supervisor", "chat")
        reason     = decision.get("reason", "")
    except Exception as e:
        print(f"  Failed to parse router response: {e}")
        supervisor = "chat"
        reason     = "Failed to parse routing decision -- defaulting to chat."

    if supervisor not in VALID_SUPERVISORS:
        print(f"  Invalid supervisor returned: {supervisor}")
        supervisor = "chat"

    print(f"  Proposed routing: {supervisor.upper()} -- {reason}")

    return {"supervisor": supervisor, "reason": reason}


# ========================
# STEP 2 — ROUTING GATE (HITL 1)
# ========================

def routing_gate(proposed: str, reason: str) -> str:
    print("\n" + "-" * 60)
    print("ROUTING DECISION")
    print("-" * 60)
    print(f"  Supervisor : {proposed.upper()}")
    print(f"  Reason     : {reason}")
    print("-" * 60)
    print("  Enter / ok / yes                       -> confirm")
    print("  knowledge / code / engineering / chat  -> override")
    print("-" * 60)

    human_response = input("\nYour decision: ").strip().lower()

    if not human_response or human_response in ("ok", "yes", "y", proposed):
        print(f"  Confirmed: {proposed.upper()}")
        return proposed

    if human_response in VALID_SUPERVISORS:
        print(f"  Overridden to: {human_response.upper()}")
        return human_response

    print(f"  '{human_response}' not recognised. Keeping: {proposed.upper()}")
    return proposed


# ========================
# STEP 3 — DISPATCH
# ========================

async def dispatch(
    supervisor: str,
    user_message: str,
    session_id: str,
    conversation_history: list,
    current_summary: str,
) -> dict:
    print(f"  Dispatching to {supervisor.upper()}...")

    # Build trimmed history to pass down to agents
    trimmed_history = build_trimmed_history(conversation_history, current_summary)

    # -- knowledge ----------------------------------------------------------
    if supervisor == "knowledge":
        result = await invoke_knowledge_supervisor(
            user_message, trimmed_history
        )
        return {
            "answer":                result["answer"],
            "pending_code_approval": False,
            "pending_code":          "",
        }

    # -- code ---------------------------------------------------------------
    elif supervisor == "code":
        phase1 = await invoke_code_supervisor(
            user_message, session_id, trimmed_history
        )

        print("\n" + "-" * 60)
        print("CODE REVIEW REQUIRED")
        print("-" * 60)
        print(phase1["code"])
        print("\n" + "-" * 60)
        print("  approve      -> run as-is")
        print("  cancel       -> abort")
        print("  (paste code) -> run your edited version")
        print("-" * 60)
        decision = input("\nYour decision: ").strip()

        phase2 = await resume_code_supervisor(
            user_message=user_message,
            human_input=decision,
            code=phase1["code"],
            session_id=session_id,
            conversation_history=trimmed_history, 
        )
        return {
            "answer":                phase2.get("answer", ""),
            "pending_code_approval": False,
            "pending_code":          "",
        }

    # -- engineering --------------------------------------------------------
    elif supervisor == "engineering":
        result = await invoke_engineering_supervisor(
            user_message, session_id, trimmed_history
        )
        return {
            "answer":                result["answer"],
            "pending_code_approval": False,
            "pending_code":          "",
        }

    # -- chat ---------------------------------------------------------------
    elif supervisor == "chat":
        result = await invoke_chat_supervisor(
            user_message, session_id, trimmed_history
        )
        return {
            "answer":                result["answer"],
            "pending_code_approval": False,
            "pending_code":          "",
        }

    # -- fallback -----------------------------------------------------------
    return {
        "answer":                "I couldn't determine how to handle that request. Please try rephrasing.",
        "pending_code_approval": False,
        "pending_code":          "",
    }


# ========================
# MAIN LOOP
# ========================

async def main():
    print("\nMulti-Supervisor Router ready.")
    print("   HITL 1 -- confirm or override routing decision.")
    print("   HITL 2 -- review generated code before execution.")
    print("   Press Enter for a new session or paste an existing session ID.")
    print("   Type 'exit' to quit.\n")

    raw        = input("Session ID (blank = new): ").strip()
    session_id = raw if raw else str(uuid.uuid4())
    print(f"   Session: {session_id}\n")

    # Load full raw history + summary for this session
    mem                  = load_session(ROUTER_MEMORY_FILE, session_id)
    conversation_history = mem.get("conversation_history", [])
    current_summary      = mem.get("conversation_summary", "")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break
        if not user_input:
            continue

        print(f"\n{'='*60}\nINPUT: {user_input[:80]}\n{'='*60}")

        # Step 1 -- route
        routing = await route(user_input, conversation_history, current_summary)

        # Step 2 -- routing gate (HITL 1)
        confirmed_supervisor = routing_gate(routing["supervisor"], routing["reason"])

        # Step 3 -- dispatch (agents receive trimmed history)
        result = await dispatch(
            confirmed_supervisor, user_input, session_id,
            conversation_history, current_summary,
        )

        # Append to full raw history
        conversation_history.append({
            "role":    "user",
            "content": user_input,
            "agent":   None,
        })
        conversation_history.append({
            "role":    "assistant",
            "content": result["answer"],
            "agent":   confirmed_supervisor,
        })

        # Summarize if history exceeds threshold (once per turn, after appending)
        current_summary = await maybe_summarize(conversation_history, current_summary)

        # Persist full raw history + summary
        save_session(ROUTER_MEMORY_FILE, session_id, {
            "conversation_history": conversation_history,
            "conversation_summary": current_summary,
        })

        print(f"\n{'='*60}")
        print(f"ANSWER [{confirmed_supervisor.upper()}]:")
        print(result["answer"])
        print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())