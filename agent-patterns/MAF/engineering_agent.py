import asyncio
import uuid
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from memory_store import load_session, save_session

from agent_framework.azure import AzureOpenAIResponsesClient

load_dotenv()

ENGINEERING_MEMORY_FILE = "engineering_memory.json"
DEBUG_LOG_FILE = "engineering_debug.log"


# ========================
# DEBUG LOGGER
# ========================

def log(session_id: str, label: str, content: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n{'='*60}\n"
        f"[{timestamp}] [{session_id[:8]}] {label}\n"
        f"{'='*60}\n"
        f"{content}\n"
    )
    print(block)
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(block)


# ========================
# MEMORY HELPERS
# ========================

def load_turns(session_id: str) -> list:
    raw = load_session(ENGINEERING_MEMORY_FILE, session_id)
    if isinstance(raw, list):
        return raw
    if raw:
        return [raw]
    return []


def save_turns(session_id: str, turns: list):
    save_session(ENGINEERING_MEMORY_FILE, session_id, turns)


def get_all_agent_memory(turns: list, key: str) -> str:
    outputs = []
    for turn in turns:
        val = turn.get(key, "")
        if val:
            outputs.append(val)
    return "\n\n---\n\n".join(outputs) if outputs else ""


# ========================
# AGENT FACTORY HELPERS
# ========================

def create_supervisor_agent(conversation_history: str):
    return AzureOpenAIResponsesClient().create_agent(
        name="Engineering Supervisor",
        description="Plans which sub-agents to call based on the user request.",
        instructions=f"""You are a supervisor deciding which agents to call.

Agents available:
- research → explanations, concepts, how things work
- code     → coding, debugging, logic, implementation
- compare  → comparisons, tradeoffs, pros/cons

Conversation history:
{conversation_history or "This is the first message."}

Rules:
- Use the conversation history for full context
- Use the latest user message for intent
- If it's a follow-up, infer intent from the history
- Avoid calling all agents unnecessarily
- Write a self-contained task string for each agent you enable —
  include all relevant context so the agent doesn't need the history
- Coding tasks → prefer code agent

Respond ONLY in this exact JSON format, nothing else:
{{
  "research": {{ "needed": true/false, "task": "self-contained task string or null" }},
  "code":     {{ "needed": true/false, "task": "self-contained task string or null" }},
  "compare":  {{ "needed": true/false, "task": "self-contained task string or null" }}
}}""",
    )


def create_research_agent(prev_research_output: str):
    return AzureOpenAIResponsesClient().create_agent(
        name="Research Agent",
        description="Explains concepts and provides research answers.",
        instructions=f"""You are a research agent.

Your previous outputs this session (do not repeat, only build on or update):
{prev_research_output or "None — this is your first response."}

Rules:
- Avoid repetition
- No code
- Be concise""",
    )


def create_code_agent(prev_code_output: str):
    return AzureOpenAIResponsesClient().create_agent(
        name="Code Agent",
        description="Writes and explains code.",
        instructions=f"""You are a code agent.

Your previous outputs this session (maintain consistency, update if needed):
{prev_code_output or "None — this is your first response."}

Rules:
- Keep consistency with your previous code
- Update existing logic if the task requires it
- Be concise""",
    )


def create_compare_agent(prev_compare_output: str):
    return AzureOpenAIResponsesClient().create_agent(
        name="Compare Agent",
        description="Compares technologies, approaches, or concepts.",
        instructions=f"""You are a comparison agent.

Your previous outputs this session (do not repeat, only build on or update):
{prev_compare_output or "None — this is your first response."}

Rules:
- No repetition
- Be structured and concise""",
    )


def create_merge_agent():
    return AzureOpenAIResponsesClient().create_agent(
        name="Merge Agent",
        description="Merges outputs from sub-agents into a final answer.",
        instructions="""You are a merge agent. You receive outputs from one or more sub-agents
and combine them into a single, coherent, well-structured response.

Rules:
- Do not repeat yourself
- Preserve all useful content
- Be clear and concise""",
    )


# ========================
# ENTRY POINT
# ========================

async def invoke_engineering_supervisor(
    user_message: str,
    session_id: str,
    conversation_history: str = "",
) -> dict:

    # ── Load turns & extract all agent memory ────────────────────────────
    turns       = load_turns(session_id)
    turn_number = len(turns) + 1

    prev_research_memory   = get_all_agent_memory(turns, "research_memory")
    prev_code_memory       = get_all_agent_memory(turns, "code_memory")
    prev_comparison_memory = get_all_agent_memory(turns, "comparison_memory")

    log(session_id, f"TURN {turn_number} — START", (
        f"User message         : {user_message}\n"
        f"Previous turns stored: {len(turns)}"
    ))

    # ── Step 1: Supervisor decides plan ──────────────────────────────────
    supervisor_prompt = f"Latest user message:\n{user_message}"

    log(session_id, f"TURN {turn_number} — SUPERVISOR INPUT", (
        f"History:\n{conversation_history or '(none)'}\n\n"
        f"Prompt:\n{supervisor_prompt}"
    ))

    supervisor_agent    = create_supervisor_agent(conversation_history)
    supervisor_response = await supervisor_agent.run(supervisor_prompt)

    try:
        raw = supervisor_response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        plan = json.loads(raw.strip())
    except Exception as e:
        log(session_id, f"TURN {turn_number} — SUPERVISOR PARSE ERROR", str(e))
        plan = {
            "research": {"needed": True,  "task": user_message},
            "code":     {"needed": False, "task": None},
            "compare":  {"needed": False, "task": None},
        }

    log(session_id, f"TURN {turn_number} — SUPERVISOR PLAN", json.dumps(plan, indent=2))

    # ── Step 2: Run needed agents in parallel ─────────────────────────────
    async def run_research():
        if not plan["research"]["needed"]:
            return None
        log(session_id, f"TURN {turn_number} — RESEARCH AGENT INPUT", (
            f"Task            : {plan['research']['task']}\n"
            f"Prev own output : {prev_research_memory or '(none)'}"
        ))
        agent = create_research_agent(prev_research_memory)
        resp  = await agent.run(plan["research"]["task"])
        log(session_id, f"TURN {turn_number} — RESEARCH AGENT OUTPUT", resp.text)
        return resp.text

    async def run_code():
        if not plan["code"]["needed"]:
            return None
        log(session_id, f"TURN {turn_number} — CODE AGENT INPUT", (
            f"Task            : {plan['code']['task']}\n"
            f"Prev own output : {prev_code_memory or '(none)'}"
        ))
        agent = create_code_agent(prev_code_memory)
        resp  = await agent.run(plan["code"]["task"])
        log(session_id, f"TURN {turn_number} — CODE AGENT OUTPUT", resp.text)
        return resp.text

    async def run_compare():
        if not plan["compare"]["needed"]:
            return None
        log(session_id, f"TURN {turn_number} — COMPARE AGENT INPUT", (
            f"Task            : {plan['compare']['task']}\n"
            f"Prev own output : {prev_comparison_memory or '(none)'}"
        ))
        agent = create_compare_agent(prev_comparison_memory)
        resp  = await agent.run(plan["compare"]["task"])
        log(session_id, f"TURN {turn_number} — COMPARE AGENT OUTPUT", resp.text)
        return resp.text

    research_result, code_result, compare_result = await asyncio.gather(
        run_research(),
        run_code(),
        run_compare(),
    )

    # ── Step 3: Merge ─────────────────────────────────────────────────────
    parts = []
    if research_result:
        parts.append(f" Explanation:\n{research_result}")
    if code_result:
        parts.append(f" Code:\n{code_result}")
    if compare_result:
        parts.append(f" Comparison:\n{compare_result}")

    if not parts:
        final_answer = "No results from any agent."
        log(session_id, f"TURN {turn_number} — MERGE", "No agent produced a result.")
    elif len(parts) == 1:
        final_answer = parts[0]
        log(session_id, f"TURN {turn_number} — MERGE", "Single agent — skipping merge step.")
    else:
        merge_input  = "\n\n".join(parts)
        merge_prompt = f"User question: {user_message}\n\nAgent outputs:\n{merge_input}"
        log(session_id, f"TURN {turn_number} — MERGE AGENT INPUT", merge_prompt)
        merge_agent    = create_merge_agent()
        merge_response = await merge_agent.run(merge_prompt)
        final_answer   = merge_response.text
        log(session_id, f"TURN {turn_number} — MERGE AGENT OUTPUT", final_answer)

    # ── Step 4: Build new turn object & append to list ────────────────────
    new_turn = {
        "turn":              turn_number,
        "timestamp":         datetime.now().isoformat(),
        "last_answer":       final_answer,
        "research_memory":   research_result or "",
        "code_memory":       code_result or "",
        "comparison_memory": compare_result or "",
    }

    turns.append(new_turn)
    save_turns(session_id, turns)

    log(session_id, f"TURN {turn_number} — SAVED TURN SNAPSHOT", json.dumps(new_turn, indent=2))
    log(session_id, f"TURN {turn_number} — END", f"Total turns in log: {len(turns)}")

    return {
        "answer": final_answer,
    }
