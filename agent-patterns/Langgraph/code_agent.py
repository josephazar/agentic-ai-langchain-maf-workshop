import os
import re
import ast
import sys
import json
import importlib
import subprocess
import tempfile
import sqlite3
from dotenv import load_dotenv
from datetime import datetime
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

# ----------- setup -----------
# loads AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY from .env
load_dotenv()

# single shared LLM instance, swap model here if needed
llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
)

MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# ----------- sqlite checkpointer -----------
# persists state across restarts, check_same_thread=False for langgraph threading
sqlite_conn = sqlite3.connect("code_checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(sqlite_conn)

# ----------- logger -----------
# flat .txt file, append-only during a session
# init_log() wipes it at session start so old runs don't pile up
LOG_FILE = "code_supervisor_log.txt"

def _write(text: str):
    # internal only, everything goes through log_prompt()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def init_log():
    # call once at startup, overwrites the previous log
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"  CODE SUPERVISOR LOG  session started {ts}\n")
        f.write("=" * 80 + "\n")

def log_prompt(node: str, prompt: str):
    # logs the exact string the LLM sees, useful for debugging context drift
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _write(f"\n  +-- PROMPT SENT TO LLM [{node.upper()}]  ({ts})")
    _write("  |")
    for line in prompt.strip().splitlines():
        _write(f"  |  {line}")
    _write("  +" + "-" * 60)

# ----------- structured models -----------
# CodeBlock for generation, DebugResult for fixing — both strip markdown from output
class CodeBlock(BaseModel):
    code: str = Field(description="Complete runnable Python code. No markdown, no backticks.")
    explanation: str = Field(description="Brief explanation of what the code does.")

class DebugResult(BaseModel):
    fixed_code: str = Field(description="Corrected Python code. No markdown, no backticks.")
    explanation: str = Field(description="What was wrong and what was fixed.")
    gave_up: bool = Field(default=False, description="True only if the error is fundamentally unsolvable.")

code_llm  = llm.with_structured_output(CodeBlock)
debug_llm = llm.with_structured_output(DebugResult)

# ----------- state -----------
class ExecutionRecord(TypedDict):
    attempt: int
    code: str
    stdout: str
    stderr: str
    success: bool

class State(TypedDict):
    messages: Annotated[list, add_messages]
    raw_messages: Annotated[list, add_messages]
    current_code: str
    explanation: str
    user_request: str
    attempt: int
    execution_history: list[ExecutionRecord]
    final_output: str
    conversation_summary: str
    status: str
    missing_env_vars: list[str]
    human_feedback: str
    code_artifacts: dict

# ----------- pre-execution checks -----------
_STDLIB = sys.stdlib_module_names if hasattr(sys, "stdlib_module_names") else set()

# maps import names to their actual pip package names where they differ
_IMPORT_TO_PIP = {
    "cv2": "opencv-python", "PIL": "Pillow", "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4", "yaml": "pyyaml", "dotenv": "python-dotenv",
    "google.cloud": "google-cloud", "tensorflow": "tensorflow", "torch": "torch",
}

def extract_imports(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    packages = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                packages.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                packages.add(node.module.split(".")[0])
    return list(packages)

def get_recent_shared_history(messages: list, limit: int = 6) -> str:
    history_lines = []
    for msg in messages[-limit:]:
        if isinstance(msg, HumanMessage):
            history_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_lines.append(f"Assistant: {msg.content}")
    return "\n".join(history_lines)

def install_missing_packages(code: str) -> list[str]:
    # tries to import each package first, only installs if missing
    installed = []
    for pkg in extract_imports(code):
        if pkg in _STDLIB:
            continue
        try:
            importlib.import_module(pkg)
        except ImportError:
            pip_name = _IMPORT_TO_PIP.get(pkg, pkg)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                installed.append(pip_name)
    return installed

def check_missing_env_vars(code: str) -> list[str]:
    # scans code for os.getenv / os.environ calls and checks if they're set
    patterns = [
        r'os\.getenv\s*\(\s*["\'](\w+)["\']\s*\)',
        r'os\.environ\s*\[\s*["\'](\w+)["\']\s*\]',
        r'os\.environ\.get\s*\(\s*["\'](\w+)["\']\s*\)',
    ]
    missing = []
    for pattern in patterns:
        for key in re.findall(pattern, code):
            if not os.getenv(key) and key not in missing:
                missing.append(key)
    return missing

def get_last_user_message(messages: list) -> str:
    # walks backwards, safe even if AIMessages are mixed in
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def extract_code_from_message(text: str) -> Optional[str]:
    # handles both fenced code blocks and raw python signals
    if "```" in text:
        lines = text.split("\n")
        code_lines, inside = [], False
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines).strip()
    python_signals = ["def ", "import ", "print(", "class ", "for ", "while ", "if __name__"]
    if any(sig in text for sig in python_signals):
        return text.strip()
    return None

# ----------- node 1: supervisor -----------
def supervisor_node(state: State) -> dict:
    # if code already exists, skip regeneration — happens after HITL resume
    if state.get("current_code"):
        return {}

    user_message   = get_last_user_message(state["messages"])
    existing_code  = extract_code_from_message(user_message)
    full_conversation = "\n".join([
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
        for msg in state["messages"]
        if isinstance(msg, (HumanMessage, AIMessage))
    ])

    # user pasted code directly, no need to generate
    if existing_code:
        return {
            "current_code":      existing_code,
            "explanation":       "User-provided code.",
            "user_request":      user_message,
            "attempt":           0,
            "execution_history": [],
            "status":            "pending_approval",
            "human_feedback":    "",
        }

    # build context sections for the prompt
    existing_artifacts = state.get("code_artifacts", {})
    artifacts_context  = ""
    if existing_artifacts:
        artifacts_context = "\n\nPreviously generated code this session:\n"
        for name, code in existing_artifacts.items():
            artifacts_context += f"\n### {name}\n```\n{code}\n```\n"

    prior_summary   = state.get("conversation_summary", "")
    summary_context = f"\n\n\n{prior_summary}\n" if prior_summary else ""

    generate_prompt = f"""You are a Python code generator.
Task: {user_message}
Cross-agent context (from router):
{summary_context if summary_context else "None"}



Rules:
- Write complete, runnable Python code
- Standard library only unless explicitly required otherwise
- No markdown or backticks in the code field
- Add helpful print() statements
- Handle edge cases
- If prior code is relevant, build on it rather than starting from scratch"""

    log_prompt("supervisor", generate_prompt)

    result: CodeBlock = code_llm.invoke(generate_prompt)

    return {
        "current_code":      result.code,
        "explanation":       result.explanation,
        "user_request":      user_message,
        "attempt":           0,
        "execution_history": [],
        "status":            "pending_approval",
        "human_feedback":    "",
    }

# ----------- node 2: HITL approval -----------
def approval_node(state: State) -> dict:
    human_response: str = interrupt({
        "type": "code_approval",
        "code": state["current_code"],
        "message": "Respond with 'approve', 'cancel', or paste edited code.",
    })

    human_response = human_response.strip()

    if human_response.lower() == "cancel":
        return {
            "status":         "cancelled",
            "human_feedback": human_response,
            "final_output":   "Execution cancelled by user.",
        }

    if human_response.lower() == "approve":
        return {"status": "approved", "human_feedback": human_response}

    # user pasted edited code
    return {
        "current_code":   human_response,
        "status":         "approved",
        "human_feedback": human_response,
    }

def route_after_approval(state: State) -> str:
    return "finalizer" if state["status"] == "cancelled" else "executor"

# ----------- node 3: executor -----------
def executor_node(state: State) -> dict:
    attempt = state["attempt"] + 1
    code    = state["current_code"]

    missing_env = check_missing_env_vars(code)
    if missing_env:
        msg = (
            "Missing environment variable(s):\n"
            + "\n".join(f"  - {k}" for k in missing_env)
            + "\n\nAdd them to your .env file and try again."
        )
        return {
            "attempt":           attempt,
            "execution_history": list(state.get("execution_history", [])),
            "status":            "missing_env",
            "missing_env_vars":  missing_env,
            "final_output":      msg,
        }

    install_missing_packages(code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
        )
        stdout  = proc.stdout.strip()
        stderr  = proc.stderr.strip()
        success = proc.returncode == 0 and not stderr
    except subprocess.TimeoutExpired:
        stdout, stderr, success = "", f"TimeoutError: exceeded {TIMEOUT_SECONDS}s.", False
    finally:
        os.unlink(tmp_path)

    record = {
        "attempt": attempt, "code": code,
        "stdout": stdout, "stderr": stderr, "success": success,
    }

    history = list(state.get("execution_history", []))
    history.append(record)
    return {
        "attempt":           attempt,
        "execution_history": history,
        "missing_env_vars":  [],
        "status":            "success" if success else "running",
    }

# ----------- node 4: debugger -----------
def debugger_node(state: State) -> dict:
    history = state["execution_history"]
    last    = history[-1]

    history_text = "".join(
        f"\n--- Attempt {r['attempt']} ---\nCode:\n{r['code']}\n"
        + (f"Stdout:\n{r['stdout']}\n" if r["stdout"] else "")
        + f"Error:\n{r['stderr']}\n"
        for r in history
    )

    debug_prompt = (
        f"Fix this failing Python code. Study all attempts.\n{history_text}\n"
        "Return COMPLETE corrected code. No markdown. "
        "Set gave_up=true only if truly unfixable."
    )

    log_prompt("debugger", debug_prompt)

    result: DebugResult = debug_llm.invoke(debug_prompt)

    if result.gave_up:
        return {
            "current_code": result.fixed_code,
            "status":       "gave_up",
            "final_output": (
                f"Gave up after {last['attempt']} attempt(s).\n"
                f"{last['stderr']}\n{result.explanation}"
            ),
        }
    return {"current_code": result.fixed_code, "status": "running"}

# ----------- node 5: finalizer -----------
def finalizer_node(state: State) -> dict:
    status  = state["status"]
    history = state["execution_history"]

    if status in ("cancelled", "gave_up", "missing_env"):
        output = state.get("final_output", "Execution did not complete.")

    elif status == "success":
        last  = history[-1]
        n     = last["attempt"]
        lines = [f"Executed successfully on attempt {n}.\n"]
        if n > 1:
            lines.append(f"Took {n-1} debug cycle(s).\n")
        lines += ["```python", last["code"], "```"]
        if last["stdout"]:
            lines += ["\nOutput:", last["stdout"]]
        output = "\n".join(lines)

    else:
        last   = history[-1] if history else {}
        output = "\n".join([
            f"Could not fix after {MAX_RETRIES} attempt(s).",
            "Last error:", last.get("stderr", ""),
            "Last code:", "```python", last.get("code", ""), "```",
        ])

    # save successful code to artifacts keyed by timestamp
    existing_artifacts = state.get("code_artifacts", {})
    new_artifacts      = {}

    if status == "success" and history:
        last_code = history[-1]["code"]
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{ts}_final.py"
        new_artifacts[filename] = last_code

    merged_artifacts = {**existing_artifacts, **new_artifacts}

    # rolling summary so the supervisor has context on future requests
    user_request  = state.get("user_request", "Unknown request")
    explanation   = state.get("explanation", "")
    last_record   = history[-1] if history else {}
    attempts_made = last_record.get("attempt", 0)
    stdout        = last_record.get("stdout", "")
    stderr        = last_record.get("stderr", "")
    missing_vars  = state.get("missing_env_vars", [])

    if status == "success":
        outcome = f"Executed successfully on attempt {attempts_made}."
        if stdout:
            outcome += f" First output line: {stdout.splitlines()[0]}"
    elif status == "cancelled":
        outcome = "User cancelled before execution."
    elif status == "gave_up":
        outcome = f"Could not fix after {attempts_made} attempt(s). Last error: {stderr[:120]}"
    elif status == "missing_env":
        outcome = f"Blocked, missing environment variables: {', '.join(missing_vars)}"
    else:
        outcome = f"Failed after {attempts_made} attempt(s). Last error: {stderr[:120]}"

    prior_summary   = state.get("conversation_summary", "")
    new_entry       = f"User asked: {user_request}. Built: {explanation}. Outcome: {outcome}"
    updated_summary = f"{prior_summary}\n{new_entry}".strip() if prior_summary else new_entry

    new_ai_message = AIMessage(content=output)

    return {
        "messages":             [new_ai_message],
        "raw_messages":         [new_ai_message],
        "final_output":         output,
        "conversation_summary": updated_summary,
        "code_artifacts":       merged_artifacts,
    }

# ----------- routing -----------
def route_after_executor(state: State) -> str:
    if state["status"] in ("success", "missing_env"):
        return "finalizer"
    if state["attempt"] >= MAX_RETRIES:
        return "finalizer"
    return "debugger"

def route_after_debugger(state: State) -> str:
    return "finalizer" if state["status"] == "gave_up" else "executor"

# ----------- graph -----------
# supervisor -> approval -> executor -> debugger (loop) -> finalizer
def build_code_supervisor_graph():
    workflow = StateGraph(State)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("approval",   approval_node)
    workflow.add_node("executor",   executor_node)
    workflow.add_node("debugger",   debugger_node)
    workflow.add_node("finalizer",  finalizer_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "approval")
    workflow.add_conditional_edges(
        "approval", route_after_approval,
        {"executor": "executor", "finalizer": "finalizer"},
    )
    workflow.add_conditional_edges(
        "executor", route_after_executor,
        {"debugger": "debugger", "finalizer": "finalizer"},
    )
    workflow.add_conditional_edges(
        "debugger", route_after_debugger,
        {"executor": "executor", "finalizer": "finalizer"},
    )
    workflow.add_edge("finalizer", END)

    return workflow.compile(checkpointer=checkpointer)

code_graph = build_code_supervisor_graph()

# ----------- display helper -----------
def _print_approval_prompt(code: str):
    # shown to the user in terminal when HITL kicks in
    print("\n" + "-" * 60)
    print("CODE REVIEW REQUIRED")
    print("-" * 60)
    print(code)
    print("\n" + "-" * 60)
    print("  approve      -> run as-is")
    print("  cancel       -> abort")
    print("  (paste code) -> run your edited version")
    print("-" * 60)

# ----------- entry points -----------
def invoke_code_supervisor(
    user_message:         str,
    session_id:           str,
    conversation_summary: str = "",
) -> dict:
    config = {"configurable": {"thread_id": session_id}}

    code_graph.invoke(
        {
            "messages":             [HumanMessage(content=user_message)],
            "raw_messages":         [HumanMessage(content=user_message)],
            "current_code":         "",
            "explanation":          "",
            "user_request":         "",
            "attempt":              0,
            "execution_history":    [],
            "final_output":         "",
            "conversation_summary": conversation_summary,
            "status":               "running",
            "missing_env_vars":     [],
            "human_feedback":       "",
            "code_artifacts":       {},
        },
        config=config,
    )

    snapshot = code_graph.get_state(config)

    if snapshot.tasks:
        code = snapshot.values.get("current_code", "")
        _print_approval_prompt(code)
        return {"status": "pending_approval", "code": code, "session_id": session_id}

    values = snapshot.values
    return {
        "status":       values.get("status", "unknown"),
        "answer":       values.get("final_output", ""),
        "code_summary": values.get("conversation_summary", ""),
        "session_id":   session_id,
    }

def resume_code_supervisor(
    human_input:          str,
    session_id:           str,
    conversation_summary: str = "",   
) -> dict:
    config = {"configurable": {"thread_id": session_id}}

    result = code_graph.invoke(Command(resume=human_input), config=config)

    return {
        "answer":       result.get("final_output", ""),
        "status":       result.get("status", "unknown"),
        "code_summary": result.get("conversation_summary", ""),
    }