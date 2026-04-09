import asyncio
import ast
import os
import re
import sys
import json
import importlib
import subprocess
import tempfile
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_framework.openai import OpenAIChatClient
from memory_store import load_session, save_session

load_dotenv()

CODE_MEMORY_FILE = "code_memory.json"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# ========================
# LOGGER
# ========================
LOG_FILE = "code_agent_log.txt"

def _write(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def init_log():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"  CODE AGENT LOG  session started {ts}\n")
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
# PRE-EXECUTION CHECKS
# ========================

_STDLIB = sys.stdlib_module_names if hasattr(sys, "stdlib_module_names") else set()

_IMPORT_TO_PIP = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "google.cloud": "google-cloud",
    "tensorflow": "tensorflow",
    "torch": "torch",
}


def extract_imports(code: str) -> list:
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


def install_missing_packages(code: str):
    for pkg in extract_imports(code):
        if pkg in _STDLIB:
            continue

        try:
            importlib.import_module(pkg)
        except ImportError:
            pip_name = _IMPORT_TO_PIP.get(pkg, pkg)
            print(f"  Installing: {pip_name}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name],
                capture_output=True,
                text=True,
            )


def check_missing_env_vars(code: str) -> list:
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


def extract_code_from_message(text: str):
    if "```" in text:
        lines = text.split("\n")
        code_lines = []
        inside = False

        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue

            if inside:
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

    python_signals = [
        "def ",
        "import ",
        "print(",
        "class ",
        "for ",
        "while ",
        "if __name__",
    ]

    if any(sig in text for sig in python_signals):
        return text.strip()

    return None


# ========================
# EXECUTOR
# ========================

def execute_code(code: str) -> dict:
    missing_env = check_missing_env_vars(code)

    if missing_env:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Missing env vars: {', '.join(missing_env)}",
            "missing_env": missing_env,
        }

    install_missing_packages(code)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        success = proc.returncode == 0 and not stderr

    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = f"TimeoutError: exceeded {TIMEOUT_SECONDS}s."
        success = False

    finally:
        os.unlink(tmp_path)

    return {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "missing_env": [],
    }


# ========================
# AGENT FACTORIES
# ========================

def create_generator_agent(history_text: str):
    history_context = (
        f"\n\nFull conversation history:\n{history_text}\n"
        if history_text else ""
    )

    instructions = f"""You are a Python code generator.
{history_context}

Rules:
- Write complete, runnable Python code
- Standard library only unless explicitly required
- No markdown or backticks — return raw code only
- Add helpful print() statements
- Handle edge cases
- Use the full conversation history to understand follow-up questions
- If the user says things like "fix that", "continue", "modify it", or "why is it failing",
  use the previous conversation history to infer what "that" refers to
"""

    log_prompt("generator_instructions", instructions)

    return OpenAIChatClient().as_agent(
        name="Code Generator",
        description="Generates complete runnable Python code.",
        instructions=instructions,
    )


def create_debugger_agent(conversation_history: str = ""):
    history_context = (
        f"\n\nConversation history for context:\n{conversation_history}\n"
        if conversation_history else ""
    )

    instructions = f"""You are a Python debugger.
{history_context}
You will receive failed execution attempts with code and error messages.

Return ONLY valid JSON in this exact format:
{{
  "fixed_code": "complete corrected python code here",
  "explanation": "what was wrong and what was fixed",
  "gave_up": false
}}
"""

    log_prompt("debugger_instructions", instructions)

    return OpenAIChatClient().as_agent(
        name="Code Debugger",
        description="Fixes failing Python code.",
        instructions=instructions,
    )


# ========================
# MAIN ENTRY POINT
# ========================

async def invoke_code_supervisor(
    user_message: str,
    session_id: str,
    conversation_history: str = "",
) -> dict:

    log_section("invoke_code_supervisor", f"user_message: {user_message}\nsession_id: {session_id}")

    existing_code = extract_code_from_message(user_message)

    if existing_code:
        log_section("generator_skipped", "User provided code directly — skipping generation.")
        return {
            "status": "pending_approval",
            "code": existing_code,
            "session_id": session_id,
        }

    generator = create_generator_agent(history_text=conversation_history)

    log_section("generator_user_message", user_message)

    response = await generator.run(user_message)
    generated_code = response.text.strip()

    log_section("generator_response", generated_code)

    if "```" in generated_code:
        generated_code = extract_code_from_message(generated_code) or generated_code

    return {
        "status": "pending_approval",
        "code": generated_code,
        "session_id": session_id,
    }


async def resume_code_supervisor(
    user_message: str,
    human_input: str,
    code: str,
    session_id: str,
    conversation_history: str = "",
) -> dict:
    log_section("resume_code_supervisor", f"human_input: {human_input}\nsession_id: {session_id}")

    mem = load_session(CODE_MEMORY_FILE, session_id)
    existing_turns = mem.get("turns", [])

    if human_input.strip().lower() == "cancel":
        return {
            "status": "cancelled",
            "answer": "Execution cancelled by user.",
            "session_id": session_id,
        }

    final_decision = human_input.strip()

    if human_input.strip().lower() != "approve":
        code = human_input.strip()
        print("  Using edited code.")

    current_code = code
    attempt = 0
    execution_history = []

    while attempt < MAX_RETRIES:
        attempt += 1

        log_section(f"executor_attempt_{attempt}", current_code)

        result = execute_code(current_code)

        log_section(f"executor_result_attempt_{attempt}",
            f"success: {result['success']}\nstdout: {result['stdout']}\nstderr: {result['stderr']}"
        )

        execution_history.append({
            "attempt": attempt,
            "code": current_code,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "success": result["success"],
        })

        if result["missing_env"]:
            msg = (
                "Missing environment variable(s):\n"
                + "\n".join(f"  - {k}" for k in result["missing_env"])
                + "\n\nAdd them to your .env file and try again."
            )
            return {
                "status": "missing_env",
                "answer": msg,
                "session_id": session_id,
            }

        if result["success"]:
            break

        if attempt >= MAX_RETRIES:
            break

        history_text = "".join(
            f"\n--- Attempt {r['attempt']} ---\n"
            f"Code:\n{r['code']}\n"
            + (f"Stdout:\n{r['stdout']}\n" if r["stdout"] else "")
            + f"Error:\n{r['stderr']}\n"
            for r in execution_history
        )

        debugger_prompt = f"""Fix this failing Python code.

Study all attempts carefully.

{history_text}
"""

        log_prompt(f"debugger_prompt_attempt_{attempt}", debugger_prompt)

        debugger = create_debugger_agent(conversation_history)
        debug_response = await debugger.run(debugger_prompt)

        log_section(f"debugger_response_attempt_{attempt}", debug_response.text)

        try:
            raw = debug_response.text.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            debug_result = json.loads(raw.strip())

        except Exception:
            debug_result = {
                "fixed_code": debug_response.text.strip(),
                "gave_up": False,
            }

        if debug_result.get("gave_up"):
            last = execution_history[-1]
            return {
                "status": "gave_up",
                "answer": f"Gave up after {attempt} attempt(s).\n{last['stderr']}",
                "session_id": session_id,
            }

        current_code = debug_result["fixed_code"]

    last = execution_history[-1]

    if last["success"]:
        lines = [f"Executed successfully on attempt {attempt}.\n"]

        if attempt > 1:
            lines.append(f"Took {attempt - 1} debug cycle(s).\n")

        lines += ["```python", last["code"], "```"]

        if last["stdout"]:
            lines += ["\nOutput:", last["stdout"]]

        final_output = "\n".join(lines)
        status = "success"

    else:
        final_output = "\n".join([
            f"Could not fix after {MAX_RETRIES} attempt(s).",
            "Last error:",
            last["stderr"],
            "Last code:",
            "```python",
            last["code"],
            "```",
        ])
        status = "failed"

    log_section("final_output", final_output)

    timestamp = datetime.now().isoformat()

    new_turn = {
        "turn": len(existing_turns) + 1,
        "timestamp": timestamp,
        "user_message": user_message,
        "generated_code": last["code"],
        "decision": final_decision,
        "status": status,
        "stdout": last["stdout"],
        "stderr": last["stderr"],
        "attempts": attempt,
    }

    existing_turns.append(new_turn)

    save_session(CODE_MEMORY_FILE, session_id, {
        "turns": existing_turns,
    })

    return {
        "status": status,
        "answer": final_output,
        "session_id": session_id,
    }