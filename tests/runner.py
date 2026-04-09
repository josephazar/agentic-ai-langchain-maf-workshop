"""
Subprocess execution engine: runs each test file and captures output.
"""

import os
import sys
import time
import socket
import subprocess
from dotenv import dotenv_values

from test_config import REPO_ROOT, PRE_CHECKS


# ---------------------------------------------------------------------------
# MCP Server lifecycle
# ---------------------------------------------------------------------------
_mcp_server_proc = None


def start_mcp_server():
    """Start the MCP server in the background. Returns True if started."""
    global _mcp_server_proc
    if _mcp_server_proc is not None:
        return True  # already running

    env = _build_env()
    mcp_path = os.path.join(REPO_ROOT, "utility", "mcp-server.py")
    if not os.path.exists(mcp_path):
        return False

    _mcp_server_proc = subprocess.Popen(
        [sys.executable, mcp_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    # Wait for it to be ready (up to 10s)
    for _ in range(20):
        time.sleep(0.5)
        try:
            s = socket.create_connection(("127.0.0.1", 8787), timeout=1)
            s.close()
            return True
        except Exception:
            continue
    return False


def stop_mcp_server():
    """Stop the MCP server if running."""
    global _mcp_server_proc
    if _mcp_server_proc is not None:
        _mcp_server_proc.terminate()
        try:
            _mcp_server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _mcp_server_proc.kill()
        _mcp_server_proc = None


def _build_env():
    """Build env dict: merge current env with .env file values."""
    env = os.environ.copy()
    dot_values = dotenv_values(os.path.join(REPO_ROOT, ".env"))
    env.update(dot_values)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_test(entry: dict) -> dict:
    """
    Execute a single test entry and return the result dict.

    Returns:
        {
            "file_path", "file_name", "category", "exit_code",
            "stdout", "stderr", "duration_s", "status",
            "raw_stdout_len", "raw_stderr_len",
            "skip_reason"  (only if skipped)
        }
    """
    file_path = entry["file_path"]
    file_name = os.path.basename(file_path)
    category = entry["category"]
    timeout = entry.get("timeout", 120)
    cwd = entry.get("cwd") or os.path.dirname(file_path)
    test_input = entry.get("test_input")

    base_result = {
        "file_path": file_path,
        "file_name": file_name,
        "category": category,
    }

    # -- Check skip_reason --
    if entry.get("skip_reason"):
        return {
            **base_result,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "duration_s": 0,
            "status": "skipped",
            "raw_stdout_len": 0,
            "raw_stderr_len": 0,
            "skip_reason": entry["skip_reason"],
        }

    # -- Run pre-check --
    pre_check_name = entry.get("pre_check")
    if pre_check_name and pre_check_name in PRE_CHECKS:
        ok, reason = PRE_CHECKS[pre_check_name]()
        if not ok:
            return {
                **base_result,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "duration_s": 0,
                "status": "skipped",
                "raw_stdout_len": 0,
                "raw_stderr_len": 0,
                "skip_reason": reason,
            }

    # -- Start MCP server if needed --
    needs_mcp = entry.get("needs_mcp", False)
    if needs_mcp:
        if not start_mcp_server():
            return {
                **base_result,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "duration_s": 0,
                "status": "skipped",
                "raw_stdout_len": 0,
                "raw_stderr_len": 0,
                "skip_reason": "Could not start MCP server",
            }

    env = _build_env()
    start = time.time()

    try:
        if category == "E":
            # Notebook execution via jupyter nbconvert
            cmd = [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                f"--ExecutePreprocessor.timeout={timeout}",
                "--ExecutePreprocessor.kernel_name=venv",
                file_path,
                "--output", f"/tmp/_test_{file_name}",
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 60,
                cwd=cwd,
                env=env,
            )
        else:
            # Python script execution
            cmd = [sys.executable, file_path]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                input=test_input,
                timeout=timeout,
                cwd=cwd,
                env=env,
            )

        duration = time.time() - start
        raw_stdout = proc.stdout or ""
        raw_stderr = proc.stderr or ""

        return {
            **base_result,
            "exit_code": proc.returncode,
            "stdout": raw_stdout[-3000:],
            "stderr": raw_stderr[-1500:],
            "duration_s": round(duration, 1),
            "status": "completed",
            "raw_stdout_len": len(raw_stdout),
            "raw_stderr_len": len(raw_stderr),
        }

    except subprocess.TimeoutExpired as e:
        duration = time.time() - start
        return {
            **base_result,
            "exit_code": None,
            "stdout": (e.stdout or "")[-3000:] if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "")[-1500:] if isinstance(e.stderr, str) else "",
            "duration_s": round(duration, 1),
            "status": "timeout",
            "raw_stdout_len": len(e.stdout or ""),
            "raw_stderr_len": len(e.stderr or ""),
        }

    except Exception as e:
        duration = time.time() - start
        return {
            **base_result,
            "exit_code": None,
            "stdout": "",
            "stderr": str(e),
            "duration_s": round(duration, 1),
            "status": "error",
            "raw_stdout_len": 0,
            "raw_stderr_len": 0,
        }
