import json
import os
from typing import Any

MEMORY_DIR = "memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

def _path(filename: str) -> str:
    return os.path.join(MEMORY_DIR, filename)

def load_session(filename: str, session_id: str) -> dict:
    """Load one session's data from a JSON file. Returns {} if not found."""
    fp = _path(filename)
    if not os.path.exists(fp):
        return {}
    with open(fp, "r", encoding="utf-8") as f:
        all_sessions = json.load(f)
    return all_sessions.get(session_id, {})

def save_session(filename: str, session_id: str, data: dict):
    """Merge updated data for one session into the JSON file."""
    fp = _path(filename)
    all_sessions = {}
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            all_sessions = json.load(f)
    all_sessions[session_id] = data
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, indent=2)

def list_sessions(filename: str) -> list[str]:
    """Return all session IDs stored in a file."""
    fp = _path(filename)
    if not os.path.exists(fp):
        return []
    with open(fp, "r", encoding="utf-8") as f:
        return list(json.load(f).keys())