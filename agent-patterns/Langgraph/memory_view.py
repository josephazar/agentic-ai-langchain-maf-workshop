"""
memory_viewer.py
─────────────────────────────────────────────────────────────────────────────
Run this from your Langgraph/ folder to inspect what each agent has stored
in memory for every session.

Usage:
    python memory_viewer.py               # show all agents, all sessions
    python memory_viewer.py chat          # show only chat agent
    python memory_viewer.py code abc-123  # show code agent, specific session
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import sqlite3
import json

# map agent name -> (db file, which state fields to display)
AGENTS = {
    "chat": {
        "db":     "chat_checkpoints.db",
        "fields": ["chat_history", "final_answer"],
    },
    "code": {
        "db":     "code_checkpoints.db",
        "fields": ["conversation_summary", "code_artifacts", "status", "final_output"],
    },
    "engineering": {
        "db":     "engineering_checkpoints.db",
        "fields": ["conversation_summary", "research_memory", "code_memory", "comparison_memory"],
    },
    "knowledge": {
        "db":     "knowledge_checkpoints.db",
        "fields": ["conversation_summary", "final_answer", "plan"],
    },
    "router": {
        "db":     "router_checkpoints.db",
        "fields": ["router_memory", "active_supervisor", "final_answer"],
    },
}

SEP  = "=" * 70
SEP2 = "-" * 70


def decode_checkpoint(blob: bytes) -> dict:
    """Try to decode a msgpack checkpoint blob into a readable dict."""
    try:
        import msgpack
        raw = msgpack.unpackb(blob, raw=False, strict_map_key=False)

        # langgraph wraps state under a 'channel_values' key
        if isinstance(raw, dict):
            return raw.get("channel_values", raw)
        return raw
    except Exception:
        pass

    # fallback: try JSON (older langgraph versions)
    try:
        return json.loads(blob.decode("utf-8", errors="replace"))
    except Exception:
        return {"_raw": repr(blob[:120]) + "..."}


def get_sessions(db_path: str) -> list[str]:
    """Return all unique thread_ids (session IDs) in a checkpoint DB."""
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:
        return []


def get_latest_checkpoint(db_path: str, thread_id: str) -> dict:
    """Return the decoded state for the most recent checkpoint of a session."""
    try:
        conn = sqlite3.connect(db_path)

        # langgraph checkpoint schema: thread_id, checkpoint_id, checkpoint (blob)
        row = conn.execute(
            """
            SELECT checkpoint FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """,
            (thread_id,),
        ).fetchone()
        conn.close()

        if not row:
            return {}

        blob = row[0]
        if isinstance(blob, (bytes, bytearray)):
            return decode_checkpoint(blob)
        if isinstance(blob, str):
            try:
                return json.loads(blob)
            except Exception:
                return {"_raw": blob[:200]}
        return {}

    except sqlite3.OperationalError as e:
        return {"_error": str(e)}


def print_field(key: str, value):
    """Pretty-print a single state field."""
    print(f"\n  {key}:")

    if value is None or value == "" or value == [] or value == {}:
        print("     (empty)")
        return

    if isinstance(value, str):
        lines = value.strip().splitlines()
        for line in lines[:40]:
            print(f"     {line}")
        if len(lines) > 40:
            print(f"     ... ({len(lines) - 40} more lines)")

    elif isinstance(value, list):
        print(f"     [{len(value)} item(s)]")
        for i, item in enumerate(value[:5]):
            if isinstance(item, dict):
                snippet = json.dumps(item, indent=6, default=str)
                for line in snippet.splitlines()[:10]:
                    print(f"     {line}")
            else:
                preview = str(item)[:200]
                print(f"     [{i}] {preview}")
        if len(value) > 5:
            print(f"     ... ({len(value) - 5} more items)")

    elif isinstance(value, dict):
        snippet = json.dumps(value, indent=4, default=str)
        for line in snippet.splitlines()[:30]:
            print(f"     {line}")
        lines = snippet.splitlines()
        if len(lines) > 30:
            print(f"     ... ({len(lines) - 30} more lines)")

    else:
        print(f"     {value}")


def show_agent(agent_name: str, filter_session: str = None):
    info    = AGENTS[agent_name]
    db_path = info["db"]
    fields  = info["fields"]

    print(f"\n{SEP}")
    print(f"  AGENT: {agent_name.upper()}   (db: {db_path})")
    print(SEP)

    sessions = get_sessions(db_path)

    if not sessions:
        print("  No sessions found (DB may not exist yet or no runs completed).")
        return

    if filter_session:
        sessions = [s for s in sessions if filter_session in s]
        if not sessions:
            print(f"  No sessions matching '{filter_session}'.")
            return

    for session_id in sessions:
        print(f"\n  {SEP2}")
        print(f"  SESSION: {session_id}")
        print(f"  {SEP2}")

        state = get_latest_checkpoint(db_path, session_id)

        if not state:
            print("  (no state data found)")
            continue

        if "_error" in state:
            print(f"  Error reading state: {state['_error']}")
            continue

        if "_raw" in state:
            print("  Could not decode checkpoint. Raw preview:")
            print(f"     {state['_raw']}")
            continue

        found_any = False
        for field in fields:
            if field in state:
                print_field(field, state[field])
                found_any = True

        if not found_any:
            print("\n  State exists but none of the tracked fields were found.")
            print("  Available keys:", list(state.keys()))


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args           = sys.argv[1:]
    agent_filter   = args[0].lower() if len(args) >= 1 else None
    session_filter = args[1]         if len(args) >= 2 else None

    if agent_filter and agent_filter not in AGENTS:
        print(f"Unknown agent '{agent_filter}'. Choose from: {', '.join(AGENTS)}")
        sys.exit(1)

    agents_to_show = [agent_filter] if agent_filter else list(AGENTS.keys())

    for agent in agents_to_show:
        show_agent(agent, session_filter)

    print(f"\n{SEP}\n")