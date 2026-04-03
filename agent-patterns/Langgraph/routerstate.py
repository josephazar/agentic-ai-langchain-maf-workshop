import sqlite3
import msgpack
from pprint import pprint
from langchain_core.messages import messages_from_dict

DB_PATH = "router_checkpoints.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get latest checkpoint per session/thread
cursor.execute("""
WITH latest AS (
    SELECT thread_id, MAX(rowid) as max_rowid
    FROM checkpoints
    GROUP BY thread_id
)
SELECT c.thread_id, c.checkpoint
FROM checkpoints c
JOIN latest l
    ON c.thread_id = l.thread_id
   AND c.rowid = l.max_rowid
ORDER BY c.rowid DESC
""")

rows = cursor.fetchall()

print("\n" + "=" * 120)
print("ROUTER SESSION VIEW")
print("=" * 120)

for idx, row in enumerate(rows):
    thread_id = row[0]
    checkpoint_blob = row[1]

    try:
        checkpoint_data = msgpack.unpackb(checkpoint_blob, raw=False)
        channel_values = checkpoint_data.get("channel_values", {})

        print("\n" + "=" * 120)
        print(f"SESSION #{idx + 1}")
        print("=" * 120)
        print(f"Thread ID           : {thread_id}")
        print(f"Session ID          : {channel_values.get('session_id', '')}")
        print(f"Active Supervisor   : {channel_values.get('active_supervisor', '')}")
        print(f"Routing Reason      : {channel_values.get('routing_reason', '')}")
        print(f"Pending Approval    : {channel_values.get('pending_code_approval', False)}")
        print(f"Router Memory       : {channel_values.get('router_memory', '')}")
        print("\nFINAL ANSWER:")
        print("-" * 120)
        print(channel_values.get("final_answer", ""))

        print("\nCONVERSATION HISTORY:")
        print("-" * 120)

        raw_messages = channel_values.get("messages", [])

        for msg_idx, raw_msg in enumerate(raw_messages):
            try:
                # Decode LangChain ExtType message
                if isinstance(raw_msg, msgpack.ExtType):
                    unpacked = msgpack.unpackb(raw_msg.data, raw=False)

                    # unpacked looks like:
                    # [module_path, class_name, data_dict, method_name]
                    msg_data = unpacked[2]

                    role = msg_data.get("type", "unknown").upper()
                    content = msg_data.get("content", "")

                    print(f"\n[{msg_idx + 1}] {role}")
                    print(content)

            except Exception as e:
                print(f"\n[{msg_idx + 1}] Could not decode message")
                print(str(e))

    except Exception as e:
        print(f"\nFailed to decode session {thread_id}")
        print(str(e))

conn.close()