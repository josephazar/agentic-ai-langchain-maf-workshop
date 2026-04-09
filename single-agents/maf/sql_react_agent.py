import asyncio
import os
import sqlite3
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "utility", "chinook.db")
) 


# ============================
# TOOLS
# ============================

def get_schema(_: str = "") -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema = {}

        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            schema[table_name] = columns

        conn.close()
        return str(schema)

    except Exception as e:
        return f"Schema Error: {str(e)}"


def query_sqlite(query: str) -> str:
    try:
        if any(x in query.lower() for x in ["drop", "delete", "update"]):
            return "Dangerous query blocked."

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(query)
        rows = cursor.fetchall()

        conn.close()

        return str(rows)

    except Exception as e:
        return f"SQL Error: {str(e)}"


# ============================
# MAIN
# ============================

async def main():
    agent = OpenAIChatClient().as_agent(
        name="SQLite Agent",
        description="Agent that queries SQLite database",
        instructions="""
You are a SQLite database assistant.

TOOLS:
- get_schema → understand database structure
- query_sqlite → execute SQL

RULES:
- ALWAYS call get_schema first if unsure
- Translate question to SQL
- Then call query_sqlite
- Return ONLY results
- Do NOT explain unless asked
""",
        tools=[get_schema, query_sqlite],
    )

    print("SQLite Agent running... (type 'exit' to quit)\n")

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = await agent.run(user_input)
        print("Assistant:", response.text)


if __name__ == "__main__":
    asyncio.run(main())