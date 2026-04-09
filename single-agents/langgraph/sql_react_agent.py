import os
import sqlite3
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

# =========================================================
# MEMORY
# =========================================================
memory = MemorySaver()

# =========================================================
# STATE
# =========================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]

# =========================================================
# DATABASE CONFIG
# =========================================================
DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "utility", "chinook.db")
)  # absolute path so it works no matter where you run the script from

# =========================================================
# TOOLS
# =========================================================

@tool
def get_schema(_: str = ""):
    """Get database schema (tables and columns)."""
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


@tool
def query_sqlite(query: str):
    """Execute SQL query on SQLite database."""
    try:
        # 🚫 basic protection
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


tools = [get_schema, query_sqlite]
tool_node = ToolNode(tools)

# =========================================================
# LLM
# =========================================================
llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
)

llm_with_tools = llm.bind_tools(tools)

# =========================================================
# AGENT NODE
# =========================================================
def agent(state: State):
    system_message = SystemMessage(content=(
        "You are a SQLite database assistant.\n\n"

        "TOOLS:\n"
        "1. get_schema → understand database\n"
        "2. query_sqlite → execute SQL\n\n"

        "RULES:\n"
        "- ALWAYS call get_schema first if unsure\n"
        "- Convert user question into SQL\n"
        "- Then call query_sqlite\n"
        "- Return ONLY the result\n"
        "- Do NOT explain unless asked\n"
    ))

    messages = [system_message] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

# =========================================================
# ROUTER
# =========================================================
def tool_router(state: State):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END

# =========================================================
# GRAPH
# =========================================================
builder = StateGraph(State)

builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tool_router, ["tools", END])
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=memory)

# =========================================================
# STREAM FUNCTION
# =========================================================
def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [("user", user_input)]},
        {"configurable": {"thread_id": "1"}},
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# =========================================================
# CLI LOOP
# =========================================================
if __name__ == "__main__":
    print("SQLite LangGraph Agent running...")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("User: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)

        except Exception as e:
            import traceback
            traceback.print_exc()
            break