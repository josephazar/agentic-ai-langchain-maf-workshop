"""
ReAct Agent with Code Interpreter (LangGraph)
- Agent writes and executes Python code at runtime
- Sandboxed via subprocess with timeout
- Test with math, data transformations, and plotting
"""

import os
import sys
import subprocess
import tempfile
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ── Sandboxed Code Execution Tool ─────────────────────────────────────────────

@tool
def execute_python(code: str) -> str:
    """
    Write and execute Python code to solve problems.
    Use this for: math calculations, data analysis, data transformations,
    generating/saving plots, or any logic that requires computation.
    Always use this tool instead of computing answers manually.
    For plots, save them to a file using matplotlib savefig().
    """
    # Write code to a temp file and run it in a subprocess (sandboxed)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        # Auto-inject matplotlib non-interactive backend for plotting
        preamble = "import matplotlib\nmatplotlib.use('Agg')\n"
        f.write(preamble + code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=30,          # kill if it runs longer than 15 seconds
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            return f"❌ Error:\n{stderr}"

        output = stdout if stdout else "✅ Code executed successfully (no output)."

        # Check if a plot was saved
        if "savefig" in code:
            output += "\n📊 Plot saved. Check the working directory for the image file."

        return output

    except subprocess.TimeoutExpired:
        return " Execution timed out (>15 seconds). Simplify the code."
    finally:
        os.unlink(tmp_path)   # clean up temp file

# ── LLM + Tools ───────────────────────────────────────────────────────────────

tools = [execute_python]
tool_node = ToolNode(tools)

llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
)
llm_with_tools = llm.bind_tools(tools)

# ── Agent Node ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a Python code interpreter agent. You have one tool: execute_python.\n\n"
    "RULES:\n"
    "1. ALWAYS write and execute Python code to solve problems — never compute manually.\n"
    "2. For plots/charts: use matplotlib and always call plt.savefig('output.png') instead of plt.show().\n"
    "3. For data analysis: use pandas or numpy as needed.\n"
    "4. If code fails, read the error, fix the code, and try again.\n"
    "5. After execution, explain the result to the user in plain language.\n"
))

def agent(state: State):
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_router(state: State):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END

# ── Graph ─────────────────────────────────────────────────────────────────────

memory = MemorySaver()

builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tool_router, ["tools", END])
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=memory)

# ── CLI ───────────────────────────────────────────────────────────────────────

def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [("user", user_input)]},
        {"configurable": {"thread_id": "1"}},
    ):
        for value in event.values():
            last_msg = value["messages"][-1]
            if last_msg.content:
                print("Assistant:", last_msg.content)

if __name__ == "__main__":
    print("ReAct Agent with Code Interpreter")
    print("Examples:")
    print("  - What is 123 * 456 + sqrt(789)?")
    print("  - Plot a sine wave and save it")
    print("  - Given [10, 20, 30, 40], compute mean, median, std")
    print("  Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except Exception:
            import traceback
            traceback.print_exc()
            break