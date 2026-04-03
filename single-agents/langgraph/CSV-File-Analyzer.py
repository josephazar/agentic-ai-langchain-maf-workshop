"""
Single Agent: CSV File Analyzer — LangGraph Implementation
-----------------------------------------------------------
An agent that can load, query, and analyze CSV files using natural language.
It translates user questions into pandas operations, executes them, and returns insights.
Optionally generates matplotlib visualizations when appropriate.

Requirements:
    pip install langgraph langchain-openai langchain-core pandas matplotlib python-dotenv
"""

import os
import io
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI window)
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ── Global DataFrame store ─────────────────────────────────────────────────────
_dataframes: dict[str, pd.DataFrame] = {}


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def load_csv(filepath: str) -> str:
    """Load a CSV file into memory from the given filepath.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        str: Confirmation message with shape and column names.
    """
    global _dataframes
    try:
        df = pd.read_csv(filepath)
        _dataframes["current"] = df
        return (
            f"CSV loaded successfully.\n"
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns.\n"
            f"Columns: {', '.join(df.columns.tolist())}\n"
            f"Preview (first 3 rows):\n{df.head(3).to_string(index=False)}"
        )
    except Exception as e:
        return f"Error loading CSV: {str(e)}"


@tool
def query_csv(pandas_code: str) -> str:
    """Execute a pandas expression against the loaded CSV DataFrame.

    The DataFrame is available as the variable `df`.
    Use this for filtering, grouping, aggregation, sorting, etc.

    Args:
        pandas_code (str): A Python expression using `df`, e.g.:
            - df.groupby('region')['revenue'].mean()
            - df[df['age'] > 30].shape[0]
            - df.describe()

    Returns:
        str: The result of the expression as a string.
    """
    if "current" not in _dataframes:
        return "No CSV loaded yet. Please load a CSV file first using load_csv."
    df = _dataframes["current"]  # noqa: F841  (used in eval)
    try:
        result = eval(pandas_code, {"df": df, "pd": pd})  # noqa: S307
        if isinstance(result, pd.DataFrame):
            return result.to_string()
        elif isinstance(result, pd.Series):
            return result.to_string()
        else:
            return str(result)
    except Exception as e:
        return f"Error executing pandas code: {str(e)}"


@tool
def visualize_csv(pandas_plot_code: str, output_path: str = "chart.png") -> str:
    """Generate a matplotlib visualization from the loaded DataFrame and save it.

    The DataFrame is available as `df`. Write standard matplotlib/pandas plot code.
    Call plt.savefig(output_path) at the end — it is injected automatically.

    Args:
        pandas_plot_code (str): Python code that creates a plot using `df` and `plt`, e.g.:
            df.groupby('region')['revenue'].sum().plot(kind='bar')
            plt.title('Revenue by Region')
            plt.tight_layout()
        output_path (str): File path to save the chart (default: chart.png).

    Returns:
        str: Path to the saved chart or an error message.
    """
    if "current" not in _dataframes:
        return "No CSV loaded yet. Please load a CSV file first using load_csv."
    df = _dataframes["current"]  # noqa: F841
    try:
        plt.figure(figsize=(10, 6))
        exec(pandas_plot_code, {"df": df, "plt": plt, "pd": pd})  # noqa: S102
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return f"Chart saved to: {output_path}"
    except Exception as e:
        plt.close()
        return f"Error generating visualization: {str(e)}"


@tool
def get_csv_info() -> str:
    """Return metadata about the currently loaded CSV file.

    Returns:
        str: Column names, dtypes, shape, null counts, and basic stats.
    """
    if "current" not in _dataframes:
        return "No CSV loaded yet."
    df = _dataframes["current"]
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    return (
        f"Shape: {df.shape}\n\n"
        f"Info:\n{info_str}\n"
        f"Null counts:\n{df.isnull().sum().to_string()}\n\n"
        f"Numeric summary:\n{df.describe().to_string()}"
    )


# ── LangGraph State ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── LLM + Tool binding ─────────────────────────────────────────────────────────

tools = [load_csv, query_csv, visualize_csv, get_csv_info]

llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="gpt-5-mini",
)

llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a CSV data analyst assistant.

You have access to these tools:
- load_csv(filepath): Load a CSV file into memory
- get_csv_info(): Get metadata, dtypes, and statistics about the loaded CSV
- query_csv(pandas_code): Run any pandas expression against the DataFrame (variable: df)
- visualize_csv(pandas_plot_code, output_path): Generate and save a matplotlib chart

RULES:
- Always load the CSV first if not already loaded
- Translate natural language questions into precise pandas expressions
- Use query_csv for any data question (averages, filters, groupby, counts, etc.)
- Use visualize_csv when the user asks for a chart, plot, or graph
- Be concise and show the actual result, not just code
- If a query fails, try a corrected version automatically
"""


# ── Graph nodes ────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """The LLM reasoning node — decides whether to call tools or respond."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route to tools if the LLM made tool calls, otherwise end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Build graph ────────────────────────────────────────────────────────────────

tool_node = ToolNode(tools)

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")  # After tools → back to agent to reason on results

graph = graph_builder.compile()


# ── Main REPL ──────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  CSV File Analyzer — LangGraph Agent")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    state: AgentState = {"messages": []}

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = graph.invoke(state)

        # Print the last AI message
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\nAssistant: {msg.content}\n")
                break


if __name__ == "__main__":
    run()