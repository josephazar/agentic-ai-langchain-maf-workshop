"""
Single Agent: CSV File Analyzer — MAF Implementation
-----------------------------------------------------
An agent that can load, query, and analyze CSV files using natural language.
It translates user questions into pandas operations, executes them, and returns insights.
Optionally generates matplotlib visualizations when appropriate.

Requirements:
    pip install agent-framework pandas matplotlib python-dotenv
"""

import os
import io
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI window)
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from agent_framework.azure import AzureOpenAIResponsesClient

load_dotenv()

# ── Global DataFrame store ─────────────────────────────────────────────────────
_dataframes: dict[str, pd.DataFrame] = {}


# ── Tools ──────────────────────────────────────────────────────────────────────

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


def get_csv_info() -> str:
    """Return metadata about the currently loaded CSV file including
    column names, dtypes, shape, null counts, and basic statistics.

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


def query_csv(pandas_code: str) -> str:
    """Execute a pandas expression against the loaded CSV DataFrame.

    The DataFrame is available as the variable `df`.
    Use this for filtering, grouping, aggregation, sorting, etc.

    Args:
        pandas_code (str): A Python/pandas expression using `df`, e.g.:
            df.groupby('region')['revenue'].mean()
            df[df['age'] > 30].shape[0]
            df.describe()

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


def visualize_csv(pandas_plot_code: str, output_path: str = "chart.png") -> str:
    """Generate a matplotlib visualization from the loaded DataFrame and save it.

    The DataFrame is available as `df`. Write standard matplotlib/pandas plot code.

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


# ── Agent setup ────────────────────────────────────────────────────────────────

tools = [load_csv, get_csv_info, query_csv, visualize_csv]

agent = AzureOpenAIResponsesClient().create_agent(
    name="CSV Analyzer Agent",
    description="An agent that loads, queries, and visualizes CSV files using natural language.",
    instructions="""You are a CSV data analyst assistant.

TOOLS:
- load_csv(filepath): Load a CSV file into memory
- get_csv_info(): Get metadata, dtypes, and statistics
- query_csv(pandas_code): Run pandas expressions against df
- visualize_csv(pandas_plot_code, output_path): Save a chart

RULES:
- At the start of EVERY message, call get_csv_info() first to check if a CSV is loaded
- If get_csv_info() returns "No CSV loaded yet", ask the user for a filepath then load it
- If data IS loaded, proceed directly with the user's question
- NEVER ask clarifying questions — make reasonable assumptions and proceed
- Translate questions into pandas and show actual results, not code
- If a query fails, try a corrected version automatically
""",

    tools=tools,
)


# ── Main REPL ──────────────────────────────────────────────────────────────────

async def run():
    import asyncio
    print("=" * 60)
    print("  CSV File Analyzer — MAF Agent")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        response = await agent.run(user_input)
        print(f"\nAssistant: {response.text}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())