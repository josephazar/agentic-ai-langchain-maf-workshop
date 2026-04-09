"""
MAF ReAct Agent with Code Interpreter
- Agent writes and executes Python code at runtime
- Sandboxed via subprocess with timeout
- Test with math, data transformations, and plotting
"""

import asyncio
import os
import sys
import subprocess
import tempfile

from dotenv import load_dotenv
load_dotenv()

from agent_framework.openai import OpenAIChatClient

# ── Sandboxed Code Execution Tool ─────────────────────────────────────────────

def execute_python(code: str) -> str:
    """
    Write and execute Python code to solve problems.
    Use this for: math calculations, data analysis, data transformations,
    generating/saving plots, or any logic that requires computation.
    Always use this tool instead of computing answers manually.
    For plots, save them to a file using matplotlib savefig().
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        preamble = "import matplotlib\nmatplotlib.use('Agg')\n"
        f.write(preamble + code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            return f" Error:\n{stderr}"

        output = stdout if stdout else " Code executed successfully (no output)."

        if "savefig" in code:
            output += "\n Plot saved. Check the working directory for the image file."

        return output

    except subprocess.TimeoutExpired:
        return " Execution timed out (>15 seconds). Simplify the code."
    finally:
        os.unlink(tmp_path)

# ── MAF Agent ─────────────────────────────────────────────────────────────────

async def main():
    agent = OpenAIChatClient().as_agent(
        name="Code Interpreter Agent",
        description="An agent that writes and executes Python code to solve problems.",
        instructions=(
            "You are a Python code interpreter agent. You have one tool: execute_python.\n\n"
            "RULES:\n"
            "1. ALWAYS write and execute Python code to solve problems — never compute manually.\n"
            "2. For plots/charts: use matplotlib and always call plt.savefig('output.png') instead of plt.show().\n"
            "3. For data analysis: use pandas or numpy as needed.\n"
            "4. If code fails, read the error, fix the code, and try again.\n"
            "5. After execution, explain the result to the user in plain language.\n"
        ),
        tools=[execute_python],
    )

    print("MAF ReAct Agent with Code Interpreter")
    print("Examples:")
    print("  - What is 123 * 456 + sqrt(789)?")
    print("  - Plot a sine wave and save it")
    print("  - Given [10, 20, 30, 40], compute mean, median, std")
    print("  Type 'quit' to exit\n")

    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        response = await agent.run(user_input)
        print("Assistant:", response.text)

if __name__ == "__main__":
    asyncio.run(main())