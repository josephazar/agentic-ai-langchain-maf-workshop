"""
LLM-as-Judge: sends captured output to Azure OpenAI and gets a pass/fail verdict.
"""

import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

JUDGE_PROMPT = """\
You are an automated test evaluator for an educational AI workshop.

Your job is to determine whether a Python script or Jupyter notebook executed \
successfully and produced output consistent with its intended purpose.

## Task Description
{task_description}

## Execution Metadata
- Exit code: {exit_code}
- Status: {status}
- Duration: {duration_s}s
- Stdout length: {raw_stdout_len} chars

## Captured Output (last 2000 chars of stdout)
```
{stdout}
```

## Captured Errors (last 1000 chars of stderr)
```
{stderr}
```

## Evaluation Criteria
1. Did the script start and run without crashing (traceback in output = fail)?
2. Does the output show evidence that the LLM was called and responded?
3. Does the output match what the task description says it should do?
4. Ignore warnings in stderr (deprecation, FutureWarning, etc.) -- only count \
actual Python tracebacks or fatal errors.
5. For agents: receiving an LLM response and then exiting cleanly counts as pass, \
even if exit code is 1 due to EOFError after stdin is exhausted.
6. For notebooks: successful conversion (exit code 0) counts as pass.
7. For smoke tests: starting up and exiting without a traceback counts as pass.

## Response Format
Return ONLY a valid JSON object, nothing else:
{{"pass": true, "score": 8, "reasoning": "The script ran and produced expected output."}}
"""


def _get_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )


def judge_result(result: dict, task_description: str) -> dict:
    """
    Send execution result to the LLM judge and return the verdict.

    Returns:
        {"pass": bool | None, "score": int, "reasoning": str}
    """
    # -- Skip judging for skipped tests --
    if result.get("status") == "skipped":
        return {
            "pass": None,
            "score": 0,
            "reasoning": f"Skipped: {result.get('skip_reason', 'unknown')}",
        }

    prompt = JUDGE_PROMPT.format(
        task_description=task_description,
        exit_code=result.get("exit_code", "N/A"),
        status=result.get("status", "unknown"),
        duration_s=result.get("duration_s", 0),
        raw_stdout_len=result.get("raw_stdout_len", 0),
        stdout=(result.get("stdout", "") or "")[-2000:],
        stderr=(result.get("stderr", "") or "")[-1000:],
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown fences)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            verdict = json.loads(raw[start:end])
            return {
                "pass": bool(verdict.get("pass", False)),
                "score": int(verdict.get("score", 5)),
                "reasoning": str(verdict.get("reasoning", "No reasoning provided")),
            }

    except Exception as e:
        pass  # Fall through to heuristic

    # -- Fallback heuristic if LLM judge fails --
    has_output = len(result.get("stdout", "")) > 50
    no_crash = result.get("exit_code") in (0, 1, None)
    has_traceback = "Traceback" in result.get("stderr", "")

    if has_output and no_crash and not has_traceback:
        return {"pass": True, "score": 5, "reasoning": "Heuristic: produced output without traceback."}
    else:
        return {"pass": False, "score": 2, "reasoning": "Heuristic: missing output or traceback detected."}
