from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json
from pathlib import Path

load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)
PROMPTS_PATH = Path(__file__).with_name("prompts") / "structured_output.json"
PROMPTS = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


# HELPER — parse JSON from model response
# Handles edge cases: ```json wrapping, text before/after JSON

def parse_json_response(text):
    # edge case 1: model wraps in ```json ... ```
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    # edge case 2: model adds text before or after the JSON
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None, "No JSON object found in response"

    try:
        return json.loads(text[start:end]), None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


# HELPER — run a prompt with automatic retry on bad JSON
# If model breaks format, feeds the error back and asks it to fix
# Same pattern as ReAct: observe the error → correct it

def prompt_with_retry(messages, max_retries=3):
    conversation = messages.copy()

    for attempt in range(1, max_retries + 1):
        print(f"  Attempt {attempt}...")

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=conversation
        )
        raw = response.choices[0].message.content
        parsed, error = parse_json_response(raw)

        # success — return immediately
        if parsed:
            return raw, parsed, None

        # failure — feed error back and ask model to fix itself
        print(f"  Failed: {error} — asking model to fix...")
        conversation.append({"role": "assistant", "content": raw})
        conversation.append({
            "role": "user",
            "content": f"""Your response could not be parsed as JSON.
Error: {error}
Your response was: {raw}

Return ONLY a valid JSON object. No markdown, no explanation, just raw JSON."""
        })

    return raw, None, f"Failed after {max_retries} attempts"


# TASK 1 — Extract person info as JSON

print("=" * 60)
print("TASK 1 — PERSON INFO EXTRACTION")
print("=" * 60)

raw1, parsed1, error1 = prompt_with_retry([
    {"role": "user", "content": PROMPTS["task_1_prompt"]}
])

print(f"\nRaw response:\n{raw1}")
if error1:
    print(f"\nFailed to parse: {error1}")
else:
    print(f"\nParsed successfully:")
    print(f"  Name: {parsed1.get('name')}")
    print(f"  Age:  {parsed1.get('age')}")
    print(f"  City: {parsed1.get('city')}")


# TASK 2 — Classify product review as JSON

print("\n" + "=" * 60)
print("TASK 2 — PRODUCT REVIEW ANALYSIS")
print("=" * 60)

raw2, parsed2, error2 = prompt_with_retry([
    {"role": "user", "content": PROMPTS["task_2_prompt"]}
])

print(f"\nRaw response:\n{raw2}")
if error2:
    print(f"\nFailed to parse: {error2}")
else:
    print(f"\nParsed successfully:")
    print(f"  Sentiment:  {parsed2.get('sentiment')}")
    print(f"  Score:      {parsed2.get('score')}")
    print(f"  Key issues: {parsed2.get('key_issues')}")


# OBSERVATION
# parse_json_response handles 3 edge cases:
#   1. Model wraps in ```json ... ```
#   2. Model adds text before/after JSON
#   3. Model returns invalid/malformed JSON
# prompt_with_retry feeds the error back to the model
# and asks it to self-correct — same pattern as ReAct
