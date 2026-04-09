from openai import AzureOpenAI
from dotenv import load_dotenv
from collections import Counter
import os
import json
from pathlib import Path

load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)
PROMPTS_PATH = Path(__file__).with_name("prompts") / "self_consistency.json"
PROMPTS = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


billing_question = PROMPTS["billing_question"]


factory_question = PROMPTS["factory_question"]


single_billing = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0,
    messages=[{"role": "user", "content": billing_question}]
).choices[0].message.content

single_factory = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0,
    messages=[{"role": "user", "content": factory_question}]
).choices[0].message.content


NUM_RUNS = 5


print("PROBLEM 1 RUNS — BILLING COMPLAINT")

billing_responses = []
for i in range(NUM_RUNS):
    answer = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0.7,
        messages=[{"role": "user", "content": billing_question}]
    ).choices[0].message.content
    billing_responses.append(answer)
    print(f"Run {i+1}: {answer}\n{'-' * 40}")


print("PROBLEM 2 RUNS — FACTORY REVENUE")

factory_responses = []
for i in range(NUM_RUNS):
    answer = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0.7,
        messages=[{"role": "user", "content": factory_question}]
    ).choices[0].message.content
    factory_responses.append(answer)
    print(f"Run {i+1}: {answer}\n{'-' * 40}")


def extract_verdict(text):
    for line in reversed(text.splitlines()):
        line = line.lower().strip()
        if "correctly" in line:
            return "correctly charged"
        if "overcharged" in line:
            return "overcharged"
        if "undercharged" in line:
            return "undercharged"
    return "unknown"

def extract_final_number(text):
    import re
    numbers = re.findall(r'\$[\d,]+(?:\.\d+)?|\b\d+(?:\.\d+)?\b', text)
    return numbers[-1] if numbers else "unknown"

billing_votes = [extract_verdict(r) for r in billing_responses]
factory_votes = [extract_final_number(r) for r in factory_responses]

billing_majority = Counter(billing_votes).most_common(1)[0][0]
factory_majority = Counter(factory_votes).most_common(1)[0][0]



print("PROBLEM 1 — BILLING COMPLAINT")

print("\n--- Single CoT (temperature=0) ---")
print(single_billing)
print(f"\nVerdict: {extract_verdict(single_billing)}")
print(f"\nVotes:         {billing_votes}")
print(f"Vote counts:   {dict(Counter(billing_votes))}")
print(f"Majority vote: {billing_majority}")



print("PROBLEM 2 — FACTORY REVENUE")

print("\n--- Single CoT (temperature=0) ---")
print(single_factory)
print(f"\nExtracted answer: {extract_final_number(single_factory)}")
print(f"\nVotes:         {factory_votes}")
print(f"Vote counts:   {dict(Counter(factory_votes))}")
print(f"Majority vote: {factory_majority}")

# =============================================================
# OBSERVATION
# Problem 1: simple enough that all runs agree
# Problem 2: rounding at steps 4+5 causes diverging answers
# Self-consistency majority voting filters out one-off mistakes
# Correct answers: Problem 1 → correctly charged | Problem 2 → $42
# =============================================================