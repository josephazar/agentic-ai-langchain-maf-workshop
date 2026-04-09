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
PROMPTS_PATH = Path(__file__).with_name("prompts") / "chain_of_thought.json"
PROMPTS = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


question_logic = PROMPTS["question_logic"]

response_direct = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
    messages=[{"role": "user", "content": question_logic}]
)

response_cot = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
    messages=[{"role": "user", "content": question_logic + " Let's think step by step."}]
)


print("PROBLEM 1 — LOGIC PUZZLE")


print("\n--- Direct Prompting ---")
print(response_direct.choices[0].message.content)

print("\n--- Chain-of-Thought Prompting ---")
print(response_cot.choices[0].message.content)


billing_complaint = PROMPTS["billing_complaint"]


zero_shot_prompt = PROMPTS["zero_shot_template"].format(billing_complaint=billing_complaint)


few_shot_prompt = PROMPTS["few_shot_template"].format(billing_complaint=billing_complaint)

response_zero_shot = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
    messages=[{"role": "user", "content": zero_shot_prompt}]
)

response_few_shot = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
    messages=[{"role": "user", "content": few_shot_prompt}]
)


print("PROBLEM 2 — BILLING COMPLAINT")


print("\n--- Zero-Shot CoT (no example) ---")
print(response_zero_shot.choices[0].message.content)

print("\n--- Few-Shot CoT (with example) ---")
print(response_few_shot.choices[0].message.content)

# =============================================================
# OBSERVATION
# Zero-shot CoT: model reasons freely, may vary in format
# Few-shot CoT: model mirrors the Step 1/2/3/4 structure
# Correct answer: $80x3 + $25 = $265, billed $275 → overcharged by $10
# =============================================================