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

PROMPTS_PATH = Path(__file__).with_name("prompts") / "zero_shot.json"
PROMPTS_CONFIG = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
prompts = [(task["name"], task["prompt"]) for task in PROMPTS_CONFIG["tasks"]]

for task_name, prompt in prompts:
    print(f"Task: {task_name}")
    print(f"Prompt: {prompt}")

    
    response_mini = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\nGPT-5 Mini:\n{response_mini.choices[0].message.content}")

    
    response_41 = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\nGPT-4.1:\n{response_41.choices[0].message.content}")


# ------------------------------------------------------------------------------
# My actual takeaways from testing zero-shot prompts:
#
# Zero-shot prompting works well for classification, summarization, and
# extraction because these tasks have clear, unambiguous instructions and
# simple output formats. Both GPT-5 Mini and GPT-4.1 performed equally well
# across all three tasks without needing any examples. The model doesn't need
# to explain or reason through anything complex; it just needs to follow a
# direct instruction.
#
#
# For clear stuff like classification, summarizing, or simple extraction, you usually don’t need to give the model examples. Just describe what you want and it’s fine, both GPT-5 Mini and GPT-4.1 did well on all those tasks, straight to the point and without needing hints.
#
# But it gets shaky fast if things aren’t straightforward:
#
# 1. If you need super consistent output (like always JSON, or always exactly one specific word), zero-shot kinda falls apart. Each model words it differently, so if you need to automatically parse the results, it’ll break your workflow unless you give a few examples of the exact format.
#
# 2. If you have business-specific rules (e.g. “Churn Risk” only applies if a contract expires within 90 days), zero-shot will usually miss the nuance and go by the generic meaning. Examples help a ton with these exceptions.
#
# 3. If you ask for a lot at once (classify + explain + suggest next step, all in one go), you’ll get mixed or incomplete responses. The model has no idea what to prioritize. Split it up, or show with examples what you want for each part.
#
# 4. If your input is kind of ambiguous (could go in two categories), zero-shot will just pick something with no explanation and isn’t consistent run to run. Examples showing tricky/edge cases make a big difference.
#
# 5. If you want it to always sound a certain way (super formal, casual, company style…), zero-shot isn’t reliable — it’ll drift in tone. System prompts or examples that match your desired vibe help lock it in.
#
# Bottom line: Zero-shot is great for stuff that’s simple, clear-cut, and doesn’t need consistency beyond “do the task.” If you need specific formatting, special rules, multi-step responses, to handle ambiguity, or to control the tone, don’t be lazy — throw in a few examples instead.
# ------------------------------------------------------------------------------