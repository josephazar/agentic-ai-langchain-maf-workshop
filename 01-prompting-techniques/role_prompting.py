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

# Role Prompting — assigning the model a specific persona via the system message
PROMPTS_PATH = Path(__file__).with_name("prompts") / "role_prompting.json"
PROMPTS = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))

for role in PROMPTS["roles"]:
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": role["system"]},
            {"role": "user", "content": role["user"]},
        ],
    )
    print(role["label"])
    print(response.choices[0].message.content)
    
print("OBSERVATION:")
print("""
- The Developer role provides structured, technical answers with best practices.
- The Data Analyst focuses on insights and practical interpretation.
- The Teacher simplifies concepts using analogies and easy language.

Conclusion:
Changing the system role significantly affects tone, depth, and explanation style.
""")