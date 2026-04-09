from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json
import re
from pathlib import Path

load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)
PROMPTS_PATH = Path(__file__).with_name("prompts") / "react_prompting.json"
PROMPTS = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


# AVAILABLE TOOLS (actions the model can take)
# In a real agent these could be web search, DB queries, APIs

def calculate(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def convert_currency(amount, from_currency, to_currency):
    # Hardcoded rates for demo purposes
    rates = {"USD_EUR": 0.92, "EUR_USD": 1.09, "USD_GBP": 0.79, "GBP_USD": 1.27}
    key = f"{from_currency}_{to_currency}"
    if key in rates:
        return str(round(amount * rates[key], 2))
    return "Unknown currency pair"


# SYSTEM PROMPT — teaches the model the ReAct format
# Model must respond in Thought/Action/Input format only

system_prompt = PROMPTS["system_prompt"]

# PROBLEM

user_question = PROMPTS["user_question"]


# ReAct LOOP
# Each iteration: model thinks + acts → we observe → feed back

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_question}
]


print("REACT LOOP")
print(f"Question: {user_question}")


MAX_STEPS = 6

for step in range(MAX_STEPS):
    
    # MODEL THINKS AND DECIDES AN ACTION
    
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0,
        messages=messages
    )
    model_output = response.choices[0].message.content
    messages.append({"role": "assistant", "content": model_output})

    print(f"\n--- Step {step + 1} ---")
    print(model_output)

   
    # CHECK IF MODEL REACHED FINAL ANSWER → STOP LOOP
    
    if "Final Answer:" in model_output:
        print("AGENT FINISHED")
        break

   
    # PARSE ACTION AND INPUT FROM MODEL OUTPUT
    
    action_match = re.search(r"Action:\s*(\w+)", model_output)
    input_match = re.search(r"Input:\s*([^\n]+)", model_output)

    if not action_match or not input_match:
        print("Could not parse action — stopping loop.")
        break

    action = action_match.group(1).strip()
    action_input = input_match.group(1).strip()

  
    # EXECUTE THE ACTION (this is the "Acting" part of ReAct)
    #
    if action == "calculate":
        observation = calculate(action_input)
    elif action == "convert_currency":
        parts = [p.strip() for p in action_input.split(",")]
        observation = convert_currency(float(parts[0]), parts[1], parts[2])
    else:
        observation = f"Unknown action: {action}"

    print(f"Observation: {observation}")

    
    # FEED OBSERVATION BACK TO MODEL FOR NEXT THOUGHT
    
    messages.append({"role": "user", "content": f"Observation: {observation}"})

# =============================================================
# OBSERVATION
# This is the foundation of agentic AI:
# The model doesn't just answer — it reasons, acts, observes,
# and loops until it reaches a final answer.
# Thought  → model decides what to do
# Action   → code executes the tool
# Observation → result fed back to model
# This pattern scales to web search, APIs, databases, etc.
# =============================================================