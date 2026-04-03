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
PROMPTS_PATH = Path(__file__).with_name("prompts") / "tree_of_thought.json"
PROMPTS = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


# PROBLEM — Planning puzzle
# You have a 4L jug and a 3L jug, no markings.
# Goal: measure exactly 2L of water.
# Good ToT problem because there are multiple branching paths
# and some lead to dead ends that need pruning.

puzzle = PROMPTS["puzzle"]


# STEP 1 — GENERATE BRANCHES
# Ask the model to explore multiple distinct approaches

branch_prompt = PROMPTS["branch_prompt_template"].format(puzzle=puzzle)

branch_response = client.chat.completions.create(
    model="gpt-4.1",
    temperature=0.7,
    messages=[{"role": "user", "content": branch_prompt}]
)
branches = branch_response.choices[0].message.content


# STEP 2 — EVALUATE BRANCHES
# Ask the model to score each branch and prune bad ones

evaluate_prompt = PROMPTS["evaluate_prompt_template"].format(
    puzzle=puzzle,
    branches=branches,
)

evaluate_response = client.chat.completions.create(
    model="gpt-4.1",
    temperature=0,
    messages=[{"role": "user", "content": evaluate_prompt}]
)
evaluation = evaluate_response.choices[0].message.content


# STEP 3 — SOLVE BEST BRANCH
# Take the most promising branch and solve it fully

solve_prompt = PROMPTS["solve_prompt_template"].format(
    puzzle=puzzle,
    branches=branches,
    evaluation=evaluation,
)

solve_response = client.chat.completions.create(
    model="gpt-4.1",
    temperature=0,
    messages=[{"role": "user", "content": solve_prompt}]
)
solution = solve_response.choices[0].message.content


# OUTPUT


print("STEP 1 — GENERATE BRANCHES")
print("(exploring multiple reasoning paths)")

print(branches)

print("STEP 2 — EVALUATE & PRUNE")
print("(score each branch, discard dead ends)")

print(evaluation)


print("STEP 3 — SOLVE BEST BRANCH")
print("(fully solve the most promising path)")

print(solution)

# =============================================================
# OBSERVATION
# ToT mimics deliberate planning by:
# 1. Generating alternatives (branching)
# 2. Evaluating and pruning bad paths
# 3. Committing to and solving the best path
# Unlike CoT which follows one linear chain,
# ToT explores the solution space before committing
# =============================================================