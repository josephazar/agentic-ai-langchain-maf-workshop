# 01 — Prompting Techniques

A collection of prompting technique demos built on **Azure OpenAI**.  
Each script is self-contained, loads its prompts from a matching JSON file in `prompts/`, and prints results directly to the terminal.



## Scripts

### `zero_shot.py`
Sends prompts to the model with **no examples** — just a plain instruction.  
Runs the same tasks on both `gpt-5-mini` and `gpt-4.1` so you can compare outputs side by side.  
Tasks cover classification, summarization, and extraction.

> **Key takeaway:** Works great for simple, clear-cut tasks. Falls short when you need consistent formatting, business-specific rules, or handling edge cases — that's when you need few-shot.

---

### `few_shot.py`
Demonstrates **in-context learning** by embedding 2–5 examples directly in the prompt.  
Includes an interactive menu to run 6 different tests:
- Fallacy detection (zero-shot vs few-shot)
- Text classification (standard vs diversified examples)
- Confidence scoring (zero-shot vs few-shot)

> **Key takeaway:** Examples lock in output format and teach the model business-specific rules that a plain instruction can't communicate.

---

### `role_prompting.py`
Assigns the model **different personas** via the system message — Developer, Data Analyst, and Teacher — and asks each the same question.  
Prints all three responses so you can see how the role changes tone, depth, and structure.

> **Key takeaway:** The same question gets a technical breakdown, a data-driven interpretation, or a beginner-friendly analogy depending solely on the role assigned.

---

### `chain_prompting.py`
Breaks a complex task into a **sequence of simpler prompts** where each output feeds the next.  
The task: article → summary → key points → email draft.  
Offers two modes via interactive menu:
- **Manual chain** — raw Azure OpenAI API, step by step
- **LangChain LCEL chain** — `prompt | llm | parser` pipeline

> **Key takeaway:** Chaining keeps each step focused and makes the pipeline reusable. The LangChain version does the same work in a fraction of the code.

---

### `chain_of_thought.py`
Compares **direct prompting** vs adding *"Let's think step by step"* to the same question.  
Runs two problems: a logic puzzle and a billing complaint calculation.  
Also shows how zero-shot CoT vs few-shot CoT affects output structure.

> **Key takeaway:** CoT improves accuracy on reasoning tasks. Few-shot CoT additionally enforces a consistent step-by-step structure the model mirrors from the examples.

---

### `self_consistency.py`
Runs the **same prompt 5 times** at `temperature=0.7` and uses majority voting to pick the most reliable answer.  
Tests two problems: a billing verdict (categorical) and a factory revenue calculation (numerical).  
Compares a single deterministic run (`temperature=0`) against the voted result.

> **Key takeaway:** On simple problems all runs agree. On multi-step math, rounding differences cause divergence — majority voting filters out one-off mistakes and improves reliability.

---

### `tree_of_thought.py`
Implements a **3-step ToT loop** with separate API calls for each phase:
1. **Branch** — generate multiple distinct solution approaches
2. **Evaluate** — score each branch and prune dead ends
3. **Solve** — fully work through the most promising path

Uses a water jug puzzle as the demo problem (good for ToT because multiple paths exist and some lead to dead ends).

> **Key takeaway:** Unlike CoT which follows one linear chain, ToT explores the solution space before committing — better for problems where the right approach isn't obvious upfront.

---

### `react_prompting.py`
Implements a full **ReAct loop** (Reason → Act → Observe) from scratch.  
The model outputs `Thought / Action / Input`, the code executes the tool, and the observation is fed back — repeated up to 6 steps or until `Final Answer:` is reached.  
Available tools: `calculate` (math) and `convert_currency` (hardcoded rates).

> **Key takeaway:** This is the foundation of agentic AI. The model doesn't just answer — it reasons, acts, and observes in a loop. Each step is a separate API call. This pattern scales directly to web search, databases, and any external API.

---

### `structured_output.py`
Forces the model to respond in **valid JSON** with automatic retry on failure.  
Includes a `parse_json_response()` helper that handles 3 edge cases (markdown fences, text before/after JSON, malformed JSON) and a `prompt_with_retry()` function that feeds parsing errors back to the model and asks it to self-correct.

> **Key takeaway:** Reliable JSON output requires defensive parsing + a retry loop. The self-correction pattern (feed the error back) is the same principle as ReAct — observe the failure and correct it.

---

### `prompt_templates.py`
Side-by-side comparison of a **raw API call** vs a **LangChain LCEL pipeline** for the same sentiment analysis task.  
Shows how `ChatPromptTemplate | AzureChatOpenAI | JsonOutputParser` replaces manual message construction.  
Runs the same chain on two different reviews to demonstrate reusability.

> **Key takeaway:** Templates decouple prompt structure from input data. The same pipeline runs on any input with a single `.invoke()` call — no copy-pasting message dicts.

---

## Prompt Configs

Each script reads from a matching JSON file:

| Script | Prompt file |
|---|---|
| `zero_shot.py` | `prompts/zero_shot.json` |
| `few_shot.py` | `prompts/few_shot.json` |
| `role_prompting.py` | `prompts/role_prompting.json` |
| `chain_prompting.py` | `prompts/chain_prompting.json` |
| `chain_of_thought.py` | `prompts/chain_of_thought.json` |
| `self_consistency.py` | `prompts/self_consistency.json` |
| `tree_of_thought.py` | `prompts/tree_of_thought.json` |
| `react_prompting.py` | `prompts/react_prompting.json` |
| `structured_output.py` | `prompts/structured_output.json` |
| `prompt_templates.py` | `prompts/prompt_templates.json` |

