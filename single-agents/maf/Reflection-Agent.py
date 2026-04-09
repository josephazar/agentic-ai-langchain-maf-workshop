import asyncio
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

load_dotenv()

client = OpenAIChatClient()

# =========================================================
# AGENTS
# =========================================================

generator_agent = client.as_agent(
    name="Essay Generator",
    description="Writes and revises 5-paragraph essays.",
    instructions="""You are an essay assistant tasked with writing excellent 5-paragraph essays.
Generate the best essay possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempts.
Return ONLY the essay, no preamble.""",
)

reflector_agent = client.as_agent(
    name="Essay Reflector",
    description="Critiques essay submissions.",
    instructions="""You are a teacher grading an essay submission.
Generate critique and recommendations for the submitted essay.
Provide detailed recommendations, including requests for length, depth, style, etc.
Return ONLY the critique, no preamble.""",
)

# =========================================================
# REFLECTION LOOP
# =========================================================

async def run_reflection_agent(user_input: str, max_loops: int = 3):
    print("\n[Starting reflection loop...]\n")

    # Step 1: Initial essay generation
    print("=" * 60)
    print("GENERATE (Round 1)")
    print("=" * 60)
    essay_response = await generator_agent.run(user_input)
    essay = essay_response.text
    print(essay)
    print()

    for i in range(2, max_loops + 1):
        # Step 2: Reflect on the essay
        print("=" * 60)
        print(f"REFLECT (critique for round {i})")
        print("=" * 60)
        critique_response = await reflector_agent.run(essay)
        critique = critique_response.text
        print(critique)
        print()

        # Step 3: Revise the essay based on critique
        print("=" * 60)
        print(f"GENERATE (Round {i})")
        print("=" * 60)
        revision_prompt = f"""Original request: {user_input}

Your previous essay:
{essay}

Critique received:
{critique}

Please write a revised and improved version of the essay."""
        essay_response = await generator_agent.run(revision_prompt)
        essay = essay_response.text
        print(essay)
        print()

    print("=" * 60)
    print("FINAL ESSAY")
    print("=" * 60)
    print(essay)
    return essay


# =========================================================
# CLI LOOP
# =========================================================

async def main():
    print("=" * 60)
    print("  Reflection Agent — MAF (AzureOpenAI)")
    print("=" * 60)
    print("The agent will generate an essay, critique it, and")
    print("refine it 3 times before giving the final result.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        await run_reflection_agent(user_input)


if __name__ == "__main__":
    asyncio.run(main())