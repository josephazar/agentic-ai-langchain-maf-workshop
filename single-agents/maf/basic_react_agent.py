import asyncio
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

load_dotenv()


# TOOLS


def simple_search(query: str) -> str:
    query = query.lower()
    if "ai" in query:
        return "AI stands for Artificial Intelligence."
    elif "python" in query:
        return "Python is used for AI, web, and automation."
    elif "langgraph" in query:
        return "LangGraph is a framework for building AI agents."
    return "No results found."


def get_weather(location: str = "San Francisco") -> str:
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."



async def main():
    agent = OpenAIChatClient().as_agent(
        name="My Azure Agent",
        description="An agent with tools.",
        instructions="""
You are a helpful assistant with access to tools.
RULES:
- If the user asks about weather → ALWAYS call get_weather
- If the user asks about AI, Python, or LangGraph → ALWAYS call simple_search
- DO NOT answer from your own knowledge
- Return ONLY the tool result
""",
        tools=[get_weather, simple_search],
    )

    print("Agent is running... (type 'exit' to quit)\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = await agent.run(user_input)
        print("Assistant:", response.text)
        
if __name__ == "__main__":
    asyncio.run(main())