import asyncio
from agent_framework.azure import AzureOpenAIResponsesClient
from dotenv import load_dotenv
import os

from tavily import TavilyClient

load_dotenv()


tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str) -> str:
    result = tavily.search(query=query, max_results=3)
    print("\n[TOOL CALLED]")
    print("Query:", query)
    # format results nicely
    formatted = []
    for r in result["results"]:
        formatted.append(f"{r['title']}\n{r['url']}\n{r['content']}")
    
    return "\n\n".join(formatted)




async def main():
    agent = AzureOpenAIResponsesClient().create_agent(
        name="My Azure Agent",
        description="An agent with Tavily web search.",
        instructions="""
You are a helpful assistant with access to a web search tool.

RULES:
- ALWAYS use the web_search tool for any question that requires information
- DO NOT answer from your own knowledge
- Return a clear and concise answer based on the tool result
""",
        tools=[web_search],
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