import asyncio
import os
from dotenv import load_dotenv

from agent_framework.azure import AzureOpenAIResponsesClient

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()



embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
    model="text-embedding-3-large"
)

vectorstore = FAISS.load_local(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "utility", "faiss_index")),
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})



def rag_search(query: str) -> str:
    results = retriever.invoke(query)

    if not results:
        return "No relevant information found."

    return "\n".join([doc.page_content for doc in results])



async def main():
    agent = AzureOpenAIResponsesClient().create_agent(
        name="RAG Agent",
        description="An agent that answers using a vector database.",
        instructions="""
You are a helpful assistant with access to a RAG tool.

RULES:
- ALWAYS call rag_search to answer user questions
- DO NOT answer from your own knowledge
- Use ONLY the retrieved context
- If nothing relevant is found, say: I don't know
- Return ONLY the tool result
""",
        tools=[rag_search],
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