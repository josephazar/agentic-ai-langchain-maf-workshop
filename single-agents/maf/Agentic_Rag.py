import os
import asyncio
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel

from agent_framework.azure import AzureOpenAIResponsesClient

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# =========================
# VECTOR STORE SETUP
# =========================
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

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# =========================
# STRUCTURED OUTPUT MODELS
# =========================
class RouteDecision(BaseModel):
    action: Literal["retrieve", "answer"]
    reason: str


class GradeDecision(BaseModel):
    binary_score: Literal["yes", "no"]


# =========================
# TOOL FUNCTIONS
# =========================
def retrieve_docs(query: str) -> str:
    docs = retriever.invoke(query)
    if not docs:
        return "NO_RESULTS"

    return "\n\n".join(doc.page_content for doc in docs)


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


# =========================
# MAIN AGENTIC RAG LOGIC
# =========================
async def route_question(agent, question: str) -> RouteDecision:
    prompt = f"""
Decide whether to retrieve or answer.

Rules:
- If the user asks about a document, article, blog, uploaded knowledge, or specific author/source → retrieve
- If the answer is general and does not require retrieval → answer
- If unsure → retrieve

Question: {question}

Return JSON:
{{
  "action": "retrieve" or "answer",
  "reason": "short reason"
}}
"""
    response = await agent.run(prompt)
    return RouteDecision.model_validate_json(response.text)


async def grade_documents(agent, question: str, context: str) -> GradeDecision:
    prompt = f"""
Decide if the retrieved context is relevant enough to answer the user's question.

Question:
{question}

Context:
{context}

Return JSON:
{{
  "binary_score": "yes" or "no"
}}
"""
    response = await agent.run(prompt)
    return GradeDecision.model_validate_json(response.text)


async def rewrite_question(agent, question: str) -> str:
    prompt = f"""
Rewrite this user question to improve vector database retrieval.
Keep the same meaning, but make it clearer and more searchable.

Question:
{question}

Return only the rewritten question text.
"""
    response = await agent.run(prompt)
    return response.text.strip()


async def generate_answer(agent, question: str, context: str) -> str:
    prompt = f"""
Answer the user's question using the provided context.

Rules:
- Use the context as your main source
- If the context is insufficient, say so clearly
- Be concise and accurate

Question:
{question}

Context:
{context}
"""
    response = await agent.run(prompt)
    return response.text.strip()


async def direct_answer(agent, question: str) -> str:
    response = await agent.run(question)
    return response.text.strip()


# =========================
# ORCHESTRATOR
# =========================
async def agentic_rag_loop(agent, user_input: str, max_rewrites: int = 2) -> str:
    print("\n==================== ROUTER ====================")
    route = await route_question(agent, user_input)
    print("Decision:", route.action)
    print("Reason:", route.reason)

    if route.action == "answer":
        print("\n================ DIRECT ANSWER ================")
        return await direct_answer(agent, user_input)

    current_question = user_input

    for attempt in range(max_rewrites + 1):
        print(f"\n================ RETRIEVE (Attempt {attempt + 1}) ================")
        context = retrieve_docs(current_question)
        print("Retrieved Context Preview:")
        print(context[:500])

        if context == "NO_RESULTS":
            relevant = GradeDecision(binary_score="no")
        else:
            print("\n================ GRADER ====================")
            relevant = await grade_documents(agent, current_question, context)
            print("Relevant:", relevant.binary_score)

        if relevant.binary_score == "yes":
            print("\n================ FINAL ANSWER ====================")
            return await generate_answer(agent, current_question, context)

        if attempt < max_rewrites:
            print("\n================ REWRITE ====================")
            current_question = await rewrite_question(agent, current_question)
            print("Rewritten Question:", current_question)
        else:
            print("\n========= MAX REWRITES REACHED - FORCING ANSWER =========")
            forced_context = "" if context == "NO_RESULTS" else context
            return await generate_answer(agent, current_question, forced_context)

    return "I could not generate an answer."


# =========================
# MAIN
# =========================
async def main():
    agent = AzureOpenAIResponsesClient().create_agent(
        name="MAF Agentic RAG Agent",
        description="A MAF-based agent that can route, grade, rewrite, and answer.",
        instructions="""
You are a helpful assistant.

Rules:
- Follow the user's task exactly
- When asked to return JSON, return valid JSON only
- Do not wrap JSON in markdown
- When asked to return only plain text, return only plain text
- Be precise and concise
""",
        tools=[get_weather, simple_search],
    )

    print("MAF Agentic RAG running... (type 'exit' to quit)\n")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        answer = await agentic_rag_loop(agent, user_input, max_rewrites=2)
        print("\nAssistant:", answer)
        print()


if __name__ == "__main__":
    asyncio.run(main())