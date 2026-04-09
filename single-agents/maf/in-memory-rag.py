"""
MAF ReAct Agent with In-Memory RAG
- Accepts PDF, TXT, CSV file uploads at runtime
- Chunks, embeds, and stores in an ephemeral in-memory vector store
- Agent answers questions from uploaded documents only
"""

import asyncio
import os

from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

# ── In-memory vector store (ephemeral per session) ────────────────────────────

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    api_version="2024-12-01-preview",
)

vector_store: FAISS | None = None


def load_document(file_path: str) -> str:
    """Load a PDF, TXT, or CSV and index it into the in-memory vector store."""
    global vector_store

    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    else:
        return f"Unsupported file type: {ext}. Please upload PDF, TXT, or CSV."

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        vector_store.add_documents(chunks)

    return f" Loaded '{os.path.basename(file_path)}' — {len(chunks)} chunks indexed."


# ── RAG Tool (passed to MAF agent) ───────────────────────────────────────────

def search_documents(query: str) -> str:
    """
    Search the uploaded documents for information relevant to the query.
    Always use this tool when the user asks any question — answers must come
    from the uploaded files only.
    """
    if vector_store is None:
        return "No documents have been loaded yet. Please upload a PDF, TXT, or CSV file first."

    results = vector_store.similarity_search(query, k=4)
    if not results:
        return "No relevant content found in the uploaded documents."

    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in results
    )


# ── MAF Agent ────────────────────────────────────────────────────────────────

async def main():
    agent = OpenAIChatClient().as_agent(
        name="RAG Agent",
        description="An agent that answers questions from uploaded documents.",
        instructions="""
You are a document assistant with access to one tool: search_documents.

RULES:
- ALWAYS call search_documents before answering any question.
- Base your answer ONLY on the tool's results — do not use outside knowledge.
- If the documents don't contain the answer, say so clearly.
- Do not ask the user for clarification before trying the tool.
""",
        tools=[search_documents],
    )

    print("MAF ReAct Agent with In-Memory RAG")
    print("Commands:")
    print("  /load <file_path>  — load a PDF, TXT, or CSV file")
    print("  quit / exit        — exit\n")

    while True:
        user_input = input("User: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if user_input.startswith("/load "):
            file_path = user_input[6:].strip().strip('"').strip("'")
            result = load_document(file_path)
            print(f"System: {result}\n")
            continue

        response = await agent.run(user_input)
        print("Assistant:", response.text)


if __name__ == "__main__":
    asyncio.run(main())