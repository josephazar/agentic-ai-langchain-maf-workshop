import os
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
INDEX_PATH = "faiss_index"

# Embeddings
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
    model="text-embedding-3-large"
)

# Check if index exists
if os.path.exists(INDEX_PATH):
    print("Loading existing FAISS index...")

    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:
    print("Creating new FAISS index...")

    docs = [
        Document(page_content="Marc Dagher is a beautiful person."),
        Document(page_content="Anthony Dagher is a kind person."),
        Document(page_content="Christian Dagher is a wonderful person."),
    ]

    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save index
    vectorstore.save_local(INDEX_PATH)
    print("FAISS index saved.")

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# Test query
query = "tell me about christian dagher"
results = retriever.invoke(query)

for doc in results:
    print("Result:", doc.page_content)