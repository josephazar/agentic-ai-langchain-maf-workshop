import os
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# ========================
# LOAD EMBEDDINGS
# ========================
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
    model="text-embedding-3-large"
)


# ========================
# LOAD FAISS
# ========================
path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "utility", "faiss_index")
)

vectorstore = FAISS.load_local(
    path,
    embeddings,
    allow_dangerous_deserialization=True
)


# ========================
# PRINT DOCUMENTS
# ========================
docs = vectorstore.docstore._dict  # internal storage

print(f"\n📦 Total documents in FAISS: {len(docs)}\n")

for i, (doc_id, doc) in enumerate(docs.items()):
    print(f"--- Document {i} ---")
    print(doc.page_content[:500])  # print first 500 chars
    print()
    print(f"📍 Loading from: {path}")