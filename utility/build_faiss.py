import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# ========================
# 1. LOAD DOCUMENTS
# ========================
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]


# ========================
# 2. SPLIT DOCUMENTS
# ========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

splits = splitter.split_documents(docs_list)


# ========================
# 3. EMBEDDINGS
# ========================
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
    model="text-embedding-3-large"
)


# ========================
# 4. PATH
# ========================
save_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "faiss_index")
)


# ========================
# 5. LOAD OR CREATE
# ========================
if os.path.exists(save_path):
    print(" Existing FAISS found → loading and adding documents...")

    vectorstore = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # ADD new documents
    vectorstore.add_documents(splits)

else:
    print(" No FAISS found → creating new index...")

    vectorstore = FAISS.from_documents(splits, embeddings)


# ========================
# 6. SAVE UPDATED INDEX
# ========================
vectorstore.save_local(save_path)

print(" FAISS index updated successfully!")
print(f" Location: {save_path}")