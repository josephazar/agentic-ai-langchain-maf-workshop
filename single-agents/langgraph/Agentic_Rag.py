import os
from dotenv import load_dotenv

from typing import Literal
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

from pydantic import BaseModel, Field

# ========================
# SETUP
# ========================
load_dotenv()

llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    decision: str
    rewrite_count: int  #  NEW

# ========================
# VECTOR STORE
# ========================
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
    model="text-embedding-3-large"
)

vectorstore = FAISS.load_local(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "utility", "faiss_index")
    ),
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever()

# ========================
# DEBUG
# ========================
DIVIDER = "=" * 60

def section(title, content):
    print(f"\n {title}\n{'-'*40}")
    print(content)

# ========================
# ROUTER
# ========================
class RouteDecision(BaseModel):
    action: Literal["retrieve", "answer"]
    reason: str

def router(state: State):
    print(f"\n{DIVIDER}\n ROUTER NODE\n{DIVIDER}")

    question = state["messages"][0].content

    decision = llm.with_structured_output(RouteDecision).invoke([
        {
            "role": "user",
            "content": f"""
Decide whether to retrieve or answer.

Rules:
- If question is about documents/blog -> retrieve
- If general knowledge -> answer
- If unsure -> retrieve

Question: {question}
"""
        }
    ])

    print(f"\n DECISION: {decision.action}")
    print(f" Reason: {decision.reason}")

    return {
        "messages": state["messages"],
        "decision": decision.action,
        "rewrite_count": state["rewrite_count"]
    }

# ========================
# RETRIEVE
# ========================
def retrieve_node(state: State):
    print(f"\n{DIVIDER}\n RETRIEVE NODE\n{DIVIDER}")

    question = state["messages"][0].content

    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs]) or "NO_RESULTS"

    section("RETRIEVED DOCS", context[:500])

    return {
        "messages": state["messages"] + [AIMessage(content=context)],
        "rewrite_count": state["rewrite_count"]
    }

# ========================
# GRADER
# ========================
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")

def grade_documents(state: State) -> Literal["generate_answer", "rewrite"]:
    print(f"\n{DIVIDER}\n GRADER NODE\n{DIVIDER}")

    question = state["messages"][0].content
    context = state["messages"][-1].content
    rewrite_count = state["rewrite_count"]

    section("QUESTION", question)
    section("CONTEXT", context[:300])
    print(f"\n Rewrite attempts: {rewrite_count}/2")

    # STOP CONDITION
    if rewrite_count >= 2:
        print("\n Max rewrites reached -> forcing answer")
        return "generate_answer"

    if context.strip() == "" or context == "NO_RESULTS":
        print("\n No useful docs -> rewrite")
        return "rewrite"

    result = llm.with_structured_output(GradeDocuments).invoke([
        {
            "role": "user",
            "content": f"""
Is this context relevant?

Question: {question}
Context: {context}

Answer only yes or no.
"""
        }
    ])

    print(f"\n Relevance: {result.binary_score}")

    if result.binary_score == "yes":
        return "generate_answer"

    return "rewrite"

# ========================
# REWRITE
# ========================
def rewrite(state: State):
    print(f"\n{DIVIDER}\n REWRITE NODE\n{DIVIDER}")

    question = state["messages"][0].content

    response = llm.invoke([
        {
            "role": "user",
            "content": f"Rewrite this for better retrieval:\n{question}"
        }
    ])

    new_count = state["rewrite_count"] + 1

    section("REWRITTEN QUESTION", response.content)
    print(f"\n Rewrite count: {new_count}/2")

    return {
        "messages": [HumanMessage(content=response.content)],
        "rewrite_count": new_count
    }

# ========================
# GENERATE ANSWER
# ========================
def generate_answer(state: State):
    print(f"\n{DIVIDER}\n GENERATE ANSWER NODE\n{DIVIDER}")

    question = state["messages"][0].content
    context = state["messages"][-1].content

    if context == "NO_RESULTS":
        context = ""

    response = llm.invoke([
        {
            "role": "user",
            "content": f"""
Answer clearly.

Question: {question}
Context: {context}
"""
        }
    ])

    section("FINAL ANSWER", response.content)

    return {"messages": [response]}

# ========================
# DIRECT ANSWER
# ========================
def direct_answer(state: State):
    print(f"\n{DIVIDER}\n DIRECT ANSWER NODE\n{DIVIDER}")

    question = state["messages"][0].content

    response = llm.invoke([
        {"role": "user", "content": question}
    ])

    section("FINAL ANSWER (NO RETRIEVAL)", response.content)

    return {"messages": [response]}

# ========================
# GRAPH
# ========================
def build_graph():
    workflow = StateGraph(State)

    workflow.add_node("router", router)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("direct_answer", direct_answer)

    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state["decision"],
        {
            "retrieve": "retrieve",
            "answer": "direct_answer",
        }
    )

    workflow.add_conditional_edges("retrieve", grade_documents)

    workflow.add_edge("rewrite", "router")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()

graph = build_graph()

# ========================
# RUN
# ========================
if __name__ == "__main__":
    print("\n Agentic RAG (FINAL - STABLE) running...\n")

    while True:
        q = input("You: ")

        if q.lower() in ["exit", "quit"]:
            break

        print(f"\n{'='*60}")
        print(f" QUESTION: {q}")
        print(f"{'='*60}")

        for _ in graph.stream({
            "messages": [{"role": "user", "content": q}],
            "decision": "",
            "rewrite_count": 0
        }):
            pass

        print(f"\n{'='*60}\n DONE\n{'='*60}\n")