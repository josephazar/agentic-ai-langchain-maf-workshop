import os
from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings


class State(TypedDict):
    messages: Annotated[list, add_messages]

memory = MemorySaver()



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



llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
)



def retrieve(state: State):
    query = state["messages"][-1].content

    results = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in results])

    return {
        "messages": [
            SystemMessage(content=f"Context:\n{context}")
        ]
    }



def agent(state: State):
    system_message = SystemMessage(content=(
        "You are a helpful assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not in the context, say: I don't know."
    ))

    messages = [system_message] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": [response]}

builder = StateGraph(State)

builder.add_node("retrieve", retrieve)
builder.add_node("agent", agent)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "agent")
builder.add_edge("agent", END)

graph = builder.compile(checkpointer=memory)


def run():
    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        for event in graph.stream(
            {"messages": [("user", user_input)]},
            {"configurable": {"thread_id": "1"}},
        ):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

if __name__ == "__main__":
    run()