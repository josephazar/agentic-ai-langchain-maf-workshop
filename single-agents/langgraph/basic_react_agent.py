from typing import Literal

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import AzureChatOpenAI
import os
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def simple_search(query: str):
    """if user asks about ai or python or langgraph return hardcoded answers"""
    query = query.lower()

    if "ai" in query:
        return "AI stands for Artificial Intelligence. It allows machines to learn and make decisions."
    elif "python" in query:
        return "Python is a programming language used for AI, web, and automation."
    elif "langgraph" in query:
        return "LangGraph is a framework for building stateful, multi-step AI agents."
    else:
        return "No results found."

@tool
def get_weather(location: str = "San Francisco"):
    """Call to get the current weather. Defaults to San Francisco if no location given."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."

tools = [get_weather, simple_search]
tool_node = ToolNode(tools)

llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini") 
)

llm_with_tools = llm.bind_tools(tools)

def tool_router(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def agent(state: State):
    system_message = SystemMessage(content=(
    "You are a helpful assistant with access to two tools:\n"
    "1. get_weather: call this for any weather-related question. If no location is given, default to San Francisco.\n"
    "2. simple_search: always call this when the user asks about AI, Python, or LangGraph.\n"
    "Always use tools when relevant — never ask the user for clarification before trying."
    "After calling a tool, return ONLY the tool's result to the user. Do not add, expand, or elaborate."
    ))
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)
    print(response)
    return {"messages": [response]}
builder = StateGraph(State)

builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tool_router, ["tools", END])
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=memory)

def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [("user", user_input)]},
        {"configurable": {"thread_id": "1"}},
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        import traceback
        traceback.print_exc()
        break
    