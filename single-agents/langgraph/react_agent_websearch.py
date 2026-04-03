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
from langchain_community.tools.tavily_search import TavilySearchResults

memory = MemorySaver()

from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]


def debug_message(prefix, msg):
    print(f"\n[{prefix}]")
    print(f"type: {type(msg)}")
    print(f"content: {getattr(msg, 'content', None)}")
    print(f"tool_calls: {getattr(msg, 'tool_calls', None)}")


search_tool = TavilySearchResults(max_results=2)

tools = [search_tool]
tool_node = ToolNode(tools)


llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="gpt-5-mini" 
)

llm_with_tools = llm.bind_tools(tools)


def tool_router(state: State):
    print("\n=== ROUTER ===")

    messages = state["messages"]
    last_message = messages[-1]

    debug_message("LAST MESSAGE", last_message)

    if last_message.tool_calls:
        print("Routing decision: TOOLS")
        return "tools"

    print("Routing decision: END")
    return END


def agent(state: State):
    print("\n=== AGENT NODE ===")

    print("\nIncoming state messages:")
    for m in state["messages"]:
        debug_message("STATE MESSAGE", m)

    system_message = SystemMessage(content=(
        "You are a helpful assistant with access to a web search tool (Tavily).\n"
        "Use the tool when:\n"
        "- The question requires up-to-date information\n"
        "- The question is about news, facts, or unknown topics\n"
        "- You are not 100% sure of the answer\n\n"
        "Do NOT use the tool for simple or well-known questions.\n"
        "Use the tool result to answer the user clearly and concisely."
    ))

    messages = [system_message] + state["messages"]

    print("\nSending to LLM...")

    response = llm_with_tools.invoke(messages)

    print("\n=== LLM RESPONSE ===")
    debug_message("LLM OUTPUT", response)

    if response.tool_calls:
        print("Decision: TOOL WILL BE USED")
    else:
        print("Decision: NO TOOL (END)")

    return {"messages": [response]}


builder = StateGraph(State)

builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tool_router, ["tools", END])
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str):
    print("NEW RUN")
    print(f"User input: {user_input}")
    for i, event in enumerate(graph.stream(
        {"messages": [("user", user_input)]},
        {"configurable": {"thread_id": "1"}},
    )):
        print(f"\n--- EVENT {i} ---")

        for node_name, value in event.items():
            print(f"\nNode executed: {node_name}")

            last_msg = value["messages"][-1]
            debug_message("NODE OUTPUT", last_msg)

            print("Assistant:", last_msg.content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)

    except Exception as e:
        print("Error:", e)
        print("Goodbye!")
        break