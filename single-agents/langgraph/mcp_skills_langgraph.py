import asyncio
import json
from typing import Annotated, Any, Dict, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

# ── Model ──────────────────────────────────────────────────────────────────────

model = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"),
)

system_prompt = SystemMessage(
    "You are a helpful AI assistant, please respond to the users query to the best of "
    "your ability! Do whatever the user ask you to do, including calling tools if "
    "necessary. If you call a tool, return the result of the tool call in your response. "
    "If you don't know the answer, just say 'I don't know'."
)


# ── Agent State ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ── Graph ──────────────────────────────────────────────────────────────────────

def create_graph(
    model: AzureChatOpenAI,
    system_prompt: SystemMessage,
    tools: list,
    tool_map: dict,
):
    bound_model = model.bind_tools(tools=tools, tool_choice="auto")

    async def tool_node(state: AgentState):
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            try:
                print(f"--- Invoking tool: {tool_call}")
                tool_result = await tool_map[tool_call["name"]].ainvoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            except Exception as e:
                error_msg = f"Error: {str(e)}. Please re-check your tool usage and try again."
                outputs.append(
                    ToolMessage(
                        content=error_msg,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        return {"messages": outputs}

    async def call_model(state: AgentState, config: RunnableConfig):
        to_send = [system_prompt] + list(state["messages"])
        print(f"--- Sending {len(to_send)} messages to model")
        response = await bound_model.ainvoke(to_send, config)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        return "continue" if last_message.tool_calls else "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ── Runner ─────────────────────────────────────────────────────────────────────

async def run_query(graph: Any, query: str) -> Dict[str, Any]:
    print(f"--- Query: '{query}'")
    result = await graph.ainvoke({"messages": [("user", query)]})
    print("--- Done.")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    async with sse_client("http://127.0.0.1:8787/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session=session)
            tool_map = {tool.name: tool for tool in tools}

            graph = create_graph(model, system_prompt, tools, tool_map)

            # ── CLI loop ───────────────────────────────────────────────────────
            print("\nLangGraph + MCP Agent  |  type 'quit' to exit\n")
            while True:
                try:
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ("quit", "exit", "q"):
                        print("Goodbye!")
                        break

                    response = await run_query(graph, user_input)
                    messages: list[BaseMessage] = response.get("messages", [])
                    print(f"\nAssistant: {messages[-1].content}\n")

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break


if __name__ == "__main__":
    asyncio.run(main())