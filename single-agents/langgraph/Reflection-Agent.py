"""
Reflection Agent — LangGraph Implementation
--------------------------------------------
Converted from the official LangGraph reflection.ipynb tutorial.

An agent that generates an essay, critiques it, then refines it iteratively.
Stops after 3 reflection loops (when messages > 6).

Flow: Generate → Reflect → Generate → Reflect → Generate → Done

Requirements:
    pip install langgraph langchain-openai langchain-core python-dotenv
"""

import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="gpt-5-mini"
)

generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an essay assistant tasked with writing excellent 5-paragraph essays."
            " Generate the best essay possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a teacher grading an essay submission. Generate critique and recommendations"
            " for the user's submission. Provide detailed recommendations, including requests"
            " for length, depth, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Chain prompts with LLM
generate = generate_prompt | llm
reflect = reflection_prompt | llm

# =========================================================
# STATE
# =========================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]

# =========================================================
# NODES
# =========================================================
def generation_node(state: State) -> State:
    return {"messages": [generate.invoke({"messages": state["messages"]})]}


def reflection_node(state: State) -> State:
    # Flip message roles: AI → Human, Human → AI
    # This is the key trick from the notebook — the reflector sees
    # the essay as a "human" submission and its critique goes back as "human" feedback
    cls_map = {"ai": HumanMessage, "human": AIMessage}

    # Keep the first message (original user request) as-is, flip the rest
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]

    res = reflect.invoke({"messages": translated})

    # Return critique as HumanMessage so the generator treats it as user feedback
    return {"messages": [HumanMessage(content=res.content)]}


# =========================================================
# ROUTER
# =========================================================
def should_continue(state: State):
    # The notebook uses message count: > 6 messages = 3 full loops done
    # 1 user request + (1 essay + 1 critique) x 3 = 7 messages
    if len(state["messages"]) > 6:
        return END
    return "reflect"


# =========================================================
# GRAPH
# =========================================================
builder = StateGraph(State)

builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)

builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# =========================================================
# CLI LOOP
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Reflection Agent — LangGraph")
    print("=" * 60)
    print("The agent will generate an essay, critique it, and")
    print("refine it 3 times before giving the final result.")
    print("Type 'exit' to quit.\n")

    thread_id = 0

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        thread_id += 1
        config = {"configurable": {"thread_id": str(thread_id)}}

        print("\n[Starting reflection loop...]\n")

        for event in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
        ):
            for node_name, node_output in event.items():
                last_msg = node_output["messages"][-1]

                if node_name == "generate":
                    print(f"{'='*60}")
                    print(f"GENERATE")
                    print(f"{'='*60}")
                    print(last_msg.content)
                    print()

                elif node_name == "reflect":
                    print(f"{'='*60}")
                    print(f"REFLECT (critique)")
                    print(f"{'='*60}")
                    print(last_msg.content)
                    print()