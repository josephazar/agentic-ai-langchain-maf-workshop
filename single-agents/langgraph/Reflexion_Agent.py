import os
import json
import datetime
from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, ValidationError
import faiss
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_community.vectorstores import FAISS

load_dotenv()

# =========================================================
# LLM
# =========================================================
llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="gpt-5-mini"
)

# =========================================================
# VECTOR MEMORY
# =========================================================
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
    model="text-embedding-3-large"
)

# =========================================================
# VECTOR MEMORY (FIXED)
# =========================================================

VECTOR_DB_PATH = "reflexion_memory"
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")

if os.path.exists(index_file):
    print(" Loading existing vector DB...")
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print(" Creating new vector DB...")

    dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(dim)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),  # fine for now
        index_to_docstore_id={}
    )

def store_reflection(question: str, reflection: str):
    doc = Document(
        page_content=f"Q: {question}\nReflection: {reflection}",
        metadata={"timestamp": datetime.datetime.now().isoformat()}
    )
    vectorstore.add_documents([doc])
    vectorstore.save_local(VECTOR_DB_PATH)


def retrieve_reflections(query: str, k: int = 3):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs]) if docs else "None"


# =========================================================
# METRICS
# =========================================================
metrics = []


# =========================================================
# TOOLS
# =========================================================
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)


# =========================================================
# SCHEMAS
# =========================================================
class Reflection(BaseModel):
    missing: str
    superfluous: str


class AnswerQuestion(BaseModel):
    answer: str
    reflection: Reflection
    search_queries: list[str]


class ReviseAnswer(AnswerQuestion):
    references: list[str]


# =========================================================
# RESPONDER
# =========================================================
class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: dict):
        question = state["messages"][0].content

        past_reflections = retrieve_reflections(question)

        memory_message = HumanMessage(
            content=f"Past mistakes to avoid:\n{past_reflections}"
        )

        base_messages = [state["messages"][0], memory_message]

        last_ai = None
        last_critic = None

        for attempt in range(3):
            messages = base_messages.copy()

            if last_ai:
                messages.append(last_ai)

            if last_critic:
                messages.append(last_critic)

            response = self.runnable.invoke(
                {"messages": messages},
                {"tags": [f"attempt:{attempt}"]}
            )

            try:
                self.validator.invoke(response)

                tool_call = response.tool_calls[0]["args"]

                # Store reflection
                if "reflection" in tool_call:
                    store_reflection(
                        question,
                        json.dumps(tool_call["reflection"])
                    )

                # Track metrics
                metrics.append({
                    "attempt": attempt,
                    "has_reflection": "reflection" in tool_call,
                    "num_queries": len(tool_call.get("search_queries", []))
                })

                return {"messages": response}

            except ValidationError as e:
                last_ai = response
                last_critic = ToolMessage(
                    content=(
                        f"{repr(e)}\n\n"
                        "Fix schema errors ONLY.\n\n"
                        + self.validator.schema_json()
                    ),
                    tool_call_id=response.tool_calls[0]["id"],
                )

        return {"messages": response}


# =========================================================
# PROMPTS
# =========================================================
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe.
3. Recommend search queries to improve your answer.

IMPORTANT:
Use past mistakes to improve your answer.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "Respond using the {function_name} function.",
        ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


# Initial
initial_chain = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer.",
    function_name=AnswerQuestion.__name__,
) | llm.bind_tools(tools=[AnswerQuestion])

validator = PydanticToolsParser(tools=[AnswerQuestion])
draft_node = ResponderWithRetries(initial_chain, validator)


# Revision
revision_chain = actor_prompt_template.partial(
    first_instruction="Revise your answer using critique and add references.",
    function_name=ReviseAnswer.__name__,
) | llm.bind_tools(tools=[ReviseAnswer])

revision_validator = PydanticToolsParser(tools=[ReviseAnswer])
revise_node = ResponderWithRetries(revision_chain, revision_validator)


# =========================================================
# TOOL NODE
# =========================================================
def run_queries(search_queries: list[str], **kwargs):
    """Run search queries using Tavily and return results."""
    return tavily_tool.batch([{"query": q} for q in search_queries])

tool_node = ToolNode([
    StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
    StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
])


# =========================================================
# GRAPH
# =========================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

builder.add_node("draft", draft_node.respond)
builder.add_node("tools", tool_node)
builder.add_node("revise", revise_node.respond)

builder.add_edge(START, "draft")
builder.add_edge("draft", "tools")
builder.add_edge("tools", "revise")


MAX_ITERATIONS = 5


def loop(state):
    count = sum(1 for m in state["messages"] if m.type in ["ai", "tool"])
    return END if count > MAX_ITERATIONS else "tools"


builder.add_conditional_edges("revise", loop, ["tools", END])

graph = builder.compile()


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    print("Reflexion Agent with Memory\n")

    def print_all_reflections():
        docs = vectorstore.similarity_search("", k=100)

        print("\n Stored Reflections:\n")
        for i, doc in enumerate(docs):
            print(f"--- {i} ---")
            print(doc.page_content)
            print(doc.metadata)
            print()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            break

        
        if user_input.lower() == "memory":
            print_all_reflections()
            continue

        events = graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            stream_mode="values",
        )

        for i, step in enumerate(events):
            msg = step["messages"][-1]
            print(f"\n--- Step {i} ({msg.type}) ---")
            msg.pretty_print()

        print("\n Metrics:")
        for m in metrics:
            print(m)

        metrics.clear()