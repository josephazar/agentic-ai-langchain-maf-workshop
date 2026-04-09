import asyncio
import os
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv
from agent_framework.openai import OpenAIChatClient

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# ========================
# LOGGER
# ========================
LOG_FILE = "knowledge_agent_log.txt"

def _write(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def init_log():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"  KNOWLEDGE AGENT LOG  session started {ts}\n")
        f.write("=" * 80 + "\n")

def log_prompt(node: str, prompt: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _write(f"\n  +-- PROMPT SENT TO LLM [{node.upper()}]  ({ts})")
    _write("  |")
    for line in prompt.strip().splitlines():
        _write(f"  |  {line}")
    _write("  +" + "-" * 60)

def log_section(label: str, content: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _write(f"\n  +-- [{label.upper()}]  ({ts})")
    _write("  |")
    for line in content.strip().splitlines():
        _write(f"  |  {line}")
    _write("  +" + "-" * 60)

init_log()

# ========================
# FAISS + TAVILY SETUP
# ========================

embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview",
    model="text-embedding-3-large"
)

vectorstore = FAISS.load_local(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "utility", "faiss_index")
    ),
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever()

tavily = TavilySearchResults(max_results=5)


# ========================
# AGENT FACTORIES
# ========================

def create_supervisor_agent(conversation_history: str):
    instructions = f"""You are a Knowledge Supervisor deciding which agents to use.

Available agents:
- rag -> retrieves from internal vector store (blog posts, documents, curated content)
- web -> searches the internet via Tavily for current or external information

Conversation history:
{conversation_history or "This is the first message."}

Rules:
- Use RAG when the question is about internal documents, blog posts, or curated knowledge
- Use Web when the question needs current events, recent data, or external sources
- Use BOTH when the question benefits from combining internal + external knowledge
- Use only ONE when the other adds no value
- If the latest message is a follow-up, use the conversation history to infer context
- Write self-contained tasks for rag/web so they have enough context
- Set task to null for agents you do NOT want to call
- The task field must be a SHORT search query (max 10 words), not a paragraph

Respond ONLY in this exact JSON format, nothing else:
{{
  "rag": {{ "task": "specific retrieval task or null" }},
  "web": {{ "task": "specific search task or null" }}
}}"""

    log_prompt("supervisor_instructions", instructions)

    return OpenAIChatClient().as_agent(
        name="Knowledge Supervisor",
        description="Decides which knowledge agents to call.",
        instructions=instructions,
    )


def create_merger_agent():
    instructions = """You are a knowledge assistant synthesizing information from multiple sources.

Instructions:
- Answer the user's question clearly and completely
- If both internal and web sources were used, synthesize them into one coherent answer
- Do not mention "RAG" or "web search" explicitly -- just answer naturally
- If context is missing or irrelevant, answer from your own knowledge
- Be concise but thorough"""

    log_prompt("merger_instructions", instructions)

    return OpenAIChatClient().as_agent(
        name="Knowledge Merger",
        description="Synthesizes RAG and web search results into a final answer.",
        instructions=instructions,
    )


# ========================
# RAG + WEB HELPERS
# ========================

async def run_rag(task: str) -> str:
    print("  RAG agent running...")
    log_section("rag_query", task)

    docs = await asyncio.to_thread(retriever.invoke, task)

    if not docs:
        log_section("rag_result", "NO_RESULTS")
        return "NO_RESULTS"

    output = "\n\n".join([d.page_content for d in docs])
    log_section("rag_result", output)
    return output


async def run_web(task: str) -> str:
    print("  Web search agent running...")
    log_section("web_query", task)

    try:
        results = await asyncio.to_thread(tavily.invoke, {"query": task})
    except Exception as e:
        log_section("web_error", str(e))
        return "NO_RESULTS"

    if not results:
        log_section("web_result", "NO_RESULTS")
        return "NO_RESULTS"

    parts = []
    for r in results:
        if isinstance(r, dict):
            parts.append(
                f"Source: {r.get('url', 'unknown')}\n{r.get('content', '')}"
            )
        else:
            parts.append(str(r))

    output = "\n\n".join(parts)
    log_section("web_result", output)
    return output


# ========================
# ENTRY POINT
# ========================

async def invoke_knowledge_supervisor(
    user_message: str,
    conversation_history: str = "",
) -> dict:

    print("  Knowledge supervisor planning...")

    log_section("invoke_start", f"user_message: {user_message}")

    supervisor_agent = create_supervisor_agent(conversation_history)

    log_section("supervisor_user_message", user_message)

    supervisor_response = await supervisor_agent.run(user_message)

    log_section("supervisor_response", supervisor_response.text)

    try:
        raw = supervisor_response.text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        plan = json.loads(raw.strip())

    except Exception as e:
        print(f"  Supervisor JSON parse failed: {e} -- defaulting to web only")
        log_section("supervisor_parse_error", str(e))

        plan = {
            "rag": {"task": None},
            "web": {"task": user_message},
        }

    rag_task = plan.get("rag", {}).get("task")
    web_task = plan.get("web", {}).get("task")

    log_section("supervisor_plan", f"RAG task: {rag_task or 'SKIPPED'}\nWeb task: {web_task or 'SKIPPED'}")

    print(f"  RAG task : {rag_task or 'SKIPPED'}")
    print(f"  Web task : {web_task or 'SKIPPED'}")

    # ========================
    # RUN AGENTS IN PARALLEL
    # ========================

    async def maybe_rag():
        return await run_rag(rag_task) if rag_task else "NOT_CALLED"

    async def maybe_web():
        return await run_web(web_task) if web_task else "NOT_CALLED"

    rag_result, web_result = await asyncio.gather(
        maybe_rag(),
        maybe_web(),
    )

    # ========================
    # MERGE
    # ========================

    print("  Merging results...")

    context_parts = []

    if rag_result not in ("NOT_CALLED", "NO_RESULTS"):
        context_parts.append(f"=== Internal Knowledge (RAG) ===\n{rag_result}")

    if web_result not in ("NOT_CALLED", "NO_RESULTS"):
        context_parts.append(f"=== Web Search Results ===\n{web_result}")

    if not context_parts:
        context_section = "No context was retrieved from any source."
    else:
        context_section = "\n\n".join(context_parts)

    merge_prompt = f"""User question:
{user_message}

Retrieved context:
{context_section}
"""

    log_prompt("merger_prompt", merge_prompt)

    merger_agent = create_merger_agent()
    merge_response = await merger_agent.run(merge_prompt)

    log_section("merger_response", merge_response.text)

    return {
        "answer": merge_response.text,
    }