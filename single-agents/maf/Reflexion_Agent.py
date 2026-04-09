import asyncio
import os
import json
import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import faiss

from agent_framework.openai import OpenAIChatClient

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

load_dotenv()

# =========================================================
# CONFIG
# =========================================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5-mini")

VECTOR_DB_PATH = "reflexion_memory"
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# =========================================================
# EMBEDDINGS + VECTOR MEMORY
# =========================================================
embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model="text-embedding-3-large",
)

index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")

if os.path.exists(index_file):
    print("Loading existing vector DB...")
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    print("Creating new vector DB...")
    dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(dim)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )


def store_reflection(question: str, reflection: dict):
    doc = Document(
        page_content=f"Q: {question}\nReflection: {json.dumps(reflection, ensure_ascii=False)}",
        metadata={"timestamp": datetime.datetime.now().isoformat()},
    )
    vectorstore.add_documents([doc])
    vectorstore.save_local(VECTOR_DB_PATH)


def retrieve_reflections(query: str, k: int = 3) -> str:
    if not vectorstore.index_to_docstore_id:
        return "None"

    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    docs = [doc for doc, score in results if score >= 0.8]

    if not docs:
        return "None"

    return "\n\n".join(d.page_content for d in docs)


def print_all_reflections():
    all_docs = list(vectorstore.docstore._dict.values())

    print("\nStored Reflections:\n")

    if not all_docs:
        print("No reflections stored yet.\n")
        return

    for i, doc in enumerate(all_docs):
        print(f"--- {i} ---")
        print(doc.page_content)
        print(doc.metadata)
        print()


# =========================================================
# SEARCH TOOL
# =========================================================
search = TavilySearchAPIWrapper()


def run_queries(search_queries: list[str]) -> str:
    """
    Run web searches for the provided queries and return a compact text bundle.
    """
    if not search_queries:
        return "No search queries provided."

    chunks = []

    for q in search_queries:
        try:
            result = search.results(q, max_results=5)

            chunks.append(f"QUERY: {q}")

            if not result:
                chunks.append("No results found.\n")
                continue

            for i, item in enumerate(result, start=1):
                title = item.get("title", "No title")
                url = item.get("url", "No url")
                content = item.get("content", "")

                chunks.append(
                    f"[{i}] {title}\n"
                    f"URL: {url}\n"
                    f"CONTENT: {content}\n"
                )

        except Exception as e:
            chunks.append(f"QUERY: {q}\nERROR: {str(e)}\n")

    return "\n".join(chunks)


# =========================================================
# OUTPUT SCHEMA WE WANT THE AGENT TO FOLLOW
# =========================================================
class FinalAnswerSchema(BaseModel):
    answer: str
    reflection: dict
    search_queries: list[str]
    references: list[str]


# =========================================================
# BUILD AGENT
# =========================================================
def build_agent():
    client = OpenAIChatClient(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        model=AZURE_OPENAI_MODEL,
    )

    agent = client.as_agent(
        name="ReflexionResearchAgent",
        description="A self-reflective researcher agent with memory and web search.",
        instructions="""
You are an expert researcher.

Your job is to answer the user's question while internally following this process:

1. Draft an answer.
2. Critique the draft harshly.
3. Produce search queries if more evidence is needed.
4. Call run_queries when needed.
5. Revise the answer using search results.
6. Return one final JSON object only.

Rules:
- Use the user's past mistakes/reflections as guidance.
- You may call run_queries more than once if needed.
- Be concise but useful.
- Do not expose chain-of-thought.
- Output must be valid JSON with exactly these keys:
  {
    "answer": string,
    "reflection": object,
    "search_queries": string[],
    "references": string[]
  }
- "reflection" should contain short fields like:
  {
    "missing": "...",
    "superfluous": "...",
    "improvements_made": "..."
  }
- "references" should be a list of URLs or source labels if available.
- Final output must be JSON only, no markdown fences.
""",
        tools=[run_queries],
    )

    return agent


# =========================================================
# SAFE JSON PARSING
# =========================================================
def parse_json_response(text: str) -> FinalAnswerSchema:
    try:
        data = json.loads(text)
        return FinalAnswerSchema(**data)
    except Exception as e:
        raise ValidationError.from_exception_data(
            title="FinalAnswerSchema",
            line_errors=[],
        ) from e


# =========================================================
# OUTER RETRY LOOP
# =========================================================
async def run_reflexion_turn(agent, user_input: str, max_attempts: int = 3):
    past_reflections = retrieve_reflections(user_input)

    prompt = f"""
Current time: {datetime.datetime.now().isoformat()}

User question:
{user_input}

Past mistakes to avoid:
{past_reflections}

Return valid JSON only.
"""

    last_text = None

    for attempt in range(max_attempts):
        print("\n" + "=" * 60)
        print(f" ATTEMPT {attempt + 1}")
        print("=" * 60)

        print("\nPROMPT SENT TO MODEL:\n")
        print(prompt)

        response = await agent.run(prompt)
        last_text = response.text

        print("\nRAW RESPONSE FROM MODEL:\n")
        print(last_text)

        try:
            parsed = parse_json_response(response.text)

            # Save reflection on every attempt
            store_reflection(user_input, parsed.reflection)
            print("\nReflection stored in memory")

            # =========================
            # REFLECTION CHECK
            # =========================
            missing = parsed.reflection.get("missing", "").lower()
            should_retry = False

            if missing and len(missing) > 20:
                print("\nReflection indicates missing info -> retrying...\n")
                should_retry = True

            # =========================
            # FORCE FINAL ON LAST ATTEMPT
            # =========================
            if attempt == max_attempts - 1:
                print("\nLast attempt reached -> forcing final answer\n")
                should_retry = False

            if not should_retry:
                print("\nFINAL ANSWER ACCEPTED\n")
                return parsed

            # =========================
            # BUILD RETRY PROMPT
            # =========================
            prompt = f"""
Your previous answer was not good enough.

Improve it using this reflection:

Reflection:
{json.dumps(parsed.reflection, indent=2)}

Fix the issues and provide a better answer.

Return valid JSON only.

Original question:
{user_input}

Past mistakes:
{past_reflections}

Previous answer:
{parsed.answer}
"""

        except Exception:
            print("\nParsing failed -> retrying...\n")

            prompt = f"""
Your previous output was invalid or not parseable.

Fix it and return valid JSON only.

Schema:
{{
  "answer": "string",
  "reflection": {{
    "missing": "string",
    "superfluous": "string",
    "improvements_made": "string"
  }},
  "search_queries": ["string"],
  "references": ["string"]
}}

Original user question:
{user_input}

Past mistakes to avoid:
{past_reflections}

Previous invalid output:
{last_text}
"""

    raise RuntimeError(f"Failed after {max_attempts} attempts.\nLast output:\n{last_text}")


# =========================================================
# CLI
# =========================================================
async def main():
    agent = build_agent()

    print("MAF Reflexion Agent with Memory\n")
    print("Commands:")
    print("  memory   -> print stored reflections")
    print("  exit     -> quit\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "memory":
            print_all_reflections()
            continue

        try:
            result = await run_reflexion_turn(agent, user_input)

            print("\n" + "=" * 60)
            print("FINAL ANSWER")
            print("=" * 60)
            print(result.answer)

            if result.search_queries:
                print("\nSearch Queries Used:")
                for q in result.search_queries:
                    print(f"- {q}")

            if result.references:
                print("\nReferences:")
                for ref in result.references:
                    print(f"- {ref}")

            print("\nReflection:")
            print(json.dumps(result.reflection, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"\nERROR: {str(e)}\n")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    asyncio.run(main())