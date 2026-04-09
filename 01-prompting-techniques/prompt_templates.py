from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json
from pathlib import Path

load_dotenv()


raw_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

langchain_model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    
)

PROMPTS_PATH = Path(__file__).with_name("prompts") / "prompt_templates.json"
PROMPTS = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
review = PROMPTS["review"]


# APPROACH 1 — RAW API CALL
# You manually write everything from scratch every single time
# messages, role, content — all hardcoded around the variable


print("APPROACH 1 — RAW API CALL")


raw_response = raw_client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini"),
    
    messages=[
        {"role": "system", "content": PROMPTS["system_message"]},
        {"role": "user", "content": PROMPTS["user_message_template"].format(review=review)}
    ]
)
raw_output = raw_response.choices[0].message.content
print(f"Raw output:\n{raw_output}")
try:
    parsed = json.loads(raw_output)
    print(f"\nParsed: {parsed}")
except:
    print("Could not parse JSON")


# APPROACH 2 — LANGCHAIN RUNNABLESEQUENCE
# Define the template once, reuse it with any input
# prompt | model | parser is one clean pipeline


print("APPROACH 2 — LANGCHAIN RUNNABLESEQUENCE")


prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPTS["system_message"]),
    ("user", PROMPTS["user_message_template"])
])

chain = prompt | langchain_model | JsonOutputParser()

langchain_output = chain.invoke({"review": review})
print(f"Parsed output: {langchain_output}")
print(f"\nSentiment:  {langchain_output.get('sentiment')}")
print(f"Score:      {langchain_output.get('score')}")
print(f"Key issues: {langchain_output.get('key_issues')}")


# APPROACH 2b — SAME CHAIN, DIFFERENT REVIEW



print("APPROACH 2b — SAME CHAIN, DIFFERENT REVIEW")


review2 = PROMPTS["review2"]
langchain_output2 = chain.invoke({"review": review2})
print(f"Sentiment:  {langchain_output2.get('sentiment')}")
print(f"Score:      {langchain_output2.get('score')}")
print(f"Key issues: {langchain_output2.get('key_issues')}")


# OBSERVATION
# Raw: manual, verbose, not reusable
# Chained: template + model + parser in one pipeline
# Both hit the same Azure OpenAI endpoint
