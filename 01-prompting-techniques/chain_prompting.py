import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

MODEL_NAME = "gpt-5-mini"
PROMPTS_PATH = Path(__file__).with_name("prompts") / "chain_prompting.json"
PROMPTS = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


article = PROMPTS["article"]


def run_manual_chain():
    print("\n--- Running Manual Chain ---\n")

    # Step 1
    print("Step 1 - Summarizing...")
    response1 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": PROMPTS["manual_summary_template"].format(article=article)}],
    )
    summary = response1.choices[0].message.content
    print("Summary:\n", summary)

    

    # Step 2
    print("Step 2 - Extracting key points...")
    response2 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": PROMPTS["manual_key_points_template"].format(summary=summary)}],
    )
    key_points = response2.choices[0].message.content
    print("Key Points:\n", key_points)

    

    # Step 3
    print("Step 3 - Drafting email...")
    response3 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": PROMPTS["manual_email_template"].format(key_points=key_points)}],
    )
    email = response3.choices[0].message.content
    print("Email:\n", email)


# =========================
# LANGCHAIN CHAIN
# =========================
def run_langchain_chain():
    print("\n--- Running LangChain LCEL Chain ---\n")

    llm = AzureChatOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-12-01-preview",
        deployment_name=MODEL_NAME,
        
    )

    parser = StrOutputParser()

    summary_prompt = PromptTemplate.from_template(
        PROMPTS["summary_template"]
    )

    keypoints_prompt = PromptTemplate.from_template(
        PROMPTS["keypoints_template"]
    )

    email_prompt = PromptTemplate.from_template(
      PROMPTS["email_template"]
    )

   
    chain = (
        summary_prompt
        | llm
        | parser
        | (lambda summary: {"summary": summary})  # pass forward
        | keypoints_prompt
        | llm
        | parser
        | (lambda key_points: {"key_points": key_points})
        | email_prompt
        | llm
        | parser
    )

   
    result = chain.invoke({"article": article})
    print("Final Email Output:\n")
    print(result)



def main():
    while True:
        print("\nChoose mode:\n")
        print("1. Manual Chain (your original)")
        print("2. LangChain Chain")
        print("0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            run_manual_chain()
        elif choice == "2":
            run_langchain_chain()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()