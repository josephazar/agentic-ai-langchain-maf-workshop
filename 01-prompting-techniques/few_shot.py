import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
from pathlib import Path

load_dotenv()

# Initialize the client
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
PROMPTS_PATH = Path(__file__).with_name("prompts") / "few_shot.json"
PROMPTS = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def run_test(prompt, test_label):
    print(f"\n--- Running {test_label} ---\n")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": PROMPTS["system_instruction"]
            },
            {"role": "user", "content": prompt}
        ],
        
    )

    print(response.choices[0].message.content.strip())





fallacy_zero_shot = PROMPTS["fallacy_zero_shot"]
fallacy_few_shot = PROMPTS["fallacy_few_shot"]




classification_few_shot = PROMPTS["classification_few_shot"]
classification_few_shot_diversed = PROMPTS["classification_few_shot_diversed"]




confidence_zero_shot = PROMPTS["confidence_zero_shot"]
confidence_few_shot = PROMPTS["confidence_few_shot"]




def main():
    options = {
        "1": (fallacy_zero_shot, "Fallacy: Zero-Shot"),
        "2": (fallacy_few_shot, "Fallacy: Few-Shot"),
        "3": (classification_few_shot, "Classification: Few-Shot"),
        "4": (classification_few_shot_diversed, "Classification: Few-Shot-Diversed"),
        "5": (confidence_zero_shot, "Confidence: Zero-Shot"),
        "6": (confidence_few_shot, "Confidence: Few-Shot"),
        "0": ("exit", "Exit Program")
    }

    while True:
        print("\nSelect a test to run:\n")

        for key, (_, label) in options.items():
            print(f"{key}. {label}")

        choice = input("\nEnter choice: ").strip()

        if choice == "0":
            print("Exiting...")
            break

        elif choice in options:
            prompt, label = options[choice]
            run_test(prompt, label)

        else:
            print("Invalid choice. Try again.")
            
if __name__ == "__main__":
    main()