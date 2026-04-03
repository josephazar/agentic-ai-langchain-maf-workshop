from dotenv import load_dotenv
from agent_framework.azure import AzureOpenAIResponsesClient

load_dotenv()

async def invoke_chat_supervisor(
    user_message: str,
    session_id: str,
    conversation_history: str = "",
) -> dict:

    agent = AzureOpenAIResponsesClient().create_agent(
        name="Chat Agent",
        instructions=f"""You are a conversational AI assistant.

Conversation history:
{conversation_history if conversation_history else "No previous conversation."}

Rules:
- Be natural and conversational
- Continue previous context when useful
- Keep responses moderately concise""",
    )

    response = await agent.run(user_message)
    return {"answer": response.text}