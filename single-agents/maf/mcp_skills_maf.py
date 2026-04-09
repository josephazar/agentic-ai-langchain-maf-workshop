import asyncio
import json
import inspect
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import TextContent, ImageContent, EmbeddedResource

load_dotenv()

# Map JSON schema types to Python types
TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def make_mcp_tool_func(session: ClientSession, tool):
    tool_name = tool.name
    tool_description = tool.description or ""
    tool_schema = tool.inputSchema or {}
    properties = tool_schema.get("properties", {})
    required = tool_schema.get("required", [])

    # Build explicit inspect.Parameter objects for each MCP tool argument
    params = []
    for param_name, param_info in properties.items():
        json_type = param_info.get("type", "string")
        python_type = TYPE_MAP.get(json_type, str)
        default = inspect.Parameter.empty if param_name in required else None
        params.append(
            inspect.Parameter(
                name=param_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=python_type,
            )
        )

    async def mcp_tool_func(*args, **kwargs):
        # Bind positional args using the parameter names
        param_names = [p.name for p in params]
        for i, val in enumerate(args):
            if i < len(param_names):
                kwargs[param_names[i]] = val

        # Keep only the actual tool arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in properties}

        print(f"[DEBUG] Calling '{tool_name}' with: {filtered_kwargs}")

        try:
            result = await session.call_tool(tool_name, arguments=filtered_kwargs)

            if not result or not result.content:
                return f"Tool '{tool_name}' returned no content."

            parts = []
            for block in result.content:
                if isinstance(block, TextContent):
                    parts.append(block.text)
                elif isinstance(block, ImageContent):
                    parts.append(f"[Image: {block.mimeType}]")
                elif isinstance(block, EmbeddedResource):
                    parts.append(str(block.resource))
                elif hasattr(block, "text"):
                    parts.append(str(block.text))
                else:
                    parts.append(str(block))

            return "\n".join(parts) if parts else f"Tool '{tool_name}' returned empty content."

        except Exception as e:
            return f"Error calling tool '{tool_name}': {str(e)}"

    # Inject the real signature so agent_framework builds the correct LLM tool schema
    mcp_tool_func.__signature__ = inspect.Signature(params)
    mcp_tool_func.__name__ = tool_name
    mcp_tool_func.__doc__ = tool_description

    return mcp_tool_func


async def main():
    async with sse_client("http://127.0.0.1:8787/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            tools = [make_mcp_tool_func(session, tool) for tool in mcp_tools.tools]

            print("Loaded tools:", [t.__name__ for t in tools])

            agent = OpenAIChatClient().as_agent(
                name="MAF MCP Agent",
                description="An agent that uses tools loaded from an MCP server.",
                instructions="""
You are a helpful assistant with access to tools.
RULES:
- If the user asks about weather → ALWAYS call get_weather
- If the user asks about time → ALWAYS call get_current_time
- DO NOT answer from your own knowledge
- Return ONLY the tool result
""",
                tools=tools,
            )

            print("MAF + MCP Agent is running... (type 'exit' to quit)\n")

            while True:
                user_input = input("User: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                response = await agent.run(user_input)
                print("Assistant:", response.text)


if __name__ == "__main__":
    asyncio.run(main())