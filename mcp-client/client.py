import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing import List, Dict, Any
import json

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            timeout=60
        )
    # methods will go here

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using GPT-4 and available tools"""

        messages = [
            {
                "role": "system",
                "content": "You are a coding assistant that talks like a pirate."
            },
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = []
        for tool in response.tools:
            available_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })

        final_text = []
        tool_results = []

        # Helper to convert message list to LangChain format
        def to_langchain_messages(msgs: List[Dict[str, Any]]):
            result = []
            for msg in msgs:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    result.append(SystemMessage(content=content))
                elif role == "user":
                    result.append(HumanMessage(content=content))
                elif role == "assistant":
                    result.append(AIMessage(content=content))
                elif role == "tool":
                    result.append(ToolMessage(tool_call_id=msg["tool_call_id"], content=content))
            return result

        while True:
            lc_messages = to_langchain_messages(messages)

            # First GPT response
            response = self.llm.invoke(
                input=lc_messages,
                tools=available_tools,
                tool_choice="auto"
            )

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    # Log tool use
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                    # Run tool and capture result
                    result = await self.session.call_tool(tool_name, tool_args)
                    tool_results.append({"call": tool_name, "result": result})

                    # Append assistant's tool call + tool result to messages
                    messages.append({
                        "role": "assistant",
                        "content": "",  # GPT tool call messages don't contain visible text
                    })
                    messages.append({
                        "role": "user",
                        "content": str(result.content)
                    })

                # Loop again with tool result
                continue

            # If no more tool calls, end conversation
            final_text.append(response.content)
            break

        return "\n".join(final_text) # weather alert for NY


    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())
