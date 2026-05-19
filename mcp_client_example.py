"""
mcp_client_example.py — Example: connect to the MCP server and call a tool.

Requires: pip install fastmcp langchain-mcp-adapters

Usage:
    python mcp_client_example.py
"""
import os
import sys
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from config import NEBIUS_API_BASE, NEBIUS_API_KEY, AGENT_MODEL


async def main():
    """Connect to the MCP server and run a query through an agent."""

    # 1. Create the MCP client
    #    Use sys.executable to ensure the SAME Python interpreter is used
    #    for the subprocess (avoids path issues on Windows)
    client = MultiServerMCPClient(
            {
                "customer_service": { # Server name
                    "transport": "stdio", # Communication method
                    "command": sys.executable,  # Python path
                    "args": [os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")],  # Server script
                }
            }
        )

    # 2. Get all tools exposed by the MCP server
    tools = await client.get_tools()
    print(f"Available MCP tools: {[t.name for t in tools]}\n")

    # 3. Create a LangGraph agent using the MCP tools
    llm = ChatOpenAI(
        model=AGENT_MODEL,
        base_url=NEBIUS_API_BASE,
        api_key=NEBIUS_API_KEY,
    )
    agent = create_react_agent(llm, tools=tools)

    # 4. Run a query
    result = await agent.ainvoke(
        {"messages": [("user", "What categories exist in the dataset?")]}
    )

    # 5. Print the result
    final_message = result["messages"][-1]
    print(f"Agent response:\n{final_message.content}")


if __name__ == "__main__":
    asyncio.run(main())
