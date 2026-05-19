# Customer Service Data Analyst Agent

A LangGraph-based ReAct agent that answers user questions about the [Bitext Customer Service Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file with your Nebius API key
echo NEBIUS_API_KEY=your-key-here > .env

# 4. Place the dataset CSV in the data/ folder
mkdir data
# Download from HuggingFace and save as data/bitext_customer_service.csv

# 5. Run the agent
python main.py

# With a named session and user profile
python main.py --session my_session --user alice
```

## Architecture

```
User query
    │
    ▼
┌──────────┐   structured     ┌─────────────────┐
│  Router   │────────────────▶│  ReAct Agent     │
│  (LLM)   │   unstructured  │  (tool-calling   │
│           │────────────────▶│   loop)          │
│           │   personal      │                  │
│           │────────────────▶│  [dataset tools  │
└──────────┘                  │   + profile      │
    │ out_of_scope            │   tools]         │
    ▼                         └─────────────────┘
┌──────────┐                          │
│ Polite   │                          ▼
│ Decline  │                  ┌─────────────────┐
└──────────┘                  │  Final Answer   │
                              └─────────────────┘
```

The **Router** is a dedicated graph node that classifies every incoming query as
`structured`, `unstructured`, `personal`, or `out_of_scope` before the agent begins
tool selection. Out-of-scope queries are declined without invoking tools.

The **ReAct Agent** uses LangGraph's `create_react_agent` with composable tools.
It can chain multiple tools for multi-step queries (e.g. filter → count).

**Episodic Memory** (Task 2a): SqliteSaver checkpointer persists conversation state
across turns and restarts. Same `--session` ID = same conversation restored.

**Semantic Memory** (Task 2b): User profile stored as JSON files in `profiles/`.
The agent can save and retrieve user facts (name, interests, preferences).

## Model Choice

| Role   | Model                              | Justification |
|--------|------------------------------------|---------------|
| Router | `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE model (30B total, 3B active) — fast (70 Tok/s), cheap ($0.10/1M in), sufficient for 4-way classification |
| Agent  | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Optimized for chat, reasoning, and tool use. Strong balance of capability and cost-efficiency for multi-step ReAct loops |

Both models served via **Nebius Token Factory**.

## Tools

| Tool | Purpose | Multi-step Example |
|------|---------|--------------------|
| `get_categories` | List all categories | — |
| `get_intents` | List intents (optionally by category) | distribution queries |
| `filter_and_count` | Filter + count rows by column/value | "How many refund requests?" |
| `show_examples` | Show N example rows with optional filters | "Show 3 examples from SHIPPING" |
| `get_sample_for_summary` | Get representative sample for LLM summarization | "Summarize FEEDBACK" |
| `dataset_overview` | High-level stats | "Tell me about the dataset" |
| `get_user_profile` | Retrieve user's stored profile | "What do you remember about me?" |
| `update_user_profile` | Save user facts (name, interests, etc.) | "My name is Alice" |

## MCP Server (Task 3)

The MCP server exposes 4 dataset tools via the Model Context Protocol using FastMCP.

### Starting the server

```bash
python mcp_server.py
```

This starts the server with **stdio transport** (default). It runs as a subprocess
and communicates via stdin/stdout — no network exposure needed.

### Connecting a client

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def main():
    async with MultiServerMCPClient({
        "customer_service": {
            "command": "python",
            "args": ["mcp_server.py"],
        }
    }) as client:
        tools = client.get_tools()
        print(f"Available tools: {[t.name for t in tools]}")

        llm = ChatOpenAI(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            base_url="https://api.studio.nebius.com/v1/",
            api_key="your-key",
        )
        agent = create_react_agent(llm, tools=tools)
        result = await agent.ainvoke(
            {"messages": [("user", "What categories exist?")]}
        )
        print(result["messages"][-1].content)

asyncio.run(main())
```

Or run the included example directly:
```bash
python mcp_client_example.py
```

### MCP tools exposed

| MCP Tool | Description |
|----------|-------------|
| `get_categories` | List all unique categories |
| `get_intents` | List intents with counts, optionally by category |
| `filter_and_count` | Filter + count rows by column/value |
| `show_examples` | Show N example rows with optional filters |

## Project Structure

```
├── main.py              # CLI entrypoint
├── agent.py             # LangGraph agent + graph definition
├── router.py            # Query router node
├── tools.py             # Tool definitions with Pydantic schemas
├── user_profile.py      # Persistent user profile (semantic memory)
├── data_loader.py       # Dataset loading
├── config.py            # Model, paths, and constants
├── mcp_server.py        # FastMCP server (Task 3)
├── mcp_client_example.py# Example MCP client connection
├── requirements.txt     # Dependencies
├── .env                 # API key (not committed)
├── profiles/            # User profile JSON files (auto-created)
├── checkpoints.db       # SQLite conversation persistence (auto-created)
└── data/                # Dataset CSV
```
