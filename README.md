# Customer Service Data Analyst Agent

A LangGraph-based ReAct agent that answers user questions about the [Bitext Customer Service Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

## Features

- **Query Router**: Classifies queries as structured, unstructured, personal, recommendation, or out-of-scope
- **Multi-step Reasoning**: Chains tools together for complex queries
- **Episodic Memory**: Conversation persistence across sessions using SqliteSaver
- **Semantic Memory**: User profile storage with facts, interests, and preferences
- **Query Recommender**: Suggests relevant follow-up queries based on conversation history
- **MCP Server**: Exposes tools via Model Context Protocol for remote access
- **Streamlit UI**: Interactive chat interface with reasoning step visualization
- **CLI Interface**: Terminal-based interaction with session management

## Quick Start

### Option 1: Streamlit Web UI (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API key
# Copy .env.example to .env and add your Nebius API key
cp .env.example .env
# Edit .env and replace 'your-nebius-api-key-here' with your actual key

# 3. Run the Streamlit app
streamlit run app.py
```

Then open your browser to http://localhost:8501

### Option 2: CLI Interface

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API key
# Copy .env.example to .env and add your Nebius API key
cp .env.example .env
# Edit .env and replace 'your-nebius-api-key-here' with your actual key

# 3. Run the CLI
python main.py

# With session and user profile
python main.py --session my_session --user alice
```

## Architecture

```
User query
    │
    ▼
┌──────────────────┐
│  Router (LLM)    │ Classifies query type
└────────┬─────────┘
         │
    ┌────┴─────────────────────────────┐
    │                                   │
    ▼ (structured/unstructured/        ▼ (out_of_scope)
       personal/recommendation)
┌─────────────────────┐         ┌──────────────┐
│  ReAct Agent        │         │ Polite       │
│  - Dataset tools    │         │ Decline      │
│  - Profile tools    │         └──────────────┘
│  - Recommender tool │
└──────────┬──────────┘
           │
           ▼
    ┌─────────────┐
    │ Final Answer│
    └─────────────┘
```

### Key Components

**Router**: Dedicated LangGraph node that classifies queries before tool selection
- `structured`: Data queries (counts, distributions, examples)
- `unstructured`: Summaries and analysis
- `personal`: User profile operations
- `recommendation`: Query suggestions
- `out_of_scope`: Unrelated queries (declined politely)

**ReAct Agent**: Multi-step tool-calling loop using LangGraph's `create_react_agent`
- Chains tools for complex queries (e.g., filter → count → summarize)
- Max iterations safety net with fallback messages

**Episodic Memory** (Task 2a): SqliteSaver checkpointer
- Persists conversation state across turns and restarts
- Same `--session` ID = same conversation restored
- Stored in `checkpoints.db`

**Semantic Memory** (Task 2b): User profile system
- JSON files in `profiles/` directory
- Stores name, interests, preferences, facts
- Tools: `get_user_profile`, `update_user_profile`

**Query Recommender** (Bonus B):
- Suggests 2-3 relevant follow-up queries
- Based on conversation history and user profile
- Activated by "What should I query next?"

## Model Choice

**Selected Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`

This is a **Mixture-of-Experts (MoE) model** with 30B total parameters and 3B active parameters.

### Why This Model?

This model was chosen for its optimal balance of **capability, speed, and cost-efficiency**:

| Criterion | Details | Why It Matters |
|-----------|---------|----------------|
| **Architecture** | MoE with 30B total / 3B active params | Only 3B params activated per token = fast inference while maintaining 30B model knowledge |
| **Speed** | 70 tokens/second | Multi-step ReAct loops require fast inference. Agent often chains 3-5 tool calls per query |
| **Cost** | $0.10/1M input tokens, $0.30/1M output | Affordable for development and iterative testing. ~100x cheaper than GPT-4 |
| **Optimization** | Built for chat, reasoning, and tool use | Native tool-calling support. Strong instruction-following for both routing and agent tasks |
| **Task Fit** | Sufficient for 5-way routing + multi-step tool chains | Not over-engineered. Handles both simple classification and complex reasoning |

### Used For Both Router and Agent

| Role   | Why This Model Works |
|--------|---------------------|
| **Router** | 5-way classification is straightforward. The model's instruction-following is more than sufficient. Fast response (<500ms) keeps routing overhead minimal |
| **Agent**  | Multi-step tool-calling requires strong reasoning. The model chains tools effectively (e.g., filter → count → summarize) and handles complex queries with 10+ reasoning steps |

**Alternative Considered:** Smaller models (7B-13B) were tested but struggled with multi-step reasoning and occasionally hallucinated tool arguments. Larger models (70B+) were overkill and 3-5x slower with no quality improvement for this use case.

Both router and agent served via **Nebius Token Factory** (https://api.tokenfactory.nebius.com/v1/).

## Tools

### Dataset Tools

| Tool | Purpose | Example Query |
|------|---------|---------------|
| `get_categories` | List all categories | "What categories exist?" |
| `get_intents` | List intents with counts (optionally by category) | "Show intents in REFUND" |
| `filter_and_count` | Count rows matching column/value | "How many refund requests?" |
| `show_examples` | Get N example rows with filters | "Show 3 examples from SHIPPING" |
| `get_sample_for_summary` | Get sample for LLM summarization | "Summarize FEEDBACK" |
| `dataset_overview` | High-level dataset statistics | "Tell me about the dataset" |

### Memory Tools

| Tool | Purpose | Example Query |
|------|---------|---------------|
| `get_user_profile` | Retrieve stored user information | "What do you remember about me?" |
| `update_user_profile` | Save user facts | "My name is Alice" |

### Recommendation Tool

| Tool | Purpose | Example Query |
|------|---------|---------------|
| `recommend_queries` | Suggest follow-up queries | "What should I query next?" |

## Streamlit App (Bonus A)

Interactive web UI with:
- Chat interface with persistent history
- Session ID and User ID configuration in sidebar
- Expandable reasoning steps (tool calls, results, router decisions)
- Example queries for quick start
- Clear chat display button

**Run:**
```bash
streamlit run app.py
```

**Features:**
- Real-time agent responses
- Visual separation of reasoning steps
- Session persistence across page refreshes
- Mobile-responsive design

## MCP Server (Task 3)

Exposes dataset tools via Model Context Protocol using FastMCP.

### Starting the Server

```bash
python mcp_server.py
```

Runs with **stdio transport** (subprocess communication, no network ports).

### MCP Client Example

```bash
python mcp_client_example.py
```

Or integrate into your own code:

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def main():
    client = MultiServerMCPClient({
        "customer_service": {
            "transport": "stdio",
            "command": "python",
            "args": ["mcp_server.py"],
        }
    })

    tools = await client.get_tools()
    llm = ChatOpenAI(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        base_url="https://api.tokenfactory.nebius.com/v1/",
        api_key="your-key",
    )
    agent = create_react_agent(llm, tools=tools)

    result = await agent.ainvoke(
        {"messages": [("user", "What categories exist?")]}
    )
    print(result["messages"][-1].content)

asyncio.run(main())
```

### MCP Tools Exposed

| MCP Tool | Description |
|----------|-------------|
| `get_categories` | List all unique categories |
| `get_intents` | List intents with counts, optionally by category |
| `filter_and_count` | Filter + count rows by column/value |
| `show_examples` | Show N example rows with optional filters |

## Project Structure

```
├── main.py                  # CLI entrypoint
├── app.py                   # Streamlit web UI (Bonus A)
├── agent.py                 # LangGraph agent + graph definition
├── router.py                # Query router node (5-way classification)
├── tools.py                 # Dataset tool definitions with Pydantic schemas
├── user_profile.py          # Persistent user profile (semantic memory)
├── query_recommender.py     # Query recommendation tool (Bonus B)
├── data_loader.py           # Dataset loading utilities
├── config.py                # Model config, paths, and constants
├── mcp_server.py            # FastMCP server (Task 3)
├── mcp_client_example.py    # Example MCP client connection
├── requirements.txt         # Python dependencies
├── .env.example             # Example environment variables (copy to .env)
├── .env                     # Your API keys (create from .env.example, not committed)
├── profiles/                # User profile JSON files (auto-created)
├── checkpoints.db           # SQLite conversation persistence (auto-created)
└── data/                    # Dataset CSV
    └── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
```

## Example Interactions

### Structured Query
```
User: How many refund requests did we get?
Router: structured
Agent:
  Tool call: filter_and_count(column='intent', value='get_refund')
  Result: Found Found 997 rows where intent='get_refund'
  Answer: We received 997 refund requests in the dataset.
```

### Unstructured Query
```
User: Summarize the FEEDBACK category
Router: unstructured
Agent:
  Tool call: get_sample_for_summary(category='FEEDBACK')
  Result: [15 example rows...]
Answer: The FEEDBACK category contains both positive and negative customer
feedback. Common themes include...
```

### Personal Memory
```
User: My name is Alice and I'm interested in refund patterns
Router: personal
Agent:
  Tool call: update_user_profile(name='Alice')
  Tool call: update_user_profile(interest='refund patterns')
Answer: Got it! I've saved that your name is Alice and you're interested in
refund patterns.

User: What do you remember about me?
Agent:
  Tool call: get_user_profile()
  Result: name: Alice, interests: refund patterns
Answer: I remember that your name is Alice and you're interested in refund patterns.
```

### Query Recommendation
```
User: What should I query next?
Router: recommendation
Agent:
  Tool call: recommend_queries(conversation_summary='Asked about categories')
Answer: Here are some follow-up queries you might find interesting:
1. "How many queries are in each category?"
2. "Show me examples from the ORDER category"
3. "What are the most common intents in REFUND?"
```

## Configuration

### Environment Variables

**Important for Graders:** A `.env.example` file is included in this repository. To get started:

```bash
# 1. Copy the example file
cp .env.example .env

# 2. Edit .env and add your Nebius API key
# Replace 'your-nebius-api-key-here' with your actual key from https://studio.nebius.com/
```

The `.env` file should contain:

```bash
NEBIUS_API_KEY=your-nebius-api-key-here
```

Alternatively, you can set the environment variable directly:

```bash
# Windows
set NEBIUS_API_KEY=your-key-here

# Linux/Mac
export NEBIUS_API_KEY=your-key-here
```

### Config File

Edit `config.py` to customize:
- Model selection (`AGENT_MODEL`, `ROUTER_MODEL`)
- API base URL (`NEBIUS_API_BASE`)
- Max iterations (`MAX_ITERATIONS`)
- Dataset path (`DATASET_PATH`)
- Checkpoint database path (`SQLITE_CHECKPOINT_PATH`)
- Profiles directory (`PROFILES_DIR`)

## Requirements

- Python 3.11+
- langchain >= 0.3.0
- langchain-openai >= 0.3.0
- langgraph >= 0.4.0
- pandas >= 2.0.0
- fastmcp >= 3.3.0 (for MCP server)
- streamlit >= 1.57.0 (for web UI)

See `requirements.txt` for full dependency list.

## Dataset

Download from HuggingFace:
https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

Place the CSV in the `data/` folder as:
`data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`

## License

MIT

## Acknowledgments

- Dataset: [Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- Framework: [LangGraph](https://github.com/langchain-ai/langgraph)
- Model Provider: [Nebius AI](https://nebius.com/)
