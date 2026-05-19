"""
agent.py — LangGraph ReAct agent with query router and persistent memory.

Graph structure:
  START -> router_node → [structured/unstructured →-> agent_node, out_of_scope -> decline_node] -> END

Task 1: Router, tools, multi-step reasoning, max iterations fallback
Task 2a: SqliteSaver checkpointer for episodic memory (conversation persistence)
Task 2b: User profile tools for semantic memory (persistent user facts)
"""

import sqlite3
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from config import (
    NEBIUS_API_BASE, NEBIUS_API_KEY, AGENT_MODEL,
    MAX_ITERATIONS, SQLITE_CHECKPOINT_PATH,
)
from router import classify_query
from tools import ALL_TOOLS
from user_profile import create_profile_tools


# ── System prompt for the ReAct agent ─────────────────────────────────
AGENT_SYSTEM_PROMPT = """\
You are a data analyst agent for a customer service dataset.
The dataset contains customer queries (instructions) and agent responses,
organized by categories and intents.

Your job is to answer the user's questions using the available tools.
Always use tools to get data — never make up numbers or examples.

For multi-step questions, chain tools as needed. For example:
- "How many refund requests?" → use filter_and_count(column='intent', value='get_refund')
- "Distribution of intents in ACCOUNT" → use get_intents(category='ACCOUNT')
- "Summarize FEEDBACK" → use get_sample_for_summary(category='FEEDBACK'), then synthesize

You also have memory capabilities:
- You remember the full conversation history within a session.
- You can store and retrieve facts about the user using the profile tools.
- When the user shares personal info (name, preferences, interests), use
  update_user_profile to save it.
- When asked "What do you remember about me?", use get_user_profile to retrieve their info.

Be concise and helpful. Show your reasoning.
"""

OUT_OF_SCOPE_MESSAGE = (
    "I'm sorry, but that question is outside the scope of what I can help with. "
    "I'm a data analyst agent for the Bitext Customer Service dataset. "
    "I can answer questions about customer service categories, intents, "
    "example queries, and response patterns. Try asking something like:\n"
    '  • "What categories exist in the dataset?"\n'
    '  • "How many refund requests did we get?"\n'
    '  • "Summarize the FEEDBACK category."'
)

MAX_ITER_FALLBACK = (
    "I've spent several steps trying to answer your question but couldn't reach "
    "a complete answer. Could you try rephrasing, or ask a simpler question?"
)


def create_agent_llm() -> ChatOpenAI:
    """Create the main LLM instance for the ReAct agent."""
    return ChatOpenAI(
        model=AGENT_MODEL,
        base_url=NEBIUS_API_BASE,
        api_key=NEBIUS_API_KEY,
        temperature=0,
    )


def create_checkpointer() -> SqliteSaver:
    """Create a persistent SQLite checkpointer for episodic memory.

    This persists conversation state across turns AND across restarts.
    The same session/thread_id will restore the full conversation history.
    """
    conn = sqlite3.connect(SQLITE_CHECKPOINT_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return checkpointer


def build_agent_graph(checkpointer=None, user_id: str = "default"):
    """Build and compile the full agent graph.

    Args:
        checkpointer: LangGraph checkpointer for persistent memory.
                      If None, creates a SqliteSaver automatically.
        user_id: User ID for the profile system (Task 2b).

    Returns:
        A compiled LangGraph graph ready for .invoke() or .astream().
    """
    llm = create_agent_llm()

    # Combine dataset tools + user profile tools
    profile_tools = create_profile_tools(user_id)
    all_tools = ALL_TOOLS + profile_tools

    # The inner ReAct agent handles tool-calling for structured/unstructured queries
    react_agent = create_react_agent(
        llm,
        tools=all_tools,
        prompt=AGENT_SYSTEM_PROMPT,
    )

    # ── Graph nodes ───────────────────────────────────────────────────

    def router_node(state: MessagesState) -> dict:
        """Classify the latest user message and store the route."""
        # Find the last human message
        last_user_msg = None
        for msg in reversed(state["messages"]):
            if hasattr(msg, "type") and msg.type == "human":
                last_user_msg = msg.content
                break

        if not last_user_msg:
            last_user_msg = state["messages"][-1].content

        route = classify_query(last_user_msg)
        print(f"  Router: classified as '{route}'")
        from langchain_core.messages import SystemMessage
        return {"messages": [SystemMessage(content=f"__ROUTE__:{route}")]}

    def decline_node(state: MessagesState) -> dict:
        """Handle out-of-scope queries with a polite decline."""
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)]}

    def agent_node(state: MessagesState) -> dict:
        """Run the ReAct agent on the query (structured or unstructured)."""
        # Remove the route metadata messages before passing to the agent
        filtered = [m for m in state["messages"] if not (
            hasattr(m, "content") and isinstance(m.content, str)
            and m.content.startswith("__ROUTE__:")
        )]
        try:
            result = react_agent.invoke({"messages": filtered})
            return {"messages": result["messages"][len(filtered):]}
        except Exception as e:
            from langchain_core.messages import AIMessage
            if "recursion" in str(e).lower() or "iteration" in str(e).lower():
                return {"messages": [AIMessage(content=MAX_ITER_FALLBACK)]}
            return {"messages": [AIMessage(content=f"An error occurred: {e}")]}

    def route_decision(state: MessagesState) -> str:
        for msg in reversed(state["messages"]):
            if hasattr(msg, "content") and isinstance(msg.content, str) \
            and msg.content.startswith("__ROUTE__:"):
                route = msg.content.split(":", 1)[1]
                if route == "out_of_scope":
                    return "decline"
                return "agent"  # structured, unstructured, AND personal all go to agent (personal added in Task 2b)
        return "agent"

    # ── Build the graph ───────────────────────────────────────────────
    graph = StateGraph(MessagesState)

    graph.add_node("router", router_node)
    graph.add_node("agent", agent_node)
    graph.add_node("decline", decline_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_decision, {
        "agent": "agent",
        "decline": "decline",
    })
    graph.add_edge("agent", END)
    graph.add_edge("decline", END)

    # Use provided checkpointer or create a persistent one
    if checkpointer is None:
        checkpointer = create_checkpointer()

    return graph.compile(checkpointer=checkpointer)
