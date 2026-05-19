"""
app.py — Streamlit chat UI for the Customer Service Data Analyst Agent.

Bonus A (+10 pts):
  1. Chat interface with user input and agent responses
  2. Displays reasoning steps (tool calls and results), not just final answer
  3. Session ID input in sidebar to switch/resume conversations

Usage:
    streamlit run app.py
"""

import streamlit as st
import sqlite3
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from config import SQLITE_CHECKPOINT_PATH, MAX_ITERATIONS
from agent import build_agent_graph

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="CS Data Analyst Agent",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")

    session_id = st.text_input(
        "Session ID",
        value="streamlit_default",
        help="Same session ID = same conversation restored, even after restart.",
    )

    user_id = st.text_input(
        "User ID",
        value="default",
        help="User ID for persistent profile (name, interests).",
    )

    if st.button("Clear chat display"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Try asking:")
    st.markdown("""
    - What categories exist?
    - How many refund requests?
    - Show 3 examples from SHIPPING
    - Summarize the FEEDBACK category
    - My name is Alice
    - What do you remember about me?
    - What should I query next?
    """)

# ── Initialize state ──────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_session" not in st.session_state:
    st.session_state.current_session = session_id

# Reset chat display if session changed
if st.session_state.current_session != session_id:
    st.session_state.chat_history = []
    st.session_state.current_session = session_id


@st.cache_resource
def get_graph(_session_id: str, _user_id: str):
    """Create and cache the agent graph with a persistent checkpointer."""
    conn = sqlite3.connect(SQLITE_CHECKPOINT_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return build_agent_graph(checkpointer=checkpointer, user_id=_user_id)


def extract_reasoning_steps(messages: list) -> list[dict]:
    """Extract reasoning steps from agent messages for display."""
    steps = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                    steps.append({
                        "type": "tool_call",
                        "content": f"Tool call: **{tc['name']}**({args_str})",
                    })
            if msg.content and not msg.content.startswith("__ROUTE__:"):
                steps.append({
                    "type": "answer",
                    "content": msg.content,
                })
        elif isinstance(msg, ToolMessage):
            content = msg.content
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            steps.append({
                "type": "tool_result",
                "content": f"Result ({msg.name}): {content}",
            })
        elif isinstance(msg, SystemMessage):
            if msg.content.startswith("__ROUTE__:"):
                route = msg.content.split(":", 1)[1]
                steps.append({
                    "type": "router",
                    "content": f"Router: **{route}**",
                })
    return steps


# ── Main chat area ────────────────────────────────────────────────────
st.title("Customer Service Data Analyst Agent")
st.caption(f"Session: `{session_id}` | User: `{user_id}`")

# Display chat history
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        with st.chat_message("user"):
            st.write(entry["content"])
    elif entry["role"] == "assistant":
        with st.chat_message("assistant"):
            if entry.get("steps"):
                with st.expander("Reasoning steps", expanded=False):
                    for step in entry["steps"]:
                        if step["type"] == "router":
                            st.info(step["content"])
                        elif step["type"] == "tool_call":
                            st.warning(step["content"])
                        elif step["type"] == "tool_result":
                            st.code(step["content"], language=None)
            st.write(entry["content"])

# ── User input ────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a question about the customer service dataset..."):

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                graph = get_graph(session_id, user_id)
                config = {
                    "configurable": {"thread_id": session_id},
                    "recursion_limit": MAX_ITERATIONS * 2 + 5,
                }

                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config,
                )

                # Find new messages from this turn
                all_msgs = result["messages"]
                new_start = len(all_msgs)
                for i in range(len(all_msgs) - 1, -1, -1):
                    if isinstance(all_msgs[i], HumanMessage) and all_msgs[i].content == user_input:
                        new_start = i + 1
                        break
                new_msgs = all_msgs[new_start:]

                steps = extract_reasoning_steps(new_msgs)

                final_answer = ""
                for step in reversed(steps):
                    if step["type"] == "answer":
                        final_answer = step["content"]
                        break

                if not final_answer:
                    final_answer = "I wasn't able to generate a response."

                reasoning_steps = [s for s in steps if s["type"] != "answer"]
                if reasoning_steps:
                    with st.expander("Reasoning steps", expanded=False):
                        for step in reasoning_steps:
                            if step["type"] == "router":
                                st.info(step["content"])
                            elif step["type"] == "tool_call":
                                st.warning(step["content"])
                            elif step["type"] == "tool_result":
                                st.code(step["content"], language=None)

                st.write(final_answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_answer,
                    "steps": reasoning_steps,
                })

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "steps": [],
                })
