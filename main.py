"""
main.py — CLI entrypoint for the Customer Service Data Analyst Agent.

Usage:
    python main.py                          # new session with random ID
    python main.py --session my_session     # resume or start a named session
    python main.py --session my_session --user alice   # with user profile

Task 2a: The --session flag restores conversation history across restarts
         via SqliteSaver. "Show me 3 more" works because the checkpointer
         replays the full conversation to the agent on each invoke.

Task 2b: The --user flag binds the profile tools to a specific user ID.
         Profile data persists as JSON files in the profiles/ directory.
"""

import argparse
import uuid

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from agent import build_agent_graph
from config import MAX_ITERATIONS


def print_step(message) -> None:
    """Pretty-print a single message from the agent's reasoning trace."""
    if isinstance(message, SystemMessage):
        if message.content.startswith("__ROUTE__:"):
            return
        print(f"  System: {message.content}")

    elif isinstance(message, AIMessage):
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                print(f"  Tool call: {tc['name']}({args_str})")
        if message.content and not message.content.startswith("__ROUTE__:"):
            print(f"Agent: {message.content}")

    elif isinstance(message, ToolMessage):
        content = message.content
        if len(content) > 500:
            content = content[:500] + "... [truncated]"
        print(f"  Result ({message.name}): {content}")


def run_cli(session_id: str, user_id: str) -> None:
    """Run the interactive CLI conversation loop.

    The checkpointer handles all conversation state. We only send the NEW
    user message each turn — the checkpointer automatically provides the
    full history to the agent via the thread_id.
    """
    graph = build_agent_graph(user_id=user_id)

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": MAX_ITERATIONS * 2 + 5,
    }

    print("=" * 60)
    print("  Customer Service Data Analyst Agent")
    print(f"  Session: {session_id}  |  User: {user_id}")
    print("  Type 'quit' or 'exit' to end the conversation.")
    print("=" * 60)

    # Check if this session has prior history (resuming a conversation)
    try:
        state = graph.get_state(config)
        if state and state.values and state.values.get("messages"):
            n_msgs = len(state.values["messages"])
            print(f"  Restored {n_msgs} messages from previous session.")
    except Exception:
        pass  # no prior state — fresh session

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        try:
            # With checkpointer: only send the new message.
            # The checkpointer automatically appends it to the stored history
            # and provides the full conversation to the agent.
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config,
            )

            # Print only the NEW messages from this turn
            # (the last human message + everything after it)
            all_msgs = result["messages"]
            # Find where our new messages start (after the last HumanMessage we sent)
            new_start = len(all_msgs)
            for i in range(len(all_msgs) - 1, -1, -1):
                if isinstance(all_msgs[i], HumanMessage) and all_msgs[i].content == user_input:
                    new_start = i + 1
                    break

            for msg in all_msgs[new_start:]:
                print_step(msg)

        except Exception as e:
            error_str = str(e)
            if "recursion" in error_str.lower():
                print(f"\nAgent: I've hit my maximum reasoning steps. "
                      f"Could you try rephrasing your question?\n")
            else:
                print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent"
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session ID for conversation persistence. "
             "Same ID = same conversation restored, even after restart.",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="default",
        help="User ID for persistent profile (name, interests, preferences).",
    )
    args = parser.parse_args()

    session_id = args.session or str(uuid.uuid4())[:8]
    run_cli(session_id, args.user)


if __name__ == "__main__":
    main()
