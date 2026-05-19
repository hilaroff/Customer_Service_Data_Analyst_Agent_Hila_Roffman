"""
user_profile.py — Persistent user profile for semantic memory (Task 2b).

Stores distilled facts about each user (name, preferences, interests) in a
JSON file, separate from conversation history. This is semantic memory —
curated facts promoted from episodic interactions.

The profile is NOT a replay of past messages. It captures:
  - User's name
  - Topics they frequently ask about
  - Stated preferences
  - Any other relevant personal facts

Exposed to the agent as two tools: get_user_profile and update_user_profile.
"""

import os
import json
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from config import PROFILES_DIR


def _profile_path(user_id: str) -> str:
    """Get the file path for a user's profile."""
    return os.path.join(PROFILES_DIR, f"{user_id}.json")


def load_profile(user_id: str) -> dict:
    """Load a user profile from disk. Returns empty profile if none exists."""
    path = _profile_path(user_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {
        "name": None,
        "interests": [],
        "preferences": [],
        "facts": [],
    }


def save_profile(user_id: str, profile: dict) -> None:
    """Save a user profile to disk."""
    os.makedirs(PROFILES_DIR, exist_ok=True)
    path = _profile_path(user_id)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


def create_profile_tools(user_id: str) -> list:
    """Create profile tools bound to a specific user_id.

    Returns a list of LangChain tools that the agent can use to
    read and update the user's persistent profile.
    """

    @tool
    def get_user_profile() -> str:
        """Retrieve the current user's profile — stored facts, name, interests, and preferences.

        Use this tool when the user asks:
        - "What do you remember about me?"
        - "What's my name?"
        - "What are my interests?"
        Or when you need to personalize a response based on past user info.
        """
        profile = load_profile(user_id)

        if not profile.get("name") and not profile.get("interests") \
           and not profile.get("preferences") and not profile.get("facts"):
            return "No profile information stored yet for this user."

        lines = []
        if profile.get("name"):
            lines.append(f"Name: {profile['name']}")
        if profile.get("interests"):
            lines.append(f"Interests: {', '.join(profile['interests'])}")
        if profile.get("preferences"):
            lines.append(f"Preferences: {', '.join(profile['preferences'])}")
        if profile.get("facts"):
            lines.append(f"Other facts: {', '.join(profile['facts'])}")
        return "User profile:\n" + "\n".join(lines)

    class UpdateProfileInput(BaseModel):
        name: Optional[str] = Field(
            default=None,
            description="The user's name, if they shared it."
        )
        interest: Optional[str] = Field(
            default=None,
            description="A topic the user is interested in (e.g. 'refund analysis', 'shipping data')."
        )
        preference: Optional[str] = Field(
            default=None,
            description="A stated preference (e.g. 'prefers detailed examples', 'likes charts')."
        )
        fact: Optional[str] = Field(
            default=None,
            description="Any other notable fact about the user (e.g. 'works in customer support')."
        )

    @tool(args_schema=UpdateProfileInput)
    def update_user_profile(
        name: Optional[str] = None,
        interest: Optional[str] = None,
        preference: Optional[str] = None,
        fact: Optional[str] = None,
    ) -> str:
        """Update the current user's persistent profile with new information.

        Use this tool when the user shares personal information during conversation:
        - Their name: "My name is Alice" → name='Alice'
        - An interest: "I'm mostly looking at refund data" → interest='refund data'
        - A preference: "I prefer seeing 5 examples at a time" → preference='prefers 5 examples'
        - A fact: "I work in the complaints department" → fact='works in complaints department'

        The profile persists across sessions and restarts.
        """
        profile = load_profile(user_id)

        updated = []
        if name:
            profile["name"] = name
            updated.append(f"name='{name}'")
        if interest and interest not in profile["interests"]:
            profile["interests"].append(interest)
            updated.append(f"interest='{interest}'")
        if preference and preference not in profile["preferences"]:
            profile["preferences"].append(preference)
            updated.append(f"preference='{preference}'")
        if fact and fact not in profile["facts"]:
            profile["facts"].append(fact)
            updated.append(f"fact='{fact}'")

        if updated:
            save_profile(user_id, profile)
            return f"Profile updated: {', '.join(updated)}"
        return "No new information to update."

    return [get_user_profile, update_user_profile]
