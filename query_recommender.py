"""
query_recommender.py - Bonus B: Query recommendation tool.

When the user asks "What should I query next?", this tool examines the
conversation history and user profile to suggest relevant follow-up queries.

The key behavior: suggest but DON'T execute. Let the user refine, then
only execute when they confirm.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import NEBIUS_API_BASE, NEBIUS_API_KEY, AGENT_MODEL
from user_profile import load_profile

RECOMMENDER_PROMPT = """\
You are a query recommendation assistant for a customer service dataset analyst.

Based on the user's conversation history and profile, suggest 2-3 relevant
follow-up queries they might find useful. The dataset contains customer service
data with categories (ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE,
ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION) and intents within each.

Available query types:
- Count queries: "How many [intent] requests did we get?"
- Distribution queries: "What is the distribution of intents in [CATEGORY]?"
- Example queries: "Show me N examples from [CATEGORY/intent]"
- Summary queries: "Summarize the [CATEGORY] category"
- Comparison queries: "Compare [CATEGORY1] and [CATEGORY2]"

Rules:
- Suggest queries the user has NOT already asked
- If the user has a profile with interests, lean toward those topics
- Present suggestions as numbered options, not as executed queries
- Keep suggestions specific and actionable
- Do NOT execute any tools -- just suggest

Conversation so far:
{history}

User profile:
{profile}

Suggest 2-3 follow-up queries:
"""


def create_recommender_tool(user_id: str) -> list:
    """Create the query recommender tool bound to a user_id."""

    class RecommenderInput(BaseModel):
        conversation_summary: str = Field(
            description="Brief summary of what the user has asked about so far in this conversation. "
                        "List the main topics/categories/intents discussed."
        )

    @tool(args_schema=RecommenderInput)
    def recommend_queries(conversation_summary: str) -> str:
        """Suggest 2-3 relevant follow-up queries based on the conversation history and user profile.

        Use this tool ONLY when the user asks:
        - "What should I query next?"
        - "What else can I ask?"
        - "Suggest a query"
        - Or similar requests for recommendations.

        IMPORTANT: Only SUGGEST queries. Do NOT execute them.
        Wait for the user to pick one or refine it, then execute only after confirmation.
        """
        profile = load_profile(user_id)

        profile_str = "No profile stored."
        if profile.get("name") or profile.get("interests"):
            parts = []
            if profile.get("name"):
                parts.append(f"Name: {profile['name']}")
            if profile.get("interests"):
                parts.append(f"Interests: {', '.join(profile['interests'])}")
            if profile.get("preferences"):
                parts.append(f"Preferences: {', '.join(profile['preferences'])}")
            if profile.get("facts"):
                parts.append(f"Facts: {', '.join(profile['facts'])}")
            profile_str = "\n".join(parts)

        llm = ChatOpenAI(
            model=AGENT_MODEL,
            base_url=NEBIUS_API_BASE,
            api_key=NEBIUS_API_KEY,
            temperature=0.7,
            max_tokens=300,
        )

        prompt = RECOMMENDER_PROMPT.format(
            history=conversation_summary,
            profile=profile_str,
        )

        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content

    return [recommend_queries]
