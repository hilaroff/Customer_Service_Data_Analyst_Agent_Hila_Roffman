"""
router.py — Query router node for the agent graph.

Classifies incoming queries BEFORE tool selection:
  - structured:      concrete, data-driven answers (counts, distributions, examples)
  - unstructured:    open-ended summarization or pattern analysis
  - personal:        user sharing info or asking what we remember (Task 2b)
  - recommendation:  user asking for query suggestions (Bonus B)
  - out_of_scope:    unrelated to the customer service dataset

Worth 15 pts. Must be a dedicated node in the graph, not just prompt instructions.
"""

from langchain_openai import ChatOpenAI
from config import NEBIUS_API_BASE, NEBIUS_API_KEY, ROUTER_MODEL

ROUTER_SYSTEM_PROMPT = """\
You are a query classifier for a customer service data analysis agent.
The agent has access to a customer service dataset with categories, intents,
customer instructions, and agent responses. The agent also has a user profile
memory system and a query recommendation feature.

Classify the user's query into EXACTLY one of these five types:

- **structured**: The user wants a concrete, data-driven answer.
  Examples: counts, distributions, listing categories/intents, showing example rows.
  e.g. "How many refund requests?", "What categories exist?", "Show me 3 examples from SHIPPING."

- **unstructured**: The user wants an open-ended summary, analysis, or pattern description.
  e.g. "Summarize the FEEDBACK category", "How do agents respond to complaints?"

- **personal**: The user is sharing personal information, asking what you remember,
  or making conversation about themselves. (Task 2b)
  e.g. "My name is Alice", "I work in complaints", "What do you remember about me?"

- **recommendation**: The user is asking for query suggestions or what to explore next. (Bonus B)
  e.g. "What should I query next?", "What else can I ask?", "Suggest something",
  "Yes, do it", "Go ahead", "Let's try that"

- **out_of_scope**: The query is NOT about the dataset, NOT personal info, NOT a recommendation.
  e.g. "Who won the Champions League?", "Write me a poem", "What's the weather?"

Reply with EXACTLY one word: structured, unstructured, personal, recommendation, or out_of_scope
"""


def create_router_llm() -> ChatOpenAI:
    """Create the LLM instance used for routing."""
    return ChatOpenAI(
        model=ROUTER_MODEL,
        base_url=NEBIUS_API_BASE,
        api_key=NEBIUS_API_KEY,
        temperature=0,
        max_tokens=10,
    )


def classify_query(query: str) -> str:
    """Classify a user query into one of: structured, unstructured, personal,
    recommendation, or out_of_scope.

    Args:
        query: The user's natural-language question.

    Returns:
        One of: 'structured', 'unstructured', 'personal', 'recommendation', 'out_of_scope'.
    """
    llm = create_router_llm()
    response = llm.invoke([
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ])
    classification = response.content.strip().lower()

    # Validate — fall back to structured if the model returns something unexpected
    valid = {"structured", "unstructured", "personal", "recommendation", "out_of_scope"}
    if classification not in valid:
        for v in valid:
            if v in classification:
                return v
        return "structured"  # safe default

    return classification
