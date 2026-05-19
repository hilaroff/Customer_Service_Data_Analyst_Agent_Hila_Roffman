"""
tools.py — Agent tools for querying the Bitext Customer Service dataset.

Each tool has:
  1. A Pydantic input schema (BaseModel) with Field descriptions
  2. A clear docstring - this is the LLM's ONLY documentation for the tool
  3. Typed return value

Design philosophy (from course): "A few well-designed tools beat many poorly
described ones." Tool descriptions are as important as the tool logic.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import pandas as pd

from data_loader import load_dataset

# ── Load the dataset once at module level ─────────────────────────────
_df: pd.DataFrame = load_dataset()


# ══════════════════════════════════════════════════════════════════════
# Tool 1: List categories
# ══════════════════════════════════════════════════════════════════════

@tool
def get_categories() -> str:
    """List all unique categories in the customer service dataset.

    Use this tool when the user asks what categories exist, or when you need
    to validate a category name before filtering.
    Returns a comma-separated list of all category names.
    """
    categories = sorted(_df["category"].unique().tolist())
    return f"The dataset contains {len(categories)} categories: {', '.join(categories)}"


# ══════════════════════════════════════════════════════════════════════
# Tool 2: List intents (optionally filtered by category)
# ══════════════════════════════════════════════════════════════════════

class GetIntentsInput(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Category name to filter by (e.g. 'ORDER', 'REFUND'). "
                    "If omitted, returns all intents across all categories."
    )

@tool(args_schema=GetIntentsInput)
def get_intents(category: Optional[str] = None) -> str:
    """List unique intents in the dataset, optionally filtered by category.

    Use this tool when the user asks about available intents, or asks for
    the distribution of intents within a specific category.
    Returns each intent with its row count.
    """
    df = _df
    if category:
        df = df[df["category"].str.upper() == category.upper()]
        if df.empty:
            return f"No data found for category '{category}'. Use get_categories to see valid names."

    counts = df["intent"].value_counts().sort_index()
    lines = [f"  {intent}: {count} rows" for intent, count in counts.items()]
    header = f"Intents in category '{category.upper()}'" if category else "All intents"
    return f"{header} ({len(counts)} intents):\n" + "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Tool 3: Filter and count rows
# ══════════════════════════════════════════════════════════════════════

class FilterCountInput(BaseModel):
    column: str = Field(
        description="Column to filter on. Must be one of: 'category', 'intent', 'flags'."
    )
    value: str = Field(
        description="Value to match (case-insensitive). "
                    "For example: column='intent', value='get_refund'."
    )

@tool(args_schema=FilterCountInput)
def filter_and_count(column: str, value: str) -> str:
    """Filter the dataset by a column value and return the count of matching rows.

    Use this when the user asks 'how many' questions like:
    - 'How many refund requests did we get?' → column='intent', value='get_refund'
    - 'How many rows are in the SHIPPING category?' → column='category', value='SHIPPING'

    Returns the count and a brief summary.
    """
    valid_columns = ["category", "intent", "flags"]
    if column not in valid_columns:
        return f"Invalid column '{column}'. Must be one of: {valid_columns}"

    mask = _df[column].str.upper() == value.upper()
    count = mask.sum()

    if count == 0:
        return (f"No rows found where {column}='{value}'. "
                f"Check spelling or use get_categories / get_intents to find valid values.")

    return f"Found {count:,} rows where {column}='{value}'."


# ══════════════════════════════════════════════════════════════════════
# Tool 4: Show example rows
# ══════════════════════════════════════════════════════════════════════

class ShowExamplesInput(BaseModel):
    n: int = Field(
        default=3,
        description="Number of example rows to return (1-10)."
    )
    category: Optional[str] = Field(
        default=None,
        description="Filter by category (e.g. 'SHIPPING'). Optional."
    )
    intent: Optional[str] = Field(
        default=None,
        description="Filter by intent (e.g. 'get_refund'). Optional."
    )

@tool(args_schema=ShowExamplesInput)
def show_examples(n: int = 3, category: Optional[str] = None, intent: Optional[str] = None) -> str:
    """Show N example rows from the dataset, optionally filtered by category and/or intent.

    Use this tool when the user asks to see examples, sample entries, or specific rows.
    Examples:
    - 'Show me 3 examples from SHIPPING' → n=3, category='SHIPPING'
    - 'Show examples of get_refund intent' → n=3, intent='get_refund'

    Returns the customer instruction and the agent response for each example.
    """
    n = max(1, min(n, 10))  # clamp to 1-10
    df = _df

    if category:
        df = df[df["category"].str.upper() == category.upper()]
    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]

    if df.empty:
        return "No matching rows found. Check the category/intent spelling."

    sample = df.sample(n=min(n, len(df)), random_state=None)
    lines = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        lines.append(
            f"--- Example {i} ---\n"
            f"  Category: {row['category']}\n"
            f"  Intent:   {row['intent']}\n"
            f"  Customer: {row['instruction']}\n"
            f"  Agent:    {row['response']}\n"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Tool 5: Summarize responses (for unstructured / open-ended queries)
# ══════════════════════════════════════════════════════════════════════

class SummarizeInput(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Category to summarize (e.g. 'FEEDBACK'). Optional."
    )
    intent: Optional[str] = Field(
        default=None,
        description="Intent to summarize (e.g. 'cancel_order'). Optional."
    )

@tool(args_schema=SummarizeInput)
def get_sample_for_summary(category: Optional[str] = None, intent: Optional[str] = None) -> str:
    """Retrieve a representative sample of customer instructions and agent responses
    for summarization or open-ended analysis.

    Use this tool for questions like:
    - 'Summarize the FEEDBACK category' → category='FEEDBACK'
    - 'How do agents respond to cancellation requests?' → intent='cancel_order'

    Returns up to 15 representative examples. The LLM should then synthesize
    these into a coherent summary for the user.
    """
    df = _df
    if category:
        df = df[df["category"].str.upper() == category.upper()]
    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]

    if df.empty:
        return "No matching rows found. Check spelling with get_categories or get_intents."

    sample = df.sample(n=min(15, len(df)), random_state=42)
    lines = []
    for _, row in sample.iterrows():
        lines.append(f"[{row['intent']}] Customer: {row['instruction']}")
        lines.append(f"  Agent: {row['response']}\n")

    header = f"Sample of {len(sample)} entries"
    if category:
        header += f" from category '{category.upper()}'"
    if intent:
        header += f" with intent '{intent}'"
    return header + ":\n\n" + "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Tool 6: Dataset overview / stats
# ══════════════════════════════════════════════════════════════════════

@tool
def dataset_overview() -> str:
    """Get a high-level overview of the entire dataset: total rows, number of
    categories, number of intents, and top-level distribution.

    Use this when the user asks general questions about the dataset size,
    structure, or overall statistics.
    """
    total = len(_df)
    n_cats = _df["category"].nunique()
    n_intents = _df["intent"].nunique()
    cat_dist = _df["category"].value_counts()

    lines = [
        f"Dataset overview:",
        f"  Total rows:   {total:,}",
        f"  Categories:   {n_cats}",
        f"  Intents:      {n_intents}",
        f"",
        f"Rows per category:",
    ]
    for cat, count in cat_dist.items():
        lines.append(f"  {cat}: {count:,}")

    return "\n".join(lines)


# ── Collect all tools for the agent ───────────────────────────────────
ALL_TOOLS = [
    get_categories,
    get_intents,
    filter_and_count,
    show_examples,
    get_sample_for_summary,
    dataset_overview,
]
