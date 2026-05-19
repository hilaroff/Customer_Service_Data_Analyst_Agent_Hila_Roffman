"""
mcp_server.py — FastMCP server exposing customer service dataset tools.

Requires: pip install fastmcp

Exposes 4 dataset tools via the Model Context Protocol (MCP).

Usage:
    # Start the server (stdio transport)
    python mcp_server.py

    # Or use fastmcp CLI:
    fastmcp run mcp_server.py
"""

import os
import sys
import pandas as pd
from fastmcp import FastMCP

# ── Load dataset ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "data", "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

# Fallback: check root-level CSV
if not os.path.exists(DATASET_PATH):
    alt = os.path.join(SCRIPT_DIR, "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
    if os.path.exists(alt):
        DATASET_PATH = alt
    else:
        print(f"ERROR: Dataset not found at {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

df = pd.read_csv(DATASET_PATH)

# ── Create MCP server ─────────────────────────────────────────────────
mcp = FastMCP("customer-service-analyst")


@mcp.tool
def get_categories() -> list[str]:
    """List all unique categories in the customer service dataset.

    Use this when you need to know what high-level categories exist.
    Returns a sorted list of category names (e.g. ACCOUNT, REFUND, SHIPPING).
    """
    return sorted(df["category"].unique().tolist())


@mcp.tool
def get_intents(category: str = None) -> dict[str, int]:
    """List unique intents with their row counts, optionally filtered by category.

    Args:
        category: Filter by this category (e.g. 'REFUND'). Case-insensitive.
                  If omitted, returns all intents across all categories.

    Returns:
        Dictionary mapping intent names to their row counts.
    """
    data = df
    if category:
        data = data[data["category"].str.upper() == category.upper()]
        if data.empty:
            return {"error": f"No data found for category '{category}'"}
    return data["intent"].value_counts().to_dict()


@mcp.tool
def filter_and_count(column: str, value: str) -> dict:
    """Filter the dataset by a column value and return the count.

    Args:
        column: Column to filter on. Must be 'category', 'intent', or 'flags'.
        value: Value to match (case-insensitive).

    Returns:
        Dictionary with column, value, and count.
    """
    valid_columns = ["category", "intent", "flags"]
    if column not in valid_columns:
        return {"error": f"Invalid column '{column}'. Must be one of: {valid_columns}"}
    mask = df[column].str.upper() == value.upper()
    return {"column": column, "value": value, "count": int(mask.sum())}


@mcp.tool
def show_examples(n: int = 3, category: str = None, intent: str = None) -> list[dict]:
    """Show N example rows from the dataset, optionally filtered.

    Args:
        n: Number of examples to return (1-10, default 3).
        category: Filter by category (e.g. 'SHIPPING'). Optional.
        intent: Filter by intent (e.g. 'get_refund'). Optional.

    Returns:
        List of dicts with keys: category, intent, instruction, response.
    """
    n = max(1, min(n, 10))
    data = df
    if category:
        data = data[data["category"].str.upper() == category.upper()]
    if intent:
        data = data[data["intent"].str.lower() == intent.lower()]
    if data.empty:
        return [{"error": "No matching rows found"}]
    sample = data.sample(n=min(n, len(data)))
    return sample[["category", "intent", "instruction", "response"]].to_dict(orient="records")


if __name__ == "__main__":
    mcp.run()
