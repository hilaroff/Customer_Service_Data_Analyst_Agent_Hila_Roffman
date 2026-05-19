"""
data_loader.py — Load the Bitext Customer Service dataset from a local CSV.
"""
import pandas as pd
from config import DATASET_PATH


def load_dataset() -> pd.DataFrame:
    """Load the Bitext dataset from the local CSV file."""
    return pd.read_csv(DATASET_PATH)


if __name__ == "__main__":
    df = load_dataset()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nCategories ({df['category'].nunique()}):")
    print(df["category"].value_counts().to_string())
    print(f"\nSample intents: {df['intent'].unique()[:10].tolist()}")