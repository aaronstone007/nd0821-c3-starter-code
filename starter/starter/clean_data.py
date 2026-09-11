"""Data-cleaning utilities for the Census income dataset.

Loads the raw ``census.csv``, strips leading/trailing whitespace from column
names and from every string cell, then writes the result to a distinct file
(``census_clean.csv``) so the raw data is never overwritten.
"""
import os

import pandas as pd

# Resolve paths relative to this module so the routine works regardless of the
# current working directory (local runs, CI, Heroku).
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_MODULE_DIR, "..", "data"))
RAW_PATH = os.path.join(_DATA_DIR, "census.csv")
CLEAN_PATH = os.path.join(_DATA_DIR, "census_clean.csv")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with whitespace stripped from headers and cells.

    Column names have leading/trailing spaces removed, and every string
    (object-dtype) value is stripped as well. Non-string columns are left
    untouched.
    """
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].str.strip()

    return cleaned


def run(raw_path: str = RAW_PATH, clean_path: str = CLEAN_PATH) -> pd.DataFrame:
    """Load the raw CSV, clean it, and save it to ``clean_path``.

    The raw file at ``raw_path`` is only read, never modified. Returns the
    cleaned DataFrame.
    """
    df = pd.read_csv(raw_path, skipinitialspace=True)
    cleaned = clean_data(df)
    cleaned.to_csv(clean_path, index=False)
    return cleaned


if __name__ == "__main__":
    result = run()
    print(f"Wrote cleaned data to {CLEAN_PATH} "
          f"({result.shape[0]} rows, {result.shape[1]} columns).")
    print("Columns:", list(result.columns))
