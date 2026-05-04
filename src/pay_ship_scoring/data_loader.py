"""Data loading utilities."""

from pathlib import Path

import pandas as pd


EXPECTED_LISTING_COLUMNS = ["listing_id", "category", "title", "description", "url"]
EXPECTED_SIGNAL_COLUMNS = ["category", "signal_term", "signal_type", "decision_impact"]


def _validate_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def load_listings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, EXPECTED_LISTING_COLUMNS, "listings.csv")
    return df


def load_signal_library(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, EXPECTED_SIGNAL_COLUMNS, "signal_library.csv")
    return df

