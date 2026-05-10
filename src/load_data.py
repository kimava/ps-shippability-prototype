"""Data loading utilities with chunked listing reads."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import List, Optional

import pandas as pd


LISTING_COLUMNS = ["ad_id", "title", "description", "category_id", "L1", "L2", "L3", "L4"]
SIGNAL_COLUMNS = ["category_scope", "signal_term", "signal_type", "decision_impact"]
TRAINING_COLUMNS = [
    "ad_id",
    "title",
    "description",
    "category_id",
    "L1",
    "L2",
    "L3",
    "L4",
    "human_label",
]


def _validate_columns(df: pd.DataFrame, expected: List[str], file_label: str) -> None:
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"{file_label} missing required columns: {missing}")


def load_signal_library(path: Path) -> pd.DataFrame:
    signal_df = pd.read_csv(path)
    _validate_columns(signal_df, SIGNAL_COLUMNS, "signal_library.csv")
    return signal_df


def iterate_listing_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        _validate_columns(chunk, LISTING_COLUMNS, "listings.csv")
        yield chunk


def load_training_labels(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    training_df = pd.read_csv(path)
    _validate_columns(training_df, TRAINING_COLUMNS, "training_labels.csv")
    return training_df


def derive_internal_category(df: pd.DataFrame) -> pd.Series:
    return (
        df["L4"]
        .fillna("")
        .astype(str)
        .str.strip()
        .where(lambda s: s != "", df["L3"].fillna("").astype(str).str.strip())
        .where(lambda s: s != "", df["L2"].fillna("").astype(str).str.strip())
        .where(lambda s: s != "", df["L1"].fillna("").astype(str).str.strip())
    )

