"""Combine deterministic rule outcomes with ML predictions."""

from __future__ import annotations

import pandas as pd

from .config import AppConfig


def combine_decisions(scored_chunk: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    chunk = scored_chunk.copy()
    chunk["final_decision"] = chunk["rule_decision"]
    chunk["final_reason"] = chunk["rule_reason"]

    review_mask = chunk["rule_decision"] == "Review"
    if not review_mask.any():
        return chunk

    ml_shippable_mask = (
        review_mask
        & (chunk["ml_label"] == "Shippable")
        & (chunk["ml_confidence"] >= config.ml_confidence_threshold)
    )
    ml_not_shippable_mask = (
        review_mask
        & (chunk["ml_label"] == "Not Shippable")
        & (chunk["ml_confidence"] >= config.ml_confidence_threshold)
    )

    chunk.loc[ml_shippable_mask, "final_decision"] = "Eligible"
    chunk.loc[ml_not_shippable_mask, "final_decision"] = "Exclude"

    confidence_text = chunk["ml_confidence"].astype(float).round(2).map(lambda value: f"{value:.2f}")
    chunk["final_reason"] = (
        "Rule: "
        + chunk["rule_reason"].astype(str)
        + " | ML: "
        + chunk["ml_label"].astype(str)
        + " ("
        + confidence_text
        + ") - "
        + chunk["ml_reason"].astype(str)
    )
    return chunk

