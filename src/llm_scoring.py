"""Mock LLM scoring layer for contextual judgement."""

from __future__ import annotations

import pandas as pd


SHIPPABLE_TERMS = [
    "small",
    "lightweight",
    "boxed",
    "can post",
    "postage available",
    "willing to post",
    "compact",
    "clothing",
]
NOT_SHIPPABLE_TERMS = [
    "collection only",
    "pickup only",
    "furniture",
    "sofa",
    "wardrobe",
    "table",
    "large",
    "heavy",
]


def _normalized_text(df: pd.DataFrame) -> pd.Series:
    return (df["title"].fillna("").astype(str) + " " + df["description"].fillna("").astype(str)).str.lower()


def apply_mock_llm_scoring(scored_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Apply heuristic LLM-style scoring only to rows with rule_decision == Review.
    Non-review rows are intentionally left as skipped.
    """
    chunk = scored_chunk.copy()
    chunk["llm_label"] = "Skipped"
    chunk["llm_confidence"] = 0.0
    chunk["llm_reason"] = "LLM scoring skipped due to deterministic rule decision."

    review_mask = chunk["rule_decision"] == "Review"
    if not review_mask.any():
        return chunk

    review_df = chunk.loc[review_mask].copy()
    text = _normalized_text(review_df)
    shippable_hits = text.apply(lambda value: sum(term in value for term in SHIPPABLE_TERMS))
    not_shippable_hits = text.apply(lambda value: sum(term in value for term in NOT_SHIPPABLE_TERMS))

    review_df["llm_label"] = "Uncertain"
    review_df["llm_confidence"] = 0.55
    review_df["llm_reason"] = "Mixed or limited shipping evidence."

    shippable_mask = shippable_hits > not_shippable_hits
    not_shippable_mask = not_shippable_hits > shippable_hits

    review_df.loc[shippable_mask, "llm_label"] = "Shippable"
    review_df.loc[shippable_mask, "llm_confidence"] = (
        0.65 + (shippable_hits[shippable_mask].clip(upper=4) * 0.08)
    ).clip(upper=0.95)
    review_df.loc[shippable_mask, "llm_reason"] = "Likely parcel-friendly based on item/shipping descriptors."

    review_df.loc[not_shippable_mask, "llm_label"] = "Not Shippable"
    review_df.loc[not_shippable_mask, "llm_confidence"] = (
        0.65 + (not_shippable_hits[not_shippable_mask].clip(upper=4) * 0.08)
    ).clip(upper=0.95)
    review_df.loc[not_shippable_mask, "llm_reason"] = "Likely collection-heavy or bulky context."

    chunk.loc[review_mask, ["llm_label", "llm_confidence", "llm_reason"]] = review_df[
        ["llm_label", "llm_confidence", "llm_reason"]
    ]
    return chunk

