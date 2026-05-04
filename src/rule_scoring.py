"""Deterministic rule-based scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .config import AppConfig

WORD_BOUNDARY_TEMPLATE = r"(?<!\w){term}(?!\w)"


@dataclass(frozen=True)
class SignalRule:
    category: str
    signal_term: str
    signal_type: str
    decision_impact: str
    weight: int
    pattern: re.Pattern[str]

    def as_text(self) -> str:
        return f"{self.signal_term} ({self.signal_type}, {self.decision_impact}, {self.weight:+d})"


class RuleScorer:
    """Reusable rule scorer with precompiled category-specific signals."""

    def __init__(self, signal_library_df: pd.DataFrame, config: AppConfig):
        self.config = config
        self.signals_by_category = self._compile_signal_rules(signal_library_df, config)

    @staticmethod
    def _compile_signal_rules(signal_library_df: pd.DataFrame, config: AppConfig) -> dict[str, list[SignalRule]]:
        grouped: dict[str, list[SignalRule]] = {}

        for signal in signal_library_df.itertuples(index=False):
            category = str(signal.category)
            signal_term = str(signal.signal_term).strip()
            signal_type = str(signal.signal_type).strip().lower()
            escaped_term = re.escape(signal_term.lower())
            compiled_pattern = re.compile(WORD_BOUNDARY_TEMPLATE.format(term=escaped_term))
            weight = config.signal_weights.get(signal_type, 0)

            grouped.setdefault(category, []).append(
                SignalRule(
                    category=category,
                    signal_term=signal_term,
                    signal_type=signal_type,
                    decision_impact=str(signal.decision_impact),
                    weight=weight,
                    pattern=compiled_pattern,
                )
            )

        return grouped

    @staticmethod
    def _normalize_text_columns(df: pd.DataFrame) -> pd.Series:
        text_series = (
            df["title"].fillna("").astype(str) + " " + df["description"].fillna("").astype(str)
        )
        return text_series.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    def score_chunk(self, listings_chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = listings_chunk.copy()
        chunk["category"] = chunk["category"].fillna("").astype(str)
        chunk["title"] = chunk["title"].fillna("").astype(str)
        chunk["description"] = chunk["description"].fillna("").astype(str)
        chunk["url"] = chunk["url"].fillna("").astype(str)
        chunk["combined_text"] = self._normalize_text_columns(chunk)

        # These are chunk-local accumulators to keep memory bounded.
        rule_scores = pd.Series(self.config.base_score, index=chunk.index, dtype="int64")
        matched_signals: dict[int, list[str]] = {idx: [] for idx in chunk.index}
        strong_negative_hit = pd.Series(False, index=chunk.index, dtype="bool")

        for category, rules in self.signals_by_category.items():
            category_mask = chunk["category"] == category
            if not category_mask.any():
                continue

            category_text = chunk.loc[category_mask, "combined_text"]
            for rule in rules:
                matched_mask = category_text.str.contains(rule.pattern, regex=True, na=False)
                if not matched_mask.any():
                    continue

                matched_index = category_text.index[matched_mask]
                rule_scores.loc[matched_index] = rule_scores.loc[matched_index] + rule.weight

                if rule.signal_type == "strong_negative":
                    strong_negative_hit.loc[matched_index] = True

                signal_text = rule.as_text()
                for row_index in matched_index:
                    matched_signals[row_index].append(signal_text)

        chunk["matched_signals"] = chunk.index.map(
            lambda idx: "; ".join(matched_signals[idx]) if matched_signals[idx] else "None"
        )
        chunk["rule_score"] = rule_scores.clip(self.config.min_score, self.config.max_score)
        chunk["rule_reason"] = "Score-based rule decision."

        if self.config.exclude_on_any_strong_negative:
            chunk.loc[strong_negative_hit, "rule_reason"] = "Strong negative signal found."

        chunk["rule_decision"] = "Exclude"
        no_strong_negative_mask = ~strong_negative_hit if self.config.exclude_on_any_strong_negative else pd.Series(
            True, index=chunk.index, dtype="bool"
        )

        eligible_mask = no_strong_negative_mask & (chunk["rule_score"] >= self.config.include_if_score_at_least)
        review_mask = (
            no_strong_negative_mask
            & (chunk["rule_score"] >= self.config.review_if_score_at_least)
            & ~eligible_mask
        )

        chunk.loc[eligible_mask, "rule_decision"] = "Eligible"
        chunk.loc[review_mask, "rule_decision"] = "Review"
        chunk.loc[eligible_mask, "rule_reason"] = "Score meets eligibility threshold."
        chunk.loc[review_mask, "rule_reason"] = "Score falls into manual review range."
        chunk.loc[~eligible_mask & ~review_mask & no_strong_negative_mask, "rule_reason"] = (
            "Score below review threshold."
        )

        return chunk

