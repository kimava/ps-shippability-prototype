"""Rule-based scoring and decision logic."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .config import ScoringConfig


WORD_BOUNDARY_TEMPLATE = r"(?<!\w){term}(?!\w)"


@dataclass(frozen=True)
class MatchedSignal:
    signal_term: str
    signal_type: str
    decision_impact: str
    weight: int

    def as_text(self) -> str:
        return f"{self.signal_term} ({self.signal_type}, {self.decision_impact}, {self.weight:+d})"


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def _build_regex(term: str) -> re.Pattern[str]:
    escaped = re.escape(str(term).strip().lower())
    return re.compile(WORD_BOUNDARY_TEMPLATE.format(term=escaped))


def _match_signals(
    listing_text: str,
    listing_category: str,
    signal_library: pd.DataFrame,
    config: ScoringConfig,
) -> list[MatchedSignal]:
    category_signals = signal_library[signal_library["category"] == listing_category]
    normalized_text = _normalize_text(listing_text)
    matches: list[MatchedSignal] = []

    for signal in category_signals.itertuples(index=False):
        pattern = _build_regex(signal.signal_term)
        if not pattern.search(normalized_text):
            continue

        signal_type = str(signal.signal_type).strip().lower()
        weight = config.signal_weights.get(signal_type, 0)
        matches.append(
            MatchedSignal(
                signal_term=str(signal.signal_term),
                signal_type=signal_type,
                decision_impact=str(signal.decision_impact),
                weight=weight,
            )
        )

    return matches


def _derive_decision(score: int, matches: list[MatchedSignal], config: ScoringConfig) -> tuple[str, str]:
    has_strong_negative = any(match.signal_type == "strong_negative" for match in matches)

    if config.exclude_on_any_strong_negative and has_strong_negative:
        reason = "Strong negative signal found."
        return "Exclude", reason

    if score >= config.include_if_score_at_least:
        return "Eligible", "Score meets eligibility threshold."
    if score >= config.review_if_score_at_least:
        return "Review", "Score falls into manual review range."
    return "Exclude", "Score below review threshold."


def score_listings(listings_df: pd.DataFrame, signal_library_df: pd.DataFrame, config: ScoringConfig) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []

    for listing in listings_df.itertuples(index=False):
        listing_text = f"{listing.title} {listing.description}"
        matches = _match_signals(
            listing_text=listing_text,
            listing_category=str(listing.category),
            signal_library=signal_library_df,
            config=config,
        )

        raw_score = config.base_score + sum(match.weight for match in matches)
        score = max(config.min_score, min(config.max_score, raw_score))
        final_decision, reason = _derive_decision(score, matches, config)

        matched_signal_text = "; ".join(match.as_text() for match in matches) if matches else "None"
        reason_detail = f"{reason} Matches: {matched_signal_text} | Score={score}."

        rows.append(
            {
                "listing_id": listing.listing_id,
                "category": listing.category,
                "title": listing.title,
                "description": listing.description,
                "matched_signals": matched_signal_text,
                "rule_score": score,
                "final_decision": final_decision,
                "reason": reason_detail,
            }
        )

    return pd.DataFrame(rows)

