"""Configuration for rule-based listing scoring."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ScoringConfig:
    """Central scoring parameters, easy to tweak in one place."""

    base_score: int = 50
    signal_weights: Dict[str, int] = field(
        default_factory=lambda: {
            "strong_negative": -45,
            "strong_positive": 20,
            "bulky": -25,
            "supporting": 0,
        }
    )
    include_if_score_at_least: int = 60
    review_if_score_at_least: int = 35
    exclude_on_any_strong_negative: bool = True
    min_score: int = 0
    max_score: int = 100

