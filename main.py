"""Hybrid scoring pipeline runner.

Run with:
    python main.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_app_config
from src.decision_combiner import combine_decisions
from src.load_data import iterate_listing_chunks, load_signal_library, load_training_labels
from src.ml_scoring import train_ml_model
from src.rule_scoring import RuleScorer

RULE_OUTPUT_COLUMNS = [
    "listing_id",
    "category",
    "title",
    "matched_signals",
    "rule_score",
    "rule_decision",
    "rule_reason",
    "url",
]

REVIEW_OUTPUT_COLUMNS = [
    "listing_id",
    "category",
    "title",
    "description",
    "matched_signals",
    "rule_score",
    "rule_decision",
    "rule_reason",
    "url",
]

HYBRID_OUTPUT_COLUMNS = [
    "listing_id",
    "category",
    "title",
    "matched_signals",
    "rule_score",
    "rule_decision",
    "rule_reason",
    "ml_label",
    "ml_confidence",
    "ml_reason",
    "final_decision",
    "final_reason",
    "url",
]


def _write_chunk(df: pd.DataFrame, output_path: Path, write_header: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, mode="w" if write_header else "a", header=write_header, index=False)


def _reset_output_files(paths: list[Path]) -> None:
    for output_path in paths:
        if output_path.exists():
            output_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid Pay & Ship listing scoring.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. Defaults to ./config.json when omitted.",
    )
    return parser.parse_args()


def run(config_path: str | None = None) -> None:
    config = load_app_config(Path(config_path) if config_path else None)
    signal_library = load_signal_library(config.signal_library_path)
    training_labels = load_training_labels(config.training_labels_path)
    ml_engine = train_ml_model(training_labels, config)
    rule_scorer = RuleScorer(signal_library, config)

    _reset_output_files(
        [
            config.rule_output_path,
            config.review_candidates_output_path,
            config.hybrid_output_path,
        ]
    )

    total_rows = 0
    first_rule_chunk = True
    first_review_chunk = True
    first_hybrid_chunk = True

    for listings_chunk in iterate_listing_chunks(config.listings_path, config.chunk_size):
        rule_scored = rule_scorer.score_chunk(listings_chunk)
        _write_chunk(rule_scored[RULE_OUTPUT_COLUMNS], config.rule_output_path, write_header=first_rule_chunk)
        first_rule_chunk = False

        review_chunk = rule_scored[rule_scored["rule_decision"] == "Review"]
        if not review_chunk.empty:
            _write_chunk(
                review_chunk[REVIEW_OUTPUT_COLUMNS],
                config.review_candidates_output_path,
                write_header=first_review_chunk,
            )
            first_review_chunk = False

        ml_scored = ml_engine.predict_review_candidates(rule_scored)
        final_chunk = combine_decisions(ml_scored, config)
        _write_chunk(
            final_chunk[HYBRID_OUTPUT_COLUMNS],
            config.hybrid_output_path,
            write_header=first_hybrid_chunk,
        )
        first_hybrid_chunk = False

        total_rows += len(final_chunk)

    # Ensure review output exists even when there are no review candidates.
    if first_review_chunk:
        pd.DataFrame(columns=REVIEW_OUTPUT_COLUMNS).to_csv(config.review_candidates_output_path, index=False)

    print(f"Processed {total_rows} listings")
    print(f"Rule output -> {config.rule_output_path}")
    print(f"Review candidates -> {config.review_candidates_output_path}")
    print(f"Hybrid output -> {config.hybrid_output_path}")


if __name__ == "__main__":
    args = _parse_args()
    run(config_path=args.config)

