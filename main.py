"""Hybrid scoring pipeline runner.

Run with:
    python main.py discover
    python main.py score
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config import AppConfig, load_app_config
from src.decision_combiner import combine_decisions
from src.load_data import iterate_listing_chunks, load_signal_library, load_training_labels
from src.ml_scoring import train_ml_model
from src.rule_scoring import RuleScorer
from src.top_terms import extract_top_terms_by_l2

BASE_OUTPUT_COLUMNS = [
    "ad_id",
    "title",
    "description",
    "category_id",
    "L1",
    "L2",
    "L3",
    "L4",
]

RULE_OUTPUT_COLUMNS = [
    *BASE_OUTPUT_COLUMNS,
    "matched_signals",
    "rule_score",
    "rule_decision",
    "rule_reason",
]

REVIEW_OUTPUT_COLUMNS = [
    *RULE_OUTPUT_COLUMNS,
]

HYBRID_OUTPUT_COLUMNS = [
    *BASE_OUTPUT_COLUMNS,
    "matched_signals",
    "rule_score",
    "rule_decision",
    "rule_reason",
    "ml_label",
    "ml_confidence",
    "ml_reason",
    "final_decision",
    "final_reason",
]

HYBRID_LITE_OUTPUT_COLUMNS = [
    "ad_id",
    "category_id",
    "L1",
    "L2",
    "L3",
    "L4",
    "matched_signals",
    "rule_score",
    "rule_decision",
    "ml_label",
    "ml_confidence",
    "final_decision",
]

ML_REVIEW_LABELLING_COLUMNS = [
    "ad_id",
    "title",
    "description",
    "category_id",
    "L1",
    "L2",
    "L3",
    "L4",
    "matched_signals",
    "rule_score",
    "rule_decision",
    "ml_label",
    "ml_confidence",
    "final_decision",
    "human_label",
]


def _write_chunk(df: pd.DataFrame, output_path: Path, write_header: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
    )


def _reset_output_files(paths: List[Path]) -> None:
    for output_path in paths:
        if output_path.exists():
            output_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pay & Ship listing scoring.")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["discover", "score"],
        help="discover: rule-only + term extraction, score: hybrid rule+ML scoring.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. Defaults to ./config.json when omitted.",
    )
    return parser.parse_args()


def _run_discover(config: AppConfig, rule_scorer: RuleScorer) -> None:
    _reset_output_files(
        [
            config.rule_output_path,
            config.review_candidates_output_path,
            config.l2_top_terms_output_path,
        ]
    )

    total_rows = 0
    first_rule_chunk = True
    first_review_chunk = True
    all_review_chunks: List[pd.DataFrame] = []

    for listings_chunk in iterate_listing_chunks(config.input_listings_path, config.chunk_size):
        rule_scored = rule_scorer.score_chunk(listings_chunk)
        _write_chunk(rule_scored[RULE_OUTPUT_COLUMNS], config.rule_output_path, write_header=first_rule_chunk)
        first_rule_chunk = False

        review_chunk = rule_scored[rule_scored["rule_decision"] == "Review"]
        all_review_chunks.append(review_chunk[RULE_OUTPUT_COLUMNS].copy())
        if not review_chunk.empty:
            _write_chunk(
                review_chunk[REVIEW_OUTPUT_COLUMNS],
                config.review_candidates_output_path,
                write_header=first_review_chunk,
            )
            first_review_chunk = False

        total_rows += len(rule_scored)

    if first_review_chunk:
        pd.DataFrame(columns=REVIEW_OUTPUT_COLUMNS).to_csv(
            config.review_candidates_output_path,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
        )

    review_candidates_df = (
        pd.concat(all_review_chunks, ignore_index=True)
        if all_review_chunks
        else pd.DataFrame(columns=REVIEW_OUTPUT_COLUMNS)
    )
    l2_top_terms_df = extract_top_terms_by_l2(
        review_candidates_df,
        config.top_terms_per_l2,
        config.min_term_frequency,
    )
    config.l2_top_terms_output_path.parent.mkdir(parents=True, exist_ok=True)
    l2_top_terms_df.to_csv(
        config.l2_top_terms_output_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
    )

    print(f"Processed {total_rows} listings in discover mode")
    print(f"Rule output -> {config.rule_output_path}")
    print(f"Review candidates -> {config.review_candidates_output_path}")
    print(f"L2 top terms -> {config.l2_top_terms_output_path}")


def _run_score(config: AppConfig, rule_scorer: RuleScorer) -> None:
    training_labels = load_training_labels(config.training_labels_path)
    ml_engine = train_ml_model(training_labels, config)
    _reset_output_files(
        [
            config.rule_output_path,
            config.rule_only_output_path,
            config.review_candidates_output_path,
            config.hybrid_output_path,
            config.hybrid_lite_output_path,
            config.ml_review_labelling_output_path,
        ]
    )

    total_rows = 0
    first_rule_chunk = True
    first_rule_only_chunk = True
    first_review_chunk = True
    first_hybrid_chunk = True
    first_hybrid_lite_chunk = True
    first_ml_review_labelling_chunk = True

    for listings_chunk in iterate_listing_chunks(config.input_listings_path, config.chunk_size):
        rule_scored = rule_scorer.score_chunk(listings_chunk)
        _write_chunk(rule_scored[RULE_OUTPUT_COLUMNS], config.rule_output_path, write_header=first_rule_chunk)
        first_rule_chunk = False
        _write_chunk(
            rule_scored[RULE_OUTPUT_COLUMNS],
            config.rule_only_output_path,
            write_header=first_rule_only_chunk,
        )
        first_rule_only_chunk = False

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
        _write_chunk(
            final_chunk[HYBRID_LITE_OUTPUT_COLUMNS],
            config.hybrid_lite_output_path,
            write_header=first_hybrid_lite_chunk,
        )
        first_hybrid_lite_chunk = False
        review_labelling_chunk = final_chunk[final_chunk["rule_decision"] == "Review"].copy()
        if not review_labelling_chunk.empty:
            review_labelling_chunk["human_label"] = ""
            _write_chunk(
                review_labelling_chunk[ML_REVIEW_LABELLING_COLUMNS],
                config.ml_review_labelling_output_path,
                write_header=first_ml_review_labelling_chunk,
            )
            first_ml_review_labelling_chunk = False

        total_rows += len(final_chunk)

    # Ensure review output exists even when there are no review candidates.
    if first_review_chunk:
        pd.DataFrame(columns=REVIEW_OUTPUT_COLUMNS).to_csv(
            config.review_candidates_output_path,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
        )
    if first_ml_review_labelling_chunk:
        pd.DataFrame(columns=ML_REVIEW_LABELLING_COLUMNS).to_csv(
            config.ml_review_labelling_output_path,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
        )

    print(f"Processed {total_rows} listings")
    print(f"Rule output -> {config.rule_output_path}")
    print(f"Rule-only output -> {config.rule_only_output_path}")
    print(f"Review candidates -> {config.review_candidates_output_path}")
    print(f"Hybrid output -> {config.hybrid_output_path}")
    print(f"Hybrid lite output -> {config.hybrid_lite_output_path}")
    print(f"ML review labelling output -> {config.ml_review_labelling_output_path}")


def run(mode: Optional[str], config_path: Optional[str] = None) -> None:
    if mode not in {"discover", "score"}:
        print("Usage: python main.py discover")
        print("   or: python main.py score")
        return

    config = load_app_config(Path(config_path) if config_path else None)
    signal_library = load_signal_library(config.signal_library_path)
    rule_scorer = RuleScorer(signal_library, config)

    if mode == "discover":
        _run_discover(config, rule_scorer)
    else:
        _run_score(config, rule_scorer)


if __name__ == "__main__":
    args = _parse_args()
    run(mode=args.mode, config_path=args.config)

