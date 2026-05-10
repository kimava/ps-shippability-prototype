"""Runtime configuration for hybrid scoring."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class AppConfig:
    # Paths
    input_listings_path: Path = Path("data/listings.csv")
    signal_library_path: Path = Path("data/signal_library.csv")
    training_labels_path: Path = Path("data/training_labels.csv")
    rule_output_path: Path = Path("outputs/scored_listings_rule.csv")
    rule_only_output_path: Path = Path("outputs/scored_listings_rule_only.csv")
    review_candidates_output_path: Path = Path("outputs/review_candidates.csv")
    l2_top_terms_output_path: Path = Path("outputs/l2_top_terms.csv")
    hybrid_output_path: Path = Path("outputs/scored_listings_hybrid.csv")
    hybrid_lite_output_path: Path = Path("outputs/scored_listings_hybrid_lite.csv")
    ml_review_labelling_output_path: Path = Path("outputs/ml_review_labelling.csv")
    model_output_path: Path = Path("models/shippability_model.pkl")

    # Scale-related controls
    chunk_size: int = 50_000

    # Toggle ML scoring layer
    enable_ml_scoring: bool = True
    ml_confidence_threshold: float = 0.75
    min_training_rows: int = 8
    top_terms_per_l2: int = 25
    min_term_frequency: int = 3

    # Rule scoring controls
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


def load_app_config(config_json_path: Optional[Path] = None) -> AppConfig:
    """
    Load config from JSON if present, falling back to AppConfig defaults.

    JSON keys can override any AppConfig field, e.g.:
    {
      "chunk_size": 100000,
      "enable_llm_scoring": true
    }
    """
    default_config = AppConfig()
    json_path = config_json_path or Path("config.json")

    if not json_path.exists():
        return default_config

    with json_path.open("r", encoding="utf-8") as file:
        raw_overrides = json.load(file)

    if not isinstance(raw_overrides, dict):
        raise ValueError("config.json must contain a JSON object.")

    compatibility_keys = {
        "enable_llm_scoring",
        "llm_confidence_threshold",
        "output_path",
        "listings_path",
    }
    allowed_keys = set(AppConfig.__dataclass_fields__.keys()) | compatibility_keys
    unknown_keys = [key for key in raw_overrides if key not in allowed_keys]
    if unknown_keys:
        raise ValueError(f"Unknown config.json keys: {unknown_keys}")

    merged_config = {**default_config.__dict__, **raw_overrides}

    # Backward compatibility with previous llm config keys.
    if "enable_llm_scoring" in raw_overrides and "enable_ml_scoring" not in raw_overrides:
        merged_config["enable_ml_scoring"] = bool(raw_overrides["enable_llm_scoring"])
    if "llm_confidence_threshold" in raw_overrides and "ml_confidence_threshold" not in raw_overrides:
        merged_config["ml_confidence_threshold"] = float(raw_overrides["llm_confidence_threshold"])
    if "output_path" in raw_overrides and "hybrid_output_path" not in raw_overrides:
        merged_config["hybrid_output_path"] = raw_overrides["output_path"]
    if "listings_path" in raw_overrides and "input_listings_path" not in raw_overrides:
        merged_config["input_listings_path"] = raw_overrides["listings_path"]

    # Normalize path fields if set as strings in JSON.
    for path_field in [
        "input_listings_path",
        "signal_library_path",
        "training_labels_path",
        "rule_output_path",
        "rule_only_output_path",
        "review_candidates_output_path",
        "l2_top_terms_output_path",
        "hybrid_output_path",
        "hybrid_lite_output_path",
        "ml_review_labelling_output_path",
        "model_output_path",
    ]:
        value = merged_config[path_field]
        if not isinstance(value, Path):
            merged_config[path_field] = Path(value)

    return AppConfig(**merged_config)

