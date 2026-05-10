"""Baseline ML scoring using scikit-learn TF-IDF + LogisticRegression."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import AppConfig
from .load_data import derive_internal_category

ALLOWED_LABELS = {"Shippable", "Not Shippable", "Review"}


def _build_text_features(df: pd.DataFrame) -> pd.Series:
    return (
        derive_internal_category(df).fillna("").astype(str)
        + " "
        + df["title"].fillna("").astype(str)
        + " "
        + df["description"].fillna("").astype(str)
    )


@dataclass
class MLScoringEngine:
    model: Optional[Pipeline]
    enabled: bool
    reason_template: str = (
        "ML prediction based on TF-IDF text features from category, title and description."
    )

    def predict_review_candidates(self, scored_chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = scored_chunk.copy()
        chunk["ml_label"] = "Skipped"
        chunk["ml_confidence"] = 0.0
        chunk["ml_reason"] = "ML skipped due to deterministic rule decision."

        if not self.enabled or self.model is None:
            return chunk

        review_mask = chunk["rule_decision"] == "Review"
        if not review_mask.any():
            return chunk

        review_text = _build_text_features(chunk.loc[review_mask])
        probabilities = self.model.predict_proba(review_text)
        labels = self.model.classes_

        predicted_idx = probabilities.argmax(axis=1)
        predicted_labels = [labels[idx] for idx in predicted_idx]
        predicted_conf = [float(probabilities[i, idx]) for i, idx in enumerate(predicted_idx)]

        chunk.loc[review_mask, "ml_label"] = predicted_labels
        chunk.loc[review_mask, "ml_confidence"] = predicted_conf
        chunk.loc[review_mask, "ml_reason"] = self.reason_template
        return chunk


def _print_evaluation_metrics(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> None:
    y_pred = model.predict(x_test)
    print("\n=== Baseline ML Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=== End Baseline ML Evaluation ===\n")


def train_ml_model(training_df: Optional[pd.DataFrame], config: AppConfig) -> MLScoringEngine:
    if not config.enable_ml_scoring:
        print("ML scoring disabled by config.")
        return MLScoringEngine(model=None, enabled=False)

    if training_df is None or training_df.empty:
        print("No training labels found; ML scoring will be skipped.")
        return MLScoringEngine(model=None, enabled=False)

    labeled_df = training_df[training_df["human_label"].isin(ALLOWED_LABELS)].copy()
    if len(labeled_df) < config.min_training_rows:
        print(
            f"Training labels too small ({len(labeled_df)} rows, minimum {config.min_training_rows}); "
            "ML scoring will be skipped."
        )
        return MLScoringEngine(model=None, enabled=False)

    x = _build_text_features(labeled_df)
    y = labeled_df["human_label"].astype(str)

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30_000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    class_counts = y.value_counts()
    can_evaluate = len(labeled_df) >= 12 and len(class_counts) > 1 and class_counts.min() >= 2

    if can_evaluate:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        model.fit(x_train, y_train)
        _print_evaluation_metrics(model, x_test, y_test)
    else:
        print("Skipping ML holdout evaluation due to limited class distribution/size.")
        model.fit(x, y)

    # Refit on all labeled data for deployment in the scoring pass.
    model.fit(x, y)
    config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.model_output_path.open("wb") as file:
        pickle.dump(model, file)

    print(f"Saved baseline model -> {config.model_output_path}")
    return MLScoringEngine(model=model, enabled=True)

