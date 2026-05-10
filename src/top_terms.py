"""Extract candidate terms from review listings grouped by L2."""

from __future__ import annotations

import re
from math import log
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple, Union

import pandas as pd

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
GENERIC_WORDS = {
    "for",
    "sale",
    "good",
    "condition",
    "used",
    "new",
    "excellent",
    "collection",
    "item",
    "items",
    "selling",
    "available",
    "please",
    "thanks",
    "great",
    "fully",
    "still",
    "brand",
    "perfect",
    "lovely",
    "nice",
    "old",
    "free",
    "included",
    "complete",
    "set",
    "bundle",
    "cash",
    "today",
    "message",
    "contact",
    "delivery",
    "post",
    "posted",
    "shipping",
    "ship",
    "collect",
    "only",
    "no",
    "yes",
    "can",
}
DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}
STOPWORDS = GENERIC_WORDS | DEFAULT_STOPWORDS
WEAK_DESCRIPTORS = {
    "best",
    "amazing",
    "quality",
    "vintage",
    "modern",
    "classic",
    "small",
    "medium",
    "large",
}
PRODUCT_HINTS = {
    "sofa",
    "wardrobe",
    "table",
    "chair",
    "bike",
    "bed",
    "mattress",
    "shoes",
    "toy",
    "plush",
    "book",
    "game",
    "phone",
    "case",
    "watch",
    "jewellery",
    "lego",
    "kitchen",
    "goal",
}

NEGATIVE_HINTS = {"sofa", "wardrobe", "table", "bike", "ride on", "bed", "mattress", "large", "heavy"}
POSITIVE_HINTS = {"shoes", "toy", "plush", "book", "game", "phone", "case", "watch", "jewellery", "lego"}


def _is_valid_token(token: str) -> bool:
    if len(token) < 3:
        return False
    if token.isdigit():
        return False
    if token in STOPWORDS:
        return False
    if not any(char.isalpha() for char in token):
        return False
    return True


def _tokenize(text: str) -> List[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if _is_valid_token(token)]


def _is_useful_bigram(left: str, right: str) -> bool:
    if left in WEAK_DESCRIPTORS and right in WEAK_DESCRIPTORS:
        return False
    if left in STOPWORDS or right in STOPWORDS:
        return False
    # Keep likely product phrases while filtering vague adjective-only pairs.
    if left in PRODUCT_HINTS or right in PRODUCT_HINTS:
        return True
    return left not in WEAK_DESCRIPTORS and right not in WEAK_DESCRIPTORS


def _ngrams(tokens: List[str]) -> List[str]:
    terms = list(tokens)
    for idx in range(len(tokens) - 1):
        left = tokens[idx]
        right = tokens[idx + 1]
        if _is_useful_bigram(left, right):
            terms.append("{0} {1}".format(left, right))
    return terms


def _suggest_signal(term: str) -> Tuple[str, str]:
    lowered = term.lower()
    if any(hint in lowered for hint in NEGATIVE_HINTS):
        return "strong_negative", "Exclude"
    if any(hint in lowered for hint in POSITIVE_HINTS):
        return "supporting", "Review"
    return "supporting", "Neutral"


def _term_usefulness_score(term: str, l2_frequency: int, global_l2_count: int, l2_total_count: int) -> float:
    idf = log((l2_total_count + 1.0) / (global_l2_count + 1.0)) + 1.0
    noun_bonus = 1.2 if any(part in PRODUCT_HINTS for part in term.split()) else 1.0
    return float(l2_frequency) * idf * noun_bonus


def extract_top_terms_by_l2(
    review_df: pd.DataFrame,
    top_terms_per_l2: int,
    min_term_frequency: int,
) -> pd.DataFrame:
    counts_by_l2: DefaultDict[str, Counter] = defaultdict(Counter)
    examples_by_l2: DefaultDict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for row in review_df.itertuples(index=False):
        l2_value = str(getattr(row, "L2", "") or "").strip() or "Unknown"
        title = str(getattr(row, "title", "") or "")
        description = str(getattr(row, "description", "") or "")
        terms = _ngrams(_tokenize(f"{title} {description}"))
        seen_terms: Set[str] = set()
        for term in terms:
            counts_by_l2[l2_value][term] += 1
            if term not in seen_terms and len(examples_by_l2[l2_value][term]) < 3 and title:
                examples_by_l2[l2_value][term].append(title)
                seen_terms.add(term)

    rows: List[Dict[str, Union[str, int]]] = []
    l2_document_frequency: Counter = Counter()
    for term_counts in counts_by_l2.values():
        for term in term_counts.keys():
            l2_document_frequency[term] += 1

    total_l2_buckets = max(len(counts_by_l2), 1)
    for l2_value in sorted(counts_by_l2.keys()):
        ranked_terms: List[Tuple[str, int, float]] = []
        for term, frequency in counts_by_l2[l2_value].items():
            if frequency < min_term_frequency:
                continue
            score = _term_usefulness_score(
                term=term,
                l2_frequency=int(frequency),
                global_l2_count=int(l2_document_frequency[term]),
                l2_total_count=total_l2_buckets,
            )
            ranked_terms.append((term, int(frequency), score))

        ranked_terms.sort(key=lambda item: (-item[2], -item[1], item[0]))
        for term, frequency, _ in ranked_terms[:top_terms_per_l2]:
            signal_type, impact = _suggest_signal(term)
            rows.append(
                {
                    "L2": l2_value,
                    "candidate_term": term,
                    "frequency": int(frequency),
                    "example_titles": " | ".join(examples_by_l2[l2_value].get(term, [])),
                    "suggested_signal_type": signal_type,
                    "suggested_decision_impact": impact,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "L2",
            "candidate_term",
            "frequency",
            "example_titles",
            "suggested_signal_type",
            "suggested_decision_impact",
        ],
    )
