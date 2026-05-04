# Pay & Ship Eligibility Scoring Prototype

## Overview

This project is a Python-based prototype designed to identify whether marketplace listings are suitable for shipping ("Pay & Ship").

It combines:

* **Rule-based keyword scoring** (deterministic signals)
* **Machine learning baseline** (text classification using scikit-learn)

The goal is to reduce manual review by automatically classifying listings as:

* Eligible (Shippable)
* Exclude (Not shippable)
* Review (Requires manual check)

---

## How it works

### 1. Rule-based scoring

All listings are first processed using keyword signals:

* Strong negative signals (e.g. "collection only", bulky items)
* Strong positive signals (e.g. "can post", "small")

Output:

* rule_score
* rule_decision (Eligible / Exclude / Review)

---

### 2. Review candidate extraction

Listings that cannot be confidently classified are marked as:

* `Review`

These listings are passed to the ML layer.

---

### 3. Machine learning scoring (baseline)

A scikit-learn model is trained on labelled data:

* TF-IDF vectorisation of title + description + category
* Logistic Regression classifier

ML is applied only to `Review` listings.

---

### 4. Final decision logic

| Rule decision | ML result                     | Final decision |
| ------------- | ----------------------------- | -------------- |
| Eligible      | -                             | Eligible       |
| Exclude       | -                             | Exclude        |
| Review        | High confidence Shippable     | Eligible       |
| Review        | High confidence Not Shippable | Exclude        |
| Review        | Low confidence                | Review         |

---

## Project structure

```
ps-prototype/
  data/
    listings.csv
    signal_library.csv
    training_labels.csv
  outputs/
    scored_listings_rule.csv
    review_candidates.csv
    scored_listings_hybrid.csv
  src/
    load_data.py
    rule_scoring.py
    ml_scoring.py
    decision_combiner.py
  models/
    shippability_model.pkl
  main.py
  config.py / config.json
```

---

## Input data

### listings.csv

```
listing_id,category,title,description,url
```

### signal_library.csv

```
category,signal_term,signal_type,decision_impact
```

### training_labels.csv (for ML)

```
listing_id,category,title,description,url,human_label
```

human_label values:

* Shippable
* Not Shippable
* Review

---

## Outputs

### 1. Rule-only results

```
outputs/scored_listings_rule.csv
```

### 2. Review candidates

```
outputs/review_candidates.csv
```

### 3. Final hybrid results

```
outputs/scored_listings_hybrid.csv
```

---

## How to run

From the project root directory:

```bash
python main.py
```

If using a custom config:

```bash
python main.py --config config.json
```

---

## Key features

* Scalable to large datasets (chunked processing)
* Category-aware keyword matching
* Hybrid rule + ML decision system
* Configurable thresholds and pipeline behaviour

---

## Future improvements

* Replace mock/heuristic logic with real LLM API
* Expand labelled dataset for stronger ML performance
* Add explainability (top features, SHAP values)
* Introduce confidence calibration

---

## Purpose

This is a prototype to:

* Validate feasibility of automated shipping eligibility classification
* Reduce manual review effort in mixed categories
* Provide a foundation for production-ready ML/AI systems
