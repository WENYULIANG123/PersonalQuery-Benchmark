#!/usr/bin/env python3
"""Validate generated query dataset JSONL field names, order, and query categories."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


DATASET_ROOT = Path("/home/wlia0047/ar57/wenyu/dataset")
EXPECTED_FIELDS = [
    "category",
    "uuid",
    "asin",
    "query_category",
    "complexity_level",
    "profile_complexity_level",
    "correct_query",
    "correct_word_count",
    "idf",
    "attrs_used",
    "has_error_query",
    "error_query",
    "injected_errors",
]
FORBIDDEN_FIELDS = {
    "sample_id",
    "user_id",
    "clean_query",
    "clean_word_count",
    "has_noisy_query",
    "noisy_query",
}
VALID_QUERY_CATEGORIES = {"wide", "deep"}
EXPECTED_ERROR_FIELDS = ["correct", "error", "error_type", "note"]


def validate_file(path: Path, counts: Counter[str]) -> None:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            item = json.loads(line)
            if list(item.keys()) != EXPECTED_FIELDS:
                raise ValueError(f"{path}:{lineno} has unexpected field order: {list(item.keys())}")
            forbidden = FORBIDDEN_FIELDS.intersection(item)
            if forbidden:
                raise ValueError(f"{path}:{lineno} contains forbidden fields: {sorted(forbidden)}")
            query_category = item["query_category"]
            if query_category not in VALID_QUERY_CATEGORIES:
                raise ValueError(f"{path}:{lineno} has invalid query_category: {query_category}")
            if not isinstance(item["idf"], (int, float)):
                raise TypeError(f"{path}:{lineno} idf must be numeric")
            if not isinstance(item["injected_errors"], list):
                raise TypeError(f"{path}:{lineno} injected_errors must be a list")
            for error_index, injected_error in enumerate(item["injected_errors"]):
                if list(injected_error.keys()) != EXPECTED_ERROR_FIELDS:
                    raise ValueError(
                        f"{path}:{lineno} injected_errors[{error_index}] has unexpected fields: "
                        f"{list(injected_error.keys())}"
                    )
                if not isinstance(injected_error["note"], str):
                    raise TypeError(f"{path}:{lineno} injected_errors[{error_index}].note must be a string")
            counts[query_category] += 1


def main() -> None:
    files = sorted(DATASET_ROOT.glob("*/data.jsonl")) + sorted(DATASET_ROOT.glob("*/paired_data.jsonl"))
    if not files:
        raise FileNotFoundError(f"No dataset JSONL files found under {DATASET_ROOT}")
    counts: Counter[str] = Counter()
    for path in files:
        validate_file(path, counts)
    print(f"[VALID] files={len(files)} query_category_counts={dict(sorted(counts.items()))}", flush=True)


if __name__ == "__main__":
    main()
