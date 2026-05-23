#!/usr/bin/env python3
"""Validate generated clean clustered query dataset JSONL files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_ROOT = Path("/home/wlia0047/ar57/wenyu/result/personal_query/11_query_dataset")
EXPECTED_FIELDS = [
    "category",
    "uuid",
    "asin",
    "cluster_label",
    "cluster_index",
    "correct_query",
    "attrs_used",
]
FORBIDDEN_FIELDS = {
    "sample_id",
    "user_id",
    "clean_query",
    "clean_word_count",
    "has_noisy_query",
    "noisy_query",
    "query_type",
    "correct_word_count",
    "idf",
    "target_depth",
    "user_avg_depth",
    "source_stage",
    "query_category",
    "complexity_level",
    "complexity_group",
    "depth",
    "has_error_query",
    "error_query",
    "injected_errors",
}


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string")
    return value


def require_int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def validate_file(path: Path, counts: Counter[str]) -> None:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            item = json.loads(line)
            if list(item.keys()) != EXPECTED_FIELDS:
                raise ValueError(f"{path}:{lineno} has unexpected field order: {list(item.keys())}")
            forbidden = FORBIDDEN_FIELDS.intersection(item)
            if forbidden:
                raise ValueError(f"{path}:{lineno} contains forbidden fields: {sorted(forbidden)}")

            require_text(item["category"], f"{path}:{lineno}.category")
            require_text(item["uuid"], f"{path}:{lineno}.uuid")
            require_text(item["asin"], f"{path}:{lineno}.asin")
            cluster_label = require_text(item["cluster_label"], f"{path}:{lineno}.cluster_label")
            cluster_index = require_int(item["cluster_index"], f"{path}:{lineno}.cluster_index")
            if cluster_index < 0:
                raise ValueError(f"{path}:{lineno}.cluster_index must be non-negative")
            require_text(item["correct_query"], f"{path}:{lineno}.correct_query")
            require_dict(item["attrs_used"], f"{path}:{lineno}.attrs_used")

            counts[f"cluster_label:{cluster_label}"] += 1
            counts[f"cluster_index:{cluster_index}"] += 1


def main() -> None:
    files = sorted(DATASET_ROOT.glob("*/data.jsonl"))
    if not files:
        raise FileNotFoundError(f"No dataset JSONL files found under {DATASET_ROOT}")
    counts: Counter[str] = Counter()
    for path in files:
        validate_file(path, counts)
    print(f"[VALID] files={len(files)} counts={dict(sorted(counts.items()))}", flush=True)


if __name__ == "__main__":
    main()
