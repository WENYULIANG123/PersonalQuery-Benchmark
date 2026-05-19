#!/usr/bin/env python3
"""Validate generated syntax-depth query dataset JSONL files."""

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
    "complexity_group",
    "depth",
    "correct_query",
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
    "query_type",
    "correct_word_count",
    "idf",
    "target_depth",
    "user_avg_depth",
    "source_stage",
    "query_category",
    "complexity_level",
}
VALID_COMPLEXITY_GROUPS = {"low", "medium", "high"}
EXPECTED_ERROR_FIELDS = ["target_token_depth"]


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string")
    return value


def require_int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def require_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    return float(value)


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def validate_injected_errors(errors: list[Any], label: str) -> None:
    for error_index, injected_error in enumerate(errors):
        error_label = f"{label}.injected_errors[{error_index}]"
        injected_error = require_dict(injected_error, error_label)
        if list(injected_error.keys()) != EXPECTED_ERROR_FIELDS:
            raise ValueError(f"{error_label} has unexpected fields: {list(injected_error.keys())}")
        target_token_depth = require_int(injected_error["target_token_depth"], f"{error_label}.target_token_depth")
        if target_token_depth < 0:
            raise ValueError(f"{error_label}.target_token_depth must be non-negative")


def validate_file(path: Path, counts: Counter[str]) -> None:
    is_paired_file = path.name == "paired_data.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            item = json.loads(line)
            if list(item.keys()) != EXPECTED_FIELDS:
                raise ValueError(f"{path}:{lineno} has unexpected field order: {list(item.keys())}")
            forbidden = FORBIDDEN_FIELDS.intersection(item)
            if forbidden:
                raise ValueError(f"{path}:{lineno} contains forbidden fields: {sorted(forbidden)}")

            complexity_group = require_text(item["complexity_group"], f"{path}:{lineno}.complexity_group")
            if complexity_group not in VALID_COMPLEXITY_GROUPS:
                raise ValueError(f"{path}:{lineno} has invalid complexity_group: {complexity_group}")

            depth = require_int(item["depth"], f"{path}:{lineno}.depth")
            if depth < 0:
                raise ValueError(f"{path}:{lineno}.depth must be non-negative")

            correct_query = require_text(item["correct_query"], f"{path}:{lineno}.correct_query")
            require_dict(item["attrs_used"], f"{path}:{lineno}.attrs_used")

            has_error_query = require_bool(item["has_error_query"], f"{path}:{lineno}.has_error_query")
            injected_errors = require_list(item["injected_errors"], f"{path}:{lineno}.injected_errors")
            validate_injected_errors(injected_errors, f"{path}:{lineno}")

            if is_paired_file and not has_error_query:
                raise ValueError(f"{path}:{lineno} paired_data row must have an error query")

            if has_error_query:
                require_text(item["error_query"], f"{path}:{lineno}.error_query")
                if not injected_errors:
                    raise ValueError(f"{path}:{lineno} noisy row injected_errors must be non-empty")
            else:
                if item["error_query"] is not None:
                    raise ValueError(f"{path}:{lineno} clean row error_query must be null")
                if injected_errors:
                    raise ValueError(f"{path}:{lineno} clean row injected_errors must be empty")

            counts[f"complexity_group:{complexity_group}"] += 1
            counts[f"has_error_query:{has_error_query}"] += 1


def main() -> None:
    files = sorted(DATASET_ROOT.glob("*/data.jsonl")) + sorted(DATASET_ROOT.glob("*/paired_data.jsonl"))
    if not files:
        raise FileNotFoundError(f"No dataset JSONL files found under {DATASET_ROOT}")
    counts: Counter[str] = Counter()
    for path in files:
        validate_file(path, counts)
    print(f"[VALID] files={len(files)} counts={dict(sorted(counts.items()))}", flush=True)


if __name__ == "__main__":
    main()
