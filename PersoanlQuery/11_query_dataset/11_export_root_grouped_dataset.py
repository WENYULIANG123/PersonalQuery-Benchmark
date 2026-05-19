#!/usr/bin/env python3
"""Rebuild root dataset/*_query.json from Stage 11 syntax-depth JSONL outputs."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/fs04/ar57/wenyu")
ROOT_DATASET_DIR = REPO_ROOT / "dataset"
STAGE11_DATASET_DIR = REPO_ROOT / "result" / "personal_query" / "11_query_dataset"
TARGET_CATEGORIES = ("Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies")
COMPLEXITY_GROUP_ORDER = {"low": 0, "medium": 1, "high": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild grouped root dataset JSON from Stage 11 data.jsonl files.")
    parser.add_argument("--categories", nargs="+", required=True)
    return parser.parse_args()


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def require_text(item: dict[str, Any], key: str, label: str) -> str:
    if key not in item:
        raise KeyError(f"{label} is missing required key: {key}")
    value = item[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label}.{key} must be a non-empty string")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"{path}:{line_number} is empty")
            rows.append(require_dict(json.loads(stripped), f"{path}:{line_number}"))
    return rows


def query_sort_key(query: dict[str, Any]) -> tuple[int, int, str]:
    complexity_group = query["complexity_group"]
    if complexity_group not in COMPLEXITY_GROUP_ORDER:
        raise ValueError(f"Unsupported complexity_group in grouped query: {complexity_group}")
    return (
        COMPLEXITY_GROUP_ORDER[complexity_group],
        query["depth"],
        query["correct_query"],
    )


def build_grouped_query(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "complexity_group": row["complexity_group"],
        "depth": row["depth"],
        "correct_query": row["correct_query"],
        "attrs_used": row["attrs_used"],
        "has_error_query": row["has_error_query"],
        "error_query": row["error_query"],
        "injected_errors": row["injected_errors"],
    }


def build_grouped_rows(category: str, flat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for row_index, row in enumerate(flat_rows):
        row_label = f"{category} data.jsonl row {row_index + 1}"
        user_id = require_text(row, "uuid", row_label)
        asin = require_text(row, "asin", row_label)
        attrs_used = require_dict(row["attrs_used"], f"{row_label}.attrs_used")
        complexity_group = require_text(row, "complexity_group", row_label)
        if complexity_group not in COMPLEXITY_GROUP_ORDER:
            raise ValueError(f"{row_label}.complexity_group is unsupported: {complexity_group}")

        key = (user_id, asin)
        if key not in grouped:
            grouped[key] = {
                "category": category,
                "uuid": user_id,
                "asin": asin,
                "attrs_used": attrs_used,
                "queries": [],
            }

        grouped[key]["queries"].append(build_grouped_query(row))

    result = []
    for key, grouped_row in grouped.items():
        grouped_row["queries"].sort(key=query_sort_key)
        seen_query_keys = [
            (item["has_error_query"], item["complexity_group"], item["depth"], item["correct_query"])
            for item in grouped_row["queries"]
        ]
        if len(seen_query_keys) != len(set(seen_query_keys)):
            raise ValueError(f"Duplicate grouped query under key {key}: {seen_query_keys}")
        result.append(grouped_row)
    return result


def export_category(category: str) -> dict[str, int]:
    data_file = STAGE11_DATASET_DIR / category / "data.jsonl"
    output_file = ROOT_DATASET_DIR / f"{category}_query.json"
    flat_rows = read_jsonl(data_file)
    grouped_rows = build_grouped_rows(category, flat_rows)
    output_file.write_text(json.dumps(grouped_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "flat_rows": len(flat_rows),
        "grouped_rows": len(grouped_rows),
    }


def main() -> None:
    args = parse_args()
    for category in args.categories:
        if category not in TARGET_CATEGORIES:
            raise ValueError(f"Unsupported category: {category}")

    for category in args.categories:
        stats = export_category(category)
        print(f"[EXPORT] {category}: flat_rows={stats['flat_rows']} grouped_rows={stats['grouped_rows']}", flush=True)


if __name__ == "__main__":
    main()
