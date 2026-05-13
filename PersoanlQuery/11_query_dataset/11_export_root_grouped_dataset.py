#!/usr/bin/env python3
"""Rebuild root dataset/*_query.json from Stage 11 JSONL outputs."""

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
QUERY_ORDER = {"wide": 0, "deep": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild grouped root dataset JSON from Stage 11 data.jsonl files.")
    parser.add_argument("--categories", nargs="+", required=True)
    return parser.parse_args()


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
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


def build_grouped_rows(category: str, flat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for row_index, row in enumerate(flat_rows):
        row_label = f"{category} data.jsonl row {row_index + 1}"
        user_id = row.get("uuid")
        asin = row.get("asin")
        if not isinstance(user_id, str) or not user_id:
            raise TypeError(f"{row_label}.uuid must be a non-empty string")
        if not isinstance(asin, str) or not asin:
            raise TypeError(f"{row_label}.asin must be a non-empty string")
        attrs_used = require_dict(row.get("attrs_used"), f"{row_label}.attrs_used")
        query_category = row.get("query_category")
        if query_category not in QUERY_ORDER:
            raise ValueError(f"{row_label}.query_category is unsupported: {query_category}")

        key = (user_id, asin)
        if key not in grouped:
            grouped[key] = {
                "category": category,
                "uuid": user_id,
                "asin": asin,
                "attrs_used": attrs_used,
                "queries": [],
            }

        # Root dataset rows have only one attrs_used field, but Stage 11 flat rows
        # may keep query-specific attrs. Prefer the wide-query attrs when available.
        if query_category == "wide":
            grouped[key]["attrs_used"] = attrs_used

        grouped[key]["queries"].append(
            {
                "query_category": query_category,
                "complexity_level": row["complexity_level"],
                "correct_query": row["correct_query"],
                "correct_word_count": row["correct_word_count"],
                "idf": row["idf"],
                "has_error_query": row["has_error_query"],
                "error_query": row["error_query"],
                "injected_errors": row["injected_errors"],
            }
        )

    result = []
    for key, grouped_row in grouped.items():
        grouped_row["queries"].sort(key=lambda item: QUERY_ORDER[item["query_category"]])
        seen_categories = [item["query_category"] for item in grouped_row["queries"]]
        if len(seen_categories) != len(set(seen_categories)):
            raise ValueError(f"Duplicate query_category under key {key}: {seen_categories}")
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
