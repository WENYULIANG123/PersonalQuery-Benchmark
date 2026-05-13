#!/usr/bin/env python3
"""Delete dataset queries whose surface complexity does not match their labels."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/fs04/ar57/wenyu")
ROOT_DATASET_DIR = REPO_ROOT / "dataset"
STAGE11_DATASET_DIR = REPO_ROOT / "result" / "personal_query" / "11_query_dataset"
TARGET_CATEGORIES = ("Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies")
REQUIRED_TOKEN = {
    "wide": "which",
    "deep": "that",
}
FORBIDDEN_TOKEN = {
    "wide": "that",
    "deep": "which",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete invalid-complexity queries from released datasets.")
    parser.add_argument("--categories", nargs="+", required=True)
    return parser.parse_args()


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def require_text(item: dict[str, Any], key: str, label: str) -> str:
    if key not in item:
        raise KeyError(f"{label} is missing required key: {key}")
    value = item[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label}.{key} must be a non-empty string")
    return value


def require_int(item: dict[str, Any], key: str, label: str) -> int:
    if key not in item:
        raise KeyError(f"{label} is missing required key: {key}")
    value = item[key]
    if not isinstance(value, int):
        raise TypeError(f"{label}.{key} must be an integer")
    return value


def require_bool(item: dict[str, Any], key: str, label: str) -> bool:
    if key not in item:
        raise KeyError(f"{label} is missing required key: {key}")
    value = item[key]
    if not isinstance(value, bool):
        raise TypeError(f"{label}.{key} must be a boolean")
    return value


def count_token(text: str, token: str) -> int:
    return len(re.findall(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE))


def query_is_valid(query: dict[str, Any], label: str) -> bool:
    query_category = require_text(query, "query_category", label)
    if query_category not in REQUIRED_TOKEN:
        raise ValueError(f"{label}.query_category is unsupported: {query_category}")

    complexity_level = require_int(query, "complexity_level", label)
    if complexity_level < 0:
        raise ValueError(f"{label}.complexity_level must be non-negative")

    texts = [require_text(query, "correct_query", label)]
    has_error_query = require_bool(query, "has_error_query", label)
    if has_error_query:
        texts.append(require_text(query, "error_query", label))

    required_token = REQUIRED_TOKEN[query_category]
    forbidden_token = FORBIDDEN_TOKEN[query_category]
    for text in texts:
        if count_token(text, required_token) != complexity_level:
            return False
        if count_token(text, forbidden_token) != 0:
            return False
    return True


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def filter_root_dataset(category: str) -> dict[str, int]:
    dataset_file = ROOT_DATASET_DIR / f"{category}_query.json"
    rows = require_list(load_json(dataset_file), str(dataset_file))

    kept_rows: list[dict[str, Any]] = []
    removed_queries = 0
    removed_records = 0
    kept_queries = 0

    for row_index, raw_row in enumerate(rows):
        row = require_dict(raw_row, f"{dataset_file}[{row_index}]")
        queries = require_list(row.get("queries"), f"{dataset_file}[{row_index}].queries")
        kept_queries_for_row = []
        for query_index, raw_query in enumerate(queries):
            query = require_dict(raw_query, f"{dataset_file}[{row_index}].queries[{query_index}]")
            if query_is_valid(query, f"{dataset_file}[{row_index}].queries[{query_index}]"):
                kept_queries_for_row.append(query)
                kept_queries += 1
            else:
                removed_queries += 1
        if kept_queries_for_row:
            row["queries"] = kept_queries_for_row
            kept_rows.append(row)
        else:
            removed_records += 1

    dataset_file.write_text(json.dumps(kept_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "rows_before": len(rows),
        "rows_after": len(kept_rows),
        "removed_records": removed_records,
        "kept_queries": kept_queries,
        "removed_queries": removed_queries,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"{path}:{line_number} is empty")
            rows.append(require_dict(json.loads(stripped), f"{path}:{line_number}"))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def filter_stage11_category(category: str) -> dict[str, Any]:
    category_dir = STAGE11_DATASET_DIR / category
    data_file = category_dir / "data.jsonl"
    paired_file = category_dir / "paired_data.jsonl"
    summary_file = category_dir / "summary.json"

    data_rows = read_jsonl(data_file)
    kept_data_rows = []
    removed_data_rows = 0
    rows_by_query_category: Counter[str] = Counter()
    rows_by_complexity: Counter[str] = Counter()

    for row_index, row in enumerate(data_rows, 1):
        if query_is_valid(row, f"{data_file}:{row_index}"):
            kept_data_rows.append(row)
            rows_by_query_category[row["query_category"]] += 1
            rows_by_complexity[f"{row['query_category']}:{row['complexity_level']}"] += 1
        else:
            removed_data_rows += 1

    kept_paired_rows = [row for row in kept_data_rows if row["has_error_query"]]
    paired_rows_by_query_category: Counter[str] = Counter(row["query_category"] for row in kept_paired_rows)

    write_jsonl(data_file, kept_data_rows)
    write_jsonl(paired_file, kept_paired_rows)

    summary = require_dict(load_json(summary_file), str(summary_file))
    summary["num_dataset_rows"] = len(kept_data_rows)
    summary["num_paired_rows"] = len(kept_paired_rows)
    summary["num_unpaired_rows"] = len(kept_data_rows) - len(kept_paired_rows)
    summary["rows_by_query_category"] = dict(sorted(rows_by_query_category.items()))
    summary["paired_rows_by_query_category"] = dict(sorted(paired_rows_by_query_category.items()))
    summary["rows_by_complexity"] = dict(sorted(rows_by_complexity.items()))
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "data_rows_before": len(data_rows),
        "data_rows_after": len(kept_data_rows),
        "data_rows_removed": removed_data_rows,
        "paired_rows_after": len(kept_paired_rows),
        "rows_by_query_category": dict(sorted(rows_by_query_category.items())),
        "paired_rows_by_query_category": dict(sorted(paired_rows_by_query_category.items())),
        "rows_by_complexity": dict(sorted(rows_by_complexity.items())),
    }


def update_aggregate_summary(categories: list[str]) -> None:
    summary_file = STAGE11_DATASET_DIR / "summary.json"
    summary = require_dict(load_json(summary_file), str(summary_file))
    category_summaries = []
    total_rows = 0
    total_paired_rows = 0

    for category in categories:
        category_summary_file = STAGE11_DATASET_DIR / category / "summary.json"
        category_summary = require_dict(load_json(category_summary_file), str(category_summary_file))
        category_summaries.append(category_summary)
        total_rows += category_summary["num_dataset_rows"]
        total_paired_rows += category_summary["num_paired_rows"]

    summary["categories"] = categories
    summary["category_summaries"] = category_summaries
    summary["num_categories"] = len(categories)
    summary["num_dataset_rows"] = total_rows
    summary["num_paired_rows"] = total_paired_rows
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for category in args.categories:
        if category not in TARGET_CATEGORIES:
            raise ValueError(f"Unsupported category: {category}")

    for category in args.categories:
        root_stats = filter_root_dataset(category)
        stage11_stats = filter_stage11_category(category)
        print(f"[FILTER] {category} root={root_stats} stage11={stage11_stats}", flush=True)

    update_aggregate_summary(args.categories)


if __name__ == "__main__":
    main()
