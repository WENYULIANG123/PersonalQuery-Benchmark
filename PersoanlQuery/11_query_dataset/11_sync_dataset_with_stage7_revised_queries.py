#!/usr/bin/env python3
"""Sync dataset/*_query.json correct queries with Stage 7 revised ground-truth queries."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path("/fs04/ar57/wenyu")
DATASET_ROOT = REPO_ROOT / "dataset"
STAGE11_HELPER = REPO_ROOT / "PersoanlQuery" / "11_query_dataset" / "11_build_query_datasets.py"
NOISY_QUERY_ROOT = REPO_ROOT / "result" / "personal_query" / "07_inject_noisy"
DATASET_TO_STAGE7_CATEGORY = {
    "wide": "acl",
    "deep": "ccomp",
}
TARGET_CATEGORIES = (
    "Baby_Products",
    "Grocery_and_Gourmet_Food",
    "Pet_Supplies",
)


def load_stage11_helpers():
    spec = importlib.util.spec_from_file_location("stage11_build_query_datasets", STAGE11_HELPER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Stage 11 helper module: {STAGE11_HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite dataset/*_query.json correct queries using Stage 7 revised ground-truth queries."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help="Explicit categories to sync. No category is inferred automatically.",
    )
    return parser.parse_args()


def parse_packed_json_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 7 noisy query file does not exist: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Stage 7 noisy query file is empty: {path}")
    if content.startswith("["):
        data = json.loads(content)
        if not isinstance(data, list):
            raise TypeError(f"Expected JSON list in {path}, got {type(data).__name__}")
        return data

    data: list[dict[str, Any]] = []
    depth = 0
    start = -1
    for idx, char in enumerate(content):
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                obj = json.loads(content[start:idx + 1])
                if not isinstance(obj, dict):
                    raise TypeError(f"Expected object in packed JSON {path}, got {type(obj).__name__}")
                data.append(obj)
                start = -1

    if depth != 0:
        raise ValueError(f"Malformed packed JSON stream in {path}")
    return data


def build_revised_index(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    revised_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            raise TypeError(f"Stage 7 record at index {idx} must be an object")
        user_id = str(item.get("user_id", "")).strip()
        asin = str(item.get("asin", "")).strip()
        query_category = str(item.get("query_category", "")).strip()
        ground_truth_query = str(item.get("ground_truth_query", "")).strip()
        noisy_query = str(item.get("noisy_query", "")).strip()
        injected_errors = item.get("injected_errors")

        if not user_id or not asin or query_category not in {"acl", "ccomp"}:
            raise ValueError(
                f"Invalid Stage 7 key fields at index {idx}: "
                f"user_id={user_id!r}, asin={asin!r}, query_category={query_category!r}"
            )
        if not ground_truth_query or not noisy_query:
            raise ValueError(f"Stage 7 record at index {idx} is missing query text")
        if not isinstance(injected_errors, list):
            raise TypeError(f"Stage 7 injected_errors at index {idx} must be a list")

        key = (user_id, asin, query_category)
        revised_index[key] = item
    return revised_index


def sync_category(category: str, helpers) -> dict[str, int]:
    dataset_file = DATASET_ROOT / f"{category}_query.json"
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_file}")

    stage7_file = NOISY_QUERY_ROOT / category / "noisy_query.json"
    revised_records = parse_packed_json_objects(stage7_file)
    revised_index = build_revised_index(revised_records)
    word_idf = helpers.get_or_build_word_idf(category)

    dataset_rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    if not isinstance(dataset_rows, list):
        raise TypeError(f"Dataset file must contain a list: {dataset_file}")

    updated_queries = 0
    unchanged_queries = 0
    seen_stage7_keys: set[tuple[str, str, str]] = set()

    for row_idx, row in enumerate(dataset_rows):
        if not isinstance(row, dict):
            raise TypeError(f"dataset[{row_idx}] in {dataset_file} must be an object")
        user_id = str(row.get("uuid", "")).strip()
        asin = str(row.get("asin", "")).strip()
        queries = row.get("queries")
        if not user_id or not asin or not isinstance(queries, list):
            raise ValueError(f"dataset[{row_idx}] in {dataset_file} is missing uuid/asin/queries")

        for query_idx, query in enumerate(queries):
            if not isinstance(query, dict):
                raise TypeError(f"dataset[{row_idx}].queries[{query_idx}] must be an object")
            dataset_query_category = str(query.get("query_category", "")).strip()
            if dataset_query_category not in DATASET_TO_STAGE7_CATEGORY:
                raise ValueError(
                    f"dataset[{row_idx}].queries[{query_idx}] has invalid query_category: {dataset_query_category!r}"
                )
            stage7_query_category = DATASET_TO_STAGE7_CATEGORY[dataset_query_category]
            key = (user_id, asin, stage7_query_category)
            stage7_item = revised_index.get(key)
            if stage7_item is None:
                unchanged_queries += 1
                continue

            revised_query = stage7_item["ground_truth_query"]
            noisy_query = stage7_item["noisy_query"]
            injected_errors = stage7_item["injected_errors"]
            query["correct_query"] = revised_query
            query["correct_word_count"] = len(revised_query.split())
            query["idf"] = helpers.compute_query_idf(revised_query, word_idf)
            query["has_error_query"] = True
            query["error_query"] = noisy_query
            query["injected_errors"] = injected_errors
            updated_queries += 1
            seen_stage7_keys.add(key)

    missing_dataset_keys = sorted(set(revised_index) - seen_stage7_keys)
    if missing_dataset_keys:
        raise ValueError(
            f"Stage 7 contains {len(missing_dataset_keys)} keys not found in dataset {category}: "
            f"{missing_dataset_keys[:5]}"
        )

    dataset_file.write_text(json.dumps(dataset_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "updated_queries": updated_queries,
        "unchanged_queries": unchanged_queries,
        "stage7_records": len(revised_index),
    }


def main() -> None:
    args = parse_args()
    helpers = load_stage11_helpers()

    for category in args.categories:
        if category not in TARGET_CATEGORIES:
            raise ValueError(f"Unsupported category: {category}")

    for category in args.categories:
        stats = sync_category(category, helpers)
        print(
            f"[SYNC] {category}: updated_queries={stats['updated_queries']} "
            f"unchanged_queries={stats['unchanged_queries']} stage7_records={stats['stage7_records']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
