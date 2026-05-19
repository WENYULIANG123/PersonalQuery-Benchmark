#!/usr/bin/env python3
"""Build per-category syntax-depth query datasets from Stage 08 and Stage 09 outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")
RESULT_ROOT = REPO_ROOT / "result" / "personal_query"
DATASET_ROOT = RESULT_ROOT / "11_query_dataset"

QUERY_CATEGORY = "syntax_depth"
SOURCE_STAGE_08 = "08_clean"
SOURCE_STAGE_09 = "09_noisy"
SYNTAX_DEPTH_GROUPS = {"low_complexity", "medium_complexity", "high_complexity"}
STAGE09_SYNTAX_DEPTH_GROUPS = {"low_complexity", "medium_complexity"}
COMPLEXITY_GROUP_MAP = {
    "low_complexity": "low",
    "medium_complexity": "medium",
    "high_complexity": "high",
}
OUTPUT_COMPLEXITY_GROUPS = {"low", "medium", "high"}
INJECTED_ERROR_FIELDS = ["target_token_depth"]
OUTPUT_FIELDS = [
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


@dataclass(frozen=True)
class DatasetPaths:
    category: str
    stage6_syntax_depth_query_file: Path
    stage7_original_query_file: Path
    stage8_retrieval_summary_file: Path
    stage9_noisy_results_file: Path
    output_dir: Path
    dataset_file: Path
    paired_dataset_file: Path
    summary_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one JSONL dataset per category from syntax-depth Stage 08 and Stage 09 outputs."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help="Explicit category names to build. No category is inferred automatically.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DATASET_ROOT),
        help="Directory where per-category dataset folders will be written.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} file does not exist: {path}")


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object, got {type(value).__name__}")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON array, got {type(value).__name__}")
    return value


def require_text(item: dict[str, Any], key: str, label: str) -> str:
    if key not in item:
        raise KeyError(f"{label} is missing required key: {key}")
    value = item[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def require_int(item: dict[str, Any], key: str, label: str) -> int:
    if key not in item:
        raise KeyError(f"{label} is missing required key: {key}")
    value = item[key]
    if not isinstance(value, int):
        raise TypeError(f"{label}.{key} must be an integer, got {type(value).__name__}")
    return value


def require_number(item: dict[str, Any], key: str, label: str) -> float:
    if key not in item:
        raise KeyError(f"{label} is missing required key: {key}")
    value = item[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{label}.{key} must be numeric, got {type(value).__name__}")
    return float(value)


def read_json_array(path: Path, label: str) -> list[Any]:
    require_file(path, label)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return require_list(data, label)


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return require_dict(data, label)


def read_adjacent_json_objects(path: Path, label: str) -> list[dict[str, Any]]:
    require_file(path, label)
    content = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    length = len(content)

    while index < length:
        while index < length and content[index].isspace():
            index += 1
        if index >= length:
            break
        obj, next_index = decoder.raw_decode(content, index)
        objects.append(require_dict(obj, f"{label} object at offset {index}"))
        index = next_index

    if not objects:
        raise ValueError(f"{label} file contains no JSON objects: {path}")
    return objects


def normalize_injected_errors(errors: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(errors, list):
        raise TypeError(f"{label} must be a list")

    normalized = []
    for idx, raw_error in enumerate(errors):
        error_label = f"{label}[{idx}]"
        error = require_dict(raw_error, error_label)
        if list(error.keys()) != INJECTED_ERROR_FIELDS:
            raise ValueError(f"{error_label} must have fields {INJECTED_ERROR_FIELDS}, got {list(error.keys())}")
        target_token_depth = require_int(error, "target_token_depth", error_label)
        normalized.append({"target_token_depth": target_token_depth})
    return normalized


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def format_stat_value(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"


def format_two_column_stats_table(rows: list[tuple[str, int | float]]) -> str:
    if not rows:
        raise ValueError("Cannot format an empty statistics table")

    string_rows = [(name, format_stat_value(value)) for name, value in rows]
    total_query_index = next((idx for idx, (name, _) in enumerate(string_rows) if name == "Total Query"), None)
    split_index = total_query_index + 1 if total_query_index is not None else (len(string_rows) + 1) // 2
    left_rows = string_rows[:split_index]
    right_rows = string_rows[split_index:]

    stat_width = max(len("Statistics"), *(len(name) for name, _ in string_rows))
    value_width = max(len("Value"), *(len(value) for _, value in string_rows))
    left_header = f"{'Statistics':<{stat_width}} {'Value':>{value_width}}"
    right_header = f"{'Statistics':<{stat_width}} {'Value':>{value_width}}"
    separator = "-" * len(left_header)

    lines = [
        f"{separator}-+-{separator}",
        f"{left_header} | {right_header}",
        f"{separator}-+-{separator}",
    ]
    for row_index, left_row in enumerate(left_rows):
        left_name, left_value = left_row
        if row_index < len(right_rows):
            right_name, right_value = right_rows[row_index]
        else:
            right_name, right_value = "", ""
        if left_name == "Total Category Users":
            lines.append(f"{separator}-+-{separator}")
        left_text = f"{left_name:<{stat_width}} {left_value:>{value_width}}"
        if right_name:
            lines.append(f"{left_text} | {right_name:<{stat_width}} {right_value:>{value_width}}")
        else:
            lines.append(left_text)
    lines.append(f"{separator}-+-{separator}")
    return "\n".join(lines)


def category_label(category: str) -> str:
    labels = {
        "Baby_Products": "Baby",
        "Grocery_and_Gourmet_Food": "Grocery",
        "Pet_Supplies": "Pet",
    }
    if category not in labels:
        raise KeyError(f"Missing statistics label for category: {category}")
    return labels[category]


def normalize_complexity_group(group: str, label: str) -> str:
    if group not in COMPLEXITY_GROUP_MAP:
        raise ValueError(f"{label} has unsupported complexity group: {group}")
    return COMPLEXITY_GROUP_MAP[group]


def print_statistics_table(aggregate_summary: dict[str, Any]) -> None:
    category_summaries = require_list(aggregate_summary.get("category_summaries"), "aggregate category_summaries")
    source_stage_counts: Counter[str] = Counter()
    complexity_group_counts: Counter[str] = Counter()
    table_rows: list[tuple[str, int | float]] = []

    for raw_summary in category_summaries:
        summary = require_dict(raw_summary, "category summary")
        label = category_label(require_text(summary, "category", "category summary"))
        rows_by_source_stage = require_dict(summary.get("rows_by_source_stage"), f"{label} rows_by_source_stage")
        rows_by_complexity_group = require_dict(
            summary.get("rows_by_complexity_group"),
            f"{label} rows_by_complexity_group",
        )
        source_stage_counts.update({str(key): int(value) for key, value in rows_by_source_stage.items()})
        complexity_group_counts.update({str(key): int(value) for key, value in rows_by_complexity_group.items()})

        table_rows.extend(
            [
                (f"{label} Total Query", int(summary["num_dataset_rows"])),
                (f"{label} Users", int(summary["num_unique_users"])),
            ]
        )

    table_rows.extend(
        [
            ("Total Category Users", sum(int(summary["num_unique_users"]) for summary in category_summaries)),
            ("Total Query", int(aggregate_summary["num_dataset_rows"])),
            ("Correct Rows", source_stage_counts[SOURCE_STAGE_08]),
            ("Writing Typo Rows", source_stage_counts[SOURCE_STAGE_09]),
            ("Low Complexity Rows", sum(value for key, value in complexity_group_counts.items() if key.endswith(":low"))),
            (
                "Medium Complexity Rows",
                sum(value for key, value in complexity_group_counts.items() if key.endswith(":medium")),
            ),
            ("High Complexity Rows", sum(value for key, value in complexity_group_counts.items() if key.endswith(":high"))),
        ]
    )

    print("\n[STATISTICS]\n" + format_two_column_stats_table(table_rows), flush=True)


def load_stage6_syntax_query_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_json_array(path, "Stage 6 syntax-depth query")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row_idx, raw_item in enumerate(rows):
        item = require_dict(raw_item, f"stage6[{row_idx}]")
        user_id = require_text(item, "user_id", f"stage6[{row_idx}]")
        asin = require_text(item, "asin", f"stage6[{row_idx}]")
        query_info = require_dict(item.get("syntax_depth_query"), f"stage6[{row_idx}].syntax_depth_query")
        query = require_text(query_info, "query", f"stage6[{row_idx}].syntax_depth_query")
        attrs_used = require_dict(query_info.get("attrs_used"), f"stage6[{row_idx}].syntax_depth_query.attrs_used")

        key = (user_id, asin)
        if key in index:
            raise ValueError(f"Duplicate Stage 6 syntax-depth key: {key}")
        index[key] = {
            "query": query,
            "attrs_used": attrs_used,
        }
    return index


def load_stage7_original_query_map(path: Path) -> dict[tuple[str, str], str]:
    rows = read_adjacent_json_objects(path, "Stage 7 original query")
    index: dict[tuple[str, str], str] = {}
    for row_idx, item in enumerate(rows):
        user_id = require_text(item, "user_id", f"stage7[{row_idx}]")
        asin = require_text(item, "asin", f"stage7[{row_idx}]")
        original_query = require_text(item, "original_query", f"stage7[{row_idx}]")
        key = (user_id, asin)
        if key in index:
            raise ValueError(f"Duplicate Stage 7 original query key: {key}")
        index[key] = original_query
    return index


def validate_stage8_record(record: dict[str, Any], label: str) -> dict[str, Any]:
    user_id = require_text(record, "user_id", label)
    asin = require_text(record, "asin", label)
    source_group = require_text(record, "syntax_depth_group", label)
    if source_group not in SYNTAX_DEPTH_GROUPS:
        raise ValueError(f"{label}.syntax_depth_group is unsupported: {source_group}")
    return {
        "user_id": user_id,
        "asin": asin,
        "complexity_group": normalize_complexity_group(source_group, f"{label}.syntax_depth_group"),
        "complexity_level": require_int(record, "syntax_depth", label),
        "depth": require_int(record, "target_depth", label),
        "query_length": require_int(record, "query_length", label),
    }


def load_stage8_clean_records(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    data = read_json_object(path, "Stage 8 retrieval summary")
    results = require_list(data.get("all_results_combined"), "Stage 8 all_results_combined")
    by_retriever: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}

    for item_idx, raw_item in enumerate(results):
        item = require_dict(raw_item, f"stage8.all_results_combined[{item_idx}]")
        retriever = require_text(item, "retriever", f"stage8.all_results_combined[{item_idx}]")
        if item.get("query_category") != QUERY_CATEGORY:
            raise ValueError(f"Stage 8 {retriever} has unexpected query_category: {item.get('query_category')}")
        if item.get("query_type") != "correct":
            raise ValueError(f"Stage 8 {retriever} has unexpected query_type: {item.get('query_type')}")
        records = require_list(item.get("all_query_records"), f"Stage 8 {retriever}.all_query_records")
        record_index: dict[tuple[str, str], dict[str, Any]] = {}
        for record_idx, raw_record in enumerate(records):
            record = require_dict(raw_record, f"Stage 8 {retriever}.all_query_records[{record_idx}]")
            normalized = validate_stage8_record(record, f"Stage 8 {retriever}.all_query_records[{record_idx}]")
            key = (normalized["user_id"], normalized["asin"])
            if key in record_index:
                raise ValueError(f"Duplicate Stage 8 key for retriever {retriever}: {key}")
            record_index[key] = normalized
        if retriever in by_retriever:
            raise ValueError(f"Duplicate Stage 8 retriever result: {retriever}")
        by_retriever[retriever] = record_index

    if not by_retriever:
        raise ValueError(f"Stage 8 retrieval summary contains no retriever results: {path}")

    retrievers = sorted(by_retriever)
    canonical_retriever = retrievers[0]
    canonical_keys = set(by_retriever[canonical_retriever])
    for retriever in retrievers[1:]:
        retriever_keys = set(by_retriever[retriever])
        if retriever_keys != canonical_keys:
            raise ValueError(
                f"Stage 8 query key mismatch between {canonical_retriever} and {retriever}: "
                f"canonical_only={len(canonical_keys - retriever_keys)}, retriever_only={len(retriever_keys - canonical_keys)}"
            )

    records = [by_retriever[canonical_retriever][key] for key in sorted(canonical_keys)]
    return records, retrievers


def stage9_payload_from_records(correct_record: dict[str, Any], noisy_record: dict[str, Any], label: str) -> dict[str, Any]:
    if correct_record.get("query_type") != "correct":
        raise ValueError(f"{label} correct record has unexpected query_type: {correct_record.get('query_type')}")
    if noisy_record.get("query_type") != "noisy":
        raise ValueError(f"{label} noisy record has unexpected query_type: {noisy_record.get('query_type')}")
    if correct_record.get("query_category") != QUERY_CATEGORY or noisy_record.get("query_category") != QUERY_CATEGORY:
        raise ValueError(f"{label} record query_category must be {QUERY_CATEGORY}")

    user_id = require_text(correct_record, "user_id", f"{label}.correct")
    asin = require_text(correct_record, "asin", f"{label}.correct")
    if require_text(noisy_record, "user_id", f"{label}.noisy") != user_id:
        raise ValueError(f"{label} correct/noisy user_id mismatch")
    if require_text(noisy_record, "asin", f"{label}.noisy") != asin:
        raise ValueError(f"{label} correct/noisy asin mismatch")

    correct_query = require_text(correct_record, "correct_query", f"{label}.correct")
    if require_text(correct_record, "query", f"{label}.correct") != correct_query:
        raise ValueError(f"{label}.correct.query does not equal correct_query")

    noisy_query = require_text(noisy_record, "noisy_query", f"{label}.noisy")
    if require_text(noisy_record, "query", f"{label}.noisy") != noisy_query:
        raise ValueError(f"{label}.noisy.query does not equal noisy_query")
    if require_text(noisy_record, "correct_query", f"{label}.noisy") != correct_query:
        raise ValueError(f"{label}.noisy.correct_query does not equal correct query")

    source_complexity_group = require_text(correct_record, "syntax_depth_group", f"{label}.correct")
    if source_complexity_group not in STAGE09_SYNTAX_DEPTH_GROUPS:
        raise ValueError(f"{label}.correct.syntax_depth_group is unsupported: {source_complexity_group}")
    if require_text(noisy_record, "syntax_depth_group", f"{label}.noisy") != source_complexity_group:
        raise ValueError(f"{label} correct/noisy syntax_depth_group mismatch")
    complexity_group = normalize_complexity_group(source_complexity_group, f"{label}.correct.syntax_depth_group")

    complexity_level = require_int(correct_record, "syntax_depth", f"{label}.correct")
    if require_int(noisy_record, "syntax_depth", f"{label}.noisy") != complexity_level:
        raise ValueError(f"{label} correct/noisy syntax_depth mismatch")

    depth = require_int(correct_record, "target_depth", f"{label}.correct")
    if require_int(noisy_record, "target_depth", f"{label}.noisy") != depth:
        raise ValueError(f"{label} correct/noisy target_depth mismatch")

    word_count = require_int(correct_record, "word_count", f"{label}.correct")
    if require_int(noisy_record, "word_count", f"{label}.noisy") != word_count:
        raise ValueError(f"{label} correct/noisy word_count mismatch")
    if len(correct_query.split()) != word_count:
        raise ValueError(f"{label} correct_query word count does not match word_count")

    error_attrs = require_dict(correct_record.get("attrs_used"), f"{label}.correct.attrs_used")
    noisy_error_attrs = require_dict(noisy_record.get("attrs_used"), f"{label}.noisy.attrs_used")
    if error_attrs != noisy_error_attrs:
        raise ValueError(f"{label} correct/noisy attrs_used mismatch")

    correct_errors = normalize_injected_errors(correct_record.get("injected_errors"), f"{label}.correct.injected_errors")
    noisy_errors = normalize_injected_errors(noisy_record.get("injected_errors"), f"{label}.noisy.injected_errors")
    if correct_errors != noisy_errors:
        raise ValueError(f"{label} correct/noisy injected_errors mismatch")

    return {
        "user_id": user_id,
        "asin": asin,
        "complexity_group": complexity_group,
        "complexity_level": complexity_level,
        "depth": depth,
        "correct_query": correct_query,
        "error_attrs": error_attrs,
        "error_query": noisy_query,
        "injected_errors": correct_errors,
    }


def merge_stage9_payload(
    index: dict[tuple[str, str], dict[str, Any]],
    key: tuple[str, str],
    payload: dict[str, Any],
    label: str,
) -> None:
    if key not in index:
        index[key] = payload
        return
    existing = index[key]
    comparable_fields = [
        "complexity_group",
        "complexity_level",
        "depth",
        "correct_query",
        "error_attrs",
        "error_query",
        "injected_errors",
    ]
    for field in comparable_fields:
        if existing[field] != payload[field]:
            raise ValueError(f"{label} conflicts with existing Stage 9 payload for key {key} on field {field}")


def load_stage9_noisy_pair_index(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    data = read_json_object(path, "Stage 9 syntax-depth correct/noisy results")
    correct_results = require_list(data.get("correct_results"), "Stage 9 correct_results")
    noisy_results = require_list(data.get("noisy_results"), "Stage 9 noisy_results")

    correct_by_retriever = {}
    noisy_by_retriever = {}
    for item_idx, item in enumerate(correct_results):
        item = require_dict(item, f"Stage 9 correct result[{item_idx}]")
        retriever = require_text(item, "retriever", f"Stage 9 correct result[{item_idx}]")
        if retriever in correct_by_retriever:
            raise ValueError(f"Duplicate Stage 9 correct retriever result: {retriever}")
        correct_by_retriever[retriever] = item
    for item_idx, item in enumerate(noisy_results):
        item = require_dict(item, f"Stage 9 noisy result[{item_idx}]")
        retriever = require_text(item, "retriever", f"Stage 9 noisy result[{item_idx}]")
        if retriever in noisy_by_retriever:
            raise ValueError(f"Duplicate Stage 9 noisy retriever result: {retriever}")
        noisy_by_retriever[retriever] = item

    if set(correct_by_retriever) != set(noisy_by_retriever):
        raise ValueError(
            f"Stage 9 correct/noisy retriever mismatch: "
            f"correct={sorted(correct_by_retriever)}, noisy={sorted(noisy_by_retriever)}"
        )

    pair_index: dict[tuple[str, str], dict[str, Any]] = {}
    for retriever in sorted(correct_by_retriever):
        correct_item = correct_by_retriever[retriever]
        noisy_item = noisy_by_retriever[retriever]
        if correct_item.get("query_category") != QUERY_CATEGORY or noisy_item.get("query_category") != QUERY_CATEGORY:
            raise ValueError(f"Stage 9 {retriever} query_category must be {QUERY_CATEGORY}")
        if correct_item.get("query_type") != "correct" or noisy_item.get("query_type") != "noisy":
            raise ValueError(f"Stage 9 {retriever} query_type mismatch")

        correct_records = {}
        for record_idx, raw_record in enumerate(
            require_list(correct_item.get("all_query_records"), f"Stage 9 {retriever}.correct records")
        ):
            record = require_dict(raw_record, f"Stage 9 {retriever}.correct record[{record_idx}]")
            pair_id = require_int(record, "pair_id", f"Stage 9 {retriever}.correct record[{record_idx}]")
            if pair_id in correct_records:
                raise ValueError(f"Duplicate Stage 9 {retriever} correct pair_id: {pair_id}")
            correct_records[pair_id] = record

        noisy_records = {}
        for record_idx, raw_record in enumerate(
            require_list(noisy_item.get("all_query_records"), f"Stage 9 {retriever}.noisy records")
        ):
            record = require_dict(raw_record, f"Stage 9 {retriever}.noisy record[{record_idx}]")
            pair_id = require_int(record, "pair_id", f"Stage 9 {retriever}.noisy record[{record_idx}]")
            if pair_id in noisy_records:
                raise ValueError(f"Duplicate Stage 9 {retriever} noisy pair_id: {pair_id}")
            noisy_records[pair_id] = record

        if set(correct_records) != set(noisy_records):
            raise ValueError(f"Stage 9 {retriever} correct/noisy pair_id sets differ")

        for pair_id in sorted(correct_records):
            label = f"Stage 9 {retriever} pair_id={pair_id}"
            payload = stage9_payload_from_records(correct_records[pair_id], noisy_records[pair_id], label)
            key = (payload["user_id"], payload["asin"])
            merge_stage9_payload(pair_index, key, payload, label)

    return pair_index, sorted(correct_by_retriever)


def build_stage8_row(
    category: str,
    record: dict[str, Any],
    stage6_index: dict[tuple[str, str], dict[str, Any]],
    stage7_original_query_map: dict[tuple[str, str], str],
) -> dict[str, Any]:
    key = (record["user_id"], record["asin"])
    if key not in stage6_index:
        raise KeyError(f"Stage 8 key is missing from Stage 6 syntax-depth queries: {key}")
    stage6_query = stage6_index[key]
    correct_query = stage7_original_query_map[key] if key in stage7_original_query_map else stage6_query["query"]
    observed_word_count = len(correct_query.split())
    if observed_word_count != record["query_length"]:
        raise ValueError(f"Stage 8 query_length mismatch for {key}: {observed_word_count} != {record['query_length']}")

    return {
        "category": category,
        "uuid": record["user_id"],
        "asin": record["asin"],
        "source_stage": SOURCE_STAGE_08,
        "query_category": QUERY_CATEGORY,
        "complexity_group": record["complexity_group"],
        "complexity_level": record["complexity_level"],
        "depth": record["depth"],
        "correct_query": correct_query,
        "attrs_used": stage6_query["attrs_used"],
        "has_error_query": False,
        "error_query": None,
        "injected_errors": [],
    }


def build_stage9_row(
    category: str,
    payload: dict[str, Any],
    stage6_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    key = (payload["user_id"], payload["asin"])
    if key not in stage6_index:
        raise KeyError(f"Stage 9 key is missing from Stage 6 syntax-depth queries: {key}")
    stage6_query = stage6_index[key]

    return {
        "category": category,
        "uuid": payload["user_id"],
        "asin": payload["asin"],
        "source_stage": SOURCE_STAGE_09,
        "query_category": QUERY_CATEGORY,
        "complexity_group": payload["complexity_group"],
        "complexity_level": payload["complexity_level"],
        "depth": payload["depth"],
        "correct_query": payload["correct_query"],
        "attrs_used": stage6_query["attrs_used"],
        "has_error_query": True,
        "error_query": payload["error_query"],
        "injected_errors": payload["injected_errors"],
    }


def build_dataset_rows(
    category: str,
    stage8_records: list[dict[str, Any]],
    stage6_index: dict[tuple[str, str], dict[str, Any]],
    stage7_original_query_map: dict[tuple[str, str], str],
    stage9_pair_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    seen_keys: set[tuple[str, str, str, str]] = set()

    for record in stage8_records:
        row = build_stage8_row(category, record, stage6_index, stage7_original_query_map)
        key = (row["category"], row["uuid"], row["asin"], row["source_stage"])
        if key in seen_keys:
            raise ValueError(f"Duplicate dataset row key: {key}")
        seen_keys.add(key)
        rows.append(row)

    for key in sorted(stage9_pair_index):
        row = build_stage9_row(category, stage9_pair_index[key], stage6_index)
        dataset_key = (row["category"], row["uuid"], row["asin"], row["source_stage"])
        if dataset_key in seen_keys:
            raise ValueError(f"Duplicate dataset row key: {dataset_key}")
        seen_keys.add(dataset_key)
        rows.append(row)

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            output_row = {field: row[field] for field in OUTPUT_FIELDS}
            f.write(json.dumps(output_row, ensure_ascii=False))
            f.write("\n")


def make_paths(category: str, output_root: Path) -> DatasetPaths:
    output_dir = output_root / category
    return DatasetPaths(
        category=category,
        stage6_syntax_depth_query_file=RESULT_ROOT / "06_query" / category / "query_by_syntax_depth.json",
        stage7_original_query_file=RESULT_ROOT / "07_inject_noisy" / category / "noisy_query.json",
        stage8_retrieval_summary_file=RESULT_ROOT / "08_retrieval" / category / "retrieval_syntax_depth_summary.json",
        stage9_noisy_results_file=RESULT_ROOT / "09_noisy_retrieval" / category / "syntax_depth_correct_vs_noisy_results.json",
        output_dir=output_dir,
        dataset_file=output_dir / "data.jsonl",
        paired_dataset_file=output_dir / "paired_data.jsonl",
        summary_file=output_dir / "summary.json",
    )


def build_one_category(paths: DatasetPaths) -> dict[str, Any]:
    stage6_index = load_stage6_syntax_query_index(paths.stage6_syntax_depth_query_file)
    stage7_original_query_map = load_stage7_original_query_map(paths.stage7_original_query_file)
    stage8_records, stage8_retrievers = load_stage8_clean_records(paths.stage8_retrieval_summary_file)
    stage9_pair_index, stage9_retrievers = load_stage9_noisy_pair_index(paths.stage9_noisy_results_file)

    dataset_rows = build_dataset_rows(
        paths.category,
        stage8_records,
        stage6_index,
        stage7_original_query_map,
        stage9_pair_index,
    )
    paired_rows = [row for row in dataset_rows if row["has_error_query"]]

    write_jsonl(paths.dataset_file, dataset_rows)
    write_jsonl(paths.paired_dataset_file, paired_rows)

    unique_users = {row["uuid"] for row in dataset_rows}
    unique_user_product_pairs = {(row["uuid"], row["asin"]) for row in dataset_rows}
    paired_users = {row["uuid"] for row in paired_rows}
    total_correct_query_words = sum(len(row["correct_query"].split()) for row in dataset_rows)
    total_writing_typo_words = sum(len(row["error_query"].split()) for row in paired_rows)
    avg_correct_query_words = total_correct_query_words / len(dataset_rows)
    avg_writing_typo_words = total_writing_typo_words / len(paired_rows)
    source_stage_counts = Counter(row["source_stage"] for row in dataset_rows)
    paired_source_stage_counts = Counter(row["source_stage"] for row in paired_rows)
    source_stage_user_counts = {
        source_stage: len({row["uuid"] for row in dataset_rows if row["source_stage"] == source_stage})
        for source_stage in sorted(source_stage_counts)
    }
    category_counts = Counter(row["query_category"] for row in dataset_rows)
    paired_category_counts = Counter(row["query_category"] for row in paired_rows)
    complexity_group_counts = Counter(
        f"{row['source_stage']}:{row['complexity_group']}" for row in dataset_rows
    )
    complexity_counts = Counter(
        f"{row['source_stage']}:{row['complexity_group']}:{row['complexity_level']}" for row in dataset_rows
    )
    depth_counts = Counter(f"{row['source_stage']}:{row['depth']}" for row in dataset_rows)
    summary = {
        "category": paths.category,
        "stage6_syntax_depth_query_file": str(paths.stage6_syntax_depth_query_file),
        "stage7_original_query_file": str(paths.stage7_original_query_file),
        "stage8_retrieval_summary_file": str(paths.stage8_retrieval_summary_file),
        "stage9_noisy_results_file": str(paths.stage9_noisy_results_file),
        "stage8_retrievers": stage8_retrievers,
        "stage9_retrievers": stage9_retrievers,
        "dataset_file": str(paths.dataset_file),
        "paired_dataset_file": str(paths.paired_dataset_file),
        "num_stage6_syntax_depth_items": len(stage6_index),
        "num_stage7_original_query_items": len(stage7_original_query_map),
        "num_stage8_clean_items": len(stage8_records),
        "num_stage9_paired_items": len(stage9_pair_index),
        "num_dataset_rows": len(dataset_rows),
        "num_paired_rows": len(paired_rows),
        "num_unpaired_rows": len(dataset_rows) - len(paired_rows),
        "num_unique_users": len(unique_users),
        "num_unique_user_product_pairs": len(unique_user_product_pairs),
        "num_paired_users": len(paired_users),
        "total_correct_query_words": total_correct_query_words,
        "avg_correct_query_words": avg_correct_query_words,
        "total_writing_typo_words": total_writing_typo_words,
        "avg_writing_typo_words": avg_writing_typo_words,
        "rows_by_source_stage": dict(sorted(source_stage_counts.items())),
        "paired_rows_by_source_stage": dict(sorted(paired_source_stage_counts.items())),
        "users_by_source_stage": source_stage_user_counts,
        "rows_by_query_category": dict(sorted(category_counts.items())),
        "paired_rows_by_query_category": dict(sorted(paired_category_counts.items())),
        "rows_by_complexity_group": dict(sorted(complexity_group_counts.items())),
        "rows_by_complexity": dict(sorted(complexity_counts.items())),
        "rows_by_depth": dict(sorted(depth_counts.items())),
    }
    write_json(paths.summary_file, summary)
    return summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    summaries = []

    for category in args.categories:
        paths = make_paths(category, output_root)
        summary = build_one_category(paths)
        summaries.append(summary)
        print(
            f"[{category}] rows={summary['num_dataset_rows']} "
            f"paired={summary['num_paired_rows']} output={summary['dataset_file']}",
            flush=True,
        )

    aggregate_summary = {
        "output_root": str(output_root),
        "categories": [summary["category"] for summary in summaries],
        "num_categories": len(summaries),
        "num_dataset_rows": sum(summary["num_dataset_rows"] for summary in summaries),
        "num_paired_rows": sum(summary["num_paired_rows"] for summary in summaries),
        "total_correct_query_words": sum(summary["total_correct_query_words"] for summary in summaries),
        "total_writing_typo_words": sum(summary["total_writing_typo_words"] for summary in summaries),
        "category_summaries": summaries,
    }
    aggregate_summary["avg_correct_query_words"] = (
        aggregate_summary["total_correct_query_words"] / aggregate_summary["num_dataset_rows"]
    )
    aggregate_summary["avg_writing_typo_words"] = (
        aggregate_summary["total_writing_typo_words"] / aggregate_summary["num_paired_rows"]
    )
    write_json(output_root / "summary.json", aggregate_summary)
    print(
        f"[ALL] categories={aggregate_summary['num_categories']} "
        f"rows={aggregate_summary['num_dataset_rows']} "
        f"paired={aggregate_summary['num_paired_rows']} "
        f"summary={output_root / 'summary.json'}",
        flush=True,
    )
    print_statistics_table(aggregate_summary)


if __name__ == "__main__":
    main()
