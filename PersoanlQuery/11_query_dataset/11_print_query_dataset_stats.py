#!/usr/bin/env python3
"""Print Stage 11 query dataset statistics from summary.json."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_FILE = Path("/home/wlia0047/ar57/wenyu/result/personal_query/11_query_dataset/summary.json")
SOURCE_STAGE_08 = "08_clean"
SOURCE_STAGE_09 = "09_noisy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Stage 11 query dataset statistics.")
    parser.add_argument("--summary-file", default=str(DEFAULT_SUMMARY_FILE), help="Path to Stage 11 summary.json.")
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


def require_int_value(item: dict[str, Any], key: str, label: str) -> int:
    if key not in item:
        raise KeyError(f"{label} is missing required key: {key}")
    value = item[key]
    if not isinstance(value, int):
        raise TypeError(f"{label}.{key} must be an integer")
    return value


def require_number_value(item: dict[str, Any], key: str, label: str) -> float:
    if key not in item:
        raise KeyError(f"{label} is missing required key: {key}")
    value = item[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{label}.{key} must be numeric")
    return float(value)


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Summary file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return require_dict(json.load(handle), str(path))


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


def build_statistics_rows(summary: dict[str, Any]) -> list[tuple[str, int | float]]:
    category_summaries = require_list(summary.get("category_summaries"), "summary.category_summaries")
    source_stage_counts: Counter[str] = Counter()
    complexity_group_counts: Counter[str] = Counter()
    table_rows: list[tuple[str, int | float]] = []

    for summary_index, raw_category_summary in enumerate(category_summaries):
        category_summary = require_dict(raw_category_summary, f"category_summaries[{summary_index}]")
        category = require_text(category_summary, "category", f"category_summaries[{summary_index}]")
        label = category_label(category)

        rows_by_source_stage = require_dict(
            category_summary.get("rows_by_source_stage"),
            f"{label}.rows_by_source_stage",
        )
        rows_by_complexity_group = require_dict(
            category_summary.get("rows_by_complexity_group"),
            f"{label}.rows_by_complexity_group",
        )
        source_stage_counts.update({str(key): int(value) for key, value in rows_by_source_stage.items()})
        complexity_group_counts.update({str(key): int(value) for key, value in rows_by_complexity_group.items()})

        table_rows.extend(
            [
                (f"{label} Total Query", require_int_value(category_summary, "num_dataset_rows", label)),
                (f"{label} Users", require_int_value(category_summary, "num_unique_users", label)),
            ]
        )

    table_rows.extend(
        [
            ("Total Category Users", sum(require_int_value(item, "num_unique_users", "category summary") for item in category_summaries)),
            ("Total Query", require_int_value(summary, "num_dataset_rows", "summary")),
            ("Correct Rows", source_stage_counts[SOURCE_STAGE_08]),
            ("Writing Typo Rows", source_stage_counts[SOURCE_STAGE_09]),
            ("Low Complexity Rows", sum(value for key, value in complexity_group_counts.items() if key.endswith(":low"))),
            ("Medium Complexity Rows", sum(value for key, value in complexity_group_counts.items() if key.endswith(":medium"))),
            ("High Complexity Rows", sum(value for key, value in complexity_group_counts.items() if key.endswith(":high"))),
        ]
    )
    return table_rows


def main() -> None:
    args = parse_args()
    summary_file = Path(args.summary_file).expanduser().resolve()
    summary = load_summary(summary_file)
    print("[STATISTICS]")
    print(format_two_column_stats_table(build_statistics_rows(summary)))


if __name__ == "__main__":
    main()
