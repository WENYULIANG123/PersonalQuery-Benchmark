#!/usr/bin/env python3
"""Build per-category clean query datasets grouped by query cluster."""

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

OUTPUT_FIELDS = [
    "category",
    "uuid",
    "asin",
    "cluster_label",
    "cluster_index",
    "correct_query",
    "attrs_used",
]


@dataclass(frozen=True)
class DatasetPaths:
    category: str
    stage6_query_file: Path
    cluster_profile_file: Path
    output_dir: Path
    dataset_file: Path
    summary_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one clean JSONL dataset per category from latest Stage 06 queries and GMM query clusters."
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


def read_json_array(path: Path, label: str) -> list[Any]:
    require_file(path, label)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return require_list(data, label)


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    require_file(path, label)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"{label}:{line_number} is empty")
            rows.append(require_dict(json.loads(stripped), f"{label}:{line_number}"))
    if not rows:
        raise ValueError(f"{label} contains no rows: {path}")
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            output_row = {field: row[field] for field in OUTPUT_FIELDS}
            f.write(json.dumps(output_row, ensure_ascii=False))
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


def load_stage6_query_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_json_array(path, "Stage 06 latest query file")
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
            raise ValueError(f"Duplicate Stage 06 key: {key}")
        index[key] = {
            "query": query,
            "attrs_used": attrs_used,
        }
    return index


def load_cluster_profile_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(path, "Cluster user profiles")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row_idx, item in enumerate(rows):
        user_id = require_text(item, "user_id", f"cluster_profiles[{row_idx}]")
        asin = require_text(item, "asin", f"cluster_profiles[{row_idx}]")
        cluster_label = require_text(item, "cluster_label", f"cluster_profiles[{row_idx}]")
        cluster_index = require_int(item, "cluster_index", f"cluster_profiles[{row_idx}]")
        key = (user_id, asin)
        if key in index:
            raise ValueError(f"Duplicate cluster profile key: {key}")
        index[key] = {
            "cluster_label": cluster_label,
            "cluster_index": cluster_index,
        }
    return index


def build_dataset_rows(
    category: str,
    stage6_index: dict[tuple[str, str], dict[str, Any]],
    cluster_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    if set(stage6_index) != set(cluster_index):
        missing_cluster = sorted(set(stage6_index) - set(cluster_index))
        extra_cluster = sorted(set(cluster_index) - set(stage6_index))
        raise ValueError(
            f"Stage 06 / cluster profile key mismatch for {category}: "
            f"missing_cluster={len(missing_cluster)}, extra_cluster={len(extra_cluster)}"
        )

    rows: list[dict[str, Any]] = []
    for key in sorted(stage6_index):
        user_id, asin = key
        stage6_query = stage6_index[key]
        cluster_profile = cluster_index[key]
        rows.append(
            {
                "category": category,
                "uuid": user_id,
                "asin": asin,
                "cluster_label": cluster_profile["cluster_label"],
                "cluster_index": cluster_profile["cluster_index"],
                "correct_query": stage6_query["query"],
                "attrs_used": stage6_query["attrs_used"],
            }
        )
    return rows


def make_paths(category: str, output_root: Path) -> DatasetPaths:
    output_dir = output_root / category
    return DatasetPaths(
        category=category,
        stage6_query_file=(
            RESULT_ROOT
            / "06_query"
            / category
            / "query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json"
        ),
        cluster_profile_file=(
            RESULT_ROOT
            / "12_complexity_analysis_clause_features"
            / category
            / "strict5550_query_gmm_user_profiles.jsonl"
        ),
        output_dir=output_dir,
        dataset_file=output_dir / "data.jsonl",
        summary_file=output_dir / "summary.json",
    )


def build_one_category(paths: DatasetPaths) -> dict[str, Any]:
    stage6_index = load_stage6_query_index(paths.stage6_query_file)
    cluster_index = load_cluster_profile_index(paths.cluster_profile_file)
    dataset_rows = build_dataset_rows(paths.category, stage6_index, cluster_index)

    write_jsonl(paths.dataset_file, dataset_rows)

    unique_users = {row["uuid"] for row in dataset_rows}
    unique_user_product_pairs = {(row["uuid"], row["asin"]) for row in dataset_rows}
    total_query_words = sum(len(row["correct_query"].split()) for row in dataset_rows)
    avg_query_words = total_query_words / len(dataset_rows)
    cluster_label_counts = Counter(row["cluster_label"] for row in dataset_rows)
    cluster_index_counts = Counter(str(row["cluster_index"]) for row in dataset_rows)

    summary = {
        "category": paths.category,
        "stage6_query_file": str(paths.stage6_query_file),
        "cluster_profile_file": str(paths.cluster_profile_file),
        "dataset_file": str(paths.dataset_file),
        "num_stage6_items": len(stage6_index),
        "num_cluster_profile_items": len(cluster_index),
        "num_dataset_rows": len(dataset_rows),
        "num_unique_users": len(unique_users),
        "num_unique_user_product_pairs": len(unique_user_product_pairs),
        "total_query_words": total_query_words,
        "avg_query_words": avg_query_words,
        "rows_by_cluster_label": dict(sorted(cluster_label_counts.items())),
        "rows_by_cluster_index": dict(sorted(cluster_index_counts.items(), key=lambda item: int(item[0]))),
    }
    write_json(paths.summary_file, summary)
    return summary


def print_statistics_table(aggregate_summary: dict[str, Any]) -> None:
    category_summaries = require_list(aggregate_summary.get("category_summaries"), "aggregate category_summaries")
    cluster_counts: Counter[str] = Counter()
    table_rows: list[tuple[str, int | float]] = []

    for raw_summary in category_summaries:
        summary = require_dict(raw_summary, "category summary")
        label = category_label(require_text(summary, "category", "category summary"))
        rows_by_cluster_label = require_dict(summary.get("rows_by_cluster_label"), f"{label} rows_by_cluster_label")
        cluster_counts.update({str(key): int(value) for key, value in rows_by_cluster_label.items()})
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
        ]
    )
    for cluster_label in sorted(cluster_counts):
        table_rows.append((f"{cluster_label} Rows", cluster_counts[cluster_label]))

    print("\n[STATISTICS]\n" + format_two_column_stats_table(table_rows), flush=True)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    summaries = []

    for category in args.categories:
        paths = make_paths(category, output_root)
        summary = build_one_category(paths)
        summaries.append(summary)
        print(f"[{category}] rows={summary['num_dataset_rows']} output={summary['dataset_file']}", flush=True)

    aggregate_summary = {
        "output_root": str(output_root),
        "categories": [summary["category"] for summary in summaries],
        "num_categories": len(summaries),
        "num_dataset_rows": sum(summary["num_dataset_rows"] for summary in summaries),
        "total_query_words": sum(summary["total_query_words"] for summary in summaries),
        "category_summaries": summaries,
    }
    aggregate_summary["avg_query_words"] = aggregate_summary["total_query_words"] / aggregate_summary["num_dataset_rows"]
    write_json(output_root / "summary.json", aggregate_summary)
    print(
        f"[ALL] categories={aggregate_summary['num_categories']} "
        f"rows={aggregate_summary['num_dataset_rows']} "
        f"summary={output_root / 'summary.json'}",
        flush=True,
    )
    print_statistics_table(aggregate_summary)


if __name__ == "__main__":
    main()
