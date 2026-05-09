#!/usr/bin/env python3
"""Build per-category clean/noisy query datasets from Stage 6 and Stage 7 outputs."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")
RESULT_ROOT = REPO_ROOT / "result" / "personal_query"
DATASET_ROOT = REPO_ROOT / "dataset"
AMAZON_REVIEWS_ROOT = REPO_ROOT / "data" / "Amazon-Reviews-2023"
IDF_CACHE_DIR = RESULT_ROOT / "11_query_dataset" / "idf_cache"
UNKNOWN_WORD_IDF = 5.0
SOURCE_TO_DATASET_QUERY_CATEGORY = {
    "acl": "wide",
    "ccomp": "deep",
}
INJECTED_ERROR_FIELDS = ["correct", "error", "error_type"]
INJECTED_ERROR_REQUIRED_FIELDS = {"correct", "error", "error_type"}
METADATA_FILES = {
    "Baby_Products": AMAZON_REVIEWS_ROOT / "meta_Baby_Products.jsonl.gz",
    "Grocery_and_Gourmet_Food": AMAZON_REVIEWS_ROOT / "raw" / "meta_categories" / "meta_Grocery_and_Gourmet_Food.jsonl.gz",
    "Pet_Supplies": AMAZON_REVIEWS_ROOT / "raw" / "meta_categories" / "meta_Pet_Supplies.jsonl.gz",
}


@dataclass(frozen=True)
class DatasetPaths:
    category: str
    clean_query_file: Path
    noisy_query_file: Path
    level_file: Path
    output_dir: Path
    dataset_file: Path
    paired_dataset_file: Path
    summary_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one JSONL dataset per category from Stage 6 clean queries and Stage 7 noisy queries."
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
    if not isinstance(value, str) or not value:
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


def load_level_index(path: Path) -> dict[str, dict[str, int]]:
    rows = read_json_array(path, "level")
    level_index: dict[str, dict[str, int]] = {}
    for idx, raw_item in enumerate(rows):
        item = require_dict(raw_item, f"level[{idx}]")
        user_id = require_text(item, "user_id", f"level[{idx}]")
        acl_level = require_int(item, "acl_level", f"level[{idx}]")
        ccomp_level = require_int(item, "ccomp_level", f"level[{idx}]")
        if user_id in level_index:
            raise ValueError(f"Duplicate user_id in level file: {user_id}")
        level_index[user_id] = {"acl": acl_level, "ccomp": ccomp_level}
    return level_index


def build_noisy_index(noisy_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    noisy_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for idx, item in enumerate(noisy_rows):
        label = f"noisy[{idx}]"
        user_id = require_text(item, "user_id", label)
        asin = require_text(item, "asin", label)
        query_category = require_text(item, "query_category", label)
        if query_category not in {"acl", "ccomp"}:
            raise ValueError(f"{label}.query_category must be 'acl' or 'ccomp': {query_category}")
        require_text(item, "ground_truth_query", label)
        require_text(item, "noisy_query", label)
        if "injected_errors" not in item:
            raise KeyError(f"{label} is missing required key: injected_errors")
        if not isinstance(item["injected_errors"], list):
            raise TypeError(f"{label}.injected_errors must be a list")

        key = (user_id, asin, query_category)
        if key in noisy_index:
            raise ValueError(f"Duplicate noisy query key: {key}")
        noisy_index[key] = item
    return noisy_index


def validate_query_info(info: dict[str, Any], label: str) -> tuple[int, str, int, dict[str, Any]]:
    level = require_int(info, "level", label)
    query = require_text(info, "query", label)
    word_count = require_int(info, "word_count", label)
    if "attrs_used" not in info:
        raise KeyError(f"{label} is missing required key: attrs_used")
    attrs_used = require_dict(info["attrs_used"], f"{label}.attrs_used")
    return level, query, word_count, attrs_used


def normalize_injected_errors(errors: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(errors, list):
        raise TypeError(f"{label} must be a list")

    normalized_errors = []
    for idx, raw_error in enumerate(errors):
        error_label = f"{label}[{idx}]"
        error = require_dict(raw_error, error_label)
        unknown_fields = sorted(set(error) - set(INJECTED_ERROR_FIELDS))
        if unknown_fields:
            raise ValueError(f"{error_label} contains unknown fields: {unknown_fields}")
        missing_required_fields = sorted(INJECTED_ERROR_REQUIRED_FIELDS - set(error))
        if missing_required_fields:
            raise KeyError(f"{error_label} is missing required fields: {missing_required_fields}")

        normalized_error = {}
        for field in INJECTED_ERROR_FIELDS:
            value = error[field] if field in error else None
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{error_label}.{field} must be a string or null")
            normalized_error[field] = value
        normalized_errors.append(normalized_error)
    return normalized_errors


def get_metadata_file(category: str) -> Path:
    if category not in METADATA_FILES:
        raise KeyError(f"Missing metadata file configuration for category: {category}")
    metadata_file = METADATA_FILES[category]
    require_file(metadata_file, f"{category} metadata")
    return metadata_file


def idf_cache_file(category: str) -> Path:
    return IDF_CACHE_DIR / f"{category}_full_word_idf.json"


def idf_cache_meta_file(category: str) -> Path:
    return IDF_CACHE_DIR / f"{category}_full_word_idf.meta.json"


def metadata_signature(metadata_file: Path) -> dict[str, Any]:
    stat = metadata_file.stat()
    return {
        "metadata_file": str(metadata_file),
        "metadata_size": stat.st_size,
        "metadata_mtime_ns": stat.st_mtime_ns,
        "idf_mode": "full_metadata",
        "formula": "log(N / (df + 1)); unknown query token idf = 5.0",
        "text_fields": ["title", "brand", "description"],
    }


def load_cached_word_idf(category: str, metadata_file: Path) -> dict[str, float] | None:
    cache_file = idf_cache_file(category)
    cache_meta_file = idf_cache_meta_file(category)
    if not cache_file.is_file() or not cache_meta_file.is_file():
        return None
    cache_meta = read_json_object(cache_meta_file, f"{category} IDF cache metadata")
    expected_signature = metadata_signature(metadata_file)
    if cache_meta.get("signature") != expected_signature:
        return None
    with cache_file.open("r", encoding="utf-8") as f:
        cached_word_idf = json.load(f)
    if not isinstance(cached_word_idf, dict):
        raise TypeError(f"IDF cache must contain a JSON object: {cache_file}")
    word_idf = {}
    for word, value in cached_word_idf.items():
        if not isinstance(word, str):
            raise TypeError(f"IDF cache contains a non-string token in {cache_file}")
        if not isinstance(value, (int, float)):
            raise TypeError(f"IDF cache value for token {word!r} must be numeric")
        word_idf[word] = float(value)
    print(f"[IDF] loaded cache category={category} tokens={len(word_idf)} file={cache_file}", flush=True)
    return word_idf


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return require_dict(data, label)


def extract_metadata_text(item: dict[str, Any]) -> str:
    description = item.get("description")
    if description is None:
        description_text = ""
    elif isinstance(description, list):
        description_text = " ".join(str(part) for part in description if part is not None)
    elif isinstance(description, str):
        description_text = description
    else:
        raise TypeError(f"Unexpected description type in metadata: {type(description).__name__}")

    fields = [
        item.get("title", ""),
        item.get("brand", ""),
        description_text,
    ]
    return " ".join(str(field) for field in fields if field)


def build_full_word_idf(metadata_file: Path) -> tuple[dict[str, float], int]:
    word_doc_freq: Counter[str] = Counter()
    total_docs = 0
    with gzip.open(metadata_file, "rt", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            item = require_dict(item, f"metadata line {total_docs + 1}")
            words = set(extract_metadata_text(item).lower().split())
            for word in words:
                if len(word) > 1:
                    word_doc_freq[word] += 1
            total_docs += 1
            if total_docs % 100000 == 0:
                print(f"[IDF] scanned docs={total_docs} metadata={metadata_file}", flush=True)

    if total_docs == 0:
        raise ValueError(f"Metadata file contains no documents: {metadata_file}")

    word_idf = {word: math.log(total_docs / (df + 1)) for word, df in word_doc_freq.items()}
    rare_word_floor = math.log(total_docs / 10)
    for word, df in word_doc_freq.items():
        if len(word) >= 4 and df < 10:
            word_idf[word] = max(word_idf[word], rare_word_floor)
    return word_idf, total_docs


def save_word_idf_cache(category: str, metadata_file: Path, word_idf: dict[str, float], total_docs: int) -> None:
    IDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = idf_cache_file(category)
    cache_meta_file = idf_cache_meta_file(category)
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(word_idf.items())), f, ensure_ascii=False)
        f.write("\n")
    cache_meta = {
        "category": category,
        "total_docs": total_docs,
        "num_tokens": len(word_idf),
        "signature": metadata_signature(metadata_file),
    }
    write_json(cache_meta_file, cache_meta)
    print(
        f"[IDF] saved cache category={category} docs={total_docs} tokens={len(word_idf)} file={cache_file}",
        flush=True,
    )


def get_or_build_word_idf(category: str) -> dict[str, float]:
    metadata_file = get_metadata_file(category)
    cached_word_idf = load_cached_word_idf(category, metadata_file)
    if cached_word_idf is not None:
        return cached_word_idf
    print(f"[IDF] building full metadata IDF category={category} metadata={metadata_file}", flush=True)
    word_idf, total_docs = build_full_word_idf(metadata_file)
    save_word_idf_cache(category, metadata_file, word_idf, total_docs)
    return word_idf


def compute_query_idf(query_text: str, word_idf: dict[str, float]) -> float:
    words = query_text.lower().split()
    if not words:
        return 0.0
    idf_values = [word_idf.get(word, UNKNOWN_WORD_IDF) for word in words]
    return float(sum(idf_values) / len(idf_values))


def build_dataset_rows(
    category: str,
    clean_rows: list[Any],
    noisy_index: dict[tuple[str, str, str], dict[str, Any]],
    word_idf: dict[str, float],
) -> list[dict[str, Any]]:
    dataset_rows: list[dict[str, Any]] = []
    seen_sample_keys: set[tuple[str, str, str, str]] = set()
    clean_source_keys: set[tuple[str, str, str]] = set()

    for idx, raw_item in enumerate(clean_rows):
        item = require_dict(raw_item, f"clean[{idx}]")
        user_id = require_text(item, "user_id", f"clean[{idx}]")
        asin = require_text(item, "asin", f"clean[{idx}]")

        for source_query_category, query_key in (("acl", "acl_query"), ("ccomp", "ccomp_query")):
            if query_key not in item:
                raise KeyError(f"clean[{idx}] is missing required key: {query_key}")
            query_info = require_dict(item[query_key], f"clean[{idx}].{query_key}")
            complexity_level, clean_query, word_count, attrs_used = validate_query_info(
                query_info, f"clean[{idx}].{query_key}"
            )
            noisy_key = (user_id, asin, source_query_category)
            noisy_item = noisy_index.get(noisy_key)
            dataset_query_category = SOURCE_TO_DATASET_QUERY_CATEGORY[source_query_category]

            sample_key = (category, user_id, asin, source_query_category)
            if sample_key in seen_sample_keys:
                raise ValueError(f"Duplicate sample key: {sample_key}")
            seen_sample_keys.add(sample_key)
            clean_source_keys.add((user_id, asin, source_query_category))

            row = {
                "category": category,
                "uuid": user_id,
                "asin": asin,
                "query_category": dataset_query_category,
                "complexity_level": complexity_level,
                "correct_query": clean_query,
                "correct_word_count": word_count,
                "idf": compute_query_idf(clean_query, word_idf),
                "attrs_used": attrs_used,
                "has_error_query": noisy_item is not None,
                "error_query": None,
                "injected_errors": [],
            }

            if noisy_item is not None:
                ground_truth_query = require_text(noisy_item, "ground_truth_query", f"noisy item {noisy_key}")
                if ground_truth_query != clean_query:
                    raise ValueError(
                        "Stage 7 ground_truth_query does not match Stage 6 clean query for "
                        f"{noisy_key}: {ground_truth_query!r} != {clean_query!r}"
                    )
                row["error_query"] = require_text(noisy_item, "noisy_query", f"noisy item {noisy_key}")
                row["injected_errors"] = normalize_injected_errors(
                    noisy_item["injected_errors"], f"noisy item {noisy_key}.injected_errors"
                )

            dataset_rows.append(row)

    noisy_keys = set(noisy_index)
    unmatched_noisy_keys = sorted(noisy_keys - clean_source_keys)
    if unmatched_noisy_keys:
        raise ValueError(f"Found {len(unmatched_noisy_keys)} noisy rows without Stage 6 clean rows: {unmatched_noisy_keys[:5]}")

    return dataset_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def make_paths(category: str, output_root: Path) -> DatasetPaths:
    output_dir = output_root / category
    return DatasetPaths(
        category=category,
        clean_query_file=RESULT_ROOT / "06_query" / category / "query.json",
        noisy_query_file=RESULT_ROOT / "07_inject_noisy" / category / "noisy_query.json",
        level_file=RESULT_ROOT / "05_syntactic_analysis" / category / "level.json",
        output_dir=output_dir,
        dataset_file=output_dir / "data.jsonl",
        paired_dataset_file=output_dir / "paired_data.jsonl",
        summary_file=output_dir / "summary.json",
    )


def build_one_category(paths: DatasetPaths) -> dict[str, Any]:
    clean_rows = read_json_array(paths.clean_query_file, f"{paths.category} clean query")
    noisy_rows = read_adjacent_json_objects(paths.noisy_query_file, f"{paths.category} noisy query")
    noisy_index = build_noisy_index(noisy_rows)
    word_idf = get_or_build_word_idf(paths.category)
    dataset_rows = build_dataset_rows(paths.category, clean_rows, noisy_index, word_idf)
    paired_rows = [row for row in dataset_rows if row["has_error_query"]]

    write_jsonl(paths.dataset_file, dataset_rows)
    write_jsonl(paths.paired_dataset_file, paired_rows)

    category_counts = Counter(row["query_category"] for row in dataset_rows)
    paired_category_counts = Counter(row["query_category"] for row in paired_rows)
    complexity_counts = Counter(
        f"{row['query_category']}:{row['complexity_level']}" for row in dataset_rows
    )
    summary = {
        "category": paths.category,
        "clean_query_file": str(paths.clean_query_file),
        "noisy_query_file": str(paths.noisy_query_file),
        "level_file": str(paths.level_file),
        "dataset_file": str(paths.dataset_file),
        "paired_dataset_file": str(paths.paired_dataset_file),
        "num_clean_stage6_items": len(clean_rows),
        "num_noisy_stage7_items": len(noisy_rows),
        "num_dataset_rows": len(dataset_rows),
        "num_paired_rows": len(paired_rows),
        "num_unpaired_rows": len(dataset_rows) - len(paired_rows),
        "rows_by_query_category": dict(sorted(category_counts.items())),
        "paired_rows_by_query_category": dict(sorted(paired_category_counts.items())),
        "rows_by_complexity": dict(sorted(complexity_counts.items())),
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
        "category_summaries": summaries,
    }
    write_json(output_root / "summary.json", aggregate_summary)
    print(
        f"[ALL] categories={aggregate_summary['num_categories']} "
        f"rows={aggregate_summary['num_dataset_rows']} "
        f"paired={aggregate_summary['num_paired_rows']} "
        f"summary={output_root / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
