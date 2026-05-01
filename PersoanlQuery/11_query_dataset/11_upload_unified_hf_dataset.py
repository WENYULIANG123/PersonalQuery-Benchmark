#!/usr/bin/env python3
"""Prepare and upload a unified Hugging Face dataset repository with three configs."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


SOURCE_ROOT = Path("/home/wlia0047/ar57/wenyu/result/personal_query/11_query_dataset")
UPLOAD_ROOT = Path("/home/wlia0047/ar57/wenyu/result/personal_query/12_personalized_query_hf")
REPO_NAME = "personalized-query"


@dataclass(frozen=True)
class UnifiedConfig:
    config_name: str
    category: str
    title: str


CONFIGS = (
    UnifiedConfig("baby", "Baby_Products", "Baby Products"),
    UnifiedConfig("grocery", "Grocery_and_Gourmet_Food", "Grocery and Gourmet Food"),
    UnifiedConfig("pets", "Pet_Supplies", "Pet Supplies"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload one unified Hugging Face dataset repository.")
    parser.add_argument("--namespace", required=True, help="Hugging Face user or organization namespace.")
    parser.add_argument("--create-repo", action="store_true", help="Create the public dataset repo before upload.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return data


def required_upload_files() -> set[Path]:
    files = {Path("README.md"), Path("summary.json")}
    for config in CONFIGS:
        files.add(Path(config.category) / "data.jsonl")
        files.add(Path(config.category) / "paired_data.jsonl")
        files.add(Path(config.category) / "summary.json")
    return files


def assert_upload_root_clean(expected_files: set[Path]) -> None:
    if not UPLOAD_ROOT.exists():
        return
    if not UPLOAD_ROOT.is_dir():
        raise NotADirectoryError(f"Upload root exists but is not a directory: {UPLOAD_ROOT}")
    existing_files = {path.relative_to(UPLOAD_ROOT) for path in UPLOAD_ROOT.rglob("*") if path.is_file()}
    unexpected_files = sorted(existing_files - expected_files)
    if unexpected_files:
        raise RuntimeError(f"Upload root contains unexpected files: {unexpected_files}")


def copy_required_files() -> None:
    expected_files = required_upload_files()
    assert_upload_root_clean(expected_files)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SOURCE_ROOT / "summary.json", UPLOAD_ROOT / "summary.json")
    for config in CONFIGS:
        source_dir = SOURCE_ROOT / config.category
        target_dir = UPLOAD_ROOT / config.category
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("data.jsonl", "paired_data.jsonl", "summary.json"):
            source_file = source_dir / filename
            if not source_file.is_file():
                raise FileNotFoundError(f"Required source file does not exist: {source_file}")
            if source_file.stat().st_size == 0:
                raise ValueError(f"Required source file is empty: {source_file}")
            shutil.copy2(source_file, target_dir / filename)


def format_dict_table(values: dict[str, Any], key_title: str, value_title: str) -> str:
    if not values:
        raise ValueError("Cannot format an empty table")
    lines = [f"| {key_title} | {value_title} |", "|---|---:|"]
    for key in sorted(values):
        lines.append(f"| `{key}` | {values[key]} |")
    return "\n".join(lines)


def build_size_table(category_summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| Config | Category | Full Rows | Paired Rows | Unpaired Rows |",
        "|---|---|---:|---:|---:|",
    ]
    for config, summary in zip(CONFIGS, category_summaries, strict=True):
        lines.append(
            f"| `{config.config_name}` | {config.title} | {summary['num_dataset_rows']} | "
            f"{summary['num_paired_rows']} | {summary['num_unpaired_rows']} |"
        )
    return "\n".join(lines)


def build_card(namespace: str, aggregate_summary: dict[str, Any]) -> str:
    category_summaries = aggregate_summary["category_summaries"]
    if len(category_summaries) != len(CONFIGS):
        raise ValueError("Aggregate summary category count does not match configured categories")

    detail_sections = []
    for config, summary in zip(CONFIGS, category_summaries, strict=True):
        if summary["category"] != config.category:
            raise ValueError(f"Summary category mismatch: {summary['category']} != {config.category}")
        detail_sections.append(
            f"""### `{config.config_name}`: {config.title}

Rows by query category:

{format_dict_table(summary["rows_by_query_category"], "query_category", "rows")}

Paired rows by query category:

{format_dict_table(summary["paired_rows_by_query_category"], "query_category", "rows")}

Rows by complexity:

{format_dict_table(summary["rows_by_complexity"], "query_category:level", "rows")}
"""
        )

    return f"""---
language:
- en
license: other
pretty_name: "Personalized Query"
task_categories:
- text-generation
- text-retrieval
tags:
- personalized-search
- personalized-query
- query-generation
- error-query
- synthetic-data
- amazon-reviews
configs:
- config_name: baby
  data_files:
  - split: full
    path: Baby_Products/data.jsonl
  - split: paired
    path: Baby_Products/paired_data.jsonl
- config_name: grocery
  data_files:
  - split: full
    path: Grocery_and_Gourmet_Food/data.jsonl
  - split: paired
    path: Grocery_and_Gourmet_Food/paired_data.jsonl
- config_name: pets
  data_files:
  - split: full
    path: Pet_Supplies/data.jsonl
  - split: paired
    path: Pet_Supplies/paired_data.jsonl
---

# Personalized Query

This repository contains three personalized product-search query datasets in one Hugging Face dataset page.

Each config corresponds to one product category:

- `baby`: Baby Products
- `grocery`: Grocery and Gourmet Food
- `pets`: Pet Supplies

Each config has two splits:

- `full`: all correct Stage 6 queries. Rows without Stage 7 error query keep `error_query` as `null`.
- `paired`: only rows where a correct query has a paired error query.

## Dataset Size

{build_size_table(category_summaries)}

Total rows across all full splits: {aggregate_summary["num_dataset_rows"]}

Total paired rows: {aggregate_summary["num_paired_rows"]}

## Category Details

{chr(10).join(detail_sections)}

## Schema

| Field | Type | Description |
|---|---|---|
| `category` | string | Product category. |
| `uuid` | string | User identifier. |
| `asin` | string | Amazon product identifier. |
| `query_category` | string | Query type: `wide` or `deep`. |
| `complexity_level` | integer | Complexity level of the generated Stage 6 query. |
| `profile_complexity_level` | integer | User profile complexity level from Stage 5. |
| `correct_query` | string | Correct query generated in Stage 6. |
| `correct_word_count` | integer | Word count of `correct_query`. |
| `attrs_used` | object | Product attributes used to generate the query. |
| `has_error_query` | boolean | Whether a Stage 7 error query is available. |
| `error_query` | string or null | Error query generated in Stage 7. |
| `injected_errors` | list | Injected error metadata with fixed fields: `correct`, `error`, `error_type`, `note`. |

## Loading

```python
from datasets import load_dataset

baby_full = load_dataset("{namespace}/{REPO_NAME}", name="baby", split="full")
baby_paired = load_dataset("{namespace}/{REPO_NAME}", name="baby", split="paired")

grocery_full = load_dataset("{namespace}/{REPO_NAME}", name="grocery", split="full")
pets_full = load_dataset("{namespace}/{REPO_NAME}", name="pets", split="full")
```

## Source Pipeline

- Stage 5 provides the user profile complexity level.
- Stage 6 generates correct personalized queries.
- Stage 7 injects user-specific error query variants when a matching error pattern is available.

## Intended Use

This dataset is intended for research on personalized product search, query generation, error-query robustness, and retrieval evaluation.

## Data Notes

The queries are synthetic outputs generated from user/product signals in the local Personal Query pipeline. The error queries are generated by injecting user-specific writing error patterns. Review license and redistribution requirements for any upstream source data before external reuse.
"""


def write_readme(namespace: str) -> None:
    aggregate_summary = read_json(SOURCE_ROOT / "summary.json")
    readme = build_card(namespace, aggregate_summary)
    (UPLOAD_ROOT / "README.md").write_text(readme, encoding="utf-8")


def verify_upload_root() -> None:
    expected_files = required_upload_files()
    actual_files = {path.relative_to(UPLOAD_ROOT) for path in UPLOAD_ROOT.rglob("*") if path.is_file()}
    missing_files = sorted(expected_files - actual_files)
    unexpected_files = sorted(actual_files - expected_files)
    if missing_files:
        raise RuntimeError(f"Upload root is missing required files: {missing_files}")
    if unexpected_files:
        raise RuntimeError(f"Upload root contains unexpected files: {unexpected_files}")


def upload(namespace: str, create_repo: bool) -> str:
    repo_id = f"{namespace}/{REPO_NAME}"
    api = HfApi()
    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=False)
        print(f"[CREATED] repo_id={repo_id}", flush=True)

    commit_info = api.upload_folder(
        folder_path=str(UPLOAD_ROOT),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload unified personalized query dataset",
    )
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    expected_remote_files = {str(path) for path in required_upload_files()}
    missing_remote_files = sorted(expected_remote_files - remote_files)
    if missing_remote_files:
        raise RuntimeError(f"Upload verification failed for {repo_id}, missing files: {missing_remote_files}")
    print(f"[UPLOADED] repo_id={repo_id} commit={commit_info.oid}", flush=True)
    return repo_id


def main() -> None:
    args = parse_args()
    copy_required_files()
    write_readme(args.namespace)
    verify_upload_root()
    repo_id = upload(args.namespace, args.create_repo)
    print(f"[SUMMARY] repo_id={repo_id} local_upload_root={UPLOAD_ROOT}", flush=True)


if __name__ == "__main__":
    main()
