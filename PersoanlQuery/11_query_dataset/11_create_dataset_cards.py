#!/usr/bin/env python3
"""Create and upload Hugging Face dataset cards for the three clustered query datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


DATASET_ROOT = Path("/home/wlia0047/ar57/wenyu/result/personal_query/11_query_dataset")


@dataclass(frozen=True)
class DatasetCardTarget:
    category: str
    repo_name: str
    pretty_name: str


CARD_TARGETS = (
    DatasetCardTarget("Baby_Products", "persona-query-baby", "Persona Query: Baby"),
    DatasetCardTarget(
        "Grocery_and_Gourmet_Food",
        "persona-query-grocery",
        "Persona Query: Grocery",
    ),
    DatasetCardTarget("Pet_Supplies", "persona-query-pets", "Persona Query: Pets"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and upload dataset cards.")
    parser.add_argument("--namespace", required=True, help="Hugging Face user or organization namespace.")
    return parser.parse_args()


def load_summary(category: str) -> dict[str, Any]:
    summary_path = DATASET_ROOT / category / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file does not exist: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if not isinstance(summary, dict):
        raise TypeError(f"Summary must be a JSON object: {summary_path}")
    required_keys = [
        "category",
        "num_dataset_rows",
        "rows_by_cluster_label",
        "rows_by_cluster_index",
    ]
    for key in required_keys:
        if key not in summary:
            raise KeyError(f"Summary is missing required key: {key}")
    return summary


def format_dict_table(values: dict[str, Any], key_title: str, value_title: str) -> str:
    if not values:
        raise ValueError("Cannot format an empty table")
    lines = [f"| {key_title} | {value_title} |", "|---|---:|"]
    for key in sorted(values):
        lines.append(f"| `{key}` | {values[key]} |")
    return "\n".join(lines)


def build_card(target: DatasetCardTarget, summary: dict[str, Any], namespace: str) -> str:
    rows_by_cluster_label = summary["rows_by_cluster_label"]
    rows_by_cluster_index = summary["rows_by_cluster_index"]

    return f"""---
language:
- en
license: other
pretty_name: "{target.pretty_name}"
task_categories:
- text-generation
- text-retrieval
tags:
- personalized-search
- query-generation
- clustered-query
- synthetic-data
- amazon-reviews
configs:
- config_name: full
  data_files:
  - split: train
    path: data.jsonl
---

# {target.pretty_name}

This dataset contains personalized product search queries for the `{target.category}` category.

Each record is built from the Personal Query pipeline:

- Stage 06 provides the latest clean syntax-depth query set.
- Stage 12 provides the `strict5550_query_gmm_user_profiles.jsonl` cluster assignment for each user-product query.

## Files

- `data.jsonl`: clean query rows with cluster labels.
- `summary.json`: generation statistics for this category.

## Dataset Size

| Metric | Count |
|---|---:|
| Total rows in `data.jsonl` | {summary["num_dataset_rows"]} |

## Rows By Cluster Label

{format_dict_table(rows_by_cluster_label, "cluster_label", "rows")}

## Rows By Cluster Index

{format_dict_table(rows_by_cluster_index, "cluster_index", "rows")}

## Schema

| Field | Type | Description |
|---|---|---|
| `category` | string | Product category. |
| `uuid` | string | User identifier. |
| `asin` | string | Amazon product identifier. |
| `cluster_label` | string | Query-cluster label such as `cluster_0`. |
| `cluster_index` | integer | Query-cluster index aligned with the cluster label. |
| `correct_query` | string | Clean query text used for evaluation. |
| `attrs_used` | object | Product attributes used to generate the query. |

## Loading

```python
from datasets import load_dataset

full = load_dataset("{namespace}/{target.repo_name}", name="full", split="train")
```

## Intended Use

This dataset is intended for research on personalized product search, clustered query style analysis, query generation, and retrieval evaluation.

## Data Notes

The queries are synthetic outputs generated from user/product signals in the local Personal Query pipeline. Review license and redistribution requirements for any upstream source data before external reuse.
"""


def write_card(category: str, content: str) -> Path:
    card_path = DATASET_ROOT / category / "README.md"
    card_path.write_text(content, encoding="utf-8")
    return card_path


def main() -> None:
    args = parse_args()
    api = HfApi()

    uploaded = []
    for target in CARD_TARGETS:
        summary = load_summary(target.category)
        if summary["category"] != target.category:
            raise ValueError(f"Summary category mismatch: {summary['category']} != {target.category}")
        card_content = build_card(target, summary, args.namespace)
        card_path = write_card(target.category, card_content)
        repo_id = f"{args.namespace}/{target.repo_name}"
        commit_info = api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add dataset card for {target.category}",
        )
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if "README.md" not in files:
            raise RuntimeError(f"Dataset card upload verification failed for {repo_id}")
        uploaded.append(repo_id)
        print(f"[CARD] category={target.category} repo_id={repo_id} commit={commit_info.oid}", flush=True)

    print(f"[SUMMARY] uploaded_cards={len(uploaded)} repos={uploaded}", flush=True)


if __name__ == "__main__":
    main()
