#!/usr/bin/env python3
"""Create one Hugging Face dataset repository per query category."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from huggingface_hub import HfApi


@dataclass(frozen=True)
class DatasetRepo:
    category: str
    repo_name: str


DATASET_REPOS = (
    DatasetRepo("Baby_Products", "persona-query-baby"),
    DatasetRepo("Grocery_and_Gourmet_Food", "persona-query-grocery"),
    DatasetRepo("Pet_Supplies", "persona-query-pets"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Hugging Face dataset repositories.")
    parser.add_argument("--namespace", required=True, help="Hugging Face user or organization namespace.")
    parser.add_argument("--private", action="store_true", help="Create private repositories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    created_repo_ids = []

    for dataset_repo in DATASET_REPOS:
        repo_id = f"{args.namespace}/{dataset_repo.repo_name}"
        url = api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=False,
        )
        created_repo_ids.append(str(url))
        print(f"[CREATED] category={dataset_repo.category} repo_id={repo_id} url={url}", flush=True)

    print(f"[SUMMARY] created={len(created_repo_ids)}", flush=True)


if __name__ == "__main__":
    main()
