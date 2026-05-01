#!/usr/bin/env python3
"""Make the three Hugging Face query dataset repositories public."""

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
    parser = argparse.ArgumentParser(description="Make Hugging Face dataset repositories public.")
    parser.add_argument("--namespace", required=True, help="Hugging Face user or organization namespace.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    updated_repo_ids = []

    for dataset_repo in DATASET_REPOS:
        repo_id = f"{args.namespace}/{dataset_repo.repo_name}"
        result = api.update_repo_visibility(repo_id=repo_id, private=False, repo_type="dataset")
        if result.get("private") is not False:
            raise RuntimeError(f"Visibility update failed for {repo_id}: {result}")

        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        if repo_info.private is not False:
            raise RuntimeError(f"Remote verification failed for {repo_id}: private={repo_info.private}")

        updated_repo_ids.append(repo_id)
        print(f"[PUBLIC] category={dataset_repo.category} repo_id={repo_id}", flush=True)

    print(f"[SUMMARY] public_repos={len(updated_repo_ids)} repos={updated_repo_ids}", flush=True)


if __name__ == "__main__":
    main()
