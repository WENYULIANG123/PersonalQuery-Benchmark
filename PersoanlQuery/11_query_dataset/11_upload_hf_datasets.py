#!/usr/bin/env python3
"""Upload per-category query datasets to Hugging Face dataset repositories."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi


DATASET_ROOT = Path("/home/wlia0047/ar57/wenyu/dataset")
REQUIRED_FILES = ("data.jsonl", "paired_data.jsonl", "summary.json")


@dataclass(frozen=True)
class DatasetUploadTarget:
    category: str
    repo_name: str


UPLOAD_TARGETS = (
    DatasetUploadTarget("Baby_Products", "persona-query-baby"),
    DatasetUploadTarget("Grocery_and_Gourmet_Food", "persona-query-grocery"),
    DatasetUploadTarget("Pet_Supplies", "persona-query-pets"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload three personal query datasets to Hugging Face.")
    parser.add_argument("--namespace", required=True, help="Hugging Face user or organization namespace.")
    return parser.parse_args()


def validate_dataset_dir(category: str) -> Path:
    dataset_dir = DATASET_ROOT / category
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    for filename in REQUIRED_FILES:
        file_path = dataset_dir / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Required dataset file does not exist: {file_path}")
        if file_path.stat().st_size == 0:
            raise ValueError(f"Required dataset file is empty: {file_path}")
    return dataset_dir


def main() -> None:
    args = parse_args()
    api = HfApi()

    uploaded = []
    for target in UPLOAD_TARGETS:
        dataset_dir = validate_dataset_dir(target.category)
        repo_id = f"{args.namespace}/{target.repo_name}"
        commit_info = api.upload_folder(
            folder_path=str(dataset_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {target.category} query dataset",
        )
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        missing_files = [filename for filename in REQUIRED_FILES if filename not in files]
        if missing_files:
            raise RuntimeError(f"Upload verification failed for {repo_id}, missing files: {missing_files}")
        uploaded.append(repo_id)
        print(f"[UPLOADED] category={target.category} repo_id={repo_id} commit={commit_info.oid}", flush=True)
        print(f"[FILES] repo_id={repo_id} files={files}", flush=True)

    print(f"[SUMMARY] uploaded={len(uploaded)} repos={uploaded}", flush=True)


if __name__ == "__main__":
    main()
