#!/usr/bin/env python3
"""Verify the unified Hugging Face dataset can be loaded for all configs and splits."""

from __future__ import annotations

import argparse

from datasets import get_dataset_split_names, load_dataset


CONFIGS = ("baby", "grocery", "pets")
SPLITS = ("full", "paired")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify unified Hugging Face dataset loading.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repository id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for config in CONFIGS:
        split_names = get_dataset_split_names(args.repo_id, config)
        if set(split_names) != set(SPLITS):
            raise RuntimeError(f"Unexpected split names for {config}: {split_names}")
        for split in SPLITS:
            dataset = load_dataset(args.repo_id, name=config, split=split, download_mode="force_redownload")
            if len(dataset) == 0:
                raise RuntimeError(f"{config}/{split} is empty")
            first_row = dataset[0]
            required_fields = {
                "uuid",
                "query_category",
                "correct_query",
                "has_error_query",
                "error_query",
                "injected_errors",
            }
            missing_fields = sorted(required_fields - set(first_row))
            if missing_fields:
                raise RuntimeError(f"{config}/{split} first row missing fields: {missing_fields}")
            print(f"[OK] config={config} split={split} rows={len(dataset)} first_uuid={first_row['uuid']}", flush=True)

    print(f"[SUMMARY] verified repo_id={args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
