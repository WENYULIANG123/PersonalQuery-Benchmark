#!/usr/bin/env python3
"""Verify the unified Hugging Face dataset can be loaded for all configs and splits."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

from datasets import get_dataset_split_names, load_dataset


CONFIGS = ("baby", "grocery", "pets")
SPLITS = ("full", "paired")
UPLOAD_SCRIPT = Path(__file__).with_name("11_upload_unified_hf_dataset.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify unified Hugging Face dataset loading.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repository id.")
    parser.add_argument(
        "--network-mode",
        choices=["ssh-socks", "direct"],
        default="ssh-socks",
        help="Network mode. ssh-socks routes Hugging Face traffic through the login node.",
    )
    return parser.parse_args()


def ensure_network(network_mode: str) -> None:
    if network_mode == "direct":
        return
    if network_mode != "ssh-socks":
        raise ValueError(f"Unsupported network mode: {network_mode}")
    if not UPLOAD_SCRIPT.is_file():
        raise FileNotFoundError(f"Required network helper script does not exist: {UPLOAD_SCRIPT}")

    spec = importlib.util.spec_from_file_location("upload_unified_hf_dataset", UPLOAD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load network helper script: {UPLOAD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ensure_ssh_socks_network()


def main() -> None:
    args = parse_args()
    ensure_network(args.network_mode)
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
                "complexity_group",
                "depth",
                "correct_query",
                "attrs_used",
                "has_error_query",
                "error_query",
                "injected_errors",
            }
            missing_fields = sorted(required_fields - set(first_row))
            if missing_fields:
                raise RuntimeError(f"{config}/{split} first row missing fields: {missing_fields}")
            forbidden_fields = {"source_stage", "query_category", "complexity_level"}
            present_forbidden_fields = sorted(forbidden_fields.intersection(first_row))
            if present_forbidden_fields:
                raise RuntimeError(f"{config}/{split} first row contains forbidden fields: {present_forbidden_fields}")
            print(f"[OK] config={config} split={split} rows={len(dataset)} first_uuid={first_row['uuid']}", flush=True)

    print(f"[SUMMARY] verified repo_id={args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
