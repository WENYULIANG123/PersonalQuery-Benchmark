#!/usr/bin/env python3
"""Unified entrypoint for stage12 complexity-analysis workflows."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

VALID_CATEGORIES = [
    "Baby_Products",
    "Grocery_and_Gourmet_Food",
    "Pet_Supplies",
]

VALID_TASKS = [
    "pipeline",
    "query_clustering",
    "review_query_alignment",
    "style_vector_probe",
]


def _load_local_module(filename: str, module_name: str):
    module_path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified stage12 workflows.")
    parser.add_argument("--category", required=True, choices=VALID_CATEGORIES)
    parser.add_argument("--task", required=True, choices=VALID_TASKS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["PQ_CATEGORY"] = args.category

    if args.task == "pipeline":
        train_module = _load_local_module(
            "train_vades_lite_sentence_latent_threshold.py",
            "stage12_train_vades_lite_sentence_latent_threshold",
        )
        train_module.CATEGORY = args.category
        train_module.INPUT_DIR = (
            train_module.REPO_ROOT
            / "result"
            / "personal_query"
            / "12_complexity_analysis_clause_features"
            / args.category
        )
        train_module.REVIEW_SOURCE_FILE = (
            train_module.REPO_ROOT
            / "result"
            / "personal_query"
            / "01_preference_extraction"
            / args.category
            / "stage1_filtered_users_reviews.json"
        )
        train_module.RAW_CANDIDATE_QUERY_FILE = (
            train_module.REPO_ROOT
            / "result"
            / "personal_query"
            / "06_query"
            / args.category
            / "query_by_syntax_depth_no_depth_check_10.json"
        )
        train_module.CANDIDATE_QUERY_FILE = (
            train_module.INPUT_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
        )
        train_module.SUMMARY_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_summary.json"
        train_module.DETAIL_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_epoch_details.jsonl"
        train_module.USER_PROFILE_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_user_profiles.jsonl"
        train_module.SENTENCE_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_sentences.jsonl"
        train_module.EXCLUDED_USER_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_excluded_users.jsonl"
        train_module.SELECTED_RECORD_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_selected_query_records.jsonl"
        train_module.REJECTED_RECORD_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_rejected_query_records.jsonl"
        train_module.QUERY_FILE = (
            train_module.REPO_ROOT
            / "result"
            / "personal_query"
            / "06_query"
            / args.category
            / f"query_by_syntax_depth_{train_module.OUTPUT_TAG}.json"
        )
        train_module.main()
        return

    if args.task == "query_clustering":
        cluster_module = _load_local_module(
            "cluster_strict5550_query_gmm_and_attach_retrieval.py",
            "stage12_cluster_strict5550_query_gmm_and_attach_retrieval",
        )
        cluster_module.run_query_gmm_pipeline(
            category=args.category,
            query_file=None,
            write_back_to_query_file=False,
            attach_retrieval=True,
        )
        return

    if args.task == "review_query_alignment":
        alignment_module = _load_local_module(
            "evaluate_review_query_alignment.py",
            "stage12_evaluate_review_query_alignment",
        )
        alignment_module.main()
        return

    if args.task == "style_vector_probe":
        probe_module = _load_local_module(
            "evaluate_vades_style_vector_probe.py",
            "stage12_evaluate_vades_style_vector_probe",
        )
        probe_module.main()
        return

    raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    main()
