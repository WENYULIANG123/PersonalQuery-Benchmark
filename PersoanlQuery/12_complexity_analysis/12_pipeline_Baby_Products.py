#!/usr/bin/env python3
"""Run pipeline task for Baby_Products."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import train_vades_lite_sentence_latent_threshold as train_module

train_module.REPO_ROOT = REPO_ROOT
train_module.CATEGORY = CATEGORY
train_module.INPUT_DIR = (
    REPO_ROOT
    / "result"
    / "personal_query"
    / "12_complexity_analysis_clause_features"
    / CATEGORY
)
train_module.REVIEW_SOURCE_FILE = (
    REPO_ROOT
    / "result"
    / "personal_query"
    / "01_preference_extraction"
    / CATEGORY
    / "stage1_filtered_users_reviews.json"
)
train_module.RAW_CANDIDATE_QUERY_FILE = (
    REPO_ROOT
    / "result"
    / "personal_query"
    / "06_query"
    / CATEGORY
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
    REPO_ROOT
    / "result"
    / "personal_query"
    / "06_query"
    / CATEGORY
    / f"query_by_syntax_depth_{train_module.OUTPUT_TAG}.json"
)

if __name__ == "__main__":
    # 过滤掉有重复 target_reviews 的用户
    train_module.filter_users_with_duplicate_reviews()
    train_module.main()
