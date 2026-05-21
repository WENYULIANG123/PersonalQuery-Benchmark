#!/usr/bin/env python3
"""Map Baby_Products review sentences and queries into one shared PCA space and compare them."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
REVIEW_SENTENCE_FILE = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY / "review_sentence_pca_distribution_sentences.jsonl"
QUERY_FEATURE_FILE = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY / "single_query_clause_features.jsonl"
OUTPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
SUMMARY_FILE = OUTPUT_DIR / "review_query_shared_pca_summary.json"
QUERY_RECORD_FILE = OUTPUT_DIR / "review_query_shared_pca_query_records.jsonl"
USER_PROFILE_FILE = OUTPUT_DIR / "review_query_shared_pca_user_profiles.jsonl"


def log(message: str) -> None:
    print(message, flush=True)


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} 为空")
    return rows


def feature_names_from_rows(rows: list[dict]) -> list[str]:
    names = list(rows[0]["features"].keys())
    if not names:
        raise ValueError("特征名为空")
    return names


def project_rows(feature_names: list[str], fit_rows: list[dict], target_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    fit_matrix = np.array(
        [[float(row["features"][name]) for name in feature_names] for row in fit_rows],
        dtype=float,
    )
    if len(fit_matrix) < 2:
        raise ValueError("用于拟合 PCA 的样本不足")

    scaler = StandardScaler()
    standardized_fit = scaler.fit_transform(fit_matrix)
    pca = PCA(n_components=1, random_state=42)
    fit_scores = pca.fit_transform(standardized_fit).ravel()
    anchor = np.array([float(row["features"]["max_dependency_depth"]) for row in fit_rows], dtype=float)
    corr = np.corrcoef(fit_scores, anchor)[0, 1]
    if np.isnan(corr):
        raise ValueError("共享 PCA 分数与 anchor 的相关性为 NaN")
    sign = 1.0 if corr >= 0 else -1.0
    fit_scores = sign * fit_scores

    target_matrix = np.array(
        [[float(row["features"][name]) for name in feature_names] for row in target_rows],
        dtype=float,
    )
    target_scores = sign * pca.transform(scaler.transform(target_matrix)).ravel()
    return fit_scores, target_scores


def summarize_array(values: np.ndarray) -> dict:
    if len(values) == 0:
        raise ValueError("无法汇总空数组")
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def main() -> None:
    log("开始读取评论句子特征")
    review_rows = load_jsonl(REVIEW_SENTENCE_FILE)
    log(f"评论句子行数: {len(review_rows)}")
    log("开始读取 query 特征")
    query_rows = load_jsonl(QUERY_FEATURE_FILE)
    log(f"query 行数: {len(query_rows)}")

    review_feature_names = feature_names_from_rows(review_rows)
    query_feature_names = feature_names_from_rows(query_rows)
    if review_feature_names != query_feature_names:
        raise ValueError("评论句子与 query 的特征名不一致")

    log("开始拟合共享 PCA 轴并投影")
    review_scores, query_scores = project_rows(review_feature_names, review_rows, query_rows)

    for row, score in zip(review_rows, review_scores):
        row["shared_pca_score"] = float(score)
    for row, score in zip(query_rows, query_scores):
        row["shared_pca_score"] = float(score)

    review_scores_by_user = defaultdict(list)
    for row in review_rows:
        review_scores_by_user[row["user_id"]].append(float(row["shared_pca_score"]))

    user_profiles = []
    for user_id in sorted(review_scores_by_user.keys()):
        score_array = np.array(review_scores_by_user[user_id], dtype=float)
        user_profiles.append({
            "user_id": user_id,
            "review_sentence_count": len(score_array),
            "shared_pca_score_summary": summarize_array(score_array),
            "shared_pca_score_mean": float(np.mean(score_array)),
            "shared_pca_score_std": float(np.std(score_array)),
            "shared_pca_score_p25": float(np.quantile(score_array, 0.25)),
            "shared_pca_score_p75": float(np.quantile(score_array, 0.75)),
        })

    profile_index = {row["user_id"]: row for row in user_profiles}
    query_records = []
    percentiles = []
    shifts = []
    valid_zscores = []
    above_mean_count = 0
    above_p75_count = 0
    below_p25_count = 0

    for row in query_rows:
        user_id = row["user_id"]
        profile = profile_index.get(user_id)
        if profile is None:
            raise ValueError(f"query user {user_id} 缺少评论 profile")
        review_score_array = np.array(review_scores_by_user[user_id], dtype=float)
        query_score = float(row["shared_pca_score"])
        percentile = float(np.mean(review_score_array <= query_score))
        shift = float(query_score - profile["shared_pca_score_mean"])
        if profile["shared_pca_score_std"] > 0:
            zscore = float(shift / profile["shared_pca_score_std"])
            valid_zscores.append(zscore)
        else:
            zscore = None

        if shift > 0:
            above_mean_count += 1
        if query_score > profile["shared_pca_score_p75"]:
            above_p75_count += 1
        if query_score < profile["shared_pca_score_p25"]:
            below_p25_count += 1

        percentiles.append(percentile)
        shifts.append(shift)
        query_records.append({
            "user_id": user_id,
            "asin": row["asin"],
            "query_type": row["query_type"],
            "target_depth": row.get("target_depth"),
            "actual_depth": row.get("actual_depth"),
            "user_avg_depth": row.get("user_avg_depth"),
            "query": row["query"],
            "shared_pca_score": query_score,
            "review_sentence_count": profile["review_sentence_count"],
            "review_shared_pca_score_summary": profile["shared_pca_score_summary"],
            "query_percentile_within_user_reviews": percentile,
            "query_shift_from_user_review_mean": shift,
            "query_zscore_within_user_reviews": zscore,
            "is_above_user_review_mean": bool(shift > 0),
            "is_above_user_review_p75": bool(query_score > profile["shared_pca_score_p75"]),
            "is_below_user_review_p25": bool(query_score < profile["shared_pca_score_p25"]),
        })

    review_score_array = np.array(review_scores, dtype=float)
    query_score_array = np.array(query_scores, dtype=float)
    percentile_array = np.array(percentiles, dtype=float)
    shift_array = np.array(shifts, dtype=float)
    ks_result = ks_2samp(review_score_array, query_score_array)
    wasserstein = wasserstein_distance(review_score_array, query_score_array)

    summary = {
        "category": CATEGORY,
        "review_sentence_file": str(REVIEW_SENTENCE_FILE),
        "query_feature_file": str(QUERY_FEATURE_FILE),
        "shared_pca_fit_source": "review_sentences_only",
        "num_review_sentence_rows": len(review_rows),
        "num_query_rows": len(query_rows),
        "num_query_users": len(profile_index),
        "feature_names": review_feature_names,
        "review_shared_pca_score_summary": summarize_array(review_score_array),
        "query_shared_pca_score_summary": summarize_array(query_score_array),
        "pooled_distribution_comparison": {
            "ks_statistic": float(ks_result.statistic),
            "ks_p_value": float(ks_result.pvalue),
            "wasserstein_distance": float(wasserstein),
        },
        "query_relative_to_user_reviews": {
            "mean_percentile": float(np.mean(percentile_array)),
            "median_percentile": float(np.median(percentile_array)),
            "mean_shift": float(np.mean(shift_array)),
            "median_shift": float(np.median(shift_array)),
            "mean_zscore": float(np.mean(valid_zscores)) if valid_zscores else None,
            "median_zscore": float(np.median(valid_zscores)) if valid_zscores else None,
            "share_above_user_review_mean": float(above_mean_count / len(query_records)),
            "share_above_user_review_p75": float(above_p75_count / len(query_records)),
            "share_below_user_review_p25": float(below_p25_count / len(query_records)),
        },
        "query_record_file": str(QUERY_RECORD_FILE),
        "user_profile_file": str(USER_PROFILE_FILE),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with QUERY_RECORD_FILE.open("w", encoding="utf-8") as handle:
        for row in query_records:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    with USER_PROFILE_FILE.open("w", encoding="utf-8") as handle:
        for row in user_profiles:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("共享 PCA 对齐分析完成")
    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "num_review_sentence_rows": len(review_rows),
        "num_query_rows": len(query_rows),
        "mean_percentile": summary["query_relative_to_user_reviews"]["mean_percentile"],
        "mean_shift": summary["query_relative_to_user_reviews"]["mean_shift"],
        "share_above_user_review_mean": summary["query_relative_to_user_reviews"]["share_above_user_review_mean"],
        "share_above_user_review_p75": summary["query_relative_to_user_reviews"]["share_above_user_review_p75"],
        "share_below_user_review_p25": summary["query_relative_to_user_reviews"]["share_below_user_review_p25"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
