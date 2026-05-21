#!/usr/bin/env python3
"""Analyze whether user-level review PCA and query PCA are aligned in the shared space."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, pearsonr, spearmanr


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
REVIEW_PROFILE_FILE = INPUT_DIR / "review_query_shared_pca_user_profiles.jsonl"
QUERY_RECORD_FILE = INPUT_DIR / "review_query_shared_pca_query_records.jsonl"
SUMMARY_FILE = INPUT_DIR / "review_query_user_style_alignment_summary.json"
USER_RECORD_FILE = INPUT_DIR / "review_query_user_style_alignment_user_records.jsonl"


def log(message: str) -> None:
    print(message, flush=True)


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} 为空")
    return rows


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


def build_user_rows(review_profiles: list[dict], query_records: list[dict]) -> list[dict]:
    review_index = {row["user_id"]: row for row in review_profiles}
    if len(review_index) != len(review_profiles):
        raise ValueError("review profile 存在重复 user_id")

    query_scores_by_user: dict[str, list[float]] = defaultdict(list)
    for row in query_records:
        user_id = row["user_id"]
        if user_id not in review_index:
            raise ValueError(f"query user {user_id} 缺少 review profile")
        query_scores_by_user[user_id].append(float(row["shared_pca_score"]))

    if not query_scores_by_user:
        raise ValueError("没有 query user 可用于用户级分析")

    user_rows = []
    for user_id in sorted(query_scores_by_user.keys()):
        query_score_array = np.array(query_scores_by_user[user_id], dtype=float)
        review_profile = review_index[user_id]
        user_rows.append({
            "user_id": user_id,
            "review_sentence_count": int(review_profile["review_sentence_count"]),
            "query_count": int(len(query_score_array)),
            "review_user_pca_mean": float(review_profile["shared_pca_score_mean"]),
            "review_user_pca_std": float(review_profile["shared_pca_score_std"]),
            "query_user_pca_mean": float(np.mean(query_score_array)),
            "query_user_pca_median": float(np.median(query_score_array)),
            "query_user_pca_std": float(np.std(query_score_array)),
        })
    return user_rows


def assign_review_tertiles(user_rows: list[dict]) -> tuple[list[dict], dict]:
    review_scores = np.array([row["review_user_pca_mean"] for row in user_rows], dtype=float)
    q1 = float(np.quantile(review_scores, 1.0 / 3.0))
    q2 = float(np.quantile(review_scores, 2.0 / 3.0))

    grouped: dict[str, list[dict]] = {"low": [], "medium": [], "high": []}
    for row in user_rows:
        score = float(row["review_user_pca_mean"])
        if score <= q1:
            label = "low"
        elif score <= q2:
            label = "medium"
        else:
            label = "high"
        enriched = dict(row)
        enriched["review_pca_group"] = label
        grouped[label].append(enriched)

    if any(len(rows) == 0 for rows in grouped.values()):
        raise ValueError("review_user_pca 三分组后出现空组")

    enriched_rows = grouped["low"] + grouped["medium"] + grouped["high"]
    return enriched_rows, {
        "q33": q1,
        "q67": q2,
    }


def summarize_group(rows: list[dict]) -> dict:
    review_values = np.array([row["review_user_pca_mean"] for row in rows], dtype=float)
    query_values = np.array([row["query_user_pca_mean"] for row in rows], dtype=float)
    return {
        "user_count": int(len(rows)),
        "review_user_pca_summary": summarize_array(review_values),
        "query_user_pca_summary": summarize_array(query_values),
    }


def main() -> None:
    log("开始读取用户评论 profile")
    review_profiles = load_jsonl(REVIEW_PROFILE_FILE)
    log(f"review profile 数: {len(review_profiles)}")
    log("开始读取 query 记录")
    query_records = load_jsonl(QUERY_RECORD_FILE)
    log(f"query 记录数: {len(query_records)}")

    log("开始构造用户级 review/query PCA")
    user_rows = build_user_rows(review_profiles, query_records)
    log(f"匹配到的 query 用户数: {len(user_rows)}")

    if len(user_rows) < 3:
        raise ValueError("用户数不足，无法进行相关性和三分组分析")

    review_values = np.array([row["review_user_pca_mean"] for row in user_rows], dtype=float)
    query_values = np.array([row["query_user_pca_mean"] for row in user_rows], dtype=float)

    pearson_stat = pearsonr(review_values, query_values)
    spearman_stat = spearmanr(review_values, query_values)
    slope, intercept = np.polyfit(review_values, query_values, deg=1)

    log("开始按 review_user_pca 三分组")
    user_rows, tertile_boundaries = assign_review_tertiles(user_rows)
    grouped: dict[str, list[dict]] = {"low": [], "medium": [], "high": []}
    for row in user_rows:
        grouped[row["review_pca_group"]].append(row)

    low_mean = float(np.mean([row["query_user_pca_mean"] for row in grouped["low"]]))
    medium_mean = float(np.mean([row["query_user_pca_mean"] for row in grouped["medium"]]))
    high_mean = float(np.mean([row["query_user_pca_mean"] for row in grouped["high"]]))
    strict_increasing = bool(low_mean < medium_mean < high_mean)
    nondecreasing = bool(low_mean <= medium_mean <= high_mean)

    kw_stat, kw_p = kruskal(
        [row["query_user_pca_mean"] for row in grouped["low"]],
        [row["query_user_pca_mean"] for row in grouped["medium"]],
        [row["query_user_pca_mean"] for row in grouped["high"]],
    )

    summary = {
        "category": CATEGORY,
        "review_profile_file": str(REVIEW_PROFILE_FILE),
        "query_record_file": str(QUERY_RECORD_FILE),
        "num_review_profiles": len(review_profiles),
        "num_query_records": len(query_records),
        "num_query_users": len(user_rows),
        "user_level_review_pca_summary": summarize_array(review_values),
        "user_level_query_pca_summary": summarize_array(query_values),
        "correlation": {
            "pearson_r": float(pearson_stat.statistic),
            "pearson_p_value": float(pearson_stat.pvalue),
            "spearman_rho": float(spearman_stat.statistic),
            "spearman_p_value": float(spearman_stat.pvalue),
            "linear_fit_slope": float(slope),
            "linear_fit_intercept": float(intercept),
        },
        "review_pca_tertiles": {
            "boundaries": tertile_boundaries,
            "groups": {
                "low": summarize_group(grouped["low"]),
                "medium": summarize_group(grouped["medium"]),
                "high": summarize_group(grouped["high"]),
            },
            "query_user_pca_mean_order": {
                "low": low_mean,
                "medium": medium_mean,
                "high": high_mean,
                "strict_increasing": strict_increasing,
                "nondecreasing": nondecreasing,
            },
            "kruskal_wallis": {
                "statistic": float(kw_stat),
                "p_value": float(kw_p),
            },
        },
        "user_record_file": str(USER_RECORD_FILE),
    }

    with USER_RECORD_FILE.open("w", encoding="utf-8") as handle:
        for row in user_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("用户级风格保持分析完成")
    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "num_query_users": len(user_rows),
        "pearson_r": summary["correlation"]["pearson_r"],
        "spearman_rho": summary["correlation"]["spearman_rho"],
        "strict_increasing": strict_increasing,
        "low_query_mean": low_mean,
        "medium_query_mean": medium_mean,
        "high_query_mean": high_mean,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
