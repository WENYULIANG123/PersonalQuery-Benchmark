#!/usr/bin/env python3
"""Search one shared PCA threshold that maximizes overall two-tier retriever gap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/fs04/ar57/wenyu")
FEATURE_FILE = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products" / "single_query_clause_features.jsonl"
RETRIEVAL_FILE = REPO_ROOT / "result" / "personal_query" / "08_retrieval" / "Baby_Products" / "retrieval_syntax_depth_summary.json"
OUTPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
SUMMARY_FILE = OUTPUT_DIR / "global_binary_pca_threshold_search.json"
MIN_TIER_RATIO = 0.10


def load_feature_rows() -> list[dict]:
    rows = [
        json.loads(line)
        for line in FEATURE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("特征文件为空")
    return rows


def load_retrieval_index() -> dict[str, dict[tuple[str, str], float]]:
    retrieval = json.loads(RETRIEVAL_FILE.read_text(encoding="utf-8"))
    retriever_index: dict[str, dict[tuple[str, str], float]] = {}
    for item in retrieval["all_results_combined"]:
        if item.get("query_category") != "syntax_depth" or item.get("query_type") != "correct":
            continue
        retriever = item["retriever"]
        per_retriever = {}
        for row in item["all_query_records"]:
            per_retriever[(row["user_id"], row["asin"])] = float(row["hit_at10"])
        retriever_index[retriever] = per_retriever
    if not retriever_index:
        raise ValueError("没有可用的检索结果")
    return retriever_index


def oriented_pca_scores(feature_rows: list[dict]) -> tuple[list[str], dict[tuple[str, str], float]]:
    feature_names = list(feature_rows[0]["features"].keys())
    feature_matrix = np.array(
        [[float(row["features"][name]) for name in feature_names] for row in feature_rows],
        dtype=float,
    )
    anchor = np.array([float(row["features"]["max_dependency_depth"]) for row in feature_rows], dtype=float)
    standardized = StandardScaler().fit_transform(feature_matrix)
    pca = PCA(n_components=1, random_state=42)
    scores = pca.fit_transform(standardized).ravel()
    corr = np.corrcoef(scores, anchor)[0, 1]
    if np.isnan(corr):
        raise ValueError("PCA score 与 anchor 的相关性为 NaN")
    if corr < 0:
        scores = -scores

    score_index = {}
    for row, score in zip(feature_rows, scores):
        score_index[(row["user_id"], row["asin"])] = float(score)
    return feature_names, score_index


def build_aligned_rows(score_index: dict[tuple[str, str], float], retriever_index: dict[str, dict[tuple[str, str], float]]) -> tuple[list[dict], list[str]]:
    retrievers = sorted(retriever_index.keys())
    first_retriever = retrievers[0]
    aligned_rows = []
    for key, hit in retriever_index[first_retriever].items():
        if key not in score_index:
            raise ValueError(f"PCA score 缺少 key={key}")
        row = {
            "user_id": key[0],
            "asin": key[1],
            "score": float(score_index[key]),
            "hits": {
                retriever: float(retriever_index[retriever][key])
                for retriever in retrievers
            },
        }
        aligned_rows.append(row)
    if not aligned_rows:
        raise ValueError("没有可对齐的 query")
    return aligned_rows, retrievers


def search_best_threshold(rows: list[dict], retrievers: list[str]) -> dict:
    sorted_rows = sorted(rows, key=lambda item: item["score"])
    n = len(sorted_rows)
    min_tier_size = int(np.ceil(n * MIN_TIER_RATIO))
    if min_tier_size <= 0:
        raise ValueError("最小 tier 大小无效")

    score_array = np.array([row["score"] for row in sorted_rows], dtype=float)
    hit_arrays = {
        retriever: np.array([row["hits"][retriever] for row in sorted_rows], dtype=float)
        for retriever in retrievers
    }
    prefix_sums = {
        retriever: np.cumsum(hit_arrays[retriever])
        for retriever in retrievers
    }

    best = None
    checked = 0
    for split_idx in range(min_tier_size, n - min_tier_size + 1):
        if split_idx == n:
            continue
        low_boundary = float(score_array[split_idx - 1])
        high_side_first_score = float(score_array[split_idx])
        if low_boundary == high_side_first_score:
            continue
        checked += 1
        per_retriever = {}
        total_abs_gap = 0.0
        min_abs_gap = None
        for retriever in retrievers:
            low_count = split_idx
            high_count = n - split_idx
            low_sum = float(prefix_sums[retriever][split_idx - 1])
            total_sum = float(prefix_sums[retriever][-1])
            high_sum = total_sum - low_sum
            low_mean = low_sum / low_count
            high_mean = high_sum / high_count
            signed_gap = high_mean - low_mean
            abs_gap = abs(signed_gap)
            direction = "increasing" if signed_gap >= 0 else "decreasing"
            per_retriever[retriever] = {
                "low_mean": float(low_mean),
                "high_mean": float(high_mean),
                "signed_gap": float(signed_gap),
                "abs_gap": float(abs_gap),
                "direction": direction,
            }
            total_abs_gap += abs_gap
            min_abs_gap = abs_gap if min_abs_gap is None else min(min_abs_gap, abs_gap)

        if min_abs_gap is None:
            raise ValueError("没有计算出 min_abs_gap")

        candidate = {
            "threshold": low_boundary,
            "next_score_after_threshold": high_side_first_score,
            "low_count": split_idx,
            "high_count": n - split_idx,
            "total_abs_gap": float(total_abs_gap),
            "average_abs_gap": float(total_abs_gap / len(retrievers)),
            "min_abs_gap": float(min_abs_gap),
            "per_retriever": per_retriever,
        }

        if best is None:
            best = candidate
            continue

        best_key = (
            best["total_abs_gap"],
            best["min_abs_gap"],
            -abs(best["low_count"] - best["high_count"]),
        )
        candidate_key = (
            candidate["total_abs_gap"],
            candidate["min_abs_gap"],
            -abs(candidate["low_count"] - candidate["high_count"]),
        )
        if candidate_key > best_key:
            best = candidate

    if best is None:
        raise ValueError("在当前最小 tier 比例约束下没有可用阈值")
    best["checked_threshold_count"] = checked
    return best


def main() -> None:
    feature_rows = load_feature_rows()
    retriever_index = load_retrieval_index()
    _, score_index = oriented_pca_scores(feature_rows)
    aligned_rows, retrievers = build_aligned_rows(score_index, retriever_index)
    best = search_best_threshold(aligned_rows, retrievers)

    summary = {
        "feature_file": str(FEATURE_FILE),
        "retrieval_file": str(RETRIEVAL_FILE),
        "aligned_query_count": len(aligned_rows),
        "retrievers": retrievers,
        "min_tier_ratio": MIN_TIER_RATIO,
        "best_threshold": best,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "aligned_query_count": len(aligned_rows),
        "threshold": best["threshold"],
        "next_score_after_threshold": best["next_score_after_threshold"],
        "low_count": best["low_count"],
        "high_count": best["high_count"],
        "total_abs_gap": best["total_abs_gap"],
        "average_abs_gap": best["average_abs_gap"],
        "min_abs_gap": best["min_abs_gap"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
