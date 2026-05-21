#!/usr/bin/env python3
"""Evaluate Baby_Products syntax-depth queries with 20 clause features and latent complexity tiers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy.stats import kruskal
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/fs04/ar57/wenyu")
FEATURE_FILE = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products" / "single_query_clause_features.jsonl"
RETRIEVAL_FILE = REPO_ROOT / "result" / "personal_query" / "08_retrieval" / "Baby_Products" / "retrieval_syntax_depth_summary.json"
OUTPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
SUMMARY_FILE = OUTPUT_DIR / "complexity_tier_hit10_summary.json"
ROW_FILE = OUTPUT_DIR / "complexity_tier_hit10_records.jsonl"
TIER_LABELS = ("low", "medium-low", "medium-high", "high")


def safe_pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))


def summarize_hits(records: list[dict]) -> dict:
    if not records:
        raise ValueError("没有可汇总的 hit@10 记录")
    hits = [float(item["hit_at10"]) for item in records]
    return {
        "count": len(hits),
        "mean": float(mean(hits)),
        "std": safe_pstdev(hits),
        "min": float(min(hits)),
        "max": float(max(hits)),
    }


def percentile_tiers(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.argsort(scores, kind="mergesort"), kind="mergesort")
    percentiles = (order + 1) / len(scores)
    tiers = np.empty(len(scores), dtype=object)
    tiers[percentiles <= 0.25] = "low"
    tiers[(percentiles > 0.25) & (percentiles <= 0.5)] = "medium-low"
    tiers[(percentiles > 0.5) & (percentiles <= 0.75)] = "medium-high"
    tiers[percentiles > 0.75] = "high"
    return percentiles, tiers


def oriented_score(values: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        raise ValueError("样本数不足，无法确定潜变量方向")
    if np.std(values) == 0 or np.std(anchor) == 0:
        raise ValueError("潜变量或 anchor 为常数")
    corr = np.corrcoef(values, anchor)[0, 1]
    if np.isnan(corr):
        raise ValueError("潜变量与 anchor 的相关性为 NaN")
    return values if corr >= 0 else -values


def fit_latent_methods(feature_matrix: np.ndarray, anchor: np.ndarray) -> dict[str, np.ndarray]:
    standardized = StandardScaler().fit_transform(feature_matrix)

    pca = PCA(n_components=1, random_state=42)
    pca_scores = oriented_score(pca.fit_transform(standardized).ravel(), anchor)

    fa = FactorAnalysis(n_components=1, random_state=42)
    fa_scores = oriented_score(fa.fit_transform(standardized).ravel(), anchor)

    gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42, reg_covar=1e-6, n_init=5)
    gmm.fit(standardized)
    gmm_labels = gmm.predict(standardized)
    gmm_probabilities = gmm.predict_proba(standardized)
    gmm_cluster_order = []
    for cluster_id in range(4):
        mask = gmm_labels == cluster_id
        if not np.any(mask):
            raise ValueError("GaussianMixture 产生空簇")
        gmm_cluster_order.append((float(np.mean(anchor[mask])), cluster_id))
    gmm_cluster_order.sort()
    gmm_rank_map = {cluster_id: rank for rank, (_, cluster_id) in enumerate(gmm_cluster_order)}
    gmm_scores = np.zeros(len(standardized), dtype=float)
    for cluster_id, rank in gmm_rank_map.items():
        gmm_scores += gmm_probabilities[:, cluster_id] * rank

    ord_cluster = AgglomerativeClustering(n_clusters=4, linkage="ward")
    ord_labels = ord_cluster.fit_predict(standardized)
    ord_cluster_order = []
    for cluster_id in range(4):
        mask = ord_labels == cluster_id
        if not np.any(mask):
            raise ValueError("AgglomerativeClustering 产生空簇")
        ord_cluster_order.append((float(np.mean(anchor[mask])), cluster_id))
    ord_cluster_order.sort()
    ord_rank_map = {cluster_id: rank for rank, (_, cluster_id) in enumerate(ord_cluster_order)}
    ord_scores = np.array([ord_rank_map[label] for label in ord_labels], dtype=float)

    return {
        "pca": pca_scores,
        "factor_analysis": fa_scores,
        "gmm": gmm_scores,
        "ordinal_clustering": ord_scores,
    }


def main() -> None:
    feature_rows = [
        json.loads(line)
        for line in FEATURE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retrieval = json.loads(RETRIEVAL_FILE.read_text(encoding="utf-8"))

    feature_index = {
        (row["user_id"], row["asin"]): row
        for row in feature_rows
    }

    sample_feature = feature_rows[0]["features"]
    feature_names = list(sample_feature.keys())
    feature_matrix = np.array(
        [[float(row["features"][name]) for name in feature_names] for row in feature_rows],
        dtype=float,
    )
    anchor = np.array([float(row["features"]["max_dependency_depth"]) for row in feature_rows], dtype=float)

    latent_outputs = fit_latent_methods(feature_matrix, anchor)
    for method_name, scores in latent_outputs.items():
        percentiles, tiers = percentile_tiers(scores)
        for idx, row in enumerate(feature_rows):
            row.setdefault("latent_scores", {})[method_name] = float(scores[idx])
            row.setdefault("latent_percentiles", {})[method_name] = float(percentiles[idx])
            row.setdefault("complexity_tiers", {})[method_name] = str(tiers[idx])

    all_records = []
    method_summary = {}
    for item in retrieval["all_results_combined"]:
        if item.get("query_category") != "syntax_depth" or item.get("query_type") != "correct":
            continue
        retriever = item["retriever"]
        query_records = item["all_query_records"]
        for method_name in latent_outputs:
            grouped = {label: [] for label in TIER_LABELS}
            matched = []
            missing = 0
            for record in query_records:
                key = (record["user_id"], record["asin"])
                feature_row = feature_index.get(key)
                if feature_row is None:
                    missing += 1
                    continue
                tier = feature_row["complexity_tiers"][method_name]
                merged = {
                    "retriever": retriever,
                    "method": method_name,
                    "user_id": record["user_id"],
                    "asin": record["asin"],
                    "hit_at10": float(record["hit_at10"]),
                    "tier": tier,
                }
                grouped[tier].append(merged)
                matched.append(merged)
                all_records.append(merged)

            grouped_hits = {label: summarize_hits(grouped[label]) for label in TIER_LABELS}
            stat, p_value = kruskal(*[[item["hit_at10"] for item in grouped[label]] for label in TIER_LABELS])
            method_summary.setdefault(method_name, {"retrievers": {}, "pooled": {}})
            method_summary[method_name]["retrievers"][retriever] = {
                "matched_count": len(matched),
                "missing_count": missing,
                "tier_counts": {label: len(grouped[label]) for label in TIER_LABELS},
                "tier_hit10": grouped_hits,
                "kruskal_wallis": {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                },
            }

    for method_name in latent_outputs:
        pooled_records = [row for row in all_records if row["method"] == method_name]
        grouped = {label: [row for row in pooled_records if row["tier"] == label] for label in TIER_LABELS}
        stat, p_value = kruskal(*[[item["hit_at10"] for item in grouped[label]] for label in TIER_LABELS])
        method_summary[method_name]["pooled"] = {
            "matched_count": len(pooled_records),
            "tier_counts": {label: len(grouped[label]) for label in TIER_LABELS},
            "tier_hit10": {label: summarize_hits(grouped[label]) for label in TIER_LABELS},
            "kruskal_wallis": {
                "statistic": float(stat),
                "p_value": float(p_value),
            },
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ROW_FILE.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    summary = {
        "feature_file": str(FEATURE_FILE),
        "retrieval_file": str(RETRIEVAL_FILE),
        "feature_names": feature_names,
        "num_feature_rows": len(feature_rows),
        "num_matched_records": len(all_records),
        "methods": method_summary,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "row_file": str(ROW_FILE),
        "num_feature_rows": len(feature_rows),
        "num_matched_records": len(all_records),
        "methods": list(method_summary.keys()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
