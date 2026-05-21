#!/usr/bin/env python3
"""Test whether query syntactic features are closer to the paired user's review features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import percentileofscore, wilcoxon
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
REVIEW_SENTENCE_FILE = INPUT_DIR / "review_sentence_pca_distribution_sentences.jsonl"
QUERY_FEATURE_FILE = INPUT_DIR / "single_query_clause_features.jsonl"
SUMMARY_FILE = INPUT_DIR / "twenty_feature_user_matching_summary.json"
QUERY_RECORD_FILE = INPUT_DIR / "twenty_feature_user_matching_query_records.jsonl"

RANDOM_SEED = 42
NUM_PERMUTATIONS = 1000
TOP_K_VALUES = [1, 5, 10, 50, 100]


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


def feature_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    matrix = np.array(
        [[float(row["features"][name]) for name in feature_names] for row in rows],
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("特征矩阵为空")
    return matrix


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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(denom == 0):
        raise ValueError("cosine similarity 遇到零向量")
    return numerator / denom


def build_user_review_vectors(review_rows: list[dict], feature_names: list[str], scaler: StandardScaler) -> dict[str, np.ndarray]:
    review_matrix = scaler.transform(feature_matrix(review_rows, feature_names))
    user_vectors: dict[str, list[np.ndarray]] = {}
    for row, vec in zip(review_rows, review_matrix):
        user_id = row["user_id"]
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("review row 缺少 user_id")
        user_vectors.setdefault(user_id, []).append(vec)

    if not user_vectors:
        raise ValueError("没有 review user vector")

    return {user_id: np.mean(np.vstack(vectors), axis=0) for user_id, vectors in user_vectors.items()}


def paired_query_vectors(query_rows: list[dict], feature_names: list[str], scaler: StandardScaler, user_review_vectors: dict[str, np.ndarray]) -> tuple[list[dict], np.ndarray, np.ndarray, list[str]]:
    query_matrix = scaler.transform(feature_matrix(query_rows, feature_names))
    paired_review_vectors = []
    kept_rows = []
    user_ids = []
    for row, query_vec in zip(query_rows, query_matrix):
        user_id = row["user_id"]
        if user_id not in user_review_vectors:
            raise ValueError(f"query user {user_id} 缺少 review vector")
        kept_rows.append(row)
        user_ids.append(user_id)
        paired_review_vectors.append(user_review_vectors[user_id])

    if not kept_rows:
        raise ValueError("没有可配对 query")
    return kept_rows, query_matrix, np.vstack(paired_review_vectors), user_ids


def permutation_baseline(query_matrix: np.ndarray, paired_review_matrix: np.ndarray, rng: np.random.Generator) -> dict:
    true_cos = cosine_similarity(query_matrix, paired_review_matrix)
    true_dist = np.linalg.norm(query_matrix - paired_review_matrix, axis=1)

    random_cos_means = []
    random_dist_means = []
    for _ in range(NUM_PERMUTATIONS):
        perm = rng.permutation(len(paired_review_matrix))
        random_review = paired_review_matrix[perm]
        random_cos_means.append(float(np.mean(cosine_similarity(query_matrix, random_review))))
        random_dist_means.append(float(np.mean(np.linalg.norm(query_matrix - random_review, axis=1))))

    random_cos = np.array(random_cos_means, dtype=float)
    random_dist = np.array(random_dist_means, dtype=float)
    true_cos_mean = float(np.mean(true_cos))
    true_dist_mean = float(np.mean(true_dist))

    cos_percentile = float(percentileofscore(random_cos, true_cos_mean, kind="weak"))
    dist_percentile = float(percentileofscore(random_dist, true_dist_mean, kind="weak"))
    cos_p = float((np.sum(random_cos >= true_cos_mean) + 1) / (NUM_PERMUTATIONS + 1))
    dist_p = float((np.sum(random_dist <= true_dist_mean) + 1) / (NUM_PERMUTATIONS + 1))

    return {
        "true_cosine_summary": summarize_array(true_cos),
        "true_euclidean_distance_summary": summarize_array(true_dist),
        "random_cosine_mean_distribution": summarize_array(random_cos),
        "random_euclidean_distance_mean_distribution": summarize_array(random_dist),
        "true_mean_cosine": true_cos_mean,
        "true_mean_euclidean_distance": true_dist_mean,
        "cosine_percentile_against_random_means": cos_percentile,
        "distance_percentile_against_random_means": dist_percentile,
        "cosine_permutation_p_greater_equal": cos_p,
        "distance_permutation_p_less_equal": dist_p,
    }


def paired_random_query_records(query_rows: list[dict], query_matrix: np.ndarray, paired_review_matrix: np.ndarray, rng: np.random.Generator) -> tuple[list[dict], dict]:
    true_cos = cosine_similarity(query_matrix, paired_review_matrix)
    true_dist = np.linalg.norm(query_matrix - paired_review_matrix, axis=1)
    perm = rng.permutation(len(paired_review_matrix))
    if np.any(perm == np.arange(len(perm))):
        for idx in np.where(perm == np.arange(len(perm)))[0]:
            swap_idx = (idx + 1) % len(perm)
            perm[idx], perm[swap_idx] = perm[swap_idx], perm[idx]
    random_review_matrix = paired_review_matrix[perm]
    random_cos = cosine_similarity(query_matrix, random_review_matrix)
    random_dist = np.linalg.norm(query_matrix - random_review_matrix, axis=1)

    cos_diff = true_cos - random_cos
    dist_diff = random_dist - true_dist
    cos_test = wilcoxon(cos_diff, alternative="greater", zero_method="zsplit")
    dist_test = wilcoxon(dist_diff, alternative="greater", zero_method="zsplit")

    records = []
    for idx, row in enumerate(query_rows):
        records.append({
            "user_id": row["user_id"],
            "asin": row["asin"],
            "query": row["query"],
            "true_cosine": float(true_cos[idx]),
            "random_cosine": float(random_cos[idx]),
            "true_minus_random_cosine": float(cos_diff[idx]),
            "true_euclidean_distance": float(true_dist[idx]),
            "random_euclidean_distance": float(random_dist[idx]),
            "random_minus_true_euclidean_distance": float(dist_diff[idx]),
            "true_cosine_greater_than_random": bool(cos_diff[idx] > 0),
            "true_distance_less_than_random": bool(dist_diff[idx] > 0),
        })

    summary = {
        "cosine_true_greater_share": float(np.mean(cos_diff > 0)),
        "distance_true_closer_share": float(np.mean(dist_diff > 0)),
        "cosine_wilcoxon_statistic": float(cos_test.statistic),
        "cosine_wilcoxon_p_value": float(cos_test.pvalue),
        "distance_wilcoxon_statistic": float(dist_test.statistic),
        "distance_wilcoxon_p_value": float(dist_test.pvalue),
        "cosine_diff_summary": summarize_array(cos_diff),
        "distance_diff_summary": summarize_array(dist_diff),
    }
    return records, summary


def nearest_neighbor_analysis(query_matrix: np.ndarray, user_review_vectors: dict[str, np.ndarray], query_user_ids: list[str]) -> dict:
    review_user_ids = sorted(user_review_vectors.keys())
    review_matrix = np.vstack([user_review_vectors[user_id] for user_id in review_user_ids])
    user_index = {user_id: idx for idx, user_id in enumerate(review_user_ids)}

    distances = cdist(query_matrix, review_matrix, metric="euclidean")
    ranks = []
    for query_idx, user_id in enumerate(query_user_ids):
        target_idx = user_index[user_id]
        order = np.argsort(distances[query_idx], kind="mergesort")
        rank = int(np.where(order == target_idx)[0][0] + 1)
        ranks.append(rank)

    rank_array = np.array(ranks, dtype=int)
    topk = {f"top{k}_accuracy": float(np.mean(rank_array <= k)) for k in TOP_K_VALUES}
    return {
        "num_review_users": len(review_user_ids),
        "rank_summary": summarize_array(rank_array.astype(float)),
        "mrr": float(np.mean(1.0 / rank_array)),
        **topk,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    log("开始读取 review sentence 20 特征")
    review_rows = load_jsonl(REVIEW_SENTENCE_FILE)
    log(f"review sentence 行数: {len(review_rows)}")

    log("开始读取 query 20 特征")
    query_rows = load_jsonl(QUERY_FEATURE_FILE)
    log(f"query 行数: {len(query_rows)}")

    feature_names = feature_names_from_rows(review_rows)
    if feature_names_from_rows(query_rows) != feature_names:
        raise ValueError("review 与 query 的 20 特征名不一致")

    scaler = StandardScaler()
    scaler.fit(feature_matrix(review_rows, feature_names))

    log("开始构造用户 review 20 维均值向量")
    user_review_vectors = build_user_review_vectors(review_rows, feature_names, scaler)
    log(f"review 用户数: {len(user_review_vectors)}")

    log("开始构造 query 与真实用户配对向量")
    paired_rows, query_matrix, paired_review_matrix, query_user_ids = paired_query_vectors(
        query_rows,
        feature_names,
        scaler,
        user_review_vectors,
    )
    log(f"配对 query 数: {len(paired_rows)}")

    log("开始 permutation baseline")
    permutation_summary = permutation_baseline(query_matrix, paired_review_matrix, rng)

    log("开始单次随机配对逐 query 对比")
    query_records, paired_random_summary = paired_random_query_records(
        paired_rows,
        query_matrix,
        paired_review_matrix,
        rng,
    )

    log("开始最近邻找回分析")
    nearest_neighbor = nearest_neighbor_analysis(query_matrix, user_review_vectors, query_user_ids)

    summary = {
        "category": CATEGORY,
        "review_sentence_file": str(REVIEW_SENTENCE_FILE),
        "query_feature_file": str(QUERY_FEATURE_FILE),
        "query_record_file": str(QUERY_RECORD_FILE),
        "feature_names": feature_names,
        "num_review_sentence_rows": len(review_rows),
        "num_review_users": len(user_review_vectors),
        "num_query_rows": len(query_rows),
        "num_paired_queries": len(paired_rows),
        "standardization_fit_source": "review_sentence_features",
        "num_permutations": NUM_PERMUTATIONS,
        "permutation_baseline": permutation_summary,
        "single_random_pairing_comparison": paired_random_summary,
        "nearest_neighbor_retrieval": nearest_neighbor,
    }

    write_jsonl(QUERY_RECORD_FILE, query_records)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "num_paired_queries": len(paired_rows),
        "true_mean_cosine": permutation_summary["true_mean_cosine"],
        "random_cosine_mean": permutation_summary["random_cosine_mean_distribution"]["mean"],
        "cosine_permutation_p": permutation_summary["cosine_permutation_p_greater_equal"],
        "true_mean_distance": permutation_summary["true_mean_euclidean_distance"],
        "random_distance_mean": permutation_summary["random_euclidean_distance_mean_distribution"]["mean"],
        "distance_permutation_p": permutation_summary["distance_permutation_p_less_equal"],
        "top1_accuracy": nearest_neighbor["top1_accuracy"],
        "top10_accuracy": nearest_neighbor["top10_accuracy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
