#!/usr/bin/env python3
"""Search shared PCA thresholds that make each retriever monotonic on train and validate on test."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/fs04/ar57/wenyu")
FEATURE_FILE = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products" / "single_query_clause_features.jsonl"
RETRIEVAL_FILE = REPO_ROOT / "result" / "personal_query" / "08_retrieval" / "Baby_Products" / "retrieval_syntax_depth_summary.json"
OUTPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
SUMMARY_FILE = OUTPUT_DIR / "global_monotonic_pca_threshold_search.json"
SEED = 42
TRAIN_RATIO = 0.64
VAL_RATIO = 0.16
TEST_RATIO = 0.20
TIER_LABELS = ("low", "medium", "high")
MIN_TIER_SIZE = 50
MONOTONIC_TOLERANCE = 1e-12
MIN_CANDIDATE_PERCENTILE = 0.10
MAX_CANDIDATE_PERCENTILE = 0.90
MAX_GRID_POINTS = 60


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_feature_rows() -> tuple[list[str], list[dict]]:
    rows = [
        json.loads(line)
        for line in FEATURE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("特征文件为空")
    feature_names = list(rows[0]["features"].keys())
    return feature_names, rows


def load_retrieval_index() -> dict[str, dict[tuple[str, str], float]]:
    retrieval = json.loads(RETRIEVAL_FILE.read_text(encoding="utf-8"))
    index: dict[str, dict[tuple[str, str], float]] = {}
    for item in retrieval["all_results_combined"]:
        if item.get("query_category") != "syntax_depth" or item.get("query_type") != "correct":
            continue
        retriever = item["retriever"]
        per_retriever = {}
        for row in item["all_query_records"]:
            key = (row["user_id"], row["asin"])
            per_retriever[key] = float(row["hit_at10"])
        index[retriever] = per_retriever
    if not index:
        raise ValueError("没有找到可用的检索结果")
    return index


def split_users(rows: list[dict]) -> dict[str, set[str]]:
    retrieval_index = load_retrieval_index()
    first_retriever = sorted(retrieval_index.keys())[0]
    per_user_labels: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = (row["user_id"], row["asin"])
        if key not in retrieval_index[first_retriever]:
            continue
        per_user_labels[row["user_id"]].append(retrieval_index[first_retriever][key])

    users = sorted(per_user_labels.keys())
    user_targets = [int(max(per_user_labels[user_id])) for user_id in users]
    if len(set(user_targets)) != 2:
        raise ValueError("用户级标签只有一个类别，无法做分层切分")

    train_val_users, test_users = train_test_split(
        users,
        test_size=TEST_RATIO,
        random_state=SEED,
        stratify=user_targets,
    )
    remaining_targets = [int(max(per_user_labels[user_id])) for user_id in train_val_users]
    val_ratio_within_train_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_users, val_users = train_test_split(
        train_val_users,
        test_size=val_ratio_within_train_val,
        random_state=SEED,
        stratify=remaining_targets,
    )
    return {
        "train": set(train_users),
        "val": set(val_users),
        "test": set(test_users),
    }


def build_scored_rows(feature_names: list[str], rows: list[dict], retrieval_index: dict[str, dict[tuple[str, str], float]], user_splits: dict[str, set[str]]) -> tuple[list[dict], list[str]]:
    first_retriever = sorted(retrieval_index.keys())[0]
    aligned_rows = []
    for row in rows:
        key = (row["user_id"], row["asin"])
        if key not in retrieval_index[first_retriever]:
            continue
        split_names = [split_name for split_name, users in user_splits.items() if row["user_id"] in users]
        if len(split_names) != 1:
            raise ValueError(f"用户 {row['user_id']} 没有唯一 split")
        aligned_rows.append({
            "user_id": row["user_id"],
            "asin": row["asin"],
            "split": split_names[0],
            "features": [float(row["features"][name]) for name in feature_names],
        })

    if not aligned_rows:
        raise ValueError("没有可对齐的特征行")

    train_matrix = np.array([row["features"] for row in aligned_rows if row["split"] == "train"], dtype=float)
    if len(train_matrix) == 0:
        raise ValueError("训练集为空，无法拟合 PCA")
    scaler = StandardScaler()
    scaler.fit(train_matrix)
    standardized_train = scaler.transform(train_matrix)
    pca = PCA(n_components=1, random_state=SEED)
    pca.fit(standardized_train)

    anchor = np.array([row["features"][feature_names.index("max_dependency_depth")] for row in aligned_rows if row["split"] == "train"], dtype=float)
    train_scores = pca.transform(standardized_train).ravel()
    corr = np.corrcoef(train_scores, anchor)[0, 1]
    if np.isnan(corr):
        raise ValueError("PCA 训练分数与 anchor 的相关性为 NaN")
    sign = 1.0 if corr >= 0 else -1.0

    scored_rows = []
    for row in aligned_rows:
        standardized = scaler.transform(np.array([row["features"]], dtype=float))
        score = float(sign * pca.transform(standardized).ravel()[0])
        score_row = {
            "user_id": row["user_id"],
            "asin": row["asin"],
            "split": row["split"],
            "score": score,
            "hits": {},
        }
        key = (row["user_id"], row["asin"])
        for retriever, retriever_index in retrieval_index.items():
            hit = retriever_index.get(key)
            if hit is None:
                raise ValueError(f"{retriever} 缺少 key={key} 的 hit@10")
            score_row["hits"][retriever] = float(hit)
        scored_rows.append(score_row)
    return scored_rows, sorted(retrieval_index.keys())


def candidate_boundaries(train_scores: np.ndarray) -> list[float]:
    low = np.quantile(train_scores, MIN_CANDIDATE_PERCENTILE)
    high = np.quantile(train_scores, MAX_CANDIDATE_PERCENTILE)
    if low >= high:
        raise ValueError("候选边界范围无效")
    quantiles = np.linspace(MIN_CANDIDATE_PERCENTILE, MAX_CANDIDATE_PERCENTILE, MAX_GRID_POINTS)
    candidates = sorted({float(np.quantile(train_scores, q)) for q in quantiles})
    if len(candidates) < 3:
        raise ValueError("候选边界数量不足")
    return candidates


def assign_tier(score: float, low_boundary: float, high_boundary: float) -> str:
    if score <= low_boundary:
        return "low"
    if score <= high_boundary:
        return "medium"
    return "high"


def is_monotonic(means: dict[str, float]) -> tuple[bool, str | None]:
    low = means["low"]
    medium = means["medium"]
    high = means["high"]
    if low <= medium + MONOTONIC_TOLERANCE and medium <= high + MONOTONIC_TOLERANCE:
        return True, "increasing"
    if low >= medium - MONOTONIC_TOLERANCE and medium >= high - MONOTONIC_TOLERANCE:
        return True, "decreasing"
    return False, None


def evaluate_thresholds(rows: list[dict], retrievers: list[str], low_boundary: float, high_boundary: float) -> dict:
    tiered = {retriever: {label: [] for label in TIER_LABELS} for retriever in retrievers}
    for row in rows:
        tier = assign_tier(row["score"], low_boundary, high_boundary)
        for retriever in retrievers:
            tiered[retriever][tier].append(float(row["hits"][retriever]))

    per_retriever = {}
    overall_min_gap = None
    total_gap = 0.0
    for retriever in retrievers:
        tier_hits = tiered[retriever]
        for label, values in tier_hits.items():
            if len(values) < MIN_TIER_SIZE:
                return {
                    "feasible": False,
                    "reason": f"{retriever} 的 {label} tier 样本不足 {MIN_TIER_SIZE}",
                }
        means = {label: float(np.mean(values)) for label, values in tier_hits.items()}
        monotonic, direction = is_monotonic(means)
        if not monotonic:
            return {
                "feasible": False,
                "reason": f"{retriever} 不单调",
            }
        gap = float(max(means.values()) - min(means.values()))
        total_gap += gap
        overall_min_gap = gap if overall_min_gap is None else min(overall_min_gap, gap)
        per_retriever[retriever] = {
            "direction": direction,
            "tier_counts": {label: len(tier_hits[label]) for label in TIER_LABELS},
            "tier_means": means,
            "gap": gap,
        }

    if overall_min_gap is None:
        raise ValueError("没有计算出 overall_min_gap")
    return {
        "feasible": True,
        "per_retriever": per_retriever,
        "overall_min_gap": float(overall_min_gap),
        "total_gap": float(total_gap),
    }


def choose_best_thresholds(train_rows: list[dict], val_rows: list[dict], retrievers: list[str]) -> dict:
    train_scores = np.array([row["score"] for row in train_rows], dtype=float)
    boundaries = candidate_boundaries(train_scores)
    best = None
    checked = 0
    for i, low_boundary in enumerate(boundaries[:-1]):
        for high_boundary in boundaries[i + 1:]:
            checked += 1
            if checked % 500 == 0:
                print(json.dumps({
                    "checked_pairs": checked,
                    "current_low_boundary": float(low_boundary),
                    "current_high_boundary": float(high_boundary),
                }, ensure_ascii=False))
            train_evaluation = evaluate_thresholds(train_rows, retrievers, low_boundary, high_boundary)
            if not train_evaluation["feasible"]:
                continue
            val_evaluation = evaluate_thresholds(val_rows, retrievers, low_boundary, high_boundary)
            if not val_evaluation["feasible"]:
                continue
            candidate = {
                "low_boundary": float(low_boundary),
                "high_boundary": float(high_boundary),
                "train_evaluation": train_evaluation,
                "val_evaluation": val_evaluation,
            }
            if best is None:
                best = candidate
                continue
            best_key = (
                best["train_evaluation"]["overall_min_gap"],
                best["val_evaluation"]["overall_min_gap"],
                best["train_evaluation"]["total_gap"],
                best["val_evaluation"]["total_gap"],
            )
            candidate_key = (
                candidate["train_evaluation"]["overall_min_gap"],
                candidate["val_evaluation"]["overall_min_gap"],
                candidate["train_evaluation"]["total_gap"],
                candidate["val_evaluation"]["total_gap"],
            )
            if candidate_key > best_key:
                best = candidate
    if best is None:
        raise ValueError("训练集和验证集上不存在满足所有检索器单调的共享阈值")
    return best


def main() -> None:
    set_seed(SEED)
    feature_names, feature_rows = load_feature_rows()
    retrieval_index = load_retrieval_index()
    user_splits = split_users(feature_rows)
    scored_rows, retrievers = build_scored_rows(feature_names, feature_rows, retrieval_index, user_splits)

    split_rows = {
        split_name: [row for row in scored_rows if row["split"] == split_name]
        for split_name in ("train", "val", "test")
    }
    best = choose_best_thresholds(split_rows["train"], split_rows["val"], retrievers)
    low_boundary = best["low_boundary"]
    high_boundary = best["high_boundary"]

    test_evaluation = evaluate_thresholds(split_rows["test"], retrievers, low_boundary, high_boundary)
    summary = {
        "feature_file": str(FEATURE_FILE),
        "retrieval_file": str(RETRIEVAL_FILE),
        "retrievers": retrievers,
        "split_sizes": {split_name: len(rows) for split_name, rows in split_rows.items()},
        "unique_user_counts": {
            split_name: len(user_splits[split_name])
            for split_name in ("train", "val", "test")
        },
        "thresholds": {
            "low_boundary": low_boundary,
            "high_boundary": high_boundary,
        },
        "train_evaluation": best["train_evaluation"],
        "val_evaluation": best["val_evaluation"],
        "test_evaluation": test_evaluation,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "thresholds": summary["thresholds"],
        "train_overall_min_gap": best["train_evaluation"]["overall_min_gap"],
        "val_overall_min_gap": best["val_evaluation"]["overall_min_gap"],
        "test_feasible": test_evaluation["feasible"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
