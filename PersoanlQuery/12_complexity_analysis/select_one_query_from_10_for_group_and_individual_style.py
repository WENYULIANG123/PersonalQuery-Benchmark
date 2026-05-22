#!/usr/bin/env python3
"""Select one query from 10 candidates per user for group and individual style alignment."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.stats import kruskal, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from extract_clause_features_single_query import extract_clause_features_from_doc, load_spacy_model  # noqa: E402


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
QUERY_10_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / "query_by_syntax_depth_no_depth_check_10.json"
REVIEW_SENTENCE_FILE = INPUT_DIR / "review_sentence_pca_distribution_sentences.jsonl"
SUMMARY_FILE = INPUT_DIR / "select_one_query_from_10_group_individual_summary.json"
CANDIDATE_FEATURE_FILE = INPUT_DIR / "query_10_candidates_clause_features.jsonl"
SELECTED_RECORD_FILE = INPUT_DIR / "selected_one_query_from_10_group_individual_records.jsonl"

RIDGE = 1e-3
ALPHA_GRID = [round(value, 2) for value in np.linspace(0.0, 1.0, 21)]


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
    for idx, row in enumerate(rows):
        if list(row["features"].keys()) != names:
            raise ValueError(f"第 {idx} 行特征名不一致")
    return names


def feature_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    matrix = np.array([[float(row["features"][name]) for name in feature_names] for row in rows], dtype=float)
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


def extract_candidate_features() -> tuple[list[str], list[dict]]:
    payload = json.loads(QUERY_10_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{QUERY_10_FILE} 必须是非空列表")

    nlp = load_spacy_model()
    rows = []
    feature_names = None
    total_users = len(payload)
    for user_index, user_record in enumerate(payload, start=1):
        user_id = user_record["user_id"]
        asin = user_record["asin"]
        candidates = user_record.get("syntax_depth_queries")
        if not isinstance(candidates, list) or len(candidates) != 10:
            raise ValueError(f"user {user_id} syntax_depth_queries 必须正好 10 条")

        query_texts = [candidate["query"] for candidate in candidates]
        for candidate, doc in zip(candidates, nlp.pipe(query_texts, batch_size=16)):
            query = candidate["query"]
            extracted = extract_clause_features_from_doc(doc, query)
            features = extracted["features"]
            if feature_names is None:
                feature_names = list(features.keys())
            elif list(features.keys()) != feature_names:
                raise ValueError("候选 query 特征名不一致")
            rows.append({
                "user_id": user_id,
                "asin": asin,
                "candidate_index": int(candidate["accepted_candidate_index"]),
                "query": query,
                "word_count": int(extracted["word_count"]),
                "target_depth": candidate["target_depth"],
                "user_avg_depth": candidate["user_avg_depth"],
                "attrs_used": candidate["attrs_used"],
                "features": features,
            })
        log(f"已抽取候选特征 user {user_index}/{total_users}: {user_id}")

    if feature_names is None:
        raise ValueError("没有抽取到候选 query 特征")
    return feature_names, rows


def compute_scatter_matrices(matrix: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    global_mean = np.mean(matrix, axis=0)
    by_user: dict[str, list[np.ndarray]] = defaultdict(list)
    for vector, label in zip(matrix, labels):
        by_user[label].append(vector)
    if len(by_user) < 2:
        raise ValueError("用户数不足，无法计算 Fisher 轴")

    dim = matrix.shape[1]
    between = np.zeros((dim, dim), dtype=float)
    within = np.zeros((dim, dim), dtype=float)
    user_means = {}
    for user_id, vectors in by_user.items():
        block = np.vstack(vectors)
        user_mean = np.mean(block, axis=0)
        user_means[user_id] = user_mean
        diff_mean = (user_mean - global_mean).reshape(-1, 1)
        between += block.shape[0] * (diff_mean @ diff_mean.T)
        centered = block - user_mean
        within += centered.T @ centered
    between /= float(matrix.shape[0])
    within /= float(matrix.shape[0] - len(by_user))
    return between, within, user_means


def fit_fisher_axis(review_rows: list[dict], feature_names: list[str]) -> tuple[StandardScaler, np.ndarray, dict[str, float]]:
    review_matrix = feature_matrix(review_rows, feature_names)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(review_matrix)
    labels = [row["user_id"] for row in review_rows]
    between, within, user_means = compute_scatter_matrices(standardized, labels)
    eigenvalues, eigenvectors = eigh(between, within + RIDGE * np.eye(within.shape[0]))
    best_idx = int(np.argmax(eigenvalues))
    weight = np.real(eigenvectors[:, best_idx]).astype(float)
    norm = float(np.linalg.norm(weight))
    if norm == 0.0:
        raise ValueError("Fisher 轴权重范数为 0")
    weight = weight / norm
    anchor_name = "max_dependency_depth"
    if anchor_name not in feature_names:
        raise ValueError("缺少 max_dependency_depth")
    if weight[feature_names.index(anchor_name)] < 0:
        weight = -weight
    review_user_scores = {user_id: float(vector @ weight) for user_id, vector in user_means.items()}
    return scaler, weight, review_user_scores


def assign_tertiles(scores: dict[str, float], user_ids: list[str]) -> tuple[dict[str, str], dict]:
    values = np.array([scores[user_id] for user_id in user_ids], dtype=float)
    low_boundary, high_boundary = np.quantile(values, [1 / 3, 2 / 3])
    if low_boundary == high_boundary:
        raise ValueError("review score 三分位阈值相同")
    groups = {}
    for user_id in user_ids:
        score = scores[user_id]
        if score <= low_boundary:
            groups[user_id] = "low"
        elif score <= high_boundary:
            groups[user_id] = "medium"
        else:
            groups[user_id] = "high"
    return groups, {"low_medium": float(low_boundary), "medium_high": float(high_boundary)}


def evaluate_selection(selected_rows: list[dict]) -> dict:
    review_scores = np.array([row["review_user_score"] for row in selected_rows], dtype=float)
    query_scores = np.array([row["candidate_fisher_score"] for row in selected_rows], dtype=float)
    abs_diffs = np.abs(query_scores - review_scores)
    pearson = pearsonr(review_scores, query_scores)
    spearman = spearmanr(review_scores, query_scores)
    grouped = defaultdict(list)
    for row in selected_rows:
        grouped[row["review_group"]].append(float(row["candidate_fisher_score"]))
    for label in ("low", "medium", "high"):
        if label not in grouped:
            raise ValueError(f"selection 缺少分组: {label}")
    group_arrays = [grouped[label] for label in ("low", "medium", "high")]
    group_test = kruskal(*group_arrays)
    means = {label: float(np.mean(grouped[label])) for label in ("low", "medium", "high")}
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "mean_absolute_query_review_diff": float(np.mean(abs_diffs)),
        "median_absolute_query_review_diff": float(np.median(abs_diffs)),
        "query_score_summary": summarize_array(query_scores),
        "review_score_summary": summarize_array(review_scores),
        "group_query_score_means": means,
        "group_query_score_order": [means["low"], means["medium"], means["high"]],
        "group_query_score_is_increasing": bool(means["low"] < means["medium"] < means["high"]),
        "group_query_score_is_decreasing": bool(means["low"] > means["medium"] > means["high"]),
        "group_query_score_range": float(max(means.values()) - min(means.values())),
        "group_query_score_kruskal_statistic": float(group_test.statistic),
        "group_query_score_kruskal_p_value": float(group_test.pvalue),
    }


def score_candidates(
    candidate_rows: list[dict],
    feature_names: list[str],
    scaler: StandardScaler,
    weight: np.ndarray,
    review_user_scores: dict[str, float],
) -> list[dict]:
    matrix = scaler.transform(feature_matrix(candidate_rows, feature_names))
    scores = matrix @ weight
    scored_rows = []
    for row, score in zip(candidate_rows, scores):
        user_id = row["user_id"]
        if user_id not in review_user_scores:
            raise ValueError(f"候选 user 缺少 review score: {user_id}")
        enriched = dict(row)
        enriched["candidate_fisher_score"] = float(score)
        enriched["review_user_score"] = float(review_user_scores[user_id])
        enriched["candidate_minus_review_score"] = float(score - review_user_scores[user_id])
        scored_rows.append(enriched)
    return scored_rows


def select_by_alpha(scored_rows: list[dict], alpha: float) -> list[dict]:
    user_ids = sorted({row["user_id"] for row in scored_rows})
    review_scores = {row["user_id"]: row["review_user_score"] for row in scored_rows}
    review_groups, _ = assign_tertiles(review_scores, user_ids)
    group_means = {
        label: float(np.mean([review_scores[user_id] for user_id in user_ids if review_groups[user_id] == label]))
        for label in ("low", "medium", "high")
    }

    all_candidate_scores = np.array([row["candidate_fisher_score"] for row in scored_rows], dtype=float)
    candidate_mean = float(np.mean(all_candidate_scores))
    candidate_std = float(np.std(all_candidate_scores))
    if candidate_std == 0.0:
        raise ValueError("候选 query score 方差为 0")

    all_review_values = np.array([review_scores[user_id] for user_id in user_ids], dtype=float)
    review_mean = float(np.mean(all_review_values))
    review_std = float(np.std(all_review_values))
    if review_std == 0.0:
        raise ValueError("review user score 方差为 0")

    grouped_candidates = defaultdict(list)
    for row in scored_rows:
        grouped_candidates[row["user_id"]].append(row)

    selected = []
    for user_id in user_ids:
        review_z = (review_scores[user_id] - review_mean) / review_std
        group_z = (group_means[review_groups[user_id]] - review_mean) / review_std
        target_z = alpha * review_z + (1.0 - alpha) * group_z

        best_row = None
        best_distance = None
        for candidate in grouped_candidates[user_id]:
            candidate_z = (candidate["candidate_fisher_score"] - candidate_mean) / candidate_std
            distance = abs(candidate_z - target_z)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_row = candidate
        if best_row is None:
            raise ValueError(f"user {user_id} 没有候选")
        enriched = dict(best_row)
        enriched["review_group"] = review_groups[user_id]
        enriched["selection_alpha"] = alpha
        enriched["selection_target_z"] = float(target_z)
        enriched["selection_distance_z"] = float(best_distance)
        selected.append(enriched)
    return selected


def main() -> None:
    log("开始读取 review sentence 特征")
    review_rows = load_jsonl(REVIEW_SENTENCE_FILE)
    review_feature_names = feature_names_from_rows(review_rows)

    log("开始抽取 10 候选 query 的 20 特征")
    candidate_feature_names, candidate_rows = extract_candidate_features()
    if candidate_feature_names != review_feature_names:
        raise ValueError("review 与 candidate query 特征名不一致")

    log("写入候选 query 特征")
    with CANDIDATE_FEATURE_FILE.open("w", encoding="utf-8") as handle:
        for row in candidate_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    log("开始拟合 review Fisher 用户风格轴")
    scaler, weight, all_review_user_scores = fit_fisher_axis(review_rows, review_feature_names)
    candidate_user_ids = sorted({row["user_id"] for row in candidate_rows})
    review_user_scores = {
        user_id: all_review_user_scores[user_id]
        for user_id in candidate_user_ids
        if user_id in all_review_user_scores
    }
    if len(review_user_scores) != len(candidate_user_ids):
        missing = sorted(set(candidate_user_ids) - set(review_user_scores))
        raise ValueError(f"候选用户缺少 review score: {missing[:5]}")

    scored_rows = score_candidates(candidate_rows, candidate_feature_names, scaler, weight, review_user_scores)

    log("开始搜索每用户 1 条 query 的选择策略")
    search_results = []
    best_selected = None
    best_summary = None
    best_objective = None
    for alpha in ALPHA_GRID:
        selected = select_by_alpha(scored_rows, alpha)
        evaluation = evaluate_selection(selected)
        objective = (
            evaluation["spearman_r"]
            + evaluation["pearson_r"]
            + 0.25 * evaluation["group_query_score_range"]
            - 0.10 * evaluation["mean_absolute_query_review_diff"]
        )
        search_result = {
            "alpha": alpha,
            "objective": float(objective),
            **evaluation,
        }
        search_results.append(search_result)
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_selected = selected
            best_summary = search_result

    if best_selected is None or best_summary is None:
        raise ValueError("没有产生最优选择")

    log("写入最优选择记录")
    with SELECTED_RECORD_FILE.open("w", encoding="utf-8") as handle:
        for row in best_selected:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    feature_weights = [
        {"feature": name, "weight": float(value), "absolute_weight": float(abs(value))}
        for name, value in zip(review_feature_names, weight)
    ]
    feature_weights.sort(key=lambda item: item["absolute_weight"], reverse=True)

    selected_candidate_indices = np.array([row["candidate_index"] for row in best_selected], dtype=float)
    summary = {
        "category": CATEGORY,
        "query_10_file": str(QUERY_10_FILE),
        "review_sentence_file": str(REVIEW_SENTENCE_FILE),
        "candidate_feature_file": str(CANDIDATE_FEATURE_FILE),
        "selected_record_file": str(SELECTED_RECORD_FILE),
        "method": {
            "axis": "review_sentence_regularized_fisher_user_discriminative_axis",
            "selection": "per_user_candidate_closest_to_alpha_user_review_score_plus_group_review_centroid",
            "alpha_grid": ALPHA_GRID,
            "objective": "pearson + spearman + 0.25*group_range - 0.10*mean_abs_query_review_diff",
        },
        "data": {
            "candidate_users": len(candidate_user_ids),
            "candidate_rows": len(candidate_rows),
            "selected_rows": len(best_selected),
        },
        "best": best_summary,
        "search_results": search_results,
        "selected_candidate_index_summary": summarize_array(selected_candidate_indices),
        "feature_weights": feature_weights,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "candidate_feature_file": str(CANDIDATE_FEATURE_FILE),
        "selected_record_file": str(SELECTED_RECORD_FILE),
        "candidate_users": len(candidate_user_ids),
        "candidate_rows": len(candidate_rows),
        "best_alpha": best_summary["alpha"],
        "best_objective": best_summary["objective"],
        "pearson_r": best_summary["pearson_r"],
        "spearman_r": best_summary["spearman_r"],
        "group_query_score_means": best_summary["group_query_score_means"],
        "group_query_score_is_increasing": best_summary["group_query_score_is_increasing"],
        "group_query_score_kruskal_p_value": best_summary["group_query_score_kruskal_p_value"],
        "mean_absolute_query_review_diff": best_summary["mean_absolute_query_review_diff"],
        "selected_candidate_index_summary": summary["selected_candidate_index_summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
