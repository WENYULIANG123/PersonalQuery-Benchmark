#!/usr/bin/env python3
"""Select one query from 10 candidates per user with Fisher and 3-component shared PCA targets."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import select_one_query_from_10_for_group_and_individual_style as base  # noqa: E402


REPO_ROOT = base.REPO_ROOT
CATEGORY = base.CATEGORY
INPUT_DIR = base.INPUT_DIR
QUERY_10_FILE = base.QUERY_10_FILE
REVIEW_SENTENCE_FILE = base.REVIEW_SENTENCE_FILE
SHARED_PCA_N_COMPONENTS = 3
SUMMARY_FILE = INPUT_DIR / "select_one_query_from_10_joint_fisher_shared_pca_k3_summary.json"
CANDIDATE_FEATURE_FILE = INPUT_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
SELECTED_RECORD_FILE = INPUT_DIR / "selected_one_query_from_10_joint_fisher_shared_pca_k3_records.jsonl"

RIDGE = base.RIDGE
ALPHA_GRID = base.ALPHA_GRID
BETA_GRID = [round(value, 2) for value in np.linspace(0.0, 1.0, 21)]


def log(message: str) -> None:
    print(message, flush=True)


def summarize_array(values: np.ndarray) -> dict:
    return base.summarize_array(values)


def load_jsonl(path: Path) -> list[dict]:
    return base.load_jsonl(path)


def feature_names_from_rows(rows: list[dict]) -> list[str]:
    return base.feature_names_from_rows(rows)


def feature_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    return base.feature_matrix(rows, feature_names)


def assign_tertiles(scores: dict[str, float], user_ids: list[str]) -> tuple[dict[str, str], dict]:
    return base.assign_tertiles(scores, user_ids)


def extract_candidate_features() -> tuple[list[str], list[dict]]:
    return base.extract_candidate_features()


def normalize_components(component_scores: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    if component_scores.ndim != 2:
        raise ValueError("component_scores 必须是二维矩阵")
    if len(means) != component_scores.shape[1] or len(stds) != component_scores.shape[1]:
        raise ValueError("均值 / 标准差维度不一致")
    return (component_scores - means) / stds


def weighted_composite_score(component_scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if component_scores.ndim != 2:
        raise ValueError("component_scores 必须是二维矩阵")
    if len(weights) != component_scores.shape[1]:
        raise ValueError("weights 维度与 component_scores 不一致")
    return component_scores @ weights


def component_signs_by_anchor(raw_scores: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    if raw_scores.ndim != 2:
        raise ValueError("raw_scores 必须是二维矩阵")
    if len(raw_scores) != len(anchor):
        raise ValueError("raw_scores 与 anchor 长度不一致")

    signs = np.ones(raw_scores.shape[1], dtype=float)
    for idx in range(raw_scores.shape[1]):
        corr = np.corrcoef(raw_scores[:, idx], anchor)[0, 1]
        if np.isnan(corr):
            raise ValueError(f"第 {idx} 个主成分与 anchor 的相关性为 NaN")
        if corr < 0:
            signs[idx] = -1.0
    return signs


def fit_shared_pca_k(review_rows: list[dict], feature_names: list[str], n_components: int) -> dict:
    review_matrix = feature_matrix(review_rows, feature_names)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(review_matrix)
    pca = PCA(n_components=n_components, random_state=42)
    raw_scores = pca.fit_transform(standardized)

    anchor = np.array([float(row["features"]["max_dependency_depth"]) for row in review_rows], dtype=float)
    component_signs = component_signs_by_anchor(raw_scores, anchor)
    signed_scores = raw_scores * component_signs

    component_means = np.mean(signed_scores, axis=0)
    component_stds = np.std(signed_scores, axis=0)
    if np.any(component_stds == 0.0):
        raise ValueError("shared PCA 组件方差为 0")

    component_weights = np.array(pca.explained_variance_ratio_, dtype=float)
    weight_sum = float(np.sum(component_weights))
    if weight_sum == 0.0:
        raise ValueError("shared PCA explained_variance_ratio 之和为 0")
    component_weights = component_weights / weight_sum

    normalized_scores = normalize_components(signed_scores, component_means, component_stds)
    composite_scores = weighted_composite_score(normalized_scores, component_weights)
    anchor_corr = np.corrcoef(composite_scores, anchor)[0, 1]
    if np.isnan(anchor_corr):
        raise ValueError("shared PCA composite score 与 anchor 的相关性为 NaN")
    composite_sign = 1.0 if anchor_corr >= 0 else -1.0
    review_scores = composite_sign * composite_scores

    review_user_scores: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(review_rows, review_scores):
        review_user_scores[row["user_id"]].append(float(score))
    review_user_means = {
        user_id: float(np.mean(np.array(scores, dtype=float)))
        for user_id, scores in review_user_scores.items()
    }

    return {
        "scaler": scaler,
        "pca": pca,
        "component_signs": component_signs,
        "composite_sign": composite_sign,
        "component_means": component_means,
        "component_stds": component_stds,
        "component_weights": component_weights,
        "review_scores": review_scores,
        "review_user_means": review_user_means,
        "anchor_correlation": float(composite_sign * anchor_corr),
        "explained_variance_ratio": np.array(pca.explained_variance_ratio_, dtype=float),
    }


def score_shared_pca_candidates_k(
    candidate_rows: list[dict],
    feature_names: list[str],
    fit_result: dict,
) -> np.ndarray:
    matrix = feature_matrix(candidate_rows, feature_names)
    raw_scores = fit_result["pca"].transform(fit_result["scaler"].transform(matrix))
    signed_scores = raw_scores * fit_result["component_signs"]
    normalized_scores = normalize_components(signed_scores, fit_result["component_means"], fit_result["component_stds"])
    composite_scores = weighted_composite_score(normalized_scores, fit_result["component_weights"])
    return fit_result["composite_sign"] * composite_scores


def score_candidates_joint(
    candidate_rows: list[dict],
    feature_names: list[str],
    fisher_scaler: StandardScaler,
    fisher_weight: np.ndarray,
    fisher_review_scores: dict[str, float],
    shared_fit_result: dict,
    shared_review_scores: dict[str, float],
) -> list[dict]:
    fisher_scored_rows = base.score_candidates(candidate_rows, feature_names, fisher_scaler, fisher_weight, fisher_review_scores)
    shared_scores = score_shared_pca_candidates_k(candidate_rows, feature_names, shared_fit_result)
    if len(fisher_scored_rows) != len(shared_scores):
        raise ValueError("候选 Fisher / shared PCA 分数数量不一致")

    scored_rows = []
    for row, shared_score in zip(fisher_scored_rows, shared_scores):
        user_id = row["user_id"]
        if user_id not in shared_review_scores:
            raise ValueError(f"候选 user 缺少 shared review score: {user_id}")
        enriched = dict(row)
        enriched["candidate_shared_pca_score"] = float(shared_score)
        enriched["review_shared_pca_score"] = float(shared_review_scores[user_id])
        enriched["candidate_minus_review_shared_pca_score"] = float(shared_score - shared_review_scores[user_id])
        scored_rows.append(enriched)
    return scored_rows


def select_by_alpha_beta(scored_rows: list[dict], alpha: float, beta: float) -> list[dict]:
    user_ids = sorted({row["user_id"] for row in scored_rows})

    fisher_review_scores = {row["user_id"]: row["review_user_score"] for row in scored_rows}
    shared_review_scores = {row["user_id"]: row["review_shared_pca_score"] for row in scored_rows}

    fisher_groups, fisher_boundaries = assign_tertiles(fisher_review_scores, user_ids)
    shared_groups, shared_boundaries = assign_tertiles(shared_review_scores, user_ids)

    fisher_group_means = {
        label: float(np.mean([fisher_review_scores[user_id] for user_id in user_ids if fisher_groups[user_id] == label]))
        for label in ("low", "medium", "high")
    }
    shared_group_means = {
        label: float(np.mean([shared_review_scores[user_id] for user_id in user_ids if shared_groups[user_id] == label]))
        for label in ("low", "medium", "high")
    }

    fisher_candidate_scores = np.array([float(row["candidate_fisher_score"]) for row in scored_rows], dtype=float)
    shared_candidate_scores = np.array([float(row["candidate_shared_pca_score"]) for row in scored_rows], dtype=float)
    fisher_candidate_mean = float(np.mean(fisher_candidate_scores))
    fisher_candidate_std = float(np.std(fisher_candidate_scores))
    shared_candidate_mean = float(np.mean(shared_candidate_scores))
    shared_candidate_std = float(np.std(shared_candidate_scores))
    if fisher_candidate_std == 0.0:
        raise ValueError("候选 Fisher score 方差为 0")
    if shared_candidate_std == 0.0:
        raise ValueError("候选 shared PCA score 方差为 0")

    fisher_review_values = np.array([fisher_review_scores[user_id] for user_id in user_ids], dtype=float)
    shared_review_values = np.array([shared_review_scores[user_id] for user_id in user_ids], dtype=float)
    fisher_review_mean = float(np.mean(fisher_review_values))
    fisher_review_std = float(np.std(fisher_review_values))
    shared_review_mean = float(np.mean(shared_review_values))
    shared_review_std = float(np.std(shared_review_values))
    if fisher_review_std == 0.0:
        raise ValueError("review Fisher score 方差为 0")
    if shared_review_std == 0.0:
        raise ValueError("review shared PCA score 方差为 0")

    grouped_candidates = defaultdict(list)
    for row in scored_rows:
        grouped_candidates[row["user_id"]].append(row)

    selected = []
    for user_id in user_ids:
        fisher_review_z = (fisher_review_scores[user_id] - fisher_review_mean) / fisher_review_std
        fisher_group_z = (fisher_group_means[fisher_groups[user_id]] - fisher_review_mean) / fisher_review_std
        fisher_target_z = alpha * fisher_review_z + (1.0 - alpha) * fisher_group_z

        shared_review_z = (shared_review_scores[user_id] - shared_review_mean) / shared_review_std
        shared_group_z = (shared_group_means[shared_groups[user_id]] - shared_review_mean) / shared_review_std
        shared_target_z = alpha * shared_review_z + (1.0 - alpha) * shared_group_z

        best_row = None
        best_distance = None
        for candidate in grouped_candidates[user_id]:
            fisher_candidate_z = (float(candidate["candidate_fisher_score"]) - fisher_candidate_mean) / fisher_candidate_std
            shared_candidate_z = (float(candidate["candidate_shared_pca_score"]) - shared_candidate_mean) / shared_candidate_std
            distance = float(
                np.sqrt(
                    (1.0 - beta) * (fisher_candidate_z - fisher_target_z) ** 2
                    + beta * (shared_candidate_z - shared_target_z) ** 2
                )
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_row = candidate

        if best_row is None:
            raise ValueError(f"user {user_id} 没有候选")

        enriched = dict(best_row)
        enriched["fisher_review_group"] = fisher_groups[user_id]
        enriched["shared_review_group"] = shared_groups[user_id]
        enriched["review_group"] = fisher_groups[user_id]
        enriched["selection_alpha"] = alpha
        enriched["selection_beta"] = beta
        enriched["selection_target_fisher_z"] = float(fisher_target_z)
        enriched["selection_target_shared_pca_z"] = float(shared_target_z)
        enriched["selection_distance_z"] = float(best_distance)
        selected.append(enriched)

    return selected


def evaluate_axis_selection(
    selected_rows: list[dict],
    review_score_key: str,
    candidate_score_key: str,
    group_key: str,
) -> dict:
    review_scores = np.array([float(row[review_score_key]) for row in selected_rows], dtype=float)
    query_scores = np.array([float(row[candidate_score_key]) for row in selected_rows], dtype=float)
    abs_diffs = np.abs(query_scores - review_scores)

    pearson = pearsonr(review_scores, query_scores)
    spearman = spearmanr(review_scores, query_scores)

    grouped = defaultdict(list)
    for row in selected_rows:
        grouped[row[group_key]].append(float(row[candidate_score_key]))
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    log("开始读取 review sentence 特征")
    review_rows = load_jsonl(REVIEW_SENTENCE_FILE)
    review_feature_names = feature_names_from_rows(review_rows)

    log("开始抽取 10 候选 query 的 20 特征")
    candidate_feature_names, candidate_rows = extract_candidate_features()
    if candidate_feature_names != review_feature_names:
        raise ValueError("review 与 candidate query 特征名不一致")

    log("写入候选 query 特征")
    write_jsonl(CANDIDATE_FEATURE_FILE, candidate_rows)

    log("开始拟合 review Fisher 用户风格轴")
    fisher_scaler, fisher_weight, fisher_review_scores_all = base.fit_fisher_axis(review_rows, review_feature_names)

    log("开始拟合 review shared PCA 3 成分轴")
    shared_fit_result = fit_shared_pca_k(review_rows, review_feature_names, SHARED_PCA_N_COMPONENTS)

    candidate_user_ids = sorted({row["user_id"] for row in candidate_rows})
    fisher_review_scores = {
        user_id: fisher_review_scores_all[user_id]
        for user_id in candidate_user_ids
        if user_id in fisher_review_scores_all
    }
    shared_review_scores = {
        user_id: shared_fit_result["review_user_means"][user_id]
        for user_id in candidate_user_ids
        if user_id in shared_fit_result["review_user_means"]
    }
    if len(fisher_review_scores) != len(candidate_user_ids):
        missing = sorted(set(candidate_user_ids) - set(fisher_review_scores))
        raise ValueError(f"候选用户缺少 Fisher review score: {missing[:5]}")
    if len(shared_review_scores) != len(candidate_user_ids):
        missing = sorted(set(candidate_user_ids) - set(shared_review_scores))
        raise ValueError(f"候选用户缺少 shared review score: {missing[:5]}")

    scored_rows = score_candidates_joint(
        candidate_rows,
        candidate_feature_names,
        fisher_scaler,
        fisher_weight,
        fisher_review_scores,
        shared_fit_result,
        shared_review_scores,
    )

    log("开始搜索 Fisher + 3 成分 shared PCA 的联合选择策略")
    search_results = []
    best_selected = None
    best_summary = None
    best_objective = None

    for alpha in ALPHA_GRID:
        for beta in BETA_GRID:
            selected = select_by_alpha_beta(scored_rows, alpha, beta)
            fisher_eval = evaluate_axis_selection(selected, "review_user_score", "candidate_fisher_score", "fisher_review_group")
            shared_eval = evaluate_axis_selection(selected, "review_shared_pca_score", "candidate_shared_pca_score", "shared_review_group")
            objective = (
                fisher_eval["pearson_r"]
                + fisher_eval["spearman_r"]
                + shared_eval["pearson_r"]
                + shared_eval["spearman_r"]
                + 0.25 * (fisher_eval["group_query_score_range"] + shared_eval["group_query_score_range"])
                - 0.10 * (fisher_eval["mean_absolute_query_review_diff"] + shared_eval["mean_absolute_query_review_diff"])
            )
            search_result = {
                "alpha": alpha,
                "beta": beta,
                "objective": float(objective),
                "fisher_pearson_r": fisher_eval["pearson_r"],
                "fisher_spearman_r": fisher_eval["spearman_r"],
                "fisher_group_range": fisher_eval["group_query_score_range"],
                "fisher_mean_absolute_query_review_diff": fisher_eval["mean_absolute_query_review_diff"],
                "fisher_group_kruskal_p_value": fisher_eval["group_query_score_kruskal_p_value"],
                "shared_pearson_r": shared_eval["pearson_r"],
                "shared_spearman_r": shared_eval["spearman_r"],
                "shared_group_range": shared_eval["group_query_score_range"],
                "shared_mean_absolute_query_review_diff": shared_eval["mean_absolute_query_review_diff"],
                "shared_group_kruskal_p_value": shared_eval["group_query_score_kruskal_p_value"],
            }
            search_results.append(search_result)
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_selected = selected
                best_summary = {
                    "alpha": alpha,
                    "beta": beta,
                    "objective": float(objective),
                    "fisher": fisher_eval,
                    "shared": shared_eval,
                }

    if best_selected is None or best_summary is None:
        raise ValueError("没有产生最优选择")

    log("写入最优选择记录")
    write_jsonl(SELECTED_RECORD_FILE, best_selected)

    selected_candidate_indices = np.array([float(row["candidate_index"]) for row in best_selected], dtype=float)
    selected_fisher_scores = np.array([float(row["candidate_fisher_score"]) for row in best_selected], dtype=float)
    selected_shared_scores = np.array([float(row["candidate_shared_pca_score"]) for row in best_selected], dtype=float)

    summary = {
        "category": CATEGORY,
        "query_10_file": str(QUERY_10_FILE),
        "review_sentence_file": str(REVIEW_SENTENCE_FILE),
        "candidate_feature_file": str(CANDIDATE_FEATURE_FILE),
        "selected_record_file": str(SELECTED_RECORD_FILE),
        "method": {
            "axis_fisher": "review_sentence_regularized_fisher_user_discriminative_axis",
            "axis_shared_pca": "review_sentence_shared_pca_axis_k3",
            "shared_pca_n_components": SHARED_PCA_N_COMPONENTS,
            "selection": "per_user_candidate_closest_to_joint_fisher_and_shared_pca_target",
            "alpha_grid": ALPHA_GRID,
            "beta_grid": BETA_GRID,
            "objective": "fisher + shared_pca correlations + 0.25*group_range - 0.10*mean_abs_diff",
        },
        "data": {
            "candidate_users": len(candidate_user_ids),
            "candidate_rows": len(candidate_rows),
            "selected_rows": len(best_selected),
        },
        "fisher_review_score_summary": summarize_array(np.array([fisher_review_scores[u] for u in candidate_user_ids], dtype=float)),
        "shared_review_score_summary": summarize_array(np.array([shared_review_scores[u] for u in candidate_user_ids], dtype=float)),
        "shared_review_sentence_score_summary": summarize_array(np.array(shared_fit_result["review_scores"], dtype=float)),
        "shared_pca": {
            "n_components": SHARED_PCA_N_COMPONENTS,
            "explained_variance_ratio": [float(value) for value in shared_fit_result["explained_variance_ratio"]],
            "component_weights": [float(value) for value in shared_fit_result["component_weights"]],
            "component_signs": [float(value) for value in shared_fit_result["component_signs"]],
            "anchor_correlation": float(shared_fit_result["anchor_correlation"]),
        },
        "best": best_summary,
        "search_results": search_results,
        "selected_candidate_index_summary": summarize_array(selected_candidate_indices),
        "selected_fisher_score_summary": summarize_array(selected_fisher_scores),
        "selected_shared_pca_score_summary": summarize_array(selected_shared_scores),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "candidate_feature_file": str(CANDIDATE_FEATURE_FILE),
        "selected_record_file": str(SELECTED_RECORD_FILE),
        "candidate_users": len(candidate_user_ids),
        "candidate_rows": len(candidate_rows),
        "best_alpha": best_summary["alpha"],
        "best_beta": best_summary["beta"],
        "best_objective": best_summary["objective"],
        "fisher_pearson_r": best_summary["fisher"]["pearson_r"],
        "fisher_spearman_r": best_summary["fisher"]["spearman_r"],
        "fisher_group_query_score_means": best_summary["fisher"]["group_query_score_means"],
        "fisher_group_query_score_is_increasing": best_summary["fisher"]["group_query_score_is_increasing"],
        "fisher_group_query_score_kruskal_p_value": best_summary["fisher"]["group_query_score_kruskal_p_value"],
        "shared_pearson_r": best_summary["shared"]["pearson_r"],
        "shared_spearman_r": best_summary["shared"]["spearman_r"],
        "shared_group_query_score_means": best_summary["shared"]["group_query_score_means"],
        "shared_group_query_score_is_increasing": best_summary["shared"]["group_query_score_is_increasing"],
        "shared_group_query_score_kruskal_p_value": best_summary["shared"]["group_query_score_kruskal_p_value"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
