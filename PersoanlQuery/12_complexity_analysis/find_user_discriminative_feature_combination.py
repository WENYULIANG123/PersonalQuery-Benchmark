#!/usr/bin/env python3
"""Find a 20-feature linear combination that best separates users."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
REVIEW_SENTENCE_FILE = INPUT_DIR / "review_sentence_pca_distribution_sentences.jsonl"
QUERY_FEATURE_FILE = INPUT_DIR / "single_query_clause_features.jsonl"
SUMMARY_FILE = INPUT_DIR / "user_discriminative_feature_combination_summary.json"
USER_SCORE_FILE = INPUT_DIR / "user_discriminative_feature_combination_user_scores.jsonl"
QUERY_SCORE_FILE = INPUT_DIR / "user_discriminative_feature_combination_query_scores.jsonl"

RANDOM_SEED = 42
TEST_SIZE = 0.4
RIDGE = 1e-3


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


def split_review_sentences(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    counts = Counter(row["user_id"] for row in rows)
    if any(count < 2 for count in counts.values()):
        bad_users = [user_id for user_id, count in counts.items() if count < 2]
        raise ValueError(f"存在评论句子少于 2 条的用户，无法做用户内留出评估: {bad_users[:5]}")

    rng = np.random.default_rng(RANDOM_SEED)
    train_rows = []
    test_rows = []
    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_user[row["user_id"]].append(row)

    for user_id in sorted(by_user.keys()):
        user_rows = list(by_user[user_id])
        indices = np.arange(len(user_rows))
        rng.shuffle(indices)
        test_count = max(1, int(round(len(user_rows) * TEST_SIZE)))
        if test_count >= len(user_rows):
            test_count = len(user_rows) - 1
        test_index_set = set(int(idx) for idx in indices[:test_count])
        for idx, row in enumerate(user_rows):
            if idx in test_index_set:
                test_rows.append(row)
            else:
                train_rows.append(row)

    if not train_rows or not test_rows:
        raise ValueError("review train/test split 为空")
    return train_rows, test_rows


def compute_scatter_matrices(matrix: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if matrix.shape[0] != len(labels):
        raise ValueError("matrix 与 labels 长度不一致")
    global_mean = np.mean(matrix, axis=0)
    by_user: dict[str, list[np.ndarray]] = defaultdict(list)
    for vector, label in zip(matrix, labels):
        by_user[label].append(vector)
    if len(by_user) < 2:
        raise ValueError("用户数不足，无法计算用户间散度")

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


def orient_weight(weight: np.ndarray, feature_names: list[str]) -> np.ndarray:
    anchor = "max_dependency_depth"
    if anchor not in feature_names:
        raise ValueError("缺少 max_dependency_depth，无法固定方向")
    anchor_weight = float(weight[feature_names.index(anchor)])
    if anchor_weight < 0:
        return -weight
    return weight


def fit_fisher_axis(train_matrix: np.ndarray, train_labels: list[str], feature_names: list[str]) -> tuple[np.ndarray, dict]:
    between, within, _ = compute_scatter_matrices(train_matrix, train_labels)
    regularized_within = within + RIDGE * np.eye(within.shape[0])
    eigenvalues, eigenvectors = eigh(between, regularized_within)
    best_idx = int(np.argmax(eigenvalues))
    weight = np.real(eigenvectors[:, best_idx]).astype(float)
    norm = float(np.linalg.norm(weight))
    if norm == 0.0:
        raise ValueError("Fisher 轴权重范数为 0")
    weight = orient_weight(weight / norm, feature_names)
    metadata = {
        "largest_generalized_eigenvalue": float(np.real(eigenvalues[best_idx])),
        "ridge": RIDGE,
    }
    return weight, metadata


def score_matrix(matrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
    scores = matrix @ weight
    if np.any(~np.isfinite(scores)):
        raise ValueError("score 中存在非有限值")
    return scores


def variance_ratio(scores: np.ndarray, labels: list[str]) -> dict:
    by_user: dict[str, list[float]] = defaultdict(list)
    for score, label in zip(scores, labels):
        by_user[label].append(float(score))
    user_means = np.array([np.mean(values) for values in by_user.values()], dtype=float)
    within_values = []
    for values in by_user.values():
        arr = np.array(values, dtype=float)
        within_values.extend((arr - np.mean(arr)).tolist())
    within_arr = np.array(within_values, dtype=float)
    between_var = float(np.var(user_means))
    within_var = float(np.var(within_arr))
    if within_var == 0.0:
        raise ValueError("用户内方差为 0，无法计算分离比")
    return {
        "user_count": int(len(by_user)),
        "between_user_variance": between_var,
        "within_user_variance": within_var,
        "between_within_variance_ratio": float(between_var / within_var),
        "user_mean_score_summary": summarize_array(user_means),
        "within_residual_score_summary": summarize_array(within_arr),
    }


def nearest_centroid_accuracy(test_matrix: np.ndarray, test_labels: list[str], train_centroids: dict[str, np.ndarray], weight: np.ndarray) -> dict:
    centroid_scores = {user_id: float(vector @ weight) for user_id, vector in train_centroids.items()}
    users = sorted(centroid_scores.keys())
    if set(test_labels) - set(users):
        raise ValueError("test 中存在 train centroid 没有覆盖的用户")
    predictions = []
    margins = []
    for vector, true_user in zip(test_matrix, test_labels):
        score = float(vector @ weight)
        distances = np.array([abs(score - centroid_scores[user_id]) for user_id in users], dtype=float)
        order = np.argsort(distances)
        predicted_user = users[int(order[0])]
        predictions.append(predicted_user)
        if len(order) < 2:
            raise ValueError("centroid 用户数不足")
        margins.append(float(distances[int(order[1])] - distances[int(order[0])]))
    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "test_count": int(len(test_labels)),
        "user_count": int(len(users)),
        "margin_summary": summarize_array(np.array(margins, dtype=float)),
    }


def build_query_alignment(
    query_rows: list[dict],
    feature_names: list[str],
    scaler: StandardScaler,
    weight: np.ndarray,
    review_user_scores: dict[str, float],
) -> tuple[list[dict], dict]:
    query_matrix = scaler.transform(feature_matrix(query_rows, feature_names))
    query_scores = score_matrix(query_matrix, weight)
    records = []
    review_scores = []
    matched_query_scores = []
    for row, query_score in zip(query_rows, query_scores):
        user_id = row["user_id"]
        if user_id not in review_user_scores:
            raise ValueError(f"query user {user_id} 缺少 review 用户分数")
        review_score = float(review_user_scores[user_id])
        records.append({
            "user_id": user_id,
            "asin": row["asin"],
            "query": row["query"],
            "user_discriminative_query_score": float(query_score),
            "review_user_centroid_score": review_score,
            "query_minus_review_score": float(query_score - review_score),
        })
        review_scores.append(review_score)
        matched_query_scores.append(float(query_score))

    review_arr = np.array(review_scores, dtype=float)
    query_arr = np.array(matched_query_scores, dtype=float)
    pearson = pearsonr(review_arr, query_arr)
    spearman = spearmanr(review_arr, query_arr)
    alignment = {
        "matched_query_count": int(len(records)),
        "pearson_r": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "review_user_score_summary": summarize_array(review_arr),
        "query_score_summary": summarize_array(query_arr),
        "query_minus_review_score_summary": summarize_array(query_arr - review_arr),
    }
    return records, alignment


def train_pairwise_style_matcher(
    query_rows: list[dict],
    query_matrix: np.ndarray,
    review_user_vectors: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict:
    groups = np.array([row["user_id"] for row in query_rows], dtype=object)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    indices = np.arange(len(query_rows))
    train_idx, test_idx = next(splitter.split(indices, groups=groups))
    train_users = np.unique(groups[train_idx])
    test_users = np.unique(groups[test_idx])
    if set(train_users) & set(test_users):
        raise ValueError("query train/test 用户集合有重叠")

    def choose_negative_user(true_user: str, candidates: np.ndarray) -> str:
        if len(candidates) < 2:
            raise ValueError("负样本候选用户不足")
        while True:
            user_id = str(candidates[int(rng.integers(0, len(candidates)))])
            if user_id != true_user:
                return user_id

    def make_pairs(row_indices: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_rows = []
        y_rows = []
        for idx in row_indices:
            true_user = query_rows[int(idx)]["user_id"]
            query_vector = query_matrix[int(idx)]
            x_rows.append(np.abs(query_vector - review_user_vectors[true_user]))
            y_rows.append(1)
            negative_user = choose_negative_user(true_user, candidates)
            x_rows.append(np.abs(query_vector - review_user_vectors[negative_user]))
            y_rows.append(0)
        return np.vstack(x_rows), np.array(y_rows, dtype=int)

    x_train, y_train = make_pairs(train_idx, train_users)
    x_test, y_test = make_pairs(test_idx, test_users)
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=1000,
    )
    model.fit(x_train, y_train)
    test_scores = model.predict_proba(x_test)[:, 1]
    return {
        "model": {
            "type": "LogisticRegression",
            "input": "absolute_difference_between_query_features_and_review_user_mean_features",
            "penalty": "l1",
            "solver": "liblinear",
            "C": 1.0,
            "class_weight": "balanced",
        },
        "split": {
            "train_queries": int(len(train_idx)),
            "test_queries": int(len(test_idx)),
            "train_users": int(len(train_users)),
            "test_users": int(len(test_users)),
        },
        "test_auc": float(roc_auc_score(y_test, test_scores)),
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
        "feature_weights": [
            {
                "feature": feature_name,
                "abs_difference_weight": float(weight),
                "absolute_weight": float(abs(weight)),
            }
            for feature_name, weight in sorted(
                zip(feature_names_from_rows(query_rows), model.coef_[0]),
                key=lambda pair: abs(float(pair[1])),
                reverse=True,
            )
        ],
        "nonzero_weight_count": int(np.sum(model.coef_[0] != 0)),
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
        raise ValueError("review 与 query 特征名不一致")

    review_users = {row["user_id"] for row in review_rows}
    query_users = {row["user_id"] for row in query_rows}
    missing_query_users = sorted(query_users - review_users)
    if missing_query_users:
        raise ValueError(f"存在 query 用户缺少 review 特征: {missing_query_users[:5]}")

    train_review_rows, test_review_rows = split_review_sentences(review_rows)
    train_labels = [row["user_id"] for row in train_review_rows]
    test_labels = [row["user_id"] for row in test_review_rows]

    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(feature_matrix(train_review_rows, feature_names))
    test_matrix = scaler.transform(feature_matrix(test_review_rows, feature_names))
    all_review_matrix = scaler.transform(feature_matrix(review_rows, feature_names))
    query_matrix = scaler.transform(feature_matrix(query_rows, feature_names))

    log("开始拟合 Fisher 用户判别轴")
    fisher_weight, fisher_metadata = fit_fisher_axis(train_matrix, train_labels, feature_names)
    _, _, train_centroids = compute_scatter_matrices(train_matrix, train_labels)
    test_scores = score_matrix(test_matrix, fisher_weight)
    train_scores = score_matrix(train_matrix, fisher_weight)
    all_review_scores = score_matrix(all_review_matrix, fisher_weight)

    log("开始计算用户分离度与留出识别效果")
    train_separation = variance_ratio(train_scores, train_labels)
    test_separation = variance_ratio(test_scores, test_labels)
    centroid_eval = nearest_centroid_accuracy(test_matrix, test_labels, train_centroids, fisher_weight)

    review_user_score_values: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(review_rows, all_review_scores):
        review_user_score_values[row["user_id"]].append(float(score))
    review_user_scores = {
        user_id: float(np.mean(values))
        for user_id, values in review_user_score_values.items()
    }
    user_score_records = [
        {
            "user_id": user_id,
            "review_sentence_count": len(review_user_score_values[user_id]),
            "review_user_discriminative_score_mean": review_user_scores[user_id],
            "review_user_discriminative_score_summary": summarize_array(np.array(review_user_score_values[user_id], dtype=float)),
        }
        for user_id in sorted(review_user_scores.keys())
    ]

    log("开始投影 query 并计算 review-query 对齐")
    query_records, query_alignment = build_query_alignment(query_rows, feature_names, scaler, fisher_weight, review_user_scores)

    log("开始训练 query-review pairwise style matcher 作为监督对照")
    review_user_vector_values: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, vector in zip(review_rows, all_review_matrix):
        review_user_vector_values[row["user_id"]].append(vector)
    review_user_vectors = {
        user_id: np.mean(np.vstack(vectors), axis=0)
        for user_id, vectors in review_user_vector_values.items()
    }
    supervised_matcher = train_pairwise_style_matcher(query_rows, query_matrix, review_user_vectors, rng)

    weights = [
        {
            "feature": feature_name,
            "weight": float(weight),
            "absolute_weight": float(abs(weight)),
        }
        for feature_name, weight in zip(feature_names, fisher_weight)
    ]
    weights.sort(key=lambda item: item["absolute_weight"], reverse=True)

    summary = {
        "category": CATEGORY,
        "review_sentence_file": str(REVIEW_SENTENCE_FILE),
        "query_feature_file": str(QUERY_FEATURE_FILE),
        "user_score_file": str(USER_SCORE_FILE),
        "query_score_file": str(QUERY_SCORE_FILE),
        "feature_names": feature_names,
        "method": {
            "name": "regularized_fisher_discriminant_axis",
            "objective": "maximize_between_user_variance_over_within_user_variance_on_review_sentences",
            "standardization_fit_on": "review_sentence_train_split",
            "test_size": TEST_SIZE,
            "random_seed": RANDOM_SEED,
            **fisher_metadata,
        },
        "data": {
            "review_sentence_count": int(len(review_rows)),
            "query_count": int(len(query_rows)),
            "user_count": int(len(review_users)),
            "train_review_sentence_count": int(len(train_review_rows)),
            "test_review_sentence_count": int(len(test_review_rows)),
        },
        "train_separation": train_separation,
        "test_separation": test_separation,
        "heldout_review_nearest_centroid_eval": centroid_eval,
        "query_review_alignment": query_alignment,
        "feature_weights": weights,
        "supervised_query_review_style_matcher": supervised_matcher,
    }

    write_jsonl(USER_SCORE_FILE, user_score_records)
    write_jsonl(QUERY_SCORE_FILE, query_records)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "user_score_file": str(USER_SCORE_FILE),
        "query_score_file": str(QUERY_SCORE_FILE),
        "test_between_within_ratio": test_separation["between_within_variance_ratio"],
        "heldout_review_nearest_centroid_accuracy": centroid_eval["accuracy"],
        "query_review_pearson_r": query_alignment["pearson_r"],
        "query_review_spearman_r": query_alignment["spearman_r"],
        "top_fisher_weights": weights[:10],
        "supervised_matcher_test_auc": supervised_matcher["test_auc"],
        "supervised_matcher_top_weights": supervised_matcher["feature_weights"][:10],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
