#!/usr/bin/env python3
"""Train a supervised 20-feature combination to identify true query-user style matches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
REVIEW_SENTENCE_FILE = INPUT_DIR / "review_sentence_pca_distribution_sentences.jsonl"
QUERY_FEATURE_FILE = INPUT_DIR / "single_query_clause_features.jsonl"
SUMMARY_FILE = INPUT_DIR / "twenty_feature_supervised_style_matcher_summary.json"
PAIR_RECORD_FILE = INPUT_DIR / "twenty_feature_supervised_style_matcher_test_pairs.jsonl"

RANDOM_SEED = 42
TEST_SIZE = 0.25
NEGATIVES_PER_POSITIVE = 1


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


def build_review_user_vectors(review_rows: list[dict], feature_names: list[str], scaler: StandardScaler) -> dict[str, np.ndarray]:
    review_matrix = scaler.transform(feature_matrix(review_rows, feature_names))
    grouped: dict[str, list[np.ndarray]] = {}
    for row, vector in zip(review_rows, review_matrix):
        user_id = row["user_id"]
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("review row 缺少 user_id")
        grouped.setdefault(user_id, []).append(vector)

    if not grouped:
        raise ValueError("没有 review user vector")
    return {user_id: np.mean(np.vstack(vectors), axis=0) for user_id, vectors in grouped.items()}


def build_query_rows(query_rows: list[dict], feature_names: list[str], scaler: StandardScaler, review_vectors: dict[str, np.ndarray]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    query_matrix = scaler.transform(feature_matrix(query_rows, feature_names))
    kept_rows = []
    kept_vectors = []
    groups = []
    for row, vector in zip(query_rows, query_matrix):
        user_id = row["user_id"]
        if user_id not in review_vectors:
            raise ValueError(f"query user {user_id} 缺少 review vector")
        kept_rows.append(row)
        kept_vectors.append(vector)
        groups.append(user_id)

    if not kept_rows:
        raise ValueError("没有可用 query")
    return kept_rows, np.vstack(kept_vectors), np.array(groups, dtype=object)


def choose_negative_user(true_user: str, candidate_users: np.ndarray, rng: np.random.Generator) -> str:
    if len(candidate_users) < 2:
        raise ValueError("负样本候选用户不足")
    while True:
        user = str(candidate_users[int(rng.integers(0, len(candidate_users)))])
        if user != true_user:
            return user


def build_pairs(
    query_rows: list[dict],
    query_matrix: np.ndarray,
    review_vectors: dict[str, np.ndarray],
    candidate_negative_users: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    x_rows = []
    y_rows = []
    pair_records = []

    for idx, (row, query_vector) in enumerate(zip(query_rows, query_matrix)):
        true_user = row["user_id"]
        true_review_vector = review_vectors[true_user]
        true_diff = np.abs(query_vector - true_review_vector)
        x_rows.append(true_diff)
        y_rows.append(1)
        pair_records.append({
            "pair_type": "true",
            "query_index": idx,
            "query_user_id": true_user,
            "matched_user_id": true_user,
            "asin": row["asin"],
        })

        for _ in range(NEGATIVES_PER_POSITIVE):
            negative_user = choose_negative_user(true_user, candidate_negative_users, rng)
            negative_diff = np.abs(query_vector - review_vectors[negative_user])
            x_rows.append(negative_diff)
            y_rows.append(0)
            pair_records.append({
                "pair_type": "random",
                "query_index": idx,
                "query_user_id": true_user,
                "matched_user_id": negative_user,
                "asin": row["asin"],
            })

    return np.vstack(x_rows), np.array(y_rows, dtype=int), pair_records


def group_split(query_rows: list[dict], groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    indices = np.arange(len(query_rows))
    train_idx, test_idx = next(splitter.split(indices, groups=groups))
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("train/test split 为空")
    return train_idx, test_idx


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def evaluate_pairs(model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> dict:
    scores = model.predict_proba(x)[:, 1]
    auc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)
    true_scores = scores[y == 1]
    random_scores = scores[y == 0]
    if len(true_scores) != len(random_scores):
        raise ValueError("当前实现要求正负样本数量相同")
    return {
        "auc": float(auc),
        "average_precision": float(ap),
        "true_score_summary": summarize_array(true_scores),
        "random_score_summary": summarize_array(random_scores),
        "true_score_higher_share": float(np.mean(true_scores > random_scores)),
        "mean_true_minus_random_score": float(np.mean(true_scores - random_scores)),
    }


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    log("开始读取 review sentence 20 特征")
    review_rows = load_jsonl(REVIEW_SENTENCE_FILE)
    log(f"review sentence 行数: {len(review_rows)}")

    log("开始读取 query 20 特征")
    raw_query_rows = load_jsonl(QUERY_FEATURE_FILE)
    log(f"query 行数: {len(raw_query_rows)}")

    feature_names = feature_names_from_rows(review_rows)
    if feature_names_from_rows(raw_query_rows) != feature_names:
        raise ValueError("review 与 query 特征名不一致")

    scaler = StandardScaler()
    scaler.fit(feature_matrix(review_rows, feature_names))

    review_vectors = build_review_user_vectors(review_rows, feature_names, scaler)
    query_rows, query_matrix, groups = build_query_rows(raw_query_rows, feature_names, scaler, review_vectors)

    train_idx, test_idx = group_split(query_rows, groups)
    train_users = np.unique(groups[train_idx])
    test_users = np.unique(groups[test_idx])
    if set(train_users) & set(test_users):
        raise ValueError("train/test 用户集合有重叠")

    log(f"train query 数: {len(train_idx)}, test query 数: {len(test_idx)}")
    log(f"train user 数: {len(train_users)}, test user 数: {len(test_users)}")

    x_train, y_train, _ = build_pairs(
        [query_rows[idx] for idx in train_idx],
        query_matrix[train_idx],
        review_vectors,
        train_users,
        rng,
    )
    x_test, y_test, test_pair_records = build_pairs(
        [query_rows[idx] for idx in test_idx],
        query_matrix[test_idx],
        review_vectors,
        test_users,
        rng,
    )

    log("开始训练 L1 logistic style matcher")
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=1000,
    )
    model.fit(x_train, y_train)

    train_eval = evaluate_pairs(model, x_train, y_train)
    test_eval = evaluate_pairs(model, x_test, y_test)
    test_scores = model.predict_proba(x_test)[:, 1]

    test_records = []
    for record, label, score in zip(test_pair_records, y_test, test_scores):
        enriched = dict(record)
        enriched["label"] = int(label)
        enriched["style_match_score"] = float(score)
        test_records.append(enriched)

    coef = model.coef_[0]
    weights = []
    for feature_name, weight in zip(feature_names, coef):
        weights.append({
            "feature": feature_name,
            "abs_difference_weight": float(weight),
            "absolute_weight": float(abs(weight)),
        })
    weights.sort(key=lambda item: item["absolute_weight"], reverse=True)

    summary = {
        "category": CATEGORY,
        "review_sentence_file": str(REVIEW_SENTENCE_FILE),
        "query_feature_file": str(QUERY_FEATURE_FILE),
        "test_pair_record_file": str(PAIR_RECORD_FILE),
        "feature_names": feature_names,
        "model": {
            "type": "LogisticRegression",
            "input": "absolute_difference_between_query_features_and_review_user_mean_features",
            "penalty": "l1",
            "solver": "liblinear",
            "C": 1.0,
            "class_weight": "balanced",
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
            "negatives_per_positive": NEGATIVES_PER_POSITIVE,
        },
        "split": {
            "train_queries": int(len(train_idx)),
            "test_queries": int(len(test_idx)),
            "train_users": int(len(train_users)),
            "test_users": int(len(test_users)),
        },
        "train_eval": train_eval,
        "test_eval": test_eval,
        "feature_weights": weights,
        "nonzero_weight_count": int(np.sum(coef != 0)),
    }

    write_jsonl(PAIR_RECORD_FILE, test_records)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "test_auc": test_eval["auc"],
        "test_average_precision": test_eval["average_precision"],
        "test_true_score_higher_share": test_eval["true_score_higher_share"],
        "test_mean_true_minus_random_score": test_eval["mean_true_minus_random_score"],
        "nonzero_weight_count": summary["nonzero_weight_count"],
        "top_weights": weights[:8],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
