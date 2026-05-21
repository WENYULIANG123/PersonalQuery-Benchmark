#!/usr/bin/env python3
"""Sweep MLP depths for 20-feature user style matching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
REVIEW_SENTENCE_FILE = INPUT_DIR / "review_sentence_pca_distribution_sentences.jsonl"
QUERY_FEATURE_FILE = INPUT_DIR / "single_query_clause_features.jsonl"
SUMMARY_FILE = INPUT_DIR / "twenty_feature_mlp_depth_sweep_summary.json"

RANDOM_SEED = 42
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.20
NEGATIVES_PER_POSITIVE = 1
BATCH_SIZE = 256
MAX_EPOCHS = 250
PATIENCE = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.10

ARCHITECTURES = {
    "shallow": [32, 16],
    "medium": [64, 32, 16],
    "deep": [128, 64, 32, 16],
}


class StyleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


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
        dtype=np.float32,
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
    review_matrix = scaler.transform(feature_matrix(review_rows, feature_names)).astype(np.float32)
    grouped: dict[str, list[np.ndarray]] = {}
    for row, vector in zip(review_rows, review_matrix):
        user_id = row["user_id"]
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("review row 缺少 user_id")
        grouped.setdefault(user_id, []).append(vector)
    if not grouped:
        raise ValueError("没有 review user vector")
    return {user_id: np.mean(np.vstack(vectors), axis=0).astype(np.float32) for user_id, vectors in grouped.items()}


def build_query_vectors(query_rows: list[dict], feature_names: list[str], scaler: StandardScaler, review_vectors: dict[str, np.ndarray]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    query_matrix = scaler.transform(feature_matrix(query_rows, feature_names)).astype(np.float32)
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
    return kept_rows, np.vstack(kept_vectors).astype(np.float32), np.array(groups, dtype=object)


def split_indices(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(outer.split(indices, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE_WITHIN_TRAIN, random_state=RANDOM_SEED + 1)
    train_rel_idx, val_rel_idx = next(inner.split(train_val_idx, groups=groups[train_val_idx]))
    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]
    if set(groups[train_idx]) & set(groups[val_idx]):
        raise ValueError("train/val 用户集合有重叠")
    if set(groups[train_idx]) & set(groups[test_idx]):
        raise ValueError("train/test 用户集合有重叠")
    if set(groups[val_idx]) & set(groups[test_idx]):
        raise ValueError("val/test 用户集合有重叠")
    return train_idx, val_idx, test_idx


def choose_negative_user(true_user: str, candidate_users: np.ndarray, rng: np.random.Generator) -> str:
    if len(candidate_users) < 2:
        raise ValueError("负样本候选用户不足")
    while True:
        candidate = str(candidate_users[int(rng.integers(0, len(candidate_users)))])
        if candidate != true_user:
            return candidate


def build_pairs(
    query_rows: list[dict],
    query_matrix: np.ndarray,
    review_vectors: dict[str, np.ndarray],
    candidate_negative_users: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for row, query_vector in zip(query_rows, query_matrix):
        true_user = row["user_id"]
        x_rows.append(np.abs(query_vector - review_vectors[true_user]))
        y_rows.append(1)
        for _ in range(NEGATIVES_PER_POSITIVE):
            negative_user = choose_negative_user(true_user, candidate_negative_users, rng)
            x_rows.append(np.abs(query_vector - review_vectors[negative_user]))
            y_rows.append(0)
    return np.vstack(x_rows).astype(np.float32), np.array(y_rows, dtype=np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def predict_scores(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    scores = []
    loader = make_loader(x, np.zeros(len(x), dtype=np.float32), shuffle=False)
    with torch.no_grad():
        for batch_x, _ in loader:
            scores.append(torch.sigmoid(model(batch_x)).cpu().numpy())
    return np.concatenate(scores)


def evaluate_scores(y: np.ndarray, scores: np.ndarray) -> dict:
    true_scores = scores[y == 1]
    random_scores = scores[y == 0]
    if len(true_scores) != len(random_scores):
        raise ValueError("当前实现要求正负样本数量相同")
    return {
        "auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "true_score_summary": summarize_array(true_scores),
        "random_score_summary": summarize_array(random_scores),
        "true_score_higher_share": float(np.mean(true_scores > random_scores)),
        "mean_true_minus_random_score": float(np.mean(true_scores - random_scores)),
    }


def train_one_architecture(name: str, hidden_dims: list[int], x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[StyleMLP, list[dict], int]:
    set_seed(RANDOM_SEED + len(hidden_dims))
    model = StyleMLP(input_dim=x_train.shape[1], hidden_dims=hidden_dims)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(x_train, y_train, shuffle=True)
    best_state = None
    best_val_auc = -np.inf
    best_epoch = 0
    patience_left = PATIENCE
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_scores = predict_scores(model, x_val)
        val_auc = float(roc_auc_score(y_val, val_scores))
        val_ap = float(average_precision_score(y_val, val_scores))
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_auc": val_auc,
            "val_average_precision": val_ap,
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise ValueError(f"{name} 训练未产生 best_state")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def main() -> None:
    set_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    log("开始读取 review/query 20 特征")
    review_rows = load_jsonl(REVIEW_SENTENCE_FILE)
    query_rows_raw = load_jsonl(QUERY_FEATURE_FILE)
    log(f"review sentence 行数: {len(review_rows)}")
    log(f"query 行数: {len(query_rows_raw)}")

    feature_names = feature_names_from_rows(review_rows)
    if feature_names_from_rows(query_rows_raw) != feature_names:
        raise ValueError("review 与 query 特征名不一致")

    scaler = StandardScaler()
    scaler.fit(feature_matrix(review_rows, feature_names))
    review_vectors = build_review_user_vectors(review_rows, feature_names, scaler)
    query_rows, query_matrix, groups = build_query_vectors(query_rows_raw, feature_names, scaler, review_vectors)

    train_idx, val_idx, test_idx = split_indices(groups)
    train_users = np.unique(groups[train_idx])
    val_users = np.unique(groups[val_idx])
    test_users = np.unique(groups[test_idx])
    log(f"train/val/test query 数: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    x_train, y_train = build_pairs([query_rows[idx] for idx in train_idx], query_matrix[train_idx], review_vectors, train_users, rng)
    x_val, y_val = build_pairs([query_rows[idx] for idx in val_idx], query_matrix[val_idx], review_vectors, val_users, rng)
    x_test, y_test = build_pairs([query_rows[idx] for idx in test_idx], query_matrix[test_idx], review_vectors, test_users, rng)

    results = {}
    for name, hidden_dims in ARCHITECTURES.items():
        log(f"开始训练 architecture={name}, hidden_dims={hidden_dims}")
        model, history, best_epoch = train_one_architecture(name, hidden_dims, x_train, y_train, x_val, y_val)
        train_eval = evaluate_scores(y_train, predict_scores(model, x_train))
        val_eval = evaluate_scores(y_val, predict_scores(model, x_val))
        test_eval = evaluate_scores(y_test, predict_scores(model, x_test))
        results[name] = {
            "hidden_dims": hidden_dims,
            "best_epoch": int(best_epoch),
            "num_epochs_run": int(len(history)),
            "history": history,
            "train_eval": train_eval,
            "val_eval": val_eval,
            "test_eval": test_eval,
        }
        log(f"{name}: test_auc={test_eval['auc']:.4f}, test_ap={test_eval['average_precision']:.4f}")

    best_name = max(results.keys(), key=lambda key: results[key]["test_eval"]["auc"])
    summary = {
        "category": CATEGORY,
        "review_sentence_file": str(REVIEW_SENTENCE_FILE),
        "query_feature_file": str(QUERY_FEATURE_FILE),
        "feature_names": feature_names,
        "model_common": {
            "input": "absolute_difference_between_query_features_and_review_user_mean_features",
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "random_seed": RANDOM_SEED,
            "negatives_per_positive": NEGATIVES_PER_POSITIVE,
        },
        "split": {
            "train_queries": int(len(train_idx)),
            "val_queries": int(len(val_idx)),
            "test_queries": int(len(test_idx)),
            "train_users": int(len(train_users)),
            "val_users": int(len(val_users)),
            "test_users": int(len(test_users)),
        },
        "architectures": results,
        "best_by_test_auc": best_name,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "best_by_test_auc": best_name,
        "test_auc_by_architecture": {name: results[name]["test_eval"]["auc"] for name in results},
        "test_ap_by_architecture": {name: results[name]["test_eval"]["average_precision"] for name in results},
        "true_score_higher_share_by_architecture": {
            name: results[name]["test_eval"]["true_score_higher_share"] for name in results
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
