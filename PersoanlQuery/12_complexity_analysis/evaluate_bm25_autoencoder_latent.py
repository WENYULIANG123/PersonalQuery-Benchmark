#!/usr/bin/env python3
"""Run a 1D autoencoder latent-score experiment for Baby_Products bm25 hit@10."""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kruskal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


REPO_ROOT = Path("/fs04/ar57/wenyu")
FEATURE_FILE = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products" / "single_query_clause_features.jsonl"
RETRIEVAL_FILE = REPO_ROOT / "result" / "personal_query" / "08_retrieval" / "Baby_Products" / "retrieval_syntax_depth_summary.json"
OUTPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
SUMMARY_FILE = OUTPUT_DIR / "bm25_autoencoder_latent_3tier_summary.json"
ROW_FILE = OUTPUT_DIR / "bm25_autoencoder_latent_3tier_records.jsonl"
TIER_LABELS = ("low", "medium", "high")
SEED = 42
TRAIN_RATIO = 0.64
VAL_RATIO = 0.16
TEST_RATIO = 0.20
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ENCODER_HIDDEN_DIM = 16
DECODER_HIDDEN_DIM = 16


class AutoEncoderLatentNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder_hidden = nn.Sequential(
            nn.Linear(input_dim, ENCODER_HIDDEN_DIM),
            nn.ReLU(),
        )
        self.encoder_latent = nn.Linear(ENCODER_HIDDEN_DIM, 1)
        self.decoder = nn.Sequential(
            nn.Linear(1, DECODER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(DECODER_HIDDEN_DIM, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder_hidden(x)
        latent = self.encoder_latent(hidden)
        reconstruction = self.decoder(latent)
        return latent.squeeze(-1), reconstruction


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bm25_rows() -> tuple[list[str], list[dict]]:
    feature_rows = [
        json.loads(line)
        for line in FEATURE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    feature_index = {(row["user_id"], row["asin"]): row for row in feature_rows}

    retrieval = json.loads(RETRIEVAL_FILE.read_text(encoding="utf-8"))
    bm25_block = None
    for item in retrieval["all_results_combined"]:
        if item.get("retriever") == "bm25" and item.get("query_category") == "syntax_depth" and item.get("query_type") == "correct":
            bm25_block = item
            break
    if bm25_block is None:
        raise ValueError("未找到 bm25 的 syntax_depth/correct 检索结果")

    merged_rows = []
    feature_names = list(feature_rows[0]["features"].keys())
    for retrieval_row in bm25_block["all_query_records"]:
        key = (retrieval_row["user_id"], retrieval_row["asin"])
        feature_row = feature_index.get(key)
        if feature_row is None:
            raise ValueError(f"bm25 检索记录缺少特征行: {key}")
        merged_rows.append({
            "user_id": retrieval_row["user_id"],
            "asin": retrieval_row["asin"],
            "hit_at10": float(retrieval_row["hit_at10"]),
            "query": feature_row["query"],
            "features": {name: float(feature_row["features"][name]) for name in feature_names},
        })
    return feature_names, merged_rows


def split_users(rows: list[dict]) -> dict[str, set[str]]:
    user_to_labels: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        user_to_labels[row["user_id"]].append(float(row["hit_at10"]))

    users = sorted(user_to_labels)
    user_targets = [int(max(user_to_labels[user_id])) for user_id in users]
    if len(set(user_targets)) != 2:
        raise ValueError("用户级标签只有一个类别，无法做分层切分")

    train_val_users, test_users = train_test_split(
        users,
        test_size=TEST_RATIO,
        random_state=SEED,
        stratify=user_targets,
    )
    remaining_targets = [int(max(user_to_labels[user_id])) for user_id in train_val_users]
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


def build_matrices(feature_names: list[str], rows: list[dict], user_splits: dict[str, set[str]]) -> tuple[dict[str, np.ndarray], dict[str, list[dict]]]:
    split_rows = {name: [] for name in user_splits}
    for row in rows:
        assigned = [split_name for split_name, users in user_splits.items() if row["user_id"] in users]
        if len(assigned) != 1:
            raise ValueError(f"用户 {row['user_id']} 未被唯一分配到一个 split")
        split_rows[assigned[0]].append(row)

    for split_name, items in split_rows.items():
        if not items:
            raise ValueError(f"{split_name} split 为空")

    train_matrix = np.array(
        [[row["features"][name] for name in feature_names] for row in split_rows["train"]],
        dtype=np.float32,
    )
    scaler = StandardScaler()
    scaler.fit(train_matrix)

    split_features = {}
    for split_name, items in split_rows.items():
        matrix = np.array(
            [[row["features"][name] for name in feature_names] for row in items],
            dtype=np.float32,
        )
        split_features[split_name] = scaler.transform(matrix).astype(np.float32)
    return split_features, split_rows


def evaluate_reconstruction_loss(model: nn.Module, features: np.ndarray, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(features).to(device)
        _, reconstruction = model(inputs)
        loss = criterion(reconstruction, inputs)
    value = float(loss.item())
    if not np.isfinite(value):
        raise ValueError("验证重构损失不是有限值")
    return value


def train_model(split_features: dict[str, np.ndarray]) -> tuple[AutoEncoderLatentNet, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoderLatentNet(input_dim=split_features["train"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    train_inputs = torch.from_numpy(split_features["train"]).to(device)

    best_state = None
    best_val_loss = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        _, reconstruction = model(train_inputs)
        loss = criterion(reconstruction, train_inputs)
        if not torch.isfinite(loss):
            raise ValueError(f"epoch {epoch} 出现非有限训练重构损失")
        loss.backward()
        optimizer.step()

        train_loss = float(loss.item())
        val_loss = evaluate_reconstruction_loss(model, split_features["val"], criterion, device)
        history.append({
            "epoch": epoch,
            "train_reconstruction_loss": train_loss,
            "val_reconstruction_loss": val_loss,
        })
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(json.dumps({
                "epoch": epoch,
                "train_reconstruction_loss": train_loss,
                "val_reconstruction_loss": val_loss,
            }, ensure_ascii=False))
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise ValueError("训练结束后没有保存到最优 autoencoder 模型")
    model.load_state_dict(best_state)
    return model, history


def extract_latent_scores(model: AutoEncoderLatentNet, split_features: dict[str, np.ndarray], split_rows: dict[str, list[dict]]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    device = next(model.parameters()).device
    score_outputs = {}
    for split_name, features in split_features.items():
        model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(features).to(device)
            latent, _ = model(inputs)
            score_outputs[split_name] = latent.cpu().numpy()

    train_hits = np.array([float(row["hit_at10"]) for row in split_rows["train"]], dtype=float)
    corr = np.corrcoef(score_outputs["train"], train_hits)[0, 1]
    if np.isnan(corr):
        raise ValueError("训练集 latent score 与 hit@10 相关性为 NaN")
    if corr < 0:
        for split_name in score_outputs:
            score_outputs[split_name] = -score_outputs[split_name]
        corr = -corr

    metrics = {
        "train_score_hit10_corr": float(corr),
    }
    return score_outputs, metrics


def compute_train_thresholds(train_scores: np.ndarray) -> tuple[float, float]:
    low_boundary = float(np.quantile(train_scores, 1.0 / 3.0))
    high_boundary = float(np.quantile(train_scores, 2.0 / 3.0))
    if low_boundary >= high_boundary:
        raise ValueError("训练集三分位边界无效")
    return low_boundary, high_boundary


def assign_tier(score: float, thresholds: tuple[float, float]) -> str:
    low_boundary, high_boundary = thresholds
    if score <= low_boundary:
        return "low"
    if score <= high_boundary:
        return "medium"
    return "high"


def summarize_hits(values: list[float]) -> dict:
    if not values:
        raise ValueError("存在空 tier，无法汇总 hit@10")
    array = np.array(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def build_test_summary(split_rows: dict[str, list[dict]], score_outputs: dict[str, np.ndarray], thresholds: tuple[float, float]) -> tuple[list[dict], dict]:
    test_rows = split_rows["test"]
    test_scores = score_outputs["test"]
    if len(test_rows) != len(test_scores):
        raise ValueError("测试集行数与 score 数量不一致")

    grouped_hits = {label: [] for label in TIER_LABELS}
    output_rows = []
    for row, score in zip(test_rows, test_scores):
        tier = assign_tier(float(score), thresholds)
        hit = float(row["hit_at10"])
        grouped_hits[tier].append(hit)
        output_rows.append({
            "user_id": row["user_id"],
            "asin": row["asin"],
            "query": row["query"],
            "hit_at10": hit,
            "latent_score": float(score),
            "tier": tier,
        })

    for label, values in grouped_hits.items():
        if not values:
            raise ValueError(f"测试集的 {label} tier 为空")

    statistic, p_value = kruskal(*[grouped_hits[label] for label in TIER_LABELS])
    summary = {
        "matched_count": len(output_rows),
        "tier_counts": {label: len(grouped_hits[label]) for label in TIER_LABELS},
        "tier_hit10": {label: summarize_hits(grouped_hits[label]) for label in TIER_LABELS},
        "kruskal_wallis": {
            "statistic": float(statistic),
            "p_value": float(p_value),
        },
    }
    return output_rows, summary


def main() -> None:
    set_seed(SEED)
    feature_names, rows = load_bm25_rows()
    user_splits = split_users(rows)
    split_features, split_rows = build_matrices(feature_names, rows, user_splits)
    model, history = train_model(split_features)
    score_outputs, model_metrics = extract_latent_scores(model, split_features, split_rows)
    thresholds = compute_train_thresholds(score_outputs["train"])
    output_rows, test_summary = build_test_summary(split_rows, score_outputs, thresholds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ROW_FILE.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "feature_file": str(FEATURE_FILE),
        "retrieval_file": str(RETRIEVAL_FILE),
        "retriever": "bm25",
        "feature_names": feature_names,
        "num_rows": len(rows),
        "split_sizes": {split_name: len(items) for split_name, items in split_rows.items()},
        "unique_user_counts": {
            split_name: len(user_splits[split_name])
            for split_name in ("train", "val", "test")
        },
        "train_thresholds": {
            "low_boundary": thresholds[0],
            "high_boundary": thresholds[1],
        },
        "model_metrics": model_metrics,
        "training_history_tail": history[-10:],
        "test_summary": test_summary,
        "row_file": str(ROW_FILE),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "row_file": str(ROW_FILE),
        "split_sizes": summary["split_sizes"],
        "unique_user_counts": summary["unique_user_counts"],
        "model_metrics": model_metrics,
        "test_summary": test_summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
