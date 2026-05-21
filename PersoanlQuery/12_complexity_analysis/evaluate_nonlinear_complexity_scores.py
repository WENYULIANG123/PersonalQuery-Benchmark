#!/usr/bin/env python3
"""Evaluate nonlinear 1D complexity scores against retrieval metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kruskal, pearsonr
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
QUERY_FEATURE_FILE = INPUT_DIR / "single_query_clause_features.jsonl"
RETRIEVAL_FILE = REPO_ROOT / "result" / "personal_query" / "08_retrieval" / CATEGORY / "retrieval_syntax_depth_summary.json"
SUMMARY_FILE = INPUT_DIR / "nonlinear_complexity_score_retrieval_summary.json"
RECORD_FILE = INPUT_DIR / "nonlinear_complexity_score_retrieval_records.jsonl"

RANDOM_SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 300
PATIENCE = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

METRIC_COLUMNS = ["hit_at10", "n_at10", "mrr_at10"]
TIER_LABELS = ["low", "medium", "high"]


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        encoder_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            current_dim = hidden_dim
        encoder_layers.append(nn.Linear(current_dim, 1))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        current_dim = 1
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            current_dim = hidden_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z.squeeze(-1)


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


def query_key(user_id: str, asin: str) -> tuple[str, str]:
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id 必须是非空字符串")
    if not isinstance(asin, str) or not asin:
        raise ValueError("asin 必须是非空字符串")
    return user_id, asin


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


def orient_score(scores: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(scores, anchor)[0, 1]
    if np.isnan(corr):
        raise ValueError("score 与 anchor 的相关性为 NaN")
    return scores if corr >= 0 else -scores


def linear_pca_score(standardized: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    score = PCA(n_components=1, random_state=RANDOM_SEED).fit_transform(standardized).ravel()
    return orient_score(score, anchor)


def kernel_pca_scores(standardized: np.ndarray, anchor: np.ndarray) -> dict[str, np.ndarray]:
    configs = {
        "kernel_pca_rbf_gamma_0_01": {"kernel": "rbf", "gamma": 0.01},
        "kernel_pca_rbf_gamma_0_05": {"kernel": "rbf", "gamma": 0.05},
        "kernel_pca_rbf_gamma_0_10": {"kernel": "rbf", "gamma": 0.10},
        "kernel_pca_poly_degree_2": {"kernel": "poly", "degree": 2, "coef0": 1.0},
    }
    output = {}
    for name, params in configs.items():
        log(f"开始计算 {name}")
        model = KernelPCA(n_components=1, random_state=RANDOM_SEED, **params)
        score = model.fit_transform(standardized).ravel()
        output[name] = orient_score(score, anchor)
    return output


def make_loader(matrix: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(matrix.astype(np.float32)))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_autoencoder_score(name: str, standardized: np.ndarray, anchor: np.ndarray, hidden_dims: list[int]) -> tuple[np.ndarray, dict]:
    log(f"开始训练 {name}, hidden_dims={hidden_dims}")
    set_seed(RANDOM_SEED + len(hidden_dims))
    model = AutoEncoder(input_dim=standardized.shape[1], hidden_dims=hidden_dims)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    loader = make_loader(standardized, shuffle=True)

    best_state = None
    best_loss = np.inf
    best_epoch = 0
    patience_left = PATIENCE
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for (batch_x,) in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "reconstruction_loss": epoch_loss})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise ValueError(f"{name} 未产生 best_state")
    model.load_state_dict(best_state)
    model.eval()
    scores = []
    with torch.no_grad():
        for (batch_x,) in make_loader(standardized, shuffle=False):
            _, z = model(batch_x)
            scores.append(z.cpu().numpy())
    score = np.concatenate(scores)
    oriented = orient_score(score, anchor)
    return oriented, {
        "hidden_dims": hidden_dims,
        "best_epoch": int(best_epoch),
        "best_reconstruction_loss": float(best_loss),
        "num_epochs_run": int(len(history)),
        "history_tail": history[-10:],
    }


def autoencoder_scores(standardized: np.ndarray, anchor: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    configs = {
        "autoencoder_shallow": [32, 16],
        "autoencoder_medium": [64, 32, 16],
        "autoencoder_deep": [128, 64, 32, 16],
    }
    scores = {}
    metadata = {}
    for name, hidden_dims in configs.items():
        score, meta = train_autoencoder_score(name, standardized, anchor, hidden_dims)
        scores[name] = score
        metadata[name] = meta
    return scores, metadata


def load_retrieval_records(query_score_rows: list[dict]) -> list[dict]:
    query_index = {query_key(row["user_id"], row["asin"]): row for row in query_score_rows}
    if len(query_index) != len(query_score_rows):
        raise ValueError("query score rows 存在重复 key")
    payload = json.loads(RETRIEVAL_FILE.read_text(encoding="utf-8"))
    combined = payload.get("all_results_combined")
    if not isinstance(combined, list) or not combined:
        raise ValueError("retrieval summary 缺少 all_results_combined")
    records = []
    missing = set()
    for retriever_result in combined:
        retriever = retriever_result.get("retriever")
        query_records = retriever_result.get("all_query_records")
        if not isinstance(retriever, str) or not retriever:
            raise ValueError("retriever 为空")
        if not isinstance(query_records, list) or not query_records:
            raise ValueError(f"{retriever} 缺少 all_query_records")
        for row in query_records:
            key = query_key(row["user_id"], row["asin"])
            score_row = query_index.get(key)
            if score_row is None:
                missing.add(key)
                continue
            base = {
                "domain": CATEGORY,
                "retriever": retriever,
                "user_id": row["user_id"],
                "asin": row["asin"],
                "query_length": float(row["query_length"]),
                "avg_idf": float(row["mean_idf"]),
                "hit_at10": float(row["hit_at10"]),
                "n_at10": float(row["n_at10"]),
                "mrr_at10": float(row["mrr_at10"]),
            }
            for key_name, value in score_row["scores"].items():
                base[key_name] = float(value)
            records.append(base)
    if missing:
        raise ValueError(f"retrieval 中有 {len(missing)} 个 query key 缺少 score")
    if not records:
        raise ValueError("没有成功对齐 retrieval records")
    return records


def assign_tiers(unique_query_rows: list[dict], score_name: str) -> dict[tuple[str, str], str]:
    values = np.array([row["scores"][score_name] for row in unique_query_rows], dtype=float)
    q33 = float(np.quantile(values, 1.0 / 3.0))
    q67 = float(np.quantile(values, 2.0 / 3.0))
    if not q33 < q67:
        raise ValueError(f"{score_name} 三分位边界异常")
    tier_by_key = {}
    for row in unique_query_rows:
        score = float(row["scores"][score_name])
        if score <= q33:
            tier = "low"
        elif score <= q67:
            tier = "medium"
        else:
            tier = "high"
        tier_by_key[query_key(row["user_id"], row["asin"])] = tier
    return tier_by_key


def metric_tests(records: list[dict], score_name: str, tier_by_key: dict[tuple[str, str], str]) -> dict:
    scoped = []
    for row in records:
        enriched = dict(row)
        enriched["tier"] = tier_by_key[query_key(row["user_id"], row["asin"])]
        scoped.append(enriched)

    output = {}
    for scope in ["pooled"] + sorted({row["retriever"] for row in scoped}):
        scope_rows = scoped if scope == "pooled" else [row for row in scoped if row["retriever"] == scope]
        grouped = {label: [row for row in scope_rows if row["tier"] == label] for label in TIER_LABELS}
        if any(len(rows) == 0 for rows in grouped.values()):
            raise ValueError(f"{score_name} {scope} 出现空 tier")
        metric_output = {}
        for metric in METRIC_COLUMNS:
            samples = [[float(row[metric]) for row in grouped[label]] for label in TIER_LABELS]
            stat, p_value = kruskal(*samples)
            means = {label: float(np.mean(samples[idx])) for idx, label in enumerate(TIER_LABELS)}
            metric_output[metric] = {
                "kruskal_statistic": float(stat),
                "kruskal_p_value": float(p_value),
                "means": means,
                "range": float(max(means.values()) - min(means.values())),
            }
        output[scope] = {
            "counts": {label: len(grouped[label]) for label in TIER_LABELS},
            "metric_tests": metric_output,
        }
    return output


def main() -> None:
    set_seed(RANDOM_SEED)
    log("开始读取 query 20 特征")
    query_rows = load_jsonl(QUERY_FEATURE_FILE)
    feature_names = feature_names_from_rows(query_rows)
    matrix = feature_matrix(query_rows, feature_names)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix).astype(np.float32)
    anchor = matrix[:, feature_names.index("max_dependency_depth")]

    scores = {"linear_pca": linear_pca_score(standardized, anchor)}
    scores.update(kernel_pca_scores(standardized, anchor))
    ae_scores, ae_metadata = autoencoder_scores(standardized, anchor)
    scores.update(ae_scores)

    query_score_rows = []
    for idx, row in enumerate(query_rows):
        query_score_rows.append({
            "user_id": row["user_id"],
            "asin": row["asin"],
            "query": row["query"],
            "scores": {name: float(score[idx]) for name, score in scores.items()},
        })

    log("开始对齐 retrieval 指标")
    records = load_retrieval_records(query_score_rows)
    log(f"对齐 retrieval 记录数: {len(records)}")

    score_summaries = {}
    tier_results = {}
    for score_name, score_array in scores.items():
        tier_by_key = assign_tiers(query_score_rows, score_name)
        tests = metric_tests(records, score_name, tier_by_key)
        corr = pearsonr(score_array, anchor)
        if np.isnan(corr.statistic) or np.isnan(corr.pvalue):
            raise ValueError(f"{score_name} 与 anchor 的 Pearson 结果为 NaN")
        score_summaries[score_name] = {
            "score_summary": summarize_array(score_array),
            "anchor_feature": "max_dependency_depth",
            "anchor_pearson_r": float(corr.statistic),
            "anchor_pearson_p_value": float(corr.pvalue),
        }
        tier_results[score_name] = tests

    ranking = sorted(
        scores.keys(),
        key=lambda name: (
            tier_results[name]["pooled"]["metric_tests"]["hit_at10"]["range"],
            -tier_results[name]["pooled"]["metric_tests"]["hit_at10"]["kruskal_p_value"],
        ),
        reverse=True,
    )

    summary = {
        "category": CATEGORY,
        "query_feature_file": str(QUERY_FEATURE_FILE),
        "retrieval_file": str(RETRIEVAL_FILE),
        "record_file": str(RECORD_FILE),
        "feature_names": feature_names,
        "num_query_rows": len(query_rows),
        "num_retrieval_records": len(records),
        "autoencoder_metadata": ae_metadata,
        "score_summaries": score_summaries,
        "tier_results": tier_results,
        "ranking_by_pooled_hit_at10_range": ranking,
    }
    with RECORD_FILE.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "ranking_by_pooled_hit_at10_range": ranking,
        "pooled_hit_at10": {
            name: tier_results[name]["pooled"]["metric_tests"]["hit_at10"] for name in ranking
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
