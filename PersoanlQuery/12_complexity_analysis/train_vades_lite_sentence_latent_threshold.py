#!/usr/bin/env python3
"""Train a sentence-level review-only VADES model with explicit user distributions and range-aware query ranking."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent

from cluster_strict5550_query_gmm_and_attach_retrieval import run_query_gmm_pipeline  # noqa: E402
from extract_clause_features_single_query import extract_clause_features_from_doc, load_spacy_model  # noqa: E402


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = os.environ.get("PQ_CATEGORY", "Baby_Products")
INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
REVIEW_SOURCE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / CATEGORY / "stage1_filtered_users_reviews.json"
)
CANDIDATE_QUERY_FILE = INPUT_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
RAW_CANDIDATE_QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / "query_by_syntax_depth_no_depth_check_10.json"

SEED = 42
DEVICE: torch.device | None = None
TRAIN_SENTENCES_PER_USER = 10
HOLDOUT_SENTENCES_PER_USER = 10
TOTAL_SENTENCES_PER_USER = TRAIN_SENTENCES_PER_USER + HOLDOUT_SENTENCES_PER_USER
OUTPUT_TAG = os.environ.get("VADES_OUTPUT_TAG", "vades_lite_sentence_user_distribution_train10_holdout10")
LATENT_DIM = int(os.environ.get("VADES_LATENT_DIM", "20"))
HIDDEN_DIM = int(os.environ.get("VADES_HIDDEN_DIM", "64"))
EPOCHS = int(os.environ.get("VADES_EPOCHS", "100"))
BATCH_SIZE = int(os.environ.get("VADES_BATCH_SIZE", "128"))
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
USER_MATCH_WEIGHT = 1.0
STYLE_RECON_WEIGHT = 0.8
SENT_KL_WEIGHT = 0.05
USER_PRIOR_KL_WEIGHT = 0.02
ABS_THRESHOLD_QUANTILE = float(os.environ.get("VADES_ABS_THRESHOLD_QUANTILE", "0.95"))
MAX_USERS_OVERRIDE = os.environ.get("VADES_MAX_USERS")
SKIP_POST_CLUSTERING = os.environ.get("VADES_SKIP_POST_CLUSTERING", "0") == "1"

SUMMARY_FILE = INPUT_DIR / f"{OUTPUT_TAG}_summary.json"
DETAIL_FILE = INPUT_DIR / f"{OUTPUT_TAG}_epoch_details.jsonl"
USER_PROFILE_FILE = INPUT_DIR / f"{OUTPUT_TAG}_user_profiles.jsonl"
SENTENCE_FILE = INPUT_DIR / f"{OUTPUT_TAG}_sentences.jsonl"
EXCLUDED_USER_FILE = INPUT_DIR / f"{OUTPUT_TAG}_excluded_users.jsonl"
SELECTED_RECORD_FILE = INPUT_DIR / f"{OUTPUT_TAG}_selected_query_records.jsonl"
REJECTED_RECORD_FILE = INPUT_DIR / f"{OUTPUT_TAG}_rejected_query_records.jsonl"
QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / f"query_by_syntax_depth_{OUTPUT_TAG}.json"


def log(message: str) -> None:
    print(message, flush=True)


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


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


def diagonal_gaussian_kl(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * (
        logvar_p
        - logvar_q
        + (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
    ).sum(dim=-1)


def standard_normal_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar).sum(dim=-1)


class SentenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        reconstruction = self.decoder(mu)
        return mu, logvar, reconstruction


class UserDistributionTable(nn.Module):
    def __init__(self, num_users: int, latent_dim: int):
        super().__init__()
        self.user_mu = nn.Embedding(num_users, latent_dim)
        self.user_logvar = nn.Embedding(num_users, latent_dim)
        nn.init.zeros_(self.user_mu.weight)
        nn.init.zeros_(self.user_logvar.weight)

    def forward(self, user_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_mu(user_index), self.user_logvar(user_index)


def normalize_text(text: str) -> str:
    return re.sub(r"\\s+", " ", text.strip())


def extract_sentences_from_review_text(text: str) -> list[str]:
    nlp = load_spacy_model()
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        normalized = normalize_text(sent.text)
        if normalized:
            sentences.append(normalized)
    return sentences


def load_filtered_user_reviews() -> list[dict]:
    if not REVIEW_SOURCE_FILE.exists():
        raise FileNotFoundError(f"缺少用户评论文件: {REVIEW_SOURCE_FILE}")
    payload = load_json(REVIEW_SOURCE_FILE)
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"{REVIEW_SOURCE_FILE} 必须是非空列表")
        return payload
    if isinstance(payload, dict):
        users = payload.get("users")
        if not isinstance(users, list) or not users:
            raise ValueError(f"{REVIEW_SOURCE_FILE} 的 users 字段必须是非空列表")
        return users
    raise ValueError(f"{REVIEW_SOURCE_FILE} 必须是列表或包含 users 的字典")


def extract_first_twenty_sentences_for_users(user_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    log("开始为用户抽取前 20 个句子")
    kept_rows: list[dict] = []
    excluded_rows: list[dict] = []
    total_users = len(user_rows)
    for user_offset, user_row in enumerate(user_rows, start=1):
        user_id = user_row.get("user_id")
        reviews = user_row.get("reviews")
        if reviews is None:
            reviews = user_row.get("results")
        if user_id is None or not isinstance(reviews, list):
            raise ValueError(f"用户记录缺少 user_id 或 reviews/results: index={user_offset}")
        collected: list[dict] = []
        for review_index, review_row in enumerate(reviews):
            review_texts = review_row.get("target_reviews")
            if review_texts is not None:
                if not isinstance(review_texts, list):
                    raise ValueError(f"target_reviews 必须是列表: user_id={user_id}, review_index={review_index}")
                candidate_texts = [text for text in review_texts if text]
            else:
                single_review_text = review_row.get("text")
                candidate_texts = [single_review_text] if single_review_text else []
            if not candidate_texts:
                continue
            for source_text in candidate_texts:
                for sentence_index, sentence_text in enumerate(extract_sentences_from_review_text(source_text)):
                    collected.append(
                        {
                            "user_id": user_id,
                            "review_index": review_index,
                            "sentence_index": sentence_index,
                            "sentence_text": sentence_text,
                            "word_count": len(sentence_text.split()),
                        }
                    )
                    if len(collected) == TOTAL_SENTENCES_PER_USER:
                        break
                if len(collected) == TOTAL_SENTENCES_PER_USER:
                    break
            if len(collected) == TOTAL_SENTENCES_PER_USER:
                break

        if len(collected) < TOTAL_SENTENCES_PER_USER:
            log(f"用户 {user_offset}/{total_users}: {user_id}, 句子数不足 20, 实际={len(collected)}，从本方法中过滤")
            excluded_rows.append(
                {
                    "user_id": user_id,
                    "available_sentence_count": int(len(collected)),
                    "required_sentence_count": TOTAL_SENTENCES_PER_USER,
                }
            )
            continue

        log(f"已完成用户 {user_offset}/{total_users}: {user_id}, 句子数={len(collected)}")
        kept_rows.extend(collected)
    return kept_rows, excluded_rows


def build_sentence_feature_rows(sentence_rows: list[dict]) -> tuple[list[dict], list[str]]:
    log("开始提取评论句法特征")
    nlp = load_spacy_model()
    feature_names: list[str] | None = None
    enriched_rows: list[dict] = []
    for row in sentence_rows:
        doc = nlp(row["sentence_text"])
        extracted = extract_clause_features_from_doc(doc, row["sentence_text"])
        if feature_names is None:
            feature_names = list(extracted.keys())
        elif list(extracted.keys()) != feature_names:
            raise ValueError(f"评论句法特征字段顺序不一致: user_id={row['user_id']}")
        enriched = dict(row)
        enriched["features"] = extracted
        enriched_rows.append(enriched)
    if feature_names is None:
        raise ValueError("没有可用评论句法特征")
    return enriched_rows, feature_names


def load_candidate_query_rows() -> list[dict]:
    if CANDIDATE_QUERY_FILE.exists():
        log(f"开始读取候选 query 特征文件: {CANDIDATE_QUERY_FILE}")
        rows = []
        with CANDIDATE_QUERY_FILE.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"{CANDIDATE_QUERY_FILE} 为空")
        return rows

    if not RAW_CANDIDATE_QUERY_FILE.exists():
        raise FileNotFoundError(f"缺少原始 10 候选 query 文件: {RAW_CANDIDATE_QUERY_FILE}")
    log(f"未发现候选特征文件，开始从原始 10 候选 query 生成: {RAW_CANDIDATE_QUERY_FILE}")
    raw_rows = load_json(RAW_CANDIDATE_QUERY_FILE)
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{RAW_CANDIDATE_QUERY_FILE} 必须是非空列表")
    rows = build_candidate_feature_rows_from_raw_query_file(raw_rows)
    write_jsonl(CANDIDATE_QUERY_FILE, rows)
    log(f"候选特征文件已写入: {CANDIDATE_QUERY_FILE}")
    return rows


def build_candidate_feature_rows_from_raw_query_file(raw_rows: list[dict]) -> list[dict]:
    nlp = load_spacy_model()
    total_users = len(raw_rows)
    rows: list[dict] = []
    for user_offset, row in enumerate(raw_rows, start=1):
        user_id = row.get("user_id")
        asin = row.get("asin")
        candidates = row.get("syntax_depth_queries")
        if user_id is None or asin is None or not isinstance(candidates, list):
            raise ValueError(f"原始 10 候选 query 记录缺少 user_id / asin / syntax_depth_queries: index={user_offset}")
        for candidate_index, candidate in enumerate(candidates, start=1):
            query_text = candidate.get("query")
            if not query_text:
                raise ValueError(f"候选 query 缺少 query 文本: user_id={user_id}, candidate_index={candidate_index}")
            doc = nlp(query_text)
            extracted = extract_clause_features_from_doc(doc, query_text)
            rows.append(
                {
                    "user_id": user_id,
                    "asin": asin,
                    "candidate_index": candidate_index,
                    "query": query_text,
                    "word_count": int(candidate.get("word_count", len(query_text.split()))),
                    "target_depth": candidate.get("target_depth"),
                    "user_avg_depth": candidate.get("user_avg_depth"),
                    "attrs_used": candidate.get("attrs_used"),
                    "features": extracted,
                }
            )
        log(f"已完成用户 {user_offset}/{total_users}: {user_id}, 10 候选已处理")
    return rows


def build_training_dataset(sentence_rows: list[dict], feature_names: list[str]) -> tuple[list[str], dict]:
    feature_matrix = np.asarray(
        [[float(row["features"][name]) for name in feature_names] for row in sentence_rows],
        dtype=np.float64,
    )
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in sentence_rows:
        grouped_rows[row["user_id"]].append(row)
    user_ids = sorted(grouped_rows.keys())
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}

    user_indices: list[int] = []
    train_mask: list[bool] = []
    holdout_mask: list[bool] = []
    for row in sentence_rows:
        user_rows = grouped_rows[row["user_id"]]
        per_user_offset = user_rows.index(row)
        user_indices.append(user_to_index[row["user_id"]])
        train_mask.append(per_user_offset < TRAIN_SENTENCES_PER_USER)
        holdout_mask.append(per_user_offset >= TRAIN_SENTENCES_PER_USER)

    dataset = {
        "scaler": scaler,
        "feature_names": feature_names,
        "sentence_rows": sentence_rows,
        "scaled_features": scaled_features,
        "feature_matrix": feature_matrix,
        "user_indices": np.asarray(user_indices, dtype=np.int64),
        "train_mask": np.asarray(train_mask, dtype=bool),
        "holdout_mask": np.asarray(holdout_mask, dtype=bool),
        "user_to_index": user_to_index,
        "grouped_rows": grouped_rows,
    }
    return user_ids, dataset


def train_vades_user_distribution_model(user_ids: list[str], dataset: dict, feature_names: list[str]) -> tuple[SentenceEncoder, UserDistributionTable, dict]:
    input_dim = len(feature_names)
    encoder = SentenceEncoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    user_table = UserDistributionTable(num_users=len(user_ids), latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(user_table.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    features_tensor = torch.tensor(dataset["scaled_features"], dtype=torch.float32, device=DEVICE)
    user_index_tensor = torch.tensor(dataset["user_indices"], dtype=torch.long, device=DEVICE)
    train_indices = np.flatnonzero(dataset["train_mask"])
    train_indices_tensor = torch.tensor(train_indices, dtype=torch.long, device=DEVICE)

    epoch_rows: list[dict] = []
    encoder.train()
    user_table.train()

    for epoch in range(1, EPOCHS + 1):
        shuffled_indices = train_indices.copy()
        np.random.shuffle(shuffled_indices)
        batch_metrics = []
        for start in range(0, len(shuffled_indices), BATCH_SIZE):
            batch_indices = shuffled_indices[start : start + BATCH_SIZE]
            batch_idx_tensor = torch.tensor(batch_indices, dtype=torch.long, device=DEVICE)
            batch_features = features_tensor[batch_idx_tensor]
            batch_user_indices = user_index_tensor[batch_idx_tensor]

            sent_mu, sent_logvar, reconstruction = encoder(batch_features)
            user_mu_all = user_table.user_mu.weight
            user_logvar_all = user_table.user_logvar.weight
            kl_matrix = []
            for sent_offset in range(sent_mu.shape[0]):
                sent_mu_row = sent_mu[sent_offset].unsqueeze(0).expand_as(user_mu_all)
                sent_logvar_row = sent_logvar[sent_offset].unsqueeze(0).expand_as(user_logvar_all)
                kl_row = diagonal_gaussian_kl(sent_mu_row, sent_logvar_row, user_mu_all, user_logvar_all)
                kl_matrix.append(kl_row)
            kl_matrix_tensor = torch.stack(kl_matrix, dim=0)
            user_match_loss = F.cross_entropy(-kl_matrix_tensor, batch_user_indices)

            style_recon_loss = F.mse_loss(reconstruction, batch_features)
            sent_kl_loss = standard_normal_kl(sent_mu, sent_logvar).mean()
            batch_user_mu, batch_user_logvar = user_table(batch_user_indices)
            user_prior_kl_loss = standard_normal_kl(batch_user_mu, batch_user_logvar).mean()

            total_loss = (
                USER_MATCH_WEIGHT * user_match_loss
                + STYLE_RECON_WEIGHT * style_recon_loss
                + SENT_KL_WEIGHT * sent_kl_loss
                + USER_PRIOR_KL_WEIGHT * user_prior_kl_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_metrics.append(
                {
                    "total_loss": float(total_loss.item()),
                    "user_match_loss": float(user_match_loss.item()),
                    "style_recon_loss": float(style_recon_loss.item()),
                    "sent_kl_loss": float(sent_kl_loss.item()),
                    "user_prior_kl_loss": float(user_prior_kl_loss.item()),
                }
            )

        epoch_summary = {
            "epoch": epoch,
            "batch_count": int(len(batch_metrics)),
            "total_loss_mean": float(np.mean([row["total_loss"] for row in batch_metrics])),
            "user_match_loss_mean": float(np.mean([row["user_match_loss"] for row in batch_metrics])),
            "style_recon_loss_mean": float(np.mean([row["style_recon_loss"] for row in batch_metrics])),
            "sent_kl_loss_mean": float(np.mean([row["sent_kl_loss"] for row in batch_metrics])),
            "user_prior_kl_loss_mean": float(np.mean([row["user_prior_kl_loss"] for row in batch_metrics])),
        }
        epoch_rows.append(epoch_summary)
    return encoder, user_table, {"epochs": epoch_rows}


def infer_user_sentence_distributions(
    encoder: SentenceEncoder,
    user_table: UserDistributionTable,
    user_ids: list[str],
    dataset: dict,
) -> tuple[list[dict], list[dict]]:
    feature_names = dataset["feature_names"]
    sentence_rows = dataset["sentence_rows"]
    feature_tensor = torch.tensor(dataset["scaled_features"], dtype=torch.float32, device=DEVICE)
    encoder.eval()
    user_table.eval()
    with torch.no_grad():
        sent_mu, sent_logvar, _ = encoder(feature_tensor)
        user_mu_weight = user_table.user_mu.weight.detach().cpu().numpy()
        user_logvar_weight = user_table.user_logvar.weight.detach().cpu().numpy()

    sentence_output_rows: list[dict] = []
    user_scores: dict[str, list[float]] = defaultdict(list)
    user_profile_rows: list[dict] = []
    user_mu_weight_tensor = user_table.user_mu.weight.detach()
    user_logvar_weight_tensor = user_table.user_logvar.weight.detach()

    for idx, row in enumerate(sentence_rows):
        user_id = row["user_id"]
        user_index = dataset["user_to_index"][user_id]
        sentence_output_rows.append(
            {
                "user_id": user_id,
                "review_index": row["review_index"],
                "sentence_index": row["sentence_index"],
                "sentence_text": row["sentence_text"],
                "word_count": row["word_count"],
                "features": row["features"],
            }
        )
        score = diagonal_gaussian_kl(
            sent_mu[idx].unsqueeze(0),
            sent_logvar[idx].unsqueeze(0),
            user_mu_weight_tensor[user_index].unsqueeze(0),
            user_logvar_weight_tensor[user_index].unsqueeze(0),
        ).item()
        user_scores[user_id].append(float(score))

    for user_id in user_ids:
        user_index = dataset["user_to_index"][user_id]
        score_array = np.asarray(user_scores[user_id], dtype=np.float64)
        user_profile_rows.append(
            {
                "user_id": user_id,
                "user_mu": user_mu_weight[user_index].tolist(),
                "user_logvar": user_logvar_weight[user_index].tolist(),
                "user_var": np.exp(user_logvar_weight[user_index]).tolist(),
                "kl_score_summary": summarize_array(score_array),
            }
        )
    return sentence_output_rows, user_profile_rows


def calibrate_absolute_threshold_with_unseen_holdout(
    encoder: SentenceEncoder,
    user_table: UserDistributionTable,
    user_ids: list[str],
    dataset: dict,
) -> dict[str, float]:
    feature_tensor = torch.tensor(dataset["scaled_features"], dtype=torch.float32, device=DEVICE)
    encoder.eval()
    user_table.eval()
    thresholds: dict[str, float] = {}
    holdout_indices = np.flatnonzero(dataset["holdout_mask"])
    with torch.no_grad():
        sent_mu, sent_logvar, _ = encoder(feature_tensor)
        user_mu_weight = user_table.user_mu.weight
        user_logvar_weight = user_table.user_logvar.weight
        grouped_holdout_scores: dict[str, list[float]] = defaultdict(list)
        for idx in holdout_indices:
            user_id = dataset["sentence_rows"][idx]["user_id"]
            user_index = dataset["user_to_index"][user_id]
            score = diagonal_gaussian_kl(
                sent_mu[idx].unsqueeze(0),
                sent_logvar[idx].unsqueeze(0),
                user_mu_weight[user_index].unsqueeze(0),
                user_logvar_weight[user_index].unsqueeze(0),
            ).item()
            grouped_holdout_scores[user_id].append(float(score))
    for user_id in user_ids:
        values = grouped_holdout_scores.get(user_id)
        if values is None:
            raise ValueError(f"用户 {user_id} 缺少 holdout 句子得分")
        thresholds[user_id] = float(np.quantile(np.asarray(values, dtype=np.float64), ABS_THRESHOLD_QUANTILE))
    return thresholds


def rank_and_select_queries(
    encoder: SentenceEncoder,
    user_table: UserDistributionTable,
    dataset: dict,
    candidate_rows: list[dict],
    user_ids: list[str],
    user_profile_rows: list[dict],
    abs_thresholds: dict[str, float],
) -> tuple[list[dict], list[dict], list[dict]]:
    feature_names = dataset["feature_names"]
    feature_scaler: StandardScaler = dataset["scaler"]

    grouped_candidates: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        grouped_candidates[row["user_id"]].append(row)

    user_profile_by_id = {row["user_id"]: row for row in user_profile_rows}
    selected_rows: list[dict] = []
    rejected_rows: list[dict] = []
    query_output_rows: list[dict] = []

    encoder.eval()
    user_table.eval()
    with torch.no_grad():
        for user_id in user_ids:
            candidates = grouped_candidates.get(user_id)
            if not candidates:
                raise ValueError(f"用户 {user_id} 没有候选 query")
            user_profile = user_profile_by_id.get(user_id)
            if user_profile is None:
                raise ValueError(f"用户 {user_id} 缺少 user profile")
            user_index = dataset["user_to_index"][user_id]
            user_mu, user_logvar = user_table(torch.tensor([user_index], dtype=torch.long, device=DEVICE))

            candidate_records: list[dict] = []
            for candidate in candidates:
                feature_vector = np.asarray(
                    [float(candidate["features"][name]) for name in feature_names],
                    dtype=np.float64,
                ).reshape(1, -1)
                scaled_vector = feature_scaler.transform(feature_vector)
                feature_tensor = torch.tensor(scaled_vector, dtype=torch.float32, device=DEVICE)
                query_mu, query_logvar, _ = encoder(feature_tensor)
                range_score = diagonal_gaussian_kl(query_mu, query_logvar, user_mu, user_logvar).item()
                passes_abs_threshold = range_score <= abs_thresholds[user_id]
                candidate_records.append(
                    {
                        **candidate,
                        "query_mu": query_mu.squeeze(0).detach().cpu().numpy().tolist(),
                        "query_logvar": query_logvar.squeeze(0).detach().cpu().numpy().tolist(),
                        "range_score": float(range_score),
                        "passes_abs_threshold": bool(passes_abs_threshold),
                    }
                )

            passed = [row for row in candidate_records if row["passes_abs_threshold"]]
            if passed:
                best = min(passed, key=lambda row: row["range_score"])
                selected_rows.append(best)
                query_output_rows.append(
                    {
                        "user_id": best["user_id"],
                        "asin": best["asin"],
                        "syntax_depth_query": {
                            "query": best["query"],
                            "word_count": int(best["word_count"]),
                            "target_depth": best["target_depth"],
                            "actual_depth": None,
                            "user_avg_depth": best["user_avg_depth"],
                            "attrs_used": best["attrs_used"],
                            "accepted_candidate_index": int(best["candidate_index"]),
                            "candidate_count": int(len(candidate_records)),
                        },
                    }
                )
            else:
                best = min(candidate_records, key=lambda row: row["range_score"])
                rejected_rows.append(best)
    return selected_rows, rejected_rows, query_output_rows


def build_summary(
    user_ids: list[str],
    sentence_rows: list[dict],
    excluded_rows: list[dict],
    feature_names: list[str],
    training_info: dict,
    user_profile_rows: list[dict],
    selected_rows: list[dict],
    rejected_rows: list[dict],
    query_output_rows: list[dict],
) -> dict:
    kl_thresholds = [row["kl_score_summary"]["q75"] for row in user_profile_rows]
    selected_scores = np.asarray([row["range_score"] for row in selected_rows], dtype=np.float64) if selected_rows else np.asarray([0.0])
    rejected_scores = np.asarray([row["range_score"] for row in rejected_rows], dtype=np.float64) if rejected_rows else np.asarray([0.0])
    return {
        "category": CATEGORY,
        "output_tag": OUTPUT_TAG,
        "device": str(DEVICE),
        "feature_names": feature_names,
        "user_count_total": int(len(user_ids) + len(excluded_rows)),
        "user_count_trained": int(len(user_ids)),
        "user_count_excluded": int(len(excluded_rows)),
        "sentence_count_total": int(len(sentence_rows)),
        "train_sentences_per_user": TRAIN_SENTENCES_PER_USER,
        "holdout_sentences_per_user": HOLDOUT_SENTENCES_PER_USER,
        "candidate_count_total": int(len(selected_rows) + len(rejected_rows)),
        "selected_query_count": int(len(selected_rows)),
        "rejected_query_count": int(len(rejected_rows)),
        "selected_user_count": int(len(query_output_rows)),
        "training": {
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "user_match_weight": USER_MATCH_WEIGHT,
            "style_recon_weight": STYLE_RECON_WEIGHT,
            "sent_kl_weight": SENT_KL_WEIGHT,
            "user_prior_kl_weight": USER_PRIOR_KL_WEIGHT,
            "epoch_summaries": training_info["epochs"],
        },
        "holdout_abs_threshold_quantile": ABS_THRESHOLD_QUANTILE,
        "user_kl_q75_summary": summarize_array(np.asarray(kl_thresholds, dtype=np.float64)),
        "selected_range_score_summary": summarize_array(selected_scores),
        "rejected_range_score_summary": summarize_array(rejected_scores),
        "summary_file": str(SUMMARY_FILE),
        "detail_file": str(DETAIL_FILE),
        "user_profile_file": str(USER_PROFILE_FILE),
        "sentence_file": str(SENTENCE_FILE),
        "excluded_user_file": str(EXCLUDED_USER_FILE),
        "selected_record_file": str(SELECTED_RECORD_FILE),
        "rejected_record_file": str(REJECTED_RECORD_FILE),
        "query_file": str(QUERY_FILE),
    }


def ensure_directories() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    QUERY_FILE.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    global DEVICE
    DEVICE = infer_device()
    set_random_seed(SEED)
    ensure_directories()
    log(f"运行设备: {DEVICE}")
    log(f"训练批大小: {BATCH_SIZE}")
    log("开始读取候选 query")
    candidate_rows = load_candidate_query_rows()
    candidate_user_ids = {row["user_id"] for row in candidate_rows}
    user_rows = load_filtered_user_reviews()
    user_rows = [row for row in user_rows if row.get("user_id") in candidate_user_ids]
    if MAX_USERS_OVERRIDE is not None:
        max_users = int(MAX_USERS_OVERRIDE)
        if max_users <= 0:
            raise ValueError("VADES_MAX_USERS 必须是正整数")
        user_rows = user_rows[:max_users]
        log(f"启用用户数限制: {len(user_rows)}")
    sentence_rows, excluded_rows = extract_first_twenty_sentences_for_users(user_rows)
    enriched_sentence_rows, feature_names = build_sentence_feature_rows(sentence_rows)
    user_ids, dataset = build_training_dataset(enriched_sentence_rows, feature_names)
    encoder, user_table, training_info = train_vades_user_distribution_model(user_ids, dataset, feature_names)
    sentence_output_rows, user_profile_rows = infer_user_sentence_distributions(encoder, user_table, user_ids, dataset)
    abs_thresholds = calibrate_absolute_threshold_with_unseen_holdout(encoder, user_table, user_ids, dataset)
    selected_rows, rejected_rows, query_output_rows = rank_and_select_queries(
        encoder=encoder,
        user_table=user_table,
        dataset=dataset,
        candidate_rows=candidate_rows,
        user_ids=user_ids,
        user_profile_rows=user_profile_rows,
        abs_thresholds=abs_thresholds,
    )

    write_jsonl(SENTENCE_FILE, sentence_output_rows)
    write_jsonl(EXCLUDED_USER_FILE, excluded_rows)
    write_jsonl(USER_PROFILE_FILE, user_profile_rows)
    write_jsonl(SELECTED_RECORD_FILE, selected_rows)
    write_jsonl(REJECTED_RECORD_FILE, rejected_rows)
    QUERY_FILE.write_text(json.dumps(query_output_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = build_summary(
        user_ids=user_ids,
        sentence_rows=sentence_output_rows,
        excluded_rows=excluded_rows,
        feature_names=feature_names,
        training_info=training_info,
        user_profile_rows=user_profile_rows,
        selected_rows=selected_rows,
        rejected_rows=rejected_rows,
        query_output_rows=query_output_rows,
    )
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(DETAIL_FILE, training_info["epochs"])
    if SKIP_POST_CLUSTERING:
        log("按配置跳过训练后的 query GMM 聚类与 retrieval attach")
    else:
        run_query_gmm_pipeline(
            category=CATEGORY,
            query_file=QUERY_FILE,
            write_back_to_query_file=False,
            attach_retrieval=True,
        )
    log(f"已写入 summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
