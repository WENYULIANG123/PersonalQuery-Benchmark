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
    torch.cuda.manual_seed_all(seed)


def require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("该脚本要求 CUDA GPU 环境，请使用 sbatch_wrapper --gpu 提交")
    return torch.device("cuda")


def normalize_review_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    if not text:
        raise ValueError("评论文本归一化后为空")
    return text


def sentence_has_non_root_dependency(doc_like) -> bool:
    tokens = [token for token in doc_like if not token.is_space]
    if not tokens:
        return False
    return any(token.head != token for token in tokens)


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} 为空")
    return rows


def load_json_list(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} 必须是非空列表")
    return data


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


def feature_names_from_rows(rows: list[dict]) -> list[str]:
    names = list(rows[0]["features"].keys())
    if not names:
        raise ValueError("特征名为空")
    for idx, row in enumerate(rows):
        if list(row["features"].keys()) != names:
            raise ValueError(f"第 {idx} 行特征名不一致")
    return names


def feature_vector(row: dict, feature_names: list[str]) -> np.ndarray:
    return np.array([float(row["features"][name]) for name in feature_names], dtype=np.float32)


def build_candidate_feature_rows_from_raw_query_file() -> list[dict]:
    if not RAW_CANDIDATE_QUERY_FILE.exists():
        raise FileNotFoundError(f"缺少原始 10 候选 query 文件: {RAW_CANDIDATE_QUERY_FILE}")

    raw_rows = load_json_list(RAW_CANDIDATE_QUERY_FILE)
    nlp = load_spacy_model()
    candidate_feature_rows = []
    expected_feature_names = None

    for row_index, raw_row in enumerate(raw_rows, start=1):
        user_id = raw_row.get("user_id")
        asin = raw_row.get("asin")
        candidates = raw_row.get("syntax_depth_queries")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError(f"第 {row_index} 条原始候选缺少合法 user_id")
        if not isinstance(asin, str) or not asin:
            raise ValueError(f"第 {row_index} 条原始候选缺少合法 asin")
        if not isinstance(candidates, list) or len(candidates) != 10:
            raise ValueError(f"user {user_id} 的 syntax_depth_queries 数量不是 10")

        query_texts = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            query_text = candidate.get("query")
            if not isinstance(query_text, str) or not query_text.strip():
                raise ValueError(f"user {user_id} candidate {candidate_index} 缺少合法 query 文本")
            query_texts.append(query_text.strip())

        for candidate_index, (candidate, doc) in enumerate(zip(candidates, nlp.pipe(query_texts, batch_size=16), strict=True), start=1):
            extracted = extract_clause_features_from_doc(doc, query_texts[candidate_index - 1])
            features = extracted["features"]
            feature_names = list(features.keys())
            if expected_feature_names is None:
                expected_feature_names = feature_names
            elif feature_names != expected_feature_names:
                raise ValueError(f"user {user_id} candidate {candidate_index} 的特征名与前文不一致")

            target_depth = candidate.get("target_depth")
            user_avg_depth = candidate.get("user_avg_depth")
            attrs_used = candidate.get("attrs_used")
            if not isinstance(target_depth, int):
                raise ValueError(f"user {user_id} candidate {candidate_index} 缺少合法 target_depth")
            if not isinstance(user_avg_depth, (int, float)):
                raise ValueError(f"user {user_id} candidate {candidate_index} 缺少合法 user_avg_depth")
            if not isinstance(attrs_used, dict):
                raise ValueError(f"user {user_id} candidate {candidate_index} 缺少合法 attrs_used")

            candidate_feature_rows.append({
                "user_id": user_id,
                "asin": asin,
                "candidate_index": candidate_index,
                "query": query_texts[candidate_index - 1],
                "word_count": int(extracted["word_count"]),
                "target_depth": int(target_depth),
                "user_avg_depth": float(user_avg_depth),
                "attrs_used": attrs_used,
                "features": features,
            })

    if expected_feature_names is None:
        raise ValueError("未能从原始 10 候选 query 文件中抽取任何特征")
    return candidate_feature_rows


def ensure_candidate_feature_file_exists() -> None:
    if CANDIDATE_QUERY_FILE.exists():
        return

    log(f"未发现候选特征文件，开始从原始 10 候选 query 生成: {RAW_CANDIDATE_QUERY_FILE}")
    candidate_feature_rows = build_candidate_feature_rows_from_raw_query_file()
    CANDIDATE_QUERY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CANDIDATE_QUERY_FILE.open("w", encoding="utf-8") as handle:
        for row in candidate_feature_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    log(f"候选特征文件已写入: {CANDIDATE_QUERY_FILE}")


def load_candidate_rows() -> tuple[list[str], list[dict], list[str]]:
    ensure_candidate_feature_file_exists()
    candidate_rows = load_jsonl(CANDIDATE_QUERY_FILE)
    feature_names = feature_names_from_rows(candidate_rows)
    counts = defaultdict(int)
    for row in candidate_rows:
        counts[row["user_id"]] += 1
    bad_users = [user_id for user_id, count in counts.items() if count != 10]
    if bad_users:
        raise ValueError(f"存在候选 query 数量不是 10 的用户: {bad_users[:5]}")
    return sorted(counts.keys()), candidate_rows, feature_names


def validate_sentence_rows(sentence_rows: list[dict], user_ids: list[str], feature_names: list[str]) -> None:
    if not sentence_rows:
        raise ValueError("句子缓存为空")
    if feature_names_from_rows(sentence_rows) != feature_names:
        raise ValueError("句子缓存特征名与候选 query 特征名不一致")

    rows_by_user: dict[str, list[dict]] = defaultdict(list)
    for row in sentence_rows:
        user_id = row.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("句子缓存存在非法 user_id")
        rows_by_user[user_id].append(row)

    expected_users = set(user_ids)
    cached_users = set(rows_by_user.keys())
    if cached_users != expected_users:
        missing_users = sorted(expected_users - cached_users)
        extra_users = sorted(cached_users - expected_users)
        raise ValueError(f"句子缓存用户集合不匹配: missing={missing_users[:5]}, extra={extra_users[:5]}")

    for user_id in user_ids:
        user_rows = rows_by_user[user_id]
        if len(user_rows) != TOTAL_SENTENCES_PER_USER:
            raise ValueError(f"user {user_id} 的句子数不是 {TOTAL_SENTENCES_PER_USER}")


def extract_first_twenty_sentences_for_users(user_ids: list[str]) -> tuple[list[dict], list[dict]]:
    source = json.loads(REVIEW_SOURCE_FILE.read_text(encoding="utf-8"))
    users = source.get("users")
    if not isinstance(users, list):
        raise TypeError("stage1_filtered_users_reviews.json 顶层 users 必须是列表")

    user_entries = {}
    for user_entry in users:
        user_id = user_entry.get("user_id")
        if user_id in user_ids:
            user_entries[user_id] = user_entry
    missing_users = [user_id for user_id in user_ids if user_id not in user_entries]
    if missing_users:
        raise ValueError(f"源评论文件缺少用户: {missing_users[:5]}")

    nlp = load_spacy_model()
    sentence_rows = []
    excluded_rows = []
    total_users = len(user_ids)
    for user_offset, user_id in enumerate(user_ids, start=1):
        user_entry = user_entries[user_id]
        review_texts = []
        results = user_entry.get("results", [])
        if not isinstance(results, list):
            raise TypeError("user.results 必须是列表")
        for product in results:
            target_reviews = product.get("target_reviews", [])
            if not isinstance(target_reviews, list):
                raise TypeError("product.target_reviews 必须是列表")
            for review in target_reviews:
                if not isinstance(review, str):
                    raise TypeError("target_reviews 中存在非字符串评论")
                if not review.strip():
                    continue
                review_texts.append(normalize_review_text(review))
        if not review_texts:
            raise ValueError(f"user {user_id} 没有可用评论文本")

        collected = []
        for review_index, review_doc in enumerate(nlp.pipe(review_texts, batch_size=16)):
            for sentence_index, sent in enumerate(review_doc.sents):
                sentence_text = sent.text.strip()
                if not sentence_text:
                    continue
                if not sentence_has_non_root_dependency(sent):
                    continue
                extracted = extract_clause_features_from_doc(sent, sentence_text)
                collected.append({
                    "user_id": user_id,
                    "review_index": review_index,
                    "sentence_index": sentence_index,
                    "sentence_text": sentence_text,
                    "word_count": extracted["word_count"],
                    "features": extracted["features"],
                })
                if len(collected) == TOTAL_SENTENCES_PER_USER:
                    break
            if len(collected) == TOTAL_SENTENCES_PER_USER:
                break
        if len(collected) < TOTAL_SENTENCES_PER_USER:
            excluded_rows.append({
                "user_id": user_id,
                "usable_sentence_count": len(collected),
            })
            log(
                f"用户 {user_offset}/{total_users}: {user_id}, 句子数不足 {TOTAL_SENTENCES_PER_USER}, "
                f"实际={len(collected)}，从本方法中过滤"
            )
            continue

        sentence_rows.extend(collected)
        log(f"已完成用户 {user_offset}/{total_users}: {user_id}, 句子数={len(collected)}")

    return sentence_rows, excluded_rows


def load_or_extract_sentence_rows(user_ids: list[str], feature_names: list[str]) -> tuple[list[dict], list[dict]]:
    if SENTENCE_FILE.exists() and EXCLUDED_USER_FILE.exists():
        log(f"发现句子缓存，开始校验并复用: {SENTENCE_FILE}")
        sentence_rows = load_jsonl(SENTENCE_FILE)
        validate_sentence_rows(sentence_rows, user_ids, feature_names)
        excluded_rows = load_jsonl(EXCLUDED_USER_FILE)
        excluded_user_ids = {row["user_id"] for row in excluded_rows}
        cached_user_ids = {row["user_id"] for row in sentence_rows}
        if cached_user_ids & excluded_user_ids:
            raise ValueError("句子缓存用户与排除用户有重叠")
        if cached_user_ids | excluded_user_ids != set(user_ids):
            raise ValueError("句子缓存与排除用户集合并后不等于原始候选用户集合")
        log(f"句子缓存校验通过，直接复用: {len(sentence_rows)} 条；排除用户数={len(excluded_rows)}")
        return sentence_rows, excluded_rows

    log(f"未发现 {TOTAL_SENTENCES_PER_USER} 句缓存，开始抽取每个用户前 {TOTAL_SENTENCES_PER_USER} 个可用评论句子")
    sentence_rows, excluded_rows = extract_first_twenty_sentences_for_users(user_ids)
    cached_user_ids = sorted({row["user_id"] for row in sentence_rows})
    validate_sentence_rows(sentence_rows, cached_user_ids, feature_names)
    SENTENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SENTENCE_FILE.open("w", encoding="utf-8") as handle:
        for row in sentence_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    with EXCLUDED_USER_FILE.open("w", encoding="utf-8") as handle:
        for row in excluded_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    log(f"句子缓存已写入: {SENTENCE_FILE}")
    log(f"排除用户文件已写入: {EXCLUDED_USER_FILE}")
    return sentence_rows, excluded_rows


def build_datasets(
    sentence_rows: list[dict],
    excluded_rows: list[dict],
    candidate_rows: list[dict],
    feature_names: list[str],
) -> tuple[list[str], dict]:
    all_rows_by_user: dict[str, list[dict]] = defaultdict(list)
    for row in sentence_rows:
        all_rows_by_user[row["user_id"]].append(row)

    train_by_user: dict[str, list[np.ndarray]] = defaultdict(list)
    train_meta_by_user: dict[str, list[dict]] = defaultdict(list)
    holdout_by_user: dict[str, list[np.ndarray]] = defaultdict(list)
    holdout_meta_by_user: dict[str, list[dict]] = defaultdict(list)
    for user_id, rows in all_rows_by_user.items():
        if len(rows) != TOTAL_SENTENCES_PER_USER:
            raise ValueError(f"user {user_id} 句子数不是 {TOTAL_SENTENCES_PER_USER}")
        train_rows = rows[:TRAIN_SENTENCES_PER_USER]
        holdout_rows = rows[TRAIN_SENTENCES_PER_USER:]
        if len(train_rows) != TRAIN_SENTENCES_PER_USER or len(holdout_rows) != HOLDOUT_SENTENCES_PER_USER:
            raise ValueError(f"user {user_id} 的 train/holdout 句子切分不正确")
        for row in train_rows:
            train_by_user[user_id].append(feature_vector(row, feature_names))
            train_meta_by_user[user_id].append(row)
        for row in holdout_rows:
            holdout_by_user[user_id].append(feature_vector(row, feature_names))
            holdout_meta_by_user[user_id].append(row)

    candidates_by_user: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        row_copy = dict(row)
        row_copy["feature_vector"] = feature_vector(row_copy, feature_names)
        candidates_by_user[row["user_id"]].append(row_copy)

    user_ids = sorted(train_by_user.keys())
    for user_id in user_ids:
        if len(train_by_user[user_id]) != TRAIN_SENTENCES_PER_USER:
            raise ValueError(f"user {user_id} 训练句子数不是 {TRAIN_SENTENCES_PER_USER}")
        if len(holdout_by_user[user_id]) != HOLDOUT_SENTENCES_PER_USER:
            raise ValueError(f"user {user_id} 校准句子数不是 {HOLDOUT_SENTENCES_PER_USER}")
        if user_id not in candidates_by_user or len(candidates_by_user[user_id]) != 10:
            raise ValueError(f"user {user_id} 候选 query 数量不是 10")

    return user_ids, {
        "original_candidate_user_count": len(user_ids) + len(excluded_rows),
        "excluded_rows": excluded_rows,
        "train_by_user": train_by_user,
        "train_meta_by_user": train_meta_by_user,
        "holdout_by_user": holdout_by_user,
        "holdout_meta_by_user": holdout_meta_by_user,
        "candidates_by_user": candidates_by_user,
    }


class SentenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
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

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(inputs)
        return self.mu_head(hidden), self.logvar_head(hidden)

    def decode(self, latent_mu: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_mu)


class UserDistributionTable(nn.Module):
    def __init__(self, num_users: int, latent_dim: int) -> None:
        super().__init__()
        self.mu = nn.Embedding(num_users, latent_dim)
        self.logvar = nn.Embedding(num_users, latent_dim)
        nn.init.normal_(self.mu.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.logvar.weight, -1.5)

    def forward(self, user_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu(user_indices), self.logvar(user_indices)


def gaussian_kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=1))


def diagonal_gaussian_kl(
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
) -> torch.Tensor:
    var_p = torch.exp(logvar_p)
    var_q = torch.exp(logvar_q)
    kl = 0.5 * (logvar_q - logvar_p + (var_p + (mu_p - mu_q).pow(2)) / var_q - 1.0)
    return torch.sum(kl, dim=-1)


def diagonal_gaussian_kl_numpy(
    mu_p: np.ndarray,
    logvar_p: np.ndarray,
    mu_q: np.ndarray,
    logvar_q: np.ndarray,
) -> float:
    var_p = np.exp(logvar_p)
    var_q = np.exp(logvar_q)
    kl = 0.5 * (logvar_q - logvar_p + (var_p + (mu_p - mu_q) ** 2) / var_q - 1.0)
    return float(np.sum(kl))


def tensor_from_numpy(array: np.ndarray) -> torch.Tensor:
    if DEVICE is None:
        raise RuntimeError("DEVICE 尚未初始化")
    return torch.from_numpy(array.astype(np.float32)).to(DEVICE)


def build_training_rows(user_ids: list[str], dataset: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    feature_rows = []
    label_rows = []
    for user_id in user_ids:
        block = np.vstack(dataset["train_by_user"][user_id]).astype(np.float32)
        for row in block:
            feature_rows.append(row)
            label_rows.append(user_to_idx[user_id])
    return np.vstack(feature_rows), np.array(label_rows, dtype=np.int64), user_to_idx


def train_vades_user_distribution_model(user_ids: list[str], dataset: dict, feature_names: list[str]) -> tuple[SentenceEncoder, UserDistributionTable, dict]:
    set_random_seed(SEED)
    raw_features, labels, user_to_idx = build_training_rows(user_ids, dataset)
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(raw_features).astype(np.float32)

    encoder = SentenceEncoder(input_dim=len(feature_names), hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    user_table = UserDistributionTable(num_users=len(user_ids), latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(user_table.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    feature_tensor = tensor_from_numpy(feature_matrix)
    label_tensor = torch.from_numpy(labels).to(DEVICE)
    num_rows = feature_tensor.shape[0]
    epoch_records = []

    for epoch in range(EPOCHS):
        encoder.train()
        user_table.train()
        permutation = torch.randperm(num_rows, device=DEVICE)

        batch_totals = []
        batch_user_match = []
        batch_recon = []
        batch_sent_kl = []
        batch_user_prior = []

        for start in range(0, num_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_rows)
            batch_indices = permutation[start:end]
            batch_features = feature_tensor[batch_indices]
            batch_labels = label_tensor[batch_indices]

            optimizer.zero_grad()

            sent_mu, sent_logvar = encoder.encode(batch_features)
            reconstruction = encoder.decode(sent_mu)

            all_user_mu = user_table.mu.weight
            all_user_logvar = user_table.logvar.weight
            expanded_sent_mu = sent_mu.unsqueeze(1)
            expanded_sent_logvar = sent_logvar.unsqueeze(1)
            expanded_user_mu = all_user_mu.unsqueeze(0)
            expanded_user_logvar = all_user_logvar.unsqueeze(0)
            kl_matrix = diagonal_gaussian_kl(
                expanded_sent_mu,
                expanded_sent_logvar,
                expanded_user_mu,
                expanded_user_logvar,
            )
            user_match_loss = F.cross_entropy(-kl_matrix, batch_labels)
            style_recon_loss = F.mse_loss(reconstruction, batch_features)
            sent_kl_loss = gaussian_kl_to_standard_normal(sent_mu, sent_logvar)
            user_mu, user_logvar = user_table(batch_labels)
            user_prior_kl_loss = gaussian_kl_to_standard_normal(user_mu, user_logvar)
            total_loss = (
                USER_MATCH_WEIGHT * user_match_loss
                + STYLE_RECON_WEIGHT * style_recon_loss
                + SENT_KL_WEIGHT * sent_kl_loss
                + USER_PRIOR_KL_WEIGHT * user_prior_kl_loss
            )
            total_loss.backward()
            optimizer.step()

            batch_totals.append(float(total_loss.detach().cpu().item()))
            batch_user_match.append(float(user_match_loss.detach().cpu().item()))
            batch_recon.append(float(style_recon_loss.detach().cpu().item()))
            batch_sent_kl.append(float(sent_kl_loss.detach().cpu().item()))
            batch_user_prior.append(float(user_prior_kl_loss.detach().cpu().item()))

        epoch_record = {
            "epoch": epoch + 1,
            "total_loss": float(np.mean(batch_totals)),
            "user_match_loss": float(np.mean(batch_user_match)),
            "style_recon_loss": float(np.mean(batch_recon)),
            "sent_kl_loss": float(np.mean(batch_sent_kl)),
            "user_prior_kl_loss": float(np.mean(batch_user_prior)),
        }
        epoch_records.append(epoch_record)
        if epoch == 0 or (epoch + 1) % 50 == 0:
            log(
                f"epoch {epoch + 1}/{EPOCHS}: total={epoch_record['total_loss']:.6f}, "
                f"user_match={epoch_record['user_match_loss']:.6f}, "
                f"recon={epoch_record['style_recon_loss']:.6f}, "
                f"sent_kl={epoch_record['sent_kl_loss']:.6f}, "
                f"user_prior_kl={epoch_record['user_prior_kl_loss']:.6f}"
            )

    return encoder, user_table, {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "epoch_records": epoch_records,
        "user_to_idx": user_to_idx,
    }


def extract_user_distributions(user_ids: list[str], training_info: dict, user_table: UserDistributionTable) -> dict:
    user_table.eval()
    user_to_idx = training_info["user_to_idx"]
    user_mu = {}
    user_logvar = {}
    user_var = {}
    with torch.no_grad():
        for user_id in user_ids:
            idx_tensor = torch.tensor([user_to_idx[user_id]], dtype=torch.long, device=DEVICE)
            mu, logvar = user_table(idx_tensor)
            mu_np = mu.cpu().numpy()[0]
            logvar_np = logvar.cpu().numpy()[0]
            user_mu[user_id] = mu_np
            user_logvar[user_id] = logvar_np
            user_var[user_id] = np.exp(logvar_np)
    return {
        "user_mu": user_mu,
        "user_logvar": user_logvar,
        "user_var": user_var,
    }


def encode_sentence_distributions(
    encoder: SentenceEncoder,
    training_info: dict,
    user_ids: list[str],
    dataset: dict,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    encoder.eval()
    scaler_mean = np.array(training_info["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(training_info["scaler_scale"], dtype=np.float32)
    encoded_train_by_user: dict[str, list[dict]] = {}
    encoded_holdout_by_user: dict[str, list[dict]] = {}
    with torch.no_grad():
        for user_id in user_ids:
            train_sentence_rows = dataset["train_meta_by_user"][user_id]
            train_feature_rows = dataset["train_by_user"][user_id]
            encoded_train_rows = []
            for meta_row, feature_row in zip(train_sentence_rows, train_feature_rows, strict=True):
                standardized = (feature_row - scaler_mean) / scaler_scale
                sent_mu, sent_logvar = encoder.encode(tensor_from_numpy(standardized.reshape(1, -1)))
                sent_mu_np = sent_mu.cpu().numpy()[0]
                sent_logvar_np = sent_logvar.cpu().numpy()[0]
                encoded_train_rows.append({
                    "sentence_text": meta_row["sentence_text"],
                    "word_count": meta_row["word_count"],
                    "sentence_mu": sent_mu_np,
                    "sentence_logvar": sent_logvar_np,
                    "sentence_var": np.exp(sent_logvar_np),
                })
            if len(encoded_train_rows) != TRAIN_SENTENCES_PER_USER:
                raise ValueError(f"user {user_id} 编码训练句子数不是 {TRAIN_SENTENCES_PER_USER}")

            holdout_sentence_rows = dataset["holdout_meta_by_user"][user_id]
            holdout_feature_rows = dataset["holdout_by_user"][user_id]
            encoded_holdout_rows = []
            for meta_row, feature_row in zip(holdout_sentence_rows, holdout_feature_rows, strict=True):
                standardized = (feature_row - scaler_mean) / scaler_scale
                sent_mu, sent_logvar = encoder.encode(tensor_from_numpy(standardized.reshape(1, -1)))
                sent_mu_np = sent_mu.cpu().numpy()[0]
                sent_logvar_np = sent_logvar.cpu().numpy()[0]
                encoded_holdout_rows.append({
                    "sentence_text": meta_row["sentence_text"],
                    "word_count": meta_row["word_count"],
                    "sentence_mu": sent_mu_np,
                    "sentence_logvar": sent_logvar_np,
                    "sentence_var": np.exp(sent_logvar_np),
                })
            if len(encoded_holdout_rows) != HOLDOUT_SENTENCES_PER_USER:
                raise ValueError(f"user {user_id} 编码校准句子数不是 {HOLDOUT_SENTENCES_PER_USER}")

            encoded_train_by_user[user_id] = encoded_train_rows
            encoded_holdout_by_user[user_id] = encoded_holdout_rows
    return encoded_train_by_user, encoded_holdout_by_user


def estimate_distribution_from_sentence_distributions(sentence_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not sentence_rows:
        raise ValueError("无法从空句子集合估计用户分布")
    mu_matrix = np.stack([row["sentence_mu"] for row in sentence_rows], axis=0)
    var_matrix = np.stack([row["sentence_var"] for row in sentence_rows], axis=0)
    user_mu = np.mean(mu_matrix, axis=0)
    second_moment = np.mean(var_matrix + mu_matrix ** 2, axis=0)
    user_var = second_moment - user_mu ** 2
    if np.any(user_var <= 0.0):
        raise ValueError("估计出的用户方差存在非正值")
    user_logvar = np.log(user_var)
    return user_mu.astype(np.float32), user_logvar.astype(np.float32), user_var.astype(np.float32)


def build_full_sentence_estimated_user_distributions(
    user_ids: list[str],
    encoded_train_sentences_by_user: dict[str, list[dict]],
) -> dict:
    user_mu = {}
    user_logvar = {}
    user_var = {}
    for user_id in user_ids:
        mu, logvar, var = estimate_distribution_from_sentence_distributions(encoded_train_sentences_by_user[user_id])
        user_mu[user_id] = mu
        user_logvar[user_id] = logvar
        user_var[user_id] = var
    return {
        "user_mu": user_mu,
        "user_logvar": user_logvar,
        "user_var": user_var,
    }


def calibrate_absolute_threshold_leave_one_out(
    user_ids: list[str],
    encoded_train_sentences_by_user: dict[str, list[dict]],
) -> tuple[dict[str, float], np.ndarray]:
    d_pos = []
    thresholds_by_user: dict[str, float] = {}
    for user_id in user_ids:
        sentence_rows = encoded_train_sentences_by_user[user_id]
        if len(sentence_rows) != TRAIN_SENTENCES_PER_USER:
            raise ValueError(f"user {user_id} 训练句子数不是 {TRAIN_SENTENCES_PER_USER}")
        user_distances = []
        for holdout_idx in range(TRAIN_SENTENCES_PER_USER):
            retained_rows = sentence_rows[:holdout_idx] + sentence_rows[holdout_idx + 1 :]
            if len(retained_rows) != TRAIN_SENTENCES_PER_USER - 1:
                raise ValueError("leave-one-out retained_rows 长度不正确")
            loo_mu, loo_logvar, _ = estimate_distribution_from_sentence_distributions(retained_rows)
            holdout_row = sentence_rows[holdout_idx]
            kl_value = diagonal_gaussian_kl_numpy(
                holdout_row["sentence_mu"],
                holdout_row["sentence_logvar"],
                loo_mu,
                loo_logvar,
            )
            d_pos.append(kl_value)
            user_distances.append(kl_value)
        user_distance_array = np.array(user_distances, dtype=np.float64)
        if len(user_distance_array) != TRAIN_SENTENCES_PER_USER:
            raise ValueError(f"user {user_id} 的 leave-one-out KL 数量不正确")
        thresholds_by_user[user_id] = float(np.quantile(user_distance_array, ABS_THRESHOLD_QUANTILE))
    d_pos_array = np.array(d_pos, dtype=np.float64)
    if len(d_pos_array) != len(user_ids) * TRAIN_SENTENCES_PER_USER:
        raise ValueError("leave-one-out 校准样本数不正确")
    return thresholds_by_user, d_pos_array


def calibrate_absolute_threshold_with_unseen_holdout(
    user_ids: list[str],
    encoded_train_sentences_by_user: dict[str, list[dict]],
    encoded_holdout_sentences_by_user: dict[str, list[dict]],
) -> tuple[dict[str, float], np.ndarray]:
    d_pos = []
    thresholds_by_user: dict[str, float] = {}
    for user_id in user_ids:
        train_rows = encoded_train_sentences_by_user[user_id]
        holdout_rows = encoded_holdout_sentences_by_user[user_id]
        if len(train_rows) != TRAIN_SENTENCES_PER_USER:
            raise ValueError(f"user {user_id} 训练句子数不是 {TRAIN_SENTENCES_PER_USER}")
        if len(holdout_rows) != HOLDOUT_SENTENCES_PER_USER:
            raise ValueError(f"user {user_id} 校准句子数不是 {HOLDOUT_SENTENCES_PER_USER}")
        user_mu, user_logvar, _ = estimate_distribution_from_sentence_distributions(train_rows)
        user_distances = []
        for holdout_row in holdout_rows:
            kl_value = diagonal_gaussian_kl_numpy(
                holdout_row["sentence_mu"],
                holdout_row["sentence_logvar"],
                user_mu,
                user_logvar,
            )
            d_pos.append(kl_value)
            user_distances.append(kl_value)
        user_distance_array = np.array(user_distances, dtype=np.float64)
        if len(user_distance_array) != HOLDOUT_SENTENCES_PER_USER:
            raise ValueError(f"user {user_id} 的 unseen holdout KL 数量不正确")
        thresholds_by_user[user_id] = float(np.quantile(user_distance_array, ABS_THRESHOLD_QUANTILE))
    d_pos_array = np.array(d_pos, dtype=np.float64)
    if len(d_pos_array) != len(user_ids) * HOLDOUT_SENTENCES_PER_USER:
        raise ValueError("unseen holdout 校准样本数不正确")
    return thresholds_by_user, d_pos_array


def select_queries_with_user_distributions(
    encoder: SentenceEncoder,
    training_info: dict,
    user_ids: list[str],
    dataset: dict,
    user_distributions: dict,
    thresholds_by_user: dict[str, float],
) -> tuple[list[dict], list[dict]]:
    encoder.eval()
    scaler_mean = np.array(training_info["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(training_info["scaler_scale"], dtype=np.float32)

    selected = []
    rejected = []
    for user_id in user_ids:
        candidate_rows = dataset["candidates_by_user"][user_id]
        user_mu_np = user_distributions["user_mu"][user_id]
        user_logvar_np = user_distributions["user_logvar"][user_id]
        user_threshold = thresholds_by_user[user_id]
        scored_candidates = []
        for candidate in candidate_rows:
            standardized = (candidate["feature_vector"] - scaler_mean) / scaler_scale
            with torch.no_grad():
                query_mu, query_logvar = encoder.encode(tensor_from_numpy(standardized.reshape(1, -1)))
            user_mu = torch.from_numpy(user_mu_np).to(DEVICE).reshape(1, -1)
            user_logvar = torch.from_numpy(user_logvar_np).to(DEVICE).reshape(1, -1)
            range_score = float(diagonal_gaussian_kl(query_mu, query_logvar, user_mu, user_logvar).cpu().item())
            scored = dict(candidate)
            scored.pop("feature_vector", None)
            scored["query_mu"] = query_mu.cpu().numpy()[0].tolist()
            scored["query_logvar"] = query_logvar.cpu().numpy()[0].tolist()
            scored["range_score"] = range_score
            scored["user_abs_threshold"] = user_threshold
            scored["passes_abs_threshold"] = bool(range_score <= user_threshold)
            scored_candidates.append(scored)
        scored_candidates.sort(key=lambda row: row["range_score"])
        passed_candidates = [row for row in scored_candidates if row["passes_abs_threshold"]]
        if passed_candidates:
            best = dict(passed_candidates[0])
            best["selection_status"] = "selected"
            best["candidate_pass_count"] = len(passed_candidates)
            selected.append(best)
            continue
        best = dict(scored_candidates[0])
        best["selection_status"] = "rejected"
        best["candidate_pass_count"] = 0
        rejected.append(best)
    return selected, rejected


def build_query_payload(selected_rows: list[dict]) -> list[dict]:
    payload = []
    for row in selected_rows:
        payload.append({
            "user_id": row["user_id"],
            "asin": row["asin"],
            "syntax_depth_query": {
                "target_depth": row["target_depth"],
                "actual_depth": "",
                "user_avg_depth": row["user_avg_depth"],
                "query": row["query"],
                "word_count": row["word_count"],
                "attrs_used": row["attrs_used"],
                "accepted_candidate_index": row["candidate_index"],
                "candidate_count": 10,
            },
        })
    return payload


def main() -> None:
    global DEVICE
    DEVICE = require_cuda_device()
    set_random_seed(SEED)
    log(f"运行设备: {DEVICE}")
    log(f"训练批大小: {BATCH_SIZE}")
    log("开始读取候选 query")
    user_ids, candidate_rows, feature_names = load_candidate_rows()
    log(f"候选用户数: {len(user_ids)}")

    sentence_rows, excluded_rows = load_or_extract_sentence_rows(user_ids, feature_names)

    log(f"开始构造 train {TRAIN_SENTENCES_PER_USER} + holdout {HOLDOUT_SENTENCES_PER_USER} 句子数据集")
    user_ids, dataset = build_datasets(sentence_rows, excluded_rows, candidate_rows, feature_names)
    log(
        f"原始候选用户数={dataset['original_candidate_user_count']}，"
        f"可用于本方法的用户数={len(user_ids)}，"
        f"因句子不足 {TOTAL_SENTENCES_PER_USER} 被过滤用户数={len(excluded_rows)}"
    )

    log("开始训练论文式句子级 VADES")
    encoder, user_table, training_info = train_vades_user_distribution_model(user_ids, dataset, feature_names)

    log("开始编码训练句子与未训练校准句子分布")
    encoded_train_sentences_by_user, encoded_holdout_sentences_by_user = encode_sentence_distributions(
        encoder, training_info, user_ids, dataset
    )

    log(f"开始基于前 {TRAIN_SENTENCES_PER_USER} 个训练句子估计用户分布")
    user_distributions = build_full_sentence_estimated_user_distributions(user_ids, encoded_train_sentences_by_user)

    log(f"开始使用后 {HOLDOUT_SENTENCES_PER_USER} 个未训练句子执行校准")
    thresholds_by_user, d_pos_array = calibrate_absolute_threshold_with_unseen_holdout(
        user_ids,
        encoded_train_sentences_by_user,
        encoded_holdout_sentences_by_user,
    )
    threshold_values = np.array([thresholds_by_user[user_id] for user_id in user_ids], dtype=np.float64)
    log(f"unseen-holdout per-user T_abs mean={float(np.mean(threshold_values)):.6f}")

    log("开始基于阈值执行 range-aware query 排序选择")
    selected_rows, rejected_rows = select_queries_with_user_distributions(
        encoder, training_info, user_ids, dataset, user_distributions, thresholds_by_user
    )

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    QUERY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with DETAIL_FILE.open("w", encoding="utf-8") as handle:
        for row in training_info["epoch_records"]:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    with USER_PROFILE_FILE.open("w", encoding="utf-8") as handle:
        for user_id in user_ids:
            handle.write(json.dumps({
                "user_id": user_id,
                "user_mu": user_distributions["user_mu"][user_id].tolist(),
                "user_logvar": user_distributions["user_logvar"][user_id].tolist(),
                "user_var": user_distributions["user_var"][user_id].tolist(),
                "user_abs_threshold": thresholds_by_user[user_id],
            }, ensure_ascii=False))
            handle.write("\n")

    with EXCLUDED_USER_FILE.open("w", encoding="utf-8") as handle:
        for row in excluded_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    with SELECTED_RECORD_FILE.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    with REJECTED_RECORD_FILE.open("w", encoding="utf-8") as handle:
        for row in rejected_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    query_payload = build_query_payload(selected_rows)
    QUERY_FILE.write_text(json.dumps(query_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("开始对写回后的 query 执行 GMM 分类并回写 cluster 标记")
    gmm_result = run_query_gmm_pipeline(
        category=CATEGORY,
        query_file=QUERY_FILE,
        write_back_to_query_file=True,
        attach_retrieval=False,
    )

    summary = {
        "category": CATEGORY,
        "method": "vades_sentence_user_distribution_range_aware_train10_holdout10_threshold",
        "review_source_file": str(REVIEW_SOURCE_FILE),
        "candidate_query_file": str(CANDIDATE_QUERY_FILE),
        "train_sentences_per_user": TRAIN_SENTENCES_PER_USER,
        "holdout_sentences_per_user": HOLDOUT_SENTENCES_PER_USER,
        "total_required_sentences_per_user": TOTAL_SENTENCES_PER_USER,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "loss_weights": {
            "user_match": USER_MATCH_WEIGHT,
            "style_recon": STYLE_RECON_WEIGHT,
            "sent_kl": SENT_KL_WEIGHT,
            "user_prior_kl": USER_PRIOR_KL_WEIGHT,
        },
        "abs_threshold_quantile": ABS_THRESHOLD_QUANTILE,
        "abs_threshold_scope": "per_user_unseen_holdout",
        "abs_threshold_per_user_summary": summarize_array(threshold_values),
        "d_pos_summary": summarize_array(d_pos_array),
        "original_candidate_user_count": dataset["original_candidate_user_count"],
        "eligible_user_count": len(user_ids),
        "excluded_user_count": len(excluded_rows),
        "user_count": len(user_ids),
        "selected_count": len(selected_rows),
        "rejected_count": len(rejected_rows),
        "selected_rate": float(len(selected_rows) / len(user_ids)),
        "rejected_rate": float(len(rejected_rows) / len(user_ids)),
        "selected_rate_over_original_candidates": float(len(selected_rows) / dataset["original_candidate_user_count"]),
        "rejected_rate_over_original_candidates": float(len(rejected_rows) / dataset["original_candidate_user_count"]),
        "selected_range_score_summary": summarize_array(np.array([row["range_score"] for row in selected_rows], dtype=float)),
        "selected_candidate_index_summary": summarize_array(np.array([float(row["candidate_index"]) for row in selected_rows], dtype=float)),
        "selected_candidate_pass_count_summary": summarize_array(np.array([float(row["candidate_pass_count"]) for row in selected_rows], dtype=float)),
        "rejected_range_score_summary": None if not rejected_rows else summarize_array(np.array([row["range_score"] for row in rejected_rows], dtype=float)),
        "train_loss_summary": summarize_array(np.array([row["total_loss"] for row in training_info["epoch_records"]], dtype=float)),
        "last_epoch_record": training_info["epoch_records"][-1],
        "sentence_file": str(SENTENCE_FILE),
        "excluded_user_file": str(EXCLUDED_USER_FILE),
        "user_profile_file": str(USER_PROFILE_FILE),
        "selected_record_file": str(SELECTED_RECORD_FILE),
        "rejected_record_file": str(REJECTED_RECORD_FILE),
        "query_file": str(QUERY_FILE),
        "query_gmm_summary_file": gmm_result["summary_file"],
        "query_gmm_user_file": gmm_result["user_file"],
        "query_gmm_feature_file": gmm_result["feature_file"],
        "query_gmm_selected_k": gmm_result["selected_k"],
        "query_gmm_cluster_counts": gmm_result["cluster_counts"],
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "selected_count": len(selected_rows),
        "rejected_count": len(rejected_rows),
        "query_file": str(QUERY_FILE),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
