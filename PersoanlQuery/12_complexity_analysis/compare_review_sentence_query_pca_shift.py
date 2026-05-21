#!/usr/bin/env python3
"""Compare Baby_Products review-sentence PCA distribution with query positions on the same axis."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from extract_clause_features_single_query import extract_clause_features_from_doc, load_spacy_model  # noqa: E402


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
MAX_REVIEW_SENTENCES_PER_USER = 5
MAX_USERS = 100
QUERY_FEATURE_FILE = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY / "single_query_clause_features.jsonl"
REVIEW_SOURCE_FILE = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / CATEGORY / "stage1_filtered_users_reviews.json"
OUTPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
SUMMARY_FILE = OUTPUT_DIR / "review_sentence_vs_query_pca_shift_summary.json"
QUERY_RECORD_FILE = OUTPUT_DIR / "review_sentence_vs_query_pca_shift_query_records.jsonl"
USER_PROFILE_FILE = OUTPUT_DIR / "review_sentence_vs_query_pca_shift_user_profiles.jsonl"


def log(message: str) -> None:
    print(message, flush=True)


def normalize_review_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    if not text:
        raise ValueError("review 文本归一化后为空")
    return text


def load_query_feature_rows() -> tuple[list[str], list[dict]]:
    rows = [
        json.loads(line)
        for line in QUERY_FEATURE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("query 特征文件为空")
    feature_names = list(rows[0]["features"].keys())
    return feature_names, rows


def select_query_rows(query_rows: list[dict]) -> list[dict]:
    if MAX_USERS is None:
        return query_rows
    ordered_user_ids = []
    seen = set()
    for row in query_rows:
        user_id = row["user_id"]
        if user_id not in seen:
            seen.add(user_id)
            ordered_user_ids.append(user_id)
        if len(ordered_user_ids) == MAX_USERS:
            break
    selected_user_ids = set(ordered_user_ids)
    selected_rows = [row for row in query_rows if row["user_id"] in selected_user_ids]
    if not selected_rows:
        raise ValueError("MAX_USERS 过滤后 query 行为空")
    return selected_rows


def build_query_user_set(query_rows: list[dict]) -> set[str]:
    ordered_user_ids = []
    seen = set()
    for row in query_rows:
        user_id = row["user_id"]
        if user_id not in seen:
            seen.add(user_id)
            ordered_user_ids.append(user_id)
    user_ids = set(ordered_user_ids)
    if not user_ids:
        raise ValueError("query 用户集合为空")
    return user_ids


def extract_target_review_texts(user_entry: dict) -> list[str]:
    # 对齐 05_syntactic_analysis_Baby_Products.py 的读取口径：
    # reviews = [] ; for p in user_data.get('results', []): reviews.extend(p.get('target_reviews', []))
    reviews = []
    results = user_entry.get("results", [])
    if not isinstance(results, list):
        raise TypeError("user.results 必须是列表")
    for product in results:
        target_reviews = product.get("target_reviews", [])
        if not isinstance(target_reviews, list):
            raise TypeError("product.target_reviews 必须是列表")
        reviews.extend(target_reviews)

    normalized_reviews = []
    for review in reviews:
        if not isinstance(review, str):
            raise TypeError("target_reviews 中存在非字符串评论")
        if not review.strip():
            continue
        normalized_reviews.append(normalize_review_text(review))
    if not normalized_reviews:
        raise ValueError("用户没有可用的 target_reviews")
    return normalized_reviews


def select_review_sentences_for_users(target_user_ids: set[str]) -> dict[str, list[str]]:
    log("开始读取评论源 JSON")
    source = json.loads(REVIEW_SOURCE_FILE.read_text(encoding="utf-8"))
    log("评论源 JSON 读取完成")
    users = source.get("users")
    if not isinstance(users, list):
        raise TypeError("stage1_filtered_users_reviews.json 顶层 users 必须是列表")

    user_entry_index = {}
    for user_entry in users:
        user_id = user_entry.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("存在非法 user_id")
        user_entry_index[user_id] = user_entry

    missing_users = sorted(target_user_ids - set(user_entry_index.keys()))
    if missing_users:
        raise ValueError(f"评论源文件缺少 query 用户: {missing_users[:5]}")

    nlp = load_spacy_model()
    log("spaCy 模型加载完成，开始为每个用户选句")
    selected_sentences = {}
    for user_id in sorted(target_user_ids):
        review_texts = extract_target_review_texts(user_entry_index[user_id])
        sentences = []
        for doc in nlp.pipe(review_texts, batch_size=8):
            for sent in doc.sents:
                sentence_text = sent.text.strip()
                if not sentence_text:
                    continue
                sentences.append(sentence_text)
                if len(sentences) == MAX_REVIEW_SENTENCES_PER_USER:
                    break
            if len(sentences) == MAX_REVIEW_SENTENCES_PER_USER:
                break
        if not sentences:
            raise ValueError(f"user {user_id} 没有可用的评论句子")
        selected_sentences[user_id] = sentences
    return selected_sentences


def sentence_has_non_root_dependency(doc) -> bool:
    tokens = [token for token in doc if not token.is_space]
    if not tokens:
        return False
    return any(token.head != token for token in tokens)


def extract_review_sentence_features(feature_names: list[str], user_sentences: dict[str, list[str]]) -> list[dict]:
    sentence_rows = []
    ordered_items = []
    sentence_texts = []
    for user_id in sorted(user_sentences.keys()):
        for idx, sentence_text in enumerate(user_sentences[user_id]):
            ordered_items.append((user_id, idx, sentence_text))
            sentence_texts.append(sentence_text)

    nlp = load_spacy_model()
    log(f"开始批量解析评论句子，总句子数: {len(sentence_texts)}")
    kept_counts_by_user = defaultdict(int)
    for (user_id, sentence_index, sentence_text), doc in zip(ordered_items, nlp.pipe(sentence_texts, batch_size=64)):
        if not sentence_has_non_root_dependency(doc):
            continue
        extracted = extract_clause_features_from_doc(doc, sentence_text)
        features = extracted["features"]
        if list(features.keys()) != feature_names:
            raise ValueError("评论句子特征名与 query 特征名不一致")
        sentence_rows.append({
            "user_id": user_id,
            "sentence_index": sentence_index,
            "sentence": sentence_text,
            "word_count": extracted["word_count"],
            "features": features,
        })
        kept_counts_by_user[user_id] += 1

    missing_users = sorted(user_id for user_id in user_sentences.keys() if kept_counts_by_user[user_id] == 0)
    if missing_users:
        raise ValueError(f"以下用户在过滤退化句后没有保留任何评论句子: {missing_users[:5]}")
    return sentence_rows


def fit_review_pca(feature_names: list[str], review_rows: list[dict]) -> tuple[StandardScaler, PCA, float]:
    review_matrix = np.array(
        [[float(row["features"][name]) for name in feature_names] for row in review_rows],
        dtype=float,
    )
    if len(review_matrix) < 2:
        raise ValueError("评论句子数量不足，无法拟合 PCA")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(review_matrix)
    pca = PCA(n_components=1, random_state=42)
    scores = pca.fit_transform(standardized).ravel()
    anchor = np.array([float(row["features"]["max_dependency_depth"]) for row in review_rows], dtype=float)
    corr = np.corrcoef(scores, anchor)[0, 1]
    if np.isnan(corr):
        raise ValueError("评论 PCA 分数与 anchor 的相关性为 NaN")
    sign = 1.0 if corr >= 0 else -1.0
    return scaler, pca, sign


def project_scores(feature_names: list[str], rows: list[dict], scaler: StandardScaler, pca: PCA, sign: float) -> np.ndarray:
    matrix = np.array(
        [[float(row["features"][name]) for name in feature_names] for row in rows],
        dtype=float,
    )
    standardized = scaler.transform(matrix)
    return sign * pca.transform(standardized).ravel()


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


def main() -> None:
    log(f"PID={os.getpid()}")
    log("开始加载 query 特征行")
    feature_names, query_rows = load_query_feature_rows()
    query_rows = select_query_rows(query_rows)
    log(f"query 行数: {len(query_rows)}")
    query_user_ids = build_query_user_set(query_rows)
    log(f"query 用户数: {len(query_user_ids)}")
    log("开始为 query 用户抽取评论句子")
    user_sentences = select_review_sentences_for_users(query_user_ids)
    log(f"评论句子用户数: {len(user_sentences)}")
    log("开始抽取评论句子特征")
    review_rows = extract_review_sentence_features(feature_names, user_sentences)
    log(f"评论句子特征行数: {len(review_rows)}")

    log("开始用评论句子拟合 PCA 轴")
    scaler, pca, sign = fit_review_pca(feature_names, review_rows)
    log("开始投影评论和 query 到统一 PCA 轴")
    review_scores = project_scores(feature_names, review_rows, scaler, pca, sign)
    query_scores = project_scores(feature_names, query_rows, scaler, pca, sign)

    for row, score in zip(review_rows, review_scores):
        row["pca_score"] = float(score)
    for row, score in zip(query_rows, query_scores):
        row["pca_score"] = float(score)

    log("开始构建用户评论分布 profile")
    review_scores_by_user: dict[str, list[float]] = defaultdict(list)
    for row in review_rows:
        review_scores_by_user[row["user_id"]].append(float(row["pca_score"]))

    user_profiles = {}
    for user_id, scores in review_scores_by_user.items():
        score_array = np.array(scores, dtype=float)
        profile = {
            "user_id": user_id,
            "review_sentence_count": len(scores),
            "review_score_summary": summarize_array(score_array),
            "review_mean": float(np.mean(score_array)),
            "review_std": float(np.std(score_array)),
            "review_p25": float(np.quantile(score_array, 0.25)),
            "review_p75": float(np.quantile(score_array, 0.75)),
        }
        user_profiles[user_id] = profile

    query_records = []
    percentiles = []
    shifts = []
    valid_zscores = []
    above_mean_count = 0
    above_p75_count = 0
    below_p25_count = 0
    for row in query_rows:
        user_id = row["user_id"]
        if user_id not in user_profiles:
            raise ValueError(f"query user {user_id} 缺少评论 profile")
        profile = user_profiles[user_id]
        review_score_array = np.array(review_scores_by_user[user_id], dtype=float)
        query_score = float(row["pca_score"])
        percentile = float(np.mean(review_score_array <= query_score))
        shift = float(query_score - profile["review_mean"])
        if profile["review_std"] > 0:
            zscore = float(shift / profile["review_std"])
            valid_zscores.append(zscore)
        else:
            zscore = None

        if shift > 0:
            above_mean_count += 1
        if query_score > profile["review_p75"]:
            above_p75_count += 1
        if query_score < profile["review_p25"]:
            below_p25_count += 1

        percentiles.append(percentile)
        shifts.append(shift)

        query_record = {
            "user_id": user_id,
            "asin": row["asin"],
            "query_type": row["query_type"],
            "target_depth": row.get("target_depth"),
            "actual_depth": row.get("actual_depth"),
            "user_avg_depth": row.get("user_avg_depth"),
            "query": row["query"],
            "query_pca_score": query_score,
            "review_sentence_count": profile["review_sentence_count"],
            "review_score_summary": profile["review_score_summary"],
            "query_percentile_within_user_reviews": percentile,
            "query_shift_from_user_review_mean": shift,
            "query_zscore_within_user_reviews": zscore,
            "is_above_user_review_mean": bool(shift > 0),
            "is_above_user_review_p75": bool(query_score > profile["review_p75"]),
            "is_below_user_review_p25": bool(query_score < profile["review_p25"]),
        }
        query_records.append(query_record)

    log("开始写输出文件")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with QUERY_RECORD_FILE.open("w", encoding="utf-8") as handle:
        for row in query_records:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    with USER_PROFILE_FILE.open("w", encoding="utf-8") as handle:
        for user_id in sorted(user_profiles.keys()):
            handle.write(json.dumps(user_profiles[user_id], ensure_ascii=False))
            handle.write("\n")

    review_score_array = np.array(review_scores, dtype=float)
    query_score_array = np.array(query_scores, dtype=float)
    percentile_array = np.array(percentiles, dtype=float)
    shift_array = np.array(shifts, dtype=float)
    ks_result = ks_2samp(review_score_array, query_score_array)
    wasserstein = wasserstein_distance(review_score_array, query_score_array)

    summary = {
        "category": CATEGORY,
        "query_feature_file": str(QUERY_FEATURE_FILE),
        "review_source_file": str(REVIEW_SOURCE_FILE),
        "max_review_sentences_per_user": MAX_REVIEW_SENTENCES_PER_USER,
        "pca_fit_source": "review_sentences_only",
        "num_query_rows": len(query_rows),
        "num_query_users": len(query_user_ids),
        "num_review_sentence_rows": len(review_rows),
        "review_sentence_count_distribution": {
            "min": int(min(profile["review_sentence_count"] for profile in user_profiles.values())),
            "max": int(max(profile["review_sentence_count"] for profile in user_profiles.values())),
            "mean": float(np.mean([profile["review_sentence_count"] for profile in user_profiles.values()])),
        },
        "review_score_summary": summarize_array(review_score_array),
        "query_score_summary": summarize_array(query_score_array),
        "pooled_distribution_comparison": {
            "ks_statistic": float(ks_result.statistic),
            "ks_p_value": float(ks_result.pvalue),
            "wasserstein_distance": float(wasserstein),
        },
        "query_relative_to_user_reviews": {
            "mean_percentile": float(np.mean(percentile_array)),
            "median_percentile": float(np.median(percentile_array)),
            "mean_shift": float(np.mean(shift_array)),
            "median_shift": float(np.median(shift_array)),
            "mean_zscore": float(np.mean(valid_zscores)) if valid_zscores else None,
            "median_zscore": float(np.median(valid_zscores)) if valid_zscores else None,
            "share_above_user_review_mean": float(above_mean_count / len(query_records)),
            "share_above_user_review_p75": float(above_p75_count / len(query_records)),
            "share_below_user_review_p25": float(below_p25_count / len(query_records)),
        },
        "query_record_file": str(QUERY_RECORD_FILE),
        "user_profile_file": str(USER_PROFILE_FILE),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log("分析完成，写出 summary")
    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "num_query_rows": len(query_rows),
        "num_review_sentence_rows": len(review_rows),
        "mean_percentile": summary["query_relative_to_user_reviews"]["mean_percentile"],
        "mean_shift": summary["query_relative_to_user_reviews"]["mean_shift"],
        "share_above_user_review_mean": summary["query_relative_to_user_reviews"]["share_above_user_review_mean"],
        "share_above_user_review_p75": summary["query_relative_to_user_reviews"]["share_above_user_review_p75"],
        "share_below_user_review_p25": summary["query_relative_to_user_reviews"]["share_below_user_review_p25"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
