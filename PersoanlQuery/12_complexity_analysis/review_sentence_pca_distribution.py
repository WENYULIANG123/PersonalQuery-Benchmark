#!/usr/bin/env python3
"""Analyze Baby_Products user review sentences on a shared PCA axis."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from extract_clause_features_single_query import extract_clause_features_from_doc, load_spacy_model  # noqa: E402


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
MAX_REVIEW_SENTENCES_PER_USER = 5
MAX_USERS = None
REVIEW_SOURCE_FILE = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / CATEGORY / "stage1_filtered_users_reviews.json"
OUTPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
SUMMARY_FILE = OUTPUT_DIR / "review_sentence_pca_distribution_summary.json"
SENTENCE_RECORD_FILE = OUTPUT_DIR / "review_sentence_pca_distribution_sentences.jsonl"
USER_PROFILE_FILE = OUTPUT_DIR / "review_sentence_pca_distribution_user_profiles.jsonl"


def log(message: str) -> None:
    print(message, flush=True)


def normalize_review_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    if not text:
        raise ValueError("review 文本归一化后为空")
    return text


def sentence_has_non_root_dependency(doc) -> bool:
    tokens = [token for token in doc if not token.is_space]
    if not tokens:
        return False
    return any(token.head != token for token in tokens)


def load_user_entries() -> list[dict]:
    source = json.loads(REVIEW_SOURCE_FILE.read_text(encoding="utf-8"))
    users = source.get("users")
    if not isinstance(users, list):
        raise TypeError("stage1_filtered_users_reviews.json 顶层 users 必须是列表")
    if MAX_USERS is not None:
        users = users[:MAX_USERS]
    if not users:
        raise ValueError("没有可用用户")
    return users


def extract_target_review_texts(user_entry: dict) -> list[str]:
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


def select_review_sentences(user_entries: list[dict]) -> dict[str, list[str]]:
    nlp = load_spacy_model()
    selected_sentences = {}
    for user_entry in user_entries:
        user_id = user_entry.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("存在非法 user_id")
        review_texts = extract_target_review_texts(user_entry)
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


def extract_sentence_features(user_sentences: dict[str, list[str]]) -> tuple[list[str], list[dict]]:
    sentence_rows = []
    nlp = load_spacy_model()
    feature_names = None
    total_users = len(user_sentences)
    for user_offset, user_id in enumerate(sorted(user_sentences.keys()), start=1):
        kept_count = 0
        sentence_texts = user_sentences[user_id]
        for sentence_index, (sentence_text, doc) in enumerate(zip(sentence_texts, nlp.pipe(sentence_texts, batch_size=16))):
            if not sentence_has_non_root_dependency(doc):
                continue
            extracted = extract_clause_features_from_doc(doc, sentence_text)
            features = extracted["features"]
            if feature_names is None:
                feature_names = list(features.keys())
            elif list(features.keys()) != feature_names:
                raise ValueError("评论句子特征名不一致")
            sentence_rows.append({
                "user_id": user_id,
                "sentence_index": sentence_index,
                "sentence": sentence_text,
                "word_count": extracted["word_count"],
                "features": features,
            })
            kept_count += 1
        if kept_count == 0:
            raise ValueError(f"user {user_id} 在过滤退化句后没有保留任何评论句子")
        log(f"已完成用户 {user_offset}/{total_users}: {user_id}, 保留句子数={kept_count}")

    if feature_names is None:
        raise ValueError("没有成功抽取任何评论句子特征")
    return feature_names, sentence_rows


def fit_pca(feature_names: list[str], sentence_rows: list[dict]) -> tuple[StandardScaler, PCA, float]:
    matrix = np.array(
        [[float(row["features"][name]) for name in feature_names] for row in sentence_rows],
        dtype=float,
    )
    if len(matrix) < 2:
        raise ValueError("评论句子数量不足，无法拟合 PCA")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)
    pca = PCA(n_components=1, random_state=42)
    scores = pca.fit_transform(standardized).ravel()
    anchor = np.array([float(row["features"]["max_dependency_depth"]) for row in sentence_rows], dtype=float)
    corr = np.corrcoef(scores, anchor)[0, 1]
    if np.isnan(corr):
        raise ValueError("评论 PCA 分数与 anchor 的相关性为 NaN")
    sign = 1.0 if corr >= 0 else -1.0
    return scaler, pca, sign


def project_scores(feature_names: list[str], sentence_rows: list[dict], scaler: StandardScaler, pca: PCA, sign: float) -> np.ndarray:
    matrix = np.array(
        [[float(row["features"][name]) for name in feature_names] for row in sentence_rows],
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
    log("开始读取用户评论数据")
    user_entries = load_user_entries()
    log(f"用户数: {len(user_entries)}")
    log("开始按用户抽取评论句子")
    user_sentences = select_review_sentences(user_entries)
    log(f"有评论句子的用户数: {len(user_sentences)}")
    log("开始抽取评论句子特征")
    feature_names, sentence_rows = extract_sentence_features(user_sentences)
    log(f"评论句子特征行数: {len(sentence_rows)}")
    log("开始拟合 PCA")
    scaler, pca, sign = fit_pca(feature_names, sentence_rows)
    scores = project_scores(feature_names, sentence_rows, scaler, pca, sign)

    for row, score in zip(sentence_rows, scores):
        row["pca_score"] = float(score)

    score_by_user = defaultdict(list)
    for row in sentence_rows:
        score_by_user[row["user_id"]].append(float(row["pca_score"]))

    user_profiles = []
    for user_id in sorted(score_by_user.keys()):
        score_array = np.array(score_by_user[user_id], dtype=float)
        user_profiles.append({
            "user_id": user_id,
            "review_sentence_count": len(score_array),
            "pca_score_summary": summarize_array(score_array),
            "pca_score_mean": float(np.mean(score_array)),
            "pca_score_std": float(np.std(score_array)),
        })

    all_scores = np.array(scores, dtype=float)
    summary = {
        "category": CATEGORY,
        "review_source_file": str(REVIEW_SOURCE_FILE),
        "max_users": MAX_USERS,
        "max_review_sentences_per_user": MAX_REVIEW_SENTENCES_PER_USER,
        "num_users": len(user_entries),
        "num_users_with_sentences": len(user_sentences),
        "num_sentence_rows": len(sentence_rows),
        "feature_names": feature_names,
        "pooled_pca_score_summary": summarize_array(all_scores),
        "user_sentence_count_summary": summarize_array(np.array([len(v) for v in score_by_user.values()], dtype=float)),
        "sentence_record_file": str(SENTENCE_RECORD_FILE),
        "user_profile_file": str(USER_PROFILE_FILE),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SENTENCE_RECORD_FILE.open("w", encoding="utf-8") as handle:
        for row in sentence_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    with USER_PROFILE_FILE.open("w", encoding="utf-8") as handle:
        for row in user_profiles:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("分析完成，写出 summary")
    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "num_users": summary["num_users"],
        "num_sentence_rows": summary["num_sentence_rows"],
        "pooled_mean": summary["pooled_pca_score_summary"]["mean"],
        "pooled_std": summary["pooled_pca_score_summary"]["std"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
