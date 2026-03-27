#!/usr/bin/env python3
import gzip
import json
import os
import shutil
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set


def log_with_timestamp(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def has_long_sentence(text: str, min_words: int) -> bool:
    """Check if text contains at least one sentence with >= min_words words."""
    if not text:
        return False
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        if len(sent.split()) >= min_words:
            return True
    return False


def count_long_sentences(text: str, min_words: int) -> int:
    """Count how many sentences have >= min_words words."""
    if not text:
        return 0
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sum(1 for sent in sentences if len(sent.split()) >= min_words)


def select_users_by_long_reviews(review_file: str, min_words: int, min_long_sentences: int) -> Dict[str, int]:
    """
    统计每个用户有多少个长句子（词数 >= min_words）。
    返回满足 min_long_sentences 条件的用户及其长句子总数。
    """
    user_long_sentence_counts: Dict[str, int] = defaultdict(int)

    log_with_timestamp("Scanning reviews for long-sentence user filtering...")
    with gzip.open(review_file, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue

            user_id = review.get("reviewerID")
            review_text = review.get("reviewText", "")
            if not user_id:
                continue

            # Count how many sentences in this review have >= min_words words
            num_long_sents = count_long_sentences(review_text, min_words)
            user_long_sentence_counts[user_id] += num_long_sents

    selected = {
        user_id: count
        for user_id, count in user_long_sentence_counts.items()
        if count >= min_long_sentences
    }

    log_with_timestamp(f"Users with >= {min_long_sentences} long sentences (>= {min_words} words each): {len(selected)}")
    return selected


def collect_reviews_for_selected_users(review_file: str, selected_user_ids: Set[str]) -> tuple:
    user_products: Dict[str, Set[str]] = defaultdict(set)
    user_product_titles: Dict[str, Dict[str, str]] = defaultdict(dict)
    asin_reviews: Dict[str, List[Dict]] = defaultdict(list)

    log_with_timestamp("Collecting product and review mappings for selected users...")
    with gzip.open(review_file, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = review.get("asin")
            reviewer_id = review.get("reviewerID")
            if not asin or not reviewer_id:
                continue

            asin_reviews[asin].append(review)

            if reviewer_id in selected_user_ids:
                user_products[reviewer_id].add(asin)
                if asin not in user_product_titles[reviewer_id]:
                    user_product_titles[reviewer_id][asin] = review.get("title", "")

    return user_products, user_product_titles, asin_reviews


def build_user_output(
    user_id: str,
    user_asins: Set[str],
    user_product_titles: Dict[str, str],
    asin_reviews: Dict[str, List[Dict]],
    output_dir: str,
    min_words: int = 25,
) -> None:
    """
    收集用户的所有长句子（词数 >= min_words），按商品组织输出。
    输出格式兼容 Stage 5：使用 target_reviews 字段存储完整评论文本。
    """
    results = []
    for asin in sorted(user_asins):
        target_reviews = []

        for review in asin_reviews.get(asin, []):
            text = review.get("reviewText", "")
            reviewer_id = review.get("reviewerID", "")
            if not text:
                continue

            if reviewer_id == user_id:
                # Check if this review has at least one sentence with >= min_words
                if has_long_sentence(text, min_words):
                    target_reviews.append(text)

        if not target_reviews:
            continue

        result = {
            "asin": asin,
            "product_title": user_product_titles.get(asin, ""),
            "target_user_id": user_id,
            "target_reviews_count": len(target_reviews),
            "target_reviews": target_reviews,
            "other_reviews_count": 0,
            "other_reviews": [],
        }
        results.append(result)

    output_data = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "total_products": len(results),
        "results": results,
    }

    output_file = os.path.join(output_dir, f"reviews_{user_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def main() -> None:
    # ============ 硬编码参数 ============
    REVIEW_FILE = "/fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz"
    OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/00_data_preparation"
    MIN_WORDS = 25            # 每句话最少词数
    MIN_LONG_SENTENCES = 20  # 每个用户最少长句子数（词数 >= MIN_WORDS）
    MAX_USERS = 10000         # 最大用户数（0表示不限制）

    if os.path.isdir(OUTPUT_DIR):
        log_with_timestamp(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 0: Batch Data Preparation (Long-Sentence User Filtering)")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Review file: {REVIEW_FILE}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    log_with_timestamp(f"Min words per sentence: {MIN_WORDS}")
    log_with_timestamp(f"Min long-sentence count per user: {MIN_LONG_SENTENCES}")
    log_with_timestamp(f"Max users: {MAX_USERS if MAX_USERS > 0 else 'unlimited'}")

    selected_user_counts = select_users_by_long_reviews(
        REVIEW_FILE,
        MIN_WORDS,
        MIN_LONG_SENTENCES,
    )

    selected_user_ids = set(selected_user_counts.keys())
    if not selected_user_ids:
        log_with_timestamp("No users matched criteria. Exiting.")
        return

    # Limit to MAX_USERS if specified
    if MAX_USERS > 0 and len(selected_user_ids) > MAX_USERS:
        selected_user_ids = set(list(sorted(selected_user_ids))[:MAX_USERS])
        log_with_timestamp(f"Limited to {MAX_USERS} users")

    user_products, user_product_titles, asin_reviews = collect_reviews_for_selected_users(
        REVIEW_FILE,
        selected_user_ids,
    )

    log_with_timestamp(f"Preparing output for {len(selected_user_ids)} users...")
    for idx, user_id in enumerate(sorted(selected_user_ids), start=1):
        log_with_timestamp(f"[{idx}/{len(selected_user_ids)}] Processing user {user_id}...")
        build_user_output(
            user_id=user_id,
            user_asins=user_products.get(user_id, set()),
            user_product_titles=user_product_titles.get(user_id, {}),
            asin_reviews=asin_reviews,
            output_dir=OUTPUT_DIR,
            min_words=MIN_WORDS,
        )

    selected_users_file = os.path.join(OUTPUT_DIR, "selected_users.json")
    selected_users = [
        {
            "user_id": user_id,
            "long_sentence_count": selected_user_counts[user_id],
            "product_count": len(user_products.get(user_id, set())),
        }
        for user_id in sorted(selected_user_ids)
    ]
    with open(selected_users_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "selection_criteria": {
                    "min_words": MIN_WORDS,
                    "min_long_sentences": MIN_LONG_SENTENCES,
                    "max_users": MAX_USERS,
                },
                "total_selected": len(selected_users),
                "selected_users": selected_users,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 0 Complete")
    log_with_timestamp(f"Selected users: {len(selected_users)}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    log_with_timestamp(f"User list: {selected_users_file}")
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    main()
