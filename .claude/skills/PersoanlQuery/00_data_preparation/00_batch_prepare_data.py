#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set


def log_with_timestamp(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def select_users_by_long_reviews(review_file: str, min_words: int, min_long_reviews: int) -> Dict[str, int]:
    long_review_counts: Dict[str, int] = defaultdict(int)

    log_with_timestamp("Scanning reviews for long-review user filtering...")
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

            if count_words(review_text) >= min_words:
                long_review_counts[user_id] += 1

    selected = {
        user_id: count
        for user_id, count in long_review_counts.items()
        if count >= min_long_reviews
    }

    log_with_timestamp(f"Users with >= {min_long_reviews} reviews of >= {min_words} words: {len(selected)}")
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
) -> None:
    results = []
    for asin in sorted(user_asins):
        target_reviews = []

        for review in asin_reviews.get(asin, []):
            text = review.get("reviewText", "")
            reviewer_id = review.get("reviewerID", "")
            if not text:
                continue

            if reviewer_id == user_id:
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
    parser = argparse.ArgumentParser(description="Stage 0: User selection by long-review count")
    parser.add_argument("--review-file", required=True, help="Path to Amazon review JSON.GZ")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--min-words", type=int, default=25, help="Min words per review")
    parser.add_argument("--min-long-reviews", type=int, default=20, help="Min long-review count per user")
    parser.add_argument("--max-other-reviews", type=int, default=20, help="Deprecated and ignored; other-user reviews are not saved")
    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        log_with_timestamp(f"Removing existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 0: Batch Data Preparation (Long-Review User Filtering)")
    log_with_timestamp("=" * 80)

    selected_user_counts = select_users_by_long_reviews(
        args.review_file,
        args.min_words,
        args.min_long_reviews,
    )

    selected_user_ids = set(selected_user_counts.keys())
    if not selected_user_ids:
        log_with_timestamp("No users matched criteria. Exiting.")
        return

    user_products, user_product_titles, asin_reviews = collect_reviews_for_selected_users(
        args.review_file,
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
            output_dir=args.output_dir,
        )

    selected_users_file = os.path.join(args.output_dir, "selected_users.json")
    selected_users = [
        {
            "user_id": user_id,
            "long_review_count": selected_user_counts[user_id],
            "product_count": len(user_products.get(user_id, set())),
        }
        for user_id in sorted(selected_user_ids)
    ]
    with open(selected_users_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "selection_criteria": {
                    "min_words": args.min_words,
                    "min_long_reviews": args.min_long_reviews,
                    "max_other_reviews": args.max_other_reviews,
                    "save_other_reviews": False,
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
    log_with_timestamp(f"Output directory: {args.output_dir}")
    log_with_timestamp(f"User list: {selected_users_file}")
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    main()
