#!/usr/bin/env python3
"""计算 Baby_Products 每个用户评论语句的平均依存句法深度。"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path


CATEGORY = "Baby_Products"
BASE_DIR = Path("/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis") / CATEGORY
INPUT_FILE = Path("/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction") / CATEGORY / "stage1_filtered_users_reviews.json"
OUTPUT_JSON = BASE_DIR / "user_average_syntax_depth.json"
MAX_REVIEWS_PER_USER = 10
BATCH_SIZE = 256


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)
    sys.stdout.flush()


@lru_cache(maxsize=1)
def _load_spacy_model():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    for pipe_name in ("ner", "lemmatizer", "textcat", "textcat_multilabel", "senter", "sentencizer"):
        if pipe_name in nlp.pipe_names:
            nlp.remove_pipe(pipe_name)
    return nlp


def _compute_doc_syntax_tree_depth(doc) -> int:
    depth_cache = {}
    max_depth = 0

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        chain = []
        current = token
        while current.i not in depth_cache and current.head != current:
            chain.append(current)
            current = current.head

        if current.i in depth_cache:
            depth = depth_cache[current.i]
        else:
            depth = 1
            depth_cache[current.i] = depth

        for chain_token in reversed(chain):
            depth += 1
            depth_cache[chain_token.i] = depth

        token_depth = depth_cache[token.i]
        if token_depth > max_depth:
            max_depth = token_depth

    if max_depth == 0:
        raise ValueError("sentence contains no valid tokens for depth computation")
    return max_depth


def compute_sentence_syntax_tree_depth(sentence: str) -> int:
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")
    sentence = sentence.strip()
    if not sentence:
        raise ValueError("sentence must be a non-empty string")

    nlp = _load_spacy_model()
    doc = nlp(sentence)
    return _compute_doc_syntax_tree_depth(doc)


def _extract_reviews(user_entry: dict) -> list[str]:
    if "results" not in user_entry:
        raise KeyError("user entry missing results")
    reviews = []
    for idx, product in enumerate(user_entry["results"]):
        if not isinstance(product, dict):
            raise TypeError(f"results[{idx}] must be an object")
        reviews_text = product.get("target_reviews")
        if reviews_text is None:
            continue
        if not isinstance(reviews_text, list):
            raise TypeError(f"results[{idx}].target_reviews must be a list")
        for review_idx, review in enumerate(reviews_text):
            if not isinstance(review, str) or not review.strip():
                raise ValueError(f"results[{idx}].target_reviews[{review_idx}] must be a non-empty string")
            reviews.append(review.strip())
            if len(reviews) == MAX_REVIEWS_PER_USER:
                return reviews
    return reviews


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"input file not found: {INPUT_FILE}")

    payload = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    users = payload.get("users")
    if not isinstance(users, list):
        raise TypeError("stage1_filtered_users_reviews.json must contain a top-level 'users' list")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    user_records = []
    depth_hist = Counter()
    total_reviews = 0
    user_inputs = []

    for user_idx, user_entry in enumerate(users):
        if not isinstance(user_entry, dict):
            raise TypeError(f"users[{user_idx}] must be an object")
        user_id = user_entry.get("user_id")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(f"users[{user_idx}].user_id must be a non-empty string")

        reviews = _extract_reviews(user_entry)
        if not reviews:
            raise ValueError(f"user {user_id} has no target_reviews")

        user_inputs.append({"user_id": user_id, "reviews": reviews})

    nlp = _load_spacy_model()
    review_stream = (review for user_input in user_inputs for review in user_input["reviews"])
    doc_iter = nlp.pipe(review_stream, batch_size=BATCH_SIZE)

    for user_idx, user_input in enumerate(user_inputs):
        user_id = user_input["user_id"]
        reviews = user_input["reviews"]
        review_depths = []
        for _ in reviews:
            depth = _compute_doc_syntax_tree_depth(next(doc_iter))
            review_depths.append(depth)
            depth_hist[depth] += 1

        avg_depth = sum(review_depths) / len(review_depths)
        user_records.append(
            {
                "user_id": user_id,
                "review_count": len(review_depths),
                "avg_depth": round(avg_depth, 4),
                "min_depth": min(review_depths),
                "max_depth": max(review_depths),
                "depths": review_depths,
            }
        )
        total_reviews += len(review_depths)
        log(
            f"[{user_idx + 1}/{len(users)}] user_id={user_id} "
            f"review_count={len(review_depths)} avg_depth={round(avg_depth, 4)} "
            f"min_depth={min(review_depths)} max_depth={max(review_depths)}"
        )

    overall_avg = sum(record["avg_depth"] * record["review_count"] for record in user_records) / total_reviews
    summary = {
        "category": CATEGORY,
        "input_file": str(INPUT_FILE),
        "user_count": len(user_records),
        "review_count": total_reviews,
        "avg_depth": round(overall_avg, 4),
        "min_depth": min(record["min_depth"] for record in user_records),
        "max_depth": max(record["max_depth"] for record in user_records),
        "depth_histogram": dict(sorted(depth_hist.items())),
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "users": user_records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    log(f"input={INPUT_FILE}")
    log(f"output={OUTPUT_JSON}")
    log(f"user_count={summary['user_count']}")
    log(f"review_count={summary['review_count']}")
    log(f"avg_depth={summary['avg_depth']}")
    log(f"min_depth={summary['min_depth']}")
    log(f"max_depth={summary['max_depth']}")


if __name__ == "__main__":
    main()
