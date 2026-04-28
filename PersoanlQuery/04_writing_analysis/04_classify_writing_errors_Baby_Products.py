#!/usr/bin/env python3
"""Use spaCy syntax only to classify writing errors as ACL/CCOMP.

Input:
  /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/Baby_Products/writing_error.json

Output:
  /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/Baby_Products/acl_ccomp_error.json
"""

import json
import re
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import spacy


CATEGORY = "Baby_Products"
WRITING_ERROR_FILE = Path(f"/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/{CATEGORY}/writing_error.json")
STAGE1_REVIEWS_FILE = Path(f"/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/{CATEGORY}/stage1_filtered_users_reviews.json")
OUTPUT_FILE = Path(f"/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/{CATEGORY}/acl_ccomp_error.json")

ACL_HEAD_DEPS = {"acl", "relcl"}
CCOMP_HEAD_DEPS = {"ccomp"}
ACL_MODIFIER_DEPS = {"amod", "acomp", "advmod", "oprd"}
ACL_NP_DEPS = {"compound", "nmod", "poss", "nsubj", "dobj", "obj", "pobj", "attr"}

_NLP = None
_NLP_LOCK = threading.Lock()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_nlp():
    global _NLP
    if _NLP is None:
        with _NLP_LOCK:
            if _NLP is None:
                _NLP = spacy.load("en_core_web_sm")
    return _NLP


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def validate_word_error(original: str, corrected: str) -> Tuple[bool, str]:
    orig = original.lower().strip()
    corr = corrected.lower().strip()
    if not orig or not corr:
        return False, "empty_error"
    if orig == corr:
        return False, "case_or_identity_error"
    if len(orig.split()) != 1 or len(corr.split()) != 1:
        return False, "non_single_word_error"
    return True, "valid"


def replace_single_word(sentence_text: str, original: str, corrected: str) -> Tuple[str, int]:
    pattern = re.compile(rf"\b{re.escape(original)}\b", flags=re.IGNORECASE)
    match = pattern.search(sentence_text)
    if match is None:
        raise ValueError(f"Original word {original!r} not found in sentence: {sentence_text}")
    replaced = sentence_text[:match.start()] + corrected + sentence_text[match.end():]
    return replaced, match.start()


def locate_sentence(nlp, review_text: str, span_text: str, original: str) -> Optional[str]:
    doc = nlp(review_text)
    span = normalize_space(span_text)
    if span:
        match = re.search(re.escape(span), review_text, flags=re.IGNORECASE)
        if match is not None:
            offset = match.start()
            end = match.end()
            for sent in doc.sents:
                if sent.start_char <= offset < sent.end_char or sent.start_char < end <= sent.end_char:
                    return sent.text

    pattern = re.compile(rf"\b{re.escape(original)}\b", flags=re.IGNORECASE)
    match = pattern.search(review_text)
    if match is None:
        return None
    for sent in doc.sents:
        if sent.start_char <= match.start() < sent.end_char:
            return sent.text
    return None


def locate_anchor_token(doc, corrected: str, replace_char: int):
    corrected_lower = corrected.lower().strip()
    candidates = [token for token in doc if token.text.lower() == corrected_lower]
    if not candidates:
        return None
    return min(candidates, key=lambda token: abs(token.idx - replace_char))


def token_in_subtree(anchor, head) -> bool:
    return anchor == head or anchor in list(head.subtree)


def classify_by_spacy(review_text: str, span_text: str, original: str, corrected: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    nlp = load_nlp()
    sentence_text = locate_sentence(nlp, review_text, span_text, original)
    if sentence_text is None:
        return None, None, None, "syntax_sentence_not_found"

    try:
        corrected_sentence, replace_char = replace_single_word(sentence_text, original, corrected)
    except ValueError:
        return None, None, None, "syntax_original_not_found"
    doc = nlp(corrected_sentence)
    anchor = locate_anchor_token(doc, corrected, replace_char)
    if anchor is None:
        return None, None, None, "syntax_anchor_not_found"

    ccomp_heads = [token for token in doc if token.dep_ in CCOMP_HEAD_DEPS]
    if anchor.dep_ == "mark" and anchor.head.dep_ in CCOMP_HEAD_DEPS:
        return "ccomp", "complement_link", "complement_linking_error", "ok"

    for head in ccomp_heads:
        if anchor.tag_ == "MD" and token_in_subtree(anchor, head):
            return "ccomp", "modal", "modal_distortion", "ok"
        if head.head == anchor or anchor in list(head.ancestors):
            return "ccomp", "ccomp", "clause_shell_typo", "ok"
        if token_in_subtree(anchor, head):
            return "ccomp", "ccomp", "clause_boundary_error", "ok"

    acl_heads = [token for token in doc if token.dep_ in ACL_HEAD_DEPS]
    for head in acl_heads:
        if not token_in_subtree(anchor, head):
            continue
        if anchor == head:
            return "acl", head.dep_, "modifier_typo", "ok"
        if anchor.pos_ in {"ADJ", "ADV"} and anchor.dep_ in ACL_MODIFIER_DEPS:
            return "acl", head.dep_, "modifier_typo", "ok"
        if anchor.pos_ in {"NOUN", "PROPN", "PRON"} and anchor.dep_ in ACL_NP_DEPS:
            return "acl", head.dep_, "np_inflection", "ok"

    return None, None, None, "syntax_no_acl_ccomp_match"


def load_review_map() -> Dict[str, List[str]]:
    if not STAGE1_REVIEWS_FILE.exists():
        raise FileNotFoundError(f"Stage 1 reviews file not found: {STAGE1_REVIEWS_FILE}")
    with STAGE1_REVIEWS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "users" not in data or not isinstance(data["users"], list):
        raise ValueError(f"Invalid Stage 1 review file structure: {STAGE1_REVIEWS_FILE}")

    review_map = {}
    for user in data["users"]:
        user_id = user.get("user_id")
        if not user_id:
            continue
        flattened_reviews = []
        for product in user.get("results", []):
            for review_text in product.get("target_reviews", []):
                if isinstance(review_text, str):
                    flattened_reviews.append(review_text)
        review_map[user_id] = flattened_reviews
    return review_map


def get_review_text(review_map: Dict[str, List[str]], user_id: str, review_index: int) -> str:
    if user_id not in review_map:
        raise ValueError(f"User {user_id} not found in Stage 1 review map")
    reviews = review_map[user_id]
    if review_index < 0 or review_index >= len(reviews):
        raise ValueError(f"Review index {review_index} out of range for user {user_id}")
    return reviews[review_index]


def classify_detail(user_id: str, detail: Dict, review_map: Dict[str, List[str]]) -> Tuple[List[Dict], Counter]:
    if "asin" not in detail:
        raise ValueError(f"Missing asin in detail: {detail}")
    if "review_index" not in detail:
        raise ValueError(f"Missing review_index in detail: {detail}")
    if "errors" not in detail or not isinstance(detail["errors"], list):
        raise ValueError(f"Missing errors list in detail: {detail}")

    review_text = get_review_text(review_map, user_id, detail["review_index"])
    grouped: Dict[Tuple[str, str, str, str], List[Dict]] = {}
    filtered_counts = Counter()
    for error in detail["errors"]:
        for required_key in ("original", "corrected", "confidence"):
            if required_key not in error:
                raise ValueError(f"Missing {required_key} in error: {error}")

        is_valid, reason = validate_word_error(error["original"], error["corrected"])
        if not is_valid:
            filtered_counts[reason] += 1
            continue

        span_text = error.get("span_text", "")
        category, region_type, error_type, reason = classify_by_spacy(
            review_text=review_text,
            span_text=span_text,
            original=error["original"],
            corrected=error["corrected"],
        )
        if category is None:
            filtered_counts[reason] += 1
            continue

        key = (category, region_type, error_type, span_text)
        grouped.setdefault(key, []).append({
            "original": error["original"],
            "corrected": error["corrected"],
            "error_type": error_type,
            "confidence": error["confidence"],
        })

    classified_details = []
    for (category, region_type, error_type, span_text), errors in grouped.items():
        item = {
            "asin": detail["asin"],
            "review_index": detail["review_index"],
            "error_category": category,
            "region_type": region_type,
            "error_type": error_type,
            "errors": errors,
        }
        if span_text:
            item["span_text"] = span_text
        classified_details.append(item)
    return classified_details, filtered_counts


def classify_user(row: Dict, review_map: Dict[str, List[str]]) -> Dict:
    for key in ("user_id", "status", "reviews_processed", "detailed_results"):
        if key not in row:
            raise ValueError(f"Missing {key} in row: {row}")
    if not isinstance(row["detailed_results"], list):
        raise ValueError(f"detailed_results must be a list for user {row['user_id']}")

    acl_error_types = Counter()
    ccomp_error_types = Counter()
    acl_region_types = Counter()
    ccomp_region_types = Counter()
    filtered_counts = Counter()
    detailed_results = []

    for detail in row["detailed_results"]:
        classified_details, detail_filtered_counts = classify_detail(row["user_id"], detail, review_map)
        filtered_counts.update(detail_filtered_counts)
        for classified_detail in classified_details:
            count = len(classified_detail["errors"])
            if classified_detail["error_category"] == "acl":
                acl_error_types[classified_detail["error_type"]] += count
                acl_region_types[classified_detail["region_type"]] += count
            elif classified_detail["error_category"] == "ccomp":
                ccomp_error_types[classified_detail["error_type"]] += count
                ccomp_region_types[classified_detail["region_type"]] += count
            else:
                raise ValueError(f"Unsupported error_category: {classified_detail['error_category']}")
            detailed_results.append(classified_detail)

    acl_count = sum(acl_error_types.values())
    ccomp_count = sum(ccomp_error_types.values())
    return {
        "user_id": row["user_id"],
        "status": row["status"],
        "reviews_processed": row["reviews_processed"],
        "acl_error_count": acl_count,
        "ccomp_error_count": ccomp_count,
        "total_errors": acl_count + ccomp_count,
        "acl_error_types": dict(acl_error_types),
        "ccomp_error_types": dict(ccomp_error_types),
        "acl_region_types": dict(acl_region_types),
        "ccomp_region_types": dict(ccomp_region_types),
        "filtered_counts": dict(filtered_counts),
        "detailed_results": detailed_results,
    }


def main() -> None:
    if not WRITING_ERROR_FILE.exists():
        raise FileNotFoundError(f"Writing error file not found: {WRITING_ERROR_FILE}")

    with WRITING_ERROR_FILE.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Input must be a JSON list: {WRITING_ERROR_FILE}")

    review_map = load_review_map()
    output_rows = [classify_user(row, review_map) for row in rows]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)

    total_acl = sum(row["acl_error_count"] for row in output_rows)
    total_ccomp = sum(row["ccomp_error_count"] for row in output_rows)
    filtered_counts = Counter()
    for row in output_rows:
        filtered_counts.update(row["filtered_counts"])
    users_with_errors = sum(1 for row in output_rows if row["total_errors"] > 0)

    log(f"Input users: {len(rows)}")
    log(f"Users with ACL/CCOMP errors: {users_with_errors}")
    log(f"ACL errors: {total_acl}")
    log(f"CCOMP errors: {total_ccomp}")
    if filtered_counts:
        log(f"Filtered outputs: {dict(filtered_counts)}")
    log(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
