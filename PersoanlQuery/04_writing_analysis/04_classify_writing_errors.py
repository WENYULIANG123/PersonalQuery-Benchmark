#!/usr/bin/env python3
"""Use spaCy syntax only to classify writing errors as ACL/CCOMP.

Input:
  /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/{category}/writing_error.json

Output:
  /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/{category}/acl_ccomp_error.json
"""

import json
import re
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import spacy


CATEGORIES = [
    "Baby_Products",
    "Grocery_and_Gourmet_Food",
    "Pet_Supplies",
]
WRITING_ANALYSIS_ROOT = Path("/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis")
STAGE1_ROOT = Path("/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction")
SYNTACTIC_ANALYSIS_ROOT = Path("/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis")
PROGRESS_INTERVAL_USERS = 25
STRUCTURE_MATCH_WINDOW = 2

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


def get_category_paths(category: str) -> Tuple[Path, Path, Path]:
    writing_error_file = WRITING_ANALYSIS_ROOT / category / "writing_error.json"
    stage1_reviews_file = STAGE1_ROOT / category / "stage1_filtered_users_reviews.json"
    output_file = WRITING_ANALYSIS_ROOT / category / "acl_ccomp_error.json"
    return writing_error_file, stage1_reviews_file, output_file


def get_syntax_cache_paths(category: str) -> Tuple[Path, Path]:
    category_dir = SYNTACTIC_ANALYSIS_ROOT / category
    return category_dir / "acl_sentences.jsonl", category_dir / "ccomp_sentences.jsonl"


def cache_key(user_id: str, review_text: str) -> Tuple[str, str]:
    return user_id, normalize_space(review_text)


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required Stage 5 syntax cache file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def load_syntax_cache(category: str) -> Dict[Tuple[str, str], Dict[str, List[Dict]]]:
    acl_file, ccomp_file = get_syntax_cache_paths(category)
    syntax_cache: Dict[Tuple[str, str], Dict[str, List[Dict]]] = {}

    log(f"[{category}] Loading Stage 5 ACL cache from: {acl_file}")
    acl_rows = 0
    for row in load_jsonl(acl_file):
        if "user_id" not in row or "sentence" not in row or "acl_info" not in row:
            raise ValueError(f"Invalid ACL syntax cache row: {row}")
        key = cache_key(row["user_id"], row["sentence"])
        syntax_cache.setdefault(key, {"acl_info": [], "ccomp_info": []})["acl_info"] = row["acl_info"]
        acl_rows += 1

    log(f"[{category}] Loading Stage 5 CCOMP cache from: {ccomp_file}")
    ccomp_rows = 0
    for row in load_jsonl(ccomp_file):
        if "user_id" not in row or "sentence" not in row or "ccomp_info" not in row:
            raise ValueError(f"Invalid CCOMP syntax cache row: {row}")
        key = cache_key(row["user_id"], row["sentence"])
        syntax_cache.setdefault(key, {"acl_info": [], "ccomp_info": []})["ccomp_info"] = row["ccomp_info"]
        ccomp_rows += 1

    log(f"[{category}] Loaded Stage 5 syntax cache: acl_rows={acl_rows}, ccomp_rows={ccomp_rows}, merged_reviews={len(syntax_cache)}")
    return syntax_cache


def load_review_map(stage1_reviews_file: Path) -> Dict[str, List[str]]:
    if not stage1_reviews_file.exists():
        raise FileNotFoundError(f"Stage 1 reviews file not found: {stage1_reviews_file}")
    with stage1_reviews_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "users" not in data or not isinstance(data["users"], list):
        raise ValueError(f"Invalid Stage 1 review file structure: {stage1_reviews_file}")

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


def format_elapsed(start_time: float) -> str:
    elapsed_seconds = int(time.time() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_review_text(review_map: Dict[str, List[str]], user_id: str, review_index: int) -> str:
    if user_id not in review_map:
        raise ValueError(f"User {user_id} not found in Stage 1 review map")
    reviews = review_map[user_id]
    if review_index < 0 or review_index >= len(reviews):
        raise ValueError(f"Review index {review_index} out of range for user {user_id}")
    return reviews[review_index]


def find_text_span(text: str, pattern_text: str) -> Optional[Tuple[int, int]]:
    if not pattern_text:
        return None
    match = re.search(re.escape(pattern_text), text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.start(), match.end()


def token_indices_in_char_span(review_text: str, start_char: int, end_char: int) -> List[int]:
    doc = load_nlp().make_doc(review_text)
    indices = []
    for token in doc:
        token_end = token.idx + len(token.text)
        if token.idx < end_char and token_end > start_char:
            indices.append(token.i)
    return indices


def locate_error_token_indices(review_text: str, span_text: str, original: str) -> Tuple[Optional[List[int]], str]:
    span = find_text_span(review_text, normalize_space(span_text))
    if span is not None:
        span_start, span_end = span
        original_match = re.search(re.escape(original), review_text[span_start:span_end], flags=re.IGNORECASE)
        if original_match is not None:
            start_char = span_start + original_match.start()
            end_char = span_start + original_match.end()
            indices = token_indices_in_char_span(review_text, start_char, end_char)
            if indices:
                return indices, "ok"
        indices = token_indices_in_char_span(review_text, span_start, span_end)
        if indices:
            return indices, "ok"

    pattern = re.compile(rf"\b{re.escape(original)}\b", flags=re.IGNORECASE)
    match = pattern.search(review_text)
    if match is None:
        return None, "syntax_original_not_found"
    indices = token_indices_in_char_span(review_text, match.start(), match.end())
    if not indices:
        return None, "syntax_anchor_not_found"
    return indices, "ok"


def is_modal_word(word: str) -> bool:
    return word.lower().strip() in {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}


def expanded_index_set(indices: List[int]) -> set:
    expanded = set()
    for index in indices:
        for candidate in range(index - STRUCTURE_MATCH_WINDOW, index + STRUCTURE_MATCH_WINDOW + 1):
            expanded.add(candidate)
    return expanded


def closest_position_match(indices: List[int], infos: List[Dict]) -> Optional[Dict]:
    if not infos:
        return None
    target_positions = expanded_index_set(indices)
    positioned_infos = [info for info in infos if isinstance(info.get("position"), int)]
    for info in positioned_infos:
        if info["position"] in target_positions:
            return info
    return None


def classify_from_stage5_cache(
    user_id: str,
    review_text: str,
    span_text: str,
    original: str,
    corrected: str,
    syntax_cache: Dict[Tuple[str, str], Dict[str, List[Dict]]],
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    key = cache_key(user_id, review_text)
    if key not in syntax_cache:
        raise ValueError(f"Stage 5 syntax cache missing review for user={user_id}")

    token_indices, reason = locate_error_token_indices(review_text, span_text, original)
    if token_indices is None:
        return None, None, None, reason

    syntax_info = syntax_cache[key]
    ccomp_match = closest_position_match(token_indices, syntax_info["ccomp_info"])
    if ccomp_match is not None:
        comp_type = ccomp_match.get("comp_type")
        if comp_type and str(comp_type).startswith("mark_"):
            return "ccomp", "complement_link", "complement_linking_error", "ok"
        if is_modal_word(original) or is_modal_word(corrected):
            return "ccomp", "modal", "modal_distortion", "ok"
        return "ccomp", "ccomp", "clause_boundary_error", "ok"

    acl_match = closest_position_match(token_indices, syntax_info["acl_info"])
    if acl_match is not None:
        acl_type = acl_match.get("acl_type")
        region_type = "relcl" if acl_type == "relcl_reference" else "acl"
        if original.lower().strip().rstrip("s") == corrected.lower().strip().rstrip("s"):
            error_type = "np_inflection"
        else:
            error_type = "modifier_typo"
        return "acl", region_type, error_type, "ok"

    return None, None, None, "syntax_no_acl_ccomp_match"


def classify_detail(
    user_id: str,
    detail: Dict,
    review_map: Dict[str, List[str]],
    syntax_cache: Dict[Tuple[str, str], Dict[str, List[Dict]]],
) -> Tuple[List[Dict], Counter]:
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
        category, region_type, error_type, reason = classify_from_stage5_cache(
            user_id=user_id,
            review_text=review_text,
            span_text=span_text,
            original=error["original"],
            corrected=error["corrected"],
            syntax_cache=syntax_cache,
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


def classify_user(
    row: Dict,
    review_map: Dict[str, List[str]],
    syntax_cache: Dict[Tuple[str, str], Dict[str, List[Dict]]],
) -> Dict:
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
        classified_details, detail_filtered_counts = classify_detail(row["user_id"], detail, review_map, syntax_cache)
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


def user_has_errors(row: Dict) -> bool:
    detailed_results = row.get("detailed_results")
    if not isinstance(detailed_results, list):
        return False
    for detail in detailed_results:
        errors = detail.get("errors") if isinstance(detail, dict) else None
        if isinstance(errors, list) and errors:
            return True
    return False


def empty_classified_user(row: Dict) -> Dict:
    for key in ("user_id", "status", "reviews_processed", "detailed_results"):
        if key not in row:
            raise ValueError(f"Missing {key} in row: {row}")
    if not isinstance(row["detailed_results"], list):
        raise ValueError(f"detailed_results must be a list for user {row['user_id']}")
    return {
        "user_id": row["user_id"],
        "status": row["status"],
        "reviews_processed": row["reviews_processed"],
        "acl_error_count": 0,
        "ccomp_error_count": 0,
        "total_errors": 0,
        "acl_error_types": {},
        "ccomp_error_types": {},
        "acl_region_types": {},
        "ccomp_region_types": {},
        "filtered_counts": {},
        "detailed_results": [],
    }


def classify_category(category: str) -> None:
    start_time = time.time()
    writing_error_file, stage1_reviews_file, output_file = get_category_paths(category)
    log(f"=== Classifying category: {category} ===")
    log(f"[{category}] Reading writing errors from: {writing_error_file}")

    if not writing_error_file.exists():
        raise FileNotFoundError(f"Writing error file not found: {writing_error_file}")

    with writing_error_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Input must be a JSON list: {writing_error_file}")
    log(f"[{category}] Loaded {len(rows)} users from writing_error.json")
    rows_with_errors = [row for row in rows if user_has_errors(row)]
    rows_without_errors = len(rows) - len(rows_with_errors)
    log(
        f"[{category}] Users requiring syntax parsing: {len(rows_with_errors)}; "
        f"users skipped with no errors: {rows_without_errors}"
    )

    log(f"[{category}] Loading Stage 1 review map from: {stage1_reviews_file}")
    review_map = load_review_map(stage1_reviews_file)
    log(f"[{category}] Loaded review map for {len(review_map)} users")
    syntax_cache = load_syntax_cache(category)

    output_rows = []
    running_acl = 0
    running_ccomp = 0
    running_filtered_counts = Counter()
    parsed_users = 0
    total_parse_users = len(rows_with_errors)
    for index, row in enumerate(rows, start=1):
        did_parse = False
        if user_has_errors(row):
            output_row = classify_user(row, review_map, syntax_cache)
            parsed_users += 1
            did_parse = True
        else:
            output_row = empty_classified_user(row)
        output_rows.append(output_row)
        running_acl += output_row["acl_error_count"]
        running_ccomp += output_row["ccomp_error_count"]
        running_filtered_counts.update(output_row["filtered_counts"])

        if (
            total_parse_users == 0
            or (did_parse and parsed_users == 1)
            or (did_parse and parsed_users % PROGRESS_INTERVAL_USERS == 0)
            or index == len(rows)
        ):
            parsed_percent = parsed_users / total_parse_users if total_parse_users else 1.0
            log(
                f"[{category}] Parsed {parsed_users}/{total_parse_users} users with errors "
                f"({parsed_percent:.2%}); "
                f"scanned={index}/{len(rows)}; skipped_no_errors={index - parsed_users}; "
                f"ACL={running_acl}; CCOMP={running_ccomp}; "
                f"filtered={sum(running_filtered_counts.values())}; "
                f"elapsed={format_elapsed(start_time)}"
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_rows_with_errors = [row for row in output_rows if row["total_errors"] > 0]
    users_without_acl_ccomp_errors = len(output_rows) - len(output_rows_with_errors)
    log(f"[{category}] Writing classified output to: {output_file}")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_rows_with_errors, f, ensure_ascii=False, indent=2)

    total_acl = sum(row["acl_error_count"] for row in output_rows)
    total_ccomp = sum(row["ccomp_error_count"] for row in output_rows)
    filtered_counts = Counter()
    for row in output_rows:
        filtered_counts.update(row["filtered_counts"])
    users_with_errors = sum(1 for row in output_rows if row["total_errors"] > 0)

    log(f"Input users: {len(rows)}")
    log(f"Users with raw writing errors: {len(rows_with_errors)}")
    log(f"Users without raw writing errors: {rows_without_errors}")
    log(f"Users with ACL/CCOMP errors: {users_with_errors}")
    log(f"Users omitted from output without ACL/CCOMP errors: {users_without_acl_ccomp_errors}")
    log(f"ACL errors: {total_acl}")
    log(f"CCOMP errors: {total_ccomp}")
    if filtered_counts:
        log(f"Filtered outputs: {dict(filtered_counts)}")
    log(f"Output written to: {output_file}")
    log(f"=== Finished category: {category}; elapsed={format_elapsed(start_time)} ===")


def main() -> None:
    for category in CATEGORIES:
        classify_category(category)


if __name__ == "__main__":
    main()
