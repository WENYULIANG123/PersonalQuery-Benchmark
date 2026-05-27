#!/usr/bin/env python3
"""Common functions for classifying writing errors as ACL/CCOMP."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


WRITING_ANALYSIS_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/04_writing_analysis")
STAGE1_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/01_preference_extraction")
SYNTACTIC_ANALYSIS_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis")
PROGRESS_INTERVAL_USERS = 25


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def normalize_space(text: str) -> str:
    return " ".join(text.split())


# 常见英语词缀
COMMON_AFFIXES = ['s', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'd', 'en', 'n']


def is_affix_variation(original: str, corrected: str) -> bool:
    """检查两个词是否仅仅是词缀变化（如 dog->dogs, jump->jumped）"""
    import string
    orig = original.lower().strip()
    corr = corrected.lower().strip()

    if orig == corr:
        return False

    # 过滤仅相差标点的错误
    PUNCT = set(string.punctuation)
    orig_no_punct = ''.join(c for c in orig if c not in PUNCT)
    corr_no_punct = ''.join(c for c in corr if c not in PUNCT)
    if orig_no_punct == corr_no_punct and orig != corr:
        return True

    # 过滤仅相差单引号的错误
    orig_no_quote = orig.replace("'", "")
    corr_no_quote = corr.replace("'", "")
    if orig_no_quote == corr_no_quote and orig != corr:
        return True

    # 计算编辑距离
    def edit_distance(s1: str, s2: str) -> int:
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    # 过滤编辑距离 <= 2 的简单词汇错误
    if len(orig) >= 2 and len(corr) >= 2:
        dist = edit_distance(orig, corr)
        if dist <= 2:
            return True

    # 过滤长度相差1且所有字符都在另一个词中的情况
    if abs(len(orig) - len(corr)) == 1:
        shorter, longer = (orig, corr) if len(orig) < len(corr) else (corr, orig)
        if all(c in longer for c in shorter):
            return True

    # 过滤主谓不一致/动词形式错误
    COMMON_VERB_VARIATIONS = {
        ('was', 'were'), ('is', 'are'), ('are', 'is'),
        ('do', 'did'), ('does', 'did'),
    }
    if (orig, corr) in COMMON_VERB_VARIATIONS or (corr, orig) in COMMON_VERB_VARIATIONS:
        return True

    # 过滤人称代词变化
    COMMON_PRONOUN_VARIATIONS = {
        ('i', 'me'), ('me', 'i'),
        ('he', 'him'), ('him', 'he'),
        ('she', 'her'), ('her', 'she'),
        ('we', 'us'), ('us', 'we'),
        ('they', 'them'), ('them', 'they'),
    }
    if (orig, corr) in COMMON_PRONOUN_VARIATIONS or (corr, orig) in COMMON_PRONOUN_VARIATIONS:
        return True

    # 确保 orig 是较短的词
    if len(orig) > len(corr):
        orig, corr = corr, orig

    if len(corr) - len(orig) < 1 or len(orig) < 3:
        return False

    for affix in COMMON_AFFIXES:
        if corr == orig + affix:
            return True

    return False


def validate_word_error(original: str, corrected: str) -> Tuple[bool, str]:
    orig = original.lower().strip()
    corr = corrected.lower().strip()
    if not orig or not corr:
        return False, "empty_error"
    if orig == corr:
        return False, "case_or_identity_error"
    if len(orig.split()) != 1 or len(corr.split()) != 1:
        return False, "non_single_word_error"
    return True, ""


def is_simple_error(original: str, corrected: str) -> Tuple[bool, str]:
    is_valid, reason = validate_word_error(original, corrected)
    if not is_valid:
        return False, reason
    if is_affix_variation(original, corrected):
        return True, "affix_variation"
    return False, ""


def get_category_paths(category: str) -> Tuple[Path, Path, Path]:
    category_dir = WRITING_ANALYSIS_ROOT / category
    writing_error_file = category_dir / "writing_error.json"
    stage1_reviews_file = STAGE1_ROOT / category / "stage1_reviews.json"
    output_file = category_dir / "acl_ccomp_error.json"
    return writing_error_file, stage1_reviews_file, output_file


def load_review_map(reviews_file: Path) -> Dict:
    with reviews_file.open("r", encoding="utf-8") as f:
        reviews_data = json.load(f)
    return {item["user_id"]: item for item in reviews_data}


def load_syntax_cache(category: str) -> Tuple[Dict, Dict]:
    cache_dir = SYNTACTIC_ANALYSIS_ROOT / category
    acl_file = cache_dir / "syntax_cache_acls.jsonl"
    ccomp_file = cache_dir / "syntax_cache_ccomps.jsonl"

    acl_cache: Dict = {}
    if acl_file.exists():
        with acl_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                user_id = row.get("user_id")
                if user_id is None:
                    raise ValueError(f"Invalid ACL syntax cache row: {row}")
                acl_cache[user_id] = row

    ccomp_cache: Dict = {}
    if ccomp_file.exists():
        with ccomp_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                user_id = row.get("user_id")
                if user_id is None:
                    raise ValueError(f"Invalid CCOMP syntax cache row: {row}")
                ccomp_cache[user_id] = row

    return acl_cache, ccomp_cache


def classify_word_error(original: str, corrected: str) -> Tuple[str, str, str, str]:
    is_simple, reason = is_simple_error(original, corrected)
    if is_simple:
        return "", "", "", reason

    return "", "", "", ""


def classify_acl_ccomp_from_word(
    original: str,
    corrected: str,
    context: str,
    region_type: str,
) -> Tuple[str, str, str, str]:
    is_simple, reason = is_simple_error(original, corrected)
    if is_simple:
        return "", "", "", reason

    if len(original.split()) != 1 or len(corrected.split()) != 1:
        return "", "", "", "non_single_word"

    return "", "", "", "unclassified"


def classify_from_stage5_cache(
    category: str,
    original: str,
    corrected: str,
    region_type: str,
    user_id: str,
    acl_cache: Dict,
    ccomp_cache: Dict,
) -> Tuple[str, str, str, str]:
    # CCOMP: 仅匹配 comp_type == 'ccomp'
    if user_id in ccomp_cache:
        ccomp_data = ccomp_cache[user_id]
        ccomp_errors = ccomp_data.get("ccomp_errors", [])
        for err in ccomp_errors:
            if err.get("comp_type") == "ccomp":
                return "ccomp", region_type, "ccomp", "ok"

    # ACL: 仅匹配 acl 和 relcl_reference
    if user_id in acl_cache:
        acl_data = acl_cache[user_id]
        acl_errors = acl_data.get("acl_errors", [])
        for err in acl_errors:
            err_type = err.get("type", "")
            if err_type == "acl":
                return "acl", region_type, "acl", "ok"
            if err_type == "relcl_reference":
                return "acl", region_type, "relcl_reference", "ok"
            # relcl_non_reference 不归类为 ACL，继续检查 CCOMP

    return "", "", "", "not_in_cache"


def classify_error(
    category: str,
    original: str,
    corrected: str,
    context: str,
    region_type: str,
    user_id: str,
    acl_cache: Dict,
    ccomp_cache: Dict,
) -> Tuple[str, str, str, str]:
    is_simple, reason = is_simple_error(original, corrected)
    if is_simple:
        return "", "", "", reason

    if len(original.split()) != 1 or len(corrected.split()) != 1:
        return "", "", "", "non_single_word"

    error_type, region_type, error_type, reason = classify_from_stage5_cache(
        category, original, corrected, region_type, user_id, acl_cache, ccomp_cache
    )
    if error_type:
        return error_type, region_type, error_type, reason

    if original.lower() in ("an", "a") or corrected.lower() in ("an", "a"):
        error_type = "np_inflection"
    elif region_type == "modifier" and abs(len(original) - len(corrected)) <= 2:
        error_type = "modifier_typo"
    else:
        error_type = "unclassified"

    return "", region_type, error_type, "fallback"


def user_has_errors(row: Dict) -> bool:
    return bool(row.get("detailed_results"))


def classify_user(row: Dict, review_map: Dict, syntax_cache: Tuple[Dict, Dict]) -> Dict:
    acl_cache, ccomp_cache = syntax_cache
    user_id = row["user_id"]
    category = row.get("category", "unknown")
    detailed_results = row.get("detailed_results", [])
    reviews_processed = row.get("reviews_processed", 0)

    output_results = []
    acl_count = 0
    ccomp_count = 0
    filtered_counts: Counter = Counter()
    acl_error_types: Counter = Counter()
    ccomp_error_types: Counter = Counter()
    acl_region_types: Counter = Counter()
    ccomp_region_types: Counter = Counter()

    for result in detailed_results:
        word_errors = result.get("word_errors", [])
        for word_error in word_errors:
            original = word_error.get("original", "")
            corrected = word_error.get("corrected", "")
            context = word_error.get("context", "")
            region_type = word_error.get("region_type", "")

            error_type, _, detail_type, reason = classify_error(
                category, original, corrected, context, region_type, user_id, acl_cache, ccomp_cache
            )

            if error_type:
                if error_type == "acl":
                    acl_count += 1
                    acl_error_types[detail_type] += 1
                    acl_region_types[region_type] += 1
                elif error_type == "ccomp":
                    ccomp_count += 1
                    ccomp_error_types[detail_type] += 1
                    ccomp_region_types[region_type] += 1

                output_results.append({
                    "original": original,
                    "corrected": corrected,
                    "context": context,
                    "region_type": region_type,
                    "error_type": error_type,
                    "detail_type": detail_type,
                    "reason": reason,
                })
            elif reason:
                filtered_counts[reason] += 1

    return {
        "user_id": user_id,
        "category": category,
        "status": row.get("status", "unknown"),
        "reviews_processed": reviews_processed,
        "acl_error_count": acl_count,
        "ccomp_error_count": ccomp_count,
        "total_errors": acl_count + ccomp_count,
        "acl_error_types": dict(acl_error_types),
        "ccomp_error_types": dict(ccomp_error_types),
        "acl_region_types": dict(acl_region_types),
        "ccomp_region_types": dict(ccomp_region_types),
        "filtered_counts": dict(filtered_counts),
        "detailed_results": output_results,
    }


def empty_classified_user(row: Dict) -> Dict:
    for key in ("user_id", "status", "reviews_processed", "detailed_results"):
        if key not in row:
            raise ValueError(f"Missing {key} in row: {row}")
    if not isinstance(row["detailed_results"], list):
        raise ValueError(f"detailed_results must be a list for user {row['user_id']}")
    return {
        "user_id": row["user_id"],
        "category": row.get("category", "unknown"),
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


def format_elapsed(start_time: float) -> str:
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m{seconds}s"


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
