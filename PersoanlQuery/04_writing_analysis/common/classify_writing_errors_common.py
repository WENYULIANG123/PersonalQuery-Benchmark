#!/usr/bin/env python3
"""Common functions for filtering simple writing errors."""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple


WRITING_ANALYSIS_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/04_writing_analysis")
STAGE1_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/01_preference_extraction")
PROGRESS_INTERVAL_USERS = 25


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# 常见英语词缀
COMMON_AFFIXES = ['s', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'd', 'en', 'n']


def _edit_distance(s1: str, s2: str) -> int:
    """计算编辑距离"""
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


def is_simple_error(original: str, corrected: str) -> Tuple[bool, str]:
    """判断是否是简单错误（应被过滤）
    
    Returns:
        (is_simple, reason): (True, reason) 如果是简单错误
                             (False, "") 如果不是简单错误
    """
    import string
    
    orig = original.lower().strip()
    corr = corrected.lower().strip()

    if not orig or not corr:
        return False, "empty"
    if orig == corr:
        return False, "identity"
    if len(orig.split()) != 1 or len(corr.split()) != 1:
        return False, "multi_word"

    # 过滤仅相差标点的错误（如 a. -> a）
    PUNCT = set(string.punctuation)
    orig_no_punct = ''.join(c for c in orig if c not in PUNCT)
    corr_no_punct = ''.join(c for c in corr if c not in PUNCT)
    if orig_no_punct == corr_no_punct and orig != corr:
        return True, "punct_only"

    # 过滤仅相差单引号的错误（如 dont -> don't）
    orig_no_quote = orig.replace("'", "")
    corr_no_quote = corr.replace("'", "")
    if orig_no_quote == corr_no_quote and orig != corr:
        return True, "quote_only"

    # 过滤编辑距离 <= 2 的简单词汇错误（如 creat->create, an->a）
    if len(orig) >= 2 and len(corr) >= 2:
        dist = _edit_distance(orig, corr)
        if dist <= 2:
            return True, "edit_dist_le2"

    # 过滤长度相差1且所有字符都在另一个词中的情况（如 a->an）
    if abs(len(orig) - len(corr)) == 1:
        shorter, longer = (orig, corr) if len(orig) < len(corr) else (corr, orig)
        if all(c in longer for c in shorter):
            return True, "char_subset"

    # 过滤常见的主谓不一致/动词形式错误
    COMMON_VERB_VARIATIONS = {
        ('was', 'were'), ('is', 'are'), ('are', 'is'),
        ('do', 'did'), ('does', 'did'),
    }
    if (orig, corr) in COMMON_VERB_VARIATIONS or (corr, orig) in COMMON_VERB_VARIATIONS:
        return True, "verb_variation"

    # 过滤人称代词变化
    COMMON_PRONOUN_VARIATIONS = {
        ('i', 'me'), ('me', 'i'),
        ('he', 'him'), ('him', 'he'),
        ('she', 'her'), ('her', 'she'),
        ('we', 'us'), ('us', 'we'),
        ('they', 'them'), ('them', 'they'),
    }
    if (orig, corr) in COMMON_PRONOUN_VARIATIONS or (corr, orig) in COMMON_PRONOUN_VARIATIONS:
        return True, "pronoun_variation"

    # 过滤词缀变化（如 dog->dogs, jump->jumped）
    if len(orig) > len(corr):
        orig, corr = corr, orig
    if len(corr) - len(orig) >= 1 and len(orig) >= 3:
        for affix in COMMON_AFFIXES:
            if corr == orig + affix:
                return True, "affix_variation"

    return False, ""


def get_category_paths(category: str) -> Tuple[Path, Path, Path]:
    """获取输入输出文件路径"""
    category_dir = WRITING_ANALYSIS_ROOT / category
    writing_error_file = category_dir / "writing_error.json"
    stage1_reviews_file = STAGE1_ROOT / category / "stage1_reviews.json"
    output_file = category_dir / "writing_error_filtered.json"
    return writing_error_file, stage1_reviews_file, output_file


def classify_user(row: Dict) -> Dict:
    """对单个用户的错误进行分类，过滤简单错误"""
    user_id = row.get("user_id", "")
    category = row.get("category", "unknown")
    detailed_results = row.get("detailed_results", [])
    reviews_processed = row.get("reviews_processed", 0)

    filtered_errors = []
    simple_counts: Counter = Counter()

    for result in detailed_results:
        word_errors = result.get("word_errors", [])
        for word_error in word_errors:
            original = word_error.get("original", "")
            corrected = word_error.get("corrected", "")
            context = word_error.get("context", "")
            region_type = word_error.get("region_type", "")

            is_simple, reason = is_simple_error(original, corrected)
            if is_simple:
                simple_counts[reason] += 1
            else:
                filtered_errors.append({
                    "original": original,
                    "corrected": corrected,
                    "context": context,
                    "region_type": region_type,
                })

    return {
        "user_id": user_id,
        "category": category,
        "status": row.get("status", "unknown"),
        "reviews_processed": reviews_processed,
        "total_errors": len(detailed_results) if detailed_results else 0,
        "filtered_errors": len(filtered_errors),
        "simple_error_counts": dict(simple_counts),
        "filtered_error_details": filtered_errors,
    }


def format_elapsed(start_time: float) -> str:
    """格式化耗时"""
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m{seconds}s"


def classify_category(category: str) -> None:
    """处理单个类别的错误分类"""
    start_time = time.time()
    writing_error_file, _, output_file = get_category_paths(category)
    log(f"=== Processing category: {category} ===")
    log(f"[{category}] Reading from: {writing_error_file}")

    if not writing_error_file.exists():
        log(f"[{category}] File not found: {writing_error_file}")
        return

    with writing_error_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    log(f"[{category}] Loaded {len(rows)} users")

    output_rows = []
    total_filtered = 0
    total_simple = 0
    simple_counts_total: Counter = Counter()

    for idx, row in enumerate(rows, start=1):
        output_row = classify_user(row)
        output_rows.append(output_row)
        total_filtered += output_row["filtered_errors"]
        total_simple += output_row["total_errors"] - output_row["filtered_errors"]
        simple_counts_total.update(output_row["simple_error_counts"])

        if idx % 100 == 0 or idx == len(rows):
            log(
                f"[{category}] Processed {idx}/{len(rows)} users; "
                f"filtered_errors={total_filtered}; simple_errors={total_simple}; "
                f"elapsed={format_elapsed(start_time)}"
            )

    # 只保留有过滤错误的用户
    output_rows_with_errors = [r for r in output_rows if r["filtered_errors"] > 0]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"[{category}] Writing {len(output_rows_with_errors)} users with errors to: {output_file}")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_rows_with_errors, f, ensure_ascii=False, indent=2)

    log(f"[{category}] Total: {len(rows)} users, {total_filtered} filtered errors, {total_simple} simple errors")
    log(f"[{category}] Simple error breakdown: {dict(simple_counts_total)}")
    log(f"=== Finished: {category}; elapsed={format_elapsed(start_time)} ===")
