#!/usr/bin/env python3
"""
Stage 4: Extract Errors from User Reviews with Simple Error Filtering

从用户评论中提取写作错误，并过滤掉简单错误（词缀变化、编辑距离<=2等）。

Input:
  - stage1_filtered_users_reviews.json (from Stage 1)
  
Output:
  - writing_error.json (过滤后的错误)

Usage:
  python 04_extract_all_user_errors.py --category Baby_Products
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# 添加 common 模块路径
_COMMON = Path(__file__).resolve().parent / "common"
sys.path.insert(0, str(_COMMON))

# LLM 客户端
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import LLMClient


# ============================================================================
# 常量
# ============================================================================

WRITING_ANALYSIS_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/04_writing_analysis")
STAGE1_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/01_preference_extraction")
PROGRESS_INTERVAL_USERS = 100

# 常见英语词缀
COMMON_AFFIXES = ['s', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'd', 'en', 'n']


# ============================================================================
# 工具函数
# ============================================================================

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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


def format_elapsed(start_time: float) -> str:
    """格式化耗时"""
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m{seconds}s"


# ============================================================================
# P3 Error Extraction
# ============================================================================

class P3ErrorExtractor:
    """使用 P3 最优模板提取错误"""
    
    P3_TEMPLATE = """Edit the following text for spelling and grammar mistakes, make minimal changes, and return only the corrected text. If the text is already correct, return it without any explanations:."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def create_p3_prompt(self, review_text: str) -> str:
        return f"""<s>[INST] {self.P3_TEMPLATE}

"{review_text}"

Please return ONLY the corrected text. If no corrections are needed, return the original text exactly as it is.
[/INST]"""
    
    def extract_errors(self, original_text: str, max_retries: int = 3) -> Dict:
        prompt = self.create_p3_prompt(original_text)
        
        for attempt in range(max_retries):
            try:
                response = self.llm_client.call(
                    prompt=prompt,
                    max_tokens=256
                )
                
                corrected_text = response.strip()
                
                return {
                    "status": "success",
                    "original": original_text,
                    "corrected": corrected_text,
                    "has_errors": original_text != corrected_text
                }
            
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    return {
                        "status": "error",
                        "original": original_text,
                        "corrected": original_text,
                        "error": str(e),
                        "has_errors": False
                    }


# ============================================================================
# 主处理函数
# ============================================================================

def extract_and_filter_errors(category: str, max_users: int = None) -> None:
    """提取并过滤错误"""
    start_time = time.time()
    
    # 路径
    input_file = STAGE1_ROOT / category / "stage1_filtered_users_reviews.json"
    output_file = WRITING_ANALYSIS_ROOT / category / "writing_error.json"
    
    log(f"=== Processing category: {category} ===")
    log(f"Reading from: {input_file}")
    
    if not input_file.exists():
        log(f"Error: File not found: {input_file}")
        return
    
    # 加载数据
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    users = data.get("users", [])
    if max_users:
        users = users[:max_users]
    
    log(f"Loaded {len(users)} users")
    
    # 初始化 LLM 客户端
    log("Initializing LLM client...")
    llm_client = LLMClient()
    extractor = P3ErrorExtractor(llm_client)
    
    # 处理每个用户
    output_rows = []
    total_errors = 0
    total_filtered = 0
    simple_counts_total: Counter = Counter()
    
    for idx, user in enumerate(users, 1):
        user_id = user.get("user_id", "")
        results = user.get("results", [])
        
        user_errors = []
        user_simple_counts: Counter = Counter()
        
        # 遍历所有评论
        for result in results:
            reviews = result.get("target_reviews", [])
            for review in reviews:
                if not review or not isinstance(review, str):
                    continue
                
                # 使用 LLM 提取错误
                extraction = extractor.extract_errors(review)
                
                if extraction.get("has_errors"):
                    original = extraction.get("original", "")
                    corrected = extraction.get("corrected", "")
                    
                    # 检查是否是简单错误
                    is_simple, reason = is_simple_error(original, corrected)
                    
                    if is_simple:
                        user_simple_counts[reason] += 1
                    else:
                        user_errors.append({
                            "original": original,
                            "corrected": corrected,
                            "context": review[:200],  # 保留上下文
                        })
        
        # 统计
        total_errors += len(user_errors) + sum(user_simple_counts.values())
        total_filtered += sum(user_simple_counts.values())
        simple_counts_total.update(user_simple_counts)
        
        # 只保留有错误且过滤后仍有剩余的用户
        if user_errors:
            output_rows.append({
                "user_id": user_id,
                "category": category,
                "status": "success",
                "total_errors": len(user_errors) + sum(user_simple_counts.values()),
                "filtered_errors": len(user_errors),
                "simple_error_counts": dict(user_simple_counts),
                "error_details": user_errors,
            })
        
        # 进度输出
        if idx % PROGRESS_INTERVAL_USERS == 0 or idx == len(users):
            elapsed = format_elapsed(start_time)
            log(
                f"[{category}] Processed {idx}/{len(users)} users; "
                f"total_errors={total_errors}; filtered={total_filtered}; "
                f"elapsed={elapsed}"
            )
    
    # 保存结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing {len(output_rows)} users with errors to: {output_file}")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)
    
    log(f"=== Finished: {category}; elapsed={format_elapsed(start_time)} ===")
    log(f"Total: {len(users)} users, {total_errors} errors, {total_filtered} filtered")
    log(f"Simple error breakdown: {dict(simple_counts_total)}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract and filter writing errors from reviews")
    parser.add_argument("--category", required=True, choices=[
        "Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"
    ])
    parser.add_argument("--max-users", type=int, default=None, help="Limit number of users to process")
    
    args = parser.parse_args()
    extract_and_filter_errors(args.category, args.max_users)


if __name__ == "__main__":
    main()
