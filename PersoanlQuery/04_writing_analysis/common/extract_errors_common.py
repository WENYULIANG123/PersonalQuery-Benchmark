#!/usr/bin/env python3
"""Common functions for extracting and filtering writing errors from reviews."""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from llm_client import LLMClient


# ============================================================================
# 常量
# ============================================================================

WRITING_ANALYSIS_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/04_writing_analysis")
STAGE1_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/01_preference_extraction")
PROGRESS_INTERVAL_USERS = 100

# 常见英语词缀
COMMON_AFFIXES = ['s', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'd', 'en', 'n']

# Prompt 文件路径
PROMPT_FILE = Path(__file__).resolve().parent / "extract_prompts.json"


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
    """判断是否是简单错误（应被过滤）"""
    import string
    
    orig = original.lower().strip()
    corr = corrected.lower().strip()

    if not orig or not corr:
        return False, "empty"
    if orig == corr:
        return False, "identity"
    if len(orig.split()) != 1 or len(corr.split()) != 1:
        return False, "multi_word"

    # 过滤仅相差标点的错误
    PUNCT = set(string.punctuation)
    orig_no_punct = ''.join(c for c in orig if c not in PUNCT)
    corr_no_punct = ''.join(c for c in corr if c not in PUNCT)
    if orig_no_punct == corr_no_punct and orig != corr:
        return True, "punct_only"

    # 过滤仅相差单引号的错误
    orig_no_quote = orig.replace("'", "")
    corr_no_quote = corr.replace("'", "")
    if orig_no_quote == corr_no_quote and orig != corr:
        return True, "quote_only"

    # 过滤编辑距离 <= 2 的简单词汇错误
    if len(orig) >= 2 and len(corr) >= 2:
        dist = _edit_distance(orig, corr)
        if dist <= 2:
            return True, "edit_dist_le2"

    # 过滤长度相差1且所有字符都在另一个词中的情况
    if abs(len(orig) - len(corr)) == 1:
        shorter, longer = (orig, corr) if len(orig) < len(corr) else (corr, orig)
        if all(c in longer for c in shorter):
            return True, "char_subset"

    # 过滤主谓不一致/动词形式错误
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

    # 过滤词缀变化
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
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """从 JSON 文件加载 prompt"""
        if not PROMPT_FILE.exists():
            raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")
        
        with PROMPT_FILE.open("r", encoding="utf-8") as f:
            prompts = json.load(f)
        
        self.template = prompts.get("p3_template", "")
        self.max_tokens = prompts.get("max_tokens", 256)
        self.max_retries = prompts.get("max_retries", 3)
        
        if not self.template:
            raise ValueError("p3_template is empty in prompt file")
    
    def create_p3_prompt(self, review_text: str) -> str:
        return f"""<s>[INST] {self.template}

"{review_text}"

Please return ONLY the corrected text. If no corrections are needed, return the original text exactly as it is.
[/INST]"""
    
    def extract_errors(self, original_text: str) -> Dict:
        prompt = self.create_p3_prompt(original_text)
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.call(
                    prompt=prompt,
                    max_tokens=self.max_tokens
                )
                
                corrected_text = response.strip()
                
                return {
                    "status": "success",
                    "original": original_text,
                    "corrected": corrected_text,
                    "has_errors": original_text != corrected_text
                }
            
            except Exception:
                if attempt < self.max_retries - 1:
                    continue
                else:
                    return {
                        "status": "error",
                        "original": original_text,
                        "corrected": original_text,
                        "has_errors": False
                    }


# ============================================================================
# 主处理函数
# ============================================================================

def extract_and_filter_errors(category: str, max_users: int = None) -> None:
    """提取并过滤错误"""
    start_time = time.time()
    
    input_file = STAGE1_ROOT / category / "stage1_filtered_users_reviews.json"
    output_file = WRITING_ANALYSIS_ROOT / category / "writing_error.json"
    
    log(f"=== Processing category: {category} ===")
    log(f"Reading from: {input_file}")
    log(f"Using prompts from: {PROMPT_FILE}")
    
    if not input_file.exists():
        log(f"Error: File not found: {input_file}")
        return
    
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    users = data.get("users", [])
    if max_users:
        users = users[:max_users]
    
    log(f"Loaded {len(users)} users")
    
    log("Initializing LLM client...")
    llm_client = LLMClient()
    extractor = P3ErrorExtractor(llm_client)
    
    output_rows = []
    total_errors = 0
    total_filtered = 0
    simple_counts_total: Counter = Counter()
    
    for idx, user in enumerate(users, 1):
        user_id = user.get("user_id", "")
        results = user.get("results", [])
        
        user_errors = []
        user_simple_counts: Counter = Counter()
        
        for result in results:
            reviews = result.get("target_reviews", [])
            for review in reviews:
                if not review or not isinstance(review, str):
                    continue
                
                extraction = extractor.extract_errors(review)
                
                if extraction.get("has_errors"):
                    original = extraction.get("original", "")
                    corrected = extraction.get("corrected", "")
                    is_simple, reason = is_simple_error(original, corrected)
                    
                    if is_simple:
                        user_simple_counts[reason] += 1
                    else:
                        user_errors.append({
                            "original": original,
                            "corrected": corrected,
                            "context": review[:200],
                        })
        
        total_errors += len(user_errors) + sum(user_simple_counts.values())
        total_filtered += sum(user_simple_counts.values())
        simple_counts_total.update(user_simple_counts)
        
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
        
        if idx % PROGRESS_INTERVAL_USERS == 0 or idx == len(users):
            log(
                f"[{category}] Processed {idx}/{len(users)} users; "
                f"total_errors={total_errors}; filtered={total_filtered}; "
                f"elapsed={format_elapsed(start_time)}"
            )
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing {len(output_rows)} users with errors to: {output_file}")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)
    
    log(f"=== Finished: {category}; elapsed={format_elapsed(start_time)} ===")
    log(f"Total: {len(users)} users, {total_errors} errors, {total_filtered} filtered")
    log(f"Simple error breakdown: {dict(simple_counts_total)}")
