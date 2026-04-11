#!/usr/bin/env python3
"""
Stage 4: P3 Optimal Template Comprehensive Error Analysis

合并脚本：结合P3错误提取和详细错误分析
1. Error extraction using P3 optimal template (MTSummit 2025, arXiv:2505.06004)
2. Detailed error recognition with position, type, and classification

Input:
  - /home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/stage1_filtered_users_reviews.json

Output:
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/writing_analysis_{user_id}.json

Usage:
  python 04_extract_all_user_errors.py
"""

import json
import os
import sys
import argparse
import difflib
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logging.getLogger('anthropic').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


# ============================================================================
# 硬编码参数
# ============================================================================

INPUT_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/stage1_filtered_users_reviews.json"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis"
MAX_USERS = 20  # 测试用：只处理前N个用户
MAX_REVIEWS = None  # None 表示处理所有
MAX_WORKERS = 20


# ============================================================================
# P3 Error Extraction
# ============================================================================

class P3ErrorExtractor:
    """Extract errors using P3 optimal template - LLM直接返回错误类型"""

    ERROR_TYPES = ["spelling", "grammar", "capitalization"]

    P3_TEMPLATE = """You are an expert at identifying and correcting errors in text.
For each error you find, you must identify:
1. The original error word (SINGLE WORD only, not phrase)
2. The corrected word (SINGLE WORD only, not phrase)
3. The error type (one of: spelling, grammar, capitalization)

IMPORTANT: original and corrected MUST be single words, not phrases or sentences.

Error type definitions:
- spelling: misspelled words (e.g., "teh" -> "the", "recieve" -> "receive")
- grammar: wrong word form (e.g., "goods" -> "good", "works" -> "work", verb tense issues, "measuring" -> "measurement")
- capitalization: wrong case (e.g., "i" -> "I", "monday" -> "Monday")

NOTE: Ignore the following (NOT errors):
- Punctuation marks (comma, period, etc.)
- Hyphenation issues (e.g., "plastic-like" vs "plastic like")
- Quote style issues
- Whitespace issues

If the text has no errors or only ignored issues, return {"corrected_text": "<original>", "errors": []}.

If there are errors, return JSON in this format:
{"corrected_text": "<corrected sentence>", "errors": [{"original": "word", "corrected": "word", "type": "error_type"}, ...]}

Analyze this text and return the result:"""

    def __init__(self):
        from llm_client import MiniMaxAnthropicClient
        self.client = MiniMaxAnthropicClient(model='MiniMax-M2.7-highspeed')

    def create_p3_prompt(self, review_text: str) -> str:
        return f"""<s>[INST] {self.P3_TEMPLATE}

"{review_text}"
[/INST]"""

    def extract_errors(self, original_text: str) -> Dict:
        prompt = self.create_p3_prompt(original_text)

        wait_seconds = 5
        max_retries = 20

        for attempt in range(max_retries):
            try:
                thinking, text = self.client.call_with_thinking(prompt, max_tokens=8192, temperature=0.3)
                result_text = text.strip()

                # 如果返回为空，保留原文
                if not result_text:
                    result_text = original_text

                # 尝试解析 JSON
                import json
                try:
                    # 提取 JSON（可能在 markdown 代码块中）
                    if "```json" in result_text:
                        start = result_text.find("```json") + 7
                        end = result_text.find("```", start)
                        result_text = result_text[start:end].strip()
                    elif "```" in result_text:
                        start = result_text.find("```") + 3
                        end = result_text.find("```", start)
                        result_text = result_text[start:end].strip()

                    result = json.loads(result_text)
                    corrected_text = result.get("corrected_text", original_text)
                    errors = result.get("errors", [])

                    # 如果没有错误但文本被修改了，保留修改
                    if not errors and corrected_text != original_text:
                        errors = [{"original": original_text, "corrected": corrected_text, "type": "unknown"}]

                    return {
                        "status": "success",
                        "original": original_text,
                        "corrected": corrected_text,
                        "has_errors": len(errors) > 0,
                        "errors": errors
                    }

                except json.JSONDecodeError:
                    # JSON 解析失败，返回原文，不标记错误
                    return {
                        "status": "success",
                        "original": original_text,
                        "corrected": original_text,
                        "has_errors": False,
                        "errors": []
                    }

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = 'rate' in error_str or 'limit' in error_str or '429' in error_str or 'too many request' in error_str
                if not is_rate_limit:
                    return {
                        "status": "error",
                        "original": original_text,
                        "corrected": original_text,
                        "error": str(e),
                        "has_errors": False,
                        "errors": []
                    }
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit, retrying in {wait_seconds}s (attempt {attempt + 1}/{max_retries})")
                    import time
                    time.sleep(wait_seconds)
                    wait_seconds += 5
                else:
                    return {
                        "status": "error",
                        "original": original_text,
                        "corrected": original_text,
                        "error": str(e),
                        "has_errors": False,
                        "errors": []
                    }


# ============================================================================
# Error Type Classification
# ============================================================================

class ErrorTypeClassifier:
    """Automatic error type classification"""

    PUNCTUATION = set('.,;:!?\'"()-[]{}""''..«»')

    @staticmethod
    def classify(original: str, corrected: str) -> dict:
        if original.lower() == corrected.lower() and original != corrected:
            return {
                "type": "capitalization",
                "description": f"Capitalization: '{original}' → '{corrected}'",
                "confidence": 0.95
            }

        if ErrorTypeClassifier._is_punctuation_change(original, corrected):
            return {
                "type": "punctuation",
                "description": ErrorTypeClassifier._describe_punctuation_change(original, corrected),
                "confidence": 0.95
            }

        if original.replace(" ", "") == corrected.replace(" ", ""):
            return {
                "type": "whitespace",
                "description": f"Whitespace adjustment: '{original}' → '{corrected}'",
                "confidence": 0.9
            }

        if original.replace("-", "") == corrected.replace("-", ""):
            return {
                "type": "formatting",
                "description": f"Hyphenation: '{original}' → '{corrected}'",
                "confidence": 0.9
            }

        if ErrorTypeClassifier._is_quote_change(original, corrected):
            return {
                "type": "formatting",
                "description": f"Quote style: '{original}' → '{corrected}'",
                "confidence": 0.85
            }

        if len(original) > 1 and len(corrected) > 1:
            edit_dist = ErrorTypeClassifier._edit_distance(original.lower(), corrected.lower())
            if edit_dist <= 2:
                return {
                    "type": "spelling",
                    "description": f"Spelling: '{original}' → '{corrected}'",
                    "confidence": 0.8
                }
            elif ErrorTypeClassifier._is_morphological_change(original, corrected):
                return {
                    "type": "grammar",
                    "description": f"Grammar: '{original}' → '{corrected}'",
                    "confidence": 0.8
                }

        return {
            "type": "grammar",
            "description": f"Correction: '{original}' → '{corrected}'",
            "confidence": 0.7
        }

    @staticmethod
    def _is_punctuation_change(original: str, corrected: str) -> bool:
        orig_punct = "".join(c for c in original if c in ErrorTypeClassifier.PUNCTUATION)
        corr_punct = "".join(c for c in corrected if c in ErrorTypeClassifier.PUNCTUATION)
        orig_alpha = "".join(c for c in original if c.isalnum() or c.isspace())
        corr_alpha = "".join(c for c in corrected if c.isalnum() or c.isspace())

        return (orig_alpha == corr_alpha and orig_punct != corr_punct)

    @staticmethod
    def _describe_punctuation_change(original: str, corrected: str) -> str:
        orig_punct = set(c for c in original if c in ErrorTypeClassifier.PUNCTUATION)
        corr_punct = set(c for c in corrected if c in ErrorTypeClassifier.PUNCTUATION)

        added = corr_punct - orig_punct
        removed = orig_punct - corr_punct

        desc = "Punctuation"
        if added:
            desc += f", added: {repr(list(added)[0])}" if len(added) == 1 else f", added: {repr(list(added))}"
        if removed:
            desc += f", removed: {repr(list(removed)[0])}" if len(removed) == 1 else f", removed: {repr(list(removed))}"

        return desc

    @staticmethod
    def _is_quote_change(original: str, corrected: str) -> bool:
        quote_chars = {'"', "'", """...""", """'""", "'", "'", "«", "»"}
        orig_has_quote = any(c in original for c in quote_chars)
        corr_has_quote = any(c in corrected for c in quote_chars)
        return orig_has_quote and corr_has_quote

    @staticmethod
    def _is_morphological_change(original: str, corrected: str) -> bool:
        suffixes = ['ed', 'ing', 'ly', 's', 'es', 'er', 'est', 'tion', 'ment', 'able', 'ible']

        orig_lower = original.lower()
        corr_lower = corrected.lower()

        for suffix in suffixes:
            if orig_lower.endswith(suffix) != corr_lower.endswith(suffix):
                return True

        return False

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return ErrorTypeClassifier._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


# ============================================================================
# Detailed Error Analysis
# ============================================================================

class DetailedErrorExtractor:
    """Extract detailed error information"""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size

    def extract_errors(self, original: str, corrected: str) -> list:
        errors = []
        error_id = 1

        matcher = difflib.SequenceMatcher(None, original, corrected)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue

            orig_text = original[i1:i2]
            corr_text = corrected[j1:j2]

            context_start = max(0, i1 - self.window_size)
            context_end = min(len(original), i2 + self.window_size)
            orig_context = original[context_start:context_end]

            context_start_corr = max(0, j1 - self.window_size)
            context_end_corr = min(len(corrected), j2 + self.window_size)
            corr_context = corrected[context_start_corr:context_end_corr]

            error_type_info = ErrorTypeClassifier.classify(orig_text, corr_text)

            error = {
                "error_id": error_id,
                "type": error_type_info["type"],
                "description": error_type_info["description"],
                "confidence": error_type_info["confidence"],
                "location": {
                    "original_position": i1,
                    "corrected_position": j1,
                    "original_snippet": self._get_snippet(original, i1, i2),
                    "corrected_snippet": self._get_snippet(corrected, j1, j2),
                },
                "details": {
                    "original": orig_text,
                    "corrected": corr_text,
                    "change_type": tag,
                    "original_length": len(orig_text),
                    "corrected_length": len(corr_text),
                },
                "context": {
                    "original": orig_context,
                    "corrected": corr_context,
                }
            }

            errors.append(error)
            error_id += 1

        return errors

    @staticmethod
    def _get_snippet(text: str, start: int, end: int, padding: int = 30) -> dict:
        snippet_start = max(0, start - padding)
        snippet_end = min(len(text), end + padding)

        snippet = text[snippet_start:snippet_end]

        rel_start = start - snippet_start
        rel_end = rel_start + (end - start)

        return {
            "text": snippet,
            "modification_start": rel_start,
            "modification_end": rel_end,
        }


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

class P3ComprehensiveAnalyzer:
    """Complete P3 analysis pipeline"""

    def __init__(self, analysis_dir: Path = None, reviews_dir: Path = None):
        self.analysis_dir = analysis_dir or Path(OUTPUT_DIR)
        self.reviews_dir = reviews_dir or Path(os.path.dirname(INPUT_FILE))
        self.p3_extractor = P3ErrorExtractor()
        self.error_extractor = DetailedErrorExtractor()

        self._merged_data = None
        self._users_map = None

    def _load_merged_file(self):
        """懒加载合并文件"""
        if self._users_map is None:
            if os.path.exists(INPUT_FILE):
                with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                    self._merged_data = json.load(f)
                self._users_map = {u.get('user_id'): u for u in self._merged_data.get('users', []) if u.get('user_id')}
                logger.info(f"Loaded {len(self._users_map)} users from merged file")
            else:
                self._users_map = {}
                logger.warning(f"Merged file not found: {INPUT_FILE}")

    def _get_user_data_from_merged(self, user_id: str) -> Optional[dict]:
        """从合并文件获取单个用户数据"""
        self._load_merged_file()
        return self._users_map.get(user_id)

    def process_user(self, user_id: str, reviews_file: Optional[str] = None, max_reviews: Optional[int] = None) -> dict:
        """Process single user: extract P3 errors and perform detailed analysis"""

        reviews_data = self._get_user_data_from_merged(user_id)

        if reviews_data is None:
            if not reviews_file:
                reviews_file = str(self.reviews_dir / f"reviews_{user_id}.json")

            if not os.path.exists(reviews_file):
                logger.error(f"Reviews file not found: {reviews_file}")
                return {"user_id": user_id, "status": "failed", "reason": "reviews_file_not_found"}

            with open(reviews_file, 'r', encoding='utf-8') as f:
                reviews_data = json.load(f)

        try:

            products = reviews_data.get('results', reviews_data.get('reviews', []))

            flattened_reviews = []
            for product in products:
                asin = product.get('asin', '')

                target_reviews = product.get('target_reviews', [])
                for review_text in target_reviews:
                    if isinstance(review_text, str):
                        flattened_reviews.append((review_text, asin))

            if max_reviews:
                flattened_reviews = flattened_reviews[:max_reviews]

            reviews_with_errors = 0
            total_errors = 0
            error_type_counts = defaultdict(int)
            detailed_results = []

            for review_idx, (review_text, asin) in enumerate(flattened_reviews):
                original = review_text

                p3_result = self.p3_extractor.extract_errors(original)

                if p3_result["status"] != "success":
                    logger.warning(f"[{user_id}] Review {review_idx}: {p3_result.get('error', 'Unknown error')}")
                    continue

                corrected = p3_result["corrected"]
                errors = p3_result.get("errors", [])

                for error in errors:
                    error_type = error.get("type", "unknown")
                    error_type_counts[error_type] += 1

                if len(errors) > 0:
                    reviews_with_errors += 1

                    # 直接使用 LLM 返回的错误信息
                    corrections = []
                    for error in errors:
                        corrections.append({
                            "original": error.get("original", ""),
                            "corrected": error.get("corrected", ""),
                            "error_type": error.get("type", "unknown")
                        })

                    detailed_results.append({
                        "original": original,
                        "corrected": corrected,
                        "corrections": corrections
                    })

                total_errors += len(errors)

                if (review_idx + 1) % 10 == 0:
                    logger.info(f"[{user_id}] Progress: {review_idx + 1}/{len(flattened_reviews)} reviews, {total_errors} errors found so far")

            logger.info(f"✅ [{user_id}] P3 comprehensive analysis completed: {len(flattened_reviews)} reviews, {total_errors} errors")

            return {
                "user_id": user_id,
                "status": "success",
                "reviews_processed": len(flattened_reviews),
                "reviews_with_errors": reviews_with_errors,
                "total_errors": total_errors,
                "error_types": dict(error_type_counts),
                "detailed_results": detailed_results,
            }

        except Exception as e:
            logger.error(f"❌ [{user_id}] Processing failed: {str(e)}")
            return {"user_id": user_id, "status": "failed", "reason": str(e)}


# ============================================================================
# Helper Functions
# ============================================================================

def log_with_timestamp(message: str):
    """打印带时间戳的日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_users_from_merged_file(input_file: str) -> List[str]:
    """从合并的用户评论文件加载用户列表"""
    log_with_timestamp(f"Loading users from merged file: {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    users = data.get('users', [])
    user_ids = [u.get('user_id') for u in users if u.get('user_id')]
    log_with_timestamp(f"Found {len(user_ids)} users in merged file")

    return user_ids


def validate_users_from_merged_file(input_file: str, user_ids: List[str]) -> Set[str]:
    """验证合并文件中的用户"""
    log_with_timestamp(f"Validating users from merged file...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    users_data = {u.get('user_id'): u for u in data.get('users', []) if u.get('user_id')}

    existing_users = set()
    for user_id in user_ids:
        if user_id in users_data:
            user = users_data[user_id]
            results = user.get('results', [])
            target_count = sum(len(p.get('target_reviews', [])) for p in results)
            existing_users.add(user_id)
            log_with_timestamp(f"  ✓ User {user_id}: {target_count} reviews")

    log_with_timestamp(f"Found {len(existing_users)} valid users")
    return existing_users


def generate_summary(output_dir: str, user_ids: List[str]) -> Dict:
    """生成所有用户的汇总统计"""
    log_with_timestamp("="*80)
    log_with_timestamp("Generating summary statistics...")
    log_with_timestamp("="*80)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_users": len(user_ids),
        "processed_users": 0,
        "failed_users": [],
        "user_summaries": {},
        "aggregate_stats": {
            "total_reviews_analyzed": 0,
            "total_words_analyzed": 0,
            "total_character_errors": 0,
            "overall_error_rate": 0.0,
            "error_type_distribution": {},
            "severity_distribution": {
                "low": 0,
                "medium": 0,
                "high": 0
            }
        }
    }

    error_type_counter = {}

    for user_id in user_ids:
        output_file = os.path.join(output_dir, f"writing_analysis_{user_id}.json")

        if not os.path.exists(output_file):
            log_with_timestamp(f"  ✗ User {user_id}: output file not found")
            summary["failed_users"].append(user_id)
            continue

        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)

            summary["processed_users"] += 1

            user_summary = {
                "user_id": user_id,
                "reviews_analyzed": user_data.get("total_reviews", 0),
                "total_words": sum(r.get("original_length", 0) for r in user_data.get("detailed_errors", [])),
                "total_character_errors": user_data.get("total_errors", 0),
                "character_error_rate": 0.0,
                "average_severity": 0.0,
                "top_error_types": []
            }

            if user_summary["reviews_analyzed"] > 0:
                user_summary["character_error_rate"] = round(
                    user_summary["total_character_errors"] / user_summary["reviews_analyzed"], 2
                )

            summary["aggregate_stats"]["total_reviews_analyzed"] += user_summary["reviews_analyzed"]
            summary["aggregate_stats"]["total_words_analyzed"] += user_summary["total_words"]
            summary["aggregate_stats"]["total_character_errors"] += user_summary["total_character_errors"]

            error_types = user_data.get("error_type_distribution", {})
            for error_type, count in error_types.items():
                error_type_counter[error_type] = error_type_counter.get(error_type, 0) + count

            top_types = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:3]
            user_summary["top_error_types"] = [{"type": t, "count": c} for t, c in top_types]

            summary["user_summaries"][user_id] = user_summary

            log_with_timestamp(
                f"  ✓ User {user_id}: {user_summary['total_character_errors']} errors "
                f"in {user_summary['reviews_analyzed']} reviews"
            )

        except Exception as e:
            log_with_timestamp(f"  ✗ User {user_id}: error reading results - {e}")
            summary["failed_users"].append(user_id)

    if summary["aggregate_stats"]["total_words_analyzed"] > 0:
        summary["aggregate_stats"]["overall_error_rate"] = round(
            summary["aggregate_stats"]["total_character_errors"] /
            summary["aggregate_stats"]["total_reviews_analyzed"],
            2
        )

    summary["aggregate_stats"]["error_type_distribution"] = dict(
        sorted(error_type_counter.items(), key=lambda x: x[1], reverse=True)
    )

    summary_file = os.path.join(output_dir, "all_users_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"Summary saved to {summary_file}")

    log_with_timestamp("="*80)
    log_with_timestamp("AGGREGATE STATISTICS")
    log_with_timestamp("="*80)
    log_with_timestamp(f"Processed users: {summary['processed_users']}/{summary['total_users']}")
    log_with_timestamp(f"Total reviews analyzed: {summary['aggregate_stats']['total_reviews_analyzed']}")
    log_with_timestamp(f"Total character errors: {summary['aggregate_stats']['total_character_errors']}")
    log_with_timestamp(f"Overall error rate: {summary['aggregate_stats']['overall_error_rate']} errors/review")

    log_with_timestamp("\nTop Error Types:")
    top_error_types = list(summary['aggregate_stats']['error_type_distribution'].items())[:5]
    for i, (error_type, count) in enumerate(top_error_types, 1):
        log_with_timestamp(f"  {i}. {error_type}: {count}")

    if summary["failed_users"]:
        log_with_timestamp(f"\nFailed users: {', '.join(summary['failed_users'])}")

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_with_timestamp("="*80)
    log_with_timestamp("Stage 4: P3 Optimal Template Comprehensive Error Analysis")
    log_with_timestamp("="*80)
    log_with_timestamp(f"Input file: {INPUT_FILE}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    log_with_timestamp(f"Max users: {MAX_USERS}")
    log_with_timestamp(f"Max reviews per user: {MAX_REVIEWS}")

    user_ids = load_users_from_merged_file(INPUT_FILE)

    if not user_ids:
        log_with_timestamp("ERROR: No users to process!")
        sys.exit(1)

    if MAX_USERS:
        user_ids = user_ids[:MAX_USERS]
        log_with_timestamp(f"Limited to {MAX_USERS} users for testing")

    existing_users = validate_users_from_merged_file(INPUT_FILE, user_ids)

    if not existing_users:
        log_with_timestamp("ERROR: No valid users found!")
        sys.exit(1)

    user_ids_to_process = sorted(list(existing_users))

    analyzer = P3ComprehensiveAnalyzer(
        analysis_dir=Path(OUTPUT_DIR),
        reviews_dir=Path(os.path.dirname(INPUT_FILE))
    )

    results = []

    log_with_timestamp("="*80)
    log_with_timestamp(f"Processing {len(user_ids_to_process)} users with P3 optimal template...")
    log_with_timestamp("="*80)

    for user_idx, user_id in enumerate(user_ids_to_process, 1):
        log_with_timestamp(f"[{user_idx}/{len(user_ids_to_process)}] Processing user: {user_id}")
        result = analyzer.process_user(user_id, None, MAX_REVIEWS)
        results.append(result)

        if result["status"] == "success":
            log_with_timestamp(f"  ✓ {result['user_id']}: {result['reviews_processed']} reviews, {result['total_errors']} errors")
        else:
            log_with_timestamp(f"  ✗ {result['user_id']}: {result.get('reason', 'Unknown error')}")

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    log_with_timestamp("="*80)
    log_with_timestamp(f"Completed: {len(successful)} success, {len(failed)} failed")
    log_with_timestamp("="*80)

    if successful:
        total_reviews = sum(r["reviews_processed"] for r in successful)
        total_errors = sum(r["total_errors"] for r in successful)
        log_with_timestamp(f"Total: {total_reviews} reviews, {total_errors} errors")

        all_error_types = defaultdict(int)
        for r in successful:
            for etype, count in r["error_types"].items():
                all_error_types[etype] += count

        log_with_timestamp("Error type distribution:")
        for etype, count in sorted(all_error_types.items(), key=lambda x: x[1], reverse=True):
            log_with_timestamp(f"  {etype}: {count}")

        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "total_users": len(user_ids_to_process),
            "processed_users": len(successful),
            "failed_users": [r["user_id"] for r in failed],
            "total_reviews": total_reviews,
            "total_errors": total_errors,
            "error_type_distribution": dict(all_error_types),
            "user_results": [
                {
                    "user_id": r["user_id"],
                    "reviews_processed": r["reviews_processed"],
                    "reviews_with_errors": r["reviews_with_errors"],
                    "total_errors": r["total_errors"],
                    "error_types": r["error_types"],
                    "detailed_results": r.get("detailed_results", [])
                }
                for r in successful
            ]
        }

        summary_file = os.path.join(OUTPUT_DIR, "writingError_user_profile.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        log_with_timestamp(f"Summary saved to {summary_file}")

    if len(successful) == 0:
        log_with_timestamp("ERROR: No users were successfully processed!")
        sys.exit(1)

    log_with_timestamp("="*80)
    log_with_timestamp("ALL PROCESSING COMPLETE!")
    log_with_timestamp("="*80)


if __name__ == "__main__":
    main()
