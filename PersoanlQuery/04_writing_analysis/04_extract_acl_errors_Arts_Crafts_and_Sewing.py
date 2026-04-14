#!/usr/bin/env python3
"""
Stage 4 (V2): Detailed Syntactic Error Extraction

基于句法结构的精细错误提取，专注于"描述细节的零件"区域：
1. acl/relcl/advcl 子树
2. 名词短语修饰链
3. 属性短语
4. 产品 feature span

在这些区域中优先提取：
- OOV 或疑似拼错词
- 复合词边界异常
- 局部短语连接异常
- 属性词 typo
- 修饰词 typo
- 复合词/连字符/空格变体
- 修饰链内部局部连接错误
- 名词短语内部的局部词形错误

Input:
  - /home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/stage1_filtered_users_reviews.json

Output:
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/detailed_errors_{user_id}.json

Usage:
  python 04_extract_detailed_errors.py
"""

import json
import os
import sys
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

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

INPUT_FILE = "/fs04/ar57/wenyu/result/personal_query/01_preference_extraction/Arts_Crafts_and_Sewing/stage1_filtered_users_reviews.json"
OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/04_writing_analysis/Arts_Crafts_and_Sewing"
MAX_USERS = 1
MAX_REVIEWS = 5
MAX_WORKERS = 50

# ACL用户画像文件（从中获取用户ID）
ACL_PROFILES_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/acl_user_profiles.json"


# ============================================================================
# 区域感知错误提取 Prompt
# ============================================================================

DETAILED_ERROR_TEMPLATE = """You are an expert at identifying subtle errors in product reviews, focusing on "detail description parts" - the small words that describe product features.

Target error regions (优先级从高到低):
1. acl/relcl/advcl subtrees - adverbial clauses, relative clauses
2. Noun phrase modifier chains - sequences like "high quality durable plastic case"
3. Attribute phrases - "X's Y", "the Y of X"
4. Product feature spans - specific product aspects being described

Priority error types to extract:
1. Attribute word typos - wrong modifier words (e.g., "convenient" -> "conveniently" when modifying verb)
2. Modifier word typos - errors in descriptive words (e.g., "smoth" -> "smooth", "diffrent" -> "different")
3. Local inflection errors inside noun phrases - (e.g., "types" when should be "type", "making" when should be "make")

IMPORTANT: Only extract errors that are:
- Single words OR small multi-word spans (max 3 words)
- Located in the detail-description parts (NOT main subject/predicate)
- Related to: typos, spelling, local inflection errors

IGNORE:
- Compound word/hyphen/space variants - (e.g., "lifelike" vs "life-like" vs "life like", "works" vs "work's")
- Local connection errors in modifier chains - missing hyphens, extra spaces, wrong boundaries
- Subject-verb agreement errors
- Whole-sentence rewrites
- Punctuation issues
- Capitalization (unless clearly a typo)
- Plural forms for multiple instances: "ratings", "settings", "options", "types", "modes", "features" when describing multiple aspects/dimensions (e.g., "endurance ratings" is valid)

Return JSON format:
{
  "regions": [
    {
      "region_type": "acl|relcl|advcl|np_modifier|attribute|feature_span",
      "span_text": "the original text span",
      "errors": [
        {
          "original": "original word(s)",
          "corrected": "corrected word(s)",
          "error_type": "attribute_typo|modifier_typo|np_inflection",
          "confidence": 0.0-1.0
        }
      ]
    }
  ]
}

If no detail-description errors found, return {"regions": []}.

Analyze this product review and identify errors in detail-description parts:

{review_text}
"""


def call_detailed_llm(review_text: str) -> Dict:
    """调用LLM进行区域感知错误提取，支持速率限制重试"""
    import time
    from llm_client import MiniMaxAnthropicClient

    prompt = f"""<s>[INST] {DETAILED_ERROR_TEMPLATE}
[/INST]"""

    wait_seconds = 5
    max_retries = 20

    for attempt in range(max_retries):
        try:
            client = MiniMaxAnthropicClient(model='MiniMax-M2.7-highspeed')
            thinking, text = client.call_with_thinking(prompt, max_tokens=8192, temperature=0.3)
            result_text = text.strip()

            if not result_text:
                result_text = '{"regions": []}'

            try:
                if "```json" in result_text:
                    start = result_text.find("```json") + 7
                    end = result_text.find("```", start)
                    result_text = result_text[start:end].strip()
                elif "```" in result_text:
                    start = result_text.find("```") + 3
                    end = result_text.find("```", start)
                    result_text = result_text[start:end].strip()

                result = json.loads(result_text)
                regions = result.get("regions", [])

                return {
                    "status": "success",
                    "original": review_text,
                    "regions": regions,
                    "has_errors": len(regions) > 0 and any(r.get("errors") for r in regions)
                }

            except json.JSONDecodeError:
                return {
                    "status": "success",
                    "original": review_text,
                    "regions": [],
                    "has_errors": False
                }

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = 'rate' in error_str or 'limit' in error_str or '429' in error_str or 'too many request' in error_str
            if not is_rate_limit:
                return {
                    "status": "error",
                    "original": review_text,
                    "regions": [],
                    "has_errors": False,
                    "error": str(e)
                }
            if attempt < max_retries - 1:
                logger.warning(f"Rate limit, retrying in {wait_seconds}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_seconds)
                wait_seconds += 5
            else:
                return {
                    "status": "error",
                    "original": review_text,
                    "regions": [],
                    "has_errors": False,
                    "error": str(e)
                }


# ============================================================================
# 错误类型分类器
# ============================================================================

class DetailedErrorClassifier:
    """区域感知错误类型分类"""

    ERROR_TYPE_MAPPING = {
        "attribute_typo": "属性词错误",
        "modifier_typo": "修饰词错误",
        "np_inflection": "名词短语词形错误"
    }

    @staticmethod
    def classify_error_type(original: str, corrected: str) -> str:
        """根据原始词和修正词判断错误类型"""
        orig_lower = original.lower().strip()
        corr_lower = corrected.lower().strip()

        # 属性词 vs 副词混淆
        if DetailedErrorClassifier._is_attribute_error(orig_lower, corr_lower):
            return "attribute_typo"

        # 拼写错误模式（编辑距离小）
        if DetailedErrorClassifier._is_spelling_error(orig_lower, corr_lower):
            return "modifier_typo"

        # 默认归类为修饰词错误
        return "modifier_typo"

    @staticmethod
    def _is_spelling_error(s1: str, s2: str) -> bool:
        """判断是否为拼写错误（编辑距离 <= 2）"""
        if len(s1) < 2 or len(s2) < 2:
            return False
        edit_dist = DetailedErrorClassifier._edit_distance(s1, s2)
        return edit_dist <= 2 and edit_dist > 0

    @staticmethod
    def _is_attribute_error(s1: str, s2: str) -> bool:
        """判断是否为属性词错误"""
        # 常见的属性词/副词对
        attribute_pairs = {
            ("convenient", "conveniently"),
            ("easy", "easily"),
            ("quick", "quickly"),
            ("slow", "slowly"),
            ("careful", "carefully"),
            ("beauty", "beautiful"),
        }
        return (s1, s2) in attribute_pairs or (s2, s1) in attribute_pairs

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return DetailedErrorClassifier._edit_distance(s2, s1)
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
# 主分析 Pipeline
# ============================================================================

class DetailedErrorAnalyzer:
    """基于区域感知的精细错误分析"""

    def __init__(self, analysis_dir: Path = None, reviews_dir: Path = None):
        self.analysis_dir = analysis_dir or Path(OUTPUT_DIR)
        self.reviews_dir = reviews_dir or Path(os.path.dirname(INPUT_FILE))
        self.classifier = DetailedErrorClassifier()

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
        """处理单个用户：提取区域感知错误"""

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
            region_type_counts = defaultdict(int)
            detailed_results = []

            for review_idx, (review_text, asin) in enumerate(flattened_reviews):
                original = review_text

                llm_result = call_detailed_llm(original)

                if llm_result["status"] != "success":
                    logger.warning(f"[{user_id}] Review {review_idx}: {llm_result.get('error', 'Unknown error')}")
                    continue

                regions = llm_result.get("regions", [])

                # 过滤并标准化错误
                for region in regions:
                    region_type = region.get("region_type", "unknown")
                    span_text = region.get("span_text", "")
                    errors = region.get("errors", [])

                    if not errors:
                        continue

                    valid_errors = []
                    for error in errors:
                        orig = error.get("original", "").strip()
                        corr = error.get("corrected", "").strip()

                        # 跳过空或相同的情况
                        if not orig or not corr or orig == corr:
                            continue

                        # 标准化错误类型
                        error_type = error.get("error_type", "")
                        if not error_type:
                            error_type = self.classifier.classify_error_type(orig, corr)

                        valid_errors.append({
                            "original": orig,
                            "corrected": corr,
                            "error_type": error_type,
                            "confidence": error.get("confidence", 0.8)
                        })

                        error_type_counts[error_type] += 1

                    if valid_errors:
                        region_type_counts[region_type] += 1
                        detailed_results.append({
                            "asin": asin,
                            "region_type": region_type,
                            "span_text": span_text,
                            "errors": valid_errors
                        })

                if any(r.get("errors") for r in regions if r.get("errors")):
                    reviews_with_errors += 1

                total_errors += sum(len(r.get("errors", [])) for r in regions if r.get("errors"))

                if (review_idx + 1) % 10 == 0:
                    logger.info(f"[{user_id}] Progress: {review_idx + 1}/{len(flattened_reviews)} reviews, {total_errors} errors found")

            logger.info(f"[{user_id}] Detailed error analysis completed: {len(flattened_reviews)} reviews, {total_errors} errors")

            return {
                "user_id": user_id,
                "status": "success",
                "reviews_processed": len(flattened_reviews),
                "reviews_with_errors": reviews_with_errors,
                "total_errors": total_errors,
                "error_types": dict(error_type_counts),
                "region_types": dict(region_type_counts),
                "detailed_results": detailed_results
            }

        except Exception as e:
            logger.error(f"[{user_id}] Processing failed: {str(e)}")
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
    # 直接从merged file获取用户，不再依赖ACL profiles
    log_with_timestamp(f"Loading users from merged file: {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    users = data.get('users', [])
    user_ids = [u.get('user_id') for u in users if u.get('user_id')]
    log_with_timestamp(f"Found {len(user_ids)} users in merged file")
    return user_ids


def validate_users_from_merged_file(input_file: str, user_ids: List[str]) -> Set[str]:
    """验证合并文件中的用户"""
    log_with_timestamp("Validating users from merged file...")
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
            log_with_timestamp(f"  User {user_id}: {target_count} reviews")
    log_with_timestamp(f"Found {len(existing_users)} valid users")
    return existing_users


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_with_timestamp("="*80)
    log_with_timestamp("Stage 4 V2: Detailed Syntactic Error Extraction")
    log_with_timestamp("="*80)
    log_with_timestamp(f"Input file: {INPUT_FILE}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    log_with_timestamp(f"Max users: {MAX_USERS}")
    log_with_timestamp(f"Max reviews per user: {MAX_REVIEWS}")
    log_with_timestamp(f"Max workers: {MAX_WORKERS}")

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

    analyzer = DetailedErrorAnalyzer(
        analysis_dir=Path(OUTPUT_DIR),
        reviews_dir=Path(os.path.dirname(INPUT_FILE))
    )

    results = []

    log_with_timestamp("="*80)
    log_with_timestamp(f"Processing {len(user_ids_to_process)} users with detailed error extraction (concurrent={MAX_WORKERS})...")
    log_with_timestamp("="*80)

    total_start = time.time()
    completed_count = 0

    def process_one_user(user_id):
        return analyzer.process_user(user_id, None, MAX_REVIEWS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_user, uid): uid for uid in user_ids_to_process}

        for future in as_completed(futures):
            uid = futures[future]
            completed_count += 1
            try:
                result = future.result()
                results.append(result)

                if result["status"] == "success":
                    log_with_timestamp(
                        f"  [{completed_count}/{len(user_ids_to_process)}] "
                        f"✓ {result['user_id']}: {result['reviews_processed']} reviews, "
                        f"{result['total_errors']} errors"
                    )
                else:
                    log_with_timestamp(
                        f"  [{completed_count}/{len(user_ids_to_process)}] "
                        f"✗ {result['user_id']}: {result.get('reason', 'Unknown error')}"
                    )
            except Exception as e:
                log_with_timestamp(f"  [{completed_count}/{len(user_ids_to_process)}] ✗ {uid}: {str(e)}")
                results.append({"user_id": uid, "status": "failed", "reason": str(e)})

    total_elapsed = time.time() - total_start

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    log_with_timestamp("="*80)
    log_with_timestamp(f"Completed: {len(successful)} success, {len(failed)} failed ({total_elapsed:.1f}s)")
    log_with_timestamp("="*80)

    if successful:
        total_reviews = sum(r["reviews_processed"] for r in successful)
        total_errors = sum(r["total_errors"] for r in successful)
        log_with_timestamp(f"Total: {total_reviews} reviews, {total_errors} errors")

        all_error_types = defaultdict(int)
        all_region_types = defaultdict(int)
        for r in successful:
            for etype, count in r["error_types"].items():
                all_error_types[etype] += count
            for rtype, count in r["region_types"].items():
                all_region_types[rtype] += count

        log_with_timestamp("Error type distribution:")
        for etype, count in sorted(all_error_types.items(), key=lambda x: x[1], reverse=True):
            log_with_timestamp(f"  {etype}: {count}")

        log_with_timestamp("Region type distribution:")
        for rtype, count in sorted(all_region_types.items(), key=lambda x: x[1], reverse=True):
            log_with_timestamp(f"  {rtype}: {count}")

        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "total_users": len(user_ids_to_process),
            "processed_users": len(successful),
            "failed_users": [r["user_id"] for r in failed],
            "total_reviews": total_reviews,
            "total_errors": total_errors,
            "error_type_distribution": dict(all_error_types),
            "region_type_distribution": dict(all_region_types),
            "user_results": [
                {
                    "user_id": r["user_id"],
                    "reviews_processed": r["reviews_processed"],
                    "reviews_with_errors": r["reviews_with_errors"],
                    "total_errors": r["total_errors"],
                    "error_types": r["error_types"],
                    "region_types": r["region_types"],
                    "detailed_results": r.get("detailed_results", [])
                }
                for r in successful
                if r["total_errors"] > 0
            ]
        }

        summary_file = os.path.join(OUTPUT_DIR, "acl_error.json")
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
