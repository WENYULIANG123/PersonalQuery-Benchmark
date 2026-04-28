#!/usr/bin/env python3
"""
Stage 4: Word-Level Error Extraction

识别用户评论中的单词级写作错误，不再进行句法类别归类。

Input:
  - /root/result/personal_query/01_preference_extraction/Baby_Products/stage1_filtered_users_reviews.json

Output:
  - /root/result/personal_query/04_writing_analysis/Baby_Products/writing_error.json

Usage:
  python 04_extract_acl_ccomp_errors_Baby_Products.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
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

INPUT_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/Baby_Products/stage1_filtered_users_reviews.json"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/Baby_Products"
QUERY_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/06_query/Baby_Products/query.json"


# ============================================================================
# Merged 错误提取 Prompt (从 JSON 文件动态加载)
# ============================================================================

PROMPT_CONFIG_FILE = "/home/wlia0047/ar57/wenyu/PersoanlQuery/04_writing_analysis/acl_ccomp_prompts.json"
def load_config():
    """从 JSON 文件加载配置"""
    with open(PROMPT_CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['base_system'], config['user_content_template'], config['max_users'], config['max_reviews'], config['max_workers'], config.get('use_minimaxio', False)

BASE_SYSTEM, USER_CONTENT_TEMPLATE, MAX_USERS, MAX_REVIEWS, MAX_WORKERS, USE_MINIMAXIO = load_config()

# 全局变量
_minimax_client = None
_first_request = True  # 首次请求时打印 system_base（用于创建缓存）


def _log(msg: str):
    """带时间戳的日志打印"""
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_minimax_client():
    """加载 MiniMax API 客户端"""
    global _minimax_client
    if _minimax_client is None:
        from llm_client import MiniMaxAnthropicClient, MiniMaxIOAnthropicClient
        if USE_MINIMAXIO:
            _minimax_client = MiniMaxIOAnthropicClient()
            _log("MiniMaxIO API 客户端初始化完成")
        else:
            _minimax_client = MiniMaxAnthropicClient()
            _log("MiniMax API 客户端初始化完成")


def call_merged_llm(reviews_text: List[str]) -> Dict:
    """调用 MiniMax API 进行单词级错误提取，使用缓存机制"""
    global _minimax_client, _first_request

    if _minimax_client is None:
        load_minimax_client()

    cache_info = {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    # 第一次请求时打印 system_base（创建缓存）
    if _first_request:
        _log(f"[Request] system_base (FIRST REQUEST - cache creation):\n{BASE_SYSTEM}")
        _first_request = False

    # 构建多评论内容
    reviews_content = "\n".join([f"[Review {i}]: {review}" for i, review in enumerate(reviews_text)])
    user_content = USER_CONTENT_TEMPLATE.replace("{reviews_text}", reviews_content)

    try:
        response, cache_info = _minimax_client.call_with_cache(
            system_base=BASE_SYSTEM,
            user_content=user_content,
            max_tokens=8192,
            temperature=0.3
        )

        # 打印缓存信息
        _log(f"[Cache] {cache_info}")

        result_text = response.strip()
        if not result_text:
            return {
                "status": "success",
                "reviews": reviews_text,
                "results": {},
                "has_errors": False
            }

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

            has_errors = False
            for idx_result in result.values():
                if isinstance(idx_result, dict):
                    if idx_result.get("errors"):
                        has_errors = True
                        break

            return {
                "status": "success",
                "reviews": reviews_text,
                "results": result,
                "has_errors": has_errors
            }

        except json.JSONDecodeError:
            return {
                "status": "success",
                "reviews": reviews_text,
                "results": {},
                "has_errors": False
            }

    except Exception as e:
        _log(f"Error calling API: {e}")
        return {
            "status": "error",
            "reviews": reviews_text,
            "results": {},
            "has_errors": False,
            "error": str(e)
        }


# ============================================================================
# 主分析 Pipeline
# ============================================================================

class MergedErrorAnalyzer:
    """单词级错误分析"""

    def __init__(self, analysis_dir: Path = None, reviews_dir: Path = None):
        self.analysis_dir = analysis_dir or Path(OUTPUT_DIR)
        self.reviews_dir = reviews_dir or Path(os.path.dirname(INPUT_FILE))

        self._merged_data = None
        self._users_map = None
        self._load_lock = threading.Lock()

    def _load_merged_file(self):
        """线程安全的懒加载合并文件"""
        if self._users_map is None:
            with self._load_lock:
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

    def _filter_errors(self, errors: List[Dict], filtered_counts: defaultdict) -> List[Dict]:
        """严格过滤错误列表，只保留单词级错误"""
        valid_errors = []
        for error in errors:
            orig = error.get("original", "").strip()
            corr = error.get("corrected", "").strip()

            if not orig or not corr or orig == corr:
                filtered_counts["empty_or_identity_error"] += 1
                continue

            if len(orig.split()) > 1 or len(corr.split()) > 1:
                filtered_counts["non_single_word_error"] += 1
                continue

            cleaned = {
                "original": orig,
                "corrected": corr,
                "confidence": error.get("confidence", 0.8)
            }
            span_text = error.get("span_text", "").strip()
            if span_text:
                cleaned["span_text"] = span_text
            valid_errors.append(cleaned)
        return valid_errors

    def process_user(self, user_id: str, reviews_file: Optional[str] = None, max_reviews: Optional[int] = None) -> dict:
        """处理单个用户：提取单词级写作错误"""

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

            error_count = 0
            filtered_counts = defaultdict(int)
            detailed_results = []

            # 一次请求提取该用户评论中的单词级错误
            review_texts = [review_text for review_text, asin in flattened_reviews]
            llm_result = call_merged_llm(review_texts)

            if llm_result["status"] != "success":
                logger.warning(f"[{user_id}] LLM call failed: {llm_result.get('error', 'Unknown error')}")
                return {"user_id": user_id, "status": "failed", "reason": "llm_call_failed"}

            results_by_idx = llm_result.get("results", {})

            # 遍历每条评论的处理结果
            for review_idx, (review_text, asin) in enumerate(flattened_reviews):
                idx_str = str(review_idx)
                review_result = results_by_idx.get(idx_str, {})

                errors = review_result.get("errors", []) if isinstance(review_result, dict) else []
                valid_errors = self._filter_errors(errors, filtered_counts)
                if valid_errors:
                    error_count += len(valid_errors)
                    detailed_results.append({
                        "asin": asin,
                        "review_index": review_idx,
                        "errors": valid_errors
                    })

            logger.info(f"[{user_id}] Error analysis completed: {len(flattened_reviews)} reviews, {error_count} errors")

            return {
                "user_id": user_id,
                "status": "success",
                "reviews_processed": len(flattened_reviews),
                "error_count": error_count,
                "total_errors": error_count,
                "filtered_counts": dict(filtered_counts),
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
    log_with_timestamp(f"Loading users from merged file: {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    users = data.get('users', [])
    user_ids = [u.get('user_id') for u in users if u.get('user_id')]
    log_with_timestamp(f"Found {len(user_ids)} users in merged file")
    return user_ids


def load_users_from_query_file(query_file: str) -> Set[str]:
    """从 query.json 加载用户ID集合"""
    if not os.path.exists(query_file):
        log_with_timestamp(f"WARNING: Query file not found: {query_file}, skipping query filter")
        return set()
    with open(query_file, 'r', encoding='utf-8') as f:
        query_data = json.load(f)
    user_ids = set()
    for item in query_data:
        uid = item.get('user_id')
        if uid:
            user_ids.add(uid)
    log_with_timestamp(f"Found {len(user_ids)} users in query file")
    return user_ids


def load_completed_user_ids(output_file: str) -> Tuple[Set[str], List[dict]]:
    """加载已完成的用户ID和历史结果，支持 JSON array 和 JSON Lines 格式"""
    if not os.path.exists(output_file):
        return set(), []

    log_with_timestamp(f"Loading existing results from {output_file}...")
    completed_ids = set()
    existing_results = []

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return set(), []

            if content.startswith('['):
                # JSON array format
                data = json.loads(content)
                for item in data:
                    if 'user_id' in item:
                        completed_ids.add(item['user_id'])
                        existing_results.append(item)
            elif content.startswith('{'):
                # JSON Lines format (one JSON object per line)
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            if 'user_id' in item:
                                completed_ids.add(item['user_id'])
                                existing_results.append(item)
                        except:
                            continue

        log_with_timestamp(f"  Found {len(completed_ids)} completed users")
        return completed_ids, existing_results

    except Exception as e:
        log_with_timestamp(f"  Error loading existing results: {e}")
        return set(), []


def write_result_incremental(result: dict, output_file: str):
    """将单个结果追加写入 JSON Lines 格式文件（临时存储）"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')


def append_to_json_array(result: dict, output_file: str):
    """将单个结果追加写入 JSON array 格式文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 读取现有数组或创建新数组
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('[') and content.endswith(']'):
                data = json.loads(content) if content != '[]' else []
            else:
                data = []
    else:
        data = []

    data.append(result)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
    log_with_timestamp("Stage 4: Word-Level Error Extraction")
    log_with_timestamp("="*80)
    log_with_timestamp(f"Input file: {INPUT_FILE}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    log_with_timestamp(f"Max users: {MAX_USERS}")
    log_with_timestamp(f"Max reviews per user: {MAX_REVIEWS}")
    log_with_timestamp(f"Max workers: {MAX_WORKERS}")

    # 加载 query.json 中的用户（跳过 level filter，直接使用 query.json 中的用户）
    user_ids = load_users_from_query_file(QUERY_FILE)

    if not user_ids:
        log_with_timestamp("ERROR: No users to process!")
        sys.exit(1)

    log_with_timestamp(f"从 query.json 加载了 {len(user_ids)} 个用户（跳过 level filter）")

    total_users = len(user_ids)  # query.json 中的总用户数（未限制前）

    if MAX_USERS:
        user_ids = list(user_ids)[:MAX_USERS]
        log_with_timestamp(f"Limited to {MAX_USERS} users for testing")

    user_ids_to_process = sorted(user_ids)

    # 加载已完成的用户，跳过重复
    output_file = os.path.join(OUTPUT_DIR, "writing_error.json")
    completed_ids, existing_results = load_completed_user_ids(output_file)

    completed_count = len(completed_ids)  # 已完成的用户数
    remaining_count = len(user_ids_to_process)  # 待处理的用户数（已去重）
    log_with_timestamp(f"总用户数: {total_users} | 已完成: {completed_count} | 待处理: {remaining_count}")

    # 初始化输出文件为空数组
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_results if existing_results else [], f, ensure_ascii=False, indent=2)

    if completed_ids:
        user_ids_to_process = [uid for uid in user_ids_to_process if uid not in completed_ids]

    analyzer = MergedErrorAnalyzer(
        analysis_dir=Path(OUTPUT_DIR),
        reviews_dir=Path(os.path.dirname(INPUT_FILE))
    )
    analyzer._load_merged_file()

    results = []

    log_with_timestamp("="*80)
    log_with_timestamp(f"Processing {len(user_ids_to_process)} users with word-level error extraction (concurrent={MAX_WORKERS})...")
    log_with_timestamp("="*80)

    total_start = time.time()
    completed_count = 0
    incremental_success = 0

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
                    incremental_success += 1
                    append_to_json_array(result, output_file)
                    log_with_timestamp(
                        f"  [{completed_count}/{len(user_ids_to_process)}] "
                        f"✓ {result['user_id']}: {result['reviews_processed']} reviews, "
                        f"errors={result['error_count']}"
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
        total_errors = sum(r["error_count"] for r in successful)

        log_with_timestamp(f"Total: {total_reviews} reviews, errors={total_errors}")

        all_filtered_counts = defaultdict(int)

        for r in successful:
            for reason, count in r.get("filtered_counts", {}).items():
                all_filtered_counts[reason] += count

        if all_filtered_counts:
            log_with_timestamp("Filtered invalid outputs:")
            for reason, count in sorted(all_filtered_counts.items(), key=lambda x: x[1], reverse=True):
                log_with_timestamp(f"  {reason}: {count}")

        # 只保存有错误的用户
        users_with_errors = [r for r in successful if r["total_errors"] > 0]

        # 只保留控制台统计和主结果文件，不再写 summary 文件

    if len(successful) == 0:
        log_with_timestamp("ERROR: No users were successfully processed!")
        sys.exit(1)

    log_with_timestamp("="*80)
    log_with_timestamp("ALL PROCESSING COMPLETE!")
    log_with_timestamp("="*80)


if __name__ == "__main__":
    main()
