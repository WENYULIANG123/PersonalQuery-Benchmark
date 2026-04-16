#!/usr/bin/env python3
"""
Stage 4 CCOMP: Complement Clause Error Extraction

专注于补语从句(CCOMP)相关错误的提取，错误类型分为4类：
1. clause-shell typo: think→thikn, believe→belive 等壳层词拼写错误
2. complement-linking error: that/if/whether 连接处错误
3. modal/auxiliary distortion: would→woudl, could→cuold 等情态动词错误
4. clause-boundary structural writing error: 从句边界结构错误（如主谓不一致）

Input:
  - /home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/Pet_Supplies/stage1_filtered_users_reviews.json

Output:
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/Pet_Supplies/ccomp_error.json

Usage:
  python 04_extract_ccomp_errors_Pet_Supplies.py
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

INPUT_FILE = "/fs04/ar57/wenyu/result/personal_query/01_preference_extraction/Pet_Supplies/stage1_filtered_users_reviews.json"
OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/04_writing_analysis/Pet_Supplies"
MAX_USERS = 10
MAX_REVIEWS = 10
MAX_WORKERS = 1

# CCOMP用户画像文件（从中获取用户ID）
CCOMP_PROFILES_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Pet_Supplies/ccomp_user_profiles.json"


# ============================================================================
# CCOMP 错误提取 Prompt (从 JSON 文件动态加载)
# ============================================================================

PROMPT_CONFIG_FILE = "/home/wlia0047/ar57/wenyu/PersoanlQuery/04_writing_analysis/ccomp_prompts.json"

def load_prompts():
    """从 JSON 文件加载 prompt 配置"""
    with open(PROMPT_CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['base_system'], config['user_content_template']

BASE_SYSTEM, USER_CONTENT_TEMPLATE = load_prompts()

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
        from llm_client import MiniMaxAnthropicClient
        _minimax_client = MiniMaxAnthropicClient()
        _log("MiniMax API 客户端初始化完成")


def call_ccomp_llm(reviews_text: List[str]) -> Dict:
    """调用 MiniMax API 进行CCOMP错误提取，使用缓存机制"""
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

            return {
                "status": "success",
                "reviews": reviews_text,
                "results": result,
                "has_errors": any(v.get("regions") for v in result.values() if isinstance(v, dict))
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
# CCOMP 错误类型分类器
# ============================================================================

class CCOMPErrorClassifier:
    """CCOMP错误类型分类"""

    ERROR_TYPE_MAPPING = {
        "clause_shell_typo": "clause壳层词拼写错误",
        "complement_linking_error": "从句连接错误",
        "modal_distortion": "情态动词错误",
        "clause_boundary_error": "从句边界结构错误"
    }

    # 常见的 clause-shell 词
    CLAUSE_SHELL_WORDS = {
        'think', 'believe', 'know', 'feel', 'want', 'hope', 'expect',
        'suppose', 'noticed', 'said', 'saw', 'heard', 'found',
        'because', 'whether', 'if', 'though', 'although', 'since',
        'that', 'what', 'whatever', 'which'
    }

    # 常见的情态动词
    MODAL_WORDS = {
        'would', 'could', 'should', 'might', 'may', 'will', 'shall',
        'must', 'ought', 'used'
    }

    @staticmethod
    def classify_error_type(original: str, corrected: str, context: str = "") -> str:
        """根据原始词和修正词判断CCOMP错误类型"""
        orig_lower = original.lower().strip()
        corr_lower = corrected.lower().strip()

        # 1. 检查是否是 clause-shell 词 typo
        if CCOMPErrorClassifier._is_clause_shell_typo(orig_lower, corr_lower):
            return "clause_shell_typo"

        # 2. 检查是否是情态动词错误
        if CCOMPErrorClassifier._is_modal_distortion(orig_lower, corr_lower):
            return "modal_distortion"

        # 3. 检查是否是从句连接错误 (that, if, whether)
        if CCOMPErrorClassifier._is_complement_linking_error(orig_lower, corr_lower):
            return "complement_linking_error"

        # 4. 检查是否是从句边界结构错误 (通常涉及动词形式变化)
        if CCOMPErrorClassifier._is_clause_boundary_error(orig_lower, corr_lower):
            return "clause_boundary_error"

        # 如果原始词是 clause-shell 词但拼写错了
        if orig_lower in CCOMPErrorClassifier.CLAUSE_SHELL_WORDS:
            return "clause_shell_typo"

        # 如果原始词是情态动词但拼写错了
        if orig_lower in CCOMPErrorClassifier.MODAL_WORDS:
            return "modal_distortion"

        # 默认归类为 clause-shell typo（最常见的CCOMP错误）
        return "clause_shell_typo"

    @staticmethod
    def _is_clause_shell_typo(s1: str, s2: str) -> bool:
        """判断是否为clause-shell词拼写错误"""
        # 常见clause-shell词的拼写变体
        shell_typo_map = {
            'thikn': 'think', 'thnk': 'think',
            'belive': 'believe', 'beleive': 'believe',
            'noticd': 'noticed', 'notcie': 'notice', 'notic': 'notice',
            'saod': 'said', 'sadi': 'said', 'soid': 'said',
            'becuase': 'because', 'becuse': 'because',
            'wether': 'whether', 'weither': 'whether',
            'woudl': 'would', 'woud': 'would',
            'shoudl': 'should', 'shuold': 'should',
            'taht': 'that', 'thta': 'that',
            'coudl': 'could', 'cuold': 'could',
        }
        return s1 in shell_typo_map and shell_typo_map[s1] == s2

    @staticmethod
    def _is_modal_distortion(s1: str, s2: str) -> bool:
        """判断是否为情态动词错误"""
        modal_typo_map = {
            'woudl': 'would', 'woud': 'would', 'woul': 'would',
            'coudl': 'could', 'cuold': 'could', 'coul': 'could',
            'shoudl': 'should', 'shoud': 'should', 'shuold': 'should',
            'migth': 'might', 'mitgh': 'might',
            'mayd': 'may',
        }
        return s1 in modal_typo_map and modal_typo_map[s1] == s2

    @staticmethod
    def _is_complement_linking_error(s1: str, s2: str) -> bool:
        """判断是否为从句连接错误"""
        # that/if/whether 的拼写错误
        linking_errors = {
            'taht': 'that', 'thta': 'that',
            'tif': 'if', 'fi': 'if',
            'wether': 'whether', 'whetehr': 'whether', 'wherher': 'whether',
        }
        return s1 in linking_errors and linking_errors[s1] == s2

    @staticmethod
    def _is_clause_boundary_error(s1: str, s2: str) -> bool:
        """判断是否为从句边界结构错误（如主谓不一致）"""
        # 常见的主谓不一致错误（动词形式错误）
        # 例如: "work" -> "works", "break" -> "breaks", "leak" -> "leaks"
        boundary_error_patterns = [
            # 第三人称单数错误
            ('work', 'works'), ('think', 'thinks'), ('believe', 'believes'),
            ('know', 'knows'), ('feel', 'feels'), ('want', 'wants'),
            ('hope', 'hopes'), ('expect', 'expects'), ('suppose', 'supposes'),
            # 过去式错误
            ('break', 'broke'), ('think', 'thought'), ('know', 'knew'),
            ('feel', 'felt'), ('want', 'wanted'), ('hope', 'hoped'),
            # 进行时错误
            ('work', 'working'), ('break', 'breaking'), ('leak', 'leaking'),
        ]
        return (s1, s2) in boundary_error_patterns or (s2, s1) in boundary_error_patterns


# ============================================================================
# 主分析 Pipeline
# ============================================================================

class CCOMPErrorAnalyzer:
    """CCOMP错误分析"""

    def __init__(self, analysis_dir: Path = None, reviews_dir: Path = None):
        self.analysis_dir = analysis_dir or Path(OUTPUT_DIR)
        self.reviews_dir = reviews_dir or Path(os.path.dirname(INPUT_FILE))
        self.classifier = CCOMPErrorClassifier()

        self._merged_data = None
        self._users_map = None
        self._load_lock = threading.Lock()

    def _load_merged_file(self):
        """线程安全的懒加载合并文件"""
        if self._users_map is None:
            with self._load_lock:
                # 双重检查锁定
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
        """处理单个用户：提取CCOMP错误"""

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

            # 将所有评论文本提取出来，一次请求发送
            review_texts = [review_text for review_text, asin in flattened_reviews]
            llm_result = call_ccomp_llm(review_texts)

            if llm_result["status"] != "success":
                logger.warning(f"[{user_id}] LLM call failed: {llm_result.get('error', 'Unknown error')}")
                return {"user_id": user_id, "status": "failed", "reason": "llm_call_failed"}

            results_by_idx = llm_result.get("results", {})

            # 遍历每条评论的处理结果
            for review_idx, (review_text, asin) in enumerate(flattened_reviews):
                idx_str = str(review_idx)
                review_result = results_by_idx.get(idx_str, {})
                regions = review_result.get("regions", []) if isinstance(review_result, dict) else []

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

                        # 跳过 corrected 是短语的情况（必须是单词）
                        if len(corr.split()) > 1:
                            continue

                        # 标准化错误类型
                        error_type = error.get("error_type", "")
                        if not error_type:
                            error_type = self.classifier.classify_error_type(orig, corr, span_text)

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

            logger.info(f"[{user_id}] CCOMP error analysis completed: {len(flattened_reviews)} reviews, {total_errors} errors")

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
    if CCOMP_PROFILES_FILE and os.path.exists(CCOMP_PROFILES_FILE):
        log_with_timestamp(f"Loading user IDs from CCOMP profiles: {CCOMP_PROFILES_FILE}...")
        with open(CCOMP_PROFILES_FILE, 'r', encoding='utf-8') as f:
            ccomp_data = json.load(f)
        user_ids = [u.get('user_id') for u in ccomp_data if u.get('user_id')]
        log_with_timestamp(f"Found {len(user_ids)} users in CCOMP profiles file")
        return user_ids

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
    log_with_timestamp("Stage 4 CCOMP: Complement Clause Error Extraction")
    log_with_timestamp("="*80)
    log_with_timestamp(f"Input file: {INPUT_FILE}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    log_with_timestamp(f"Max users: {MAX_USERS}")
    log_with_timestamp(f"Max reviews per user: {MAX_REVIEWS}")
    log_with_timestamp(f"Max workers: {MAX_WORKERS}")

    # 直接从merged文件获取所有用户ID
    log_with_timestamp(f"Loading users from merged file: {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_user_ids = [u.get('user_id') for u in data.get('users', []) if u.get('user_id')]
    log_with_timestamp(f"Found {len(all_user_ids)} users in merged file")

    if not all_user_ids:
        log_with_timestamp("ERROR: No users to process!")
        sys.exit(1)

    # 限制用户数量
    user_ids_to_process = sorted(list(set(all_user_ids)))
    if MAX_USERS:
        user_ids_to_process = user_ids_to_process[:MAX_USERS]
        log_with_timestamp(f"Limited to {MAX_USERS} users for testing")

    analyzer = CCOMPErrorAnalyzer(
        analysis_dir=Path(OUTPUT_DIR),
        reviews_dir=Path(os.path.dirname(INPUT_FILE))
    )
    # 预加载merged文件，避免多线程竞态条件
    analyzer._load_merged_file()

    results = []

    log_with_timestamp("="*80)
    log_with_timestamp(f"Processing {len(user_ids_to_process)} users with CCOMP error extraction (concurrent={MAX_WORKERS})...")
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

        # 只保存有错误的用户
        users_with_errors = [r for r in successful if r["total_errors"] > 0]
        users_with_errors_ids = [r["user_id"] for r in users_with_errors]

        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "total_users": len(user_ids_to_process),
            "processed_users": len(successful),
            "users_with_errors": len(users_with_errors),
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
                for r in users_with_errors
            ]
        }

        summary_file = os.path.join(OUTPUT_DIR, "ccomp_error.json")
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
