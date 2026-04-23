#!/usr/bin/env python3
"""
Stage 4: Combined ACL + CCOMP Error Extraction

同时识别 ACL (Attribute/Modifier) 和 CCOMP (Complement Clause) 错误：
1. ACL: 属性词/修饰词错误 - convenient, smooth, different 等词的拼写错误
2. CCOMP: 补语从句错误 - think/believe/would/that 等词的拼写错误

Input:
  - /root/result/personal_query/01_preference_extraction/Pet_Supplies/stage1_filtered_users_reviews.json

Output:
  - /root/result/personal_query/04_writing_analysis/Pet_Supplies/acl_ccomp_error.json

Usage:
  python 04_extract_acl_ccomp_errors_Pet_Supplies.py
"""

import json
import os
import sys
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
import time

sys.path.insert(0, '/workspace/PersonalQuery/PersoanlQuery')

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

INPUT_FILE = "/workspace/result/personal_query/01_preference_extraction/Pet_Supplies/stage1_filtered_users_reviews.json"
LEVEL_FILE = "/workspace/result/personal_query/05_syntactic_analysis/Pet_Supplies/level.json"
OUTPUT_DIR = "/workspace/result/personal_query/04_writing_analysis/Pet_Supplies"
QUERY_FILE = "/workspace/result/personal_query/06_query/Pet_Supplies/query.json"


# ============================================================================
# Merged 错误提取 Prompt (从 JSON 文件动态加载)
# ============================================================================

PROMPT_CONFIG_FILE = "/workspace/PersonalQuery/PersoanlQuery/04_writing_analysis/acl_ccomp_prompts.json"

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
    """调用 MiniMax API 进行 ACL+CCOMP 错误提取，使用缓存机制"""
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
                    if idx_result.get("acl_regions") or idx_result.get("ccomp_regions"):
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
# ACL 错误类型分类器
# ============================================================================

class ACLErrorClassifier:
    """ACL错误类型分类"""

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
        if ACLErrorClassifier._is_attribute_error(orig_lower, corr_lower):
            return "attribute_typo"

        # 拼写错误模式（编辑距离小）
        if ACLErrorClassifier._is_spelling_error(orig_lower, corr_lower):
            return "modifier_typo"

        # 默认归类为修饰词错误
        return "modifier_typo"

    @staticmethod
    def _is_spelling_error(s1: str, s2: str) -> bool:
        """判断是否为拼写错误（编辑距离 <= 2）"""
        if len(s1) < 2 or len(s2) < 2:
            return False
        edit_dist = ACLErrorClassifier._edit_distance(s1, s2)
        return edit_dist <= 2 and edit_dist > 0

    @staticmethod
    def _is_attribute_error(s1: str, s2: str) -> bool:
        """判断是否为属性词错误"""
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
            return ACLErrorClassifier._edit_distance(s2, s1)
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

        # 4. 检查是否是从句边界结构错误
        if CCOMPErrorClassifier._is_clause_boundary_error(orig_lower, corr_lower):
            return "clause_boundary_error"

        # 如果原始词是 clause-shell 词但拼写错了
        if orig_lower in CCOMPErrorClassifier.CLAUSE_SHELL_WORDS:
            return "clause_shell_typo"

        # 如果原始词是情态动词但拼写错了
        if orig_lower in CCOMPErrorClassifier.MODAL_WORDS:
            return "modal_distortion"

        return "clause_shell_typo"

    @staticmethod
    def _is_clause_shell_typo(s1: str, s2: str) -> bool:
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
        linking_errors = {
            'taht': 'that', 'thta': 'that',
            'tif': 'if', 'fi': 'if',
            'wether': 'whether', 'whetehr': 'whether', 'wherher': 'whether',
        }
        return s1 in linking_errors and linking_errors[s1] == s2

    @staticmethod
    def _is_clause_boundary_error(s1: str, s2: str) -> bool:
        boundary_error_patterns = [
            ('work', 'works'), ('think', 'thinks'), ('believe', 'believes'),
            ('know', 'knows'), ('feel', 'feels'), ('want', 'wants'),
            ('hope', 'hopes'), ('expect', 'expects'), ('suppose', 'supposes'),
            ('break', 'broke'), ('think', 'thought'), ('know', 'knew'),
            ('feel', 'felt'), ('want', 'wanted'), ('hope', 'hoped'),
            ('work', 'working'), ('break', 'breaking'), ('leak', 'leaking'),
        ]
        return (s1, s2) in boundary_error_patterns or (s2, s1) in boundary_error_patterns


# ============================================================================
# 主分析 Pipeline
# ============================================================================

class MergedErrorAnalyzer:
    """ACL+CCOMP 错误分析"""

    def __init__(self, analysis_dir: Path = None, reviews_dir: Path = None):
        self.analysis_dir = analysis_dir or Path(OUTPUT_DIR)
        self.reviews_dir = reviews_dir or Path(os.path.dirname(INPUT_FILE))
        self.acl_classifier = ACLErrorClassifier()
        self.ccomp_classifier = CCOMPErrorClassifier()

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

    def _filter_errors(self, errors: List[Dict], error_category: str) -> List[Dict]:
        """过滤错误列表，只保留单词级别的修正"""
        valid_errors = []
        for error in errors:
            orig = error.get("original", "").strip()
            corr = error.get("corrected", "").strip()

            if not orig or not corr or orig == corr:
                continue

            # 跳过 corrected 是短语的情况（必须是单词）
            if len(corr.split()) > 1:
                continue

            error_type = error.get("error_type", "")
            if not error_type:
                if error_category == "acl":
                    error_type = self.acl_classifier.classify_error_type(orig, corr)
                else:
                    error_type = self.ccomp_classifier.classify_error_type(orig, corr, "")

            valid_errors.append({
                "original": orig,
                "corrected": corr,
                "error_type": error_type,
                "confidence": error.get("confidence", 0.8)
            })
        return valid_errors

    def process_user(self, user_id: str, reviews_file: Optional[str] = None, max_reviews: Optional[int] = None) -> dict:
        """处理单个用户：同时提取 ACL 和 CCOMP 错误"""

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

            acl_error_count = 0
            ccomp_error_count = 0
            acl_error_type_counts = defaultdict(int)
            ccomp_error_type_counts = defaultdict(int)
            acl_region_type_counts = defaultdict(int)
            ccomp_region_type_counts = defaultdict(int)
            detailed_results = []

            # 一次请求同时提取 ACL 和 CCOMP 错误
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

                # 处理 ACL 错误
                acl_regions = review_result.get("acl_regions", []) if isinstance(review_result, dict) else []
                for region in acl_regions:
                    region_type = region.get("region_type", "unknown")
                    span_text = region.get("span_text", "")
                    errors = region.get("errors", [])

                    valid_errors = self._filter_errors(errors, "acl")
                    if valid_errors:
                        acl_region_type_counts[region_type] += 1
                        for err in valid_errors:
                            acl_error_type_counts[err["error_type"]] += 1
                        detailed_results.append({
                            "asin": asin,
                            "error_category": "acl",
                            "region_type": region_type,
                            "span_text": span_text,
                            "errors": valid_errors
                        })

                # 处理 CCOMP 错误
                ccomp_regions = review_result.get("ccomp_regions", []) if isinstance(review_result, dict) else []
                for region in ccomp_regions:
                    region_type = region.get("region_type", "unknown")
                    span_text = region.get("span_text", "")
                    errors = region.get("errors", [])

                    valid_errors = self._filter_errors(errors, "ccomp")
                    if valid_errors:
                        ccomp_region_type_counts[region_type] += 1
                        for err in valid_errors:
                            ccomp_error_type_counts[err["error_type"]] += 1
                        detailed_results.append({
                            "asin": asin,
                            "error_category": "ccomp",
                            "region_type": region_type,
                            "span_text": span_text,
                            "errors": valid_errors
                        })

            acl_total = sum(acl_error_type_counts.values())
            ccomp_total = sum(ccomp_error_type_counts.values())

            logger.info(f"[{user_id}] Merged error analysis completed: {len(flattened_reviews)} reviews, ACL={acl_total} errors, CCOMP={ccomp_total} errors")

            return {
                "user_id": user_id,
                "status": "success",
                "reviews_processed": len(flattened_reviews),
                "acl_error_count": acl_total,
                "ccomp_error_count": ccomp_total,
                "total_errors": acl_total + ccomp_total,
                "acl_error_types": dict(acl_error_type_counts),
                "ccomp_error_types": dict(ccomp_error_type_counts),
                "acl_region_types": dict(acl_region_type_counts),
                "ccomp_region_types": dict(ccomp_region_type_counts),
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


def load_level_filtered_users(level_file: str) -> Set[str]:
    """从 level.json 加载用户，仅返回 acl_level > 0 或 ccomp_level > 0 的用户"""
    log_with_timestamp(f"Loading users from level file: {level_file}...")
    if not os.path.exists(level_file):
        log_with_timestamp(f"WARNING: Level file not found: {level_file}, skipping level filter")
        return set()
    with open(level_file, 'r', encoding='utf-8') as f:
        level_data = json.load(f)
    filtered_users = set()
    for item in level_data:
        uid = item.get('user_id')
        acl_level = item.get('acl_level', 0)
        ccomp_level = item.get('ccomp_level', 0)
        if uid and (acl_level > 0 or ccomp_level > 0):
            filtered_users.add(uid)
    log_with_timestamp(f"Found {len(filtered_users)} users with acl_level > 0 or ccomp_level > 0")
    return filtered_users


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
    log_with_timestamp("Stage 4: Combined ACL + CCOMP Error Extraction")
    log_with_timestamp("="*80)
    log_with_timestamp(f"Input file: {INPUT_FILE}")
    log_with_timestamp(f"Level file: {LEVEL_FILE}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    log_with_timestamp(f"Max users: {MAX_USERS}")
    log_with_timestamp(f"Max reviews per user: {MAX_REVIEWS}")
    log_with_timestamp(f"Max workers: {MAX_WORKERS}")

    # 加载 level 过滤后的用户
    level_users = load_level_filtered_users(LEVEL_FILE)

    # 加载 query.json 中的用户
    query_users = load_users_from_query_file(QUERY_FILE)

    user_ids = load_users_from_merged_file(INPUT_FILE)

    if not user_ids:
        log_with_timestamp("ERROR: No users to process!")
        sys.exit(1)

    # 如果有 level 过滤，则取交集
    if level_users:
        user_ids = [uid for uid in user_ids if uid in level_users]
        log_with_timestamp(f"After level filter: {len(user_ids)} users")

    # 如果有 query 过滤，则取交集（确保只处理 query.json 中存在的用户）
    if query_users:
        user_ids = [uid for uid in user_ids if uid in query_users]
        log_with_timestamp(f"After query filter: {len(user_ids)} users")

    if MAX_USERS:
        user_ids = user_ids[:MAX_USERS]
        log_with_timestamp(f"Limited to {MAX_USERS} users for testing")

    existing_users = validate_users_from_merged_file(INPUT_FILE, user_ids)

    if not existing_users:
        log_with_timestamp("ERROR: No valid users found!")
        sys.exit(1)

    user_ids_to_process = sorted(list(existing_users))

    # 加载已完成的用户，跳过重复
    output_file = os.path.join(OUTPUT_DIR, "acl_ccomp_error.json")
    completed_ids, existing_results = load_completed_user_ids(output_file)

    # 初始化输出文件为空数组
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_results if existing_results else [], f, ensure_ascii=False, indent=2)

    if completed_ids:
        original_count = len(user_ids_to_process)
        user_ids_to_process = [uid for uid in user_ids_to_process if uid not in completed_ids]
        log_with_timestamp(f"跳过 {len(completed_ids)} 个已完成用户，剩余 {len(user_ids_to_process)} 个待处理")

    analyzer = MergedErrorAnalyzer(
        analysis_dir=Path(OUTPUT_DIR),
        reviews_dir=Path(os.path.dirname(INPUT_FILE))
    )
    analyzer._load_merged_file()

    results = []

    log_with_timestamp("="*80)
    log_with_timestamp(f"Processing {len(user_ids_to_process)} users with ACL+CCOMP error extraction (concurrent={MAX_WORKERS})...")
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
                        f"ACL={result['acl_error_count']}, CCOMP={result['ccomp_error_count']} errors"
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
        total_acl_errors = sum(r["acl_error_count"] for r in successful)
        total_ccomp_errors = sum(r["ccomp_error_count"] for r in successful)
        total_errors = sum(r["total_errors"] for r in successful)

        log_with_timestamp(f"Total: {total_reviews} reviews, ACL={total_acl_errors}, CCOMP={total_ccomp_errors} errors")

        all_acl_error_types = defaultdict(int)
        all_ccomp_error_types = defaultdict(int)
        all_acl_region_types = defaultdict(int)
        all_ccomp_region_types = defaultdict(int)

        for r in successful:
            for etype, count in r["acl_error_types"].items():
                all_acl_error_types[etype] += count
            for etype, count in r["ccomp_error_types"].items():
                all_ccomp_error_types[etype] += count
            for rtype, count in r["acl_region_types"].items():
                all_acl_region_types[rtype] += count
            for rtype, count in r["ccomp_region_types"].items():
                all_ccomp_region_types[rtype] += count

        log_with_timestamp("ACL Error type distribution:")
        for etype, count in sorted(all_acl_error_types.items(), key=lambda x: x[1], reverse=True):
            log_with_timestamp(f"  {etype}: {count}")

        log_with_timestamp("CCOMP Error type distribution:")
        for etype, count in sorted(all_ccomp_error_types.items(), key=lambda x: x[1], reverse=True):
            log_with_timestamp(f"  {etype}: {count}")

        # 只保存有错误的用户
        users_with_errors = [r for r in successful if r["total_errors"] > 0]

        # 保存详情结果（JSON Lines 格式，已在处理时流水写入）

    if len(successful) == 0:
        log_with_timestamp("ERROR: No users were successfully processed!")
        sys.exit(1)

    log_with_timestamp("="*80)
    log_with_timestamp("ALL PROCESSING COMPLETE!")
    log_with_timestamp("="*80)


if __name__ == "__main__":
    main()
