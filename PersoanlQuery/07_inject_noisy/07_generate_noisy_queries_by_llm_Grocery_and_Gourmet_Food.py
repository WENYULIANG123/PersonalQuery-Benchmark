#!/usr/bin/env python3
"""
基于目标查询深度保持的查询语句噪声注入 - Grocery_and_Gourmet_Food
"""

import sys
import json
import time
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from datetime import datetime
from functools import lru_cache

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery')

# ========================================
# 配置加载
# ========================================
CATEGORY = "Grocery_and_Gourmet_Food"

# 全局配置
NOISY_CONFIG_FILE = '/home/wlia0047/ar57/wenyu/PersoanlQuery/07_inject_noisy/noisy_query_config.json'


# ========================================
# 加载配置和 prompt 模板
# ========================================
def get_required_config_value(config: dict, *keys):
    current = config
    current_path = []
    for key in keys:
        current_path.append(str(key))
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"配置缺少字段: {'.'.join(current_path)}")
        current = current[key]
    return current


with open(NOISY_CONFIG_FILE, 'r', encoding='utf-8') as f:
    _NOISY_CONFIG = json.load(f)

_CATEGORY_CONFIG = get_required_config_value(_NOISY_CONFIG, 'categories', CATEGORY)

NUM_USERS_TO_TEST = get_required_config_value(_NOISY_CONFIG, 'num_users_to_test')
MAX_WORKERS = get_required_config_value(_NOISY_CONFIG, 'max_workers')
USE_MINIMAXIO = get_required_config_value(_NOISY_CONFIG, 'use_minimaxio')
INJECT_ERROR_COUNT = get_required_config_value(_NOISY_CONFIG, 'inject_error_count')
if not isinstance(MAX_WORKERS, int) or MAX_WORKERS <= 0:
    raise ValueError(f"配置 max_workers 必须是正整数: {MAX_WORKERS!r}")
EFFECTIVE_MAX_WORKERS = MAX_WORKERS
LLM_MAX_RETRIES = 6
LLM_MAX_OUTPUT_TOKENS = 32768

QUERY_FILE = get_required_config_value(_CATEGORY_CONFIG, 'query_file')
USER_ERROR_FILE = get_required_config_value(_CATEGORY_CONFIG, 'user_error_file')
NOISY_OUTPUT_FILE = get_required_config_value(_CATEGORY_CONFIG, 'noisy_output_file')
NOISY_PROMPT_FILE = get_required_config_value(_NOISY_CONFIG, 'prompt_file')
SYNTAX_DEPTH_QUERY_FILE = (
    f'/home/wlia0047/ar57/wenyu/result/personal_query/06_query/{CATEGORY}/query_by_syntax_depth.json'
)
INJECTION_SOURCE = 'syntax_depth_preserve_depth'
BATCH_SIZE = 256

# 加载噪声 prompt 模板
with open(NOISY_PROMPT_FILE, 'r', encoding='utf-8') as f:
    _NOISY_PROMPTS = json.load(f)

NOISY_SYSTEM_BASE = get_required_config_value(_NOISY_PROMPTS, f"system_base_{CATEGORY}")
NOISY_USER_CONTENT_TEMPLATE = get_required_config_value(_NOISY_PROMPTS, "user_content_noisy")
LAST_TWO_LAYER_ANCHOR_INSERTION_SYSTEM_CONTENT = get_required_config_value(
    _NOISY_PROMPTS, "last_two_layer_anchor_insertion_system_content"
)
LAST_TWO_LAYER_ANCHOR_INSERTION_TEMPLATE = get_required_config_value(
    _NOISY_PROMPTS, "last_two_layer_anchor_insertion_user_content"
)
LAST_TWO_LAYER_ANCHOR_INSERTION_SYSTEM_BASE = (
    f"{NOISY_SYSTEM_BASE}\n\n{LAST_TWO_LAYER_ANCHOR_INSERTION_SYSTEM_CONTENT}"
)
CACHE_PREWARM_USER_CONTENT = get_required_config_value(_NOISY_PROMPTS, "cache_prewarm_user_content")
ATTRIBUTE_TYPE_LABELS = {
    'A1': 'product_type',
    'A2': 'brand',
    'A3': 'price',
    'A4': 'appearance',
    'A5': 'use_case',
    'A6': 'detailed',
    'A7': 'material',
    'A8': 'safety',
    'A9': 'durability',
    'A10': 'ease_of_use',
    'A11': 'temperature_resistance',
    'A12': 'surface',
    'A13': 'reusability',
    'A14': 'size',
    'A15': 'weight',
    'A16': 'compatibility',
    'A17': 'flavor',
    'A18': 'quality',
}


# ========================================
# 日志
# ========================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========================================
# 加载用户错误数据
# ========================================
def load_user_errors(error_file: str) -> dict:
    if not os.path.exists(error_file):
        raise FileNotFoundError(f"错误文件不存在: {error_file}")

    with open(error_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 支持数组格式（直接是用户列表）和字典格式（有 user_results 字段）
    if isinstance(data, list):
        users_list = data
    else:
        users_list = data.get('user_results', [])

    user_errors = {}
    for user in users_list:
        uid = user['user_id']
        if user['total_errors'] == 0 or not user.get('detailed_results'):
            continue

        acl_patterns = []
        ccomp_patterns = []

        for detail in user['detailed_results']:
            error_category = detail.get('error_category', '')
            if error_category not in ('acl', 'ccomp'):
                continue

            seen = set()
            patterns = []
            for err in detail.get('errors', []):
                orig = err.get('original', '')
                corr = err.get('corrected', '')
                key = (orig, corr)
                if key not in seen:
                    seen.add(key)
                    patterns.append({
                        'original': orig,
                        'corrected': corr,
                        'error_type': err.get('error_type', 'unknown'),
                    })

            if error_category == 'acl':
                acl_patterns.extend(patterns)
            elif error_category == 'ccomp':
                ccomp_patterns.extend(patterns)

        if acl_patterns or ccomp_patterns:
            user_errors[uid] = {'acl': acl_patterns, 'ccomp': ccomp_patterns}

    log(f"加载了 {len(user_errors)} 个有错误的用户")
    return user_errors


def filter_error_patterns(error_patterns: list) -> list:
    if not error_patterns:
        return []

    filtered = []
    for ep in error_patterns:
        orig = ep.get("original", "")
        corr = ep.get("corrected", "")

        if not orig or not corr:
            continue
        if '-' in orig or '-' in corr:
            continue
        if ' ' in orig or ' ' in corr:
            if orig.strip() != corr.strip():
                continue
        if is_pure_suffix_change(orig, corr):
            continue
        if is_pure_punctuation_change(orig, corr):
            continue
        if is_apostrophe_only_change(orig, corr):
            continue
        if is_case_only_change(orig, corr):
            continue

        filtered.append(ep)
    return filtered


def is_pure_suffix_change(orig: str, corr: str) -> bool:
    common_prefix_len = 0
    min_len = min(len(orig), len(corr))
    for i in range(min_len):
        if orig[i].lower() == corr[i].lower():
            common_prefix_len += 1
        else:
            break
    if common_prefix_len == 0:
        return False
    orig_suffix = orig[common_prefix_len:] if common_prefix_len < len(orig) else ''
    corr_suffix = corr[common_prefix_len:] if common_prefix_len < len(corr) else ''
    core_suffixes = {'ing', 'ed', 's', 'er', 'est', 'ly', 'd', 'en', 'ment', 'tion', 'ness'}
    if len(corr_suffix) == 0 and len(orig_suffix) > 0:
        return orig_suffix.lower() in core_suffixes
    if len(orig_suffix) == 0 and len(corr_suffix) > 0:
        return corr_suffix.lower() in core_suffixes
    if len(corr_suffix) > 0 and corr_suffix.lower() in core_suffixes:
        return True
    if orig_suffix.lower() in core_suffixes and corr_suffix.lower() in core_suffixes:
        return True
    return False


def is_pure_punctuation_change(orig: str, corr: str) -> bool:
    import string
    orig_only_punct = all(c not in string.ascii_letters + string.digits + string.whitespace for c in orig)
    corr_only_punct = all(c not in string.ascii_letters + string.digits + string.whitespace for c in corr)
    if orig_only_punct and corr_only_punct:
        return True
    return False


def is_apostrophe_only_change(orig: str, corr: str) -> bool:
    orig_clean = orig.replace("'", "").replace("'", "")
    corr_clean = corr.replace("'", "").replace("'", "")
    if orig_clean.lower() == corr_clean.lower():
        return True
    return False


def is_case_only_change(orig: str, corr: str) -> bool:
    if orig.lower() == corr.lower() and orig != corr:
        return True
    return False


# ========================================
# LLM 调用
# ========================================
_minimax_client = None
_cache_prewarmed = False


def load_minimax_client():
    global _minimax_client
    if _minimax_client is None:
        from llm_client import MiniMaxAnthropicClient, MiniMaxIOAnthropicClient
        if USE_MINIMAXIO:
            _minimax_client = MiniMaxIOAnthropicClient()
            log(f"MiniMaxIO API 客户端初始化完成: client={type(_minimax_client).__name__}, model={_minimax_client.model}")
        else:
            _minimax_client = MiniMaxAnthropicClient()
            log(f"MiniMax API 客户端初始化完成: client={type(_minimax_client).__name__}, model={_minimax_client.model}")


def prewarm_noisy_cache(system_base: str) -> None:
    global _cache_prewarmed
    if _cache_prewarmed:
        return
    if _minimax_client is None:
        load_minimax_client()
    if not isinstance(system_base, str) or not system_base.strip():
        raise ValueError("system_base 必须是非空字符串")

    log(f"[CachePrewarm] system_base:\n{system_base}")
    log(f"[CachePrewarm] user_content:\n{CACHE_PREWARM_USER_CONTENT}")
    response, cache_info = _minimax_client.call_with_cache(
        system_base=system_base,
        user_content=CACHE_PREWARM_USER_CONTENT,
        max_tokens=256,
        temperature=0.0,
        max_retries=LLM_MAX_RETRIES,
        retry_on_empty_response=False,
    )
    log(f"[CachePrewarm Cache] {cache_info}")
    log(f"[CachePrewarm Response] response:\n{response}")
    if not response:
        log("[ERROR] CachePrewarm empty response, marked failed without retry")
        return
    _cache_prewarmed = True


def call_llm(prompt: str, system_base: str = None) -> str:
    global _minimax_client
    if _minimax_client is None:
        load_minimax_client()

    cache_info = {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    log(f"[Request] user_content:\n{prompt}")

    log(f"[LLM] client={type(_minimax_client).__name__}, model={_minimax_client.model}, use_cache={bool(system_base)}")
    if system_base:
        response, cache_info = _minimax_client.call_with_cache(
            system_base=system_base,
            user_content=prompt,
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            temperature=1.0,
            max_retries=LLM_MAX_RETRIES,
            stream=True,
            retry_on_empty_response=False,
        )
    else:
        response = _minimax_client.call(prompt=prompt, max_tokens=LLM_MAX_OUTPUT_TOKENS, temperature=1.0, max_retries=LLM_MAX_RETRIES)

    log(f"[Cache] {cache_info}")
    log(f"[Response] response:\n{response}")
    if not response:
        log("[ERROR] LLM response empty, marked failed without retry")
    return response


def fix_incomplete_json(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    if not text.startswith('{') and not text.startswith('['):
        return text
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    if open_brackets > close_brackets:
        text = text + ']' * (open_brackets - close_brackets)
    if open_braces > close_braces:
        text = text + '}' * (open_braces - close_braces)
    return text


def count_keyword_in_query(query: str, keyword: str) -> int:
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")
    if not isinstance(keyword, str) or not keyword:
        raise ValueError("keyword 必须是非空字符串")
    token_strip_chars = " \t\n\r\f\v.,;:!?\"“”‘’()[]{}<>"
    keyword_lower = keyword.lower()
    count = 0
    for raw_token in query.split():
        token = raw_token.strip(token_strip_chars)
        if token.lower() == keyword_lower:
            count += 1
    return count


def build_complexity_constraint_text(query_category: str, target_level: int) -> str:
    if query_category == 'acl':
        return (
            f"This query is an ACL / wide query with complexity level {target_level}.\n"
            f"- The rewritten correct query must contain exactly {target_level} occurrence(s) of the word 'which'.\n"
            "- Count only standalone word tokens; contractions such as \"which's\" do not count as 'which'.\n"
            "- The rewritten correct query must contain zero occurrences of the word 'that'.\n"
            "- Do not increase or decrease the ACL complexity level during rewriting.\n"
        )
    if query_category == 'ccomp':
        return (
            f"This query is a CCOMP / deep query with complexity level {target_level}.\n"
            f"- The rewritten correct query must contain exactly {target_level} occurrence(s) of the word 'that'.\n"
            "- Count only standalone word tokens; contractions such as \"that's\" do not count as 'that'.\n"
            "- The rewritten correct query must contain zero occurrences of the word 'which'.\n"
            "- Do not increase or decrease the CCOMP complexity level during rewriting.\n"
        )
    raise ValueError(f"未知 query_category: {query_category}")


def revised_query_matches_expected_complexity(query: str, query_category: str, target_level: int) -> bool:
    which_count = count_keyword_in_query(query, 'which')
    that_count = count_keyword_in_query(query, 'that')
    if query_category == 'acl':
        return which_count == target_level and that_count == 0
    if query_category == 'ccomp':
        return that_count == target_level and which_count == 0
    raise ValueError(f"未知 query_category: {query_category}")


def build_noisy_prompt(query: str, error_patterns: list) -> tuple:
    system_base = NOISY_SYSTEM_BASE
    if error_patterns:
        error_lines = []
        for i, ep in enumerate(error_patterns[:10], 1):
            orig = ep.get('original', '')
            corr = ep.get('corrected', '')
            err_type = ep.get('error_type', 'unknown')
            error_lines.append(f"{i}. '{corr}' -> '{orig}' (error_type: {err_type})")
        errors_section = "User's typical spelling error patterns:\n" + "\n".join(error_lines) + "\n"
    else:
        errors_section = ""

    user_content = NOISY_USER_CONTENT_TEMPLATE.format(
        query=query, errors_section=errors_section, inject_count=INJECT_ERROR_COUNT
    )
    return system_base, user_content


def build_anchor_rewrite_prompt(query: str, error_patterns: list, query_category: str, target_level: int) -> tuple:
    system_base = NOISY_SYSTEM_BASE
    error_lines = []
    for i, ep in enumerate(error_patterns[:10], 1):
        orig = ep.get('original', '')
        corr = ep.get('corrected', '')
        err_type = ep.get('error_type', 'unknown')
        error_lines.append(f"{i}. '{corr}' -> '{orig}' (error_type: {err_type})")
    errors_section = "User's typical spelling error patterns:\n" + "\n".join(error_lines) + "\n"
    complexity_section = build_complexity_constraint_text(query_category, target_level)
    user_content = (
        f"Original correct query:\n{query}\n\n"
        f"{errors_section}"
        f"{complexity_section}"
        "Task:\n"
        "1. The original query currently does not contain a usable exact anchor for the user's real spelling patterns.\n"
        "2. First minimally rewrite the correct query so that at least one exact 'correct' text from the listed patterns appears naturally.\n"
        "3. The rewritten correct query must stay grammatical, natural, and preserve the original product, brand, price, attributes, search intent, and complexity style.\n"
        f"4. Then inject 1-{INJECT_ERROR_COUNT} user spelling errors into that rewritten correct query.\n"
        "5. You must use only exact listed pairs. Do not invent approximations.\n\n"
        "Output format (JSON):\n"
        "{\n"
        '  "revised_correct_query": "...",\n'
        '  "noisy_query": "...",\n'
        '  "injected_errors": [\n'
        '    {"correct": "...", "error": "...", "error_type": "..."}\n'
        "  ]\n"
        "}\n\n"
        "Important:\n"
        "- revised_correct_query must be a natural search query\n"
        "- noisy_query must be derived from revised_correct_query\n"
        "- injected_errors.correct must literally appear in revised_correct_query\n"
        "- Preserve product attributes, search intent, and required ACL/CCOMP complexity pattern exactly"
    )
    return system_base, user_content


def parse_noisy_response(text_content: str, original_query: str) -> dict:
    try:
        text_content = fix_incomplete_json(text_content)
        json_match = re.search(r'\{[\s\S]*\}', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)

        noisy_query = data.get('noisy_query', '').strip()
        if not noisy_query:
            return None
        injected_errors = data.get('injected_errors', [])
        return {'noisy_query': noisy_query, 'injected_errors': injected_errors if isinstance(injected_errors, list) else []}
    except Exception as e:
        log(f"    [DEBUG] JSON解析失败: {e}")
        if text_content.strip():
            return {'noisy_query': text_content.strip(), 'injected_errors': []}
    return None


def parse_anchor_rewrite_response(text_content: str) -> dict:
    try:
        text_content = fix_incomplete_json(text_content)
        json_match = re.search(r'\{[\s\S]*\}', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)
        revised_correct_query = data.get('revised_correct_query', '').strip()
        noisy_query = data.get('noisy_query', '').strip()
        injected_errors = data.get('injected_errors', [])
        if not revised_correct_query or not noisy_query:
            return None
        return {
            'revised_correct_query': revised_correct_query,
            'noisy_query': noisy_query,
            'injected_errors': injected_errors if isinstance(injected_errors, list) else [],
        }
    except Exception as e:
        log(f"    [DEBUG] 锚点改写JSON解析失败: {e}")
    return None


def build_real_error_pairs(error_patterns: list) -> set:
    real_error_pairs = set()
    for idx, pattern in enumerate(error_patterns):
        if not isinstance(pattern, dict):
            raise TypeError(f"error_patterns[{idx}] 必须是 dict")
        original = pattern.get('original')
        corrected = pattern.get('corrected')
        if not isinstance(original, str) or not original:
            raise ValueError(f"error_patterns[{idx}].original 必须是非空字符串")
        if not isinstance(corrected, str) or not corrected:
            raise ValueError(f"error_patterns[{idx}].corrected 必须是非空字符串")
        real_error_pairs.add((corrected, original))
        real_error_pairs.add((corrected.lower(), original.lower()))
    if not real_error_pairs:
        raise ValueError("真实错误模式不能为空")
    return real_error_pairs


def injected_errors_match_real_patterns(injected_errors: list, error_patterns: list) -> bool:
    if not isinstance(injected_errors, list) or not injected_errors:
        return False
    real_error_pairs = build_real_error_pairs(error_patterns)
    for idx, injected_error in enumerate(injected_errors):
        if not isinstance(injected_error, dict):
            raise TypeError(f"injected_errors[{idx}] 必须是 dict")
        correct = injected_error.get('correct')
        error = injected_error.get('error')
        if not isinstance(correct, str) or not isinstance(error, str):
            return False
        if (correct, error) not in real_error_pairs and (correct.lower(), error.lower()) not in real_error_pairs:
            return False
    return True


def query_contains_exact_anchor(query: str, correct_text: str) -> bool:
    if not isinstance(query, str) or not isinstance(correct_text, str) or not correct_text:
        return False
    escaped = re.escape(correct_text)
    if re.fullmatch(r"[A-Za-z0-9']+", correct_text):
        return re.search(rf"\b{escaped}\b", query, flags=re.IGNORECASE) is not None
    return re.search(escaped, query, flags=re.IGNORECASE) is not None


def query_count_exact_anchor(query: str, correct_text: str) -> int:
    if not isinstance(query, str) or not isinstance(correct_text, str) or not correct_text:
        return 0
    escaped = re.escape(correct_text)
    if re.fullmatch(r"[A-Za-z0-9']+", correct_text):
        pattern = re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)
    else:
        pattern = re.compile(escaped, flags=re.IGNORECASE)
    return sum(1 for _ in pattern.finditer(query))


def query_has_any_real_anchor(query: str, error_patterns: list) -> bool:
    for idx, pattern in enumerate(error_patterns):
        if not isinstance(pattern, dict):
            raise TypeError(f"error_patterns[{idx}] 必须是 dict")
        correct = pattern.get('corrected')
        if not isinstance(correct, str) or not correct:
            raise ValueError(f"error_patterns[{idx}].corrected 必须是非空字符串")
        if query_contains_exact_anchor(query, correct):
            return True
    return False


def injected_errors_have_query_anchor(query: str, injected_errors: list) -> bool:
    if not isinstance(injected_errors, list) or not injected_errors:
        return False
    for idx, injected_error in enumerate(injected_errors):
        if not isinstance(injected_error, dict):
            raise TypeError(f"injected_errors[{idx}] 必须是 dict")
        correct = injected_error.get('correct')
        if not isinstance(correct, str):
            return False
        if not query_contains_exact_anchor(query, correct):
            return False
    return True


def injected_errors_align_with_queries(ground_truth_query: str, noisy_query: str, injected_errors: list) -> bool:
    if not isinstance(ground_truth_query, str) or not isinstance(noisy_query, str):
        return False
    if not isinstance(injected_errors, list) or not injected_errors:
        return False
    for idx, injected_error in enumerate(injected_errors):
        if not isinstance(injected_error, dict):
            raise TypeError(f"injected_errors[{idx}] 必须是 dict")
        correct = injected_error.get('correct')
        error = injected_error.get('error')
        if not isinstance(correct, str) or not isinstance(error, str):
            return False
        if not query_contains_exact_anchor(ground_truth_query, correct):
            return False
        if not query_contains_exact_anchor(noisy_query, error):
            return False
    return True


@lru_cache(maxsize=1)
def _load_spacy_model():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    for pipe_name in ("ner", "lemmatizer", "textcat", "textcat_multilabel", "senter", "sentencizer"):
        if pipe_name in nlp.pipe_names:
            nlp.remove_pipe(pipe_name)
    return nlp


def compute_doc_token_depths(doc) -> tuple[dict[int, int], int]:
    depth_cache = {}
    token_depths = {}
    max_depth = 0

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        chain = []
        current = token
        while current.i not in depth_cache and current.head != current:
            chain.append(current)
            current = current.head

        if current.i in depth_cache:
            depth = depth_cache[current.i]
        else:
            depth = 1
            depth_cache[current.i] = depth

        for chain_token in reversed(chain):
            depth += 1
            depth_cache[chain_token.i] = depth

        token_depth = depth_cache[token.i]
        token_depths[token.i] = token_depth
        if token_depth > max_depth:
            max_depth = token_depth

    if max_depth == 0:
        raise ValueError("sentence contains no valid tokens for depth computation")
    return token_depths, max_depth


def extract_last_two_layer_tokens(doc) -> tuple[list[dict], int]:
    token_depths, max_depth = compute_doc_token_depths(doc)
    target_depths = {max_depth}
    if max_depth > 1:
        target_depths.add(max_depth - 1)

    last_two_tokens = []
    for token in doc:
        if token.i not in token_depths:
            continue
        depth = token_depths[token.i]
        if depth not in target_depths:
            continue
        last_two_tokens.append({
            'text': token.text,
            'index': token.i,
            'start': token.idx,
            'end': token.idx + len(token.text),
            'depth': depth,
            'dep': token.dep_,
            'pos': token.pos_,
        })

    if not last_two_tokens:
        raise ValueError("last two syntax layers contain no valid tokens")
    return last_two_tokens, max_depth


def extract_query_tokens(doc) -> tuple[list[dict], int]:
    token_depths, max_depth = compute_doc_token_depths(doc)
    query_tokens = []
    for token in doc:
        if token.i not in token_depths:
            continue
        query_tokens.append({
            'text': token.text,
            'index': token.i,
            'start': token.idx,
            'end': token.idx + len(token.text),
            'depth': token_depths[token.i],
            'dep': token.dep_,
            'pos': token.pos_,
        })
    if not query_tokens:
        raise ValueError("query tokens 不能为空")
    return query_tokens, max_depth


def replace_span(text: str, start: int, end: int, replacement: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    if not isinstance(replacement, str) or not replacement:
        raise ValueError("replacement 必须是非空字符串")
    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError("start/end 必须是整数")
    if start < 0 or end < start or end > len(text):
        raise ValueError(f"无效 span: start={start}, end={end}, text_len={len(text)}")
    return text[:start] + replacement + text[end:]


def is_single_token_injection_pattern(pattern: dict) -> bool:
    if not isinstance(pattern, dict):
        raise TypeError("pattern 必须是 dict")
    original = pattern.get('original')
    corrected = pattern.get('corrected')
    if not isinstance(original, str) or not original:
        raise ValueError("pattern.original 必须是非空字符串")
    if not isinstance(corrected, str) or not corrected:
        raise ValueError("pattern.corrected 必须是非空字符串")
    if original == corrected:
        return False
    if re.search(r"\s", original) or re.search(r"\s", corrected):
        return False
    return True


def build_injectable_patterns(error_patterns: list) -> list:
    injectable = []
    seen = set()
    seen_corrected = set()
    for idx, pattern in enumerate(error_patterns):
        if not isinstance(pattern, dict):
            raise TypeError(f"error_patterns[{idx}] 必须是 dict")
        if not is_single_token_injection_pattern(pattern):
            continue
        if 'error_type' not in pattern or not isinstance(pattern['error_type'], str) or not pattern['error_type']:
            raise ValueError(f"error_patterns[{idx}].error_type 必须是非空字符串")
        if pattern['corrected'] in seen_corrected:
            continue
        key = (pattern['original'], pattern['corrected'], pattern['error_type'])
        if key in seen:
            continue
        seen.add(key)
        seen_corrected.add(pattern['corrected'])
        injectable.append(pattern)
    return injectable


def select_direct_anchor(query_tokens: list, error_patterns: list) -> tuple[dict, dict] | None:
    ordered_tokens = sorted(query_tokens, key=lambda item: item['index'])
    for token_info in ordered_tokens:
        token_text = token_info['text']
        for pattern in error_patterns:
            corrected = pattern['corrected']
            if token_text.lower() == corrected.lower():
                return token_info, pattern
    return None


def build_public_query_tokens(query_tokens: list) -> list:
    public_tokens = []
    ordered_tokens = sorted(query_tokens, key=lambda item: item['index'])
    for idx, token_info in enumerate(ordered_tokens):
        if not isinstance(token_info, dict):
            raise TypeError(f"query_tokens[{idx}] 必须是 dict")
        if 'text' not in token_info or not isinstance(token_info['text'], str) or not token_info['text']:
            raise ValueError(f"query_tokens[{idx}].text 必须是非空字符串")
        public_tokens.append({
            'token_index': token_info['index'],
            'text': token_info['text'],
            'depth': token_info['depth'],
            'dependency': token_info['dep'],
            'pos': token_info['pos'],
        })
    return public_tokens


def build_public_correct_anchor_values(selected_patterns: list) -> list:
    selected_patterns = normalize_selected_anchor_patterns(selected_patterns)
    return [
        {
            'correct_text': pattern['corrected'],
        }
        for pattern in selected_patterns
    ]


def build_anchor_insertion_pattern_candidates(error_patterns: list) -> list:
    candidates = []
    for idx, pattern in enumerate(error_patterns):
        if not isinstance(pattern, dict):
            raise TypeError(f"error_patterns[{idx}] 必须是 dict")
        for required_key in ('corrected', 'original', 'error_type'):
            if required_key not in pattern:
                raise KeyError(f"error_patterns[{idx}] 缺少字段: {required_key}")
            if not isinstance(pattern[required_key], str) or not pattern[required_key]:
                raise ValueError(f"error_patterns[{idx}].{required_key} 必须是非空字符串")
        candidates.append({
            'correct_text': pattern['corrected'],
            'user_error_text': pattern['original'],
            'noise_type': pattern['error_type'],
        })
    return candidates


def normalize_selected_anchor_patterns(error_patterns: list) -> list:
    if not isinstance(error_patterns, list):
        raise TypeError("selected anchor patterns 必须是列表")
    normalized = []
    seen_corrected = set()
    for idx, pattern in enumerate(error_patterns):
        if not isinstance(pattern, dict):
            raise TypeError(f"selected anchor patterns[{idx}] 必须是 dict")
        for required_key in ('corrected', 'original', 'error_type'):
            if required_key not in pattern:
                raise KeyError(f"selected anchor patterns[{idx}] 缺少字段: {required_key}")
            if not isinstance(pattern[required_key], str) or not pattern[required_key].strip():
                raise ValueError(f"selected anchor patterns[{idx}].{required_key} 必须是非空字符串")
        corrected = pattern['corrected'].strip()
        if corrected in seen_corrected:
            raise ValueError(f"selected anchor patterns 中 correct_text 不能重复: {corrected}")
        seen_corrected.add(corrected)
        normalized.append({
            'corrected': corrected,
            'original': pattern['original'].strip(),
            'error_type': pattern['error_type'].strip(),
        })
    if not normalized:
        raise ValueError("selected anchor patterns 不能为空")
    return normalized


def build_llm_anchor_insertion_prompt(
    query: str,
    query_info: dict,
    query_tokens: list,
    selected_patterns: list,
) -> tuple:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    if not isinstance(query_info, dict):
        raise TypeError("query_info 必须是对象")
    if not isinstance(query_tokens, list) or not query_tokens:
        raise ValueError("query_tokens 必须是非空列表")
    selected_patterns = normalize_selected_anchor_patterns(selected_patterns)
    user_content = LAST_TWO_LAYER_ANCHOR_INSERTION_TEMPLATE.format(
        query=query.strip(),
        product_attributes_json=json.dumps(build_public_attribute_values(query_info), ensure_ascii=False, indent=2),
        query_tokens_json=json.dumps(build_public_query_tokens(query_tokens), ensure_ascii=False, indent=2),
        correct_texts_json=json.dumps(build_public_correct_anchor_values(selected_patterns), ensure_ascii=False, indent=2),
    )
    return LAST_TWO_LAYER_ANCHOR_INSERTION_SYSTEM_BASE, user_content


def parse_llm_anchor_insertion_response(text_content: str, selected_patterns: list) -> list[str] | None:
    if not isinstance(text_content, str) or not text_content.strip():
        raise ValueError("LLM anchor insertion response 不能为空")
    raw_text = text_content.strip()
    if raw_text.startswith("```"):
        fence_match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
        if not fence_match:
            raise ValueError("LLM anchor insertion response 的代码围栏格式不合法")
        raw_text = fence_match.group(1).strip()
    data = json.loads(raw_text)
    if not isinstance(data, dict):
        raise TypeError("LLM anchor insertion response 必须是 JSON object")
    if data.get('status') == 'IMPOSSIBLE':
        reason = data.get('reason', '')
        if not isinstance(reason, str):
            raise TypeError("IMPOSSIBLE.reason 必须是字符串")
        return None

    candidates = data.get('candidates')
    if not isinstance(candidates, list):
        raise TypeError("candidates 必须是列表")
    if len(candidates) != 10:
        raise ValueError(f"候选句子数量必须正好为 10 个，当前为 {len(candidates)}")

    valid_correct_texts = {pattern['corrected'] for pattern in normalize_selected_anchor_patterns(selected_patterns)}
    candidate_queries = []
    seen = set()
    for candidate_idx, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            log(f"[CandidateSkip] candidates[{candidate_idx}] 不是 dict")
            continue
        query_text = candidate.get('query')
        if not isinstance(query_text, str) or not query_text.strip():
            log(f"[CandidateSkip] candidates[{candidate_idx}].query 非法")
            continue
        normalized_query = query_text.strip()
        if normalized_query in seen:
            continue
        if not any(query_count_exact_anchor(normalized_query, correct_text) > 0 for correct_text in valid_correct_texts):
            log(f"[CandidateSkip] candidates[{candidate_idx}] 未包含任何 correct_text")
            continue
        seen.add(normalized_query)
        candidate_queries.append(normalized_query)
    return candidate_queries


def select_anchor_insertion_pattern(decision: dict, error_patterns: list) -> dict:
    matches = [
        pattern for pattern in error_patterns
        if (
            pattern['corrected'] == decision['correct_text']
            and pattern['original'] == decision['user_error_text']
            and pattern['error_type'] == decision['noise_type']
        )
    ]
    if not matches:
        raise ValueError(
            "LLM 选择了不存在的错误模式: "
            f"correct_text={decision['correct_text']}, "
            f"user_error_text={decision['user_error_text']}, "
            f"noise_type={decision['noise_type']}"
        )
    if len(matches) > 1:
        raise ValueError(
            "LLM 选择的错误模式无法唯一定位: "
            f"correct_text={decision['correct_text']}, "
            f"user_error_text={decision['user_error_text']}, "
            f"noise_type={decision['noise_type']}"
        )
    return matches[0]


def _attribute_sort_key(key: str) -> tuple[int, str]:
    if not isinstance(key, str):
        raise TypeError("attrs_used 的键必须是字符串")
    suffix_match = re.search(r"(\d+)$", key)
    if suffix_match:
        return int(suffix_match.group(1)), key
    return 10**9, key


def extract_ordered_query_attributes(query_info: dict) -> list[dict]:
    if not isinstance(query_info, dict):
        raise TypeError("query_info 必须是对象")
    attrs_used = query_info.get('attrs_used')
    if not isinstance(attrs_used, dict) or not attrs_used:
        raise ValueError("query_info.attrs_used 必须是非空对象")

    ordered_items = sorted(attrs_used.items(), key=lambda item: _attribute_sort_key(item[0]))
    if len(ordered_items) != 5:
        raise ValueError(f"query_info.attrs_used 必须正好包含 5 个属性值，当前为 {len(ordered_items)}")

    ordered_attrs = []
    for idx, (attr_key, attr_value) in enumerate(ordered_items):
        if not isinstance(attr_value, str) or not attr_value.strip():
            raise ValueError(f"query_info.attrs_used[{attr_key}] 必须是非空字符串")
        ordered_attrs.append({
            'attr_key': attr_key,
            'attr_type': ATTRIBUTE_TYPE_LABELS.get(attr_key, 'unknown'),
            'attr_value': attr_value.strip(),
        })
    return ordered_attrs


def build_public_attribute_values(query_info: dict) -> list[dict]:
    return [
        {
            'attr_type': item['attr_type'],
            'attr_value': item['attr_value'],
        }
        for item in extract_ordered_query_attributes(query_info)
    ]


def find_exact_anchor_span(query: str, anchor_text: str):
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")
    if not isinstance(anchor_text, str) or not anchor_text:
        raise ValueError("anchor_text 必须是非空字符串")

    escaped = re.escape(anchor_text)
    if re.fullmatch(r"[A-Za-z0-9']+", anchor_text):
        pattern = re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)
    else:
        pattern = re.compile(escaped, flags=re.IGNORECASE)

    matches = list(pattern.finditer(query))
    if not matches:
        return None
    if len(matches) > 1:
        return 'multiple'
    match = matches[0]
    return match.start(), match.end()


def find_token_info_at_span(doc, token_depths: dict[int, int], start: int, end: int) -> dict | None:
    for token in doc:
        if token.idx == start and token.idx + len(token.text) == end:
            depth = token_depths.get(token.i)
            if depth is None:
                raise ValueError(f"token depth 缺失: token_index={token.i}")
            return {
                'text': token.text,
                'index': token.i,
                'start': token.idx,
                'end': token.idx + len(token.text),
                'depth': depth,
                'dep': token.dep_,
                'pos': token.pos_,
            }
    return None


def validate_depth_only_syntax(query: str, expected_depth: int) -> str | None:
    if not isinstance(query, str) or not query:
        raise ValueError("query 必须是非空字符串")
    if not isinstance(expected_depth, int) or expected_depth <= 0:
        raise ValueError("expected_depth 必须是正整数")
    doc = _load_spacy_model()(query)
    _, actual_depth = compute_doc_token_depths(doc)
    if actual_depth != expected_depth:
        return 'syntax_depth_changed_after_injection'
    return None


def build_anchor_rewrite_result(
    original_query: str,
    revised_correct_query: str,
    patterns: list,
    expected_depth: int,
    query_info: dict,
) -> tuple[dict | None, str | None]:
    if not isinstance(original_query, str) or not original_query.strip():
        raise ValueError("original_query 必须是非空字符串")
    if not isinstance(revised_correct_query, str) or not revised_correct_query.strip():
        raise ValueError("revised_correct_query 必须是非空字符串")
    if revised_correct_query.strip() == original_query.strip():
        return None, 'llm_rewrite_not_new'

    patterns = normalize_selected_anchor_patterns(patterns)

    ordered_attrs = extract_ordered_query_attributes(query_info)
    for attr_item in ordered_attrs:
        if query_count_exact_anchor(revised_correct_query, attr_item['attr_value']) != 1:
            return None, 'llm_rewrite_attribute_count_invalid'

    revised_doc = _load_spacy_model()(revised_correct_query)
    revised_token_depths, revised_max_depth = compute_doc_token_depths(revised_doc)
    if revised_max_depth != expected_depth:
        return None, 'syntax_depth_changed_after_rewrite'

    selected_anchor_spec = None
    for pattern in patterns:
        anchor_text = pattern['corrected']
        escaped = re.escape(anchor_text)
        if re.fullmatch(r"[A-Za-z0-9']+", anchor_text):
            anchor_pattern = re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)
        else:
            anchor_pattern = re.compile(escaped, flags=re.IGNORECASE)
        anchor_match = next(anchor_pattern.finditer(revised_correct_query), None)
        if anchor_match is None:
            continue
        anchor_start, anchor_end = anchor_match.start(), anchor_match.end()
        token_info = find_token_info_at_span(revised_doc, revised_token_depths, anchor_start, anchor_end)
        if token_info is None:
            return None, 'llm_correct_anchor_token_not_found'
        selected_anchor_spec = {
            'pattern': pattern,
            'source_start': anchor_start,
            'source_end': anchor_end,
            'token_info': token_info,
        }
        break

    if selected_anchor_spec is None:
        return None, 'llm_correct_anchor_not_inserted'

    return build_multi_anchor_result_from_specs(
        revised_correct_query,
        [selected_anchor_spec],
        expected_depth,
    )


def iter_nonspace_token_spans(text: str) -> list:
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    return [
        {'text': match.group(), 'start': match.start(), 'end': match.end()}
        for match in re.finditer(r"\S+", text)
    ]


def extract_inserted_token_spans(original_query: str, revised_query: str) -> list | None:
    original_spans = iter_nonspace_token_spans(original_query)
    revised_spans = iter_nonspace_token_spans(revised_query)
    original_tokens = [item['text'] for item in original_spans]
    revised_tokens = [item['text'] for item in revised_spans]
    if not original_tokens or not revised_tokens:
        raise ValueError("original_query 和 revised_query 都必须包含至少一个 token")

    matcher = SequenceMatcher(a=original_tokens, b=revised_tokens, autojunk=False)
    inserted_indices = []
    preserved_count = 0
    for tag, original_start, original_end, revised_start, revised_end in matcher.get_opcodes():
        if tag == 'equal':
            preserved_count += original_end - original_start
            continue
        if tag == 'insert':
            inserted_indices.extend(range(revised_start, revised_end))
            continue
        return None
    if preserved_count != len(original_tokens):
        return None
    return [revised_spans[index] for index in inserted_indices]


def core_token_span(token_text: str, token_start: int) -> tuple[str, int, int] | None:
    if not isinstance(token_text, str):
        raise TypeError("token_text 必须是字符串")
    if not isinstance(token_start, int) or token_start < 0:
        raise ValueError("token_start 必须是非负整数")
    token_strip_chars = " \t\n\r\f\v.,;:!?\"“”‘’()[]{}<>"
    left = 0
    right = len(token_text)
    while left < right and token_text[left] in token_strip_chars:
        left += 1
    while right > left and token_text[right - 1] in token_strip_chars:
        right -= 1
    if left == right:
        return None
    return token_text[left:right], token_start + left, token_start + right


def build_anchor_insertion_result(
    query: str,
    revised_correct_query: str,
    pattern: dict,
    expected_max_depth: int,
) -> tuple[dict | None, str | None]:
    inserted_spans = extract_inserted_token_spans(query, revised_correct_query)
    if inserted_spans is None:
        return None, 'llm_original_tokens_changed'
    if not inserted_spans:
        return None, 'llm_no_inserted_tokens'

    anchor_spans = []
    for inserted_span in inserted_spans:
        core_span = core_token_span(inserted_span['text'], inserted_span['start'])
        if core_span is None:
            continue
        core_text, core_start, core_end = core_span
        if core_text == pattern['corrected']:
            anchor_spans.append((core_start, core_end))

    if not anchor_spans:
        return None, 'llm_correct_anchor_not_inserted'
    if len(anchor_spans) > 1:
        return None, 'llm_correct_anchor_insert_count_invalid'

    anchor_start, anchor_end = anchor_spans[0]
    noisy_query = replace_span(revised_correct_query, anchor_start, anchor_end, pattern['original'])
    post_injection_status = validate_post_injection_syntax(
        noisy_query,
        pattern['original'],
        anchor_start,
        expected_max_depth,
    )
    if post_injection_status is not None:
        return None, post_injection_status

    return {
        'pattern': pattern,
        'revised_correct_query': revised_correct_query,
        'noisy_query': noisy_query,
        'token_info': {
            'text': pattern['corrected'],
            'index': -1,
            'start': anchor_start,
            'end': anchor_end,
            'depth': expected_max_depth,
            'dep': '',
            'pos': '',
        },
    }, None


def select_anchor_insertion_with_llm(
    query: str,
    query_tokens: list,
    query_info: dict,
    error_patterns: list,
    expected_depth: int,
) -> tuple[dict | None, str | None]:
    selected_patterns = normalize_selected_anchor_patterns(error_patterns)
    if not selected_patterns:
        return None, 'no_token_error_pattern'
    system_base, user_content = build_llm_anchor_insertion_prompt(query, query_info, query_tokens, selected_patterns)
    try:
        response = call_llm(user_content, system_base=system_base)
        if not response or not response.strip():
            return None, 'empty_llm_response'
        candidate_queries = parse_llm_anchor_insertion_response(response, selected_patterns)
    except Exception as exc:
        log(f"LLM anchor 响应解析失败，记任务失败: {type(exc).__name__}: {exc}")
        return None, 'llm_anchor_response_invalid'
    if candidate_queries is None:
        return None, 'no_suitable_rewrite'
    last_failure_status = None
    for candidate_query in candidate_queries:
        candidate_result, failure_status = build_anchor_rewrite_result(
            query,
            candidate_query,
            selected_patterns,
            expected_depth,
            query_info,
        )
        last_failure_status = failure_status
        if candidate_result is not None:
            log("LLM 候选句子通过校验并被选中")
            return candidate_result, None
    return None, last_failure_status or 'no_candidate_passed_validation'


def build_injected_error(pattern: dict, token_info: dict, mode: str) -> dict:
    return {
        'correct': pattern['corrected'],
        'error': pattern['original'],
        'error_type': pattern['error_type'],
        'target_token': token_info['text'],
        'target_token_index': token_info['index'],
        'target_token_depth': token_info['depth'],
        'injection_mode': mode,
    }


def build_injected_errors(anchor_entries: list, mode: str) -> list:
    if not isinstance(anchor_entries, list) or not anchor_entries:
        raise ValueError("anchor_entries 必须是非空列表")
    return [
        build_injected_error(anchor_entry['pattern'], anchor_entry['token_info'], mode)
        for anchor_entry in anchor_entries
    ]


def normalize_result_field_values(value):
    if value == '':
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        normalized = []
        for idx, item in enumerate(value):
            if not isinstance(item, str):
                raise TypeError(f"result field list item[{idx}] 必须是字符串")
            normalized.append(item)
        return normalized
    raise TypeError("result field 必须是字符串或字符串列表")


def select_direct_anchor_matches(last_two_tokens: list, error_patterns: list) -> list[dict] | None:
    if not isinstance(last_two_tokens, list) or not last_two_tokens:
        raise ValueError("last_two_tokens 必须是非空列表")
    ordered_tokens = sorted(last_two_tokens, key=lambda item: (-item['depth'], item['index']))
    selected = []
    used_token_indices = set()
    seen_corrected = set()

    for idx, pattern in enumerate(error_patterns):
        if not isinstance(pattern, dict):
            raise TypeError(f"error_patterns[{idx}] 必须是 dict")
        corrected = pattern.get('corrected')
        if not isinstance(corrected, str) or not corrected.strip():
            raise ValueError(f"error_patterns[{idx}].corrected 必须是非空字符串")
        corrected = corrected.strip()
        if corrected in seen_corrected:
            raise ValueError(f"selected_patterns 中 correct_text 不能重复: {corrected}")
        seen_corrected.add(corrected)

        matched_token = None
        for token_info in ordered_tokens:
            if token_info['index'] in used_token_indices:
                continue
            if token_info['text'].lower() == corrected.lower():
                matched_token = token_info
                break
        if matched_token is None:
            return None
        used_token_indices.add(matched_token['index'])
        selected.append({
            'pattern': pattern,
            'source_start': matched_token['start'],
            'source_end': matched_token['end'],
            'token_info': matched_token,
        })

    return selected


def apply_span_replacements(text: str, anchor_specs: list) -> tuple[str, list[dict]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text 必须是非空字符串")
    if not isinstance(anchor_specs, list) or not anchor_specs:
        raise ValueError("anchor_specs 必须是非空列表")

    ordered_specs = sorted(anchor_specs, key=lambda item: item['source_start'])
    last_end = -1
    for idx, spec in enumerate(ordered_specs):
        if not isinstance(spec, dict):
            raise TypeError(f"anchor_specs[{idx}] 必须是 dict")
        for required_key in ('pattern', 'source_start', 'source_end', 'token_info'):
            if required_key not in spec:
                raise KeyError(f"anchor_specs[{idx}] 缺少字段: {required_key}")
        if not isinstance(spec['source_start'], int) or not isinstance(spec['source_end'], int):
            raise TypeError(f"anchor_specs[{idx}].source_start/source_end 必须是整数")
        if spec['source_start'] < 0 or spec['source_end'] < spec['source_start'] or spec['source_end'] > len(text):
            raise ValueError(
                f"anchor_specs[{idx}] 的 source span 无效: start={spec['source_start']}, end={spec['source_end']}, text_len={len(text)}"
            )
        if spec['source_start'] < last_end:
            raise ValueError("anchor_specs 中的替换 span 不能重叠")
        if not isinstance(spec['pattern'], dict):
            raise TypeError(f"anchor_specs[{idx}].pattern 必须是 dict")
        if 'original' not in spec['pattern'] or 'corrected' not in spec['pattern']:
            raise KeyError(f"anchor_specs[{idx}].pattern 缺少 original/corrected")
        if not isinstance(spec['pattern']['original'], str) or not spec['pattern']['original'].strip():
            raise ValueError(f"anchor_specs[{idx}].pattern.original 必须是非空字符串")
        if not isinstance(spec['pattern']['corrected'], str) or not spec['pattern']['corrected'].strip():
            raise ValueError(f"anchor_specs[{idx}].pattern.corrected 必须是非空字符串")
        last_end = spec['source_end']

    current_text = text
    offset = 0
    applied_specs = []
    for spec in ordered_specs:
        start = spec['source_start'] + offset
        end = spec['source_end'] + offset
        replacement = spec['pattern']['original']
        current_text = current_text[:start] + replacement + current_text[end:]
        applied_spec = dict(spec)
        applied_spec['replacement'] = replacement
        applied_spec['final_start'] = start
        applied_spec['final_end'] = start + len(replacement)
        applied_specs.append(applied_spec)
        offset += len(replacement) - (spec['source_end'] - spec['source_start'])

    return current_text, applied_specs


def validate_post_injection_syntax_multi(noisy_query: str, anchor_entries: list, expected_max_depth: int) -> str | None:
    if not isinstance(noisy_query, str) or not noisy_query:
        raise ValueError("noisy_query 必须是非空字符串")
    if not isinstance(anchor_entries, list) or not anchor_entries:
        raise ValueError("anchor_entries 必须是非空列表")
    if not isinstance(expected_max_depth, int) or expected_max_depth <= 0:
        raise ValueError("expected_max_depth 必须是正整数")

    noisy_doc = _load_spacy_model()(noisy_query)
    noisy_token_depths, noisy_max_depth = compute_doc_token_depths(noisy_doc)
    if noisy_max_depth != expected_max_depth:
        if noisy_max_depth < expected_max_depth:
            return 'syntax_depth_too_shallow_after_injection'
        return 'syntax_depth_too_deep_after_injection'

    for idx, entry in enumerate(anchor_entries):
        if not isinstance(entry, dict):
            raise TypeError(f"anchor_entries[{idx}] 必须是 dict")
        for required_key in ('replacement', 'final_start', 'final_end'):
            if required_key not in entry:
                raise KeyError(f"anchor_entries[{idx}] 缺少字段: {required_key}")
        if not isinstance(entry['replacement'], str) or not entry['replacement']:
            raise ValueError(f"anchor_entries[{idx}].replacement 必须是非空字符串")
        if not isinstance(entry['final_start'], int) or not isinstance(entry['final_end'], int):
            raise TypeError(f"anchor_entries[{idx}].final_start/final_end 必须是整数")
        if 'token_info' not in entry or not isinstance(entry['token_info'], dict):
            raise KeyError(f"anchor_entries[{idx}] 缺少 token_info")
        found = False
        for token_info in noisy_doc:
            if (
                token_info.text == entry['replacement']
                and token_info.idx == entry['final_start']
                and token_info.idx + len(token_info.text) == entry['final_end']
            ):
                if noisy_token_depths.get(token_info.i) is None:
                    raise ValueError(f"anchor_entries[{idx}] 的 token depth 缺失")
                found = True
                break
        if not found:
            return 'injected_error_token_not_found_after_injection'
    return None


def build_multi_anchor_result_from_specs(base_query: str, anchor_specs: list, expected_max_depth: int) -> tuple[dict | None, str | None]:
    if not isinstance(base_query, str) or not base_query.strip():
        raise ValueError("base_query 必须是非空字符串")
    if not isinstance(anchor_specs, list) or not anchor_specs:
        raise ValueError("anchor_specs 必须是非空列表")
    if not isinstance(expected_max_depth, int) or expected_max_depth <= 0:
        raise ValueError("expected_max_depth 必须是正整数")

    noisy_query, applied_specs = apply_span_replacements(base_query, anchor_specs)
    depth_status = validate_post_injection_syntax_multi(noisy_query, applied_specs, expected_max_depth)
    if depth_status is not None:
        return None, depth_status

    return {
        'revised_correct_query': base_query,
        'noisy_query': noisy_query,
        'anchor_entries': applied_specs,
    }, None


def validate_post_injection_syntax(
    noisy_query: str,
    error_text: str,
    expected_start: int,
    expected_max_depth: int,
) -> str | None:
    if not isinstance(noisy_query, str) or not noisy_query:
        raise ValueError("noisy_query 必须是非空字符串")
    if not isinstance(error_text, str) or not error_text:
        raise ValueError("error_text 必须是非空字符串")
    if not isinstance(expected_start, int) or expected_start < 0:
        raise ValueError("expected_start 必须是非负整数")
    if not isinstance(expected_max_depth, int) or expected_max_depth <= 0:
        raise ValueError("expected_max_depth 必须是正整数")

    expected_end = expected_start + len(error_text)
    noisy_doc = _load_spacy_model()(noisy_query)
    noisy_token_depths, noisy_max_depth = compute_doc_token_depths(noisy_doc)
    if noisy_max_depth != expected_max_depth:
        if noisy_max_depth < expected_max_depth:
            return 'syntax_depth_too_shallow_after_injection'
        return 'syntax_depth_too_deep_after_injection'
    for token_info in noisy_doc:
        if (
            token_info.text == error_text
            and token_info.idx == expected_start
            and token_info.idx + len(token_info.text) == expected_end
        ):
            if noisy_token_depths.get(token_info.i) is None:
                raise ValueError("injected token depth 缺失")
            return None
    return 'injected_error_token_not_found_after_injection'


def inject_by_last_two_layers(
    query: str,
    query_tokens: list,
    error_patterns: list,
    query_info: dict,
    expected_depth: int,
) -> dict:
    injectable_patterns = build_injectable_patterns(error_patterns)
    if not injectable_patterns:
        return {'status': 'no_token_error_pattern'}

    direct_anchor = select_direct_anchor(query_tokens, injectable_patterns)
    if direct_anchor:
        token_info, pattern = direct_anchor
        revised_correct_query = query
        noisy_query = replace_span(query, token_info['start'], token_info['end'], pattern['original'])
        anchor_entries = [{
            'pattern': pattern,
            'source_start': token_info['start'],
            'source_end': token_info['end'],
            'token_info': token_info,
            'replacement': pattern['original'],
            'final_start': token_info['start'],
            'final_end': token_info['start'] + len(pattern['original']),
        }]
        mode = 'direct_anchor'
    else:
        insertion_result, failure_status = select_anchor_insertion_with_llm(
            query,
            query_tokens,
            query_info,
            injectable_patterns,
            expected_depth,
        )
        if failure_status is not None:
            return {'status': failure_status}
        revised_correct_query = insertion_result['revised_correct_query']
        noisy_query = insertion_result['noisy_query']
        anchor_entries = insertion_result['anchor_entries']
        mode = 'rewrite_then_inject'

    injected_errors = build_injected_errors(anchor_entries, mode)
    if noisy_query == revised_correct_query:
        return {'status': 'no_injection'}
    if not injected_errors_match_real_patterns(injected_errors, injectable_patterns):
        return {'status': 'pattern_mismatch'}
    if not injected_errors_align_with_queries(revised_correct_query, noisy_query, injected_errors):
        return {'status': 'no_anchor'}
    post_injection_status = validate_post_injection_syntax_multi(
        noisy_query,
        anchor_entries,
        expected_depth,
    )
    if post_injection_status is not None:
        return {'status': post_injection_status}

    return {
        'status': 'success',
        'source_query': query,
        'ground_truth_query': revised_correct_query,
        'noisy_query': noisy_query,
        'injected_errors': injected_errors,
        'query_rewritten': revised_correct_query != query,
        'injection_mode': mode,
        'anchor_entries': anchor_entries,
        'selected_tokens': [anchor_entry['token_info'] for anchor_entry in anchor_entries],
    }


# ========================================
# 增量写入辅助函数
# ========================================
def write_noisy_result_incremental(result_item: dict, output_file: str):
    """将单个句法树最后两层注入结果增量写入简化文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    for required_key in ('uid', 'asin', 'source_query', 'status'):
        if required_key not in result_item:
            raise KeyError(f"result_item 缺少字段: {required_key}")
    if not isinstance(result_item['uid'], str) or not result_item['uid'].strip():
        raise ValueError("result_item.uid 必须是非空字符串")
    if not isinstance(result_item['asin'], str):
        raise TypeError("result_item.asin 必须是字符串")
    if not isinstance(result_item['source_query'], str) or not result_item['source_query'].strip():
        raise ValueError("result_item.source_query 必须是非空字符串")
    if not isinstance(result_item['status'], str) or not result_item['status']:
        raise ValueError("result_item.status 必须是非空字符串")
    if result_item['status'] != 'success':
        return

    noisy_query = ''
    noise_type = ''
    correct_text = ''
    noisy_text = ''
    anchor_replaced_text = ''
    injection_target_depth = ''
    original_query = result_item['source_query']
    if result_item['status'] == 'success':
        if 'ground_truth_query' not in result_item:
            raise KeyError("成功结果缺少 ground_truth_query")
        if not isinstance(result_item['ground_truth_query'], str) or not result_item['ground_truth_query'].strip():
            raise ValueError("成功结果 ground_truth_query 必须是非空字符串")
        if 'noisy_query' not in result_item:
            raise KeyError("成功结果缺少 noisy_query")
        if not isinstance(result_item['noisy_query'], str) or not result_item['noisy_query'].strip():
            raise ValueError("成功结果 noisy_query 必须是非空字符串")
        if 'injected_errors' not in result_item:
            raise KeyError("成功结果缺少 injected_errors")
        injected_errors = result_item['injected_errors']
        if not isinstance(injected_errors, list) or not injected_errors:
            raise ValueError("成功结果 injected_errors 必须是非空列表")
        if 'anchor_entries' not in result_item:
            raise KeyError("成功结果缺少 anchor_entries")
        anchor_entries = result_item['anchor_entries']
        if not isinstance(anchor_entries, list) or not anchor_entries:
            raise ValueError("成功结果 anchor_entries 必须是非空列表")
        if len(anchor_entries) != len(injected_errors):
            raise ValueError("成功结果 anchor_entries 与 injected_errors 数量必须一致")
        for idx, injected_error in enumerate(injected_errors):
            if not isinstance(injected_error, dict):
                raise TypeError(f"成功结果 injected_errors[{idx}] 必须是 dict")
            if 'error_type' not in injected_error:
                raise KeyError(f"成功结果 injected_errors[{idx}] 缺少 error_type")
            if not isinstance(injected_error['error_type'], str) or not injected_error['error_type'].strip():
                raise ValueError(f"成功结果 injected_errors[{idx}].error_type 必须是非空字符串")
            for required_key in ('correct', 'error', 'target_token', 'injection_mode'):
                if required_key not in injected_error:
                    raise KeyError(f"成功结果 injected_errors[{idx}] 缺少 {required_key}")
                if not isinstance(injected_error[required_key], str) or not injected_error[required_key].strip():
                    raise ValueError(f"成功结果 injected_errors[{idx}].{required_key} 必须是非空字符串")
            if injected_error['injection_mode'] not in ('direct_anchor', 'rewrite_then_inject'):
                raise ValueError(f"未知 injection_mode: {injected_error['injection_mode']}")
        for idx, anchor_entry in enumerate(anchor_entries):
            if not isinstance(anchor_entry, dict):
                raise TypeError(f"成功结果 anchor_entries[{idx}] 必须是 dict")
            for required_key in ('pattern', 'token_info', 'source_start', 'source_end', 'final_start', 'final_end', 'replacement'):
                if required_key not in anchor_entry:
                    raise KeyError(f"成功结果 anchor_entries[{idx}] 缺少 {required_key}")
            if not isinstance(anchor_entry['pattern'], dict):
                raise TypeError(f"成功结果 anchor_entries[{idx}].pattern 必须是 dict")
            if not isinstance(anchor_entry['token_info'], dict):
                raise TypeError(f"成功结果 anchor_entries[{idx}].token_info 必须是 dict")
            if not isinstance(anchor_entry['replacement'], str) or not anchor_entry['replacement'].strip():
                raise ValueError(f"成功结果 anchor_entries[{idx}].replacement 必须是非空字符串")
        noisy_query = result_item['noisy_query']
        noisy_doc = _load_spacy_model()(noisy_query)
        noisy_token_depths, _ = compute_doc_token_depths(noisy_doc)
        noise_type = [anchor_entry['pattern']['error_type'] for anchor_entry in anchor_entries]
        correct_text = [anchor_entry['pattern']['corrected'] for anchor_entry in anchor_entries]
        noisy_text = [anchor_entry['pattern']['original'] for anchor_entry in anchor_entries]
        anchor_replaced_text = [anchor_entry['token_info']['text'] for anchor_entry in anchor_entries]
        original_query = result_item['ground_truth_query']
        injection_target_depth = []
        for idx, anchor_entry in enumerate(anchor_entries):
            matched_token = None
            for token in noisy_doc:
                if (
                    token.text == anchor_entry['replacement']
                    and token.idx == anchor_entry['final_start']
                    and token.idx + len(token.text) == anchor_entry['final_end']
                ):
                    matched_token = token
                    break
            if matched_token is None:
                raise ValueError(f"成功结果 anchor_entries[{idx}] 在 noisy_query 中未找到对应 token")
            actual_depth = noisy_token_depths.get(matched_token.i)
            if actual_depth is None:
                raise ValueError(f"成功结果 anchor_entries[{idx}] 的 token depth 缺失")
            injection_target_depth.append(actual_depth)

        def pack_output_field(values: list[str]):
            if not values:
                return ''
            if len(values) == 1:
                return values[0]
            return values

    output_data = {
        'user_id': result_item['uid'],
        'asin': result_item['asin'],
        'original_query': original_query,
        'noisy_query': noisy_query,
        'noise_type': pack_output_field(noise_type) if result_item['status'] == 'success' else '',
        'correct_text': pack_output_field(correct_text) if result_item['status'] == 'success' else '',
        'noisy_text': pack_output_field(noisy_text) if result_item['status'] == 'success' else '',
        'anchor_replaced_text': pack_output_field(anchor_replaced_text) if result_item['status'] == 'success' else '',
        'injection_target_depth': pack_output_field(injection_target_depth) if result_item['status'] == 'success' else '',
        'injection_source': INJECTION_SOURCE,
    }
    # 追加到文件
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        f.write('\n')


# ========================================
# 主函数
# ========================================
def load_appended_json_objects(file_path: str) -> list:
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return []

    decoder = json.JSONDecoder()
    objects = []
    idx = 0
    while idx < len(content):
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx >= len(content):
            break
        decoded, end = decoder.raw_decode(content, idx)
        if isinstance(decoded, list):
            if objects:
                raise ValueError(f"{file_path} 中 JSON array 前存在其他 JSON 对象")
            tail = content[end:].strip()
            if tail:
                raise ValueError(f"{file_path} 中 JSON array 后存在额外内容")
            return decoded
        if not isinstance(decoded, dict):
            raise TypeError(f"{file_path} 中追加 JSON 记录必须是对象")
        objects.append(decoded)
        idx = end
    return objects


def completed_key_from_fields(user_id: str, asin: str) -> tuple:
    return user_id, asin


def completed_key_from_record(record: dict) -> tuple:
    query_info = record['syntax_depth_query']
    return completed_key_from_fields(
        record['user_id'],
        record['asin'],
    )


def load_completed_query_keys(output_file: str) -> set:
    completed = set()
    for idx, item in enumerate(load_appended_json_objects(output_file)):
        if not isinstance(item, dict):
            raise TypeError(f"已有结果第 {idx} 条必须是对象")
        for required_key in ('user_id', 'asin', 'original_query', 'noisy_query', 'noise_type', 'correct_text', 'noisy_text', 'anchor_replaced_text'):
            if required_key not in item:
                raise KeyError(f"已有结果第 {idx} 条缺少字段: {required_key}")
        if not isinstance(item['user_id'], str) or not item['user_id'].strip():
            raise ValueError(f"已有结果第 {idx} 条 user_id 必须是非空字符串")
        if not isinstance(item['asin'], str):
            raise TypeError(f"已有结果第 {idx} 条 asin 必须是字符串")
        if not isinstance(item['original_query'], str) or not item['original_query'].strip():
            raise ValueError(f"已有结果第 {idx} 条 original_query 必须是非空字符串")
        if not isinstance(item['noisy_query'], str):
            raise TypeError(f"已有结果第 {idx} 条 noisy_query 必须是字符串")
        if not isinstance(item['noise_type'], (str, list)):
            raise TypeError(f"已有结果第 {idx} 条 noise_type 必须是字符串或字符串列表")
        if not isinstance(item['correct_text'], (str, list)):
            raise TypeError(f"已有结果第 {idx} 条 correct_text 必须是字符串或字符串列表")
        if not isinstance(item['noisy_text'], (str, list)):
            raise TypeError(f"已有结果第 {idx} 条 noisy_text 必须是字符串或字符串列表")
        if not isinstance(item['anchor_replaced_text'], (str, list)):
            raise TypeError(f"已有结果第 {idx} 条 anchor_replaced_text 必须是字符串或字符串列表")
        if 'injection_source' not in item:
            continue
        if not isinstance(item['injection_source'], str) or not item['injection_source'].strip():
            raise ValueError(f"已有结果第 {idx} 条 injection_source 必须是非空字符串")
        if item['injection_source'] != INJECTION_SOURCE:
            continue
        noise_type_values = normalize_result_field_values(item['noise_type'])
        correct_text_values = normalize_result_field_values(item['correct_text'])
        noisy_text_values = normalize_result_field_values(item['noisy_text'])
        anchor_replaced_text_values = normalize_result_field_values(item['anchor_replaced_text'])
        if item['noisy_query'] == '':
            if any(
                len(values) != 0
                for values in (noise_type_values, correct_text_values, noisy_text_values, anchor_replaced_text_values)
            ):
                raise ValueError(f"已有结果第 {idx} 条为空 noisy_query 时，其它结果字段也必须为空")
            continue
        else:
            lengths = {
                len(noise_type_values),
                len(correct_text_values),
                len(noisy_text_values),
                len(anchor_replaced_text_values),
            }
            if len(lengths) != 1 or 0 in lengths:
                raise ValueError(f"已有结果第 {idx} 条多值结果字段必须保持一致的单值/多值结构")
        completed.add(completed_key_from_fields(
            item['user_id'],
            item['asin'],
        ))
    return completed


def merge_filtered_user_errors(raw_user_errors: dict) -> dict:
    merged_by_user = {}
    for uid, err_data in raw_user_errors.items():
        if not isinstance(err_data, dict):
            raise TypeError(f"user_errors[{uid}] 必须是 dict")
        merged = []
        seen = set()
        for category in ('acl', 'ccomp'):
            for pattern in filter_error_patterns(err_data.get(category, [])):
                if 'error_type' not in pattern or not isinstance(pattern['error_type'], str) or not pattern['error_type']:
                    raise ValueError(f"user_errors[{uid}] 缺少有效 error_type")
                key = (pattern['original'], pattern['corrected'], pattern['error_type'])
                if key in seen:
                    continue
                seen.add(key)
                merged.append(pattern)
        if merged:
            merged_by_user[uid] = merged
    return merged_by_user


def load_syntax_depth_query_records(file_path: str) -> list:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"syntax-depth query file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise TypeError("query_by_syntax_depth.json 顶层必须是列表")
    return payload


def validate_syntax_depth_query_record(record: dict, idx: int) -> None:
    if not isinstance(record, dict):
        raise TypeError(f"records[{idx}] 必须是对象")
    for required_key in ('user_id', 'asin', 'syntax_depth_query'):
        if required_key not in record:
            raise KeyError(f"records[{idx}] 缺少字段: {required_key}")
    if not isinstance(record['user_id'], str) or not record['user_id'].strip():
        raise ValueError(f"records[{idx}].user_id 必须是非空字符串")
    if not isinstance(record['asin'], str):
        raise TypeError(f"records[{idx}].asin 必须是字符串")
    query_info = record['syntax_depth_query']
    if not isinstance(query_info, dict):
        raise TypeError(f"records[{idx}].syntax_depth_query 必须是对象")
    for required_key in ('query', 'word_count', 'actual_depth', 'attrs_used'):
        if required_key not in query_info:
            raise KeyError(f"records[{idx}].syntax_depth_query 缺少字段: {required_key}")
    if not isinstance(query_info['query'], str) or not query_info['query'].strip():
        raise ValueError(f"records[{idx}].syntax_depth_query.query 必须是非空字符串")
    if not isinstance(query_info['word_count'], int):
        raise TypeError(f"records[{idx}].syntax_depth_query.word_count 必须是整数")
    if not isinstance(query_info['actual_depth'], int):
        raise TypeError(f"records[{idx}].syntax_depth_query.actual_depth 必须是整数")
    attrs_used = query_info['attrs_used']
    if not isinstance(attrs_used, dict) or len(attrs_used) != 5:
        raise ValueError(f"records[{idx}].syntax_depth_query.attrs_used 必须是包含 5 个属性值的对象")
    for attr_key, attr_value in attrs_used.items():
        if not isinstance(attr_key, str) or not attr_key:
            raise TypeError(f"records[{idx}].syntax_depth_query.attrs_used 的键必须是非空字符串")
        if not isinstance(attr_value, str) or not attr_value.strip():
            raise ValueError(f"records[{idx}].syntax_depth_query.attrs_used[{attr_key}] 必须是非空字符串")


def build_syntax_depth_query_tasks(records: list, user_errors: dict, completed_keys: set) -> tuple[list, dict]:
    tasks = []
    stats = {
        'no_user_errors': 0,
        'already_completed': 0,
        'eligible_before_limit': 0,
    }

    for idx, record in enumerate(records):
        validate_syntax_depth_query_record(record, idx)
        uid = record['user_id']
        errors = user_errors.get(uid)
        if not errors:
            stats['no_user_errors'] += 1
            continue

        query_info = record['syntax_depth_query']
        completed_key = completed_key_from_record(record)
        if completed_key in completed_keys:
            stats['already_completed'] += 1
            continue

        stats['eligible_before_limit'] += 1
        tasks.append({
            'uid': uid,
            'asin': record['asin'],
            'source_query_type': 'syntax_depth_query',
            'source_query': query_info['query'].strip(),
            'source_word_count': query_info['word_count'],
            'source_syntax_tree_depth': query_info['actual_depth'],
            'target_syntax_tree_depth': query_info.get('target_depth'),
            'level': query_info.get('target_depth'),
            'errors': errors,
            'query_info': dict(query_info),
        })
        if len(tasks) >= NUM_USERS_TO_TEST:
            break
    return tasks, stats


def process_one_query_task(task: dict, doc) -> dict:
    uid = task['uid']
    query = task['source_query']
    try:
        query_tokens, computed_depth = extract_query_tokens(doc)
    except Exception as exc:
        return {
            'uid': uid,
            'asin': task['asin'],
            'source_query': query,
            'status': 'syntax_parse_error',
            'error': str(exc),
        }

    try:
        injected = inject_by_last_two_layers(
            query,
            query_tokens,
            task['errors'],
            task['query_info'],
            computed_depth,
        )
    except Exception as exc:
        return {
            'uid': uid,
            'asin': task['asin'],
            'source_query': query,
            'status': 'task_unexpected_exception',
            'source_query_type': task['source_query_type'],
            'error': f'{type(exc).__name__}: {exc}',
        }
    if injected['status'] != 'success':
        return {
            'uid': uid,
            'asin': task['asin'],
            'source_query': query,
            'status': injected['status'],
            'source_query_type': task['source_query_type'],
        }

    return {
        'uid': uid,
        'asin': task['asin'],
        'source_query_type': task['source_query_type'],
        'source_query': injected['source_query'],
        'ground_truth_query': injected['ground_truth_query'],
        'ground_truth_word_count': len(injected['ground_truth_query'].split()),
        'noisy_query': injected['noisy_query'],
        'injected_errors': injected['injected_errors'],
        'user_errors': task['errors'],
        'query_info': task['query_info'],
        'query_rewritten': injected['query_rewritten'],
        'source_word_count': task['source_word_count'],
        'source_syntax_tree_depth': task['source_syntax_tree_depth'],
        'target_syntax_tree_depth': task['target_syntax_tree_depth'],
        'computed_syntax_tree_depth': computed_depth,
        'query_tokens': query_tokens,
        'selected_tokens': injected['selected_tokens'],
        'anchor_entries': injected['anchor_entries'],
        'injection_mode': injected['injection_mode'],
        'status': 'success',
    }


def main():
    log(f"=== 基于目标查询深度保持的噪声注入开始 (Category: {CATEGORY}) ===")
    log(f"加载 syntax-depth query from {SYNTAX_DEPTH_QUERY_FILE}...")
    syntax_depth_records = load_syntax_depth_query_records(SYNTAX_DEPTH_QUERY_FILE)
    log(f"加载了 {len(syntax_depth_records)} 个用户级 syntax-depth query 记录")

    log(f"加载用户错误 from {USER_ERROR_FILE}...")
    raw_user_errors = load_user_errors(USER_ERROR_FILE)
    user_errors = merge_filtered_user_errors(raw_user_errors)
    log(f"过滤并合并 ACL/CCOMP 后，有错误模式的用户数: {len(user_errors)}")

    completed_keys = load_completed_query_keys(NOISY_OUTPUT_FILE)
    log(f"已有 {INJECTION_SOURCE} 完成记录数: {len(completed_keys)}")

    tasks, build_stats = build_syntax_depth_query_tasks(syntax_depth_records, user_errors, completed_keys)
    log(f"无可用用户错误的用户级 query 数: {build_stats['no_user_errors']}")
    log(f"已完成跳过用户级 query 数: {build_stats['already_completed']}")
    log(f"达到数量限制前的候选用户级 query 数: {build_stats['eligible_before_limit']}")
    log(f"本次任务数: {len(tasks)}")
    if not tasks:
        raise ValueError("没有可处理的 query_by_syntax_depth 任务")

    log("LLM token 候选已启用：LLM 基于原句 token 生成候选 noisy query，本地校验注入前后整体深度保持不变")
    prewarm_noisy_cache(LAST_TWO_LAYER_ANCHOR_INSERTION_SYSTEM_BASE)

    nlp = _load_spacy_model()
    log(f"并发 worker 数: {EFFECTIVE_MAX_WORKERS}")
    status_counts = {}
    success_count = 0
    rewritten_success_count = 0
    direct_success_count = 0
    rewrite_success_count = 0
    total_start = time.time()

    task_docs = list(zip(tasks, nlp.pipe((task['source_query'] for task in tasks), batch_size=BATCH_SIZE)))
    with ThreadPoolExecutor(max_workers=EFFECTIVE_MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(process_one_query_task, task, doc): task
            for task, doc in task_docs
        }
        for processed_count, future in enumerate(as_completed(future_to_task), start=1):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                log(
                    f"[ERROR] 任务异常退出，已降级为失败: "
                    f"user={task['uid'][:20]} err={type(exc).__name__}: {exc}"
                )
                result = {
                    'uid': task['uid'],
                    'asin': task['asin'],
                    'source_query': task['source_query'],
                    'source_query_type': task.get('source_query_type'),
                    'status': 'task_unexpected_exception',
                    'error': f'{type(exc).__name__}: {exc}',
                }
            try:
                write_noisy_result_incremental(result, NOISY_OUTPUT_FILE)
            except Exception as exc:
                log(
                    f"[ERROR] 结果写盘失败，已降级为失败: "
                    f"user={task['uid'][:20]} err={type(exc).__name__}: {exc}"
                )
                result = {
                    'uid': task['uid'],
                    'asin': task['asin'],
                    'source_query': task['source_query'],
                    'source_query_type': task.get('source_query_type'),
                    'status': 'task_unexpected_exception',
                    'error': f'{type(exc).__name__}: {exc}',
                }

            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1

            if status == 'success':
                success_count += 1
                if result['query_rewritten']:
                    rewritten_success_count += 1
                if result['injection_mode'] == 'direct_anchor':
                    direct_success_count += 1
                elif result['injection_mode'] == 'rewrite_then_inject':
                    rewrite_success_count += 1

            log(
                f"  [进度 {processed_count}/{len(tasks)}] "
                f"成功:{success_count} 直接注入:{direct_success_count} "
                f"LLM重写后注入:{rewrite_success_count} "
                f"改写成功:{rewritten_success_count} "
                f"当前状态:{status} user={task['uid'][:20]}"
            )

    elapsed = time.time() - total_start
    log(f"\n{'='*60}")
    log("==================== 统计结果 ====================")
    log(f"总任务数: {len(tasks)}")
    log(f"成功注入: {success_count} ({success_count/len(tasks)*100:.1f}%)")
    log(f"直接命中并注入: {direct_success_count}")
    log(f"LLM 重写后注入: {rewrite_success_count}")
    log(f"其中 query_rewritten=True: {rewritten_success_count}")
    for status in sorted(status_counts):
        log(f"  - {status}: {status_counts[status]}")
    log(f"耗时: {elapsed:.1f}s")
    log(f"结果保存到: {NOISY_OUTPUT_FILE}")


if __name__ == '__main__':
    main()
