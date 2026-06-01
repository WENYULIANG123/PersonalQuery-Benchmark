"""噪声注入脚本通用函数模块"""
import sys
import json
import os
import re
from datetime import datetime
from functools import lru_cache


# ========================================
# 编辑距离工具
# ========================================
def edit_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离"""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
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


def find_edit_distance_one_matches(query: str, error_patterns: list) -> tuple:
    """查找 query 中与错误模式的 corrected（正确词）编辑距离为1的独立单词

    Returns:
        tuple: (matched_pattern, query_word_in_query, noisy_word_to_replace_with) 或 (None, None, None)
    """
    for pattern in error_patterns:
        original = pattern.get('original', '').lower()
        corrected = pattern.get('corrected', '').lower()

        if not original or not corrected:
            continue

        # 查找 query 中与 corrected（正确词）编辑距离为1的独立单词
        # 注意：我们只检查 corrected，因为用户是把 corrected 拼成了 original
        # 所以 query 中应该能找到 corrected（正确词），然后替换成 original（错误词）
        for match in re.finditer(r'(?<![a-zA-Z])[a-zA-Z]+(?![a-zA-Z])', query):
            word = match.group().lower()
            start, end = match.start(), match.end()

            # 只有当 query 中的词与 corrected（正确词）编辑距离为1时才匹配
            if edit_distance(word, corrected) == 1:
                # 找到了匹配：query 中的词是正确词的拼写变体，替换成错误形式 original
                return (pattern, query[start:end], original)

    return (None, None, None)


def inject_error_by_edit_distance(clean_query: str, matched_pattern: dict, query_word: str, noisy_word: str) -> str:
    """通过编辑距离匹配注入错误，只替换独立单词"""
    # 只替换独立的单词（前后有空格或标点的）
    pattern = re.compile(r'(?<![a-zA-Z])' + re.escape(query_word) + r'(?![a-zA-Z])', re.IGNORECASE)
    noisy_query = pattern.sub(noisy_word, clean_query, count=1)
    return noisy_query


# ========================================
# 配置
# ========================================
ATTRIBUTE_TYPE_LABELS = {
    'A1': 'product_type', 'A2': 'brand', 'A3': 'price', 'A4': 'appearance',
    'A5': 'use_case', 'A6': 'detailed', 'A7': 'material', 'A8': 'safety',
    'A9': 'durability', 'A10': 'ease_of_use', 'A11': 'temperature_resistance',
    'A12': 'surface', 'A13': 'reusability', 'A14': 'size', 'A15': 'weight',
    'A16': 'compatibility', 'A17': 'flavor', 'A18': 'quality',
}


def get_required_config_value(config: dict, *keys):
    """获取配置值"""
    current = config
    current_path = []
    for key in keys:
        current_path.append(str(key))
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"配置缺少字段: {'.'.join(current_path)}")
        current = current[key]
    return current


def load_noisy_config():
    """加载全局配置"""
    config_path = os.path.join(os.path.dirname(__file__), 'noisy_query_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_noisy_prompts(CATEGORY: str):
    """加载 prompt 模板"""
    config = load_noisy_config()
    prompt_file = config.get('prompt_file', os.path.join(os.path.dirname(__file__), 'noisy_query_prompts.json'))
    with open(prompt_file, 'r', encoding='utf-8') as f:
        _NOISY_PROMPTS = json.load(f)
    NOISY_SYSTEM_BASE = get_required_config_value(_NOISY_PROMPTS, f"system_base_{CATEGORY}")
    NOISY_USER_CONTENT_TEMPLATE = get_required_config_value(_NOISY_PROMPTS, "user_content_noisy")
    return {
        'NOISY_SYSTEM_BASE': NOISY_SYSTEM_BASE,
        'NOISY_USER_CONTENT_TEMPLATE': NOISY_USER_CONTENT_TEMPLATE,
    }


# ========================================
# 日志
# ========================================
def log(msg):
    """统一日志函数"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========================================
# LLM 调用
# ========================================
_NOISY_CONFIG = None  # 延迟加载配置

def _get_noisy_config():
    """获取配置（延迟加载）"""
    global _NOISY_CONFIG
    if _NOISY_CONFIG is None:
        _NOISY_CONFIG = load_noisy_config()
    return _NOISY_CONFIG

def _use_minimax_io() -> bool:
    """判断是否使用 MiniMax IO"""
    config = _get_noisy_config()
    return config.get('use_minimaxio', True)

@lru_cache(maxsize=1)
def load_llm_client():
    """加载 LLM 客户端（根据 use_minimaxio 选择）"""
    sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery')
    from llm_client import MiniMaxAnthropicClient, MiniMaxIOAnthropicClient
    if _use_minimax_io():
        return MiniMaxIOAnthropicClient()
    else:
        return MiniMaxAnthropicClient()

def prewarm_noisy_cache(system_base: str, user_content_template: str) -> None:
    """预热缓存 - 发送真实prompt内容来创建缓存"""
    # 构建示例prompt用于缓存创建
    sample_errors = [
        {'original': 'recieved', 'corrected': 'received', 'error_type': 'writing_error'},
        {'original': 'then', 'corrected': 'than', 'error_type': 'writing_error'},
    ]
    sample_attrs = {
        'A1': 'Baby Stroller',
        'A2': 'safe',
        'A3': 'affordable',
    }
    sample_user_content = build_noisy_prompt(sample_errors, user_content_template, inject_count=2, attrs_used=sample_attrs)
    log(f"[CACHE_PREWARM] 发送请求: system_base={system_base[:80]}...")
    log(f"[CACHE_PREWARM] user_content: {sample_user_content[:300]}...")
    response, token_usage = call_llm(sample_user_content, system_base)
    log(f"[CACHE_PREWARM] 响应: {response[:200] if response else 'None'}...")
    log(f"[CACHE_PREWARM] token_usage: {token_usage}")

def call_llm(prompt: str, system_base: str = None) -> tuple:
    """调用 LLM，根据 use_minimaxio 选择客户端

    Returns:
        tuple: (response_text, token_usage_dict)
    """
    client = load_llm_client()
    response, token_usage = client.call_with_cache(system_base=system_base, user_content=prompt, max_tokens=32768, stream=True)
    return response, token_usage


def fix_incomplete_json(text: str) -> str:
    """修复不完整的 JSON"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith('"') and not text.endswith('"'):
        text = text + '"'
    if text.startswith('{') and not text.endswith('}'):
        bracket_count = 0
        for i, c in enumerate(text):
            if c == '{':
                bracket_count += 1
            elif c == '}':
                bracket_count -= 1
        if bracket_count > 0:
            text = text + '}' * bracket_count
    return text


def parse_noisy_response(text_content: str) -> dict:
    """解析噪声响应"""
    try:
        return json.loads(text_content)
    except json.JSONDecodeError:
        return {}


# ========================================
# 用户错误数据处理
# ========================================
def load_user_errors(error_file: str) -> dict:
    """加载用户错误数据 - writing_error.json 格式（支持读取不完整的 JSON）"""
    if not os.path.exists(error_file):
        raise FileNotFoundError(f"错误文件不存在: {error_file}")

    try:
        with open(error_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # 如果 JSON 不完整，尝试逐行解析
        data = []
        with open(error_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 尝试找到完整的 JSON 对象
            depth = 0
            start = 0
            for i, c in enumerate(content):
                if c == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(content[start:i+1])
                            data.append(obj)
                        except json.JSONDecodeError:
                            pass
            if isinstance(data, list) and len(data) > 0 and 'user_results' in data[0]:
                data = data[0]['user_results']
            elif not isinstance(data, list):
                data = []

    if isinstance(data, list):
        users_list = data
    else:
        users_list = data.get('user_results', [])
    user_errors = {}
    for user in users_list:
        uid = user['user_id']
        if user['total_errors'] == 0 or not user.get('error_details'):
            continue
        seen = set()
        all_patterns = []
        for detail in user['error_details']:
            orig = detail.get('original', '')
            corr = detail.get('corrected', '')
            if not orig or not corr:
                continue
            key = (orig, corr)
            if key not in seen:
                seen.add(key)
                all_patterns.append({
                    'original': orig,
                    'corrected': corr,
                    'error_type': 'writing_error',
                })
        if all_patterns:
            user_errors[uid] = {'writing': all_patterns}
    log(f"加载了 {len(user_errors)} 个有错误的用户")
    return user_errors


def merge_filtered_user_errors(raw_user_errors: dict) -> dict:
    """过滤并合并用户错误"""
    merged_by_user = {}
    for uid, err_data in raw_user_errors.items():
        if not isinstance(err_data, dict):
            raise TypeError(f"user_errors[{uid}] 必须是 dict")
        merged = []
        seen = set()
        for pattern in filter_error_patterns(err_data.get('writing', [])):
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


def filter_error_patterns(error_patterns: list) -> list:
    """返回所有错误模式，不过滤"""
    return error_patterns if error_patterns else []


def is_pure_suffix_change(orig: str, corr: str) -> bool:
    if not orig or not corr:
        return False
    if orig.endswith(corr) and len(orig) > len(corr):
        return True
    if corr.endswith(orig) and len(corr) > len(orig):
        return True
    return False


def is_pure_punctuation_change(orig: str, corr: str) -> bool:
    orig_clean = ''.join(c for c in orig if c.isalnum())
    corr_clean = ''.join(c for c in corr if c.isalnum())
    return orig_clean == corr_clean


def is_apostrophe_only_change(orig: str, corr: str) -> bool:
    orig_clean = orig.replace("'", "").replace("'", "").replace("`", "")
    corr_clean = corr.replace("'", "").replace("'", "").replace("`", "")
    if orig_clean == corr_clean and orig != corr:
        return True
    return False


def is_case_only_change(orig: str, corr: str) -> bool:
    if orig.lower() == corr.lower() and orig != corr:
        return True
    return False


# ========================================
# 锚点工具
# ========================================
def is_single_token_injection_pattern(pattern: dict) -> bool:
    """检查是否单 token 注入模式"""
    orig = pattern.get('original', '')
    corr = pattern.get('corrected', '')
    return bool(orig.strip() and corr.strip() and ' ' not in orig.strip() and ' ' not in corr.strip())


def build_injectable_patterns(error_patterns: list) -> list:
    """构建可注入的模式"""
    return [p for p in error_patterns if is_single_token_injection_pattern(p)]


# ========================================
# 结果构建
# ========================================
def build_anchor_insertion_result(base_query: str, modified_query: str, anchor_specs: list, injection_mode: str) -> dict:
    """构建锚点插入结果"""
    return {
        'query': modified_query,
        'base_query': base_query,
        'anchor_specs': anchor_specs,
        'injection_mode': injection_mode,
        'query_rewritten': modified_query != base_query,
    }


def build_injected_error(pattern: dict, token_info: dict, mode: str) -> dict:
    """构建注入的错误"""
    return {
        'error': pattern.get('original', ''),
        'corrected': pattern.get('corrected', ''),
        'error_type': pattern.get('error_type', 'unknown'),
        'mode': mode,
    }


def validate_injected_errors(injected_errors: list, injectable_patterns: list, noisy_query: str, clean_query: str = "") -> tuple:
    """验证注入的错误是否有效

    验证策略：
    - 每个错误词(error)必须是给定的错误模式之一（original 字段）
    - 错误词(error)必须存在于 noisy_query 中（大小写不敏感）
    - 至少注入1个错误

    Returns:
        tuple: (is_valid: bool, valid_errors: list, reason: str)
    """
    if not injectable_patterns:
        return True, injected_errors, "no patterns to validate"

    if not injected_errors:
        return False, [], "no errors returned from LLM"

    # 构建错误词集合（大小写不敏感）
    error_words = {p['original'].lower() for p in injectable_patterns}
    noisy_query_lower = noisy_query.lower()

    valid_errors = []
    for err in injected_errors:
        err_text = err.get('error', '')     # 错误词
        err_text_lower = err_text.lower()

        # 检查是否使用了给定的错误词（大小写不敏感）
        if err_text_lower not in error_words:
            continue

        # 检查 noisy_query 中是否包含这个错误（大小写不敏感）
        if err_text_lower not in noisy_query_lower:
            continue

        valid_errors.append(err)

    # 只要有至少1个有效错误就返回成功
    if valid_errors:
        return True, valid_errors, f"{len(valid_errors)} valid error(s)"

    return False, [], "no valid errors found"



# ========================================
# IO 工具
# ========================================

def write_noisy_result_incremental(result_item: dict, output_file: str):
    """每完成一个就立即写入文件（追加模式）"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result_item, ensure_ascii=False) + '\n')





def completed_key_from_record(record: dict) -> tuple:
    """从记录构建完成 key"""
    return (record.get('user_id', ''), record.get('asin', ''))


def load_appended_json_objects(file_path: str) -> list:
    """加载 JSON 对象（支持 JSON Lines 和标准 JSON 数组格式）"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if not content:
        return []
    
    # 如果是 JSON 数组格式（以 '[' 开头）
    if content.startswith('['):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return []
    
    # 否则按 JSON Lines 格式解析（每行一个 JSON）
    objects = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return objects



def load_completed_query_keys(output_file: str) -> set:
    """加载已完成的查询 key"""
    records = load_appended_json_objects(output_file)
    return {completed_key_from_record(r) for r in records if r.get('status') == 'success'}


# ========================================
# 查询记录处理
# ========================================
def load_query_records(file_path: str) -> list:
    """加载查询记录"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"查询文件不存在: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return payload.get('records', [])


def build_query_tasks(records: list, user_errors: dict, completed_keys: set) -> list:
    """构建查询任务"""
    tasks = []
    for record in records:
        uid = record.get('user_id')
        asin = record.get('asin')
        
        # 提取 query 字符串（支持多种格式）
        syntax_depth_query = record.get('syntax_depth_query')
        acl_query = record.get('acl_query')
        ccomp_query = record.get('ccomp_query')
        
        clean_query = ''
        query_info = None
        
        if isinstance(syntax_depth_query, dict):
            clean_query = syntax_depth_query.get('query', '')
            query_info = syntax_depth_query
        elif isinstance(acl_query, dict):
            clean_query = acl_query.get('query', '')
            query_info = acl_query
        elif isinstance(ccomp_query, dict):
            clean_query = ccomp_query.get('query', '')
            query_info = ccomp_query
        elif isinstance(acl_query, str):
            clean_query = acl_query
        elif isinstance(ccomp_query, str):
            clean_query = ccomp_query
        else:
            clean_query = record.get('query', '')
        
        if not uid or not asin or not clean_query:
            continue
        
        record_key = (uid, asin)
        if record_key in completed_keys:
            continue
        
        errors_data = user_errors.get(uid, {})
        # load_user_errors 返回格式是 {'writing': [...]}
        if isinstance(errors_data, dict):
            errors = errors_data.get('writing', [])
        else:
            errors = errors_data if isinstance(errors_data, list) else []

        # 提取 attrs_used（如果有）
        attrs_used = None
        if isinstance(syntax_depth_query, dict):
            attrs_used = syntax_depth_query.get('attrs_used')
        elif query_info and isinstance(query_info, dict):
            attrs_used = query_info.get('attrs_used')

        tasks.append({
            'uid': uid,
            'asin': asin,
            'clean_query': clean_query,
            'query_info': query_info,
            'errors': errors,
            'attrs_used': attrs_used,
        })
    return tasks


def validate_attrs_preserved(attrs_used: dict, noisy_query: str) -> tuple:
    """验证属性值是否在 noisy_query 中出现且仅出现一次

    参考06_query的validate_query_uses_exactly_five_attrs：
    - 每个属性key对应的value必须在noisy_query中作为独立单词出现且仅出现1次
    - 使用单词边界匹配，避免子串误匹配（如"Storage"不会被"Food Storage"匹配）

    Returns:
        tuple: (is_valid: bool, invalid_attrs: list, reason: str)
    """
    if not attrs_used:
        return True, [], "no attrs to validate"

    invalid = []
    noisy_lower = noisy_query.lower()

    for key, value in attrs_used.items():
        if not isinstance(value, str):
            continue
        value_lower = value.lower()
        # 使用单词边界匹配，确保是独立单词而非子串
        pattern = r'(?<![a-zA-Z])' + re.escape(value_lower) + r'(?![a-zA-Z])'
        matches = re.findall(pattern, noisy_lower)
        count = len(matches)
        if count != 1:
            invalid.append(f"{key}={value_lower!r} 出现 {count} 次")

    if invalid:
        return False, invalid, f"attrs invalid: {', '.join(invalid)}"

    return True, [], "all attrs preserved exactly once"


# ========================================
# Prompt 构建
# ========================================
def build_noisy_prompt(error_patterns: list, user_content_template: str, inject_count: int = 3, attrs_used: dict = None) -> str:
    """构建噪声查询 prompt - 只基于错误模式和属性值生成查询"""
    # 只列出错误词（original），不给出正确词
    errors_section = "User's typical spelling errors (typos the user commonly makes):\n"
    for ep in error_patterns[:5]:
        errors_section += f"- {ep['original']}\n"

    attrs_list = ""
    if attrs_used:
        attrs_list = "\n".join([f"- {v}" for v in attrs_used.values()])

    user_content = user_content_template.format(
        errors_section=errors_section,
        inject_count=min(inject_count, len(error_patterns)),
        attrs_list=attrs_list
    )
    return user_content


# ========================================
# JSON 数组写入（标准格式）
# ========================================
def write_json_array(items: list, output_file: str, append: bool = True):
    """写入标准 JSON 数组格式（带缩进），支持追加模式"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    existing_items = []
    if append and os.path.exists(output_file):
        existing_items = load_appended_json_objects(output_file)
    
    # 去重：基于 uid 和 asin
    existing_keys = {(item.get('uid'), item.get('asin')) for item in existing_items}
    new_items = [item for item in items if (item.get('uid'), item.get('asin')) not in existing_keys]
    
    all_items = existing_items + new_items

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)


# ========================================
# 查询任务处理（共同逻辑）
# ========================================
def process_noisy_query_task(task: dict, noisy_user_content_template: str, inject_error_count: int, noisy_system_base: str, task_idx: int = None) -> dict:
    """处理单个查询任务的共同逻辑

    Args:
        task: 任务字典，包含 uid, asin, clean_query, errors, attrs_used
        noisy_user_content_template: prompt 模板
        inject_error_count: 注入错误数量
        noisy_system_base: system prompt
        task_idx: 任务索引（用于日志，可选）

    Returns:
        dict: 结果字典
    """
    uid = task['uid']
    asin = task['asin']
    clean_query = task['clean_query']
    error_patterns = task['errors']
    attrs_used = task.get('attrs_used')

    _log = log  # 使用模块的 log 函数

    try:
        injectable_patterns = build_injectable_patterns(error_patterns)

        if injectable_patterns and len(injectable_patterns) > 0:
            # ========== LLM 调用部分 ==========
            user_content = build_noisy_prompt(
                injectable_patterns,
                noisy_user_content_template, inject_error_count,
                attrs_used=attrs_used
            )
            log(f"[LLM REQUEST] user_content:\n{user_content}\n[/LLM REQUEST]")
            response, token_usage = call_llm(user_content, noisy_system_base)
            log(f"[LLM] input_tokens={token_usage.get('input_tokens',0)} output_tokens={token_usage.get('output_tokens',0)} cache_read={token_usage.get('cache_read_input_tokens',0)} cache_create={token_usage.get('cache_creation_input_tokens',0)}")
            response = fix_incomplete_json(response)
            parsed = parse_noisy_response(response)

            # 检查是否有 candidates 字段（10个候选）
            candidates = parsed.get('candidates', []) if parsed else []

            if candidates:
                # 从10个候选中选择第一个符合要求的
                selected_candidate = None
                for i, cand in enumerate(candidates):
                    noisy_query = cand.get('noisy_query', '')
                    injected_errors = cand.get('injected_errors', [])

                    # 验证属性值是否被保留
                    attr_valid, invalid_attrs, attr_msg = validate_attrs_preserved(attrs_used, noisy_query)
                    if not attr_valid:
                        _log(f"[DEBUG] 候选 {i} 属性验证失败: {attr_msg} | noisy_query: {noisy_query[:60]}")
                        continue

                    # 验证注入的错误是否有效
                    is_valid, valid_errors, validate_msg = validate_injected_errors(
                        injected_errors, injectable_patterns, noisy_query
                    )
                    if not is_valid:
                        _log(f"[DEBUG] 候选 {i} 错误验证失败: {validate_msg} | injected_errors: {injected_errors}")
                        continue

                    # 找到一个有效的候选
                    selected_candidate = {
                        'noisy_query': noisy_query,
                        'valid_errors': valid_errors,
                    }
                    break

                if selected_candidate:
                    # 构建 error -> corrected 映射，用于派生 clean_query
                    error_to_correct = {p['original'].lower(): p['corrected'] for p in injectable_patterns}
                    noisy_query = selected_candidate['noisy_query']

                    # 通过替换错误词得到派生 clean_query
                    derived_clean = noisy_query
                    for e in selected_candidate['valid_errors']:
                        err_word = e.get('error', '')
                        corr_word = error_to_correct.get(err_word.lower(), err_word)
                        # 不区分大小写替换
                        pattern = re.compile(re.escape(err_word), re.IGNORECASE)
                        derived_clean = pattern.sub(corr_word, derived_clean, count=1)

                    result = {
                        'uid': uid, 'asin': asin,
                        'clean_query': derived_clean,
                        'noisy_query': noisy_query,
                        'query_rewritten': noisy_query != derived_clean,
                        'injection_mode': 'llm_generated', 'status': 'success',
                        'injected_errors': [
                            build_injected_error({'original': e.get('error', ''), 'corrected': error_to_correct.get(e.get('error', '').lower(), ''), 'error_type': e.get('error_type', 'writing_error')}, {}, 'llm')
                            for e in selected_candidate['valid_errors']
                        ],
                    }
                    return result

                # 所有候选都不符合要求
                debug_prefix = f"任务 {task_idx}" if task_idx is not None else ""
                _log(f"[DEBUG] {debug_prefix} 所有10个候选都不符合要求")
                _log(f"  clean_query: {clean_query[:80]}")
                _log(f"  attrs_used: {attrs_used}")
                _log(f"  injectable_patterns: {injectable_patterns[:3]}")
                _log(f"  candidates count: {len(candidates)}")
                return {
                    'uid': uid, 'asin': asin, 'clean_query': clean_query,
                    'noisy_query': clean_query, 'query_rewritten': False,
                    'injection_mode': 'all_candidates_failed', 'status': 'success',
                    'injected_errors': [],
                    'debug_response': "all 10 candidates validation failed",
                }

            # 旧格式兼容：单个 noisy_query
            elif parsed and 'noisy_query' in parsed:
                noisy_query = parsed['noisy_query']
                injected_errors = parsed.get('injected_errors', [])

                # 验证属性值是否被保留
                attr_valid, invalid_attrs, attr_msg = validate_attrs_preserved(attrs_used, noisy_query)
                if not attr_valid:
                    _log(f"[DEBUG] 属性值验证失败 - {attr_msg}")
                    _log(f"  clean_query: {clean_query[:80]}")
                    _log(f"  attrs_used: {attrs_used}")
                    _log(f"  invalid_attrs: {invalid_attrs}")
                    _log(f"  noisy_query: {noisy_query[:80]}")
                    return {
                        'uid': uid, 'asin': asin, 'clean_query': clean_query,
                        'noisy_query': clean_query, 'query_rewritten': False,
                        'injection_mode': 'attrs_validation_failed', 'status': 'success',
                        'injected_errors': [],
                        'debug_response': f"attrs validation: {attr_msg}",
                    }

                # 验证注入的错误是否有效
                is_valid, valid_errors, validate_msg = validate_injected_errors(
                    injected_errors, injectable_patterns, noisy_query
                )

                if not is_valid:
                    debug_prefix = f"任务 {task_idx}" if task_idx is not None else ""
                    _log(f"[DEBUG] {debug_prefix} 验证失败 - {validate_msg}")
                    _log(f"  clean_query: {clean_query[:80]}")
                    _log(f"  injectable_patterns: {injectable_patterns[:3]}")
                    _log(f"  parsed noisy_query: {parsed.get('noisy_query', '')[:80] if parsed else 'N/A'}")
                    _log(f"  parsed injected_errors: {parsed.get('injected_errors', []) if parsed else []}")
                    return {
                        'uid': uid, 'asin': asin, 'clean_query': clean_query,
                        'noisy_query': clean_query, 'query_rewritten': False,
                        'injection_mode': 'validation_failed', 'status': 'success',
                        'injected_errors': [],
                        'debug_response': f"validation: {validate_msg}",
                    }

                # 构建 error -> corrected 映射，用于派生 clean_query
                error_to_correct = {p['original'].lower(): p['corrected'] for p in injectable_patterns}

                # 通过替换错误词得到派生 clean_query
                derived_clean = noisy_query
                for e in valid_errors:
                    err_word = e.get('error', '')
                    corr_word = error_to_correct.get(err_word.lower(), err_word)
                    pattern = re.compile(re.escape(err_word), re.IGNORECASE)
                    derived_clean = pattern.sub(corr_word, derived_clean, count=1)

                result = {
                    'uid': uid, 'asin': asin,
                    'clean_query': derived_clean,
                    'noisy_query': noisy_query, 'query_rewritten': noisy_query != derived_clean,
                    'injection_mode': 'llm_generated', 'status': 'success',
                    'injected_errors': [
                        build_injected_error({'original': e.get('error', ''), 'corrected': error_to_correct.get(e.get('error', '').lower(), ''), 'error_type': e.get('error_type', 'writing_error')}, {}, 'llm')
                        for e in valid_errors
                    ],
                }
                return result
            else:
                # 调试信息
                debug_prefix = f"任务 {task_idx}" if task_idx is not None else ""
                _log(f"[DEBUG] {debug_prefix} LLM 解析失败")
                _log(f"  clean_noisy_query: {clean_query[:50]}...")
                _log(f"  injectable_patterns: {injectable_patterns[:2]}")
                _log(f"  response: {response[:200] if response else 'empty'}...")
                _log(f"  parsed: {parsed}")
                return {
                    'uid': uid, 'asin': asin, 'clean_query': clean_query,
                    'noisy_query': clean_query, 'query_rewritten': False,
                    'injection_mode': 'llm_parse_failed', 'status': 'success',
                    'injected_errors': [],
                    'debug_response': response[:500] if response else '',
                }

        return {
            'uid': uid, 'asin': asin, 'clean_query': clean_query,
            'noisy_query': clean_query, 'query_rewritten': False,
            'injection_mode': 'no_pattern', 'status': 'success',
            'injected_errors': [],
        }
    except Exception as e:
        debug_prefix = f"任务 {task_idx}" if task_idx is not None else ""
        _log(f"[DEBUG] {debug_prefix} 异常: {e}")
        return {
            'uid': uid, 'asin': asin, 'clean_query': clean_query,
            'status': 'error', 'error': str(e),
        }
