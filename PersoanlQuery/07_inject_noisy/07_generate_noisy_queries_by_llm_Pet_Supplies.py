#!/usr/bin/env python3
"""
基于 LLM 的查询语句噪声注入 - Baby_Products
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery/07_inject_noisy')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery/07_inject_noisy/common')

from common import (
    log, get_required_config_value, load_noisy_config, load_noisy_prompts,
    load_user_errors, merge_filtered_user_errors,
    call_llm, prewarm_noisy_cache, parse_noisy_response, fix_incomplete_json,
    build_injectable_patterns, build_noisy_prompt, build_injected_error,
    write_noisy_result_incremental, load_completed_query_keys, load_appended_json_objects, write_json_array, validate_injected_errors, validate_attrs_preserved,
    load_query_records, build_query_tasks, process_noisy_query_task,
)

# ========================================
# 配置加载
# ========================================
CATEGORY = "Pet_Supplies"

_NOISY_CONFIG = load_noisy_config()
_CATEGORY_CONFIG = get_required_config_value(_NOISY_CONFIG, 'categories', CATEGORY)

MAX_WORKERS = get_required_config_value(_NOISY_CONFIG, 'max_workers')
INJECT_ERROR_COUNT = get_required_config_value(_NOISY_CONFIG, 'inject_error_count')
NUM_USERS_TO_TEST = _NOISY_CONFIG.get('num_users_to_test')  # 可选，None 表示不限制
if not isinstance(MAX_WORKERS, int) or MAX_WORKERS <= 0:
    raise ValueError(f"配置 max_workers 必须是正整数: {MAX_WORKERS!r}")
EFFECTIVE_MAX_WORKERS = MAX_WORKERS

QUERY_FILE = get_required_config_value(_CATEGORY_CONFIG, 'query_file')
USER_ERROR_FILE = get_required_config_value(_CATEGORY_CONFIG, 'user_error_file')
NOISY_OUTPUT_FILE = get_required_config_value(_CATEGORY_CONFIG, 'noisy_output_file')

_PROMPTS = load_noisy_prompts(CATEGORY)
NOISY_SYSTEM_BASE = _PROMPTS['NOISY_SYSTEM_BASE']
NOISY_USER_CONTENT_TEMPLATE = _PROMPTS['NOISY_USER_CONTENT_TEMPLATE']



# ========================================
# 主逻辑
# ========================================

def main():
    """主函数"""
    log(f"开始噪声注入任务 - {CATEGORY}")
    
    # 预热 LLM 缓存
    log("预热 LLM 缓存...")
    prewarm_noisy_cache(NOISY_SYSTEM_BASE, NOISY_USER_CONTENT_TEMPLATE)
    log("缓存预热完成")
    
    raw_user_errors = load_user_errors(USER_ERROR_FILE)
    user_errors = merge_filtered_user_errors(raw_user_errors)
    log(f"过滤并合并后，有错误模式的用户数: {len(user_errors)}")
    
    query_records = load_query_records(QUERY_FILE)
    
    # 根据 num_users_to_test 限制用户数量
    if NUM_USERS_TO_TEST is not None:
        unique_users = list(dict.fromkeys(r.get('user_id') for r in query_records if r.get('user_id')))[:NUM_USERS_TO_TEST]
        user_set = set(unique_users)
        query_records = [r for r in query_records if r.get('user_id') in user_set]
        log(f"限制用户数: {NUM_USERS_TO_TEST}, 筛选后 {len(query_records)} 条查询记录")
    else:
        log(f"加载了 {len(query_records)} 条查询记录")
    
    completed_keys = load_completed_query_keys(NOISY_OUTPUT_FILE)
    log(f"已有 {len(completed_keys)} 条完成的记录")
    
    tasks = build_query_tasks(query_records, user_errors, completed_keys)
    log(f"本次任务数: {len(tasks)}")
    
    if not tasks:
        raise ValueError("没有可处理的任务")
    
    log(f"并发 worker 数: {EFFECTIVE_MAX_WORKERS}")
    status_counts = {}
    success_count = 0
    all_results = []  # 收集所有成功结果
    
    with ThreadPoolExecutor(max_workers=EFFECTIVE_MAX_WORKERS) as executor:
        futures = {executor.submit(process_noisy_query_task, task, NOISY_USER_CONTENT_TEMPLATE, INJECT_ERROR_COUNT, NOISY_SYSTEM_BASE): task for task in tasks}
        for processed_count, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                log(f"[ERROR] 任务异常退出: user={task['uid'][:20]} err={type(exc).__name__}: {exc}")
                result = {
                    'uid': task['uid'], 'asin': task['asin'],
                    'clean_query': task['clean_query'],
                    'status': 'task_exception', 'error': f'{type(exc).__name__}: {exc}',
                }
            
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1

            # 只有真正成功注入错误的才保存
            if status == 'success' and result.get('query_rewritten'):
                success_count += 1
                all_results.append(result)
                write_json_array(all_results, NOISY_OUTPUT_FILE)

            log(f"  [进度 {processed_count}/{len(tasks)}] 成功:{success_count} 当前状态:{status} user={task['uid'][:20]}")
    
    log(f"\n{'='*60}")
    log("==================== 统计结果 ====================")
    log(f"总任务数: {len(tasks)}")
    log(f"成功注入: {success_count} ({success_count/len(tasks)*100:.1f}%)")
    for status in sorted(status_counts):
        log(f"  - {status}: {status_counts[status]}")
    log("正在写入结果文件...")
    
    # 写入标准 JSON 数组格式
    write_json_array(all_results, NOISY_OUTPUT_FILE)
    log(f"结果已写入（标准JSON数组格式）: {NOISY_OUTPUT_FILE}")


if __name__ == '__main__':
    main()
