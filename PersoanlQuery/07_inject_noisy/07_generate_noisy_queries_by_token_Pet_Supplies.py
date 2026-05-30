#!/usr/bin/env python3
"""基于 Token 级别错误选择的噪声注入脚本 (Baby Products)

使用 token_level_error_selector 对每个 query 选择最合适的 token 进行错误注入。
"""

import sys
import json
import os
from datetime import datetime
from pathlib import Path

# 添加当前目录到 sys.path
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common.token_level_error_selector import (
    select_and_apply_error, process_user_query_task,
    apply_error_to_query
)
from common.common import (
    load_user_errors, load_query_records, build_query_tasks,
    write_json_array, log, load_completed_query_keys
)


# ========================================
# 配置
# ========================================
CONFIG = {
    'category': 'Pet_Supplies',
    'query_file': '/home/wlia0047/ar57/wenyu/result/personal_query/06_query/Pet_Supplies/query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json',
    'user_error_file': '/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/Pet_Supplies/writing_error.json',
    'output_file': '/home/wlia0047/ar57/wenyu/result/personal_query/07_inject_noisy/Pet_Supplies/noisy_query_by_token.json',
    'max_workers': 10,
    'strategy': 'highest',  # 'highest' 或 'sample'
    'top_n': 3,             # 从 top-n 中选择
}


def process_batch(tasks, completed_keys):
    """批量处理任务"""
    results = []
    total = len(tasks)

    for idx, task in enumerate(tasks, 1):
        uid = task['uid']
        asin = task['asin']
        clean_query = task['clean_query']
        user_errors = task['errors']
        attrs_used = task.get('attrs_used')

        record_key = (uid, asin)
        if record_key in completed_keys:
            log(f"[{idx}/{total}] 跳过已完成: uid={uid[:12]}, asin={asin}")
            continue

        try:
            # 使用 token 选择器处理
            result = select_and_apply_error(
                query=clean_query,
                user_errors=user_errors,
                attrs_used=attrs_used,
                strategy=CONFIG['strategy'],
                top_n=CONFIG['top_n']
            )

            # 应用错误到查询
            noisy_query = clean_query
            if result.get('applied_error') and result.get('selected_token'):
                noisy_query = apply_error_to_query(
                    clean_query,
                    result['applied_error'],
                    result['selected_token']
                )

            output_item = {
                'uid': uid,
                'asin': asin,
                'clean_query': clean_query,
                'noisy_query': noisy_query,
                'query_rewritten': noisy_query != clean_query,
                'selected_token': result.get('selected_token'),
                'score': result.get('score', 0.0),
                'applied_error': result.get('applied_error'),
                'similar_cases': result.get('similar_cases', []),
                'status': 'success',
            }

            results.append(output_item)

            if idx % 100 == 0 or idx == total:
                log(f"[{idx}/{total}] 处理中: uid={uid[:12]}, token={result.get('selected_token')}, score={result.get('score', 0):.2f}")

        except Exception as e:
            log(f"[{idx}/{total}] 错误: uid={uid[:12]}, error={e}")
            results.append({
                'uid': uid,
                'asin': asin,
                'clean_query': clean_query,
                'noisy_query': clean_query,
                'query_rewritten': False,
                'selected_token': None,
                'score': 0.0,
                'applied_error': None,
                'status': 'error',
                'error': str(e),
            })

    return results


def main():
    start_time = datetime.now()
    log("=" * 60)
    log("基于 Token 级别错误选择的噪声注入 (Baby Products)")
    log("=" * 60)

    category = CONFIG['category']
    query_file = CONFIG['query_file']
    user_error_file = CONFIG['user_error_file']
    output_file = CONFIG['output_file']

    log(f"Query 文件: {query_file}")
    log(f"用户错误文件: {user_error_file}")
    log(f"输出文件: {output_file}")
    log(f"策略: strategy={CONFIG['strategy']}, top_n={CONFIG['top_n']}")

    # 加载用户错误数据
    log("加载用户错误数据...")
    user_errors = load_user_errors(user_error_file)
    log(f"有错误数据的用户: {len(user_errors)}")

    # 加载查询记录
    log("加载查询记录...")
    query_records = load_query_records(query_file)
    log(f"查询记录总数: {len(query_records)}")

    # 构建查询任务
    log("构建查询任务...")
    completed_keys = load_completed_query_keys(output_file)
    log(f"已完成的任务: {len(completed_keys)}")

    tasks = build_query_tasks(query_records, user_errors, completed_keys)
    log(f"待处理任务: {len(tasks)}")

    if not tasks:
        log("没有待处理的任务，退出")
        return

    # 批量处理
    log("开始处理...")
    results = process_batch(tasks, completed_keys)
    log(f"处理完成: {len(results)} 个结果")

    # 写入结果
    if results:
        log(f"写入结果到: {output_file}")
        write_json_array(results, output_file, append=True)
        log(f"写入完成")

    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"总耗时: {elapsed:.1f} 秒")
    log(f"处理速度: {len(results) / elapsed:.1f} 条/秒")

    # 统计
    success_count = sum(1 for r in results if r.get('status') == 'success')
    rewritten_count = sum(1 for r in results if r.get('query_rewritten'))
    log(f"成功: {success_count}, 注入错误: {rewritten_count}")

    log("=" * 60)
    log("完成")
    log("=" * 60)


if __name__ == '__main__':
    main()
