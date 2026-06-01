#!/usr/bin/env python3
"""使用增强版 LambdaMART 模型为 Baby Products 生成噪声查询

使用训练好的增强版模型对每个 query 的 token 评分，选择最高分的 token 注入错误。
"""

import json
import os
import sys
import re
import glob
from pathlib import Path
from datetime import datetime
from collections import Counter

import lightgbm as lgb

# 添加当前目录到 sys.path
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from token_level_error_selector import apply_error_to_query
from common import (
    load_user_errors, load_query_records, build_query_tasks,
    write_json_array, log, load_completed_query_keys
)
from token_level_lambdamart_enhanced import (
    get_user_error_words, tokenize_query, remove_stop_words,
    compute_features_enhanced, classify_error_type, char_similarity,
    edit_distance
)


# ========================================
# 配置
# ========================================
def build_config(category: str) -> dict:
    """根据类别构建配置"""
    base = '/home/wlia0047/ar57/wenyu'
    return {
        'category': category,
        'query_file': f'{base}/result/personal_query/06_query/{category}/query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json',
        'user_error_file': f'{base}/result/personal_query/04_writing_analysis/{category}/writing_error.json',
        'model_file': f'{base}/result/personal_query/07_inject_noisy/models/lambdamart_{category}_enhanced.json',
        'output_file': f'{base}/result/personal_query/07_inject_noisy/{category}/noisy_query.json',
    }


CONFIG = build_config('Baby_Products')

# 特征名称（必须与训练时一致）
FEATURE_NAMES = [
    'topk_sim', 'char_score', 'bigram_score', 'surface_difficulty', 'attr_risk',
    'phonetic_sim', 'visual_sim', 'spelling_sim',
    'phonetic_ratio', 'visual_ratio', 'spelling_ratio',
    'is_color', 'is_size', 'is_brand', 'is_flavor', 'is_material', 'is_species',
    'user_error_rate', 'token_len', 'has_upper', 'has_digit', 'has_rare_char',
    'has_double_letter', 'vowel_ratio'
]


def find_similar_error(token: str, user_errors: list, threshold: float = 0.7) -> dict:
    """找到与 token 最相似的用户历史错误

    关键：用户把 corrected（正确词）拼成了 original（错误词）
    所以我们应该找与 token（正确词）高度相似的 original

    注入规则：
    - 首字母尾字母必须相同（与 original 匹配）
    - 如果 token 就是 original，则直接匹配

    返回的 error_case 会被用来把 token 替换成 original（用户的错误形式）
    """
    if not user_errors:
        return None

    best_error = None

    for err in user_errors:
        original = err.get('original', '')
        corrected = err.get('corrected', '')
        if not original or not corrected:
            continue

        token_lower = token.lower()
        original_lower = original.lower()

        # 如果 token 就是 original，直接匹配
        if token_lower == original_lower:
            best_error = err
            break

        # 首字母尾字母必须相同（与 original 匹配）
        if not token_lower or not original_lower:
            continue
        if token_lower[0] == original_lower[0] and token_lower[-1] == original_lower[-1]:
            best_error = err
            break

    return best_error


def predict_and_select_token(model, tokens: list, corrected_words: set, error_words: set,
                             char_freq: Counter, bigram_freq: Counter,
                             error_types: dict, phonetic_errors: list,
                             visual_errors: list, spelling_errors: list,
                             user_errors: list, category: str, user_error_count: int) -> tuple:
    """使用 LambdaMART 模型预测并选择要注入错误的 token

    Returns:
        (selected_token, score, error_case)
    """
    if not tokens:
        return None, 0.0, None

    # 计算每个 token 的特征
    X = []
    valid_tokens = []
    for token in tokens:
        features = compute_features_enhanced(
            token,
            corrected_words, error_words, char_freq, bigram_freq,
            error_types, phonetic_errors, visual_errors, spelling_errors,
            attrs_used=None,
            category=category,
            user_error_count=user_error_count
        )
        X.append([features.get(f, 0.0) for f in FEATURE_NAMES])
        valid_tokens.append(token)

    # 模型预测
    scores = model.predict(X)

    # 找到分数最高的 token
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_token = valid_tokens[best_idx]
    best_score = scores[best_idx]

    # 找到最相似的错误
    best_error = find_similar_error(best_token, user_errors)

    return best_token, best_score, best_error


def process_batch_lambdamart(tasks: list, model, completed_keys: set):
    """使用 LambdaMART 模型批量处理任务"""
    results = []
    total = len(tasks)

    for idx, task in enumerate(tasks, 1):
        uid = task['uid']
        asin = task['asin']
        clean_query = task['clean_query']
        user_errors = task['errors']

        record_key = (uid, asin)
        if record_key in completed_keys:
            log(f"[{idx}/{total}] 跳过已完成: uid={uid[:12]}, asin={asin}")
            continue

        try:
            # Tokenize
            tokens = tokenize_query(clean_query)
            clean_tokens = remove_stop_words(tokens)

            if not clean_tokens or not user_errors:
                # 没有有效 token 或没有用户错误
                results.append({
                    'uid': uid,
                    'asin': asin,
                    'clean_query': clean_query,
                    'noisy_query': clean_query,
                    'query_rewritten': False,
                    'selected_token': None,
                    'score': 0.0,
                    'applied_error': None,
                    'status': 'skipped',
                    'reason': 'no_tokens_or_errors',
                })
                continue

            # 提取用户错误特征
            (
                corrected_words, error_words, char_freq, bigram_freq,
                error_types, phonetic_errors, visual_errors, spelling_errors
            ) = get_user_error_words(user_errors)

            # LambdaMART 预测并选择 token
            selected_token, score, error_case = predict_and_select_token(
                model, clean_tokens,
                corrected_words, error_words, char_freq, bigram_freq,
                error_types, phonetic_errors, visual_errors, spelling_errors,
                user_errors=user_errors,
                category=CONFIG['category'],
                user_error_count=len(user_errors)
            )

            # 应用错误（选择分数最高的 token，只要有匹配的错误就注入）
            noisy_query = clean_query
            applied_error = None
            if selected_token and error_case:
                noisy_query = apply_error_to_query(clean_query, error_case, selected_token)
                applied_error = {
                    'original': error_case.get('original'),
                    'corrected': error_case.get('corrected'),
                    'error_type': error_case.get('error_type', 'writing_error'),
                    'similarity': char_similarity(selected_token.lower(), error_case.get('original', '').lower()),
                }

            output_item = {
                'uid': uid,
                'asin': asin,
                'clean_query': clean_query,
                'noisy_query': noisy_query,
                'query_rewritten': noisy_query != clean_query,
                'selected_token': selected_token,
                'score': float(score) if score else 0.0,
                'applied_error': applied_error,
                'status': 'success',
            }

            results.append(output_item)

            if idx % 500 == 0 or idx == total:
                log(f"[{idx}/{total}] 处理中: uid={uid[:12]}, token={selected_token}, score={score:.2f}")

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


def main(category: str = None):
    if category is None:
        category = CONFIG['category']

    config = build_config(category)

    print("=" * 60)
    print(f"LambdaMART 噪声注入 ({category})")
    print("=" * 60)

    category = config['category']
    query_file = config['query_file']
    user_error_file = config['user_error_file']
    model_file = config['model_file']
    output_file = config['output_file']

    print(f"Query 文件: {query_file}")
    print(f"用户错误文件: {user_error_file}")
    print(f"模型文件: {model_file}")
    print(f"输出文件: {output_file}")

    # 加载模型
    print("\n加载 LambdaMART 模型...")
    model = lgb.Booster(model_file=model_file)
    print("模型加载成功")

    # 加载用户错误数据
    print("\n加载用户错误数据...")
    user_errors = load_user_errors(user_error_file)
    print(f"有错误数据的用户: {len(user_errors)}")

    # 加载查询记录
    print("\n加载查询记录...")
    query_records = load_query_records(query_file)
    print(f"查询记录总数: {len(query_records)}")

    # 构建查询任务
    print("\n构建查询任务...")
    completed_keys = load_completed_query_keys(output_file)
    print(f"已完成的任务: {len(completed_keys)}")

    tasks = build_query_tasks(query_records, user_errors, completed_keys)
    print(f"待处理任务: {len(tasks)}")

    if not tasks:
        print("没有待处理的任务，退出")
        return

    # 批量处理
    print("\n开始处理...")
    results = process_batch_lambdamart(tasks, model, completed_keys)
    print(f"处理完成: {len(results)} 个结果")

    # 只保留注入了错误的记录
    injected_results = [r for r in results if r.get('query_rewritten')]

    # 写入结果
    if injected_results:
        print(f"\n写入结果到: {output_file}")
        write_json_array(injected_results, output_file, append=False)  # 不追加，覆盖写入
        print(f"写入完成: {len(injected_results)} 条记录")

    # 统计
    success_count = sum(1 for r in results if r.get('status') == 'success')
    rewritten_count = len(injected_results)
    total_count = len(results)
    print(f"\n统计:")
    print(f"  总记录: {total_count}")
    print(f"  成功: {success_count}")
    print(f"  注入错误: {rewritten_count}")
    print(f"  注入率: {rewritten_count/total_count*100:.1f}%")

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == '__main__':
    import sys
    category = sys.argv[1] if len(sys.argv) > 1 else None
    main(category)
