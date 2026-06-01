#!/usr/bin/env python3
"""基于首尾字母匹配的噪声查询生成 - Baby_Products

不使用 LambdaMART，直接用首尾字母匹配规则注入错误。
"""

import json
import re
import sys
from pathlib import Path

STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'also', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'im', 'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'cant',
    'couldnt', 'shouldnt', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt',
    'havent', 'hadnt', 'hes', 'shes', 'its', 'theyre', 'weve', 'ive',
    'youre', 'thats', 'whats', 'whos', 'wheres', 'whens', 'hows',
    'lets', 'thats', 'theres', 'heres', 'whys', 'cuz', 'cause', 'bc',
}


def tokenize_query(query):
    query_fixed = re.sub(r"(\w)'(\w)", r"\1'\2", query)
    tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)*", query_fixed)
    return tokens


def remove_stop_words(tokens):
    result = []
    for token in tokens:
        token_lower = token.lower()
        if token.isdigit():
            continue
        if len(token) < 2:
            continue
        if token_lower not in STOP_WORDS:
            result.append(token)
    return result


def find_matching_error(token, user_errors):
    """找到首尾字母匹配的错误"""
    token_lower = token.lower()

    for err in user_errors:
        original = err.get('original', '').lower()
        corrected = err.get('corrected', '').lower()
        if not original or not corrected:
            continue

        # 如果 token == original，直接匹配
        if token_lower == original:
            return err

        # 首字母尾字母必须相同
        if token_lower and original:
            if token_lower[0] == original[0] and token_lower[-1] == original[-1]:
                return err

    return None


def apply_error_to_query(query, error_case, token):
    """将 token 替换为 error_case 的 original"""
    original = error_case.get('original', '')
    pattern = re.compile(re.escape(token), re.IGNORECASE)
    noisy_query = pattern.sub(original, query, count=1)
    return noisy_query


def main():
    category = 'Baby_Products'
    base = '/home/wlia0047/ar57/wenyu'

    query_file = f'{base}/result/personal_query/06_query/{category}/query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json'
    user_error_file = f'{base}/result/personal_query/04_writing_analysis/{category}/writing_error.json'
    output_file = f'{base}/result/personal_query/07_inject_noisy/{category}/noisy_query_firstlast.json'

    print("=" * 60)
    print(f"首尾匹配噪声注入 ({category})")
    print("=" * 60)

    # 加载用户错误
    print("\n加载用户错误数据...")
    with open(user_error_file, 'r', encoding='utf-8') as f:
        user_errors_data = json.load(f)

    user_errors_dict = {}
    for user in user_errors_data:
        uid = user['user_id']
        error_details = user.get('error_details', [])
        if error_details:
            user_errors_dict[uid] = error_details
    print(f"有错误数据的用户: {len(user_errors_dict)}")

    # 加载查询
    print("\n加载查询记录...")
    with open(query_file, 'r', encoding='utf-8') as f:
        query_data = json.load(f)
    if isinstance(query_data, dict):
        queries = query_data.get('records', [])
    else:
        queries = query_data
    print(f"查询记录总数: {len(queries)}")

    # 处理
    print("\n开始处理...")
    results = []
    total = len(queries)
    injected = 0

    for idx, rec in enumerate(queries, 1):
        uid = rec.get('uid') or rec.get('user_id')
        asin = rec.get('asin')
        query = rec.get('query') or rec.get('clean_query') or rec.get('syntax_depth_query', {}).get('query', '')

        if not uid or not asin or not query:
            continue

        user_errors = user_errors_dict.get(uid, [])
        if not user_errors:
            continue

        tokens = tokenize_query(query)
        clean_tokens = remove_stop_words(tokens)

        if not clean_tokens:
            continue

        # 遍历所有 token，找第一个匹配的
        selected_token = None
        error_case = None
        noisy_query = query

        for token in clean_tokens:
            match = find_matching_error(token, user_errors)
            if match:
                selected_token = token
                error_case = match
                noisy_query = apply_error_to_query(query, match, token)
                break

        if selected_token and error_case:
            injected += 1
            results.append({
                'uid': uid,
                'asin': asin,
                'clean_query': query,
                'noisy_query': noisy_query,
                'query_rewritten': True,
                'selected_token': selected_token,
                'applied_error': {
                    'original': error_case.get('original'),
                    'corrected': error_case.get('corrected'),
                    'error_type': error_case.get('error_type', 'writing_error'),
                },
                'status': 'success',
            })
        else:
            results.append({
                'uid': uid,
                'asin': asin,
                'clean_query': query,
                'noisy_query': query,
                'query_rewritten': False,
                'selected_token': None,
                'applied_error': None,
                'status': 'no_match',
            })

        if idx % 1000 == 0 or idx == total:
            print(f"[{idx}/{total}] 已注入: {injected}")

    # 只保留注入错误的记录
    injected_results = [r for r in results if r.get('query_rewritten')]

    print(f"\n处理完成: {len(results)} 条记录")
    print(f"注入成功: {len(injected_results)} 条")

    # 写入
    if injected_results:
        print(f"\n写入结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(injected_results, f, ensure_ascii=False, indent=2)
        print(f"写入完成: {len(injected_results)} 条记录")

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == '__main__':
    main()