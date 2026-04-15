#!/usr/bin/env python3
"""
快速全量评估脚本 - 支持多检索器 + ACL/CCOMP 分组交叉对比
包括: bge, e5, minilm, star, gritlm (密集) + bm25 (稀疏)
包含 ACL/CCOMP 混淆因素分析 (Check 1-4 + Bootstrap CI)
"""

# 完全离线模式 - 避免 HuggingFace 网络验证
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import time
import pickle
import json
import gzip
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
from scipy import stats
import statsmodels.formula.api as smf

# 设置路径
sys.path.insert(0, '/fs04/ar57/wenyu/PersoanlQuery/12_retrieval')

# ============ 配置 ============
CACHE_DIR = "/home/wlia0047/ar57_scratch/wenyu/result/personal_query/08_retrieval/retriever_Grocery_and_Gourmet_Food_cache"
QUERY_CACHE_BASE_DIR = "/home/wlia0047/ar57_scratch/wenyu/result/personal_query/08_retrieval/query_cache_Grocery_and_Gourmet_Food"
QUERY_TYPES = ['correct', 'noisy']  # 两种查询类型
QUERIES_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/06_query/Grocery_and_Gourmet_Food/ccomp_query.json"
OUTPUT_DIR = "/home/wlia0047/ar57_scratch/wenyu/result/personal_query/08_retrieval/Grocery_and_Gourmet_Food"
META_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl.gz"
CATEGORY_NAME = "_Grocery_and_Gourmet_Food"

# 要评估的检索器列表
RETRIEVERS = ['bge', 'e5', 'minilm', 'star', 'gritlm', 'bm25']
DENSE_RETRIEVERS = ['bge', 'e5', 'minilm', 'star', 'gritlm']

# IDF 分层配置 (基于查询词的平均IDF, 根据实际分布 [2.97, 5.79] 调整)
IDF_BINS = [(2.5, 3.5), (3.5, 4.5), (4.5, 5.0), (5.0, float('inf'))]
IDF_BIN_LABELS = ['IDF[2.5-3.5)', 'IDF[3.5-4.5)', 'IDF[4.5-5.0)', 'IDF[5.0+)']

# ============ 日志 ============
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ============ 缓存完整性检查 ============
def validate_cache() -> bool:
    """检查缓存目录中的文件是否完整且有效"""
    log("\n检查缓存完整性...")

    if not os.path.exists(CACHE_DIR):
        log(f"  错误: 缓存目录不存在: {CACHE_DIR}")
        return False

    issues = []

    # 检查密集检索器文件及数据完整性
    for retriever in DENSE_RETRIEVERS:
        # 查找该检索器的所有版本
        # 文件格式: {retriever}_{hash}_{suffix}.npy/.pkl
        # hash 本身可能包含下划线，所以需要从后往前推断
        retriever_files = {}
        for f in os.listdir(CACHE_DIR):
            if not (f.startswith(f'{retriever}_') and f.endswith(('.npy', '.pkl'))):
                continue

            # 提取 suffix（最后一部分）
            suffix = f.rsplit('_', 1)[-1]

            # 提取 hash（去掉 retriever_ 前缀和 suffix 后缀）
            # 例如: bge_457d1871f380782c05a5d94e656fef2c_embeddings.npy
            # -> 去掉前缀 retriever_ 和后缀 _embeddings.npy
            middle = f[len(f'{retriever}_'):]
            hash_id = middle[:-len(suffix) - 1]  # -1 for the underscore before suffix

            if hash_id not in retriever_files:
                retriever_files[hash_id] = set()
            retriever_files[hash_id].add(suffix)

        if not retriever_files:
            issues.append(f"  缺失: {retriever} 检索器缓存文件")
            continue

        # 检查每个版本是否完整
        for hash_id, suffixes in retriever_files.items():
            required_files = ['embeddings.npy', 'doc_ids.pkl', 'config.pkl', 'metadata.pkl']
            for suffix in required_files:
                full_file = f"{retriever}_{hash_id}_{suffix}"
                file_path = os.path.join(CACHE_DIR, full_file)
                if suffix not in suffixes:
                    issues.append(f"  缺失: {retriever} ({hash_id[:8]}...) - {suffix}")
                elif os.path.getsize(file_path) == 0:
                    issues.append(f"  空文件: {full_file}")

            # 验证 embeddings 和 doc_ids 数量是否匹配
            embeddings_path = os.path.join(CACHE_DIR, f"{retriever}_{hash_id}_embeddings.npy")
            doc_ids_path = os.path.join(CACHE_DIR, f"{retriever}_{hash_id}_doc_ids.pkl")
            if os.path.exists(embeddings_path) and os.path.exists(doc_ids_path):
                try:
                    embeddings = np.load(embeddings_path, mmap_mode='r')
                    n_embeddings = embeddings.shape[0]
                    with open(doc_ids_path, 'rb') as f:
                        doc_ids = pickle.load(f)
                    n_doc_ids = len(doc_ids)

                    if n_embeddings != n_doc_ids:
                        issues.append(f"  数据不一致: {retriever} ({hash_id[:8]}...) - embeddings数量({n_embeddings}) != doc_ids数量({n_doc_ids})")

                    # 检查 doc_ids 是否有重复
                    if len(doc_ids) != len(set(doc_ids)):
                        duplicates = len(doc_ids) - len(set(doc_ids))
                        issues.append(f"  数据错误: {retriever} ({hash_id[:8]}...) - doc_ids中有 {duplicates} 个重复项")

                    log(f"  {retriever} ({hash_id[:8]}...): embeddings={n_embeddings}, doc_ids={n_doc_ids}")
                except Exception as e:
                    issues.append(f"  验证失败: {retriever} ({hash_id[:8]}...) - {str(e)}")

    # 检查 BM25 文件
    bm25_files = [f for f in os.listdir(CACHE_DIR) if f.startswith('bm25_') and f.endswith('.pkl')]
    if not bm25_files:
        issues.append("  缺失: bm25 检索器缓存文件")
    else:
        for f in bm25_files:
            file_path = os.path.join(CACHE_DIR, f)
            if os.path.getsize(file_path) == 0:
                issues.append(f"  空文件: {f}")
            else:
                # 验证 BM25 数据可加载
                try:
                    with open(file_path, 'rb') as fp:
                        bm25 = pickle.load(fp)
                    # 检查 BM25 是否有 search 方法
                    if not hasattr(bm25, 'search'):
                        issues.append(f"  数据错误: {f} - BM25对象缺少search方法")
                    else:
                        log(f"  bm25 ({f.split('_')[1][:8]}...): 可正常加载")
                except Exception as e:
                    issues.append(f"  验证失败: {f} - {str(e)}")

    # 检查查询缓存
    log("  检查查询缓存...")
    for query_type in QUERY_TYPES:
        query_cache_dir = os.path.join(QUERY_CACHE_BASE_DIR, f'persona_{query_type}_query')
        if not os.path.exists(query_cache_dir):
            issues.append(f"  缺失: 查询缓存目录 persona_{query_type}_query")
            continue

        for retriever in RETRIEVERS:
            cache_file = os.path.join(query_cache_dir, f'{retriever}__persona_{query_type}_cache.pkl')
            if not os.path.exists(cache_file):
                issues.append(f"  缺失: {retriever} ({query_type}) 查询缓存")
            else:
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    n_users = len(cache_data)
                    log(f"  {retriever} ({query_type}): {n_users} 用户")
                except Exception as e:
                    issues.append(f"  验证失败: {retriever} ({query_type}) - {str(e)}")

    if issues:
        log("  缓存完整性检查未通过:")
        for issue in issues:
            log(issue)
        return False

    log("  缓存完整性检查通过 ✓")
    return True

# ============ ACL/CCOMP 混淆因素分析 ============
# 模块级变量，动态从查询文件获取
UNIQUE_GROUPS = [0, 1, 2, 3]  # 默认值，会在 load_raw_queries 时更新
GROUP_FIELD = 'ccomp'  # 默认值，会在 load_raw_queries 时更新

def load_raw_queries() -> Tuple[Dict, List[int], str]:
    """加载原始查询数据用于分析，返回 (groups_dict, unique_groups, group_field_name)"""
    global UNIQUE_GROUPS, GROUP_FIELD

    with open(QUERIES_FILE, 'r') as f:
        data = json.load(f)

    # 检测字段名：ccomp 或 acl
    sample = data[0] if data else {}
    if 'queries' in sample:
        first_query = sample.get('queries', [{}])[0]
        GROUP_FIELD = 'acl' if 'acl' in first_query else 'ccomp'
    else:
        GROUP_FIELD = 'acl' if 'acl' in sample else 'ccomp'

    # 动态收集所有唯一的 group 值
    group_values = set()
    for item in data:
        if 'queries' in item:
            for q in item['queries']:
                gv = q.get(GROUP_FIELD, 0)
                group_values.add(gv)
        else:
            gv = item.get(f'target_{GROUP_FIELD}', item.get(GROUP_FIELD, 0))
            group_values.add(gv)

    UNIQUE_GROUPS = sorted(group_values)
    groups_dict = {g: [] for g in UNIQUE_GROUPS}

    for item in data:
        if 'queries' in item:
            asin = item.get('asin', '')
            for q in item['queries']:
                gv = q.get(GROUP_FIELD, 0)
                query = q.get('filled_query', '') or q.get('generated_query', '') or q.get('query', '')
                word_count = q.get('word_count', 0)
                if query and asin:
                    groups_dict[gv].append({
                        'query': query,
                        'asin': asin,
                        'word_count': word_count,
                        'ccomp_ratio': 0.0
                    })
        else:
            gv = item.get(f'target_{GROUP_FIELD}', item.get(GROUP_FIELD, 0))
            query = item.get('filled_query', '') or item.get('generated_query', '') or item.get('query', '')
            asin = item.get('asin', '')
            word_count = item.get('word_count') or 0
            ccomp_ratio = item.get('persona', {}).get(f'{GROUP_FIELD}_sentence_ratio', 0.0)
            if query and asin:
                groups_dict[gv].append({
                    'query': query,
                    'asin': asin,
                    'word_count': word_count,
                    'ccomp_ratio': ccomp_ratio
                })

    return groups_dict, UNIQUE_GROUPS, GROUP_FIELD

def check1_query_length(ccomp_groups):
    """Check 1: Query 长度分析"""
    log("\n" + "=" * 80)
    log("Check 1: Query 长度分析")
    log("=" * 80)

    results = {}
    for ccomp in UNIQUE_GROUPS:
        lengths = [q['word_count'] for q in ccomp_groups[ccomp]]
        results[ccomp] = {
            'count': len(lengths),
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'median': np.median(lengths)
        }

    header = f"{GROUP_FIELD.upper():<10} {'N':<8} {'Mean':<10} {'Std':<10} {'Min':<8} {'Max':<8} {'Median':<10}"
    log(header)
    log("-" * 80)
    for ccomp in UNIQUE_GROUPS:
        r = results[ccomp]
        log(f"{GROUP_FIELD.upper()}{ccomp}      {r['count']:<8} {r['mean']:<10.2f} {r['std']:<10.2f} {r['min']:<8} {r['max']:<8} {r['median']:<10.2f}")

    # Kruskal-Wallis 检验
    groups = [[q['word_count'] for q in ccomp_groups[c]] for c in UNIQUE_GROUPS]
    stat, p_value = stats.kruskal(*groups)
    log(f"\n  Kruskal-Wallis 检验: H={stat:.4f}, p={p_value:.4f}")
    if p_value < 0.05:
        log("  结论: 各组query长度存在显著差异 (p < 0.05) ⚠️ 长度可能是混淆因素!")
    else:
        log("  结论: 各组query长度无显著差异 (p >= 0.05)")
    return results

def check2_pos_ratio(ccomp_groups):
    """Check 2: Content/Function Word Ratio"""
    log("\n" + "=" * 80)
    log("Check 2: Content/Function Word Ratio (POS Tagging)")
    log("=" * 80)

    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
    except:
        log("  spaCy 未安装，跳过POS分析")
        return None

    CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    FUNCTION_POS = {'DET', 'ADP', 'AUX', 'PART', 'SCONJ', 'CCONJ', 'PRON'}

    results = {}
    all_ratios = {c: [] for c in UNIQUE_GROUPS}

    for ccomp in UNIQUE_GROUPS:
        content_list, function_list, ratios = [], [], []
        for q in ccomp_groups[ccomp]:
            doc = nlp(q['query'])
            content = sum(1 for t in doc if t.pos_ in CONTENT_POS)
            function = sum(1 for t in doc if t.pos_ in FUNCTION_POS)
            total = content + function
            ratio = content / total if total > 0 else 0
            content_list.append(content)
            function_list.append(function)
            ratios.append(ratio)
            all_ratios[ccomp].append(ratio)

        results[ccomp] = {
            'count': len(ratios),
            'mean_content': np.mean(content_list),
            'mean_function': np.mean(function_list),
            'mean_ratio': np.mean(ratios),
            'std_ratio': np.std(ratios)
        }

    header = f"{GROUP_FIELD.upper():<10} {'N':<8} {'Content':<10} {'Function':<10} {'Ratio':<10} {'Std':<10}"
    log(header)
    log("-" * 80)
    for ccomp in UNIQUE_GROUPS:
        r = results[ccomp]
        log(f"{GROUP_FIELD.upper()}{ccomp}      {r['count']:<8} {r['mean_content']:<10.2f} {r['mean_function']:<10.2f} {r['mean_ratio']:<10.4f} {r['std_ratio']:<10.4f}")

    # Kruskal-Wallis 检验
    stat, p_value = stats.kruskal(*[all_ratios[c] for c in UNIQUE_GROUPS])
    log(f"\n  Kruskal-Wallis 检验: H={stat:.4f}, p={p_value:.4f}")
    if p_value < 0.05:
        log("  结论: 各组ratio存在显著差异 (p < 0.05)")
    else:
        log("  结论: 各组ratio无显著差异 (p >= 0.05)")
    return results

def check3_mean_idf(ccomp_groups):
    """Check 3: Mean IDF"""
    log("\n" + "=" * 80)
    log("Check 3: Mean IDF")
    log("=" * 80)

    # 加载商品文本
    log("  计算IDF...")
    STOPWORDS = set(['i', 'me', 'my', 'we', 'our', 'you', 'the', 'a', 'an', 'and', 'but', 'if', 'or',
                     'of', 'at', 'by', 'for', 'with', 'is', 'are', 'was', 'were', 'be', 'been',
                     'do', 'does', 'did', 'will', 'would', 'that', 'which', 'who', 'this', 'these',
                     'it', 'its', 'to', 'in', 'on', 'as', 'from'])

    word_doc_freq = Counter()
    total_docs = 0
    with gzip.open(META_FILE, 'rt', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            title = item.get('title', '') or ''
            words = set(title.lower().split())
            for w in words:
                if w not in STOPWORDS:
                    word_doc_freq[w] += 1
            total_docs += 1

    log(f"  语料库文档数: {total_docs}, 词汇量: {len(word_doc_freq)}")

    def compute_query_idf(query):
        words = query.lower().split()
        non_stop = [w for w in words if w not in STOPWORDS]
        if not non_stop:
            return 0.0
        idf_sum = 0.0
        for w in non_stop:
            df = word_doc_freq.get(w, 0)
            idf_sum += np.log(total_docs / df) if df > 0 else np.log(total_docs + 1)
        return idf_sum / len(non_stop)

    results = {}
    idf_groups = {c: [] for c in UNIQUE_GROUPS}

    for ccomp in UNIQUE_GROUPS:
        idfs = [compute_query_idf(q['query']) for q in ccomp_groups[ccomp]]
        idf_groups[ccomp] = idfs
        results[ccomp] = {
            'count': len(idfs),
            'mean_idf': np.mean(idfs),
            'std_idf': np.std(idfs),
            'median_idf': np.median(idfs)
        }

    header = f"{GROUP_FIELD.upper():<10} {'N':<8} {'Mean IDF':<12} {'Std':<10} {'Median':<10}"
    log(header)
    log("-" * 80)
    for ccomp in UNIQUE_GROUPS:
        r = results[ccomp]
        log(f"{GROUP_FIELD.upper()}{ccomp}      {r['count']:<8} {r['mean_idf']:<12.4f} {r['std_idf']:<10.4f} {r['median_idf']:<10.4f}")

    # Mann-Whitney U test
    for c1, c2 in [(0, 2), (0, 3), (1, 2), (1, 3)]:
        stat, p = stats.mannwhitneyu(idf_groups[c1], idf_groups[c2], alternative='two-sided')
        sig = "**" if p < 0.05 else ""
        log(f"\n  {GROUP_FIELD.upper()}{c1} vs {GROUP_FIELD.upper()}{c2}: U={stat:.1f}, p={p:.4f} {sig}")

    if any(stats.mannwhitneyu(idf_groups[0], idf_groups[c], alternative='two-sided')[1] < 0.05 for c in [1, 2, 3]):
        log("  结论: 各组IDF存在显著差异 (p < 0.05) ⚠️ IDF可能是混淆因素!")
    else:
        log("  结论: 各组IDF无显著差异 (p >= 0.05)")
    return results

def check4_oracle_random(ccomp_groups):
    """Check 4: Oracle-aware Random Retriever"""
    log("\n" + "=" * 80)
    log("Check 4: Oracle-aware Random Retriever")
    log("=" * 80)

    # 加载商品类目 (使用 'category' 字段，可能是list或string)
    cat_to_asins = defaultdict(list)
    asin_to_cat = {}
    with gzip.open(META_FILE, 'rt', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            asin = item.get('asin', '')
            cat_raw = item.get('category', '')
            # category可能是list或string
            if isinstance(cat_raw, list):
                cat = cat_raw[-1] if cat_raw else 'unknown'
            elif isinstance(cat_raw, str):
                cat = cat_raw
            else:
                cat = 'unknown'
            if cat and cat != 'unknown':
                cat_to_asins[cat].append(asin)
                asin_to_cat[asin] = cat

    log(f"  共有 {len(cat_to_asins)} 个类目")

    results = {}
    hit_groups = {c: [] for c in UNIQUE_GROUPS}
    n_trials = 10

    for ccomp in UNIQUE_GROUPS:
        for q in ccomp_groups[ccomp]:
            asin = q['asin']
            if asin not in asin_to_cat:
                continue
            cat = asin_to_cat[asin]
            same_cat = cat_to_asins.get(cat, [])
            if len(same_cat) < 2:
                hit_groups[ccomp].append(0.0)
                continue
            hits = 0
            for _ in range(n_trials):
                sampled = np.random.choice(same_cat, size=min(10, len(same_cat)), replace=False)
                hits += 1 if asin in sampled else 0
            hit_groups[ccomp].append(hits / n_trials)

        if hit_groups[ccomp]:
            results[ccomp] = {
                'count': len(hit_groups[ccomp]),
                'mean_hit': np.mean(hit_groups[ccomp]),
                'std_hit': np.std(hit_groups[ccomp])
            }

    header = f"{GROUP_FIELD.upper():<10} {'N':<8} {'Hit@10':<12} {'Std':<10}"
    log(header)
    log("-" * 80)
    for ccomp in UNIQUE_GROUPS:
        r = results.get(ccomp, {})
        if r:
            log(f"{GROUP_FIELD.upper()}{ccomp}      {r['count']:<8} {r['mean_hit']:<12.4f} {r['std_hit']:<10.4f}")

    # Kruskal-Wallis
    stat, p_value = stats.kruskal(*[hit_groups[c] for c in UNIQUE_GROUPS])
    log(f"\n  Kruskal-Wallis 检验: H={stat:.4f}, p={p_value:.4f}")
    if p_value < 0.05:
        log("  结论: 各组oracle random hit存在显著差异 (p < 0.05) ⚠️ ground truth产品可能天生更容易被找到!")
    else:
        log("  结论: 各组oracle random hit无显著差异 (p >= 0.05)")
    return results

def run_confound_analysis():
    """运行所有混淆因素分析"""
    log("\n" + "=" * 80)
    log(f"{GROUP_FIELD.upper()} 混淆因素分析")
    log("=" * 80)

    ccomp_groups, _, _ = load_raw_queries()
    log(f"加载了 {sum(len(g) for g in ccomp_groups.values())} 个查询")

    check1_query_length(ccomp_groups)
    check2_pos_ratio(ccomp_groups)
    check3_mean_idf(ccomp_groups)
    check4_oracle_random(ccomp_groups)

    log("\n" + "=" * 80)
    log("混淆因素分析完成")
    log("=" * 80)

# ============ 评估指标计算 ============
def compute_metrics(relevant_asin: str, retrieved_asins: List[str], k_values: List[int]) -> Dict:
    metrics = {}
    for k in k_values:
        top_k = retrieved_asins[:k]
        metrics[f'P@{k}'] = 1.0 if relevant_asin in top_k else 0.0
        if relevant_asin in top_k:
            rank = top_k.index(relevant_asin) + 1
            metrics[f'N@{k}'] = 1.0 / np.log2(rank + 1)
            metrics[f'MR@{k}'] = 1.0 / rank
        else:
            metrics[f'N@{k}'] = 0.0
            metrics[f'MR@{k}'] = 0.0
        metrics[f'H@{k}'] = 1.0 if relevant_asin in top_k else 0.0
    return metrics

def compute_average_metrics(all_metrics: List[Dict], k_values: List[int]) -> Dict:
    avg_metrics = {}
    for k in k_values:
        avg_metrics[f'P@{k}'] = np.mean([m.get(f'P@{k}', 0.0) for m in all_metrics])
        avg_metrics[f'N@{k}'] = np.mean([m.get(f'N@{k}', 0.0) for m in all_metrics])
        avg_metrics[f'MR@{k}'] = np.mean([m.get(f'MR@{k}', 0.0) for m in all_metrics])
        avg_metrics[f'H@{k}'] = np.mean([m.get(f'H@{k}', 0.0) for m in all_metrics])
    return avg_metrics

# ============ Bootstrap CI ============
def compute_bootstrap_ci(all_metrics: List[Dict], k_values: List[int], n_bootstrap: int = 1000, ci: float = 0.95) -> Dict:
    """计算Bootstrap置信区间"""
    np.random.seed(42)
    n_samples = len(all_metrics)
    if n_samples < 2:
        return {}

    alpha = 1 - ci
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    metric_keys = [f'P@{k}' for k in k_values] + [f'N@{k}' for k in k_values] + [f'MR@{k}' for k in k_values] + [f'H@{k}' for k in k_values]

    bootstrap_results = {}
    for key in metric_keys:
        values = np.array([m.get(key, 0.0) for m in all_metrics])
        bootstrapped_means = []
        for _ in range(n_bootstrap):
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_values = values[sample_indices]
            bootstrapped_means.append(np.mean(sample_values))
        bootstrapped_means = np.array(bootstrapped_means)
        bootstrap_results[key] = {
            'mean': np.mean(bootstrapped_means),
            'std': np.std(bootstrapped_means),
            'ci_lower': np.percentile(bootstrapped_means, lower_percentile),
            'ci_upper': np.percentile(bootstrapped_means, upper_percentile),
        }
    return bootstrap_results

def print_bootstrap_ci_table(all_results: List[Dict], k_values: List[int]):
    log("\n" + "=" * 100)
    log("Bootstrap CI (95%) - P@10")
    log("=" * 100)

    header = f"{'检索器':<12} {GROUP_FIELD.upper():<10} {'Mean':<10} {'Std':<10} {'CI Lower':<12} {'CI Upper':<12}"
    log(header)
    log("-" * 100)

    for r in all_results:
        retriever = r['retriever']
        for ccomp in UNIQUE_GROUPS:
            ci = r['bootstrap_ci'].get(ccomp, {}).get('P@10', {})
            if ci:
                row = f"{retriever:<12} {GROUP_FIELD.upper()}{ccomp}   {ci['mean']:.4f}     {ci['std']:.4f}     {ci['ci_lower']:.4f}      {ci['ci_upper']:.4f}"
                log(row)

    log("-" * 100)
    # 总体 CI
    for r in all_results:
        ci = r['bootstrap_ci'].get('overall', {}).get('P@10', {})
        if ci:
            log(f"{r['retriever']:<12} overall   {ci['mean']:.4f}     {ci['std']:.4f}     {ci['ci_lower']:.4f}      {ci['ci_upper']:.4f}")

# ============ 数据加载 ============
def load_dense_retriever(retriever_name: str) -> Tuple[np.ndarray, List[str], int]:
    embeddings_path = None
    for f in os.listdir(CACHE_DIR):
        if f.startswith(f'{retriever_name}_') and f.endswith('_embeddings.npy'):
            embeddings_path = os.path.join(CACHE_DIR, f)
            break
    if embeddings_path is None:
        raise FileNotFoundError(f"{retriever_name} embeddings not found")

    log(f"  [{retriever_name}] 加载: {os.path.getsize(embeddings_path)/1024/1024:.1f} MB")
    mmap_array = np.load(embeddings_path, mmap_mode='r')
    embeddings = mmap_array[:].copy()

    doc_ids_path = embeddings_path.replace('_embeddings.npy', '_doc_ids.pkl')
    with open(doc_ids_path, 'rb') as f:
        doc_ids = pickle.load(f)

    return embeddings, doc_ids, embeddings.shape[1]

def load_bm25_retriever():
    """加载 BM25 检索器"""
    bm25_path = None
    for f in os.listdir(CACHE_DIR):
        if f.startswith('bm25_') and f.endswith('.pkl'):
            bm25_path = os.path.join(CACHE_DIR, f)
            break
    if bm25_path is None:
        raise FileNotFoundError("BM25 cache not found")

    log(f"  [bm25] 加载: {os.path.getsize(bm25_path)/1024/1024:.1f} MB")
    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)
    return bm25

def load_query_cache(retriever_name: str, query_type: str = 'correct') -> Dict:
    """加载查询缓存

    Args:
        retriever_name: 检索器名称
        query_type: 查询类型 ('correct' 或 'noisy')
    """
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'persona_{query_type}_query',
        f'{retriever_name}__persona_{query_type}_cache.pkl'
    )
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def load_user_queries(query_type: str = 'correct') -> Tuple[Dict[str, List[Dict]], Dict[str, int], List[Tuple[int, float, float]]]:
    """加载用户查询，每个查询项包含word_count和ccomp_ratio（POS ratio代理）

    Args:
        query_type: 查询类型 ('correct' 使用 filled_query, 'noisy' 使用 noisy_query)
    """
    global GROUP_FIELD, UNIQUE_GROUPS

    with open(QUERIES_FILE, 'r') as f:
        data = json.load(f)
    user_queries = {}
    user_to_ccomp = {}
    all_query_metadata = []  # (user_idx, word_count, ccomp_ratio)
    idx = 0

    # 检测字段名：ccomp 或 acl
    sample = data[0] if data else {}
    if 'queries' in sample:
        first_query = sample.get('queries', [{}])[0]
        GROUP_FIELD = 'acl' if 'acl' in first_query else 'ccomp'
    else:
        GROUP_FIELD = 'acl' if 'acl' in sample else 'ccomp'

    # 动态收集所有唯一的 group 值
    group_values = set()
    for item in data:
        if 'queries' in item:
            for q in item['queries']:
                gv = q.get(GROUP_FIELD, 0)
                group_values.add(gv)
        else:
            gv = item.get(f'target_{GROUP_FIELD}', item.get(GROUP_FIELD, 0))
            group_values.add(gv)
    UNIQUE_GROUPS = sorted(group_values)

    # 根据 query_type 确定查询文本字段
    # correct: correct_query (ground truth) / filled_query / generated_query / query
    # noisy: noisy_query
    def get_query_text(q):
        if query_type == 'noisy':
            return q.get('noisy_query', '') or q.get('query', '')
        else:
            # 对于 ground truth (acl=0), 使用 correct_query
            # 对于其他版本, 使用 filled_query
            if q.get('is_ground_truth', False):
                return q.get('correct_query', '') or q.get('filled_query', '') or q.get('query', '')
            return q.get('filled_query', '') or q.get('generated_query', '') or q.get('query', '')

    # 支持两种格式：
    # 1. 新嵌套格式：[{"user_id": ..., "asin": ..., "queries": [{filled_query, ccomp/acl, word_count}, ...]}]
    # 2. 旧平铺格式：[{"user_id": ..., "filled_query": ..., "target_ccomp/acl": ...}]
    items = data if isinstance(data, list) else []

    for item in items:
        # 新嵌套格式
        if 'queries' in item:
            user_id = item.get('user_id', '')
            asin = item.get('asin', '')
            for q in item['queries']:
                query_text = get_query_text(q)
                gv = q.get(GROUP_FIELD, 0)
                word_count = q.get('word_count', 0)
                if user_id not in user_queries:
                    user_queries[user_id] = []
                    user_to_ccomp[user_id] = gv
                if query_text and asin:
                    user_queries[user_id].append({
                        'query': query_text,
                        'asin': asin,
                        'word_count': word_count,
                        'ccomp_ratio': 0.0,
                        'ccomp': gv
                    })
                    all_query_metadata.append((idx, word_count, 0.0))
                    idx += 1
        else:
            # 旧平铺格式
            user_id = item.get('user_id')
            if not user_id:
                continue
            query_text = get_query_text(item)
            asin = item.get('asin', '')
            gv = item.get(f'target_{GROUP_FIELD}', item.get(GROUP_FIELD, 0))
            word_count = item.get('word_count') or 0
            ccomp_ratio = item.get('persona', {}).get(f'{GROUP_FIELD}_sentence_ratio', 0.0)
            if user_id not in user_queries:
                user_queries[user_id] = []
                user_to_ccomp[user_id] = gv
            if query_text and asin:
                user_queries[user_id].append({
                    'query': query_text,
                    'asin': asin,
                    'word_count': word_count,
                    'ccomp_ratio': ccomp_ratio,
                    'ccomp': gv
                })
                all_query_metadata.append((idx, word_count, ccomp_ratio))
                idx += 1
    return user_queries, user_to_ccomp, all_query_metadata

def build_word_idf_dict(meta_file: str, sample_size: int = 50000) -> Dict[str, float]:
    """从商品元数据语料库构建词的IDF字典（采样版本加速）"""
    import gzip
    word_doc_freq = defaultdict(int)
    total_sampled = 0

    log(f"Building word IDF from corpus (sampling {sample_size} docs)...")
    with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                item = json.loads(line)
                # 从 title + brand + description 提取词
                text = ' '.join(filter(None, [
                    item.get('title', ''),
                    item.get('brand', ''),
                    ' '.join(item.get('description', []))
                ])).lower()
                words = set(text.split())
                for w in words:
                    if len(w) > 1:  # 过滤单字符
                        word_doc_freq[w] += 1
                total_sampled += 1
            except Exception:
                continue

    N = total_sampled
    word_idf = {}
    for w, df in word_doc_freq.items():
        word_idf[w] = np.log(N / (df + 1))  # +1 平滑

    # 也计算字符ngram（用于商品品牌等专有名词）
    for w, df in word_doc_freq.items():
        if len(w) >= 4 and df < 10:
            # 罕见词给高IDF
            word_idf[w] = max(word_idf.get(w, 0), np.log(N / 10))

    log(f"  IDF vocabulary: {len(word_idf)} words, {total_sampled} docs sampled")
    return word_idf


def compute_query_idf(query_text: str, word_idf: Dict[str, float]) -> float:
    """计算查询的平均IDF（使用预计算的词IDF）"""
    words = query_text.lower().split()
    if not words:
        return 0.0
    idf_values = [word_idf.get(w, 5.0) for w in words]  # 未知词给中等IDF=5
    return np.mean(idf_values)


def compute_idf(queries: List[str], doc_count: int) -> float:
    """计算查询的平均IDF（简化版：使用词频）"""
    from collections import Counter
    words = ' '.join(queries).lower().split()
    word_freq = Counter(words)
    if not word_freq:
        return 0.0
    # 平均 IDF = log(N / df)，这里简化为平均词频的倒数
    avg_df = sum(word_freq.values()) / len(word_freq) if word_freq else 1
    return np.log(doc_count / avg_df + 1)

def compute_oracle_random_baseline(relevant_asin: str, doc_ids: List[str], n_trials: int = 100, seed: int = 42) -> Dict:
    """计算oracle-aware随机基线：给定相关文档在随机位置时的期望性能"""
    np.random.seed(seed)
    n_docs = len(doc_ids)
    if relevant_asin not in doc_ids:
        return {'P@10': 0.0, 'N@10': 0.0}
    rel_idx = doc_ids.index(relevant_asin)
    p10_list = []
    n10_list = []
    for _ in range(n_trials):
        # 随机打乱位置（保留relevant_asin在某个随机位置）
        random_pos = np.random.randint(0, n_docs)
        top10_positions = list(range(random_pos, min(random_pos + 10, n_docs)))
        if rel_idx in top10_positions:
            rank = top10_positions.index(rel_idx) + 1
            p10_list.append(1.0)
            n10_list.append(1.0 / np.log2(rank + 1))
        else:
            p10_list.append(0.0)
            n10_list.append(0.0)
    return {'P@10': np.mean(p10_list), 'N@10': np.mean(n10_list)}

# ============ 搜索器 ============
class DenseSearcher:
    """密集检索器搜索器 (GPU 矩阵乘法 + 余弦相似度)"""
    def __init__(self, embeddings: np.ndarray, doc_ids: List[str], retriever_name: str):
        self.doc_ids = doc_ids
        self.retriever_name = retriever_name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # 归一化 doc embeddings 以支持余弦相似度
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        normalized_embeddings = embeddings / norms
        self.embeddings_tensor = torch.from_numpy(normalized_embeddings).float().to(self.device)

    def search_batch(self, query_embeddings: List[np.ndarray], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        if not query_embeddings:
            return []
        query_tensor = torch.from_numpy(np.array(query_embeddings)).float().to(self.device)
        # 归一化 query embeddings
        q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1, q_norms)
        query_tensor = query_tensor / torch.from_numpy(q_norms).float().to(self.device)
        # 余弦相似度 = 归一化点积
        scores = torch.mm(query_tensor, self.embeddings_tensor.T)
        results = []
        for i in range(len(query_embeddings)):
            top_scores, top_indices = torch.topk(scores[i], min(top_k, len(self.doc_ids)))
            results.append([(self.doc_ids[idx.item()], top_scores[j].item()) for j, idx in enumerate(top_indices)])
        return results

class BM25Searcher:
    """BM25 搜索器 (文本搜索)"""
    def __init__(self, bm25_retriever):
        self.bm25 = bm25_retriever

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        results = []
        for query in queries:
            # BM25 search returns [(asin, score), ...]
            search_results = self.bm25.search(query, top_k=top_k)
            results.append(search_results)
        return results

# ============ 评估 ============
def evaluate_dense_retriever(retriever_name: str, user_queries: Dict, user_to_ccomp: Dict, k_values: List[int], word_idf: Dict[str, float] = None, query_type: str = 'correct') -> Dict:
    log(f"\n{'='*60}")
    log(f"检索器: {retriever_name.upper()} (密集) - {query_type.upper()}")
    log(f"{'='*60}")

    embeddings, doc_ids, dim = load_dense_retriever(retriever_name)
    query_cache = load_query_cache(retriever_name, query_type)
    searcher = DenseSearcher(embeddings, doc_ids, retriever_name)

    matched_users = [uid for uid in user_queries.keys() if uid in query_cache]
    log(f"  匹配用户: {len(matched_users)}")

    ccomp_groups = {g: [] for g in UNIQUE_GROUPS}
    all_metrics = []
    eval_start = time.time()

    # 分组统计: word_count bins 和 ccomp_ratio bins
    word_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, float('inf'))]
    word_bin_labels = ['很短(1-15)', '短(15-20)', '中(20-25)', '长(25-30)', '很长(30+)']
    ratio_bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
    ratio_bin_labels = ['很低(0-0.05)', '低(0.05-0.1)', '中(0.1-0.2)', '高(0.2-0.5)', '很高(0.5+)']

    word_count_groups = {label: [] for label in word_bin_labels}
    ccomp_ratio_groups = {label: [] for label in ratio_bin_labels}

    # IDF 分层分组
    idf_bin_groups = {label: [] for label in IDF_BIN_LABELS}
    # IDF × ccomp 交叉分组: {(idf_label, ccomp): [metrics]}
    idf_ccomp_cross = defaultdict(list)
    # 收集所有 query 原始数据用于 OLS 回归
    all_query_records = []

    for user_idx, user_id in enumerate(matched_users):
        queries = user_queries[user_id]
        cached_queries = query_cache[user_id]
        ccomp = user_to_ccomp.get(user_id, 0)

        query_embeddings = []
        query_asins = []
        query_texts = []
        query_word_counts = []
        query_ccomp_ratios = []
        query_idf_values = []
        query_ccomps = []  # 每条查询自己的ccomp值

        for q in queries:
            query_text = q['query']
            relevant_asin = q['asin']
            word_count = q.get('word_count', 0)
            ccomp_ratio = q.get('ccomp_ratio', 0.0)
            q_ccomp = q.get('ccomp', 0)  # 每条查询自己的ccomp值
            if query_text in cached_queries:
                query_embeddings.append(cached_queries[query_text])
                query_asins.append(relevant_asin)
                query_texts.append(query_text)
                query_word_counts.append(word_count)
                query_ccomp_ratios.append(ccomp_ratio)
                query_ccomps.append(q_ccomp)
                # 计算 IDF
                q_idf = compute_query_idf(query_text, word_idf) if word_idf else 0.0
                query_idf_values.append(q_idf)

        if not query_embeddings:
            continue

        results = searcher.search_batch(query_embeddings, top_k=max(k_values))

        for i, (retrieved, relevant_asin) in enumerate(zip(results, query_asins)):
            retrieved_asins = [r[0] for r in retrieved]
            metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
            ccomp = query_ccomps[i]  # 使用该查询自己的ccomp值
            all_metrics.append(metrics)
            ccomp_groups[ccomp].append(metrics)

            # 记录每条 query 的原始数据（用于 OLS 回归和 Paired Difference 分析）
            all_query_records.append({
                'user_id': user_id,
                'asin': relevant_asin,
                GROUP_FIELD: ccomp,
                'mean_idf': query_idf_values[i],
                'query_length': query_word_counts[i],
                f'{GROUP_FIELD}_ratio': query_ccomp_ratios[i],
                'p_at1': float(metrics.get('P@1', 0.0)),
                'p_at3': float(metrics.get('P@3', 0.0)),
                'p_at5': float(metrics.get('P@5', 0.0)),
                'p_at10': float(metrics.get('P@10', 0.0)),
                'n_at10': float(metrics.get('N@10', 0.0)),
                'mrr_at10': float(metrics.get('MR@10', 0.0)),
                'hit_at10': float(metrics.get('H@10', 0.0)),
            })

            # word_count 分组
            wc = query_word_counts[i]
            for (low, high), label in zip(word_bins, word_bin_labels):
                if low <= wc < high:
                    word_count_groups[label].append(metrics)
                    break

            # ccomp_ratio (POS proxy) 分组
            cr = query_ccomp_ratios[i]
            for (low, high), label in zip(ratio_bins, ratio_bin_labels):
                if low <= cr < high:
                    ccomp_ratio_groups[label].append(metrics)
                    break

            # IDF 分组
            q_idf = query_idf_values[i]
            for (low, high), label in zip(IDF_BINS, IDF_BIN_LABELS):
                if low <= q_idf < high:
                    idf_bin_groups[label].append(metrics)
                    idf_ccomp_cross[(label, ccomp)].append(metrics)
                    break

        if (user_idx + 1) % 100 == 0:
            elapsed = time.time() - eval_start
            log(f"    进度: {user_idx+1}/{len(matched_users)} ({100*(user_idx+1)/len(matched_users):.1f}%)")

    eval_time = time.time() - eval_start
    overall_metrics = compute_average_metrics(all_metrics, k_values)

    ccomp_metrics = {}
    ccomp_counts = {}
    for ccomp in UNIQUE_GROUPS:
        if ccomp_groups[ccomp]:
            ccomp_metrics[ccomp] = compute_average_metrics(ccomp_groups[ccomp], k_values)
            ccomp_counts[ccomp] = len(ccomp_groups[ccomp])
        else:
            ccomp_metrics[ccomp] = {k: 0.0 for k in [f'P@{i}' for i in k_values] + [f'N@{i}' for i in k_values] + [f'MR@{i}' for i in k_values] + [f'H@{i}' for i in k_values]}
            ccomp_counts[ccomp] = 0

    # 计算 word_count 分组统计
    word_count_analysis = {}
    for label in word_bin_labels:
        if word_count_groups[label]:
            word_count_analysis[label] = {
                'count': len(word_count_groups[label]),
                'metrics': compute_average_metrics(word_count_groups[label], k_values)
            }

    # 计算 ccomp_ratio 分组统计
    ccomp_ratio_analysis = {}
    for label in ratio_bin_labels:
        if ccomp_ratio_groups[label]:
            ccomp_ratio_analysis[label] = {
                'count': len(ccomp_ratio_groups[label]),
                'metrics': compute_average_metrics(ccomp_ratio_groups[label], k_values)
            }

    # 计算 IDF 分组统计
    idf_analysis = {}
    for label in IDF_BIN_LABELS:
        if idf_bin_groups[label]:
            idf_analysis[label] = {
                'count': len(idf_bin_groups[label]),
                'metrics': compute_average_metrics(idf_bin_groups[label], k_values)
            }

    # 计算 IDF × ccomp 交叉分组统计
    idf_ccomp_analysis = {}
    for (idf_label, ccomp_val), metrics_list in idf_ccomp_cross.items():
        if metrics_list:
            idf_ccomp_analysis[(idf_label, ccomp_val)] = {
                'count': len(metrics_list),
                'metrics': compute_average_metrics(metrics_list, k_values)
            }

    return {
        'retriever': retriever_name, 'dim': dim, 'type': 'dense', 'num_users': len(matched_users),
        'num_queries': len(all_metrics), 'eval_time_seconds': eval_time,
        'metrics': overall_metrics, 'ccomp_metrics': ccomp_metrics, 'ccomp_counts': ccomp_counts,
        'raw_metrics_per_ccomp': ccomp_groups, 'all_raw_metrics': all_metrics,
        'word_count_analysis': word_count_analysis,
        'ccomp_ratio_analysis': ccomp_ratio_analysis,
        'idf_analysis': idf_analysis,
        'idf_ccomp_cross': idf_ccomp_analysis,
        'all_query_records': all_query_records
    }

def evaluate_bm25_retriever(user_queries: Dict, user_to_ccomp: Dict, k_values: List[int], word_idf: Dict[str, float] = None, query_type: str = 'correct') -> Dict:
    log(f"\n{'='*60}")
    log(f"检索器: BM25 (稀疏) - {query_type.upper()}")
    log(f"{'='*60}")

    bm25 = load_bm25_retriever()
    searcher = BM25Searcher(bm25)

    matched_users = list(user_queries.keys())
    log(f"  用户数: {len(matched_users)}")

    eval_start = time.time()

    ccomp_groups = {g: [] for g in UNIQUE_GROUPS}
    all_metrics = []

    # 分组统计
    word_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, float('inf'))]
    word_bin_labels = ['很短(1-15)', '短(15-20)', '中(20-25)', '长(25-30)', '很长(30+)']
    ratio_bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
    ratio_bin_labels = ['很低(0-0.05)', '低(0.05-0.1)', '中(0.1-0.2)', '高(0.2-0.5)', '很高(0.5+)']
    word_count_groups = {label: [] for label in word_bin_labels}
    ccomp_ratio_groups = {label: [] for label in ratio_bin_labels}

    # IDF 分层分组
    idf_bin_groups = {label: [] for label in IDF_BIN_LABELS}
    idf_ccomp_cross = defaultdict(list)
    # 收集所有 query 原始数据用于 OLS 回归
    all_query_records = []

    for user_idx, user_id in enumerate(matched_users):
        queries = user_queries[user_id]

        query_texts = [q['query'] for q in queries]
        query_asins = [q['asin'] for q in queries]
        query_word_counts = [q.get('word_count', 0) for q in queries]
        query_ccomp_ratios = [q.get('ccomp_ratio', 0.0) for q in queries]
        query_ccomps = [q.get('ccomp', 0) for q in queries]  # 每条查询自己的ccomp值
        query_idf_values = [compute_query_idf(q['query'], word_idf) if word_idf else 0.0 for q in queries]

        if not query_texts:
            continue

        results = searcher.search_batch(query_texts, top_k=max(k_values))

        for i, (retrieved, relevant_asin) in enumerate(zip(results, query_asins)):
            retrieved_asins = [r[0] for r in retrieved]
            metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
            ccomp = query_ccomps[i]  # 使用该查询自己的ccomp值
            all_metrics.append(metrics)
            ccomp_groups[ccomp].append(metrics)

            # 记录每条 query 的原始数据（用于 OLS 回归和 Paired Difference 分析）
            all_query_records.append({
                'user_id': user_id,
                'asin': relevant_asin,
                GROUP_FIELD: ccomp,
                'mean_idf': query_idf_values[i],
                'query_length': query_word_counts[i],
                f'{GROUP_FIELD}_ratio': query_ccomp_ratios[i],
                'p_at1': float(metrics.get('P@1', 0.0)),
                'p_at3': float(metrics.get('P@3', 0.0)),
                'p_at5': float(metrics.get('P@5', 0.0)),
                'p_at10': float(metrics.get('P@10', 0.0)),
                'n_at10': float(metrics.get('N@10', 0.0)),
                'mrr_at10': float(metrics.get('MR@10', 0.0)),
                'hit_at10': float(metrics.get('H@10', 0.0)),
            })

            # word_count 分组
            wc = query_word_counts[i]
            for (low, high), label in zip(word_bins, word_bin_labels):
                if low <= wc < high:
                    word_count_groups[label].append(metrics)
                    break

            # ccomp_ratio 分组
            cr = query_ccomp_ratios[i]
            for (low, high), label in zip(ratio_bins, ratio_bin_labels):
                if low <= cr < high:
                    ccomp_ratio_groups[label].append(metrics)
                    break

            # IDF 分组
            q_idf = query_idf_values[i]
            for (low, high), label in zip(IDF_BINS, IDF_BIN_LABELS):
                if low <= q_idf < high:
                    idf_bin_groups[label].append(metrics)
                    idf_ccomp_cross[(label, ccomp)].append(metrics)
                    break

        if (user_idx + 1) % 100 == 0:
            elapsed = time.time() - eval_start
            log(f"    进度: {user_idx+1}/{len(matched_users)} ({100*(user_idx+1)/len(matched_users):.1f}%)")

    eval_time = time.time() - eval_start
    overall_metrics = compute_average_metrics(all_metrics, k_values)

    ccomp_metrics = {}
    ccomp_counts = {}
    for ccomp in UNIQUE_GROUPS:
        if ccomp_groups[ccomp]:
            ccomp_metrics[ccomp] = compute_average_metrics(ccomp_groups[ccomp], k_values)
            ccomp_counts[ccomp] = len(ccomp_groups[ccomp])
        else:
            ccomp_metrics[ccomp] = {k: 0.0 for k in [f'P@{i}' for i in k_values] + [f'N@{i}' for i in k_values] + [f'MR@{i}' for i in k_values] + [f'H@{i}' for i in k_values]}
            ccomp_counts[ccomp] = 0

    # 计算 word_count 分组统计
    word_count_analysis = {}
    for label in word_bin_labels:
        if word_count_groups[label]:
            word_count_analysis[label] = {
                'count': len(word_count_groups[label]),
                'metrics': compute_average_metrics(word_count_groups[label], k_values)
            }

    # 计算 ccomp_ratio 分组统计
    ccomp_ratio_analysis = {}
    for label in ratio_bin_labels:
        if ccomp_ratio_groups[label]:
            ccomp_ratio_analysis[label] = {
                'count': len(ccomp_ratio_groups[label]),
                'metrics': compute_average_metrics(ccomp_ratio_groups[label], k_values)
            }

    # 计算 IDF 分组统计
    idf_analysis = {}
    for label in IDF_BIN_LABELS:
        if idf_bin_groups[label]:
            idf_analysis[label] = {
                'count': len(idf_bin_groups[label]),
                'metrics': compute_average_metrics(idf_bin_groups[label], k_values)
            }

    # 计算 IDF × ccomp 交叉分组统计
    idf_ccomp_analysis = {}
    for (idf_label, ccomp_val), metrics_list in idf_ccomp_cross.items():
        if metrics_list:
            idf_ccomp_analysis[(idf_label, ccomp_val)] = {
                'count': len(metrics_list),
                'metrics': compute_average_metrics(metrics_list, k_values)
            }

    return {
        'retriever': 'bm25', 'dim': 0, 'type': 'sparse', 'num_users': len(matched_users),
        'num_queries': len(all_metrics), 'eval_time_seconds': eval_time,
        'metrics': overall_metrics, 'ccomp_metrics': ccomp_metrics, 'ccomp_counts': ccomp_counts,
        'raw_metrics_per_ccomp': ccomp_groups, 'all_raw_metrics': all_metrics,
        'word_count_analysis': word_count_analysis,
        'ccomp_ratio_analysis': ccomp_ratio_analysis,
        'idf_analysis': idf_analysis,
        'idf_ccomp_cross': idf_ccomp_analysis,
        'all_query_records': all_query_records
    }

# ============ 表格打印 ============
def print_overall_table(all_results: List[Dict], k_values: List[int]):
    header = f"{'检索器':<10} {'类型':<8} {'Dim':<5}"
    for k in k_values:
        header += f" {'P@'+str(k):<12}"
    for k in k_values:
        header += f" {'N@'+str(k):<12}"
    header += f" {'时间(s)':<10}"

    log("\n" + "=" * 100)
    log(f"总体评估结果{CATEGORY_NAME}")
    log("=" * 100)
    log(header)
    log("-" * 100)

    for r in all_results:
        m = r['metrics']
        row = f"{r['retriever']:<10} {r['type']:<8} {r['dim']:<5}"
        for k in k_values:
            row += f" {m[f'P@{k}']:.4f}     "
        for k in k_values:
            row += f" {m[f'N@{k}']:.4f}     "
        row += f" {r['eval_time_seconds']:.2f}"
        log(row)

def print_ccomp_table(all_results: List[Dict], k_values: List[int]):
    log("\n" + "=" * 120)
    log(f"{GROUP_FIELD.upper()} 分组评估结果{CATEGORY_NAME}")
    log("=" * 120)

    for ccomp in UNIQUE_GROUPS:
        log(f"\n--- {GROUP_FIELD.upper()}={ccomp} ---")
        header = f"{'检索器':<10} {'类型':<8}"
        for k in k_values:
            header += f" {'P@'+str(k):<12}"
        header += f" {'用户数':<8}"
        log(header)
        log("-" * 120)

        for r in all_results:
            m = r['ccomp_metrics'].get(ccomp, {})
            count = r['ccomp_counts'].get(ccomp, 0)
            row = f"{r['retriever']:<10} {r['type']:<8}"
            for k in k_values:
                row += f" {m.get(f'P@{k}', 0.0):.4f}     "
            row += f" {count:<8}"
            log(row)

def print_cross_tab(all_results: List[Dict], metric: str, query_type: str = ''):
    suffix = f" [{query_type.upper()} 版本]" if query_type else ""
    log(f"\n{'='*80}")
    log(f"交叉对比表{CATEGORY_NAME}: {metric}{suffix}")
    log(f"{'='*80}")

    header = f"{'检索器':<12}"
    for ccomp in UNIQUE_GROUPS:
        header += f" {GROUP_FIELD.upper()+str(ccomp):<12}"
    header += f" {'平均':<10}"
    log(header)
    log("-" * 80)

    totals = {0: 0, 1: 0, 2: 0, 3: 0, 'count': 0}
    for r in all_results:
        row = f"{r['retriever']:<12}"
        row_sum = 0
        row_count = 0
        for ccomp in UNIQUE_GROUPS:
            val = r['ccomp_metrics'].get(ccomp, {}).get(metric, 0.0)
            row += f" {val:.4f}     "
            row_sum += val
            totals[ccomp] += val
            row_count += 1
        row_avg = row_sum / row_count if row_count > 0 else 0.0
        totals['count'] += 1
        row += f" {row_avg:.4f}"
        log(row)

    if totals['count'] > 0:
        avg_row = f"{'平均':<12}"
        for ccomp in UNIQUE_GROUPS:
            avg_val = totals[ccomp] / totals['count']
            avg_row += f" {avg_val:.4f}     "
        log("-" * 80)
        log(avg_row)


def print_summary_table_wide(all_results: List[Dict], query_type: str = 'CORRECT'):
    """宽格式汇总表：每行一个检索器，列是 (指标 × 分组) 的所有组合

    格式：
    检索器 | P@1_ACL0 | P@1_ACL1 | ... | P@10_ACL3 | 平均
    """
    log(f"\n{'='*100}")
    log(f"汇总表（宽格式）| {query_type} | 每行一个检索器，列为 (指标×分组)")
    log(f"{'='*100}")

    metrics = ['P@1', 'P@3', 'P@5', 'P@10', 'N@10', 'MR@10', 'H@10']
    groups = [f"{GROUP_FIELD.upper()}{g}" for g in UNIQUE_GROUPS]

    # 构建表头 - 使用固定宽度列（每列12字符，右对齐）
    COL_W = 12
    header = f"{'检索器':<10}"
    for m in metrics:
        for g in groups:
            label = f"{m}_{g}"
            header += f" {label:>{COL_W}}"
        header += f" {m:>{COL_W}}"
    header += f" {'总平均':>{COL_W}}"
    log(header)
    log("-" * 100)

    # 构建数据
    retrievers = sorted(set(r['retriever'] for r in all_results))

    for retriever in retrievers:
        r = next((x for x in all_results if x['retriever'] == retriever), None)
        if not r:
            continue

        row = f"{retriever:<10}"
        all_vals = []

        for m in metrics:
            metric_vals = []
            for ccomp in UNIQUE_GROUPS:
                val = r['ccomp_metrics'].get(ccomp, {}).get(m, 0.0)
                row += f" {val:>{COL_W}.4f}"
                metric_vals.append(val)
                all_vals.append(val)
            metric_avg = sum(metric_vals) / len(metric_vals) if metric_vals else 0.0
            row += f" {metric_avg:>{COL_W}.4f}"
            all_vals.append(metric_avg)

        total_avg = sum(all_vals) / len(all_vals) if all_vals else 0.0
        row += f" {total_avg:>{COL_W}.4f}"
        log(row)

    log("-" * 100)


def print_summary_table_long(all_results: List[Dict], query_type: str = 'CORRECT'):
    """长格式汇总表：每行是 (检索器, 指标, 分组) 的一个单元格值

    格式：
    检索器 | 指标 | CCOMP0 | CCOMP1 | CCOMP2 | CCOMP3 | 平均
    """
    log(f"\n{'='*100}")
    log(f"汇总表（长格式）| {query_type} | 每行一个指标，列为各分组")
    log(f"{'='*100}")

    header = f"{'指标':<10}"
    for ccomp in UNIQUE_GROUPS:
        header += f" {GROUP_FIELD.upper()+str(ccomp):<12}"
    header += f" {'平均':<12}"
    log(header)
    log("-" * 80)

    metrics = ['P@1', 'P@3', 'P@5', 'P@10', 'N@10', 'MR@10', 'H@10']
    retrievers = sorted(set(r['retriever'] for r in all_results))

    for metric in metrics:
        row = f"{metric:<10}"
        metric_group_vals = []

        for ccomp in UNIQUE_GROUPS:
            group_vals = []
            for r in all_results:
                val = r['ccomp_metrics'].get(ccomp, {}).get(metric, 0.0)
                group_vals.append(val)
            group_avg = sum(group_vals) / len(group_vals) if group_vals else 0.0
            row += f" {group_avg:.4f}      "
            metric_group_vals.append(group_avg)

        overall_avg = sum(metric_group_vals) / len(metric_group_vals) if metric_group_vals else 0.0
        row += f" {overall_avg:.4f}      "
        log(row)

    # 添加每个检索器的详细数据
    log(f"\n{'='*100}")
    log(f"检索器详细数据 | {query_type}")
    log(f"{'='*100}")

    for retriever in retrievers:
        r = next((x for x in all_results if x['retriever'] == retriever), None)
        if not r:
            continue

        log(f"\n{retriever}:")
        sub_header = f"{'指标':<10}"
        for ccomp in UNIQUE_GROUPS:
            sub_header += f" {GROUP_FIELD.upper()+str(ccomp):<12}"
        sub_header += f" {'平均':<12}"
        log(sub_header)
        log("-" * 70)

        for metric in metrics:
            row = f"{metric:<10}"
            metric_vals = []
            for ccomp in UNIQUE_GROUPS:
                val = r['ccomp_metrics'].get(ccomp, {}).get(metric, 0.0)
                row += f" {val:.4f}      "
                metric_vals.append(val)
            metric_avg = sum(metric_vals) / len(metric_vals) if metric_vals else 0.0
            row += f" {metric_avg:.4f}      "
            log(row)


# ============ 分层分析：IDF × CCOMP ============
def run_ols_ccomp_analysis(all_results: List[Dict]):
    """OLS回归分析: 控制IDF、查询长度、词汇复杂度后，检验acl/ccomp效应是否独立"""
    log("\n" + "=" * 80)
    log(f"6C. OLS回归分析: {GROUP_FIELD.upper()}效应是否独立于IDF等混淆因素")
    log("=" * 80)
    log(f"目的: 在控制 mean_idf + query_length + {GROUP_FIELD}_ratio 后，检验 {GROUP_FIELD}_high 的净效应")
    log("")

    # 合并所有检索器的 query records
    all_records = []
    for r in all_results:
        retriever = r['retriever']
        for rec in r.get('all_query_records', []):
            rec_copy = rec.copy()
            rec_copy['retriever'] = retriever
            all_records.append(rec_copy)

    if not all_records:
        log("  (无 query records 数据，跳过 OLS 分析)")
        return

    df = pd.DataFrame(all_records)
    # 动态阈值：使用中位数分组
    threshold = UNIQUE_GROUPS[len(UNIQUE_GROUPS) // 2] if UNIQUE_GROUPS else 2
    df[f'{GROUP_FIELD}_high'] = (df[GROUP_FIELD] >= threshold).astype(int)
    log(f"  分组字段: {GROUP_FIELD}, 高低分界阈值: {threshold}")
    log(f"  有效分组: {UNIQUE_GROUPS}")

    log(f"  总样本数: {len(df)} queries")
    log(f"  {GROUP_FIELD}_high=1: {df[f'{GROUP_FIELD}_high'].sum()}, {GROUP_FIELD}_high=0: {(df[f'{GROUP_FIELD}_high']==0).sum()}")
    log(f"  检索器数: {df['retriever'].nunique()}")
    log("")

    # 分检索器运行 OLS，收集结果
    ols_table = []
    for retriever in sorted(df['retriever'].unique()):
        df_r = df[df['retriever'] == retriever].copy()
        formula = f'p_at10 ~ {GROUP_FIELD}_high + mean_idf + query_length + {GROUP_FIELD}_ratio'

        try:
            df_r_clean = df_r[['p_at10', f'{GROUP_FIELD}_high', 'mean_idf', 'query_length', f'{GROUP_FIELD}_ratio']].dropna()
            if len(df_r_clean) < 10:
                ols_table.append({'retriever': retriever, 'n': len(df_r_clean), 'r2': float('nan'),
                                 'coef': float('nan'), 't': float('nan'), 'p': float('nan'), 'sig': ''})
                continue

            model = smf.ols(formula, data=df_r_clean).fit()
            p_val = model.pvalues.get(f'{GROUP_FIELD}_high', float('nan'))
            coef = model.params.get(f'{GROUP_FIELD}_high', float('nan'))
            t_val = model.tvalues.get(f'{GROUP_FIELD}_high', float('nan'))
            sig = "***" if not np.isnan(p_val) and p_val < 0.001 else \
                  "**"  if not np.isnan(p_val) and p_val < 0.01  else \
                  "*"   if not np.isnan(p_val) and p_val < 0.05  else ""
            ols_table.append({
                'retriever': retriever,
                'n': len(df_r_clean),
                'r2': model.rsquared,
                'coef': coef,
                't': t_val,
                'p': p_val,
                'sig': sig
            })
        except Exception as e:
            ols_table.append({'retriever': retriever, 'n': 0, 'r2': float('nan'),
                             'coef': float('nan'), 't': float('nan'), 'p': float('nan'), 'sig': f'ERR:{e}'})

    # 汇总 OLS
    formula_pooled = f'p_at10 ~ {GROUP_FIELD}_high + mean_idf + query_length + {GROUP_FIELD}_ratio + C(retriever)'
    try:
        df_clean = df[['p_at10', f'{GROUP_FIELD}_high', 'mean_idf', 'query_length', f'{GROUP_FIELD}_ratio', 'retriever']].dropna()
        model_pooled = smf.ols(formula_pooled, data=df_clean).fit()
        p_val = model_pooled.pvalues.get(f'{GROUP_FIELD}_high', float('nan'))
        coef = model_pooled.params.get(f'{GROUP_FIELD}_high', float('nan'))
        t_val = model_pooled.tvalues.get(f'{GROUP_FIELD}_high', float('nan'))
        sig = "***" if not np.isnan(p_val) and p_val < 0.001 else \
              "**"  if not np.isnan(p_val) and p_val < 0.01  else \
              "*"   if not np.isnan(p_val) and p_val < 0.05  else ""
        ols_table.append({
            'retriever': 'Pooled',
            'n': len(df_clean),
            'r2': model_pooled.rsquared,
            'coef': coef,
            't': t_val,
            'p': p_val,
            'sig': sig
        })
    except Exception as e:
        ols_table.append({'retriever': 'Pooled', 'n': 0, 'r2': float('nan'),
                         'coef': float('nan'), 't': float('nan'), 'p': float('nan'), 'sig': f'ERR:{e}'})

    # 打印 OLS 表格
    log("")
    header = f"{'Retriever':<12} {'N':<6} {'R2':<8} {'Coef':<10} {'t':<8} {'P>|t|':<10} {'Sig':<5}"
    log(header)
    log("-" * 70)
    for row in ols_table:
        r2_str = f"{row['r2']:.4f}" if not np.isnan(row['r2']) else "NaN"
        coef_str = f"{row['coef']:.5f}" if not np.isnan(row['coef']) else "NaN"
        t_str = f"{row['t']:.3f}" if not np.isnan(row['t']) else "NaN"
        p_str = f"{row['p']:.4f}" if not np.isnan(row['p']) else "NaN"
        log(f"{row['retriever']:<12} {row['n']:<6} {r2_str:<8} {coef_str:<10} {t_str:<8} {p_str:<10} {row['sig']:<5}")
    log("")

# ============ Paired Difference 分析 ============
def run_paired_difference_analysis(all_results: List[Dict]):
    """Paired Difference 分析：比较两个极端版本（低 vs 高）的检索性能差异"""
    log("\n" + "=" * 80)
    log(f"{GROUP_FIELD.upper()} Paired Difference 分析 (极端版本: {GROUP_FIELD}=0 vs {GROUP_FIELD}=max)")
    log("=" * 80)

    # 收集所有 (user_id, retriever, group_value, hit10) 记录
    records = []
    for r in all_results:
        retriever = r['retriever']
        for rec in r.get('all_query_records', []):
            records.append({
                'user_id': rec.get('user_id', ''),
                'retriever': retriever,
                'group': rec.get(GROUP_FIELD, 0),
                'hit10': rec.get('hit_at10', 0.0),
            })

    if not records:
        log("  (无 query records 数据，跳过 Paired Difference 分析)")
        return

    df = pd.DataFrame(records)
    n_retrievers = df['retriever'].nunique()
    log(f"  总查询记录: {len(df)} ({n_retrievers}检索器 × {len(df)//n_retrievers}查询/检索器)")

    # 获取所有分组值
    all_groups = sorted(df['group'].unique())
    if len(all_groups) < 2:
        log("  分组不足，跳过 Paired Difference 分析")
        return

    group_low = all_groups[0]  # 最低版本
    group_high = all_groups[-1]  # 最高版本
    log(f"  最低版本: {GROUP_FIELD}={group_low}, 最高版本: {GROUP_FIELD}={group_high}")
    log("")

    # 构建 pivot 表：(user_id, retriever) × group_value = hit10
    df_filtered = df[df['group'].isin([group_low, group_high])]
    pivot = df_filtered.pivot_table(
        index=['user_id', 'retriever'],
        columns='group',
        values='hit10'
    ).reset_index()

    if pivot.empty:
        log("  (无法构建 pivot 表，跳过分析)")
        return

    # 确保有 user_id 列
    if 'user_id' not in pivot.columns:
        log("  (缺少 user_id 列，跳过分析)")
        return

    pivot.columns = ['user_id', 'retriever'] + [f'g{c}' for c in pivot.columns[2:]]
    pivot['diff'] = pivot[f'g{group_high}'] - pivot[f'g{group_low}']

    n_ret = len(pivot['retriever'].unique())
    pairs_per_ret = len(pivot) // n_ret
    log(f"  配对样本数: {len(pivot)} ({n_ret}检索器 × {pairs_per_ret}对/检索器，每对 = 同用户{group_high} vs {group_low})")

    # Paired bootstrap per retriever
    np.random.seed(42)
    n_bootstrap = 10000
    results = []

    for retriever in sorted(pivot['retriever'].unique()):
        df_r = pivot[pivot['retriever'] == retriever].copy()
        diffs = df_r['diff'].dropna().values
        n = len(diffs)

        if n < 2:
            results.append({
                'retriever': retriever, 'n': n,
                'mean_diff': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan'),
                't': float('nan'), 'p': float('nan'), 'sig': ''
            })
            continue

        # Paired t-test
        from scipy.stats import ttest_1samp
        t, p = ttest_1samp(diffs, 0)

        # Bootstrap CI
        boot_means = []
        for _ in range(n_bootstrap):
            resample = np.random.choice(diffs, size=n, replace=True)
            boot_means.append(resample.mean())
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        results.append({
            'retriever': retriever, 'n': n,
            'mean_diff': diffs.mean(), 'ci_low': ci_low, 'ci_high': ci_high,
            't': t, 'p': p, 'sig': sig
        })

    # 打印结果表格
    log("")
    header = f"{'Retriever':<12} {'N':<6} {'MeanDiff':<10} {'95% CI':<18} {'t':<8} {'P>|t|':<10} {'Sig':<5}"
    log(header)
    log("-" * 80)
    for row in results:
        mean_str = f"{row['mean_diff']:.4f}" if not np.isnan(row['mean_diff']) else "NaN"
        ci_str = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]" if not np.isnan(row['ci_low']) else "[NaN, NaN]"
        t_str = f"{row['t']:.3f}" if not np.isnan(row['t']) else "NaN"
        p_str = f"{row['p']:.4f}" if not np.isnan(row['p']) else "NaN"
        log(f"{row['retriever']:<12} {row['n']:<6} {mean_str:<10} {ci_str:<18} {t_str:<8} {p_str:<10} {row['sig']:<5}")
    log("")

    # 汇总：所有检索器合并
    all_diffs = pivot['diff'].dropna().values
    if len(all_diffs) >= 2:
        t_all, p_all = ttest_1samp(all_diffs, 0)
        boot_means = []
        for _ in range(n_bootstrap):
            resample = np.random.choice(all_diffs, size=len(all_diffs), replace=True)
            boot_means.append(resample.mean())
        ci_low_all, ci_high_all = np.percentile(boot_means, [2.5, 97.5])
        sig_all = "***" if p_all < 0.001 else "**" if p_all < 0.01 else "*" if p_all < 0.05 else ""
        log("-" * 80)
        mean_all_str = f"{all_diffs.mean():.4f}"
        ci_all_str = f"[{ci_low_all:.4f}, {ci_high_all:.4f}]"
        t_all_str = f"{t_all:.3f}"
        p_all_str = f"{p_all:.4f}"
        log(f"{'Pooled':<12} {len(all_diffs):<6} {mean_all_str:<10} {ci_all_str:<18} {t_all_str:<8} {p_all_str:<10} {sig_all:<5}")
        log("")

# ============ Paired OLS 回归：ACL/CCOMP 纯效应分析 ============
def run_ccomp_pure_effect_regression(all_results: List[Dict]):
    """Paired OLS 回归：控制长度差和 IDF 差后，分析 acl/ccomp 的纯效应

    对每个用户计算：
    - diff_p10 = acl/ccomp=max 的 P@10 - acl/ccomp=0 的 P@10
    - diff_length = acl/ccomp=max 的长度 - acl/ccomp=0 的长度
    - diff_idf = acl/ccomp=max 的 IDF - acl/ccomp=0 的 IDF

    然后运行 OLS: diff_p10 ~ diff_length + diff_idf
    - 截距 = 控制长度和 IDF 差异后的 acl/ccomp 纯效应
    - diff_length 系数 = 长度每多 1 词，P@10 变化多少
    - diff_idf 系数 = IDF 每多 1，P@10 变化多少
    """
    log("\n" + "=" * 80)
    log(f"{GROUP_FIELD.upper()} 纯效应回归分析 (控制 diff_length + diff_idf)")
    log("=" * 80)
    log("模型: diff_p10 ~ diff_length + diff_idf")
    log(f"  • 截距 = 控制长度差和IDF差后的 {GROUP_FIELD} 纯效应")
    log("  • diff_length 系数 = 长度每多1词，P@10 变化多少")
    log("  • diff_idf 系数 = IDF每多1，P@10 变化多少")
    log("")

    # 收集所有 (user_id, retriever, group, p_at10, query_length, mean_idf) 记录
    records = []
    for r in all_results:
        retriever = r['retriever']
        for rec in r.get('all_query_records', []):
            records.append({
                'user_id': rec.get('user_id', ''),
                'retriever': retriever,
                'group': rec.get(GROUP_FIELD, 0),
                'p_at10': rec.get('p_at10', 0.0),
                'query_length': rec.get('query_length', 0.0),
                'mean_idf': rec.get('mean_idf', 0.0),
            })

    if not records:
        log("  (无 query records 数据，跳过纯效应回归分析)")
        return

    df = pd.DataFrame(records)
    n_retrievers = df['retriever'].nunique()
    log(f"  总查询记录: {len(df)} ({n_retrievers} 检索器)")

    # 获取分组边界
    all_groups = sorted(df['group'].unique())
    if len(all_groups) < 2:
        log("  分组不足，跳过纯效应回归分析")
        return

    group_low = all_groups[0]   # ccomp=0
    group_high = all_groups[-1]  # ccomp=max
    log(f"  极端版本: {GROUP_FIELD}={group_low} (低) vs {GROUP_FIELD}={group_high} (高)")
    log("")

    # 构建配对数据：(user_id, retriever) 配对
    df_low = df[df['group'] == group_low][['user_id', 'retriever', 'p_at10', 'query_length', 'mean_idf']].copy()
    df_high = df[df['group'] == group_high][['user_id', 'retriever', 'p_at10', 'query_length', 'mean_idf']].copy()

    df_low.columns = ['user_id', 'retriever', 'p_at10_low', 'length_low', 'idf_low']
    df_high.columns = ['user_id', 'retriever', 'p_at10_high', 'length_high', 'idf_high']

    paired = pd.merge(df_low, df_high, on=['user_id', 'retriever'], how='inner')
    if paired.empty:
        log("  (无法配对，跳过纯效应回归分析)")
        return

    # 计算差值
    paired['diff_p10'] = paired['p_at10_high'] - paired['p_at10_low']
    paired['diff_length'] = paired['length_high'] - paired['length_low']
    paired['diff_idf'] = paired['idf_high'] - paired['idf_low']

    n_pairs = len(paired)
    n_ret = len(paired['retriever'].unique())
    pairs_per_ret = n_pairs // n_ret if n_ret > 0 else n_pairs
    log(f"  配对样本数: {n_pairs} ({n_ret} 检索器 × {pairs_per_ret} 对/检索器)")
    log(f"  diff_p10: mean={paired['diff_p10'].mean():.4f}, std={paired['diff_p10'].std():.4f}")
    log(f"  diff_length: mean={paired['diff_length'].mean():.2f}, std={paired['diff_length'].std():.2f}")
    log(f"  diff_idf: mean={paired['diff_idf'].mean():.4f}, std={paired['diff_idf'].std():.4f}")
    log("")

    # 分检索器运行 OLS
    ols_table = []
    formula = 'diff_p10 ~ diff_length + diff_idf'

    for retriever in sorted(paired['retriever'].unique()):
        df_r = paired[paired['retriever'] == retriever].copy()
        df_r_clean = df_r[['diff_p10', 'diff_length', 'diff_idf']].dropna()

        if len(df_r_clean) < 10:
            ols_table.append({
                'retriever': retriever, 'n': len(df_r_clean),
                'r2': float('nan'), 'intercept': float('nan'), 'intercept_p': float('nan'),
                'coef_length': float('nan'), 'coef_length_p': float('nan'),
                'coef_idf': float('nan'), 'coef_idf_p': float('nan'),
            })
            continue

        model = smf.ols(formula, data=df_r_clean).fit()

        ols_table.append({
            'retriever': retriever,
            'n': len(df_r_clean),
            'r2': model.rsquared,
            'intercept': model.params.get('Intercept', float('nan')),
            'intercept_p': model.pvalues.get('Intercept', float('nan')),
            'coef_length': model.params.get('diff_length', float('nan')),
            'coef_length_p': model.pvalues.get('diff_length', float('nan')),
            'coef_idf': model.params.get('diff_idf', float('nan')),
            'coef_idf_p': model.pvalues.get('diff_idf', float('nan')),
        })

    # 汇总：所有检索器合并
    df_all_clean = paired[['diff_p10', 'diff_length', 'diff_idf']].dropna()
    if len(df_all_clean) >= 10:
        model_pooled = smf.ols(formula, data=df_all_clean).fit()
        ols_table.append({
            'retriever': 'Pooled',
            'n': len(df_all_clean),
            'r2': model_pooled.rsquared,
            'intercept': model_pooled.params.get('Intercept', float('nan')),
            'intercept_p': model_pooled.pvalues.get('Intercept', float('nan')),
            'coef_length': model_pooled.params.get('diff_length', float('nan')),
            'coef_length_p': model_pooled.pvalues.get('diff_length', float('nan')),
            'coef_idf': model_pooled.params.get('diff_idf', float('nan')),
            'coef_idf_p': model_pooled.pvalues.get('diff_idf', float('nan')),
        })

    # 打印 OLS 表格
    header = f"{'Retriever':<10} {'N':<6} {'R2':<8} {'Intercept':<12} {'P>|t|':<10} {'Coef_len':<10} {'P>|t|':<10} {'Coef_idf':<10} {'P>|t|':<10}"
    log(header)
    log("-" * 110)
    for row in ols_table:
        r2_str = f"{row['r2']:.4f}" if not np.isnan(row['r2']) else "NaN"
        int_str = f"{row['intercept']:.5f}" if not np.isnan(row['intercept']) else "NaN"
        int_p_str = f"{row['intercept_p']:.4f}" if not np.isnan(row['intercept_p']) else "NaN"
        len_str = f"{row['coef_length']:.5f}" if not np.isnan(row['coef_length']) else "NaN"
        len_p_str = f"{row['coef_length_p']:.4f}" if not np.isnan(row['coef_length_p']) else "NaN"
        idf_str = f"{row['coef_idf']:.5f}" if not np.isnan(row['coef_idf']) else "NaN"
        idf_p_str = f"{row['coef_idf_p']:.4f}" if not np.isnan(row['coef_idf_p']) else "NaN"
        log(f"{row['retriever']:<10} {row['n']:<6} {r2_str:<8} {int_str:<12} {int_p_str:<10} {len_str:<10} {len_p_str:<10} {idf_str:<10} {idf_p_str:<10}")
    log("")
    log("  解读:")
    log(f"  • Intercept = 控制 diff_length 和 diff_idf 后，{GROUP_FIELD} 从低到高的 P@10 纯效应")
    log("  • Coef_len = 长度每多 1 词，diff_p10 变化多少")
    log("  • Coef_idf = IDF 每多 1，diff_p10 变化多少")
    log("")


# ============ 主流程 ============
def print_query_type_comparison(all_results_by_type: Dict[str, List[Dict]], k_values: List[int]):
    """打印 correct vs noisy 配对比较表（宽格式）

    对于每个 noisy 查询，找到其同 CCOMP 级别的 correct 查询进行配对
    配对 key: (user_id, asin, ccomp)
    输出宽格式表格，包含所有指标
    """
    log("\n" + "=" * 100)
    log(f"CORRECT vs NOISY 配对比较（宽格式 | 基于 (user_id, asin, {GROUP_FIELD.upper()}) 匹配）")
    log("=" * 100)

    # 获取所有检索器
    retrievers = set()
    for qt_results in all_results_by_type.values():
        for r in qt_results:
            retrievers.add(r['retriever'])
    retrievers = sorted(retrievers)

    # 定义所有指标及其对应的字段
    METRICS = [
        ('P@1', 'p_at1'),
        ('P@3', 'p_at3'),
        ('P@5', 'p_at5'),
        ('P@10', 'p_at10'),
        ('N@10', 'n_at10'),
        ('MR@10', 'mrr_at10'),
        ('H@10', 'hit_at10'),
    ]

    group_field = GROUP_FIELD  # 'ccomp' or 'acl'

    # 构建宽格式表头 - 每个指标3个子列，固定宽度
    # 格式: 检索器 | CORR | NOISY | DIFF | CORR | NOISY | DIFF | ...
    header = f"{'检索器':<10}"
    sep = " " * 1
    for metric_name, _ in METRICS:
        header += sep + f"{metric_name:>7} {metric_name:>7} {metric_name:>7}"
    log(header)
    log("-" * 100)

    # 对每个检索器计算所有指标的配对比较
    for retriever in retrievers:
        correct_results = next((x for x in all_results_by_type.get('correct', []) if x['retriever'] == retriever), None)
        noisy_results = next((x for x in all_results_by_type.get('noisy', []) if x['retriever'] == retriever), None)

        if not correct_results or not noisy_results:
            continue

        row = f"{retriever:<10}"

        for metric_name, metric_field in METRICS:
            correct_dict = {}
            noisy_dict = {}

            for rec in correct_results.get('all_query_records', []):
                key = (rec['user_id'], rec.get('asin', ''), rec.get(group_field, -1))
                correct_dict[key] = rec.get(metric_field, 0)

            for rec in noisy_results.get('all_query_records', []):
                key = (rec['user_id'], rec.get('asin', ''), rec.get(group_field, -1))
                noisy_dict[key] = rec.get(metric_field, 0)

            # 找到共同的查询对
            common_keys = set(correct_dict.keys()) & set(noisy_dict.keys())

            if not common_keys:
                row += sep + f"{'N/A':>7} {'N/A':>7} {'N/A':>7}"
                continue

            # 计算配对差异
            correct_vals = [correct_dict[k] for k in common_keys]
            noisy_vals = [noisy_dict[k] for k in common_keys]
            diffs = [noisy_vals[i] - correct_vals[i] for i in range(len(common_keys))]

            mean_correct = sum(correct_vals) / len(correct_vals)
            mean_noisy = sum(noisy_vals) / len(noisy_vals)
            mean_diff = sum(diffs) / len(diffs)

            row += sep + f"{mean_correct:7.4f} {mean_noisy:7.4f} {mean_diff:+7.4f}"
        log(row)

    log("-" * 100)


def main():
    log("=" * 60)
    log(f"快速全量评估 - 多检索器 + {GROUP_FIELD.upper()}分组 + 交叉对比")
    log("=" * 60)

    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    # 检查缓存完整性
    # if not validate_cache():
    #     log("缓存完整性检查失败，退出。")
    #     return

    log("\n加载用户数据...")
    # 先加载 correct 版本获取统计信息
    user_queries_correct, user_to_ccomp, _ = load_user_queries('correct')
    log(f"  用户数: {len(user_queries_correct)}")

    # 按每条查询的ccomp值计数，而非按用户
    ccomp_dist = defaultdict(int)
    for user_qs in user_queries_correct.values():
        for q in user_qs:
            ccomp_dist[q.get('ccomp', 0)] += 1
    log(f"  {GROUP_FIELD.upper()}分布: {dict(sorted(ccomp_dist.items()))}")

    # 计算查询长度分组统计
    all_word_counts = [q.get('word_count') or 0 for user_qs in user_queries_correct.values() for q in user_qs]
    q25, q50, q75 = np.percentile(all_word_counts, [25, 50, 75])
    log(f"  Query长度分布: min={min(all_word_counts)}, Q25={q25:.0f}, Q50={q50:.0f}, Q75={q75:.0f}, max={max(all_word_counts)}")

    # 构建词IDF字典（用于分层分析）
    log("\n构建词IDF字典（用于分层分析）...")
    word_idf = build_word_idf_dict(META_FILE, sample_size=50000)
    # 检查IDF分布
    sample_idfs = [compute_query_idf(q['query'], word_idf) for user_qs in user_queries_correct.values() for q in user_qs]
    if sample_idfs:
        log(f"  Query IDF分布: min={min(sample_idfs):.2f}, mean={np.mean(sample_idfs):.2f}, max={max(sample_idfs):.2f}")

    k_values = [1, 3, 5, 10]

    # 分别对 correct 和 noisy 进行评估
    all_results_by_type = {}

    for query_type in QUERY_TYPES:
        log("\n" + "#" * 80)
        log(f"# 查询类型: {query_type.upper()}")
        log("#" * 80)

        # 根据 query_type 加载对应的查询
        user_queries, user_to_ccomp, _ = load_user_queries(query_type)
        log(f"  用户数: {len(user_queries)}")

        query_type_results = []

        # 评估密集检索器
        for retriever_name in DENSE_RETRIEVERS:
            try:
                result = evaluate_dense_retriever(retriever_name, user_queries, user_to_ccomp, k_values, word_idf, query_type)
                result['query_type'] = query_type
                query_type_results.append(result)
            except FileNotFoundError as e:
                log(f"  跳过 {retriever_name}: {e}")
            except Exception as e:
                log(f"  错误 {retriever_name}: {e}")

        # 评估 BM25
        try:
            result = evaluate_bm25_retriever(user_queries, user_to_ccomp, k_values, word_idf, query_type)
            result['query_type'] = query_type
            query_type_results.append(result)
        except Exception as e:
            log(f"  BM25 错误: {e}")

        all_results_by_type[query_type] = query_type_results

    # 合并所有结果用于后续分析
    all_results = all_results_by_type.get('correct', []) + all_results_by_type.get('noisy', [])

    # 计算 Oracle-Aware Random Baseline
    log("\n计算 Oracle-Aware Random Baseline...")
    # 加载 doc_ids 用于随机基线
    embeddings_path = None
    for f in os.listdir(CACHE_DIR):
        if f.startswith('bge_') and f.endswith('_doc_ids.pkl'):
            embeddings_path = os.path.join(CACHE_DIR, f)
            break
    with open(embeddings_path, 'rb') as f:
        doc_ids = pickle.load(f)
    n_docs = len(doc_ids)
    log(f"  总文档数: {n_docs}")

    # ========== ACL 组间性能比较 (CORRECT 版本) ==========
    if all_results_by_type.get('correct'):
        # 打印宽格式汇总表
        print_summary_table_wide(all_results_by_type['correct'], 'CORRECT')

    # ========== ACL 组间性能比较 (NOISY 版本) ==========
    if all_results_by_type.get('noisy'):
        # 打印宽格式汇总表
        print_summary_table_wide(all_results_by_type['noisy'], 'NOISY')

    # ========== CORRECT vs NOISY 配对比较 ==========
    # 基于 (user_id, asin) 匹配，对每个 noisy 查询找到对应的 correct 查询进行配对对比
    if all_results_by_type.get('noisy') and all_results_by_type.get('correct'):
        print_query_type_comparison(all_results_by_type, k_values)

    # 计算并打印 Bootstrap CI
    log("\n计算 Bootstrap CI (n=1000, CI=95%)...")
    for r in all_results:
        r['bootstrap_ci'] = {}
        for ccomp in UNIQUE_GROUPS:
            if r['raw_metrics_per_ccomp'].get(ccomp):
                r['bootstrap_ci'][ccomp] = compute_bootstrap_ci(
                    r['raw_metrics_per_ccomp'][ccomp], k_values, n_bootstrap=1000, ci=0.95
                )
        if r.get('all_raw_metrics'):
            r['bootstrap_ci']['overall'] = compute_bootstrap_ci(
                r['all_raw_metrics'], k_values, n_bootstrap=1000, ci=0.95
            )

    # 打印 Oracle-Aware Random Baseline
    log("\n" + "=" * 80)
    log("Oracle-Aware Random Baseline")
    log("=" * 80)
    if all_results and all_results[0].get('all_raw_metrics'):
        log(f"  理论随机P@10 = 10/{n_docs} = {10/n_docs:.6f}")
        log(f"  理论随机N@10 ≈ {np.mean([1/np.log2(r+2) if r < 10 else 0 for r in range(n_docs)])*10:.6f}")

    # 6C. OLS回归分析：CCOMP效应是否独立于IDF等混淆因素 (使用 CORRECT 版本)
    if all_results_by_type.get('correct'):
        run_ols_ccomp_analysis(all_results_by_type['correct'])

    # 6D. Paired Difference 分析 (使用 CORRECT 版本)
    if all_results_by_type.get('correct'):
        run_paired_difference_analysis(all_results_by_type['correct'])

    # 6E. ACL/CCOMP 纯效应回归 (使用 CORRECT 版本)
    if all_results_by_type.get('correct'):
        run_ccomp_pure_effect_regression(all_results_by_type['correct'])

    # 运行 CCOMP 混淆因素分析 (Check 1-4 + Bootstrap CI)
    run_confound_analysis()

    # 保存结果（处理 tuple key 等不可 JSON 序列化的问题）
    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj

    output_file = os.path.join(OUTPUT_DIR, "retrieval_all_summary.json")
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'query_types': QUERY_TYPES,
            'results_by_type': sanitize_for_json(all_results_by_type),
            'retrievers': sanitize_for_json(all_results)
        }, f, indent=2, default=str)
    log(f"\n结果已保存到: {output_file}")

    log("\n" + "=" * 60)
    log("评估完成!")
    log("=" * 60)

if __name__ == "__main__":
    main()
