#!/usr/bin/env python3
"""ACL/CCOMP 句法分析（spaCy模型分析）- 合并版本

同时计算 ACL (Adjectival Clause) 和 CCOMP (Complement Clause) 特征

用法:
    python 05_syntactic_analysis.py [category]
    category 可选: Arts_Crafts_and_Sewing (默认), Grocery_and_Gourmet_Food, Pet_Supplies
"""
import json, numpy as np
import warnings
import time
import os
import sys
from datetime import datetime
from collections import Counter
warnings.filterwarnings('ignore')
import spacy
nlp = spacy.load('en_core_web_sm')

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

# ============ 路径配置 ============
CATEGORY = "Grocery_and_Gourmet_Food"
BASE_DIR = f'/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/{CATEGORY}'
os.makedirs(BASE_DIR, exist_ok=True)

ACL_OUTPUT_JSONL = f'{BASE_DIR}/acl_sentences.jsonl'
ACL_STATS_JSON = f'{BASE_DIR}/acl_stats.json'
ACL_USERS_JSON = f'{BASE_DIR}/acl_user_profiles.json'

CCOMP_OUTPUT_JSONL = f'{BASE_DIR}/ccomp_sentences.jsonl'
CCOMP_STATS_JSON = f'{BASE_DIR}/ccomp_stats.json'
CCOMP_USERS_JSON = f'{BASE_DIR}/ccomp_user_profiles.json'
CCOMP_USERS_JSON = f'{BASE_DIR}/ccomp_user_profiles.json'

ALL_USERS_FILE = f'/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/{CATEGORY}/stage1_filtered_users_reviews.json'
STAGE1_ATTR_FILE = f'/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/{CATEGORY}/attributes_{CATEGORY}.json'

# ============ 标记词 ============
ACL_MARKERS = {'that', 'whether', 'if', 'what', 'which', 'who', 'whom', 'whose', 'whatever', 'whichever', 'whoever'}
CCOMP_MARKERS = {'that', 'whether', 'if', 'what', 'which', 'who', 'whom', 'whose'}


# ============ ACL 分析 ============
def compute_acl_gaps(doc, acl_info):
    if len(acl_info) < 2:
        return []
    positions = [info['position'] for info in acl_info]
    gaps = []
    for i in range(len(positions) - 1):
        gap = positions[i + 1] - positions[i] - 1
        gaps.append(gap)
    return gaps


def analyze_acl_in_doc(doc):
    """分析文档中所有 ACL (Adjectival Clause)"""
    results = []
    for token in doc:
        acl_info = None

        # 1. spaCy dep_='acl'
        if token.dep_ == 'acl':
            marker_word = None
            marker_dep = None
            complementizer = None

            for child in token.children:
                if child.dep_ == 'mark':
                    marker_word = child.text.lower()
                    marker_dep = 'mark'
                    complementizer = child.text.lower()
                    break

            if marker_word is None:
                for child in token.children:
                    if child.dep_ == 'comp':
                        marker_word = child.text.lower()
                        marker_dep = 'comp'
                        complementizer = child.text.lower()
                        break

            head_noun = token.head.text if token.head else ''
            head_pos = token.head.pos_ if token.head else ''

            acl_info = {
                'acl_type': 'acl',
                'marker': marker_word,
                'complementizer': complementizer,
                'head_word': head_noun,
                'head_pos': head_pos,
                'verb_word': token.text,
                'verb_pos': token.pos_,
                'position': token.i
            }

        # 2. relcl 参考
        elif token.dep_ == 'relcl':
            rel_pronoun = None
            for child in token.children:
                if child.dep_ in ('nsubj', 'dobj', 'pobj', 'attr'):
                    rel_pronoun = child.text.lower()
                    break
            if rel_pronoun is None:
                rel_pronoun = token.text.lower()

            acl_info = {
                'acl_type': 'relcl_reference',
                'marker': rel_pronoun,
                'complementizer': None,
                'head_word': token.head.text if token.head else '',
                'head_pos': token.head.pos_ if token.head else '',
                'verb_word': token.text,
                'verb_pos': token.pos_,
                'position': token.i
            }

        # 3. mark 依存
        elif token.dep_ == 'mark':
            marker_lower = token.text.lower()
            if marker_lower in ACL_MARKERS:
                head = token.head
                if head:
                    acl_type = 'acl'
                    if head.dep_ == 'relcl':
                        acl_type = 'relcl_reference'
                    acl_info = {
                        'acl_type': acl_type,
                        'marker': marker_lower,
                        'complementizer': marker_lower,
                        'head_word': head.text if head else '',
                        'head_pos': head.pos_ if head else '',
                        'verb_word': head.text if head else '',
                        'verb_pos': head.pos_ if head else '',
                        'position': token.i
                    }

        if acl_info:
            results.append(acl_info)
    return results


# ============ CCOMP 分析 ============
def compute_ccomp_gaps(doc, ccomp_info):
    if len(ccomp_info) < 2:
        return []
    positions = [info['position'] for info in ccomp_info]
    gaps = []
    for i in range(len(positions) - 1):
        gap = positions[i + 1] - positions[i] - 1
        gaps.append(gap)
    return gaps


def analyze_ccomp_in_doc(doc):
    """分析文档中所有 CCOMP (Complement Clause)"""
    results = []
    for token in doc:
        comp_info = None

        # 1. dep_='ccomp'
        if token.dep_ == 'ccomp':
            marker_word = None
            marker_dep = None
            complementizer = None

            for child in token.children:
                if child.dep_ == 'mark':
                    marker_word = child.text.lower()
                    marker_dep = 'mark'
                    break

            if marker_word is None:
                for child in token.children:
                    if child.dep_ == 'comp':
                        complementizer = child.text.lower()
                        break

            comp_info = {
                'comp_type': 'ccomp',
                'marker': marker_word,
                'complementizer': complementizer,
                'head_word': token.head.text if token.head else '',
                'head_pos': token.head.pos_ if token.head else '',
                'verb_word': token.text,
                'verb_pos': token.pos_,
                'position': token.i
            }

        # 2. dep_='xcomp'
        elif token.dep_ == 'xcomp':
            marker_word = None
            for child in token.children:
                if child.dep_ == 'aux' and child.text.lower() == 'to':
                    marker_word = 'to'
                    break

            comp_info = {
                'comp_type': 'xcomp',
                'marker': marker_word if marker_word else 'bare_infinitive',
                'complementizer': None,
                'head_word': token.head.text if token.head else '',
                'head_pos': token.head.pos_ if token.head else '',
                'verb_word': token.text,
                'verb_pos': token.pos_,
                'position': token.i
            }

        # 3. mark 依存
        elif token.dep_ == 'mark':
            marker_lower = token.text.lower()
            if marker_lower in CCOMP_MARKERS:
                head = token.head
                if head:
                    comp_info = {
                        'comp_type': 'mark_' + head.dep_,
                        'marker': marker_lower,
                        'complementizer': marker_lower,
                        'head_word': head.text if head else '',
                        'head_pos': head.pos_ if head else '',
                        'verb_word': head.text if head else '',
                        'verb_pos': head.pos_ if head else '',
                        'position': token.i
                    }

        if comp_info:
            results.append(comp_info)
    return results


# ============ 标签函数 ============
def get_acl_freq_label(ratio):
    if ratio < 0.03:
        return 'low'
    elif ratio < 0.1:
        return 'medium'
    return 'high'

def get_acl_density_label(pps):
    if pps < 0.03:
        return 'simple'
    elif pps < 0.1:
        return 'medium'
    return 'complex'

def get_ccomp_freq_label(ratio):
    if ratio < 0.05:
        return 'low'
    elif ratio < 0.15:
        return 'medium'
    return 'high'

def get_ccomp_density_label(pps):
    if pps < 0.05:
        return 'simple'
    elif pps < 0.15:
        return 'medium'
    return 'complex'

def get_length_label(avg_len):
    if avg_len < 22:
        return 'short'
    elif avg_len < 28:
        return 'medium'
    return 'long'


# ============ 处理单个用户（同时计算 ACL 和 CCOMP） ============
def process_user(user_data):
    reviews = []
    for p in user_data.get('results', []):
        reviews.extend(p.get('target_reviews', []))

    user_id = user_data['user_id']
    if not reviews:
        return None, None, None

    # ACL 统计
    total_sentences = 0
    sentences_with_acl = 0
    total_acl_count = 0
    total_token_count = 0
    acl_type_counter = Counter()
    acl_marker_counter = Counter()
    all_acl_gaps = []
    acl_sentence_details = []

    # CCOMP 统计
    sentences_with_ccomp = 0
    total_ccomp_count = 0
    ccomp_type_counter = Counter()
    ccomp_marker_counter = Counter()
    all_ccomp_gaps = []
    ccomp_sentence_details = []

    for r in reviews:
        if not r:
            continue
        doc = nlp(r)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        n = len(tokens)
        if n == 0:
            continue

        total_sentences += 1
        total_token_count += n

        # ACL 分析
        acl_info = analyze_acl_in_doc(doc)
        acl_count = len(acl_info)
        total_acl_count += acl_count
        if acl_count > 0:
            sentences_with_acl += 1

        for info in acl_info:
            acl_type_counter[info['acl_type']] += 1
            if info.get('marker'):
                acl_marker_counter[info['marker']] += 1

        acl_gaps = compute_acl_gaps(doc, acl_info)
        all_acl_gaps.extend(acl_gaps)

        acl_sentence_details.append({
            'user_id': user_id,
            'sentence': r,
            'token_count': n,
            'acl_count': acl_count,
            'has_acl': acl_count > 0,
            'acl_info': acl_info
        })

        # CCOMP 分析
        ccomp_info = analyze_ccomp_in_doc(doc)
        ccomp_count = len(ccomp_info)
        total_ccomp_count += ccomp_count
        if ccomp_count > 0:
            sentences_with_ccomp += 1

        for info in ccomp_info:
            ccomp_type_counter[info['comp_type']] += 1
            if info.get('marker'):
                ccomp_marker_counter[info['marker']] += 1

        ccomp_gaps = compute_ccomp_gaps(doc, ccomp_info)
        all_ccomp_gaps.extend(ccomp_gaps)

        ccomp_sentence_details.append({
            'user_id': user_id,
            'sentence': r,
            'token_count': n,
            'ccomp_count': ccomp_count,
            'has_ccomp': ccomp_count > 0,
            'ccomp_info': ccomp_info
        })

    if total_sentences == 0:
        return None, None, None

    # ACL 指标
    acl_sentence_ratio = sentences_with_acl / total_sentences
    acl_per_sentence = total_acl_count / total_sentences
    words_per_acl = total_token_count / total_acl_count if total_acl_count > 0 else None
    acl_mean_gap = float(np.mean(all_acl_gaps)) if all_acl_gaps else None
    acl_median_gap = float(np.median(all_acl_gaps)) if all_acl_gaps else None

    acl_result = {
        'user_id': user_id,
        'acl_sentence_ratio': acl_sentence_ratio,
        'acl_per_sentence': acl_per_sentence,
        'avg_sentence_length': total_token_count / total_sentences,
        'words_per_acl': words_per_acl,
        'freq_label': get_acl_freq_label(acl_sentence_ratio),
        'density_label': get_acl_density_label(acl_per_sentence),
        'length_label': get_length_label(total_token_count / total_sentences),
        'total_sentences': total_sentences,
        'total_acl_count': total_acl_count,
        'acl_type_distribution': dict(acl_type_counter),
        'marker_distribution': dict(acl_marker_counter),
        'mean_gap': acl_mean_gap,
        'median_gap': acl_median_gap
    }

    # CCOMP 指标
    ccomp_sentence_ratio = sentences_with_ccomp / total_sentences
    ccomp_per_sentence = total_ccomp_count / total_sentences
    words_per_ccomp = total_token_count / total_ccomp_count if total_ccomp_count > 0 else None
    ccomp_mean_gap = float(np.mean(all_ccomp_gaps)) if all_ccomp_gaps else None
    ccomp_median_gap = float(np.median(all_ccomp_gaps)) if all_ccomp_gaps else None

    ccomp_result = {
        'user_id': user_id,
        'ccomp_sentence_ratio': ccomp_sentence_ratio,
        'ccomp_per_sentence': ccomp_per_sentence,
        'avg_sentence_length': total_token_count / total_sentences,
        'words_per_ccomp': words_per_ccomp,
        'freq_label': get_ccomp_freq_label(ccomp_sentence_ratio),
        'density_label': get_ccomp_density_label(ccomp_per_sentence),
        'length_label': get_length_label(total_token_count / total_sentences),
        'total_sentences': total_sentences,
        'total_ccomp_count': total_ccomp_count,
        'ccomp_type_distribution': dict(ccomp_type_counter),
        'marker_distribution': dict(ccomp_marker_counter),
        'mean_gap': ccomp_mean_gap,
        'median_gap': ccomp_median_gap
    }

    return acl_result, ccomp_result, acl_sentence_details, ccomp_sentence_details


# ============ 统计输出函数 ============
def compute_distribution(vals, max_val_override=None):
    max_val = max_val_override if max_val_override is not None else vals.max()
    max_bin = int(max_val * 100) + 2
    dist = []
    for i in range(max_bin):
        low = i / 100.0
        high = (i + 1) / 100.0
        if i == max_bin - 1:
            count = int(np.sum(vals >= low))
        else:
            count = int(np.sum((vals >= low) & (vals < high)))
        pct = count / len(vals) * 100
        label = f'[{low:.2f}, {high:.2f})' if i < max_bin - 1 else f'[{low:.2f}, +∞)'
        dist.append({'interval': label, 'count': count, 'percentage': round(pct, 2)})
    return dist


def count_labels(users, key):
    labels = [u[key] for u in users]
    cnt = Counter(labels)
    total_u = len(users)
    result = []
    for label in ['low', 'medium', 'high', 'simple', 'complex', 'short', 'medium', 'long']:
        if label in cnt:
            result.append({'label': label, 'count': cnt[label], 'percentage': round(cnt[label]/total_u*100, 2)})
    return result


def print_stats(users, type_name, ratio_key, pps_key, wpc_key, type_dist_key, marker_dist_key, gap_key, gap_median_key):
    """打印统计信息"""
    ratio_vals = np.array([u[ratio_key] for u in users])
    pps_vals = np.array([u[pps_key] for u in users])
    len_vals = np.array([u['avg_sentence_length'] for u in users])
    tpr_vals = np.array([u[wpc_key] for u in users if u[wpc_key] is not None])
    mean_gap_vals = np.array([u[gap_key] for u in users if u[gap_key] is not None])
    median_gap_vals = np.array([u[gap_median_key] for u in users if u[gap_median_key] is not None])

    # ratio 分布
    log('')
    log(f'========== {type_name}_sentence_ratio 分布 ==========')
    log(f'最小值: {ratio_vals.min():.4f}, 最大值: {ratio_vals.max():.4f}, 均值: {ratio_vals.mean():.4f}')

    # pps 分布
    log('')
    log(f'========== {type_name}_per_sentence 分布 ==========')
    log(f'最小值: {pps_vals.min():.4f}, 最大值: {pps_vals.max():.4f}, 均值: {pps_vals.mean():.4f}')

    # 句子长度
    log('')
    log(f'========== avg_sentence_length 分布 ==========')
    log(f'最小值: {len_vals.min():.1f}, 最大值: {len_vals.max():.1f}, 均值: {len_vals.mean():.1f}')

    # words_per_X 分布
    log('')
    log(f'========== words_per_{type_name.upper()} 分布 ==========')
    log(f'有效用户数: {len(tpr_vals)}')
    if len(tpr_vals) > 0:
        log(f'最小值: {tpr_vals.min():.1f}, 最大值: {tpr_vals.max():.1f}, 均值: {tpr_vals.mean():.1f}, 中位数: {np.median(tpr_vals):.1f}')

        intervals = [
            (0, 5, '0-5'), (5, 10, '5-10'), (10, 15, '10-15'), (15, 20, '15-20'),
            (20, 25, '20-25'), (25, 30, '25-30'), (30, 35, '30-35'),
            (35, 40, '35-40'), (40, 45, '40-45'), (45, 50, '45-50'), (50, float('inf'), '50+')
        ]
        log(f'{"区间":<15} │ {"人数":>6} │ {"占比":>8}')
        log('─' * 40)
        wpr_dist = []
        for low, high, label in intervals:
            if high == float('inf'):
                count = int(np.sum(tpr_vals >= low))
            else:
                count = int(np.sum((tpr_vals >= low) & (tpr_vals < high)))
            pct = count / len(tpr_vals) * 100 if len(tpr_vals) > 0 else 0
            bar = '█' * int(pct)
            log(f'{label:<15} │ {count:>6} │ {pct:>7.2f}% │ {bar}')
            wpr_dist.append({'interval': label, 'count': count, 'percentage': round(pct, 2)})
    else:
        wpr_dist = []

    # gap 统计
    log('')
    log(f'========== {type_name} 间隔统计 ==========')
    log(f'有效用户数: {len(mean_gap_vals)}')
    if len(mean_gap_vals) > 0:
        log(f'mean_gap: 最小={mean_gap_vals.min():.2f}, 最大={mean_gap_vals.max():.2f}, 均值={mean_gap_vals.mean():.2f}, 中位数={np.median(mean_gap_vals):.2f}')
        log(f'median_gap: 最小={median_gap_vals.min():.2f}, 最大={median_gap_vals.max():.2f}, 均值={median_gap_vals.mean():.2f}, 中位数={np.median(median_gap_vals):.2f}')

        intervals = [
            (0, 5, '0-5'), (5, 10, '5-10'), (10, 15, '10-15'),
            (15, 20, '15-20'), (20, 25, '20-25'), (25, 30, '25-30'),
            (30, float('inf'), '30+')
        ]
        log(f'{"区间":<15} │ {"用户数":>8} │ {"占比":>8}')
        log('─' * 45)
        gap_dist = []
        for low, high, label in intervals:
            if high == float('inf'):
                count = int(np.sum(mean_gap_vals >= low))
            else:
                count = int(np.sum((mean_gap_vals >= low) & (mean_gap_vals < high)))
            pct = count / len(mean_gap_vals) * 100 if len(mean_gap_vals) > 0 else 0
            bar = '█' * int(pct / 2)
            log(f'{label:<15} │ {count:>8} │ {pct:>7.2f}% │ {bar}')
            gap_dist.append({'interval': label, 'count': count, 'percentage': round(pct, 2)})
    else:
        gap_dist = []

    # 类型分布
    global_type_counter = Counter()
    global_marker_counter = Counter()
    for u in users:
        for typ, cnt in u[type_dist_key].items():
            global_type_counter[typ] += cnt
        for marker, cnt in u.get(marker_dist_key, {}).items():
            global_marker_counter[marker] += cnt

    total_count = sum(global_type_counter.values())
    log('')
    log(f'========== {type_name} 类型分布（全局） ==========')
    log(f'总{type_name}数: {total_count}')
    log(f'{"类型":<30} │ {"数量":>6} │ {"占比":>8}')
    log('─' * 60)
    for typ, count in sorted(global_type_counter.items(), key=lambda x: -x[1]):
        pct = count / total_count * 100 if total_count > 0 else 0
        bar = '█' * int(pct / 2)
        log(f'{typ:<30} │ {count:>6} │ {pct:>7.1f}% │ {bar}')

    # 引导词分布
    total_markers = sum(global_marker_counter.values())
    log('')
    log(f'========== {type_name} 引导词分布（全局） ==========')
    log(f'总引导词数: {total_markers}')
    log(f'{"引导词":<15} │ {"数量":>6} │ {"占比":>8}')
    log('─' * 40)
    for marker, count in sorted(global_marker_counter.items(), key=lambda x: -x[1])[:20]:
        pct = count / total_markers * 100 if total_markers > 0 else 0
        bar = '█' * int(pct / 2)
        log(f'{marker:<15} │ {count:>6} │ {pct:>7.1f}% │ {bar}')

    return {
        'total_count': total_count,
        'dist_wpr': wpr_dist,
        'dist_gap': gap_dist,
        'global_type_counter': dict(global_type_counter),
        'global_marker_counter': dict(global_marker_counter),
        'ratio_vals': ratio_vals,
        'pps_vals': pps_vals,
        'len_vals': len_vals,
        'tpr_vals': tpr_vals,
        'mean_gap_vals': mean_gap_vals,
        'median_gap_vals': median_gap_vals,
    }


def build_summary_stats(stats, users, type_name, ratio_key, pps_key, wpc_key):
    ratio_vals = stats['ratio_vals']
    pps_vals = stats['pps_vals']
    len_vals = stats['len_vals']
    tpr_vals = stats['tpr_vals']
    mean_gap_vals = stats['mean_gap_vals']
    median_gap_vals = stats['median_gap_vals']

    return {
        'model': 'en_core_web_sm',
        'total_users': len(users),
        'valid_users': len(users),
        'total_sentences': sum(u['total_sentences'] for u in users),
        f'total_{type_name}s': stats['total_count'],
        f'{type_name}_sentence_ratio': {
            'min': round(float(ratio_vals.min()), 4),
            'max': round(float(ratio_vals.max()), 4),
            'mean': round(float(ratio_vals.mean()), 4)
        },
        f'{type_name}_per_sentence': {
            'min': round(float(pps_vals.min()), 4),
            'max': round(float(pps_vals.max()), 4),
            'mean': round(float(pps_vals.mean()), 4)
        },
        'avg_sentence_length': {
            'min': round(float(len_vals.min()), 1),
            'max': round(float(len_vals.max()), 1),
            'mean': round(float(len_vals.mean()), 1)
        },
        f'words_per_{type_name}': {
            'min': round(float(tpr_vals.min()), 1) if len(tpr_vals) > 0 else None,
            'max': round(float(tpr_vals.max()), 1) if len(tpr_vals) > 0 else None,
            'mean': round(float(tpr_vals.mean()), 1) if len(tpr_vals) > 0 else None,
            'median': round(float(np.median(tpr_vals)), 1) if len(tpr_vals) > 0 else None
        },
        'mean_gap': {
            'min': round(float(mean_gap_vals.min()), 2) if len(mean_gap_vals) > 0 else None,
            'max': round(float(mean_gap_vals.max()), 2) if len(mean_gap_vals) > 0 else None,
            'mean': round(float(mean_gap_vals.mean()), 2) if len(mean_gap_vals) > 0 else None,
            'median': round(float(np.median(mean_gap_vals)), 2) if len(mean_gap_vals) > 0 else None
        },
        'median_gap': {
            'min': round(float(median_gap_vals.min()), 2) if len(median_gap_vals) > 0 else None,
            'max': round(float(median_gap_vals.max()), 2) if len(median_gap_vals) > 0 else None,
            'mean': round(float(median_gap_vals.mean()), 2) if len(median_gap_vals) > 0 else None,
            'median': round(float(np.median(median_gap_vals)), 2) if len(median_gap_vals) > 0 else None
        },
        f'{type_name}_type_distribution': stats['global_type_counter'],
        'marker_distribution': stats['global_marker_counter'],
        'elapsed_seconds': 0,
        'distribution_ratio': compute_distribution(ratio_vals),
        'distribution_pps': compute_distribution(pps_vals, max_val_override=0.3),
        f'distribution_wpr': stats['dist_wpr'],
        f'distribution_gap': stats['dist_gap']
    }


# ============ 主函数 ============
def main():
    start_time = time.time()

    log('=' * 60)
    log(f'ACL/CCOMP 句法分析（合并版）- {CATEGORY}')
    log('=' * 60)

    # 加载用户数据
    log(f'加载用户数据: {ALL_USERS_FILE}')
    with open(ALL_USERS_FILE, 'r', encoding='utf-8') as f:
        all_users_data = json.load(f)
    user_list = all_users_data['users']
    total = len(user_list)
    log(f'总用户数: {total}')

    # 加载 Stage 1 属性
    log(f'加载Stage 1属性数据: {STAGE1_ATTR_FILE}')
    with open(STAGE1_ATTR_FILE, 'r', encoding='utf-8') as f:
        stage1_data = json.load(f)
    asin_to_attrs = {}
    for p in stage1_data.get('products', []):
        asin = p.get('asin', '')
        if asin:
            asin_to_attrs[asin] = {
                'A1_product_type': p.get('A1_product_type'),
                'A2_brand': p.get('A2_brand'),
                'A3_price': p.get('A3_price'),
                'A4_appearance': p.get('A4_appearance'),
                'A5_use_case': p.get('A5_use_case'),
            }
    log(f'Stage 1有效商品数: {len(asin_to_attrs)}')

    # 打开输出文件
    fp_acl = open(ACL_OUTPUT_JSONL, 'w', encoding='utf-8')
    fp_ccomp = open(CCOMP_OUTPUT_JSONL, 'w', encoding='utf-8')

    acl_users = []
    ccomp_users = []
    acl_sentence_count = 0
    ccomp_sentence_count = 0

    for idx, user_data in enumerate(user_list):
        if idx % 500 == 0:
            elapsed = time.time() - start_time
            log(f'进度: {idx}/{total} ({idx/total*100:.1f}%), 耗时: {elapsed:.1f}s')

        acl_result, ccomp_result, acl_sents, ccomp_sents = process_user(user_data)

        if acl_result is not None:
            # 添加商品属性
            user_id = acl_result['user_id']
            user_products = []
            for product in user_data.get('results', []):
                asin = product.get('asin', '')
                if asin in asin_to_attrs:
                    user_products.append({'asin': asin, **asin_to_attrs[asin]})
            acl_result['products'] = user_products
            ccomp_result['products'] = user_products

            acl_users.append(acl_result)
            ccomp_users.append(ccomp_result)

            for s in acl_sents:
                fp_acl.write(json.dumps(s, ensure_ascii=False) + '\n')
            acl_sentence_count += len(acl_sents)

            for s in ccomp_sents:
                fp_ccomp.write(json.dumps(s, ensure_ascii=False) + '\n')
            ccomp_sentence_count += len(ccomp_sents)

    fp_acl.close()
    fp_ccomp.close()

    elapsed = time.time() - start_time
    log(f'进度: {total}/{total} (100.0%), 耗时: {elapsed:.1f}s')
    log(f'ACL 有效用户: {len(acl_users)}, 句子: {acl_sentence_count}')
    log(f'CCOMP 有效用户: {len(ccomp_users)}, 句子: {ccomp_sentence_count}')

    # 打印 ACL 统计
    log('\n' + '=' * 60)
    log('ACL 统计')
    log('=' * 60)
    acl_stats = print_stats(
        acl_users, 'acl',
        'acl_sentence_ratio', 'acl_per_sentence', 'words_per_acl',
        'acl_type_distribution', 'marker_distribution',
        'mean_gap', 'median_gap'
    )

    # 打印 CCOMP 统计
    log('\n' + '=' * 60)
    log('CCOMP 统计')
    log('=' * 60)
    ccomp_stats = print_stats(
        ccomp_users, 'ccomp',
        'ccomp_sentence_ratio', 'ccomp_per_sentence', 'words_per_ccomp',
        'ccomp_type_distribution', 'marker_distribution',
        'mean_gap', 'median_gap'
    )

    total_time = time.time() - start_time

    # 保存 ACL 结果
    log(f'\n保存ACL用户档案到: {ACL_USERS_JSON}')
    with open(ACL_USERS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(acl_users, fp, ensure_ascii=False, indent=2)

    acl_summary = build_summary_stats(acl_stats, acl_users, 'acl', 'acl_sentence_ratio', 'acl_per_sentence', 'words_per_acl')
    acl_summary['elapsed_seconds'] = round(total_time, 1)

    log(f'保存ACL统计到: {ACL_STATS_JSON}')
    with open(ACL_STATS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(acl_summary, fp, ensure_ascii=False, indent=2)

    # 保存 CCOMP 结果
    log(f'\n保存CCOMP用户档案到: {CCOMP_USERS_JSON}')
    with open(CCOMP_USERS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(ccomp_users, fp, ensure_ascii=False, indent=2)

    ccomp_summary = build_summary_stats(ccomp_stats, ccomp_users, 'ccomp', 'ccomp_sentence_ratio', 'ccomp_per_sentence', 'words_per_ccomp')
    ccomp_summary['elapsed_seconds'] = round(total_time, 1)

    log(f'保存CCOMP统计到: {CCOMP_STATS_JSON}')
    with open(CCOMP_STATS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(ccomp_summary, fp, ensure_ascii=False, indent=2)

    log(f'\n总耗时: {total_time:.1f}s')


if __name__ == '__main__':
    main()
