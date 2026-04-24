#!/usr/bin/env python3
"""ACL/CCOMP/AttrDensity 句法分析（spaCy模型分析）- Baby_Products

同时计算：
- ACL (Adjectival Clause) - 形容词性从句
- CCOMP (Complement Clause) - 补语从句
- AttrDensity (Attribute Density) - 属性词密度
"""
import json, numpy as np
import warnings
import time
import os
import sys
import re
from datetime import datetime
from collections import Counter
warnings.filterwarnings('ignore')
import spacy
nlp = spacy.load('en_core_web_sm')
from sklearn.linear_model import LinearRegression

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

# ============ 路径配置 ============
CATEGORY = "Baby_Products"
BASE_DIR = f'/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/{CATEGORY}'
os.makedirs(BASE_DIR, exist_ok=True)

ACL_OUTPUT_JSONL = f'{BASE_DIR}/acl_sentences.jsonl'
ACL_STATS_JSON = f'{BASE_DIR}/acl_stats.json'
ACL_USERS_JSON = f'{BASE_DIR}/acl_user_profiles.json'

CCOMP_OUTPUT_JSONL = f'{BASE_DIR}/ccomp_sentences.jsonl'
CCOMP_STATS_JSON = f'{BASE_DIR}/ccomp_stats.json'
CCOMP_USERS_JSON = f'{BASE_DIR}/ccomp_user_profiles.json'

ATTR_DENSITY_OUTPUT_JSONL = f'{BASE_DIR}/attr_density_sentences.jsonl'
ATTR_DENSITY_STATS_JSON = f'{BASE_DIR}/attr_density_stats.json'
ATTR_DENSITY_USERS_JSON = f'{BASE_DIR}/attr_density_user_profiles.json'

ALL_USERS_FILE = f'/home/wlia0047/ar57_scratch/wenyu/result/personal_query/01_preference_extraction/{CATEGORY}/stage1_filtered_users_reviews.json'
STAGE1_ATTR_FILE = f'/home/wlia0047/ar57_scratch/wenyu/result/personal_query/01_preference_extraction/{CATEGORY}/attributes_{CATEGORY}.json'

# ============ 标记词 ============
# ACL 标记词（形容词性从句、状语从句、主语从句等）
ACL_MARKERS = {'that', 'whether', 'if', 'what', 'which', 'who', 'whom', 'whose',
               'whatever', 'whichever', 'whoever', 'when', 'where', 'why', 'how',
               'because', 'although', 'while', 'until', 'unless', 'since', 'before', 'after', 'though', 'if'}
# CCOMP 标记词
CCOMP_MARKERS = {'that', 'whether', 'if', 'what', 'which', 'who', 'whom', 'whose'}

# ============ 属性词典 ============
ATTRIBUTE_DICTIONARY = {
    'color': {'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white', 'gray', 'grey',
              'brown', 'navy', 'gold', 'silver', 'bronze', 'copper', 'beige', 'cream', 'ivory', 'tan', 'teal',
              'maroon', 'burgundy', 'coral', 'lavender', 'violet', 'indigo', 'colorful', 'bright', 'dark', 'light',
              'vibrant', 'pastel', 'neon', 'solid', 'striped', 'patterned', 'multicolored'},
    'material': {'plastic', 'metal', 'wood', 'wooden', 'paper', 'cardboard', 'fabric', 'cotton', 'polyester',
                 'silk', 'leather', 'rubber', 'glass', 'ceramic', 'porcelain', 'stone', 'marble', 'bamboo',
                 'steel', 'aluminum', 'copper', 'brass', 'iron', 'silver', 'gold', 'carbon', 'fiberglass',
                 'nylon', 'canvas', 'linen', 'velvet', 'wool', 'acrylic', 'vinyl', 'mesh', 'foam',
                 'resin', 'polymer', 'composite', 'natural', 'synthetic', 'eco-friendly', 'recycled'},
    'function': {'easy', 'simple', 'convenient', 'practical', 'functional', 'versatile', 'multi-purpose',
                'portable', 'compact', 'lightweight', 'heavy-duty', 'professional', 'beginner-friendly',
                'automatic', 'manual', 'rechargeable', 'battery-operated', 'cordless', 'electric', 'digital',
                'adjustable', 'foldable', 'detachable', 'removable', 'washable', 'waterproof', 'water-resistant',
                'heat-resistant', 'fireproof', 'scratch-resistant', 'stain-resistant', 'fade-resistant'},
    'appearance': {'beautiful', 'elegant', 'stylish', 'modern', 'classic', 'vintage', 'rustic', 'minimalist',
                  'decorative', 'ornate', 'sleek', 'smooth', 'textured', 'glossy', 'matte', 'shiny',
                  'transparent', 'opaque', 'translucent', 'printed', 'embossed', 'laser-cut', 'handmade',
                  'craftsman', 'artisan', 'delicate', 'intricate', 'simple', 'plain', 'fancy', 'exquisite'},
    'size': {'small', 'medium', 'large', 'tiny', 'mini', 'giant', 'big', 'compact', 'oversized', 'full-size',
            'standard', 'massive', 'petite', 'slim', 'thick', 'thin', 'long', 'short', 'tall', 'wide', 'narrow'},
    'quality': {'high-quality', 'premium', 'cheap', 'affordable', 'expensive', 'value', 'worth', 'durable',
               'sturdy', 'solid', 'fragile', 'flimsy', 'robust', 'heavy', 'strong', 'weak'},
    'safety': {'safe', 'unsafe', 'non-toxic', 'toxic', 'hazardous', 'dangerous', 'eco-friendly', 'green',
              'organic', 'natural', 'chemical-free', 'phthalate-free', 'bpa-free', 'lead-free', 'food-grade',
              'medical-grade', 'hypoallergenic', 'allergy-free', 'child-safe', 'pet-safe'},
    'usecase': {'indoor', 'outdoor', 'indoor-outdoor', 'kitchen', 'bathroom', 'bedroom',
               'living room', 'office', 'classroom', 'studio', 'workshop', 'garage', 'garden', 'camping',
               'travel', 'portable', 'home', 'commercial', 'professional', 'personal', 'gift', 'beginner',
               'expert', 'children', 'kids', 'adults', 'seniors', 'family'},
}
ALL_ATTRIBUTE_WORDS = set()
for category_words in ATTRIBUTE_DICTIONARY.values():
    ALL_ATTRIBUTE_WORDS.update(category_words)


# ============ ACL 分析（句法树变宽特征）============
# ACL 让句法树变宽，包括：
# - acl: 形容词性从句
# - relcl: 关系从句
# - advcl: 状语从句（可并列，让树变宽）
# - conj: 并列结构（让树变宽）
# - parataxis: 并列结构
# 注意：csubj/csubjpass 属于"变深"特征，在 CCOMP 中统计

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
    results = []
    for token in doc:
        acl_info = None

        # 1. acl: 形容词性从句
        if token.dep_ == 'acl':
            marker_word = None
            complementizer = None
            for child in token.children:
                if child.dep_ == 'mark':
                    marker_word = child.text.lower()
                    complementizer = child.text.lower()
                    break
            if marker_word is None:
                for child in token.children:
                    if child.dep_ == 'comp':
                        marker_word = child.text.lower()
                        complementizer = child.text.lower()
                        break
            acl_info = {
                'acl_type': 'acl',
                'marker': marker_word,
                'complementizer': complementizer,
                'head_word': token.head.text if token.head else '',
                'verb_word': token.text,
                'position': token.i
            }

        # 2. relcl: 关系从句
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
                'verb_word': token.text,
                'position': token.i
            }

        # 3. advcl: 状语从句（让句法树变宽）
        elif token.dep_ == 'advcl':
            marker_word = None
            for child in token.children:
                if child.dep_ == 'mark':
                    marker_word = child.text.lower()
                    break
            acl_info = {
                'acl_type': 'advcl',
                'marker': marker_word,
                'complementizer': None,
                'head_word': token.head.text if token.head else '',
                'verb_word': token.text,
                'position': token.i
            }

        # 4. conj: 并列结构（让句法树变宽）
        elif token.dep_ == 'conj':
            acl_info = {
                'acl_type': 'conj',
                'marker': None,
                'complementizer': None,
                'head_word': token.head.text if token.head else '',
                'verb_word': token.text,
                'position': token.i
            }

        # 5. parataxis: 并列结构（让句法树变宽）
        elif token.dep_ == 'parataxis':
            acl_info = {
                'acl_type': 'parataxis',
                'marker': None,
                'complementizer': None,
                'head_word': token.head.text if token.head else '',
                'verb_word': token.text,
                'position': token.i
            }

        # 6. mark 标记词开头的从句
        elif token.dep_ == 'mark':
            marker_lower = token.text.lower()
            if marker_lower in ACL_MARKERS:
                head = token.head
                if head:
                    if head.dep_ == 'relcl':
                        acl_type = 'relcl_reference'
                    elif head.dep_ == 'advcl':
                        acl_type = 'advcl'
                    elif head.dep_ == 'csubj':
                        acl_type = 'csubj'
                    elif head.dep_ == 'csubjpass':
                        acl_type = 'csubjpass'
                    else:
                        acl_type = 'acl'
                    acl_info = {
                        'acl_type': acl_type,
                        'marker': marker_lower,
                        'complementizer': marker_lower,
                        'head_word': head.text if head else '',
                        'verb_word': head.text if head else '',
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
    results = []
    for token in doc:
        comp_info = None
        if token.dep_ == 'ccomp':
            marker_word = None
            complementizer = None
            for child in token.children:
                if child.dep_ == 'mark':
                    marker_word = child.text.lower()
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
                'verb_word': token.text,
                'position': token.i
            }
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
                'verb_word': token.text,
                'position': token.i
            }
        # csubj/csubjpass: 主语从句/被动主语从句（让句法树变深）
        elif token.dep_ == 'csubj':
            comp_info = {
                'comp_type': 'csubj',
                'marker': None,
                'complementizer': None,
                'head_word': token.head.text if token.head else '',
                'verb_word': token.text,
                'position': token.i
            }
        elif token.dep_ == 'csubjpass':
            comp_info = {
                'comp_type': 'csubjpass',
                'marker': None,
                'complementizer': None,
                'head_word': token.head.text if token.head else '',
                'verb_word': token.text,
                'position': token.i
            }
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
                        'verb_word': head.text if head else '',
                        'position': token.i
                    }
        if comp_info:
            results.append(comp_info)
    return results


# ============ AttrDensity 分析 ============
def is_attribute_word_spacy(token):
    return token.pos_ in ('ADJ', 'NOUN') and not token.is_punct and not token.is_space

def count_attributes_in_sentence_spacy(doc):
    category_counts = Counter()
    total_count = 0
    for token in doc:
        if is_attribute_word_spacy(token):
            category_counts[token.pos_] += 1
            total_count += 1
    return total_count, dict(category_counts)

def analyze_attributes_in_doc_spacy(doc):
    results = []
    for token in doc:
        if is_attribute_word_spacy(token):
            results.append({
                'word': token.text,
                'category': token.pos_,
                'pos': token.pos_,
                'dep': token.dep_,
                'position': token.i
            })
    return results

def compute_attr_gaps_spacy(doc):
    attr_positions = [token.i for token in doc if is_attribute_word_spacy(token)]
    gaps = []
    for i in range(len(attr_positions) - 1):
        gap = attr_positions[i + 1] - attr_positions[i] - 1
        gaps.append(gap)
    return gaps


# ============ 标签函数 ============
def get_acl_freq_label(ratio):
    if ratio < 0.03: return 'low'
    elif ratio < 0.1: return 'medium'
    return 'high'

def get_acl_density_label(pps):
    if pps < 0.03: return 'simple'
    elif pps < 0.1: return 'medium'
    return 'complex'

def get_ccomp_freq_label(ratio):
    if ratio < 0.05: return 'low'
    elif ratio < 0.15: return 'medium'
    return 'high'

def get_ccomp_density_label(pps):
    if pps < 0.05: return 'simple'
    elif pps < 0.15: return 'medium'
    return 'complex'

def get_attr_freq_label(ratio):
    if ratio < 0.5: return 'low'
    elif ratio < 1.0: return 'medium'
    return 'high'

def get_attr_density_label(wpa):
    if wpa is None: return 'none'
    if wpa < 5: return 'very_high'
    elif wpa < 10: return 'high'
    elif wpa < 15: return 'medium'
    elif wpa < 20: return 'low'
    return 'very_low'

def get_length_label(avg_len):
    if avg_len < 22: return 'short'
    elif avg_len < 28: return 'medium'
    return 'long'


# ============ 处理单个用户（ACL + CCOMP + AttrDensity） ============
def process_user(user_data):
    reviews = []
    for p in user_data.get('results', []):
        reviews.extend(p.get('target_reviews', []))

    user_id = user_data['user_id']
    if not reviews:
        return None, None, None, None, None, None

    # ACL
    total_sentences = 0
    sentences_with_acl = 0
    total_acl_count = 0
    total_token_count = 0
    acl_type_counter = Counter()
    acl_marker_counter = Counter()
    all_acl_gaps = []
    acl_sentence_details = []

    # CCOMP
    sentences_with_ccomp = 0
    total_ccomp_count = 0
    ccomp_type_counter = Counter()
    ccomp_marker_counter = Counter()
    all_ccomp_gaps = []
    ccomp_sentence_details = []

    # AttrDensity
    sentences_with_attr = 0
    total_attr_count = 0
    attr_category_counter = Counter()
    all_attr_gaps = []
    attr_sentence_details = []
    regression_data = []
    token_lengths_for_eq = []
    attr_counts_for_eq = []

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

        # ACL
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
            'user_id': user_id, 'sentence': r, 'token_count': n,
            'acl_count': acl_count, 'has_acl': acl_count > 0, 'acl_info': acl_info
        })

        # CCOMP
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
            'user_id': user_id, 'sentence': r, 'token_count': n,
            'ccomp_count': ccomp_count, 'has_ccomp': ccomp_count > 0, 'ccomp_info': ccomp_info
        })

        # AttrDensity
        attr_count, attr_cat_dist = count_attributes_in_sentence_spacy(doc)
        attr_info = analyze_attributes_in_doc_spacy(doc)
        total_attr_count += attr_count
        if attr_count > 0:
            sentences_with_attr += 1
        for cat, cnt in attr_cat_dist.items():
            attr_category_counter[cat] += cnt
        wpa = n / attr_count if attr_count > 0 else None
        attr_gaps = compute_attr_gaps_spacy(doc)
        all_attr_gaps.extend(attr_gaps)
        regression_data.append((attr_count, n))
        token_lengths_for_eq.append(n)
        attr_counts_for_eq.append(attr_count)
        attr_sentence_details.append({
            'user_id': user_id, 'sentence': r, 'token_count': n,
            'attr_count': attr_count, 'words_per_attribute': wpa,
            'has_attr': attr_count > 0, 'attr_categories': attr_cat_dist, 'attr_info': attr_info
        })

    if total_sentences == 0:
        return None, None, None, None, None, None

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

    # AttrDensity 指标
    attr_sentence_ratio = sentences_with_attr / total_sentences
    attr_per_sentence = total_attr_count / total_sentences
    words_per_attr = total_token_count / total_attr_count if total_attr_count > 0 else None
    attr_mean_gap = float(np.mean(all_attr_gaps)) if all_attr_gaps else None
    attr_median_gap = float(np.median(all_attr_gaps)) if all_attr_gaps else None

    # 线性回归
    model_wpa = None
    model_overhead = None
    model_r2 = None
    if len(regression_data) >= 3:
        X = np.array([[c] for c, _ in regression_data])
        y = np.array([l for _, l in regression_data])
        model = LinearRegression()
        model.fit(X, y)
        model_wpa = float(model.coef_[0])
        model_overhead = float(model.intercept_)
        model_r2 = float(model.score(X, y))

    # expected_query_length
    eq_lengths = [n for n, c in zip(token_lengths_for_eq, attr_counts_for_eq) if c >= 2]
    expected_query_length = float(np.mean(eq_lengths)) if eq_lengths else None

    attr_result = {
        'user_id': user_id,
        'attr_sentence_ratio': attr_sentence_ratio,
        'attr_per_sentence': attr_per_sentence,
        'avg_sentence_length': total_token_count / total_sentences,
        'words_per_attribute': words_per_attr,
        'freq_label': get_attr_freq_label(attr_sentence_ratio),
        'density_label': get_attr_density_label(words_per_attr),
        'length_label': get_length_label(total_token_count / total_sentences),
        'total_sentences': total_sentences,
        'total_attr_count': total_attr_count,
        'category_distribution': dict(attr_category_counter),
        'mean_gap': attr_mean_gap,
        'median_gap': attr_median_gap,
        'model_words_per_attribute': model_wpa,
        'model_base_overhead': model_overhead,
        'model_r2': model_r2,
        'expected_query_length': expected_query_length
    }

    return (acl_result, ccomp_result, attr_result,
            acl_sentence_details, ccomp_sentence_details, attr_sentence_details)


# ============ 统计函数 ============
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


def print_clause_stats(users, type_name, ratio_key, pps_key, wpc_key, type_dist_key, marker_dist_key, gap_key, gap_median_key):
    ratio_vals = np.array([u[ratio_key] for u in users])
    pps_vals = np.array([u[pps_key] for u in users])
    len_vals = np.array([u['avg_sentence_length'] for u in users])
    tpr_vals = np.array([u[wpc_key] for u in users if u[wpc_key] is not None])
    mean_gap_vals = np.array([u[gap_key] for u in users if u[gap_key] is not None])
    median_gap_vals = np.array([u[gap_median_key] for u in users if u[gap_median_key] is not None])

    log(f'  {type_name}_sentence_ratio: min={ratio_vals.min():.4f}, max={ratio_vals.max():.4f}, mean={ratio_vals.mean():.4f}')
    log(f'  {type_name}_per_sentence: min={pps_vals.min():.4f}, max={pps_vals.max():.4f}, mean={pps_vals.mean():.4f}')
    log(f'  avg_sentence_length: min={len_vals.min():.1f}, max={len_vals.max():.1f}, mean={len_vals.mean():.1f}')

    if len(tpr_vals) > 0:
        log(f'  words_per_{type_name}: n={len(tpr_vals)}, min={tpr_vals.min():.1f}, max={tpr_vals.max():.1f}, mean={tpr_vals.mean():.1f}, median={np.median(tpr_vals):.1f}')

    if len(mean_gap_vals) > 0:
        log(f'  mean_gap: n={len(mean_gap_vals)}, min={mean_gap_vals.min():.2f}, max={mean_gap_vals.max():.2f}, mean={mean_gap_vals.mean():.2f}')

    # 类型分布
    global_type_counter = Counter()
    global_marker_counter = Counter()
    for u in users:
        for typ, cnt in u[type_dist_key].items():
            global_type_counter[typ] += cnt
        for marker, cnt in u.get(marker_dist_key, {}).items():
            global_marker_counter[marker] += cnt

    total_count = sum(global_type_counter.values())
    log(f'  总{type_name}数: {total_count}')

    return {
        'total_count': total_count,
        'ratio_vals': ratio_vals,
        'pps_vals': pps_vals,
        'len_vals': len_vals,
        'tpr_vals': tpr_vals,
        'mean_gap_vals': mean_gap_vals,
        'median_gap_vals': median_gap_vals,
        'global_type_counter': dict(global_type_counter),
        'global_marker_counter': dict(global_marker_counter),
    }


def print_attr_stats(users):
    ratio_vals = np.array([u['attr_sentence_ratio'] for u in users])
    aps_vals = np.array([u['attr_per_sentence'] for u in users])
    len_vals = np.array([u['avg_sentence_length'] for u in users])
    wpa_vals = np.array([u['words_per_attribute'] for u in users if u['words_per_attribute'] is not None])
    mean_gap_vals = np.array([u['mean_gap'] for u in users if u['mean_gap'] is not None])
    model_wpa_vals = np.array([u['model_words_per_attribute'] for u in users if u['model_words_per_attribute'] is not None])
    eq_len_vals = np.array([u['expected_query_length'] for u in users if u['expected_query_length'] is not None])

    log(f'  attr_sentence_ratio: min={ratio_vals.min():.4f}, max={ratio_vals.max():.4f}, mean={ratio_vals.mean():.4f}')
    log(f'  attr_per_sentence: min={aps_vals.min():.4f}, max={aps_vals.max():.4f}, mean={aps_vals.mean():.4f}')
    log(f'  avg_sentence_length: min={len_vals.min():.1f}, max={len_vals.max():.1f}, mean={len_vals.mean():.1f}')

    if len(wpa_vals) > 0:
        log(f'  words_per_attribute: n={len(wpa_vals)}, min={wpa_vals.min():.1f}, max={wpa_vals.max():.1f}, mean={wpa_vals.mean():.1f}, median={np.median(wpa_vals):.1f}')
    if len(model_wpa_vals) > 0:
        log(f'  model_words_per_attribute: n={len(model_wpa_vals)}, min={model_wpa_vals.min():.2f}, max={model_wpa_vals.max():.2f}, mean={model_wpa_vals.mean():.2f}')
    if len(eq_len_vals) > 0:
        log(f'  expected_query_length: n={len(eq_len_vals)}, min={eq_len_vals.min():.1f}, max={eq_len_vals.max():.1f}, mean={eq_len_vals.mean():.1f}')
    if len(mean_gap_vals) > 0:
        log(f'  mean_gap: n={len(mean_gap_vals)}, min={mean_gap_vals.min():.2f}, max={mean_gap_vals.max():.2f}, mean={mean_gap_vals.mean():.2f}')

    global_cat_counter = Counter()
    for u in users:
        for cat, cnt in u['category_distribution'].items():
            global_cat_counter[cat] += cnt

    total_attrs = sum(global_cat_counter.values())
    log(f'  总属性词数: {total_attrs}')

    return {
        'ratio_vals': ratio_vals,
        'aps_vals': aps_vals,
        'len_vals': len_vals,
        'wpa_vals': wpa_vals,
        'mean_gap_vals': mean_gap_vals,
        'model_wpa_vals': model_wpa_vals,
        'eq_len_vals': eq_len_vals,
        'global_cat_counter': dict(global_cat_counter),
        'total_attrs': total_attrs,
    }


# ============ 主函数 ============
def main():
    start_time = time.time()

    log('=' * 60)
    log(f'ACL/CCOMP/AttrDensity 句法分析（全合并版）- {CATEGORY}')
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
    fp_attr = open(ATTR_DENSITY_OUTPUT_JSONL, 'w', encoding='utf-8')

    acl_users, ccomp_users, attr_users = [], [], []
    acl_sent_count, ccomp_sent_count, attr_sent_count = 0, 0, 0

    for idx, user_data in enumerate(user_list):
        if idx % 500 == 0:
            elapsed = time.time() - start_time
            log(f'进度: {idx}/{total} ({idx/total*100:.1f}%), 耗时: {elapsed:.1f}s')

        result = process_user(user_data)
        acl_res, ccomp_res, attr_res, acl_sents, ccomp_sents, attr_sents = result

        if acl_res is not None:
            user_id = acl_res['user_id']
            user_products = []
            for product in user_data.get('results', []):
                asin = product.get('asin', '')
                if asin in asin_to_attrs:
                    user_products.append({'asin': asin, **asin_to_attrs[asin]})
            acl_res['products'] = user_products
            ccomp_res['products'] = user_products
            attr_res['products'] = user_products

            acl_users.append(acl_res)
            ccomp_users.append(ccomp_res)
            attr_users.append(attr_res)

            for s in acl_sents:
                fp_acl.write(json.dumps(s, ensure_ascii=False) + '\n')
            acl_sent_count += len(acl_sents)

            for s in ccomp_sents:
                fp_ccomp.write(json.dumps(s, ensure_ascii=False) + '\n')
            ccomp_sent_count += len(ccomp_sents)

            for s in attr_sents:
                fp_attr.write(json.dumps(s, ensure_ascii=False) + '\n')
            attr_sent_count += len(attr_sents)

    fp_acl.close()
    fp_ccomp.close()
    fp_attr.close()

    elapsed = time.time() - start_time
    log(f'进度: {total}/{total} (100.0%), 耗时: {elapsed:.1f}s')
    log(f'ACL: {len(acl_users)} 用户, {acl_sent_count} 句子')
    log(f'CCOMP: {len(ccomp_users)} 用户, {ccomp_sent_count} 句子')
    log(f'AttrDensity: {len(attr_users)} 用户, {attr_sent_count} 句子')

    # 打印统计
    log(f'\n--- ACL 统计 ---')
    acl_stats = print_clause_stats(acl_users, 'acl', 'acl_sentence_ratio', 'acl_per_sentence', 'words_per_acl',
                                   'acl_type_distribution', 'marker_distribution', 'mean_gap', 'median_gap')

    log(f'\n--- CCOMP 统计 ---')
    ccomp_stats = print_clause_stats(ccomp_users, 'ccomp', 'ccomp_sentence_ratio', 'ccomp_per_sentence', 'words_per_ccomp',
                                    'ccomp_type_distribution', 'marker_distribution', 'mean_gap', 'median_gap')

    log(f'\n--- AttrDensity 统计 ---')
    attr_stats = print_attr_stats(attr_users)

    total_time = time.time() - start_time

    # 保存 ACL
    log(f'\n保存ACL用户档案到: {ACL_USERS_JSON}')
    with open(ACL_USERS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(acl_users, fp, ensure_ascii=False, indent=2)
    acl_summary = {
        'model': 'en_core_web_sm',
        'total_users': total,
        'valid_users': len(acl_users),
        'total_sentences': acl_sent_count,
        'total_acls': acl_stats['total_count'],
        'acl_sentence_ratio': {'mean': round(float(acl_stats['ratio_vals'].mean()), 4)},
        'acl_per_sentence': {'mean': round(float(acl_stats['pps_vals'].mean()), 4)},
        'avg_sentence_length': {'mean': round(float(acl_stats['len_vals'].mean()), 1)},
        'acl_type_distribution': acl_stats['global_type_counter'],
        'marker_distribution': acl_stats['global_marker_counter'],
        'elapsed_seconds': round(total_time, 1),
    }
    log(f'保存ACL统计到: {ACL_STATS_JSON}')
    with open(ACL_STATS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(acl_summary, fp, ensure_ascii=False, indent=2)

    # 保存 CCOMP
    log(f'\n保存CCOMP用户档案到: {CCOMP_USERS_JSON}')
    with open(CCOMP_USERS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(ccomp_users, fp, ensure_ascii=False, indent=2)
    ccomp_summary = {
        'model': 'en_core_web_sm',
        'total_users': total,
        'valid_users': len(ccomp_users),
        'total_sentences': ccomp_sent_count,
        'total_ccomps': ccomp_stats['total_count'],
        'ccomp_sentence_ratio': {'mean': round(float(ccomp_stats['ratio_vals'].mean()), 4)},
        'ccomp_per_sentence': {'mean': round(float(ccomp_stats['pps_vals'].mean()), 4)},
        'avg_sentence_length': {'mean': round(float(ccomp_stats['len_vals'].mean()), 1)},
        'ccomp_type_distribution': ccomp_stats['global_type_counter'],
        'marker_distribution': ccomp_stats['global_marker_counter'],
        'elapsed_seconds': round(total_time, 1),
    }
    log(f'保存CCOMP统计到: {CCOMP_STATS_JSON}')
    with open(CCOMP_STATS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(ccomp_summary, fp, ensure_ascii=False, indent=2)

    # 保存 AttrDensity
    log(f'\n保存AttrDensity用户档案到: {ATTR_DENSITY_USERS_JSON}')
    with open(ATTR_DENSITY_USERS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(attr_users, fp, ensure_ascii=False, indent=2)
    attr_summary = {
        'model': 'en_core_web_sm',
        'total_users': total,
        'valid_users': len(attr_users),
        'total_sentences': attr_sent_count,
        'total_attrs': attr_stats['total_attrs'],
        'attr_sentence_ratio': {'mean': round(float(attr_stats['ratio_vals'].mean()), 4)},
        'attr_per_sentence': {'mean': round(float(attr_stats['aps_vals'].mean()), 4)},
        'avg_sentence_length': {'mean': round(float(attr_stats['len_vals'].mean()), 1)},
        'category_distribution': attr_stats['global_cat_counter'],
        'elapsed_seconds': round(total_time, 1),
    }
    log(f'保存AttrDensity统计到: {ATTR_DENSITY_STATS_JSON}')
    with open(ATTR_DENSITY_STATS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(attr_summary, fp, ensure_ascii=False, indent=2)

    log(f'\n总耗时: {total_time:.1f}s')


if __name__ == '__main__':
    main()
