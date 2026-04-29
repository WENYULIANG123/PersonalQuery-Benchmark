#!/usr/bin/env python3
"""ACL/CCOMP/AttrDensity 句法分析（spaCy模型分析）- Pet_Supplies

同时计算：
- ACL (Adjectival Clause) - 形容词性从句
- CCOMP (Complement Clause) - 补语从句
- AttrDensity (Attribute Density) - 属性词密度

同时输出 user_profiles.json 和 sentence-level JSONL 文件
"""
import json, numpy as np
import warnings
import time
import os
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
CATEGORY = "Pet_Supplies"
BASE_DIR = f'/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/{CATEGORY}'
os.makedirs(BASE_DIR, exist_ok=True)

ACL_USERS_JSON = f'{BASE_DIR}/acl_user_profiles.json'
CCOMP_USERS_JSON = f'{BASE_DIR}/ccomp_user_profiles.json'
ATTR_DENSITY_USERS_JSON = f'{BASE_DIR}/attr_density_user_profiles.json'
ACL_OUTPUT_JSONL = f'{BASE_DIR}/acl_sentences.jsonl'
CCOMP_OUTPUT_JSONL = f'{BASE_DIR}/ccomp_sentences.jsonl'

ALL_USERS_FILE = f'/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/{CATEGORY}/stage1_filtered_users_reviews.json'
STAGE1_ATTR_FILE = f'/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/{CATEGORY}/attributes_{CATEGORY}.json'

# ============ ACL 分析（句法树变宽特征）============
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

        if acl_info:
            results.append(acl_info)
    return results


# ============ CCOMP 分析 ============
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
        return None, None, None

    # ACL
    total_sentences = 0
    sentences_with_acl = 0
    total_acl_count = 0
    total_token_count = 0
    acl_type_counter = Counter()
    acl_marker_counter = Counter()
    acl_sentence_details = []

    # CCOMP
    sentences_with_ccomp = 0
    total_ccomp_count = 0
    ccomp_type_counter = Counter()
    ccomp_marker_counter = Counter()
    ccomp_sentence_details = []

    # AttrDensity
    sentences_with_attr = 0
    total_attr_count = 0
    attr_category_counter = Counter()
    regression_data = []

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
        acl_sentence_details.append({
            'user_id': user_id,
            'sentence': r,
            'token_count': n,
            'acl_count': acl_count,
            'has_acl': acl_count > 0,
            'acl_info': acl_info,
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
        ccomp_sentence_details.append({
            'user_id': user_id,
            'sentence': r,
            'token_count': n,
            'ccomp_count': ccomp_count,
            'has_ccomp': ccomp_count > 0,
            'ccomp_info': ccomp_info,
        })

        # AttrDensity
        attr_count, attr_cat_dist = count_attributes_in_sentence_spacy(doc)
        total_attr_count += attr_count
        if attr_count > 0:
            sentences_with_attr += 1
        for cat, cnt in attr_cat_dist.items():
            attr_category_counter[cat] += cnt
        regression_data.append((attr_count, n))

    if total_sentences == 0:
        return None, None, None, [], []

    # ACL 指标
    acl_sentence_ratio = sentences_with_acl / total_sentences
    acl_per_sentence = total_acl_count / total_sentences
    words_per_acl = total_token_count / total_acl_count if total_acl_count > 0 else None
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
    }

    # CCOMP 指标
    ccomp_sentence_ratio = sentences_with_ccomp / total_sentences
    ccomp_per_sentence = total_ccomp_count / total_sentences
    words_per_ccomp = total_token_count / total_ccomp_count if total_ccomp_count > 0 else None
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
    }

    # AttrDensity 指标
    attr_sentence_ratio = sentences_with_attr / total_sentences
    attr_per_sentence = total_attr_count / total_sentences
    words_per_attr = total_token_count / total_attr_count if total_attr_count > 0 else None

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
        'model_words_per_attribute': model_wpa,
        'model_base_overhead': model_overhead,
        'model_r2': model_r2,
    }

    return acl_result, ccomp_result, attr_result, acl_sentence_details, ccomp_sentence_details


# ============ 主函数 ============
def main():
    start_time = time.time()

    log('=' * 60)
    log(f'ACL/CCOMP/AttrDensity 句法分析（简化版）- {CATEGORY}')
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

    acl_users, ccomp_users, attr_users = [], [], []
    acl_sentence_count = 0
    ccomp_sentence_count = 0

    fp_acl = open(ACL_OUTPUT_JSONL, 'w', encoding='utf-8')
    fp_ccomp = open(CCOMP_OUTPUT_JSONL, 'w', encoding='utf-8')

    try:
        for idx, user_data in enumerate(user_list):
            if idx % 500 == 0:
                elapsed = time.time() - start_time
                log(f'进度: {idx}/{total} ({idx/total*100:.1f}%), 耗时: {elapsed:.1f}s')

            result = process_user(user_data)
            acl_res, ccomp_res, attr_res, acl_sentence_details, ccomp_sentence_details = result

            for sentence_detail in acl_sentence_details:
                fp_acl.write(json.dumps(sentence_detail, ensure_ascii=False) + '\n')
                acl_sentence_count += 1
            for sentence_detail in ccomp_sentence_details:
                fp_ccomp.write(json.dumps(sentence_detail, ensure_ascii=False) + '\n')
                ccomp_sentence_count += 1

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
    finally:
        fp_acl.close()
        fp_ccomp.close()

    elapsed = time.time() - start_time
    log(f'进度: {total}/{total} (100.0%), 耗时: {elapsed:.1f}s')
    log(f'ACL: {len(acl_users)} 用户')
    log(f'CCOMP: {len(ccomp_users)} 用户')
    log(f'AttrDensity: {len(attr_users)} 用户')
    log(f'ACL sentence rows: {acl_sentence_count} -> {ACL_OUTPUT_JSONL}')
    log(f'CCOMP sentence rows: {ccomp_sentence_count} -> {CCOMP_OUTPUT_JSONL}')

    total_time = time.time() - start_time

    # 保存 ACL
    log(f'\n保存ACL用户档案到: {ACL_USERS_JSON}')
    with open(ACL_USERS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(acl_users, fp, ensure_ascii=False, indent=2)

    # 保存 CCOMP
    log(f'保存CCOMP用户档案到: {CCOMP_USERS_JSON}')
    with open(CCOMP_USERS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(ccomp_users, fp, ensure_ascii=False, indent=2)

    # 保存 AttrDensity
    log(f'保存AttrDensity用户档案到: {ATTR_DENSITY_USERS_JSON}')
    with open(ATTR_DENSITY_USERS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(attr_users, fp, ensure_ascii=False, indent=2)

    log(f'\n总耗时: {total_time:.1f}s')


if __name__ == '__main__':
    main()
