#!/usr/bin/env python3
"""relcl区间统计（spaCy模型分析）"""
import json, numpy as np
import warnings
import time
from datetime import datetime
from collections import Counter
warnings.filterwarnings('ignore')
import spacy
nlp = spacy.load('en_core_web_sm')

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

OUTPUT_JSONL = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/relcl_sentences.jsonl'
STATS_JSON = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/relcl_stats.json'
USERS_JSON = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/relcl_user_profiles.json'

def count_tokens(text):
    """快速分词计数"""
    return len([t for t in text.split() if t.strip()])

def get_relcl_type(relcl_head):
    """分析relcl的句法形式

    Returns: (pronoun, gram_role) 元组
    - pronoun: 'that', 'which', 'who', 'whose', 'where', 'what', 'other', 'null'
    - gram_role: 'nsubj', 'dobj', 'pobj', 'poss', 'other', 'unknown'
    """
    # 关系代词列表
    rel_pronouns = {'that', 'which', 'who', 'whom', 'whose', 'what'}

    # 1. 检查relcl token本身是否是关系代词
    head_lower = relcl_head.text.lower()
    if head_lower in rel_pronouns:
        return head_lower, relcl_head.dep_

    # 2. 查找relcl head的所有子节点的文本，检查是否包含关系代词
    for child in relcl_head.children:
        child_lower = child.text.lower()
        if child_lower in rel_pronouns:
            return child_lower, child.dep_

    # 3. 检查nsubj（省略关系代词的情况，如 "the book I bought"）
    for child in relcl_head.children:
        if child.dep_ == 'nsubj':
            return 'null', 'nsubj_ellipsis'

    # 4. 检查其他可能的角色：dobj, pobj, attr, oprd 等
    for child in relcl_head.children:
        dep = child.dep_
        if dep in ('dobj', 'pobj', 'attr', 'oprd', 'nsubjpass', 'dative'):
            return 'null', dep

    # 5. 检查advmod, mark等修饰词
    for child in relcl_head.children:
        if child.dep_ in ('advmod', 'mark'):
            return 'null', 'ellipsis'

    # 6. 尝试查找wh-开头的词
    for child in relcl_head.children:
        child_lower = child.text.lower()
        if child_lower.startswith(('wh', 'how')):
            return 'other', child.dep_

    # 7. 如果head本身有文本且像关系词
    if head_lower.startswith(('that', 'which', 'who', 'what', 'where', 'how')):
        return 'other', relcl_head.dep_

    # 8. 递归检查子节点的子节点（更深层解析）
    for child in relcl_head.children:
        for subchild in child.children:
            sub_lower = subchild.text.lower()
            if sub_lower in rel_pronouns:
                return sub_lower, subchild.dep_
            if sub_lower.startswith(('wh', 'how')):
                return 'other', subchild.dep_

    # 9. 如果有任意子节点，至少返回pronoun='null'和最可能的角色
    if len(list(relcl_head.children)) > 0:
        # 取第一个子节点的dep作为gram_role
        first_child = list(relcl_head.children)[0]
        return 'null', first_child.dep_

    # 10. spaCy标记为relcl但完全无法解析的情况
    # 使用token本身的dep_作为fallback
    return 'null', relcl_head.dep_

def analyze_relcl_in_doc(doc):
    """分析文档中所有relcl的句法类型

    Returns: list of (pronoun, gram_role, head_text) 元组
    """
    results = []
    for token in doc:
        if token.dep_ == 'relcl':
            pronoun, gram_role = get_relcl_type(token)
            results.append({
                'pronoun': pronoun,
                'gram_role': gram_role,
                'head_word': token.text,
                'head_pos': token.pos_
            })
    return results

def get_freq_label(ratio):
    """relcl使用频率标签"""
    if ratio < 0.1:
        return 'low'
    elif ratio < 0.5:
        return 'medium'
    else:
        return 'high'

def get_density_label(pps):
    """relcl使用密度标签"""
    if pps < 0.1:
        return 'simple'
    elif pps < 0.5:
        return 'medium'
    else:
        return 'complex'

def get_length_label(avg_len):
    """句子长度标签"""
    if avg_len < 22:
        return 'short'
    elif avg_len < 28:
        return 'medium'
    else:
        return 'long'

def process_user(user_data, idx, total):
    reviews = []
    for p in user_data.get('results', []):
        reviews.extend(p.get('target_reviews', []))

    user_id = user_data['user_id']

    # 筛选15-35词的句子
    valid_sentences = []
    for r in reviews:
        if not r:
            continue
        n = count_tokens(r)
        if 15 <= n <= 35:
            valid_sentences.append(r)

    if not valid_sentences:
        return None, []

    # 只对有效句子进行spaCy解析
    total_sentences = 0
    sentences_with_relcl = 0
    total_relcl_count = 0
    total_token_count = 0
    sentence_details = []
    relcl_type_counter = Counter()  # 统计用户使用的relcl类型

    for r in valid_sentences:
        doc = nlp(r)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        n = len(tokens)
        if n == 0:
            continue

        # 分析relcl类型
        relcl_info = analyze_relcl_in_doc(doc)
        relcl_count = len(relcl_info)

        total_sentences += 1
        total_relcl_count += relcl_count
        total_token_count += n
        if relcl_count > 0:
            sentences_with_relcl += 1

        # 统计relcl类型
        for info in relcl_info:
            key = f"{info['pronoun']}_{info['gram_role']}"
            relcl_type_counter[key] += 1

        sentence_details.append({
            'user_id': user_id,
            'sentence': r,
            'token_count': n,
            'relcl_count': relcl_count,
            'has_relcl': relcl_count > 0,
            'relcl_types': relcl_info
        })

    if total_sentences == 0:
        return None, []

    # 四个指标
    relcl_sentence_ratio = sentences_with_relcl / total_sentences
    relcl_per_sentence = total_relcl_count / total_sentences
    avg_sentence_length = total_token_count / total_sentences
    # 每多少个有效单词使用一个relcl（如果relcl_count > 0）
    words_per_relcl = total_token_count / total_relcl_count if total_relcl_count > 0 else None

    # relcl类型统计（转换为dict）
    relcl_type_dist = dict(relcl_type_counter)

    return {
        'user_id': user_id,
        'relcl_sentence_ratio': relcl_sentence_ratio,
        'relcl_per_sentence': relcl_per_sentence,
        'avg_sentence_length': avg_sentence_length,
        'words_per_relcl': words_per_relcl,
        'freq_label': get_freq_label(relcl_sentence_ratio),
        'density_label': get_density_label(relcl_per_sentence),
        'length_label': get_length_label(avg_sentence_length),
        'total_sentences': total_sentences,
        'total_relcl_count': total_relcl_count,
        'relcl_type_distribution': relcl_type_dist
    }, sentence_details

def main():
    start_time = time.time()
    log('处理用户数据...')

    # 从单个JSON文件加载所有用户
    ALL_USERS_FILE = '/fs04/ar57/wenyu/result/personal_query/01_preference_extraction/stage1_filtered_users_reviews.json'
    log(f'加载用户数据: {ALL_USERS_FILE}')
    with open(ALL_USERS_FILE, 'r', encoding='utf-8') as f:
        all_users_data = json.load(f)
    user_list = all_users_data['users']
    total = len(user_list)
    log(f'总用户数: {total}')

    # 加载Stage 1商品属性
    STAGE1_ATTR_FILE = '/fs04/ar57/wenyu/result/personal_query/01_preference_extraction/attributes_Arts_Crafts_and_Sewing.json'
    log(f'加载Stage 1属性数据: {STAGE1_ATTR_FILE}')
    with open(STAGE1_ATTR_FILE, 'r', encoding='utf-8') as f:
        stage1_data = json.load(f)
    # 构建asin -> 属性映射
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

    # 清空并打开输出文件（流式写入）
    fp_out = open(OUTPUT_JSONL, 'w', encoding='utf-8')

    users = []
    sentence_count = 0
    for idx, user_data in enumerate(user_list):
        if idx % 500 == 0:
            elapsed = time.time() - start_time
            log(f'进度: {idx}/{total} ({idx/total*100:.1f}%), 耗时: {elapsed:.1f}s')
        result, sentences = process_user(user_data, idx, total)
        if result is not None:
            # 添加用户的商品属性信息
            user_id = result['user_id']
            user_products = []
            for product in user_data.get('results', []):
                asin = product.get('asin', '')
                if asin in asin_to_attrs:
                    user_products.append({
                        'asin': asin,
                        **asin_to_attrs[asin]
                    })
            result['products'] = user_products
            users.append(result)
            # 流式写入每个句子
            for s in sentences:
                fp_out.write(json.dumps(s, ensure_ascii=False) + '\n')
            sentence_count += len(sentences)

    fp_out.close()

    elapsed = time.time() - start_time
    log(f'进度: {total}/{total} (100.0%), 耗时: {elapsed:.1f}s')
    log(f'有效用户数: {len(users)}')
    log(f'有效句子数: {sentence_count}')

    log(f'保存句子结果到: {OUTPUT_JSONL}')

    # 统计各指标
    ratio_vals = np.array([u['relcl_sentence_ratio'] for u in users])
    pps_vals = np.array([u['relcl_per_sentence'] for u in users])
    len_vals = np.array([u['avg_sentence_length'] for u in users])
    tpr_vals = np.array([u['words_per_relcl'] for u in users if u['words_per_relcl'] is not None])

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

    # 统计标签分布
    def count_labels(key):
        labels = [u[key] for u in users]
        cnt = Counter(labels)
        total_u = len(users)
        result = []
        for label in ['low', 'medium', 'high', 'simple', 'complex', 'short', 'medium', 'long']:
            if label in cnt:
                result.append({'label': label, 'count': cnt[label], 'percentage': round(cnt[label]/total_u*100, 2)})
        return result

    log('')
    log('=' * 60)
    log('relcl_sentence_ratio 分布（有relcl的句子占比）')
    log('=' * 60)
    log(f'{"区间":<20} │ {"人数":>6} │ {"占比":>8}')
    log('─' * 60)
    dist_ratio = compute_distribution(ratio_vals)
    for d in dist_ratio:
        if d['count'] > 0:
            bar = '█' * int(d['percentage'] / 2)
            log(f'{d["interval"]:<20} │ {d["count"]:>6} │ {d["percentage"]:>7.1f}% │ {bar}')
    log(f'最小值: {ratio_vals.min():.4f}, 最大值: {ratio_vals.max():.4f}, 均值: {ratio_vals.mean():.4f}')

    log('')
    log('=' * 60)
    log('relcl_per_sentence 分布（平均每句relcl数）')
    log('=' * 60)
    log(f'{"区间":<20} │ {"人数":>6} │ {"占比":>8}')
    log('─' * 60)
    dist_pps = compute_distribution(pps_vals, max_val_override=0.3)
    for d in dist_pps:
        if d['count'] > 0:
            bar = '█' * int(d['percentage'] / 2)
            log(f'{d["interval"]:<20} │ {d["count"]:>6} │ {d["percentage"]:>7.1f}% │ {bar}')
    log(f'最小值: {pps_vals.min():.4f}, 最大值: {pps_vals.max():.4f}, 均值: {pps_vals.mean():.4f}')

    log('')
    log('=' * 60)
    log('avg_sentence_length 分布（平均句子长度）')
    log('=' * 60)
    log(f'最小值: {len_vals.min():.1f}, 最大值: {len_vals.max():.1f}, 均值: {len_vals.mean():.1f}')

    log('')
    log('=' * 60)
    log('words_per_relcl 分布（每多少有效单词使用一个relcl）')
    log('=' * 60)
    log(f'有效用户数（含relcl）: {len(tpr_vals)}')
    log(f'最小值: {tpr_vals.min():.1f}, 最大值: {tpr_vals.max():.1f}, 均值: {tpr_vals.mean():.1f}')
    log(f'中位数: {np.median(tpr_vals):.1f}')

    # words_per_relcl 区间分布（每5为区间）
    wpr_intervals = [
        (0, 5, '0-5'), (5, 10, '5-10'), (10, 15, '10-15'), (15, 20, '15-20'),
        (20, 25, '20-25'), (25, 30, '25-30'), (30, 35, '30-35'),
        (35, 40, '35-40'), (40, 45, '40-45'), (45, 50, '45-50'), (50, float('inf'), '50+')
    ]
    log(f'{"区间":<15} │ {"人数":>6} │ {"占比":>8}')
    log('─' * 40)
    wpr_dist = []
    for low, high, label in wpr_intervals:
        if high == float('inf'):
            count = int(np.sum(tpr_vals >= low))
        else:
            count = int(np.sum((tpr_vals >= low) & (tpr_vals < high)))
        pct = count / len(tpr_vals) * 100 if len(tpr_vals) > 0 else 0
        bar = '█' * int(pct)
        log(f'{label:<15} │ {count:>6} │ {pct:>7.2f}% │ {bar}')
        wpr_dist.append({'interval': label, 'count': count, 'percentage': round(pct, 2)})

    # 标签分布
    log('')
    log('=' * 60)
    log('用户写作特征标签分布')
    log('=' * 60)

    log('relcl使用频率 (freq_label):')
    for item in count_labels('freq_label'):
        bar = '█' * int(item['percentage'] / 3)
        log(f'  {item["label"]:<10} │ {item["count"]:>5} │ {item["percentage"]:>6.1f}% │ {bar}')

    log('relcl使用密度 (density_label):')
    for item in count_labels('density_label'):
        bar = '█' * int(item['percentage'] / 3)
        log(f'  {item["label"]:<10} │ {item["count"]:>5} │ {item["percentage"]:>6.1f}% │ {bar}')

    log('句子长度 (length_label):')
    for item in count_labels('length_label'):
        bar = '█' * int(item['percentage'] / 3)
        log(f'  {item["label"]:<10} │ {item["count"]:>5} │ {item["percentage"]:>6.1f}% │ {bar}')

    # 组合标签分布
    log('')
    log('=' * 60)
    log('组合标签分布 (freq+density+length)')
    log('=' * 60)
    combo_labels = [f'{u["freq_label"]}_{u["density_label"]}_{u["length_label"]}' for u in users]
    combo_cnt = Counter(combo_labels)
    total_u = len(users)
    for combo, count in sorted(combo_cnt.items(), key=lambda x: -x[1])[:15]:
        pct = count / total_u * 100
        bar = '█' * int(pct / 2)
        log(f'{combo:<25} │ {count:>5} │ {pct:>6.1f}% │ {bar}')

    # relcl句法类型统计
    log('')
    log('=' * 60)
    log('relcl句法类型分布（全局）')
    log('=' * 60)

    # 汇总所有用户的relcl类型
    global_type_counter = Counter()
    for u in users:
        for typ, cnt in u['relcl_type_distribution'].items():
            global_type_counter[typ] += cnt

    total_relcls = sum(global_type_counter.values())
    log(f'总relcl数: {total_relcls}')
    log(f'{"类型":<30} │ {"数量":>6} │ {"占比":>8}')
    log('─' * 60)
    for typ, count in sorted(global_type_counter.items(), key=lambda x: -x[1]):
        pct = count / total_relcls * 100
        bar = '█' * int(pct / 2)
        log(f'{typ:<30} │ {count:>6} │ {pct:>7.1f}% │ {bar}')

    total_time = time.time() - start_time

    # 保存用户档案
    log(f'\n保存用户档案到: {USERS_JSON}')
    with open(USERS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(users, fp, ensure_ascii=False, indent=2)

    summary_stats = {
        'model': 'en_core_web_sm',
        'total_users': total,
        'valid_users': len(users),
        'total_sentences': sentence_count,
        'total_relcls': total_relcls,
        'relcl_sentence_ratio': {
            'min': round(float(ratio_vals.min()), 4),
            'max': round(float(ratio_vals.max()), 4),
            'mean': round(float(ratio_vals.mean()), 4)
        },
        'relcl_per_sentence': {
            'min': round(float(pps_vals.min()), 4),
            'max': round(float(pps_vals.max()), 4),
            'mean': round(float(pps_vals.mean()), 4)
        },
        'avg_sentence_length': {
            'min': round(float(len_vals.min()), 1),
            'max': round(float(len_vals.max()), 1),
            'mean': round(float(len_vals.mean()), 1)
        },
        'words_per_relcl': {
            'min': round(float(tpr_vals.min()), 1),
            'max': round(float(tpr_vals.max()), 1),
            'mean': round(float(tpr_vals.mean()), 1),
            'median': round(float(np.median(tpr_vals)), 1)
        },
        'relcl_type_distribution': dict(global_type_counter),
        'elapsed_seconds': round(total_time, 1),
        'distribution_ratio': dist_ratio,
        'distribution_pps': dist_pps,
        'distribution_wpr': wpr_dist
    }

    log(f'\n总耗时: {total_time:.1f}s')

    # 保存统计结果到JSON
    log(f'保存统计结果到: {STATS_JSON}')
    with open(STATS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(summary_stats, fp, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()