#!/usr/bin/env python3
"""补语从句特征区间统计（spaCy模型分析）ccomp/xcomp"""
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

OUTPUT_JSONL = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/ccomp_sentences.jsonl'
STATS_JSON = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/ccomp_stats.json'
USERS_JSON = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/ccomp_user_profiles.json'

# 补语从句引导词
CCOMP_MARKERS = {'that', 'whether', 'if', 'what', 'which', 'who', 'whom', 'whose'}

def count_tokens(text):
    """快速分词计数"""
    return len([t for t in text.split() if t.strip()])

def compute_ccomp_gaps(doc, ccomp_info):
    """计算一个句子中相邻ccomp之间的非ccomp词数量

    Returns:
        list: 相邻ccomp之间的间隔列表（单词数）
    """
    if len(ccomp_info) < 2:
        return []

    # 获取所有ccomp的位置
    ccomp_positions = [info['position'] for info in ccomp_info]

    # 计算相邻ccomp之间的间隔
    gaps = []
    for i in range(len(ccomp_positions) - 1):
        gap = ccomp_positions[i + 1] - ccomp_positions[i] - 1
        gaps.append(gap)

    return gaps

def analyze_ccomp_in_doc(doc):
    """分析文档中所有补语从句（ccomp/xcomp）

    ccomp: 从句补语（及物动词后的that从句）- He said that...
    xcomp: 开放从句补语（to+动词结构）- She decided to go

    Returns: list of dict with ccomp/xcomp info
    """
    results = []

    for token in doc:
        comp_info = None

        # 1. spaCy的dep_='ccomp'补语从句
        if token.dep_ == 'ccomp':
            # 查找引导词（mark或comp）
            marker_word = None
            marker_dep = None
            complementizer = None  # that, whether, if等

            for child in token.children:
                if child.dep_ == 'mark':
                    marker_word = child.text.lower()
                    marker_dep = 'mark'
                    break

            # 检查是否是that引导的省略句（没有mark的情况）
            if marker_word is None:
                # 查找comp标记
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

        # 2. spaCy的dep_='xcomp'开放从句补语
        elif token.dep_ == 'xcomp':
            # xcomp通常是 to + verb 结构
            marker_word = None
            verb_infinitive = token.text

            # 查找to标记
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
                'verb_word': verb_infinitive,
                'verb_pos': token.pos_,
                'position': token.i
            }

        # 3. 检查mark依存的that/whether/if（某些从句结构）
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

def get_freq_label(ratio):
    """补语从句使用频率标签"""
    if ratio < 0.05:
        return 'low'
    elif ratio < 0.15:
        return 'medium'
    else:
        return 'high'

def get_density_label(pps):
    """补语从句使用密度标签"""
    if pps < 0.05:
        return 'simple'
    elif pps < 0.15:
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

    if not reviews:
        return None, []

    # 直接对每条评论进行spaCy解析，不拆句子不过滤长度
    total_sentences = 0
    sentences_with_ccomp = 0
    total_ccomp_count = 0
    total_token_count = 0
    sentence_details = []
    ccomp_type_counter = Counter()  # 统计用户使用的补语从句类型
    marker_counter = Counter()  # 统计引导词使用
    all_gaps = []  # 收集所有间隔

    for r in reviews:
        if not r:
            continue
        doc = nlp(r)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        n = len(tokens)
        if n == 0:
            continue

        # 分析补语从句类型
        ccomp_info = analyze_ccomp_in_doc(doc)
        ccomp_count = len(ccomp_info)

        total_sentences += 1
        total_ccomp_count += ccomp_count
        total_token_count += n
        if ccomp_count > 0:
            sentences_with_ccomp += 1

        # 统计补语从句类型
        for info in ccomp_info:
            key = info['comp_type']
            ccomp_type_counter[key] += 1
            # 统计引导词
            if info.get('marker'):
                marker_counter[info['marker']] += 1

        # 计算ccomp间隔（每写一个ccomp前平均隔多少个非ccomp词）
        gaps = compute_ccomp_gaps(doc, ccomp_info)
        all_gaps.extend(gaps)

        sentence_details.append({
            'user_id': user_id,
            'sentence': r,
            'token_count': n,
            'ccomp_count': ccomp_count,
            'has_ccomp': ccomp_count > 0,
            'ccomp_info': ccomp_info
        })

    if total_sentences == 0:
        return None, []

    # 四个指标
    ccomp_sentence_ratio = sentences_with_ccomp / total_sentences
    ccomp_per_sentence = total_ccomp_count / total_sentences
    avg_sentence_length = total_token_count / total_sentences
    # 每多少个有效单词使用一个补语从句（如果ccomp_count > 0）
    words_per_ccomp = total_token_count / total_ccomp_count if total_ccomp_count > 0 else None

    # 补语从句类型统计（转换为dict）
    ccomp_type_dist = dict(ccomp_type_counter)
    marker_dist = dict(marker_counter)

    # 计算ccomp间隔（每写一个ccomp前平均隔多少个非ccomp词）
    mean_gap = float(np.mean(all_gaps)) if all_gaps else None
    median_gap = float(np.median(all_gaps)) if all_gaps else None

    return {
        'user_id': user_id,
        'ccomp_sentence_ratio': ccomp_sentence_ratio,
        'ccomp_per_sentence': ccomp_per_sentence,
        'avg_sentence_length': avg_sentence_length,
        'words_per_ccomp': words_per_ccomp,
        'freq_label': get_freq_label(ccomp_sentence_ratio),
        'density_label': get_density_label(ccomp_per_sentence),
        'length_label': get_length_label(avg_sentence_length),
        'total_sentences': total_sentences,
        'total_ccomp_count': total_ccomp_count,
        'ccomp_type_distribution': ccomp_type_dist,
        'marker_distribution': marker_dist,
        'mean_gap': mean_gap,
        'median_gap': median_gap
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
    ratio_vals = np.array([u['ccomp_sentence_ratio'] for u in users])
    pps_vals = np.array([u['ccomp_per_sentence'] for u in users])
    len_vals = np.array([u['avg_sentence_length'] for u in users])
    tpr_vals = np.array([u['words_per_ccomp'] for u in users if u['words_per_ccomp'] is not None])
    mean_gap_vals = np.array([u['mean_gap'] for u in users if u['mean_gap'] is not None])
    median_gap_vals = np.array([u['median_gap'] for u in users if u['median_gap'] is not None])

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
    log('ccomp_sentence_ratio 分布（有补语从句的句子占比）')
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
    log('ccomp_per_sentence 分布（平均每句补语从句数）')
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
    log('words_per_ccomp 分布（每多少有效单词使用一个补语从句）')
    log('=' * 60)
    log(f'有效用户数（含补语从句）: {len(tpr_vals)}')
    if len(tpr_vals) > 0:
        log(f'最小值: {tpr_vals.min():.1f}, 最大值: {tpr_vals.max():.1f}, 均值: {tpr_vals.mean():.1f}')
        log(f'中位数: {np.median(tpr_vals):.1f}')

        # words_per_ccomp 区间分布（每5为区间）
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
    else:
        wpr_dist = []

    # ccomp间隔统计（每写一个ccomp前平均隔多少个非ccomp词）
    log('')
    log('=' * 60)
    log('ccomp间隔统计（每写一个ccomp前平均隔多少个非ccomp词）')
    log('=' * 60)
    log(f'有效用户数（含>=2个ccomp）: {len(mean_gap_vals)}')
    if len(mean_gap_vals) > 0:
        log(f'mean_gap:')
        log(f'  最小值: {mean_gap_vals.min():.2f}, 最大值: {mean_gap_vals.max():.2f}')
        log(f'  均值: {mean_gap_vals.mean():.2f}, 中位数: {np.median(mean_gap_vals):.2f}')
        log(f'median_gap:')
        log(f'  最小值: {median_gap_vals.min():.2f}, 最大值: {median_gap_vals.max():.2f}')
        log(f'  均值: {median_gap_vals.mean():.2f}, 中位数: {np.median(median_gap_vals):.2f}')

        # 按每5个单词区间统计
        log(f'{"区间":<15} │ {"用户数":>8} │ {"占比":>8}')
        log('─' * 45)
        gap_intervals = [
            (0, 5, '0-5'), (5, 10, '5-10'), (10, 15, '10-15'),
            (15, 20, '15-20'), (20, 25, '20-25'), (25, 30, '25-30'),
            (30, float('inf'), '30+')
        ]
        gap_dist = []
        for low, high, label in gap_intervals:
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

    # 标签分布
    log('')
    log('=' * 60)
    log('用户写作特征标签分布')
    log('=' * 60)

    log('补语从句使用频率 (freq_label):')
    for item in count_labels('freq_label'):
        bar = '█' * int(item['percentage'] / 3)
        log(f'  {item["label"]:<10} │ {item["count"]:>5} │ {item["percentage"]:>6.1f}% │ {bar}')

    log('补语从句使用密度 (density_label):')
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

    # 补语从句类型统计
    log('')
    log('=' * 60)
    log('补语从句类型分布（全局）')
    log('=' * 60)

    # 汇总所有用户的补语从句类型
    global_type_counter = Counter()
    global_marker_counter = Counter()
    for u in users:
        for typ, cnt in u['ccomp_type_distribution'].items():
            global_type_counter[typ] += cnt
        for marker, cnt in u.get('marker_distribution', {}).items():
            global_marker_counter[marker] += cnt

    total_ccomps = sum(global_type_counter.values())
    log(f'总补语从句数: {total_ccomps}')
    log(f'{"类型":<30} │ {"数量":>6} │ {"占比":>8}')
    log('─' * 60)
    for typ, count in sorted(global_type_counter.items(), key=lambda x: -x[1]):
        pct = count / total_ccomps * 100 if total_ccomps > 0 else 0
        bar = '█' * int(pct / 2)
        log(f'{typ:<30} │ {count:>6} │ {pct:>7.1f}% │ {bar}')

    # 引导词分布
    log('')
    log('=' * 60)
    log('补语从句引导词分布（全局）')
    log('=' * 60)
    total_markers = sum(global_marker_counter.values())
    log(f'总引导词数: {total_markers}')
    log(f'{"引导词":<15} │ {"数量":>6} │ {"占比":>8}')
    log('─' * 40)
    for marker, count in sorted(global_marker_counter.items(), key=lambda x: -x[1])[:20]:
        pct = count / total_markers * 100 if total_markers > 0 else 0
        bar = '█' * int(pct / 2)
        log(f'{marker:<15} │ {count:>6} │ {pct:>7.1f}% │ {bar}')

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
        'total_ccomps': total_ccomps,
        'ccomp_sentence_ratio': {
            'min': round(float(ratio_vals.min()), 4),
            'max': round(float(ratio_vals.max()), 4),
            'mean': round(float(ratio_vals.mean()), 4)
        },
        'ccomp_per_sentence': {
            'min': round(float(pps_vals.min()), 4),
            'max': round(float(pps_vals.max()), 4),
            'mean': round(float(pps_vals.mean()), 4)
        },
        'avg_sentence_length': {
            'min': round(float(len_vals.min()), 1),
            'max': round(float(len_vals.max()), 1),
            'mean': round(float(len_vals.mean()), 1)
        },
        'words_per_ccomp': {
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
        'ccomp_type_distribution': dict(global_type_counter),
        'marker_distribution': dict(global_marker_counter),
        'elapsed_seconds': round(total_time, 1),
        'distribution_ratio': dist_ratio,
        'distribution_pps': dist_pps,
        'distribution_wpr': wpr_dist,
        'distribution_gap': gap_dist
    }

    log(f'\n总耗时: {total_time:.1f}s')

    # 保存统计结果到JSON
    log(f'保存统计结果到: {STATS_JSON}')
    with open(STATS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(summary_stats, fp, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
