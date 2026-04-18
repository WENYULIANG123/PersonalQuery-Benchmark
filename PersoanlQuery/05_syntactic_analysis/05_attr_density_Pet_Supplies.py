#!/usr/bin/env python3
"""属性提及密度分析（spaCy模型分析）
统计每个句子中属性词（颜色、材质、功能等）的提及次数，
计算 words_per_attribute = 句子总词数 / 属性提及数
"""
import json, numpy as np
import warnings
import os
import time
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

OUTPUT_JSONL = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/Pet_Supplies/attr_density_sentences.jsonl'
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
STATS_JSON = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/Pet_Supplies/attr_density_stats.json'
USERS_JSON = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/Pet_Supplies/attr_density_user_profiles.json'

# 属性词典：颜色、材质、功能、外观等
ATTRIBUTE_DICTIONARY = {
    # 颜色类
    'color': {'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white', 'gray', 'grey',
              'brown', 'navy', 'gold', 'silver', 'bronze', 'copper', 'beige', 'cream', 'ivory', 'tan', 'teal',
              'maroon', 'burgundy', 'coral', 'lavender', 'violet', 'indigo', 'colorful', 'bright', 'dark', 'light',
              'vibrant', 'pastel', 'neon', 'solid', 'striped', 'patterned', 'multicolored', 'hue', 'shade', 'tone'},

    # 材质类
    'material': {'plastic', 'metal', 'wood', 'wooden', 'paper', 'cardboard', 'fabric', 'cotton', 'polyester',
                 'silk', 'leather', 'rubber', 'glass', 'ceramic', 'porcelain', 'stone', 'marble', 'bamboo',
                 'steel', 'aluminum', 'copper', 'brass', 'iron', 'silver', 'gold', 'carbon', 'fiberglass',
                 'nylon', 'canvas', 'linen', 'velvet', 'wool', 'acrylic', 'vinyl', 'mesh', 'foam', 'rubber',
                 'resin', 'acrylic', 'polymer', 'composite', 'natural', 'synthetic', 'eco-friendly', 'recycled'},

    # 功能类
    'function': {'easy', 'simple', 'convenient', 'practical', 'functional', 'versatile', 'multi-purpose',
                'portable', 'compact', 'lightweight', 'heavy-duty', 'professional', 'beginner-friendly',
                'automatic', 'manual', 'rechargeable', 'battery-operated', 'cordless', 'electric', 'digital',
                'adjustable', 'foldable', 'detachable', 'removable', 'washable', 'waterproof', 'water-resistant',
                'heat-resistant', 'fireproof', 'scratch-resistant', 'stain-resistant', 'fade-resistant',
                'rust-proof', 'shockproof', 'slip-proof', 'non-slip', 'anti-slip', 'self-adhesive'},

    # 外观/设计类
    'appearance': {'beautiful', 'elegant', 'stylish', 'modern', 'classic', 'vintage', 'rustic', 'minimalist',
                  'decorative', 'ornate', 'sleek', 'smooth', 'textured', 'glossy', 'matte', 'shiny',
                  'transparent', 'opaque', 'translucent', 'printed', 'embossed', 'laser-cut', 'handmade',
                  'craftsman', 'artisan', 'delicate', 'intricate', 'simple', 'plain', 'fancy', 'exquisite'},

    # 尺寸类
    'size': {'small', 'medium', 'large', 'tiny', 'mini', 'giant', 'big', 'compact', 'oversized', 'full-size',
            'standard', 'massive', 'petite', 'slim', 'thick', 'thin', 'long', 'short', 'tall', 'wide', 'narrow'},

    # 质量类
    'quality': {'high-quality', 'premium', 'cheap', 'affordable', 'expensive', 'value', 'worth', 'durable',
               'sturdy', 'solid', 'fragile', 'flimsy', 'robust', 'heavy', 'solid', 'strong', 'weak',
               'professional', 'consumer', 'industrial', 'commercial', 'budget', 'luxury', 'economical'},

    # 品牌类（从A2_brand提取的高频品牌 + 通用词）
    'brand': {'swarovski', 'beadaholique', 'darice', 'sizzix', 'generic', 'brand-name', 'original',
             'authentic', 'genuine', 'knockoff', 'counterfeit', 'replica', 'imitation', 'official', 'licensed'},

    # 价格类
    'price': {'cheap', 'expensive', 'affordable', 'budget', 'luxury', 'premium', 'economical', 'value',
              'worth', 'costly', 'cost-effective', 'pricey', 'inexpensive', 'mid-range', 'high-end',
              'low-cost', 'best-value', 'overpriced', 'underpriced', 'reasonable', 'unreasonable',
              'discount', 'sale', 'clearance', 'deal', 'bargain'},

    # 安全/环保类
    'safety': {'safe', 'unsafe', 'non-toxic', 'toxic', 'hazardous', 'dangerous', 'eco-friendly', 'green',
              'organic', 'natural', 'chemical-free', 'phthalate-free', 'bpa-free', 'lead-free', 'food-grade',
              'medical-grade', 'hypoallergenic', 'allergy-free', 'child-safe', 'pet-safe'},

    # 使用场景类
    'usecase': {'indoor', 'outdoor', 'indoor-outdoor', 'indoor/outdoor', 'kitchen', 'bathroom', 'bedroom',
               'living room', 'office', 'classroom', 'studio', 'workshop', 'garage', 'garden', 'camping',
               'travel', 'portable', 'home', 'commercial', 'professional', 'personal', 'gift', 'beginner',
               'expert', 'children', 'kids', 'adults', 'seniors', 'family'},
}

# 展平属性词典为词集合
ALL_ATTRIBUTE_WORDS = set()
for category_words in ATTRIBUTE_DICTIONARY.values():
    ALL_ATTRIBUTE_WORDS.update(category_words)


def count_tokens(text):
    """快速分词计数"""
    return len([t for t in text.split() if t.strip()])


def count_attributes_in_sentence_spacy(doc):
    """统计句子中形容词/名词作为属性词（基于spaCy POS）

    Returns:
        tuple: (属性总数, 类别分布dict) - 类别为pos标签
    """
    category_counts = Counter()
    total_count = 0

    for token in doc:
        # 形容词(ADJ)和名词(NOUN)认为是属性词
        if token.pos_ in ('ADJ', 'NOUN') and not token.is_punct and not token.is_space:
            category_counts[token.pos_] += 1
            total_count += 1

    return total_count, dict(category_counts)


def count_attributes_in_sentence(sentence):
    """统计句子中提及的属性词数量（基于词典匹配，保留兼容）

    Returns:
        tuple: (属性总数, 属性类别分布dict)
    """
    # 转小写并分词
    words = re.findall(r'\b\w+\b', sentence.lower())

    category_counts = Counter()
    total_count = 0

    for word in words:
        for category, category_words in ATTRIBUTE_DICTIONARY.items():
            if word in category_words:
                category_counts[category] += 1
                total_count += 1

    return total_count, dict(category_counts)


def analyze_attributes_in_doc_spacy(doc):
    """分析文档中所有形容词/名词作为属性（基于spaCy POS）

    Returns: list of dict with attribute info
    """
    results = []

    for token in doc:
        # 形容词和名词认为是属性词
        if token.pos_ in ('ADJ', 'NOUN') and not token.is_punct and not token.is_space:
            results.append({
                'word': token.text,
                'category': token.pos_,  # 使用POS标签作为类别
                'pos': token.pos_,
                'dep': token.dep_,
                'position': token.i
            })

    return results


def analyze_attributes_in_doc(doc):
    """分析文档中所有属性提及（基于词典匹配，保留兼容）

    Returns: list of dict with attribute info
    """
    results = []

    # 获取句子文本
    sentence_text = doc.text
    words = re.findall(r'\b\w+\b', sentence_text.lower())

    for word in words:
        for category, category_words in ATTRIBUTE_DICTIONARY.items():
            if word in category_words:
                # 查找该词在doc中的token位置
                for token in doc:
                    if token.text.lower() == word and not token.is_punct and not token.is_space:
                        results.append({
                            'word': token.text,
                            'category': category,
                            'pos': token.pos_,
                            'dep': token.dep_,
                            'position': token.i
                        })
                        break  # 每个词只匹配一次

    return results


def is_attribute_word(word):
    """判断一个词是否是属性词"""
    word_lower = word.lower()
    for category_words in ATTRIBUTE_DICTIONARY.values():
        if word_lower in category_words:
            return True
    return False


def is_attribute_word_spacy(token):
    """判断一个token是否是属性词（基于spaCy POS）"""
    return token.pos_ in ('ADJ', 'NOUN') and not token.is_punct and not token.is_space


def compute_attr_gaps_spacy(doc):
    """计算一个句子中相邻属性词之间的非属性词数量（基于spaCy POS）

    Returns:
        list: 相邻属性词之间的间隔列表（单词数）
    """
    # 获取所有属性词（形容词/名词）的位置
    attr_positions = []
    for token in doc:
        if is_attribute_word_spacy(token):
            attr_positions.append(token.i)

    # 计算相邻属性词之间的间隔
    gaps = []
    for i in range(len(attr_positions) - 1):
        gap = attr_positions[i + 1] - attr_positions[i] - 1
        gaps.append(gap)

    return gaps


def compute_attr_gaps(sentence):
    """计算一个句子中相邻属性词之间的非属性词数量（基于词典）

    Returns:
        list: 相邻属性词之间的间隔列表（单词数）
    """
    words = sentence.split()
    gaps = []

    # 找到所有属性词的位置
    attr_positions = []
    for i, word in enumerate(words):
        if is_attribute_word(word):
            attr_positions.append(i)

    # 计算相邻属性词之间的间隔（不包含第一个属性词前的内容）
    for i in range(len(attr_positions) - 1):
        gap = attr_positions[i + 1] - attr_positions[i] - 1
        gaps.append(gap)

    return gaps


def get_freq_label(ratio):
    """属性提及频率标签"""
    if ratio < 0.5:
        return 'low'
    elif ratio < 1.0:
        return 'medium'
    else:
        return 'high'


def get_density_label(wpa):
    """属性密度标签（words_per_attribute越低 = 密度越高）"""
    if wpa is None:
        return 'none'
    if wpa < 5:
        return 'very_high'
    elif wpa < 10:
        return 'high'
    elif wpa < 15:
        return 'medium'
    elif wpa < 20:
        return 'low'
    else:
        return 'very_low'


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

    # 直接对每条评论进行spaCy解析，不拆句子
    total_sentences = 0
    sentences_with_attr = 0
    total_attr_count = 0
    total_token_count = 0
    sentence_details = []
    category_counter = Counter()  # 统计各类别属性词使用

    # 线性回归数据收集：(属性数量, 句子长度)
    regression_data = []

    # 属性词间隔数据收集
    all_gaps = []

    # 方案2数据收集：含>=2属性的句子
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

        # 统计属性词（基于spaCy POS：形容词/名词）
        attr_count, category_dist = count_attributes_in_sentence_spacy(doc)
        attr_info = analyze_attributes_in_doc_spacy(doc)

        total_sentences += 1
        total_attr_count += attr_count
        total_token_count += n
        if attr_count > 0:
            sentences_with_attr += 1

        # 统计属性类别
        for cat, cnt in category_dist.items():
            category_counter[cat] += cnt

        # 计算该句子的 words_per_attribute
        wpa = n / attr_count if attr_count > 0 else None

        # 收集线性回归数据
        regression_data.append((attr_count, n))

        # 收集属性词间隔数据（基于spaCy POS）
        gaps = compute_attr_gaps_spacy(doc)
        all_gaps.extend(gaps)

        # 收集方案2数据
        token_lengths_for_eq.append(n)
        attr_counts_for_eq.append(attr_count)

        sentence_details.append({
            'user_id': user_id,
            'sentence': r,
            'token_count': n,
            'attr_count': attr_count,
            'words_per_attribute': wpa,
            'has_attr': attr_count > 0,
            'attr_categories': category_dist,
            'attr_info': attr_info
        })

    if total_sentences == 0:
        return None, []

    # 四个指标
    attr_sentence_ratio = sentences_with_attr / total_sentences
    attr_per_sentence = total_attr_count / total_sentences
    avg_sentence_length = total_token_count / total_sentences
    # 每多少个有效单词使用一个属性词（如果attr_count > 0）
    words_per_attr = total_token_count / total_attr_count if total_attr_count > 0 else None

    # 属性类别统计（转换为dict）
    category_dist = dict(category_counter)

    # 线性回归模型：从用户评论直接学——每个属性用多少词
    # y = sentence_length, X = n_attributes
    # model.coef_ = 每个属性的平均词数（斜率）
    # model.intercept_ = 框架开销（0个属性时的句子长度）
    model_words_per_attribute = None
    model_base_overhead = None
    model_r2 = None
    if len(regression_data) >= 3:
        X = np.array([[attr_count] for attr_count, _ in regression_data])
        y = np.array([length for _, length in regression_data])
        model = LinearRegression()
        model.fit(X, y)
        model_words_per_attribute = float(model.coef_[0])
        model_base_overhead = float(model.intercept_)
        model_r2 = float(model.score(X, y))

    # 方案2：只考虑含>=2个属性的句子（过滤掉纯情感句和纯叙事句）
    # 在循环中已经计算过，直接使用已有的attr_count
    attr_sentences_for_eq_lengths = [n for n, attr_count in zip(token_lengths_for_eq, attr_counts_for_eq) if attr_count >= 2]
    if len(attr_sentences_for_eq_lengths) > 0:
        expected_query_length = float(np.mean(attr_sentences_for_eq_lengths))
    else:
        expected_query_length = None

    # 属性词间隔分析（每写一个属性词前平均隔多少个非属性词）
    mean_gap = float(np.mean(all_gaps)) if len(all_gaps) > 0 else None
    median_gap = float(np.median(all_gaps)) if len(all_gaps) > 0 else None

    return {
        'user_id': user_id,
        'attr_sentence_ratio': attr_sentence_ratio,
        'attr_per_sentence': attr_per_sentence,
        'avg_sentence_length': avg_sentence_length,
        'words_per_attribute': words_per_attr,
        'freq_label': get_freq_label(attr_sentence_ratio),
        'density_label': get_density_label(words_per_attr),
        'length_label': get_length_label(avg_sentence_length),
        'total_sentences': total_sentences,
        'total_attr_count': total_attr_count,
        'category_distribution': category_dist,
        # 线性回归模型结果
        'model_words_per_attribute': model_words_per_attribute,
        'model_base_overhead': model_base_overhead,
        'model_r2': model_r2,
        # 方案2：含>=2属性的句子的平均长度（更精确的query长度估计）
        'expected_query_length': expected_query_length,
        # 属性词间隔分析
        'mean_gap': mean_gap,
        'median_gap': median_gap
    }, sentence_details


def main():
    start_time = time.time()
    log('处理用户数据...')

    # 从单个JSON文件加载所有用户
    ALL_USERS_FILE = '/fs04/ar57/wenyu/result/personal_query/01_preference_extraction/Pet_Supplies/stage1_filtered_users_reviews.json'
    log(f'加载用户数据: {ALL_USERS_FILE}')
    with open(ALL_USERS_FILE, 'r', encoding='utf-8') as f:
        all_users_data = json.load(f)
    user_list = all_users_data['users']
    total = len(user_list)
    log(f'总用户数: {total}')

    # 加载Stage 1商品属性
    STAGE1_ATTR_FILE = '/fs04/ar57/wenyu/result/personal_query/01_preference_extraction/Pet_Supplies/attributes_Pet_Supplies.json'
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
    ratio_vals = np.array([u['attr_sentence_ratio'] for u in users])
    aps_vals = np.array([u['attr_per_sentence'] for u in users])
    len_vals = np.array([u['avg_sentence_length'] for u in users])
    wpa_vals = np.array([u['words_per_attribute'] for u in users if u['words_per_attribute'] is not None])
    # 线性回归模型指标
    model_wpa_vals = np.array([u['model_words_per_attribute'] for u in users if u['model_words_per_attribute'] is not None])
    model_overhead_vals = np.array([u['model_base_overhead'] for u in users if u['model_base_overhead'] is not None])
    model_r2_vals = np.array([u['model_r2'] for u in users if u['model_r2'] is not None])
    # 方案2：expected_query_length
    eq_len_vals = np.array([u['expected_query_length'] for u in users if u['expected_query_length'] is not None])
    # 属性词间隔分析
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
        for label in ['low', 'medium', 'high', 'very_high', 'very_low', 'simple', 'complex', 'short', 'medium', 'long', 'none']:
            if label in cnt:
                result.append({'label': label, 'count': cnt[label], 'percentage': round(cnt[label]/total_u*100, 2)})
        return result

    log('')
    log('=' * 60)
    log('attr_sentence_ratio 分布（有属性提及的句子占比）')
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
    log('attr_per_sentence 分布（平均每句属性词数）')
    log('=' * 60)
    log(f'{"区间":<20} │ {"人数":>6} │ {"占比":>8}')
    log('─' * 60)
    dist_aps = compute_distribution(aps_vals, max_val_override=5.0)
    for d in dist_aps:
        if d['count'] > 0:
            bar = '█' * int(d['percentage'] / 2)
            log(f'{d["interval"]:<20} │ {d["count"]:>6} │ {d["percentage"]:>7.1f}% │ {bar}')
    log(f'最小值: {aps_vals.min():.4f}, 最大值: {aps_vals.max():.4f}, 均值: {aps_vals.mean():.4f}')

    log('')
    log('=' * 60)
    log('avg_sentence_length 分布（平均句子长度）')
    log('=' * 60)
    log(f'最小值: {len_vals.min():.1f}, 最大值: {len_vals.max():.1f}, 均值: {len_vals.mean():.1f}')

    log('')
    log('=' * 60)
    log('words_per_attribute 分布（每多少有效单词使用一个属性词）')
    log('=' * 60)
    log(f'有效用户数（含属性词）: {len(wpa_vals)}')
    if len(wpa_vals) > 0:
        log(f'最小值: {wpa_vals.min():.1f}, 最大值: {wpa_vals.max():.1f}, 均值: {wpa_vals.mean():.1f}')
        log(f'中位数: {np.median(wpa_vals):.1f}')

        # words_per_attribute 区间分布
        wpa_intervals = [
            (0, 3, '0-3'), (3, 5, '3-5'), (5, 7, '5-7'), (7, 10, '7-10'),
            (10, 15, '10-15'), (15, 20, '15-20'), (20, 30, '20-30'), (30, float('inf'), '30+')
        ]
        log(f'{"区间":<15} │ {"人数":>6} │ {"占比":>8}')
        log('─' * 40)
        wpa_dist = []
        for low, high, label in wpa_intervals:
            if high == float('inf'):
                count = int(np.sum(wpa_vals >= low))
            else:
                count = int(np.sum((wpa_vals >= low) & (wpa_vals < high)))
            pct = count / len(wpa_vals) * 100 if len(wpa_vals) > 0 else 0
            bar = '█' * int(pct)
            log(f'{label:<15} │ {count:>6} │ {pct:>7.2f}% │ {bar}')
            wpa_dist.append({'interval': label, 'count': count, 'percentage': round(pct, 2)})
    else:
        wpa_dist = []

    # 线性回归模型结果统计
    log('')
    log('=' * 60)
    log('线性回归模型结果（从用户评论学习）')
    log('=' * 60)

    if len(model_wpa_vals) > 0:
        log(f'model_words_per_attribute（模型斜率 = 每个属性的平均词数）:')
        log(f'  有效用户数: {len(model_wpa_vals)}')
        log(f'  最小值: {model_wpa_vals.min():.2f}, 最大值: {model_wpa_vals.max():.2f}, 均值: {model_wpa_vals.mean():.2f}')
        log(f'  中位数: {np.median(model_wpa_vals):.2f}')

        # 区间分布
        mwpa_intervals = [
            (0, 3, '0-3'), (3, 5, '3-5'), (5, 7, '5-7'), (7, 10, '7-10'),
            (10, 15, '10-15'), (15, 20, '15-20'), (20, 30, '20-30'), (30, float('inf'), '30+')
        ]
        log(f'  {"区间":<15} │ {"人数":>6} │ {"占比":>8}')
        log(f'  {"─" * 40}')
        mwpa_dist = []
        for low, high, label in mwpa_intervals:
            if high == float('inf'):
                count = int(np.sum(model_wpa_vals >= low))
            else:
                count = int(np.sum((model_wpa_vals >= low) & (model_wpa_vals < high)))
            pct = count / len(model_wpa_vals) * 100 if len(model_wpa_vals) > 0 else 0
            bar = '█' * int(pct)
            log(f'  {label:<15} │ {count:>6} │ {pct:>7.2f}% │ {bar}')
            mwpa_dist.append({'interval': label, 'count': count, 'percentage': round(pct, 2)})
    else:
        mwpa_dist = []

    if len(model_overhead_vals) > 0:
        log(f'model_base_overhead（模型截距 = 0个属性时的句子长度）:')
        log(f'  最小值: {model_overhead_vals.min():.1f}, 最大值: {model_overhead_vals.max():.1f}, 均值: {model_overhead_vals.mean():.1f}')

    if len(model_r2_vals) > 0:
        log(f'model_r2（R² 决定系数）:')
        log(f'  最小值: {model_r2_vals.min():.3f}, 最大值: {model_r2_vals.max():.3f}, 均值: {model_r2_vals.mean():.3f}')
        log(f'  中位数: {np.median(model_r2_vals):.3f}')

    # 方案2：expected_query_length（仅含>=2属性的句子平均长度）
    log('')
    log('=' * 60)
    log('方案2：expected_query_length（仅含>=2属性的句子平均长度）')
    log('=' * 60)
    log(f'有效用户数（含>=2属性的句子）: {len(eq_len_vals)}')
    if len(eq_len_vals) > 0:
        log(f'最小值: {eq_len_vals.min():.1f}, 最大值: {eq_len_vals.max():.1f}, 均值: {eq_len_vals.mean():.1f}')
        log(f'中位数: {np.median(eq_len_vals):.1f}')

        # 区间分布
        eq_intervals = [
            (0, 10, '0-10'), (10, 15, '10-15'), (15, 20, '15-20'),
            (20, 25, '20-25'), (25, 30, '25-30'), (30, 35, '30-35'), (35, float('inf'), '35+')
        ]
        log(f'  {"区间":<15} │ {"人数":>6} │ {"占比":>8}')
        log(f'  {"─" * 40}')
        eq_dist = []
        for low, high, label in eq_intervals:
            if high == float('inf'):
                count = int(np.sum(eq_len_vals >= low))
            else:
                count = int(np.sum((eq_len_vals >= low) & (eq_len_vals < high)))
            pct = count / len(eq_len_vals) * 100 if len(eq_len_vals) > 0 else 0
            bar = '█' * int(pct / 2)
            log(f'  {label:<15} │ {count:>6} │ {pct:>7.2f}% │ {bar}')
            eq_dist.append({'interval': label, 'count': count, 'percentage': round(pct, 2)})
    else:
        eq_dist = []

    # 属性词间隔统计（mean_gap）
    log('')
    log('=' * 60)
    log('属性词间隔统计（每写一个属性词前平均隔多少个非属性词）')
    log('=' * 60)
    log(f'有效用户数: {len(mean_gap_vals)}')
    if len(mean_gap_vals) > 0:
        log(f'mean_gap:')
        log(f'  最小值: {mean_gap_vals.min():.2f}, 最大值: {mean_gap_vals.max():.2f}')
        log(f'  均值: {mean_gap_vals.mean():.2f}, 中位数: {np.median(mean_gap_vals):.2f}')
        log(f'median_gap:')
        log(f'  最小值: {median_gap_vals.min():.2f}, 最大值: {median_gap_vals.max():.2f}')
        log(f'  均值: {median_gap_vals.mean():.2f}, 中位数: {np.median(median_gap_vals):.2f}')

        # 按每5个单词区间统计
        gap_intervals = [
            (0, 5, '0-5'), (5, 10, '5-10'), (10, 15, '10-15'),
            (15, 20, '15-20'), (20, 25, '20-25'), (25, 30, '25-30'), (30, float('inf'), '30+')
        ]
        log(f'  {"区间":<15} │ {"用户数":>8} │ {"占比":>8}')
        log(f'  {"─" * 45}')
        gap_dist = []
        for low, high, label in gap_intervals:
            if high == float('inf'):
                count = int(np.sum(mean_gap_vals >= low))
            else:
                count = int(np.sum((mean_gap_vals >= low) & (mean_gap_vals < high)))
            pct = count / len(mean_gap_vals) * 100 if len(mean_gap_vals) > 0 else 0
            bar = '█' * int(pct / 2)
            log(f'  {label:<15} │ {count:>8} │ {pct:>7.2f}% │ {bar}')
            gap_dist.append({'interval': label, 'count': count, 'percentage': round(pct, 2)})
    else:
        gap_dist = []

    # 标签分布
    log('')
    log('=' * 60)
    log('用户写作特征标签分布')
    log('=' * 60)

    log('属性提及频率 (freq_label):')
    for item in count_labels('freq_label'):
        bar = '█' * int(item['percentage'] / 3)
        log(f'  {item["label"]:<10} │ {item["count"]:>5} │ {item["percentage"]:>6.1f}% │ {bar}')

    log('属性密度 (density_label):')
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

    # 属性类别统计
    log('')
    log('=' * 60)
    log('属性类别分布（全局）')
    log('=' * 60)

    # 汇总所有用户的属性类别
    global_cat_counter = Counter()
    for u in users:
        for cat, cnt in u['category_distribution'].items():
            global_cat_counter[cat] += cnt

    total_attrs = sum(global_cat_counter.values())
    log(f'总属性词数: {total_attrs}')
    log(f'{"类别":<20} │ {"数量":>6} │ {"占比":>8}')
    log('─' * 60)
    for cat, count in sorted(global_cat_counter.items(), key=lambda x: -x[1]):
        pct = count / total_attrs * 100 if total_attrs > 0 else 0
        bar = '█' * int(pct / 2)
        log(f'{cat:<20} │ {count:>6} │ {pct:>7.1f}% │ {bar}')

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
        'total_attrs': total_attrs,
        'attr_sentence_ratio': {
            'min': round(float(ratio_vals.min()), 4),
            'max': round(float(ratio_vals.max()), 4),
            'mean': round(float(ratio_vals.mean()), 4)
        },
        'attr_per_sentence': {
            'min': round(float(aps_vals.min()), 4),
            'max': round(float(aps_vals.max()), 4),
            'mean': round(float(aps_vals.mean()), 4)
        },
        'avg_sentence_length': {
            'min': round(float(len_vals.min()), 1),
            'max': round(float(len_vals.max()), 1),
            'mean': round(float(len_vals.mean()), 1)
        },
        'words_per_attribute': {
            'min': round(float(wpa_vals.min()), 1) if len(wpa_vals) > 0 else None,
            'max': round(float(wpa_vals.max()), 1) if len(wpa_vals) > 0 else None,
            'mean': round(float(wpa_vals.mean()), 1) if len(wpa_vals) > 0 else None,
            'median': round(float(np.median(wpa_vals)), 1) if len(wpa_vals) > 0 else None
        },
        'model_words_per_attribute': {
            'min': round(float(model_wpa_vals.min()), 2) if len(model_wpa_vals) > 0 else None,
            'max': round(float(model_wpa_vals.max()), 2) if len(model_wpa_vals) > 0 else None,
            'mean': round(float(model_wpa_vals.mean()), 2) if len(model_wpa_vals) > 0 else None,
            'median': round(float(np.median(model_wpa_vals)), 2) if len(model_wpa_vals) > 0 else None
        },
        'model_base_overhead': {
            'min': round(float(model_overhead_vals.min()), 1) if len(model_overhead_vals) > 0 else None,
            'max': round(float(model_overhead_vals.max()), 1) if len(model_overhead_vals) > 0 else None,
            'mean': round(float(model_overhead_vals.mean()), 1) if len(model_overhead_vals) > 0 else None
        },
        'model_r2': {
            'min': round(float(model_r2_vals.min()), 3) if len(model_r2_vals) > 0 else None,
            'max': round(float(model_r2_vals.max()), 3) if len(model_r2_vals) > 0 else None,
            'mean': round(float(model_r2_vals.mean()), 3) if len(model_r2_vals) > 0 else None,
            'median': round(float(np.median(model_r2_vals)), 3) if len(model_r2_vals) > 0 else None
        },
        'expected_query_length': {
            'min': round(float(eq_len_vals.min()), 1) if len(eq_len_vals) > 0 else None,
            'max': round(float(eq_len_vals.max()), 1) if len(eq_len_vals) > 0 else None,
            'mean': round(float(eq_len_vals.mean()), 1) if len(eq_len_vals) > 0 else None,
            'median': round(float(np.median(eq_len_vals)), 1) if len(eq_len_vals) > 0 else None
        },
        'category_distribution': dict(global_cat_counter),
        'elapsed_seconds': round(total_time, 1),
        'distribution_ratio': dist_ratio,
        'distribution_aps': dist_aps,
        'distribution_wpa': wpa_dist,
        'distribution_model_wpa': mwpa_dist,
        'distribution_expected_query_length': eq_dist,
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
        'distribution_gap': gap_dist
    }

    log(f'\n总耗时: {total_time:.1f}s')

    # 保存统计结果到JSON
    log(f'保存统计结果到: {STATS_JSON}')
    with open(STATS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(summary_stats, fp, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
