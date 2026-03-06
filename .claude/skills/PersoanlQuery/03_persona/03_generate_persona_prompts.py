#!/usr/bin/env python3
"""
Category-Aware Persona Generation v2

关键改进：
1. 【保留类别内独特属性】不使用全局 common_attrs 过滤，而是按类别内相对独特性评分
2. 【跨类别主题识别】识别语义关联的主题（如 color-focused, texture-focused）
3. 【技能迁移模式】突出用户如何将一个领域的专业知识迁移到另一个领域
"""

import json
import os
import sys
import gzip
import math
from datetime import datetime
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

def load_metadata(meta_file):
    print(f"Loading metadata from {meta_file}...")
    metadata = {}
    try:
        if meta_file.endswith('.gz'):
            open_func = gzip.open
        else:
            open_func = open
        with open_func(meta_file, 'rt', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
                for item in data:
                    asin = item.get('asin')
                    if asin:
                        metadata[asin] = item.get('category', [])
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            asin = item.get('asin')
                            if asin:
                                metadata[asin] = item.get('category', [])
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        print(f"Warning: Failed to load metadata: {e}")
    print(f"Loaded {len(metadata)} products from metadata")
    return metadata

def get_min_category_from_list(category_list):
    if not category_list or not isinstance(category_list, list):
        return 'Unknown'
    last_category = category_list[-1] if category_list else 'Unknown'
    if '&' in last_category:
        parts = last_category.split('&')
        for part in reversed(parts):
            min_cat = part.strip()
            if min_cat:
                return min_cat
    return last_category.strip() if last_category else 'Unknown'

STOPWORDS = {
    'description', 'picture', 'pictures', 'correct', 'visual', 'representation',
    'matches', 'match', 'exact', 'exactly', 'looks', 'looking', 'look',
    'nice', 'good', 'great', 'beautiful', 'excellent', 'quality', 'high',
    'well', 'made', 'better', 'best', 'perfect', 'wonderful', 'amazing',
    'lovely', 'pretty', 'fine', 'decent', 'ok', 'okay',
    'various', 'many', 'much', 'more', 'most', 'some', 'several', 'enough',
    'size', 'sizes', 'pack', 'piece', 'pieces', 'set', 'sets', 'lot',
    'time', 'times', 'day', 'days', 'first', 'last', 'long', 'fast', 'quick',
    'use', 'used', 'using', 'work', 'works', 'working', 'get', 'got',
    'need', 'needed', 'want', 'wanted', 'like', 'liked', 'love', 'loved',
    'make', 'made', 'making', 'take', 'took', 'give', 'gave', 'find', 'found',
    'ships', 'shipped', 'shipping', 'delivery', 'arrived', 'arrives',
    'packaged', 'package', 'order', 'ordered', 'received', 'purchase',
    'on-time', 'on time', 'well pkd', 'well-packaged',
    'really', 'very', 'just', 'also', 'even', 'still', 'already',
    'always', 'never', 'ever', 'quite', 'pretty', 'rather', 'really',
    'thing', 'things', 'item', 'items', 'product', 'products',
    'recommend', 'recommended', 'expected', 'exactly',
    'quality', 'delivery', 'packaging', 'usage', 'price', 'value',
    'service', 'experience', 'overall', 'general', 'basic', 'standard'
}

def is_valid_attribute(attr):
    attr_lower = attr.lower().strip()
    if not attr_lower: return False
    if len(attr_lower) < 3: return False
    if attr_lower in STOPWORDS: return False
    if attr_lower.isdigit(): return False
    words = attr_lower.split()
    stopword_count = sum(1 for w in words if w in STOPWORDS)
    if len(words) > 0 and stopword_count / len(words) > 0.6:
        return False
    return True

def normalize_attribute(attr):
    attr = attr.strip()
    prefixes_to_remove = ['Exact ', 'Very ', 'Really ', 'Quite ', 'Pretty ']
    for prefix in prefixes_to_remove:
        if attr.lower().startswith(prefix.lower()):
            attr = attr[len(prefix):].strip()
    return attr

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def extract_validated_attributes(user_id, match_data, metadata=None, holdout_asins=None):
    if holdout_asins is None: holdout_asins = set()
    user_attrs = []
    attr_pref_category_freq = defaultdict(Counter)
    attr_product_category_freq = defaultdict(Counter)

    results = match_data.get('results', [])
    for item in results:
        if item.get('status') != 'success':
            continue
        if item.get('asin') in holdout_asins:
            continue

        selected_attrs = item.get('selected_attributes', [])
        asin = item.get('asin')
        category = 'Unknown'

        if metadata and asin in metadata:
            categories = metadata[asin]
            category = get_min_category_from_list(categories) if categories else 'Unknown'

        for attr_dict in selected_attrs:
            if not is_valid_attribute(attr_dict.get('attribute', '')):
                continue

            attr = normalize_attribute(attr_dict['attribute'])
            pref_category = attr_dict.get('preference_category', 'unknown')

            user_attrs.append({
                'attribute': attr,
                'asin': asin,
                'category': category,
                'preference_category': pref_category
            })

            attr_pref_category_freq[pref_category][attr] += 1
            attr_product_category_freq[category][attr] += 1

    if not user_attrs:
        log_with_timestamp(f"  No valid attributes found for user {user_id}")
        return [], Counter()

    return user_attrs, attr_pref_category_freq

def cluster_categories_by_semantics(categories):
    """
    将类别按语义聚类
    例如：Die-Cuts, Embossing, Ink Pads -> Paper Crafting
         Yarn, Fabric, Elastic, Lace -> Sewing & Textiles
         Paints, Markers, Crayons -> Art Supplies
    """
    paper_crafting = ['die-cuts', 'embossing', 'ink pads', 'cling stamps', 'refills',
                      'stamping', 'paper', 'cardstock', 'cuttlebug']
    sewing_textiles = ['yarn', 'fabric', 'elastic', 'lace', 'ribbons', 'bobbins',
                       'interfacing', 'sewing', 'sewing machine needles', 'doll making',
                       'parts', 'scissors', 'rulers']
    art_supplies = ['paints', 'markers', 'crayons', 'pastels', 'pencils', 'easels']

    clusters = {
        'Paper Crafting': [],
        'Sewing & Textiles': [],
        'Art Supplies': [],
        'Other': []
    }

    for cat in categories:
        cat_lower = cat.lower()
        if any(keyword in cat_lower for keyword in paper_crafting):
            clusters['Paper Crafting'].append(cat)
        elif any(keyword in cat_lower for keyword in sewing_textiles):
            clusters['Sewing & Textiles'].append(cat)
        elif any(keyword in cat_lower for keyword in art_supplies):
            clusters['Art Supplies'].append(cat)
        else:
            clusters['Other'].append(cat)

    return clusters

def group_attributes_by_category(user_attrs):
    """
    将属性按类别分组
    """
    category_attrs = defaultdict(lambda: {'attrs': Counter(), 'total': 0})

    for attr_dict in user_attrs:
        attr = attr_dict['attribute']
        category = attr_dict['category']
        category_attrs[category]['attrs'][attr] += 1
        category_attrs[category]['total'] += 1

    return category_attrs

def identify_cross_category_themes(user_attrs):
    """
    识别跨类别语义主题

    关键改进：
    1. 同一属性出现在多个类别（如 Color 在 Paints, Fabric, Yarn）
    2. 相关属性形成主题（如 color-themed: Color, Color Accuracy, Color Variety）
    3. 识别技能迁移模式（如: 尺寸敏感: Width, Weight, Size）
    """
    # 定义语义主题
    themes = {
        'color_focused': {
            'keywords': ['color', 'color accuracy', 'color quality', 'color variety',
                        'color palette', 'color vibrancy', 'colour'],
            'name': 'Color Expertise',
            'description': 'transfers color matching skills across different media'
        },
        'texture_material': {
            'keywords': ['softness', 'material quality', 'fabric quality', 'finish',
                        'sturdiness', 'texture', 'feel', 'hand'],
            'name': 'Material & Texture Focus',
            'description': 'prioritizes tactile qualities and material properties'
        },
        'precision_sizing': {
            'keywords': ['width', 'weight', 'size', 'thickness', 'length',
                        'dimension', 'measurement', 'fit'],
            'name': 'Precision & Sizing',
            'description': 'demands exact measurements and precise specifications'
        },
        'utility_functional': {
            'keywords': ['ease of use', 'functionality', 'application', 'versatility',
                        'compatibility', 'convenient', 'practical'],
            'name': 'Utility & Functionality',
            'description': 'values practical, functional performance'
        },
        'aesthetic_design': {
            'keywords': ['design', 'design appeal', 'design quality', 'design variety',
                        'pattern', 'style', 'aesthetic', 'visual'],
            'name': 'Aesthetic & Design',
            'description': 'has a strong eye for visual design and patterns'
        },
        'value_economy': {
            'keywords': ['affordability', 'value for money', 'price', 'cost',
                        'economical', 'budget', 'inexpensive'],
            'name': 'Value Conscious',
            'description': 'seeks good value and affordability'
        }
    }

    # 分析每个主题
    cross_category_themes = {}

    for theme_key, theme_info in themes.items():
        theme_attrs = defaultdict(lambda: {'categories': set(), 'freq': 0})

        for attr_dict in user_attrs:
            attr = attr_dict['attribute'].lower()
            category = attr_dict['category']

            # 检查是否匹配主题关键词
            for keyword in theme_info['keywords']:
                if keyword in attr:
                    theme_attrs[keyword]['categories'].add(category)
                    theme_attrs[keyword]['freq'] += 1
                    break  # 每个属性只匹配一个关键词

        # 只保留跨类别的主题（至少出现在2个类别中）
        cross_attrs = [
            {
                'keyword': keyword,
                'categories': sorted(list(data['categories'])),
                'freq': data['freq']
            }
            for keyword, data in theme_attrs.items()
            if len(data['categories']) >= 2
        ]

        if cross_attrs:
            cross_category_themes[theme_key] = {
                'name': theme_info['name'],
                'description': theme_info['description'],
                'attributes': sorted(cross_attrs, key=lambda x: x['freq'], reverse=True)
            }

    return cross_category_themes

def get_category_specific_attributes(user_attrs, min_attrs_per_category=2):
    """
    获取每个类别的 top 属性

    关键改进：使用类别内相对独特性评分，不依赖全局 common_attrs 过滤
    保留类别内的 top 属性，即使它们在全局是"常见"的
    """
    category_attrs = group_attributes_by_category(user_attrs)
    category_specific = {}

    for category, data in category_attrs.items():
        if data['total'] >= min_attrs_per_category:
            # 计算类别内独特性分数
            # 分数 = 频率 × log(频率) → 给高频属性更高权重
            attr_scores = {}
            for attr, freq in data['attrs'].items():
                # 使用频率 × (1 + log(频率)) 作为评分
                # 这样高频属性会得到更高分，但仍保持相对顺序
                score = freq * (1 + math.log(freq + 1))
                attr_scores[attr] = {'freq': freq, 'score': score}

            # 按评分排序，取 top 5
            sorted_attrs = sorted(attr_scores.items(),
                                  key=lambda x: x[1]['score'],
                                  reverse=True)[:5]

            category_specific[category] = [
                {
                    'attribute': attr,
                    'freq': data['freq'],
                    'score': round(data['score'], 2)
                }
                for attr, data in sorted_attrs
            ]

    return category_specific

def get_overall_top_attrs(user_attrs, top_n=15):
    """
    获取整体 top 属性（不进行全局过滤）
    """
    attr_counter = Counter()
    attr_categories = {}

    for attr_dict in user_attrs:
        attr = attr_dict['attribute']
        category = attr_dict['category']
        attr_counter[attr] += 1
        if attr not in attr_categories:
            attr_categories[attr] = category

    top_attrs = []
    for attr, freq in attr_counter.most_common(top_n):
        top_attrs.append({
            'attribute': attr,
            'freq': freq,
            'category': attr_categories.get(attr, 'Unknown')
        })

    return top_attrs

def generate_category_aware_persona_prompt_v2(user_id, user_attrs, category_specific_attrs,
                                               cross_category_themes, category_clusters):
    """
    生成类别感知的画像提示词 v2

    关键改进：
    1. 不过滤全局 common_attrs - 保留类别内 top 属性
    2. 突出跨类别语义主题和技能迁移模式
    3. Prompt 中明确说明用户如何在不同领域间迁移技能
    """

    # 构建类别上下文
    category_context_lines = []
    for cluster_name, categories in category_clusters.items():
        if categories:
            category_context_lines.append(f"- {cluster_name}: {', '.join(categories[:3])}")

    category_context = "\n".join(category_context_lines) if category_context_lines else "Mixed product types"

    # 构建类别特定属性展示（不过滤，直接显示 top 3）
    category_attrs_text = []
    for category, attrs in sorted(category_specific_attrs.items(),
                                   key=lambda x: sum(a['freq'] for a in x[1]),
                                   reverse=True)[:5]:  # 只显示前5个类别
        attrs_list = "\n    ".join([
            f"- {a['attribute']} ({a['freq']}x)"
            for a in attrs[:3]  # 每个类别显示 top 3
        ])
        category_attrs_text.append(f"**{category}:**\n    {attrs_list}")

    # 构建跨类别主题展示
    themes_text = []
    for theme_key, theme_data in sorted(cross_category_themes.items(),
                                       key=lambda x: sum(a['freq'] for a in x[1]['attributes']),
                                       reverse=True)[:4]:  # 显示前4个主题
        attrs_desc = ", ".join([a['keyword'] for a in theme_data['attributes'][:3]])
        categories = set()
        for attr_data in theme_data['attributes']:
            categories.update(attr_data['categories'])
        categories_str = ", ".join(sorted(list(categories))[:4])

        themes_text.append(
            f"- **{theme_data['name']}**: {attrs_desc}\n"
            f"  (across: {categories_str})\n"
            f"  → {theme_data['description']}"
        )

    # 组装最终文本
    if category_attrs_text:
        category_section = "\n\n**2. CATEGORY-SPECIFIC TOP PRIORITIES:**\n" + "\n".join(category_attrs_text)
    else:
        category_section = ""

    if themes_text:
        themes_section = "\n\n**3. CROSS-CATEGORY EXPERTISE & SKILL TRANSFER:**\n" + "\n".join(themes_text)
    else:
        themes_section = ""

    # 获取整体 top 属性
    overall_top = get_overall_top_attrs(user_attrs, top_n=10)
    overall_text = "\n".join([
        f"- {a['attribute']} ({a['freq']}x, from {a['category']})"
        for a in overall_top[:10]
    ])

    prompt = f"""You are creating a user persona that a HUMAN can easily understand and relate to.

**USER PREFERENCE DATA:**

**IMPORTANT - PRODUCT CATEGORY CONTEXT:**
This user reviews products across MULTIPLE categories:
{category_context}

The persona MUST reflect their expertise and how they transfer skills across these different product types.

**1. OVERALL TOP ATTRIBUTES:**
{overall_text}{category_section}{themes_section}

**WRITING GUIDELINES (CRITICAL):**

1.  **START WITH A SPECIFIC PERSONALITY TYPE**
    - Use ONE simple personality word: fussy, picky, practical, creative, perfectionist
    - DO NOT use: multi-disciplinary, detail-oriented maker, craft enthusiast
    - INSTEAD use: "This fussy person...", "This practical sewer...", "This picky creator..."

2.  **DESCRIBE WHAT THEY ACTUALLY MAKE**
    - Be specific: "makes clothes for her family", "creates greeting cards", "sews costumes"
    - DO NOT say: "creates projects", "makes items", "produces crafts"
    - Examples: "She sews dresses and costumes for her grandkids", "They make cards for birthdays"

3.  **EXPLAIN SKILL TRANSFER IN CONCRETE TERMS**
    - Show HOW they use skills from one area in another
    - Example: "Uses the color matching she learned from sewing to pick paints"
    - Example: "Applies the same care for fabric quality when choosing cardstock"
    - DO NOT say: "transfers skills seamlessly", "applies high standards"

4.  **MENTION SPECIFIC PRODUCT REQUIREMENTS**
    - For textiles: "wants soft lace that doesn't scratch", "needs fabric colors that match exactly"
    - For paper crafts: "looks for templates that cut cleanly", "wants ink that doesn't smudge"
    - For tools: "wants scissors that feel comfortable in hand", "needs templates that are easy to trace"

5.  **USE SIMPLE, EVERYDAY LANGUAGE**
    - DO NOT use: multi-disciplinary creative, professional-quality, seamless transfer
    - INSTEAD use: "works with different materials", "wants things that work well", "cares about quality"
    - Write like you're describing a real person to a friend

6.  **KEEP IT SHORT AND SPECIFIC**
    - Length: 80-100 words
    - One idea per sentence
    - Focus on ACTIONS and PREFERENCES, not abstract qualities

**BAD EXAMPLES (DO NOT COPY):**
❌ "This multi-disciplinary creative seamlessly transfers high standards"
❌ "A detail-oriented maker who demands professional-quality results"

**GOOD EXAMPLES (FOLLOW THIS STYLE):**
✅ "This picky sewer makes clothes for her family. She uses the same care for fabric color when choosing paint for her craft projects. She wants soft lace that doesn't scratch skin and easy-to-use templates that cut cleanly on the first try."

✅ "This practical person sews costumes and makes greeting cards. They apply their fabric quality standards to cardstock - if it doesn't feel right, they won't buy it. They want tools that work well without fussing with complicated instructions."

**OUTPUT FORMAT:**
- Output ONLY the persona description
- Start with personality type
- Then what they make
- Then how skills transfer
- Then specific requirements
- NO introduction, NO conclusion
"""
    return prompt

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Category-Aware Persona Prompts v2")
    parser.add_argument("--input-dir",
                        default="/home/wlia0047/wenyu/result/user_profile/01_matching/results",
                        help="Directory containing match_*.json files")
    parser.add_argument("--meta-file",
                        default="/fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz",
                        help="Metadata file")
    parser.add_argument("--output-dir",
                        default="/home/wlia0047/wenyu/result/user_profile/persona_prompts_category_aware_v2",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metadata = load_metadata(args.meta_file)

    log_with_timestamp("Loading validated matches...")
    match_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.startswith('match_') and f.endswith('.json') and 'backup' not in f
    ])
    log_with_timestamp(f"Found {len(match_files)} match files.")

    # 处理每个用户
    processed_count = 0
    for filename in match_files:
        match_file = os.path.join(args.input_dir, filename)
        try:
            with open(match_file, 'r') as f:
                match_data = json.load(f)

            user_id = match_data.get('user_id')
            if not user_id:
                continue

            log_with_timestamp(f"\n{'='*60}")
            log_with_timestamp(f"Processing user: {user_id}")

            # 提取属性
            user_attrs, _ = extract_validated_attributes(user_id, match_data, metadata)

            if not user_attrs:
                continue

            # 按类别统计
            category_attrs = group_attributes_by_category(user_attrs)
            log_with_timestamp(f"  Product categories: {len(category_attrs)}")
            for cat, data in sorted(category_attrs.items(),
                                   key=lambda x: x[1]['total'],
                                   reverse=True)[:3]:
                log_with_timestamp(f"    - {cat}: {data['total']} attributes")

            # 获取类别特定属性（不过滤全局 common）
            category_specific = get_category_specific_attributes(user_attrs)
            log_with_timestamp(f"  Category-specific top attributes:")
            for cat, attrs in sorted(category_specific.items(),
                                   key=lambda x: sum(a['freq'] for a in x[1]),
                                   reverse=True)[:3]:
                log_with_timestamp(f"    - {cat}: {', '.join([a['attribute'] for a in attrs[:2]])}")

            # 识别跨类别主题
            cross_themes = identify_cross_category_themes(user_attrs)
            if cross_themes:
                log_with_timestamp(f"  Cross-category themes: {len(cross_themes)}")
                for theme_key, theme_data in list(cross_themes.items())[:2]:
                    log_with_timestamp(f"    - {theme_data['name']}: {len(theme_data['attributes'])} attributes")

            # 语义聚类
            all_categories = list(category_specific.keys())
            category_clusters = cluster_categories_by_semantics(all_categories)

            # 生成 Prompt
            prompt = generate_category_aware_persona_prompt_v2(
                user_id, user_attrs, category_specific, cross_themes, category_clusters
            )

            # 准备输出数据
            output_file = os.path.join(args.output_dir, f"persona_prompt_{user_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'version': 'category_aware_v2',
                    'source_file': match_file,
                    'training_set_count': len([r for r in match_data.get('results', []) if r.get('status')=='success']),
                    'category_distribution': {
                        cat: data['total']
                        for cat, data in sorted(category_attrs.items(),
                                               key=lambda x: x[1]['total'],
                                               reverse=True)
                    },
                    'category_specific_attributes': category_specific,
                    'cross_category_themes': cross_themes,
                    'prompt': prompt
                }, f, indent=2)

            log_with_timestamp(f"Saved: {output_file}")
            processed_count += 1

        except Exception as e:
            log_with_timestamp(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    log_with_timestamp(f"\nTotal users processed: {processed_count}")
    log_with_timestamp(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
