#!/usr/bin/env python3
"""
Validated Persona Generation
(Based on Metadata-Verified Match Results)

核心逻辑：
1. 使用 **Validated Match Results** (match_[USER_ID].json) 作为输入
2. 仅提取经过 Step 2 验证的属性 (final_match.selected_attributes)
3. 统计这些验证属性的频率，找出用户真正"Verified"的偏好
4. 结合商品类目信息生成 Persona Prompt
"""

import json
import os
import sys
import gzip
from datetime import datetime
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

def load_metadata(meta_file):
    """
    加载商品元数据，返回 {asin: category_list} 映射
    """
    print(f"Loading metadata from {meta_file}...")
    metadata = {}

    try:
        # Check if file is gzipped
        if meta_file.endswith('.gz'):
            open_func = gzip.open
        else:
            open_func = open

        with open_func(meta_file, 'rt', encoding='utf-8') as f:
            # Detect format
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
    """
    从类目列表中获取最小类目
    """
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

# 停用词列表
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
    'recommend', 'recommended', 'expected', 'exactly', 'exactly',
    'quality', 'delivery', 'packaging', 'usage', 'price', 'value',
    'service', 'experience', 'overall', 'general', 'basic', 'standard'
}

def is_valid_attribute(attr):
    """
    判断属性是否有效
    """
    attr_lower = attr.lower().strip()
    if not attr_lower: return False
    if len(attr_lower) < 3: return False # 放宽一点限制
    if attr_lower in STOPWORDS: return False
    if attr_lower.isdigit(): return False
    
    words = attr_lower.split()
    stopword_count = sum(1 for w in words if w in STOPWORDS)
    if len(words) > 0 and stopword_count / len(words) > 0.6: # 放宽一点限制
        return False
        
    return True

def normalize_attribute(attr):
    """
    标准化属性
    """
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
    """
    从 Validated Match Results 提取属性
    """
    if holdout_asins is None: holdout_asins = set()
    
    user_attrs = []
    attr_pref_category_freq = defaultdict(Counter)
    attr_product_category_freq = defaultdict(Counter)
    
    results = match_data.get('results', [])
    
    for item in results:
        # 只处理验证成功的
        if item.get('status') != 'success':
            continue
            
        asin = item.get('asin', '')
        if asin in holdout_asins:
            continue
            
        # 获取商品类目
        if metadata and asin in metadata:
            category_list = metadata[asin]
            product_category = get_min_category_from_list(category_list)
        else:
            product_category = 'Unknown'
            
        # 获取 Validated Attributes
        # 路径: final_match -> selected_attributes
        final_match = item.get('final_match', {})
        selected_attributes = final_match.get('selected_attributes', [])
        
        if not selected_attributes:
            continue
            
        for attr_obj in selected_attributes:
            attr_name = attr_obj.get('attribute', '')
            dimension = attr_obj.get('dimension', 'General') # e.g. "quality", "feature"
            
            normalized_entity = normalize_attribute(attr_name)
            
            if is_valid_attribute(normalized_entity):
                user_attrs.append(normalized_entity)
                attr_pref_category_freq[normalized_entity][dimension] += 1
                attr_product_category_freq[normalized_entity][product_category] += 1
                
    user_attr_freq = Counter(user_attrs)
    
    uniqueness_scores = {}
    for attr, freq in user_attr_freq.items():
        uniqueness = freq # 简单频率
        
        if attr_pref_category_freq[attr]:
            top_pref_category = attr_pref_category_freq[attr].most_common(1)[0][0]
        else:
            top_pref_category = 'General'
            
        if attr_product_category_freq[attr]:
            top_product_category = attr_product_category_freq[attr].most_common(1)[0][0]
        else:
            top_product_category = 'Unknown'
            
        uniqueness_scores[attr] = {
            'user_freq': freq,
            'uniqueness': uniqueness,
            'preference_category': top_pref_category,
            'product_category': top_product_category
        }
        
    sorted_attrs = sorted(uniqueness_scores.items(),
                          key=lambda x: x[1]['uniqueness'],
                          reverse=True)
                          
    return sorted_attrs, user_attr_freq

def generate_validated_persona_prompt(user_id, unique_attrs, total_products):
    """
    生成高度差异化的 Persona Prompt，强制语义变换，禁止直接使用属性原词
    """
    # 区分通用属性和专业属性
    general_traits = ["quality", "high quality", "easy to use", "ease of use", "versatility", "performance", "good", "great"]
    
    unique_summary = []
    common_summary = []
    
    for attr, scores in unique_attrs[:20]:
        pref_cat = scores.get('preference_category', 'General')
        prod_cat = scores.get('product_category', 'Unknown')
        freq = scores.get('user_freq', 0)
        
        attr_entry = f"- {attr} (Verified in {freq} products) | Cat: {pref_cat} | Product: {prod_cat}"
        
        if any(g in attr.lower() for g in general_traits):
            common_summary.append(attr_entry)
        else:
            unique_summary.append(attr_entry)
            
    unique_text = "\n".join(unique_summary) if unique_summary else "No highly specific attributes found."
    common_text = "\n".join(common_summary) if common_summary else "No common attributes recorded."
    
    prompt = f"""You are a specialist in practical consumer behavior analysis. Your goal is to construct a HIGHLY DIFFERENTIATED and GROUNDED persona for User {user_id}.

**INPUT DATA (VERIFIED PREFERENCES):**

**1. KEY TECHNICAL/UNIQUE ATTRIBUTES:**
{unique_text}

**2. BASELINE NEEDS:**
{common_text}

**STRICT NARRATIVE CONSTRAINTS (MANDATORY):**

1.  **NO VERBATIM ATTRIBUTES**: Do not use the names of the attributes verbatim.
2.  **GROUNDED SEMANTIC TRANSFORMATION**: Convert each attribute into a *practical behavior, a specific quality requirement, or a workplace standard*.
    - *Instead of "tactile poetry"*, say "needs beads with consistent diameter and uniform finish for high-density weaving."
    - *Instead of "creative alchemy"*, say "rigorously tests adhesive bond strength on heavy 300gsm cardstock."
3.  **FORBIDDEN WORDS (AVOID FLOWERY METAPHORS)**: Absolutely DO NOT use the following words or concepts: "Alchemy", "Poetry", "Soul", "Magic", "Maestro", "Conductor", "Whimsical", "Ecosystem", "Symbiosis", "Artistic Spirit".
4.  **OPERATIONAL CONTEXT**: Describe the user's *physical workspace and actions*. Are they at a cluttered workbench using high-pressure tools? Are they at a clean desk using precision tweezers and magnifying lamps? Tie this to the Product Categories provided.
5.  **PRACTICAL ARCHETYPE**: Define them by their *output and technical focus* (e.g., "The Technical Die-Cutter Operator," "The Bulk Jewelry Component Assembler," "The High-Precision Paper Engineer").
6.  **TONE**: Professional, matter-of-fact, and observational. Describe what they *do* and what they *demand* from their tools in a real-world setting.

**OUTPUT FORMAT:**
- Output ONLY the synthesized description.
- Length: 150-200 words.
- NO introductory fluff. Start directly with the persona's core technical identity.
"""
    return prompt

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Personas from Validated Matches")
    parser.add_argument("--input-dir",
                        default="/home/wlia0047/wenyu/result/user_profile/01_matching/results",
                        help="Directory containing match_*.json files")
    parser.add_argument("--holdout-dir",
                        default="/home/wlia0047/wenyu/result/user_profile/02_processing",
                        help="Directory containing holdout data (query_*.json)")
    parser.add_argument("--meta-file",
                        default="/home/wlia0047/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz",
                        help="Metadata file")
    parser.add_argument("--output-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/persona_prompts",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metadata = load_metadata(args.meta_file)
    
    log_with_timestamp("Loading validated matches...")
    
    # 查找 match_*.json 文件
    match_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.startswith('match_') and f.endswith('.json') and 'backup' not in f
    ])
    
    log_with_timestamp(f"Found {len(match_files)} match files.")

    for filename in match_files:
        match_file = os.path.join(args.input_dir, filename)
        try:
            with open(match_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {match_file}: {e}")
            continue
            
        user_id = data.get('user_id')
        if not user_id: continue
        
        log_with_timestamp(f"Processing {user_id}...")
        
        # 加载该用户的 Holdout Data
        holdout_asins = set()
        if args.holdout_dir:
            holdout_file = os.path.join(args.holdout_dir, f"query_{user_id}.json")
            if os.path.exists(holdout_file):
                try:
                    with open(holdout_file, 'r', encoding='utf-8') as hf:
                        h_data = json.load(hf)
                        # 兼容 V2 Split 格式
                        if isinstance(h_data, dict) and 'holdout_asins' in h_data:
                            holdout_asins.update(h_data.get('holdout_asins', []))
                        # 兼容旧格式
                        elif isinstance(h_data, list):
                             for item in h_data:
                                 if isinstance(item, dict) and 'asin' in item:
                                     holdout_asins.add(item['asin'])
                        elif isinstance(h_data, dict):
                            # v1 dict format
                            items = h_data.get('query_results', []) or h_data.get('holdout_results', [])
                            for item in items:
                                 if isinstance(item, dict) and 'asin' in item:
                                     holdout_asins.add(item['asin'])
                                     
                    log_with_timestamp(f"  Loaded {len(holdout_asins)} holdout ASINs (excluded from persona)")
                except Exception as e:
                    log_with_timestamp(f"  Error loading holdout file {holdout_file}: {e}")
            else:
                log_with_timestamp(f"  Warning: No holdout file found for {user_id} at {holdout_file}. Skipping.")
                continue

        if not holdout_asins:
             log_with_timestamp(f"  Warning: Holdout set is empty for {user_id}. Skipping to enforce strict evaluation.")
             continue
        
        # 提取验证属性 (传入 holdout_asins 进行过滤)
        unique_attrs, user_freq = extract_validated_attributes(
            user_id, data, metadata, holdout_asins
        )
        
        if not unique_attrs:
            log_with_timestamp(f"  Warning: No valid attributes found after filtering holdout set.")
            continue

        # Log Top 5
        log_with_timestamp(f"  Top 5 Verified Attributes (Training Set):")
        for attr, scores in unique_attrs[:5]:
            log_with_timestamp(
                f"    - {attr}: freq={scores['user_freq']} "
                f"(Cat: {scores['product_category']})"
            )
            
        # 生成 Prompt
        total_products = len([r for r in data.get('results', []) 
                              if r.get('status')=='success' and r.get('asin') not in holdout_asins])
                              
        prompt = generate_validated_persona_prompt(user_id, unique_attrs, total_products)
        
        # 保存
        output_data = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'version': 'validated_matches_with_strict_holdout',
            'source_file': match_file,
            'holdout_excluded_count': len(holdout_asins),
            'training_set_count': total_products,
            'top_unique_attributes': [
                {'attribute': attr, **scores}
                for attr, scores in unique_attrs[:20]
            ],
            'prompt': prompt
        }
        
        output_file = os.path.join(args.output_dir, f"persona_prompt_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        log_with_timestamp(f"  Saved prompt to {output_file}")

    log_with_timestamp("Done!")

if __name__ == "__main__":
    main()
