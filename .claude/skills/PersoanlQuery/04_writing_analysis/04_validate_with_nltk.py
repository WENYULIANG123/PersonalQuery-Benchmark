#!/usr/bin/env python3
"""
Stage 5 验证脚本：使用 NLTK 词典验证字符级拼写错误

验证逻辑：
1. 原文单词不在 NLTK 词典中
2. 修正后的单词在 NLTK 词典中
3. 排除时态变化、单复数、标点符号等非拼写错误
"""

import json
import os
import sys
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

# 导入 NLTK
import nltk
from nltk.corpus import words

# 确保下载了词典数据
try:
    nltk.data.find('corpora/words')
except LookupError:
    log_with_timestamp("Downloading NLTK words corpus...")
    nltk.download('words')

# 加载英语词典（约23万词）
ENGLISH_WORDS = set(words.words())
log_with_timestamp(f"Loaded NLTK dictionary with {len(ENGLISH_WORDS):,} words")

def _is_morphological_variant(original: str, corrected: str) -> bool:
    """
    检查是否为形态变化（时态、单复数等）
    
    Examples:
    - use → used (时态)
    - color → colors (单复数)
    - run → running (ing形式)
    """
    original = original.lower()
    corrected = corrected.lower()
    
    # 完全相同
    if original == corrected:
        return True
    
    # 时态变化检查
    # ed 后缀
    if corrected == original + 'ed' or corrected == original + 'd':
        return True
    # ing 后缀
    if corrected == original + 'ing':
        return True
    # 双写辅音 + ed/ing
    if len(original) > 1 and original[-1] == corrected[-2] == original[-1]:
        if corrected.endswith(original[-1] + 'ed') or corrected.endswith(original[-1] + 'ing'):
            return True
    
    # 单复数变化
    # 简单复数 s/es
    if corrected == original + 's' or corrected == original + 'es':
        return True
    # y → ies
    if original.endswith('y') and corrected == original[:-1] + 'ies':
        return True
    # f/fe → ves
    if original.endswith('f') and corrected == original[:-1] + 'ves':
        return True
    if original.endswith('fe') and corrected == original[:-2] + 'ves':
        return True
    
    # 不规则动词形式（常见）
    irregular_pairs = [
        ('go', 'went'), ('go', 'goes'), ('go', 'going'),
        ('do', 'did'), ('do', 'does'), ('do', 'doing'),
        ('have', 'had'), ('have', 'has'), ('have', 'having'),
        ('make', 'made'), ('make', 'makes'), ('make', 'making'),
        ('take', 'took'), ('take', 'takes'), ('take', 'taking'),
        ('come', 'came'), ('come', 'comes'), ('come', 'coming'),
        ('see', 'saw'), ('see', 'sees'), ('see', 'seeing'),
        ('get', 'got'), ('get', 'gets'), ('get', 'getting')
    ]
    
    for base, variant in irregular_pairs:
        if (original == base and corrected == variant) or (original == variant and corrected == base):
            return True
    
    return False

def _is_punctuation_or_hyphen_diff(original: str, corrected: str) -> bool:
    import string
    punct_hyphen = string.punctuation
    
    if original.endswith("'s") and corrected == original[:-2]:
        return True
    
    if original.endswith("'") and corrected == original[:-1]:
        return True
    
    original_clean = original.translate(str.maketrans('', '', punct_hyphen))
    corrected_clean = corrected.translate(str.maketrans('', '', punct_hyphen))
    
    if original_clean == corrected_clean:
        return True
    
    original_normalized = original.replace('-', ' ').replace('  ', ' ').strip()
    corrected_normalized = corrected.replace('-', ' ').replace('  ', ' ').strip()
    
    if original_normalized == corrected_normalized:
        return True
    
    original_no_space = original_clean.replace(' ', '')
    corrected_no_space = corrected_clean.replace(' ', '')
    
    if original_no_space == corrected_no_space:
        return True
    
    return False

def validate_with_nltk(error_data: Dict) -> Dict:
    all_errors = error_data.get('detailed_character_errors', [])
    log_with_timestamp(f"Found {len(all_errors)} character-level errors to validate")
    
    brand_names = set(name.lower() for name in error_data.get('brand_names', []))
    if brand_names:
        log_with_timestamp(f"Loaded {len(brand_names)} brand/product names for filtering")
    
    stats = {
        'total_errors': len(all_errors),
        'morphological_filtered': 0,
        'punctuation_filtered': 0,
        'brand_name_filtered': 0,
        'original_in_dict': 0,
        'corrected_not_in_dict': 0,
        'both_not_in_dict': 0,
        'validated_errors': 0
    }
    
    validated_errors = []
    rejected_errors = defaultdict(list)
    
    for error in all_errors:
        original = error.get('original', '').strip().lower()
        corrected = error.get('corrected', '').strip().lower()
        
        if not original or not corrected:
            continue
        
        if _is_morphological_variant(original, corrected):
            stats['morphological_filtered'] += 1
            rejected_errors['morphological'].append(f"{original} → {corrected}")
            continue
        
        if _is_punctuation_or_hyphen_diff(original, corrected):
            stats['punctuation_filtered'] += 1
            rejected_errors['punctuation'].append(f"{original} → {corrected}")
            continue
        
        if brand_names and (original in brand_names or corrected in brand_names):
            stats['brand_name_filtered'] += 1
            rejected_errors['brand_name'].append(f"{original} → {corrected}")
            log_with_timestamp(f"  ⏭️  {original} → {corrected} (brand/product name, skipped)")
            continue
        
        # 3. 词典验证
        original_in_dict = original in ENGLISH_WORDS
        corrected_in_dict = corrected in ENGLISH_WORDS
        
        # 记录统计
        if original_in_dict:
            stats['original_in_dict'] += 1
            rejected_errors['original_in_dict'].append(f"{original} → {corrected}")
        elif not original_in_dict:
            if not corrected_in_dict:
                stats['both_not_in_dict'] += 1
                stats['validated_errors'] += 1
                validated_errors.append(error)
                log_with_timestamp(f"  ✅ Validated (both not in dict): {original} → {corrected}")
            else:
                stats['validated_errors'] += 1
                validated_errors.append(error)
                log_with_timestamp(f"  ✅ Validated: {original} → {corrected}")
        else:
            stats['corrected_not_in_dict'] += 1
            rejected_errors['corrected_not_in_dict'].append(f"{original} → {corrected}")
    
    # 创建验证结果
    validation_result = {
        'timestamp': datetime.now().isoformat(),
        'original_error_count': len(all_errors),
        'validated_error_count': len(validated_errors),
        'validation_stats': stats,
        'rejection_reasons': {
            reason: {
                'count': len(errors),
                'examples': errors[:5]  # 前5个例子
            }
            for reason, errors in rejected_errors.items()
        },
        'validated_errors': validated_errors
    }
    
    # 打印统计摘要
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("VALIDATION SUMMARY")
    log_with_timestamp("="*60)
    log_with_timestamp(f"Total errors analyzed: {stats['total_errors']}")
    log_with_timestamp(f"Filtered - Morphological variants: {stats['morphological_filtered']}")
    log_with_timestamp(f"Filtered - Punctuation/hyphen differences: {stats['punctuation_filtered']}")
    log_with_timestamp(f"Filtered - Brand/product names: {stats['brand_name_filtered']}")
    log_with_timestamp(f"Rejected - Original word in dictionary: {stats['original_in_dict']}")
    log_with_timestamp(f"Accepted - Both words NOT in dictionary: {stats['both_not_in_dict']}")
    log_with_timestamp(f"Accepted - Original not in dict, corrected in dict: {stats['validated_errors'] - stats['both_not_in_dict']}")
    log_with_timestamp(f"VALIDATED character-level errors: {stats['validated_errors']}")
    log_with_timestamp(f"Validation rate: {stats['validated_errors']/stats['total_errors']*100:.1f}%")
    log_with_timestamp("="*60 + "\n")
    
    return validation_result

def main():
    parser = argparse.ArgumentParser(description='Validate character-level errors using NLTK dictionary')
    parser.add_argument('--input-file', type=str, required=True,
                      help='Path to writing analysis result file')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save validation results')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取原始结果
    log_with_timestamp(f"Reading error analysis from {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        error_data = json.load(f)
    
    user_id = error_data.get('user_id', 'unknown')
    log_with_timestamp(f"Validating errors for user {user_id}")
    
    # 执行验证
    validation_result = validate_with_nltk(error_data)
    
    validation_file = os.path.join(args.output_dir, f'character_errors_validation_{user_id}.json')
    with open(validation_file, 'w', encoding='utf-8') as f:
        json.dump(validation_result, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"Validation details saved to {validation_file}")
    
    error_data['detailed_character_errors'] = validation_result['validated_errors']
    error_data['total_character_errors'] = len(validation_result['validated_errors'])
    error_data['nltk_validation_performed'] = True
    error_data['nltk_validation_stats'] = validation_result['validation_stats']
    
    if error_data.get('total_words', 0) > 0:
        error_data['character_error_rate'] = round(
            len(validation_result['validated_errors']) / error_data['total_words'] * 100, 3
        )
    
    with open(args.input_file, 'w', encoding='utf-8') as f:
        json.dump(error_data, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"Updated analysis (validated) saved to {args.input_file}")
    
    return validation_result

if __name__ == "__main__":
    main()