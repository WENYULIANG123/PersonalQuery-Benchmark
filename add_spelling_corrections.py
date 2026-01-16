#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从拼写分析文件中提取拼写错误信息，并添加到处理后的句子中
"""

import json
import re
from collections import defaultdict

def load_spelling_errors(spelling_file):
    """
    从拼写分析文件中提取所有错误信息
    返回一个字典：{original_word: [(corrected_word, reason, error_type), ...]}
    """
    spelling_errors = defaultdict(list)
    
    with open(spelling_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 遍历所有用户
    for user in data.get('users', []):
        error_types = user.get('error_types', {})
        
        # 遍历所有错误类型
        for error_type, errors in error_types.items():
            if not isinstance(errors, list):
                continue
            
            for error in errors:
                original_word = error.get('original_word', '').strip()
                corrected_word = error.get('corrected_word', '').strip()
                reason = error.get('reason', '')
                
                if original_word and corrected_word:
                    spelling_errors[original_word].append({
                        'corrected_word': corrected_word,
                        'reason': reason,
                        'error_type': error_type
                    })
    
    return spelling_errors

def find_spelling_errors_in_sentence(sentence, spelling_errors):
    """
    在句子中查找拼写错误
    返回找到的错误列表
    """
    found_errors = []
    
    # 将句子转换为小写用于匹配（但保留原始大小写信息）
    sentence.lower()
    
    # 检查每个可能的错误单词
    for original_word, corrections in spelling_errors.items():
        # 尝试不同的匹配方式
        # 1. 精确匹配（考虑大小写变化）
        pattern = re.compile(r'\b' + re.escape(original_word) + r'\b', re.IGNORECASE)
        matches = pattern.finditer(sentence)
        
        for match in matches:
            # 获取匹配的原始文本（保留大小写）
            matched_text = sentence[match.start():match.end()]
            
            # 为每个匹配添加所有可能的纠正
            for correction in corrections:
                found_errors.append({
                    'original_word': matched_text,  # 保留原始大小写
                    'corrected_word': correction['corrected_word'],
                    'reason': correction['reason'],
                    'error_type': correction['error_type'],
                    'position': match.start(),
                    'position_end': match.end()
                })
    
    # 去重：如果同一个位置有多个匹配，只保留一个
    if found_errors:
        # 按位置排序
        found_errors.sort(key=lambda x: x['position'])
        # 去除重复位置
        unique_errors = []
        seen_positions = set()
        for error in found_errors:
            pos_key = (error['position'], error['position_end'])
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_errors.append(error)
        found_errors = unique_errors
    
    return found_errors

def add_spelling_corrections(input_file, spelling_file, output_file):
    """
    将拼写纠正信息添加到处理后的句子中
    """
    print(f"正在加载拼写分析文件: {spelling_file}")
    spelling_errors = load_spelling_errors(spelling_file)
    print(f"找到 {len(spelling_errors)} 个不同的错误单词")
    
    print(f"正在加载处理后的句子文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = data.get('sentences', [])
    print(f"找到 {len(sentences)} 个句子")
    
    # 统计信息
    sentences_with_errors = 0
    total_errors_found = 0
    
    # 处理每个句子
    for sentence_data in sentences:
        original_text = sentence_data.get('original', '')
        if not original_text:
            continue
        
        # 查找拼写错误
        found_errors = find_spelling_errors_in_sentence(original_text, spelling_errors)
        
        if found_errors:
            sentences_with_errors += 1
            total_errors_found += len(found_errors)
            
            # 添加拼写纠正信息
            sentence_data['spelling_corrections'] = found_errors
            
            # 同时创建一个纠正后的版本
            corrected_text = original_text
            # 从后往前替换，避免位置偏移问题
            for error in sorted(found_errors, key=lambda x: x['position'], reverse=True):
                original_word = error['original_word']
                corrected_word = error['corrected_word']
                # 替换时保持原始大小写风格
                if original_word.islower():
                    replacement = corrected_word.lower()
                elif original_word.isupper():
                    replacement = corrected_word.upper()
                elif original_word.istitle():
                    replacement = corrected_word.title()
                else:
                    replacement = corrected_word
                
                # 使用原始位置进行替换
                start = error['position']
                end = error['position_end']
                corrected_text = corrected_text[:start] + replacement + corrected_text[end:]
            
            sentence_data['corrected_original'] = corrected_text
    
    # 更新元数据
    data['spelling_correction_stats'] = {
        'total_sentences': len(sentences),
        'sentences_with_errors': sentences_with_errors,
        'total_errors_found': total_errors_found,
        'error_rate': f"{sentences_with_errors / len(sentences) * 100:.2f}%" if sentences else "0%"
    }
    
    # 保存结果
    print(f"正在保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("\n处理完成！")
    print(f"总句子数: {len(sentences)}")
    print(f"包含错误的句子数: {sentences_with_errors}")
    print(f"找到的错误总数: {total_errors_found}")
    print(f"错误率: {data['spelling_correction_stats']['error_rate']}")

if __name__ == '__main__':
    input_file = 'reviews_sentences_nlp_processed.json'
    spelling_file = 'spelling_analysis_combined_batch_ulhdemmplr.json'
    output_file = 'reviews_sentences_nlp_processed.json'
    
    add_spelling_corrections(input_file, spelling_file, output_file)
