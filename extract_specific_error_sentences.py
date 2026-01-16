#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取包含特定错误类型的句子
只提取这5种错误类型：
- Deletion
- Insertion
- Transposition
- Scramble
- Substitution
"""

import json
from collections import defaultdict

def extract_sentences_with_specific_errors(input_file, output_file):
    """
    提取包含这5种错误类型之一的句子，输出JSON格式，包含错误单词和修正单词
    """
    # 只提取这5种错误类型
    allowed_error_types = {
        "Deletion",
        "Insertion",
        "Transposition",
        "Scramble",
        "Substitution"
    }
    
    # 使用字典存储，key为review，value为错误列表
    # 格式: {review: [{"error_type": ..., "original_word": ..., "corrected_word": ..., "reason": ..., "review_id": ...}, ...]}
    matched_data = defaultdict(list)
    
    # 统计信息
    users_with_errors = 0
    total_errors = 0
    
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共 {len(data.get('users', []))} 个用户")
    print(f"只提取这5种错误类型: {allowed_error_types}")
    print("\n正在处理...")
    
    # 遍历所有用户
    for user in data.get('users', []):
        user.get('user_id', 'unknown')
        error_types = user.get('error_types', {})
        user_has_target_errors = False
        
        # 只提取这5种错误类型中的句子和错误信息
        for error_type, error_list in error_types.items():
            # 只处理允许的错误类型
            if error_type not in allowed_error_types:
                continue
            
            if isinstance(error_list, list) and len(error_list) > 0:
                user_has_target_errors = True
                for error_item in error_list:
                    if isinstance(error_item, dict) and 'review' in error_item:
                        review = error_item.get('review', '').strip()
                        if review:
                            # 构建错误信息
                            error_info = {
                                "error_type": error_type,
                                "original_word": error_item.get('original_word', ''),
                                "corrected_word": error_item.get('corrected_word', ''),
                                "reason": error_item.get('reason', ''),
                                "review_id": error_item.get('review_id', '')
                            }
                            # 使用review作为key，同一个review可能有多个错误
                            matched_data[review].append(error_info)
                            total_errors += 1
        
        if user_has_target_errors:
            users_with_errors += 1
    
    # 转换为列表格式，每个条目包含review和对应的错误列表
    # 只保留那些至少包含这5种错误类型之一的句子
    result_list = []
    for review, errors in matched_data.items():
        # 过滤：只保留这5种错误类型
        filtered_errors = [
            e for e in errors 
            if e.get('error_type') in allowed_error_types
        ]
        
        # 只保留至少有一个这5种错误的句子
        if filtered_errors:
            # 将review中的换行符替换为空格
            review_single_line = review.replace('\n', ' ').replace('\r', ' ')
            # 去除多余的空格
            review_single_line = ' '.join(review_single_line.split())
            
            result_list.append({
                "review": review_single_line,
                "errors": filtered_errors,
                "error_count": len(filtered_errors)
            })
    
    # 按review排序（为了可重复性）
    result_list.sort(key=lambda x: x['review'])
    
    # 保存到JSON文件
    print(f"\n包含这5种错误类型的用户数: {users_with_errors}")
    print(f"提取了 {len(result_list)} 条唯一的句子")
    print(f"总共 {total_errors} 个错误")
    print(f"正在保存到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)
    
    print(f"完成！结果已保存到: {output_file}")
    print("格式：JSON文件，每个条目包含review和对应的错误列表（只包含这5种错误类型：Deletion、Insertion、Transposition、Scramble、Substitution）")

if __name__ == '__main__':
    input_file = 'spelling_analysis_combined_batch_ulhdemmplr.json'
    output_file = 'extracted_sentences_specific_errors.json'
    
    extract_sentences_with_specific_errors(input_file, output_file)
