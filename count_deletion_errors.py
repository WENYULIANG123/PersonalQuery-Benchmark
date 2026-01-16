#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 spelling_analysis_combined_batch_ulhdemmplr.json 中包含 Deletion 错误的句子数量
"""

import json

def count_deletion_errors(input_file):
    """
    统计包含 Deletion 错误的句子数量
    """
    # 使用set存储包含Deletion错误的唯一句子
    deletion_reviews = set()
    
    # 统计信息
    total_deletion_errors = 0
    users_with_deletion = 0
    
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共 {len(data.get('users', []))} 个用户")
    print("\n正在处理...")
    
    # 遍历所有用户
    for user in data.get('users', []):
        error_types = user.get('error_types', {})
        deletion_errors = error_types.get('Deletion', [])
        
        if deletion_errors and len(deletion_errors) > 0:
            users_with_deletion += 1
            user.get('user_id', 'unknown')
            
            # 提取所有包含Deletion错误的句子
            for error_item in deletion_errors:
                if isinstance(error_item, dict) and 'review' in error_item:
                    review = error_item.get('review', '').strip()
                    if review:
                        deletion_reviews.add(review)
                        total_deletion_errors += 1
    
    print("\n统计结果:")
    print(f"包含 Deletion 错误的用户数: {users_with_deletion}")
    print(f"包含 Deletion 错误的唯一句子数: {len(deletion_reviews)}")
    print(f"Deletion 错误总数: {total_deletion_errors}")
    
    return len(deletion_reviews), total_deletion_errors, users_with_deletion

if __name__ == '__main__':
    input_file = 'spelling_analysis_combined_batch_ulhdemmplr.json'
    count_deletion_errors(input_file)
