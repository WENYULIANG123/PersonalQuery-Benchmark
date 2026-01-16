#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证提取的错误类型
"""

import json
from collections import Counter

data = json.load(open('extracted_sentences_specific_errors.json'))

print(f'总条目数: {len(data)}')
print(f'总错误数: {sum(item["error_count"] for item in data)}')

error_types = []
[error_types.extend([e['error_type'] for e in item['errors']]) for item in data]

print('\n错误类型统计:')
print(dict(Counter(error_types)))
print(f'\n所有错误类型: {set(error_types)}')

print('\n各错误类型详细统计:')
for et in ['Deletion', 'Insertion', 'Transposition', 'Scramble', 'Substitution']:
    count = error_types.count(et)
    sentences = len([item for item in data if any(e['error_type'] == et for e in item['errors'])])
    print(f'  {et}: {count}个错误, {sentences}个句子')
