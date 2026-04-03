#!/usr/bin/env python3
"""检查训练数据样本"""

from datasets import load_from_disk

data = load_from_disk('./data/ling_conversion')
print('Columns:', data['train'].column_names)

# 直接迭代
train_data = list(data['train'])
print(f'\n训练集大小: {len(train_data)}')

print('\n前10个训练样本:')
for i in range(10):
    row = train_data[i]
    s1 = row['sentence1']
    s2 = row['sentence2']
    source = row['source']

    # 计算前缀相同比例
    words1 = s1.lower().split()
    words2 = s2.lower().split()
    prefix_len = 0
    for j in range(min(len(words1), len(words2))):
        if words1[j] == words2[j]:
            prefix_len += 1
        else:
            break

    ratio = prefix_len / len(words1) if words1 else 0

    print(f'\n样本{i+1} [{source}]:')
    print(f'  s1: {s1}')
    print(f'  s2: {s2}')
    print(f'  前缀相同: {prefix_len}/{len(words1)} ({ratio*100:.1f}%)')
