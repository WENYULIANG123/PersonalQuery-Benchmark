#!/usr/bin/env python3
"""简单对比输入和生成结果"""

import json

# 读取预测结果
with open('/home/wlia0047/ar57_scratch/wenyu/lingconv_predictions.txt') as f:
    preds = [line.strip() for line in f if line.strip()]

# 加载数据集信息
import sys
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis')

# 读取训练数据的前几个样本作为参考
from transformers import T5Tokenizer
from datasets import load_dataset

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# 直接加载原始数据
data_files = {
    "train": "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis/data/ling_conversion/train.parquet"
}
ds = load_dataset("parquet", data_files=data_files)["train"]

print("对比输入和生成结果：")
print("=" * 80)
for i in range(min(10, len(preds), len(ds))):
    sample = ds[i]
    src = sample.get('sentence1', sample.get('input', 'N/A'))[:80]
    tgt = sample.get('sentence2', sample.get('target', 'N/A'))[:80]
    pred = preds[i] if i < len(preds) else 'N/A'
    print(f"样本 {i}:")
    print(f"  输入: {src}")
    print(f"  目标: {tgt}")
    print(f"  生成: {pred}")
    print()
