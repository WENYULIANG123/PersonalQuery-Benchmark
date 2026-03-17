#!/usr/bin/env python3
import json
from pathlib import Path

holdout_file = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data/user_A13OFOB1394G31/holdout.json')

with open(holdout_file) as f:
    holdout = json.load(f)['pairs']

print("="*80)
print("HOLDOUT 数据分析")
print("="*80)

print("\n前3个holdout样本:")
for i, pair in enumerate(holdout[:3], 1):
    print(f"\n样本 {i}:")
    print(f"  query (CLEAN)  : {pair['query'][:60]}...")
    print(f"  positive (NOISY): {pair['positive'][:60]}...")
    print(f"  ASIN: {pair['asin']}")

print("\n" + "="*80)
print("评估逻辑问题")
print("="*80)

print("""
当前评估代码 (第372行):
  query = pair.get('query', '')  # 使用CLEAN query
  model.encode(query)            # 编码clean query
  
这意味着:
  ✗ 没有测试模型对NOISY query的理解能力
  ✗ 无法验证noisy→clean映射是否学成功
  
正确的评估应该:
  ✓ 用NOISY query来编码
  ✓ 检索文档
  ✓ 看能否找到正确商品
  
这才能测试"noisy query correction"的效果！
""")

print("="*80)
print("重新分析结果")
print("="*80)

print("""
Experiment A 为什么有效？
  - 用User A的(clean, noisy)对训练
  - 评估用User A的clean query
  - 但由于训练数据很少(36个样本)
  - 模型可能过拟合到User A的特定领域(card making)
  - 所以在这个领域的检索能工作

Experiment B 为什么无效？
  - 用其他10个用户的(clean, noisy)对训练
  - 这些用户可能来自不同的领域/产品类别
  - 评估用User A的clean query
  - 模型被混合的信号搞confused
  - 在任何领域都无法给出有意义的检索结果
  - MRR = 0

真正的问题: 评估逻辑没有测试模型的核心目标！
""")

