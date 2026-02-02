---
name: Product-Attribute-Sample
description: 从 AmazonSKB 中随机提取 500 个商品的 3 个属性及最小类目，并生成 CSV。要求商品属性不少于 3 个。
allowed-tools: Bash(python:*)
---

# Product-Attribute-Sample

此技能用于快速从 AmazonSKB 中采样商品数据及其属性，主要用于测试或生成查询输入数据。该技能会自动过滤掉属性少于 3 个的商品。

## 核心流程

当你需要采样商品数据时，执行以下步骤：

1.  **触发采样**：运行内置脚本进行数据提取。
2.  **结果导出**：脚本会自动将结果保存为 CSV 文件。

## 使用方式

### 直接执行采样
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && conda activate /home/wlia0047/ar57_scratch/wenyu/stark && python -u /home/wlia0047/ar57/wenyu/.claude/skills/product-attribute-sample/sample_products.py --size 500"
```

### 参数说明
- `--size`: 采样商品数量，默认为 500。
- `--output`: 输出 CSV 路径，默认为 `/home/wlia0047/ar57/wenyu/result/sample_product_attributes.csv`。

## [CRITICAL] 执行标准
- **强制使用 Wrapper**：必须通过 `sbatch_wrapper.py` 运行 Python 脚本，以确保资源分配和环境正确。
- **路径规范**：所有输出文件应默认保存在 `/home/wlia0047/ar57/wenyu/result/` 目录下。
