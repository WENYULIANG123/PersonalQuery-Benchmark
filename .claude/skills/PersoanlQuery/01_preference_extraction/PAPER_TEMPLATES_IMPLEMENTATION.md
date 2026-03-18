# 论文模板实现指南

**日期**: 2026-03-18  
**基础**: Appendix A - Figure 4 & Figure 5  
**版本**: 1.0

---

## 📄 概览

本指南说明如何在项目中严格使用论文提供的两个LLM提示模板进行实体提取和方面合并。

### 两个模板

| 模板 | 论文位置 | 脚本 | 功能 |
|------|---------|------|------|
| **Template 1** | Figure 4 | `01_aspect_extraction.py` | 从评论中提取方面+情感 |
| **Template 2** | Figure 5 | `01_aspect_consolidation.py` | 合并和规范化方面 |

---

## 🔧 Template 1: 方面抽取

### 核心设计

**输入**: 原始评论文本  
**输出**: 方面列表，每个方面有情感标签

### 提示词结构

```
[基础说明] + [Few-shot示例] + [用户评论] + [输出格式要求]
```

### 实现特点

| 特点 | 说明 |
|------|------|
| **情感分类** | POSITIVE / MIXED / NEGATIVE（3级） |
| **最多方面数** | 5个 |
| **方面粒度** | 词或短语（word or multi-word phrase） |
| **Few-shot示例** | 4个默认示例，可自定义 |

### 使用命令

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python3 .claude/skills/PersoanlQuery/01_preference_extraction/01_aspect_extraction.py \
     --input-file result/personal_query/00_data_preparation/reviews_{USER_ID}.json \
     --output-dir result/personal_query/01_preference_extraction \
     --max-workers 50 \
     --include-other-users"
```

### 输出格式

```json
{
  "user_id": "A123",
  "timestamp": "2026-03-18T10:30:00",
  "template_version": "Appendix_A_Figure_4",
  "total_products": 100,
  "results": [
    {
      "asin": "B01...",
      "product_title": "Glitter Glue",
      "target_aspects": [
        {
          "aspect": "glitter glue",
          "sentiment": "POSITIVE",
          "evidence": "Great glitter glue"
        }
      ],
      "other_aspects": {
        "reviewer_A": [
          {
            "aspect": "bottle size",
            "sentiment": "NEGATIVE",
            "evidence": "The bottle is quite small"
          }
        ]
      },
      "metadata": {
        "extraction_method": "aspect_extraction_v1_paper_template",
        "template_version": "Appendix_A_Figure_4",
        "target_aspects_count": 5,
        "timestamp": "2026-03-18T10:30:00"
      }
    }
  ]
}
```

---

## 🔄 Template 2: 方面合并

### 核心设计

**输入**: 方面列表（可能重复）  
**输出**: 规范化的方面映射（低级 → 高级）

### 提示词结构

```
[基础说明] + [Few-shot示例] + [方面列表] + [输出格式要求]
```

### 实现特点

| 特点 | 说明 |
|------|------|
| **合并方向** | 低级 (specific) → 高级 (general) |
| **输出格式** | JSON object (key-value mapping) |
| **Few-shot示例** | 4个默认示例，可自定义 |
| **双模式支持** | LLM驱动 或 规则驱动 |

### 使用命令

#### 方式1：LLM驱动（更灵活）

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python3 .claude/skills/PersoanlQuery/01_preference_extraction/01_aspect_consolidation.py \
     --input-file result/personal_query/01_preference_extraction/aspects_{USER_ID}.json \
     --output-dir result/personal_query/01_preference_extraction \
     --use-llm"
```

#### 方式2：规则驱动（更快、更便宜）

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python3 .claude/skills/PersoanlQuery/01_preference_extraction/01_aspect_consolidation.py \
     --input-file result/personal_query/01_preference_extraction/aspects_{USER_ID}.json \
     --output-dir result/personal_query/01_preference_extraction"
```

### 输出格式

```json
{
  "user_id": "A123",
  "timestamp": "2026-03-18T10:30:00",
  "template_version": "Appendix_A_Figure_5",
  "consolidation_method": "rule_based",
  "total_products": 100,
  "consolidation_statistics": {
    "total_processed": 100,
    "total_original_aspects": 450,
    "total_canonical_aspects": 280
  },
  "results": [
    {
      "asin": "B01...",
      "product_title": "Glitter Glue",
      "consolidation_mapping": {
        "glitter glue": "glitter_glue",
        "works well": "functionality",
        "beautiful color": "appearance"
      },
      "consolidated_aspects": [
        {
          "aspect": "glitter glue",
          "sentiment": "POSITIVE",
          "aspect_original": "glitter glue",
          "aspect_canonical": "glitter_glue"
        }
      ],
      "consolidation_method": "rule_based",
      "metadata": {
        "original_unique_count": 5,
        "canonical_unique_count": 3,
        "consolidation_rate": 0.4,
        "total_aspects": 8,
        "timestamp": "2026-03-18T10:30:00"
      }
    }
  ]
}
```

---

## 🔗 完整工作流（Template 1 + Template 2）

```
Step 0: 数据准备
  输入: reviews_{USER_ID}.json (Stage 0 输出)
  
  ↓
  
Step 1: 方面抽取 (Template 1)
  脚本: 01_aspect_extraction.py
  输入: 原始评论文本
  输出: aspects_{USER_ID}.json
  
  ↓
  
Step 1.5: 方面合并 (Template 2)
  脚本: 01_aspect_consolidation.py
  输入: aspects_{USER_ID}.json
  输出: consolidated_aspects_{USER_ID}.json
  
  ↓
  
Step 2+: 后续处理
  使用规范化后的方面进行：
  - 个性化分析
  - 聚类
  - 总结生成
  - 评估
```

### 执行时间估算

| 步骤 | 处理时间 | 说明 |
|------|---------|------|
| **Template 1** | 30-60分钟 | 100产品，LLM调用 |
| **Template 2** | 5-15分钟 | 100产品，规则驱动 |
| **Template 2 (LLM)** | 20-40分钟 | 100产品，LLM驱动 |
| **总计** | 35-100分钟 | 取决于所选方式 |

---

## 📊 Template 1 vs Template 2 的对比

### 设计差异

| 维度 | Template 1 | Template 2 |
|------|-----------|-----------|
| **目标** | 抽取原始信息 | 规范化重复信息 |
| **输入** | 自由文本 | 结构化列表 |
| **输出** | 方面+情感 | 方面映射 |
| **调用方向** | 一次/评论 | 一次/产品 |
| **LLM成本** | 高（多个评论） | 中等（单次） |

### 信息流

```
原始评论 → [Template 1] → 原始方面
  ↓
  生成: aspects_{USER_ID}.json
  
规范化 ← [Template 2] ← 原始方面
  ↓
  生成: consolidated_aspects_{USER_ID}.json
```

---

## 🎯 Template 选择指南

### 什么时候使用 Template 1？

✅ **使用场景**:
- 需要原始、细粒度的用户反馈
- 情感分析是核心目标
- 需要保留完整的证据（原始文本）
- 下游任务需要灵活的方面定义

❌ **不适合**:
- 方面已经规范化
- 只需要高级摘要
- 成本优先于精度

### 什么时候使用 Template 2？

✅ **使用场景**:
- 需要规范化和去重
- 构建产品知识库
- 需要跨产品/用户的一致性
- 上游已有方面提取结果

❌ **不适合**:
- 需要保留细粒度差异
- 方面还很新且未标准化
- 不想失去任何信息

### 推荐的组合

**最佳实践**:
```
Template 1 (方面抽取) → Template 2 (规范化) → 后续分析
```

**理由**:
1. Template 1 捕获完整的用户反馈
2. Template 2 创建可操作的规范表示
3. 两者结合兼顾精度和可用性

---

## 🧪 测试和验证

### 单元测试

```bash
# 测试 Template 1 的提示词解析
python3 -c "from 01_aspect_extraction import parse_aspect_extraction_response; ..."

# 测试 Template 2 的规则匹配
python3 -c "from 01_aspect_consolidation import consolidate_aspects_rule_based; ..."
```

### 质量检查

**Template 1 输出验证**:
- [ ] 所有sentiment值在 {POSITIVE, MIXED, NEGATIVE}
- [ ] 最多5个方面/产品
- [ ] 方面字段非空

**Template 2 输出验证**:
- [ ] 所有原始方面都有映射
- [ ] 没有循环映射 (A→B, B→A)
- [ ] 规范化形式使用下划线 (_) 而不是空格

---

## 📈 性能指标

### Template 1 效果

| 指标 | 目标 | 说明 |
|------|------|------|
| **精度 (Precision)** | ≥85% | 提取的方面是正确的 |
| **召回 (Recall)** | ≥75% | 捕获评论中的大部分方面 |
| **情感准确度** | ≥80% | 情感分类正确 |

### Template 2 效果

| 指标 | 目标 | 说明 |
|------|------|------|
| **去重率** | 30-50% | 原始方面 → 规范化方面 |
| **映射准确度** | ≥90% | 低级→高级的映射正确 |
| **覆盖率** | 100% | 所有方面都有映射 |

---

## 🔧 高级用法

### 自定义Few-shot示例

#### Template 1

```python
custom_examples = [
    {
        "review": "Your custom review text here",
        "aspects": [
            {"aspect": "aspect_name", "sentiment": "POSITIVE"}
        ]
    }
]

prompt = get_aspect_extraction_prompt(few_shot_examples=custom_examples)
```

#### Template 2

```python
custom_examples = [
    {
        "low_level": ["custom aspect 1", "custom aspect 2"],
        "high_level": {
            "custom aspect 1": "consolidated_form_1"
        }
    }
]

prompt = get_aspect_consolidation_prompt(
    aspects_to_consolidate=[...],
    few_shot_examples=custom_examples
)
```

### 自定义合并规则

编辑 `consolidate_aspects_rule_based()` 中的 `consolidation_rules` 字典:

```python
consolidation_rules = {
    "your_aspect": "canonical_form",
    "another_aspect": "canonical_form",
    ...
}
```

---

## 🚀 与现有系统的集成

### 与 Stage 1 v2 的关系

| 脚本 | 优势 | 使用场景 |
|------|------|---------|
| **01_extract_preferences_v2** | 维度+方面双视角，置信度评分 | 完整分析 |
| **01_aspect_extraction** | 严格遵循论文模板1 | 学术重现 |
| **01_aspect_consolidation** | 严格遵循论文模板2 | 生产规范化 |

### 推荐的组合

```
场景1: 学术研究
  Template 1 → Template 2 → 论文中的方法论复现

场景2: 生产系统
  Stage 1 v2 (維度+方面) → 后续分析

场景3: 混合方案
  Stage 1 v2 → 验证→ Template 1 + Template 2 → 对比
```

---

## 📋 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| Template 1输出无法解析 | LLM返回非JSON格式 | 检查few-shot示例，调整提示词 |
| Template 2映射不完整 | 某些方面没有规则 | 添加更多规则或使用LLM模式 |
| 合并率太高/太低 | 规则设置不当 | 调整阈值，审查规则库 |
| LLM调用超时 | 方面列表过大 | 分批处理 |

---

## 📚 参考

**论文原始位置**:
- Template 1: Appendix A - Figure 4
- Template 2: Appendix A - Figure 5

**相关论文**:
- [2024] A Systematic Review of Aspect-Based Sentiment Analysis
- [2020] Issues and Challenges of Aspect-Based Sentiment Analysis

---

**版本历史**

| 版本 | 日期 | 内容 |
|------|------|------|
| 1.0 | 2026-03-18 | 初始实现：Template 1 + Template 2 |

---

当前任务已完成，请做下一个任务的指示。
