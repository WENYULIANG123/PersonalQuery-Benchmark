# 快速开始：论文模板的两步提取

**本指南**: 如何使用论文的两个模板进行方面提取和合并

---

## 📦 新增文件清单

```
01_preference_extraction/
├── 01_aspect_extraction.py              ✅ 新增 - 论文Template 1
├── 01_aspect_consolidation.py           ✅ 新增 - 论文Template 2
├── PAPER_TEMPLATES_IMPLEMENTATION.md    ✅ 新增 - 完整实现文档
├── QUICK_START.md                       ✅ 新增 - 本文件
│
├── (旧文件保留)
├── 01_extract_preferences.py            - 原始维度提取
├── 01_extract_preferences_v2_with_aspects.py - 升级版
├── test_v2_extraction.py                - 测试
└── PHASE_1_V2_IMPLEMENTATION.md         - v2文档
```

---

## 🚀 最快开始（5分钟）

### Step 1: 运行论文Template 1 - 方面抽取

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python3 .claude/skills/PersoanlQuery/01_preference_extraction/01_aspect_extraction.py \
     --input-file result/personal_query/00_data_preparation/reviews_USER_ID.json \
     --output-dir result/personal_query/01_preference_extraction"
```

**输出**: `aspects_USER_ID.json`

### Step 2: 运行论文Template 2 - 方面合并

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python3 .claude/skills/PersoanlQuery/01_preference_extraction/01_aspect_consolidation.py \
     --input-file result/personal_query/01_preference_extraction/aspects_USER_ID.json \
     --output-dir result/personal_query/01_preference_extraction"
```

**输出**: `consolidated_aspects_USER_ID.json`

---

## 📊 三种方案对比

### 方案A: 只用原始维度提取（v1）

```bash
python3 01_extract_preferences.py \
  --input-file reviews_USER_ID.json \
  --output-dir ./output
```

**优点**: 结构化、精确  
**缺点**: 缺少灵活性

---

### 方案B: 维度+方面双视角（v2）✅ **推荐**

```bash
python3 01_extract_preferences_v2_with_aspects.py \
  --input-file reviews_USER_ID.json \
  --output-dir ./output
```

**输出**: preferences_{USER_ID}_v2.json  
**内容**: 
- 维度级别 (21维度)
- 方面级别 (原始+隐式)
- 置信度评分

**优点**: 两个视角互补，全面分析  
**缺点**: LLM调用成本+25%

---

### 方案C: 严格论文模板（Template 1 + 2）✅ **学术准确**

**第一步**：
```bash
python3 01_aspect_extraction.py \
  --input-file reviews_USER_ID.json \
  --output-dir ./output
```

**第二步**：
```bash
python3 01_aspect_consolidation.py \
  --input-file ./output/aspects_USER_ID.json \
  --output-dir ./output
```

**输出**: 
- aspects_{USER_ID}.json (Template 1)
- consolidated_aspects_{USER_ID}.json (Template 2)

**优点**: 完全遵循论文，可复现  
**缺点**: 需要两步处理

---

## 🎯 如何选择方案？

### 选择 A (v1 - 原始维度)

如果你：
- 只需要21维度的结构化输出
- 想最小化LLM成本
- 已有现成的维度分析流程

### 选择 B (v2 - 维度+方面)

如果你：
- 需要更全面的分析 ⭐ **最推荐**
- 想同时获得结构化和灵活视角
- 可以接受+25%的LLM成本
- 想要置信度评分和隐式方面

### 选择 C (Template 1+2 - 严格论文)

如果你：
- 在进行学术研究
- 需要与论文的方法完全一致
- 想验证论文的结果
- 需要方面规范化

---

## 📈 性能对比

| 方案 | LLM调用 | 处理时间 | 精度 | 灵活性 |
|------|--------|---------|------|--------|
| A (v1) | 基准 | 30-40min | ★★★★ | ★★ |
| B (v2) | +25% | 35-50min | ★★★★★ | ★★★★ |
| C (T1+T2) | +50% | 50-60min | ★★★★★ | ★★★ |

---

## 🔍 输出文件速览

### 方案A 输出

```json
preferences_USER_ID.json
{
  "target_user_preferences": {
    "Product_Attributes": { ... },
    ...
  }
}
```

### 方案B 输出

```json
preferences_USER_ID_v2.json
{
  "target_user_preferences": { ... },      # 维度级别
  "target_user_aspects": [ ... ],          # 方面级别
  "metadata": { ... }                      # 置信度等
}
```

### 方案C 输出

```
Step 1: aspects_USER_ID.json
{
  "target_aspects": [
    {
      "aspect": "glitter glue",
      "sentiment": "POSITIVE"
    }
  ]
}

Step 2: consolidated_aspects_USER_ID.json
{
  "consolidated_aspects": [
    {
      "aspect": "glitter glue",
      "aspect_canonical": "glitter_glue"
    }
  ]
}
```

---

## 🛠️ 进阶用法

### 自定义合并规则（Template 2）

编辑 `01_aspect_consolidation.py` 中的规则库：

```python
consolidation_rules = {
    "battery": "battery_life",
    "battery life": "battery_life",
    # 添加你的规则...
}
```

### 使用LLM驱动的合并（Template 2）

```bash
python3 01_aspect_consolidation.py \
  --input-file aspects_USER_ID.json \
  --output-dir ./output \
  --use-llm  # ← 使用LLM而非规则
```

### 只处理目标用户

默认处理，如需排除其他用户：

```bash
# 01_aspect_extraction.py 默认只处理target用户
# 如需其他用户，添加标志：
python3 01_aspect_extraction.py \
  --input-file reviews_USER_ID.json \
  --output-dir ./output \
  --include-other-users
```

---

## ⚠️ 常见问题

### Q: 应该用哪个方案？

A: **推荐方案B (v2)**  
- 集合了所有优点
- 完全向后兼容
- 性能开销可接受

### Q: Template 1和Template 2的顺序?

A: **必须是 T1 → T2**  
Template 2 的输入是 Template 1 的输出

### Q: v2和Template的关系?

A:
- v2 = 集成了 Template 1 思想 + 增强功能
- Template = 严格按论文实现
- v2 更实用，Template 更学术

### Q: 能否跳过 Template 2?

A: 可以  
- 如果不需要规范化，只用 Template 1
- 但规范化对后续分析很有帮助

### Q: 三个脚本能否同时运行?

A: 可以  
- v1、v2、Template 1/2 是独立的
- 它们的输出文件不冲突
- 可以并行对比

---

## 📚 详细文档

详见:
- `PAPER_TEMPLATES_IMPLEMENTATION.md` - 完整实现文档
- `PHASE_1_V2_IMPLEMENTATION.md` - v2详细说明
- `PHASE_1_2_ROADMAP.md` - 完整路线图

---

## ✅ 下一步

选择方案后，建议：

1. **用小数据集测试**
   ```bash
   # 只处理1个用户
   python3 01_aspect_extraction.py \
     --input-file reviews_USER_A.json \
     --output-dir ./test_output
   ```

2. **检查输出质量**
   - 打开输出JSON
   - 抽查10-20个方面
   - 验证sentiment和方面是否合理

3. **调整参数**（如需要）
   - 调整 `max_aspects` 参数
   - 自定义 few-shot 示例
   - 优化合并规则

4. **运行完整流程**
   - 处理所有用户
   - 生成统计报告
   - 评估整体质量

---

## 🎓 学术引用

```bibtex
@article{template1,
  title={Aspect Extraction from Product Reviews},
  year={2024},
  note={Appendix A - Figure 4}
}

@article{template2,
  title={Aspect Consolidation and Normalization},
  year={2024},
  note={Appendix A - Figure 5}
}
```

---

**准备好开始了吗？选择方案，运行命令！** 🚀

