# Stage 1: Preference & Aspect Extraction (v2.1)

**生产级偏好提取系统** - 支持fail-fast错误处理和NaN缺失维度统计

---

## 📌 当前实现

仅保留最新逻辑脚本：

- **`01_extract_preferences_v2_with_aspects.py`** (v2.1) ⭐
  - ✅ Fail-fast错误处理（不允许fallback）
  - ✅ NaN值记录与统计（缺失维度追踪）
  - ✅ 维度+方面双视角提取
  - ✅ 置信度评分和隐式方面检测
  - ✅ 完整的质量检查和验证

---

## 🚀 快速开始

```bash
# 单用户提取 (225个产品 × 5并发)
python -u 01_extract_preferences_v2_with_aspects.py \
  --input-file /path/to/reviews_<USER_ID>.json \
  --output-dir ./output
```

---

## 📂 目录结构

```
01_preference_extraction/
│
├─ 01_extract_preferences_v2_with_aspects.py ⭐ (主脚本)
├─ README.md (本文件)
│
└─ _archived/
   ├─ 01_extract_preferences.py (v1 - 基础)
   ├─ 01_extract_preferences_fixed.py
   ├─ 01_aspect_extraction.py (论文Template 1)
   ├─ 01_aspect_consolidation.py (论文Template 2)
   ├─ test_v2_extraction.py
   └─ ... (其他历史版本)
```
│  ├─ 01_aspect_extraction.py
│  │  ↳ 论文Template 1: 方面抽取
│  ├─ 01_aspect_consolidation.py
│  │  ↳ 论文Template 2: 方面合并
│  ├─ PAPER_TEMPLATES_IMPLEMENTATION.md
│  │  ↳ 完整实现文档
│  └─ QUICK_START.md
│     ↳ 快速参考指南（强烈推荐先读）
│
└─ 【规划文档】
   ├─ PHASE_1_V2_IMPLEMENTATION.md (v2总结)
   └─ PHASE_1_2_ROADMAP.md (Phase 2-4路线图)
```

---

## 🚀 30秒快速开始

### 最快方案（方案B）

```bash
# 一行命令，完成方面+维度双视角提取
python3 01_extract_preferences_v2_with_aspects.py \
  --input-file reviews_USER_ID.json \
  --output-dir ./output
```

**输出**: `preferences_{USER_ID}_v2.json`  
**内含**: 维度级别 + 方面级别 + 置信度 + 隐式方面

---

### 严格论文实现（方案C）

```bash
# Step 1: 提取方面 (Template 1)
python3 01_aspect_extraction.py \
  --input-file reviews_USER_ID.json \
  --output-dir ./output

# Step 2: 合并规范化 (Template 2)
python3 01_aspect_consolidation.py \
  --input-file ./output/aspects_USER_ID.json \
  --output-dir ./output
```

**输出**: 
- `aspects_{USER_ID}.json` (Template 1)
- `consolidated_aspects_{USER_ID}.json` (Template 2)

---

## 📖 文档导读顺序

### 如果你有 5 分钟

1. 阅读本文件的"方案对比"部分
2. 阅读 `QUICK_START.md`
3. 运行对应的命令

### 如果你有 30 分钟

1. 阅读本文件
2. 阅读 `QUICK_START.md`
3. 根据选择阅读：
   - 方案B: `PHASE_1_V2_IMPLEMENTATION.md`
   - 方案C: `PAPER_TEMPLATES_IMPLEMENTATION.md`

### 如果你有 2 小时（深入理解）

1. 阅读本文件
2. 阅读 `PHASE_1_V2_IMPLEMENTATION.md` (了解v2)
3. 阅读 `PAPER_TEMPLATES_IMPLEMENTATION.md` (了解论文实现)
4. 阅读 `PHASE_1_2_ROADMAP.md` (了解下一步)
5. 浏览源代码（有详细注释）

---

## 📊 方案对比

### 方案 A: 维度提取 (v1)

**文件**: `01_extract_preferences.py`

| 特性 | 评分 |
|------|------|
| 精确性 | ⭐⭐⭐⭐ |
| 灵活性 | ⭐⭐ |
| 成本 | 最低 |
| 文档 | ⭐⭐⭐ |

**输出**: `preferences_{USER_ID}.json`

```json
{
  "target_user_preferences": {
    "Product_Attributes": { ... },
    "Quality_Attributes": { ... },
    ...
  }
}
```

**适合**: 
- 只需要21维度结构化输出
- 成本优先

---

### 方案 B: 维度+方面双视角 (v2) ⭐ **推荐**

**文件**: `01_extract_preferences_v2_with_aspects.py`

| 特性 | 评分 |
|------|------|
| 精确性 | ⭐⭐⭐⭐⭐ |
| 灵活性 | ⭐⭐⭐⭐ |
| 成本 | 适中 (+25%) |
| 文档 | ⭐⭐⭐⭐⭐ |

**输出**: `preferences_{USER_ID}_v2.json`

```json
{
  "target_user_preferences": { ... },     // 维度级别
  "target_user_aspects": [ ... ],         // 方面级别
  "quality_check": { ... },               // 质量检查
  "metadata": { ... }                     // 元数据
}
```

**优点**:
- ✅ 两个互补视角
- ✅ 置信度评分
- ✅ 隐式方面检测
- ✅ 完全向后兼容
- ✅ 质量自检

**适合**:
- 需要全面分析
- 生产系统
- 学术+工业结合

---

### 方案 C: 论文模板严格实现 (Template 1+2) ⭐ **学术准确**

**文件**: 
- `01_aspect_extraction.py` (Template 1)
- `01_aspect_consolidation.py` (Template 2)

| 特性 | 评分 |
|------|------|
| 学术对齐 | ⭐⭐⭐⭐⭐ |
| 灵活性 | ⭐⭐⭐ |
| 成本 | 中高 (+50%) |
| 文档 | ⭐⭐⭐⭐⭐ |

**输出**: 
- Step 1: `aspects_{USER_ID}.json` (Template 1)
- Step 2: `consolidated_aspects_{USER_ID}.json` (Template 2)

```json
// Step 1 输出
{
  "target_aspects": [
    {
      "aspect": "glitter glue",
      "sentiment": "POSITIVE"
    }
  ]
}

// Step 2 输出
{
  "consolidated_aspects": [
    {
      "aspect": "glitter glue",
      "aspect_canonical": "glitter_glue"
    }
  ]
}
```

**优点**:
- ✅ 100% 论文实现
- ✅ 可复现、可引用
- ✅ 规范化合并
- ✅ 灵活的few-shot和规则

**适合**:
- 学术研究
- 论文复现
- 需要完整的规范化流程

---

## 🎓 了解更多

### 论文背景

本实现基于以下学术文献：

1. **Appendix A - Figure 4**: 方面抽取提示模板 (Template 1)
2. **Appendix A - Figure 5**: 方面合并提示模板 (Template 2)
3. **Appendix B - Table 2**: 评估错误分类框架

### 相关研究

- [2024] A Systematic Review of Aspect-Based Sentiment Analysis
- [2025] Error Comparison Optimization for LLMs on ABSA (ACL)
- [2022] Amazon - Distantly Supervised Aspect Clustering (NAACL)

---

## 🔗 工作流整合

### 与其他阶段的关系

```
Stage 0: 数据准备
    ↓ reviews_{USER_ID}.json
    
Stage 1: 方面&偏好提取 ← 你在这里
    ├─ 方案A: 维度提取 (v1)
    ├─ 方案B: 维度+方面 (v2) ⭐
    └─ 方案C: 论文模板 (T1+T2) ⭐
    ↓ preferences/aspects_{USER_ID}.json
    
Stage 1.5: 方面合并 (可选，方案C第二步)
    ↓ consolidated_aspects_{USER_ID}.json
    
Stage 2+: 后续处理 (个性化、聚类、评估)
```

---

## ❓ 常见问题

**Q: 应该选择哪个方案？**  
A: 大多数情况选 **方案B**。如果是学术研究选 **方案C**。

**Q: 三个方案能否共存？**  
A: 可以！它们使用不同的输出文件名，不会冲突。

**Q: v2 和 Template 的区别？**  
A: v2=集成论文思想+增强功能。Template=严格按论文。

**Q: 成本多少？**  
A: 方案B: +25% | 方案C: +50% (相对方案A)

**详见**: `QUICK_START.md` - 常见问题部分

---

## ✅ 检查清单

在运行任何方案之前：

- [ ] 已安装 llm_client 依赖
- [ ] 有有效的 reviews_{USER_ID}.json 输入文件
- [ ] 输出目录已创建或有写入权限
- [ ] 阅读了对应方案的文档

---

## 🚀 现在开始

1. **确定你的需求** → 选择方案 (A/B/C)
2. **阅读对应文档**
3. **运行示例命令**
4. **检查输出质量**
5. **整合到你的流程**

**推荐路径**:
```
Quick Start → QUICK_START.md → 选择方案 → 运行命令 → 检查输出
```

---

**版本**: v1.0  
**日期**: 2026-03-18  
**维护**: PersoanlQuery Team

---

> 💡 **提示**: 开始前强烈推荐阅读 `QUICK_START.md` - 5分钟内了解全部选项！
