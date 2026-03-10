# 📄 PersonalQuery 论文完整清单

## 🎉 文件列表（按重要性排序）

### 核心论文文件
1. ✅ **personal_query_paper_cn.md** (38K) - 完整中文版（最新）
2. ✅ **personal_query_paper.md** (45K) - 完整英文版（含DiD方法）
3. ✅ **personal_query_paper.pdf** (101K) - 英文PDF编译结果
4. ✅ **personal_query_paper.tex** (18K) - 英文LaTeX源文件

### 更新说明文档
5. ✅ **CHINESE_VERSION_UPDATE.md** (6.2K) - 中文版本详细更新说明
6. ✅ **BIAS_CORRECTION_UPDATE.md** (4.7K) - 英文版本详细更新说明

### 辅助文档
7. ✅ **README_CN.md** (7.2K) - 中文版本使用指南
8. ✅ **SUMMARY.md** (3.9K) - 快速摘要
9. ✅ **references.bib** (2.4K) - BibTeX参考文献

---

## 🚀 快速开始

### 1. 查看论文

```bash
# 中文版（推荐）
cat personal_query_paper_cn.md

# 英文版
cat personal_query_paper.md
```

### 2. 编译LaTeX

```bash
cd /fs04/ar57/wenyu/papers/PersonalQuery

# 英文版本
pdflatex personal_query_paper.tex
pdflatex personal_query_paper.tex

# 中文版本（转换Markdown为LaTeX）
pandoc personal_query_paper_cn.md -o personal_query_paper_cn.tex
pdflatex personal_query_paper_cn.tex
pdflatex personal_query_paper_cn.tex
```

---

## 🎯 核心创新：基础相似度偏差修正

### 四种评估方法

| 方法 | 优势 | 适用场景 |
|------|------|---------|
| **DiD** | 最严谨，完全消除偏差 | 基准比较 |
| **向量正交化** | 无需重训练 | 嵌入评估 |
| **对比提示** | 可解释性强 | 实际部署 |
| **相似度惩罚** | 自适应调整 | 连续评分 |

### 关键实验结果

```
优化前：差异化比率 = 1.80×
优化后：差异化比率 = 2.70×  ↑ 50%

DiD评分提升：
PersonalQuery: 11.2分
通用查询: 0.0分
差异化: ∞

基础相似度>0.9的用户：
原始改进：+8.2分
DiD改进：+12.1分
放大倍数：1.48×
```

---

## 📊 论文对比

| 方面 | 英文版 | 中文版 |
|------|--------|--------|
| 总页数 | 8页 | 9页（LaTeX编译） |
| 文件大小 | 45K (MD) | 38K (MD) |
| 术语 | 大量英文 | 英文+中文 |
| 结构 | 完整 | 完整 |
| 表格 | 6个 | 6个 |
| 代码示例 | 英文 | 英文+中文 |

---

## 🔍 按需求查阅

### 📖 想了解论文内容？

1. **快速了解**：阅读 `SUMMARY.md`
2. **详细阅读**：阅读 `personal_query_paper_cn.md`
3. **英文对比**：阅读 `personal_query_paper.md`

### 💡 想了解创新方法？

1. **中文详解**：`CHINESE_VERSION_UPDATE.md` 第1节
2. **英文详解**：`BIAS_CORRECTION_UPDATE.md` 第1节
3. **代码实现**：`CHINESE_VERSION_UPDATE.md` 第2.6节

### 📝 想要投稿？

1. **国际会议**：使用 `personal_query_paper.pdf`
2. **中文会议**：转换 `personal_query_paper_cn.md` 为LaTeX
3. **详细参数**：查阅 `CHINESE_VERSION_UPDATE.md`

### 🔬 想看实验数据？

所有数据都在：
- `personal_query_paper_cn.md` 第4.5节
- `personal_query_paper.md` 第4.5节
- 包含6个详细表格

---

## 📈 关键数据

### 评估结果

| 方法 | 风格距离 ↓ | 语义相似度 ↑ | DiD分数 |
|------|-----------|-------------|---------|
| 通用查询 | 0.412 | 0.65 | 0.0 |
| 仅属性 | 0.298 | 0.78 | 4.2 |
| 单轮LLM | 0.215 | 0.82 | 6.8 |
| **PersonalQuery** | **0.142** | **0.89** | **11.2** |

### 收敛分析

| 轮次 | 风格距离 | 改进 | 收敛率 |
|------|---------|------|--------|
| 0 | 0.287 | - | 0% |
| 1 | 0.198 | +0.089 | 42% |
| 2 | 0.156 | +0.042 | 71% |
| 3 | 0.142 | +0.014 | 87% |
| 4 | 0.139 | +0.003 | 92% |
| 5 | 0.138 | +0.001 | 95% |

### 偏差修正效果

| 相似度范围 | 用户数 | 原始增益 | DiD增益 | 放大倍数 |
|-----------|--------|---------|---------|----------|
| [0.9, 1.0] | 12 | +8.2 | +12.1 | **1.48×** |
| [0.8, 0.9) | 18 | +9.1 | +11.5 | 1.26× |
| [0.7, 0.8) | 14 | +10.3 | +11.0 | 1.07× |
| [0.6, 0.7) | 6 | +11.8 | +11.2 | 0.95× |

---

## 🛠️ 技术规格

### 16维风格特征

```
长度特征：
- tokens_per_sent（平均每句词数）
- char_per_tok（平均每词字符数）
- n_tokens（总词数）

词汇特征：
- ttr_lemma_chunks_100（类型-标记比）
- lexical_density（词汇密度）

词性分布：
- upos_dist_NOUN, VERB, ADJ, ADV, PRON
- upos_dist_DET, AUX, PART, SCONJ, CCONJ, ADP
```

### 12阶段流水线

```
0-1: 数据准备 & 偏好提取（LLM）
2-3: 分类 & 数据划分（Target/Public）
4: 接地气画像生成（技能/使用场景/情感）
5-6: 风格分析与特征提取（16维）
7-8: 查询生成 & 迭代优化（特征感知）
9-10: 噪声注入（CNN拼写模型）
11-12: 评估（LLM 5规则 + DiD修正）
```

---

## 🎓 推荐阅读顺序

### 第一次阅读（快速了解）

1. `SUMMARY.md` - 5分钟快速了解
2. `personal_query_paper_cn.md` - 阅读摘要和结论
3. `README_CN.md` - 了解如何使用

### 第二次阅读（深入理解）

1. `CHINESE_VERSION_UPDATE.md` - 了解创新方法
2. `personal_query_paper_cn.md` - 阅读全文（特别是第4.5节）
3. `BIAS_CORRECTION_UPDATE.md` - 英文版详细说明

### 第三次阅读（深度研究）

1. `personal_query_paper_cn.md` - 逐节精读
2. `CHINESE_VERSION_UPDATE.md` - 实验数据和算法
3. `README_CN.md` - 技术细节和使用指南

---

## 📝 投稿建议

### 国际会议（推荐）

1. **SIGIR** - 信息检索顶会
2. **WSDM** - Web搜索和数据挖掘
3. **WWW** - Web研究顶会
4. **ACL** - 自然语言处理顶会

### 中文会议（推荐）

1. **中国数据库大会（CCDB）**
2. **中国计算机大会（CCF）**
3. **中文信息处理国际学术研讨会（CIPS）**

### 投稿材料

**英文投稿**：
- ✅ `personal_query_paper.pdf`（直接使用）
- ✅ `personal_query_paper.tex`（源文件）

**中文投稿**：
- ✅ `personal_query_paper_cn.md`（转换为LaTeX）
- ✅ `README_CN.md`（使用指南）

---

## 🌟 核心贡献

### 1. 方法论创新

✅ **首次识别**个性化评估中的基础相似度偏差问题
✅ **提出四种**互补的偏差修正方法
✅ **实证证明**传统指标系统性低估个性化价值

### 2. 实验验证

✅ **DiD评分**将差异化比率从1.8×提升到2.7×（+50%）
✅ **相似度>0.9的用户**从偏差修正中获益1.48倍
✅ **掩盖的个性化价值高达48%**，传统评估严重低估

### 3. 实践意义

✅ 为个性化系统评估提供更严谨的方法
✅ 揭示"主流用户"评估的盲区
✅ 提供可复用的评估框架

---

## 📞 联系方式

**论文作者**：Anonymous
**最后更新**：2026-03-09
**版本**：v2.0 CN（with Bias Correction）

---

## 🎉 总结

你现在拥有了：

✅ **1个完整的中文论文**（38K，含DiD方法）
✅ **1个完整的英文论文**（45K，含DiD方法）
✅ **1个LaTeX源文件**（可重新编译）
✅ **1个PDF编译结果**（101K，8页）
✅ **3个详细更新说明**（中文+英文）
✅ **1个使用指南**（中文）
✅ **1个快速摘要**（方便分享）

**推荐立即使用**：
- 中文版本：`personal_query_paper_cn.md`
- 英文版本：`personal_query_paper.pdf`

**特别强调**：
核心创新是**基础相似度偏差修正方法**，这是对个性化评估的重大改进！

---

*Created by AI Assistant | PersonalQuery v2.0*
