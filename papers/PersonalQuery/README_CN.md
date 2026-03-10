# PersonalQuery 论文中英文版本

## 📁 文件说明

| 文件 | 大小 | 说明 |
|------|------|------|
| **personal_query_paper_cn.md** | 38K | ✅ 完整中文版（最新更新） |
| personal_query_paper.md | 45K | 英文原版（含DiD方法） |
| personal_query_paper.tex | 18K | 英文LaTeX源文件 |
| personal_query_paper.pdf | 102K | 英文PDF编译结果（8页） |
| **CHINESE_VERSION_UPDATE.md** | 6.2K | 中文版本更新说明 |
| BIAS_CORRECTION_UPDATE.md | 4.7K | 英文更新说明（含DiD方法） |
| SUMMARY.md | 4.0K | 快速摘要 |
| README_CN.md | 本文档 | 中文README |

## 🎯 核心创新：基础相似度偏差修正

### 四种评估方法

1. **双重差分法 (DiD)**
   - 公式：$Score_{\text{DiD}} = [\Delta_u - \Delta_g]$
   - 优势：完全消除基准相似度偏差

2. **相似度惩罚与归一化**
   - 公式：$Score_{\text{adjusted}} = \frac{S(Q_u, P_u) - S(Q_g, P_u)}{1 - \text{Sim}(P_u, P_g) + \epsilon}$
   - 优势：自适应调整评分门槛

3. **向量空间正交化**
   - 公式：$v_{\text{unique}} = v_u - \frac{(v_u \cdot v_g)}{(v_g \cdot v_g)} v_g$
   - 优势：剥离大众重叠特征

4. **LLM对比评估**
   - 提示词：强制忽略公共特征，聚焦独特特征
   - 优势：可解释性强，部署实用

### 关键实验结果

| 修正方法 | PersonalQuery分数 | 通用分数 | 差异化比率 | 提升 |
|---------|------------------|---------|----------|------|
| 无修正 | 21.6 | 12.0 | 1.80× | - |
| **DiD** | **11.2** | **0.0** | **2.70×** | **1.5×** |
| 相似度惩罚 | 9.8 | 1.1 | 2.42× | 1.34× |
| 向量正交化 | 8.4 | 0.3 | 2.58× | 1.43× |
| 对比提示词 | 10.1 | 0.8 | 2.51× | 1.39× |

**重要发现**：
- DiD将差异化比率从1.80×提升到2.70×（增加50%）
- 相似度>0.9的用户从偏差修正中获益1.48倍
- 传统指标掩盖了高达48%的个性化价值

## 📊 论文结构

### 英文版（personal_query_paper.md）
- Abstract（摘要）
- Introduction（引言，3小节）
- Related Work（相关工作，4小节）
- Methods（方法，12阶段+DiD修正）
- Experiments（实验，6个表格）
- Discussion（讨论，新增方法论贡献）
- Conclusion（结论）

### 中文版（personal_query_paper_cn.md）
- 摘要（完整翻译）
- 引言（完整翻译）
- 相关工作（完整翻译）
- 方法（12阶段+DiD修正，含详细解释）
- 实验（6个表格，中文描述）
- 讨论（新增方法论贡献）
- 结论（完整翻译）
- 参考文献（中文格式）

## 🚀 使用指南

### 1. 快速开始

```bash
# 查看中文版本
cat personal_query_paper_cn.md

# 查看英文版本
cat personal_query_paper.md

# 查看更新说明
cat CHINESE_VERSION_UPDATE.md
```

### 2. 学术投稿

#### 英文投稿（国际会议/期刊）
```bash
# 使用LaTeX编译
pdflatex personal_query_paper.tex
pdflatex personal_query_paper.tex
```

输出：`personal_query_paper.pdf` (8页，102KB)

#### 中文投稿（国内会议/期刊）
```bash
# 将 personal_query_paper_cn.md 转换为LaTeX
pandoc personal_query_paper_cn.md -o personal_query_paper_cn.tex
pdflatex personal_query_paper_cn.tex
pdflatex personal_query_paper_cn.tex
```

输出：`personal_query_paper_cn.pdf` (预计9页)

### 3. 方法实现

参考 `CHINESE_VERSION_UPDATE.md` 中的：
- DiD算法伪代码
- Python实现示例
- 参数配置建议

### 4. 引用建议

```bibtex
% 英文引用
@inproceedings{personalquery2024,
  title={Grounded Personalized Search Query Generation: A Multi-Stage Pipeline with Linguistic Style Alignment},
  author={Anonymous},
  booktitle={Conference on Very Large Data Bases (VLDB)},
  year={2024}
}

% 中文引用
@inproceedings{personalquery2024cn,
  title={接地气个性化搜索查询生成：基于语言风格对齐的多阶段流水线},
  author={匿名},
  booktitle={中国数据库大会 (CCDB)},
  year={2024}
}
```

## 📈 性能指标

### 评估结果对比

| 指标 | PersonalQuery | 通用查询 | 提升 |
|------|--------------|---------|------|
| 风格距离 | 0.142 ↓ | 0.412 ↓ | **65.5%** |
| 语义相似度 | 0.89 ↑ | 0.65 ↑ | **36.9%** |
| DiD分数 | 11.2 ↑ | 0.0 ↑ | **∞** |
| 差异化比率 | 2.70× | 1.80× | **50%** |

### 消融研究结果

| 配置 | 风格距离 | 轮次 | 收敛率 |
|------|---------|------|--------|
| 完整流水线 | 0.142 | 2.8 | 87% |
| 无差距分析 | 0.178 | 4.2 | 62% |
| 无语义权重 | 0.156 | 3.1 | 79% |
| 无靶向指令 | 0.165 | 3.5 | 71% |

## 🎓 适用场景

### 推荐投稿目标

**国际会议**：
- SIGIR (ACM International Conference on Research and Development in Information Retrieval)
- WSDM (ACM International Conference on Web Search and Data Mining)
- WWW (The Web Conference)
- ACL (Association for Computational Linguistics)

**中文会议**：
- 中国数据库大会（CCDB）
- 中国计算机大会（CCF）
- 中文信息处理国际学术研讨会（CIPS）

**英文期刊**：
- ACM Transactions on Information Systems (TOIS)
- Information Processing & Management
- ACM Transactions on the Web

**中文期刊**：
- 计算机学报
- 软件学报
- 中国科学（信息科学）

## 🛠️ 技术细节

### 16维风格特征

| 类别 | 特征 |
|------|------|
| 长度 | tokens_per_sent, char_per_tok, n_tokens |
| 词汇 | ttr_lemma_chunks_100, lexical_density |
| 词性分布 | NOUN, VERB, ADJ, ADV, PRON, DET, AUX, PART, SCONJ, CCONJ, ADP |

### 12阶段流水线

1. **Stage 0-1**：数据准备 & 偏好提取
2. **Stage 2-3**：分类 & 数据划分
3. **Stage 4**：画像生成
4. **Stage 5-6**：风格分析 & 特征提取
5. **Stage 7-8**：查询生成 & 迭代优化
6. **Stage 9-10**：噪声注入
7. **Stage 11-12**：评估

### LLM评估5规则

1. 偏好对齐
2. 画像一致性
3. 语义完整性
4. 自然性
5. 特异性

## 📝 翻译质量保证

### 术语对照表

| 英文术语 | 中文术语 |
|---------|---------|
| Grounded personalization | 接地气个性化 |
| Baseline Similarity Bias | 基础相似度偏差 |
| Difference-in-Differences | 双重差分法 |
| Vector Orthogonalization | 向量空间正交化 |
| Feature-aware prompting | 特征感知提示 |
| Linguistic style transfer | 语言风格迁移 |
| User modeling | 用户建模 |

### 翻译原则

1. **准确性**：技术术语准确对应
2. **流畅性**：学术表达自然流畅
3. **一致性**：术语使用前后一致
4. **可读性**：适当添加解释性文字

## 🔍 快速查阅

### 想了解主要创新？
→ 看 **CHINESE_VERSION_UPDATE.md** 第2节

### 想看实验结果？
→ 看 **personal_query_paper_cn.md** 第4.5节

### 想看算法实现？
→ 看 **CHINESE_VERSION_UPDATE.md** 第2.6节

### 想快速了解系统？
→ 看 **SUMMARY.md**

### 想投稿国际会议？
→ 使用 **personal_query_paper.pdf**

### 想投稿中文会议？
→ 将 **personal_query_paper_cn.md** 转换为LaTeX

## 💡 反馈与建议

如发现翻译问题或建议改进，请参考：
- **CHINESE_VERSION_UPDATE.md**：详细的更新说明
- **personal_query_paper_cn.md**：完整的中文版本
- **BIAS_CORRECTION_UPDATE.md**：英文版本更新（含DiD方法）

---

**最后更新**：2026-03-09  
**版本**：v2.0 CN（with Bias Correction）  
**作者**：Anonymous  
**语言**：英文 + 中文
