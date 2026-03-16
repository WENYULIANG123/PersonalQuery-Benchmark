# Stage 10 噪声注入逻辑完整分析

## 📋 概述

**文件**: `10_generate_noisy_queries.py`
**策略**: 纯个性化拼写错误注入（基于用户真实历史）
**修改率**: 约 35% (16/46 查询)

---

## 🎯 核心逻辑流程

### 1. 初始化阶段

```python
# 使用 HybridErrorInjector
injector = HybridErrorInjector(WRITING_ANALYSIS_FILE)
```

**HybridErrorInjector** 特点：
- ✅ 仅使用个性化拼写错误（从 Stage 5 writing analysis）
- ❌ 禁用语法错误注入
- ❌ 禁用统计错误注入
- ✅ 如果无法匹配，保持查询不变（不强制注入）

### 2. 错误提取器（PersonalizedSpellingErrorExtractor）

从用户写作历史中提取拼写错误模式：

**数据来源**: Stage 5 writing_analysis_A13OFOB1394G31.json

**提取逻辑**:
```python
错误模式 = {
    "正确单词": "用户的错误拼写"
}

例如：
{
    "machine": "maching",
    "gluing": "glueing",
    "sympathy": "sympathy"  # 注意：某些看起来相同，但可能有Unicode差异
}
```

### 3. 错误注入器（PersonalizedSpellingInjector）

**注入策略**:

#### 步骤1: 精确匹配
```python
if query中的词 in 用户错误模式:
    将正确词 → 错误拼写
```

#### 步骤2: 模糊匹配（相似度 ≥ 0.75）
```python
if similarity(query中的词, 用户错误的"正确修正") ≥ 0.75:
    将正确词 → 错误拼写
```

**示例**:
- 用户错误: "recieve" → "receive"
- 查询词: "received" (与 "receive" 相似度 0.83)
- 结果: "received" → "recieved"

### 4. 目标词选择策略

**选择条件**:
1. 词长 ≥ 4个字母
2. 词必须全是字母（纯数字排除）
3. 必须在用户错误模式中（精确或模糊匹配）

**选择概率**:
- 所有候选词等概率随机选择
- 精确匹配优先于模糊匹配

---

## 📊 实际注入结果

### 被修改的 16 个查询

| ASIN | 原词 | 修改后 | 类型 | 在目标产品中 |
|------|------|--------|------|-------------|
| B00HHEX8SS | machines | maching | 删除 'es' | ❌ 不在 |
| B00MCH9BJA | machine | maching | 删除 'e' | ✅ 在 |
| B005CNACDK | machine | maching | 删除 'e' | ❌ 不在 |
| B00IH26OC6 | machine | maching | 删除 'e' | ❌ 不在 |
| B004KW7H82 | machine | maching | 删除 'e' | ✅ 在 |
| B00DV8XPAU | machine | maching | 删除 'e' | ❌ 不在 |
| B00DG8V5WA | gluing | glueing | 改变 'i' → 'e' | ❌ 不在 |
| B00CDRL174 | machine | maching | 删除 'e' | ✅ 在 |
| B00902XQ6O | machine | maching | 删除 'e' | ❌ 不在 |
| B00C9UW1HY | sympathy | sympathy | 可能Unicode | ❌ 不在 |
| ... | ... | ... | ... | ... |

### 修改模式总结

**主要错误类型**:
1. **"machine" → "maching"** (11次): 删除词尾元音
2. **"gluing" → "glueing"** (1次): 元音替换
3. **"sympathy" → "sympathy"** (3次): 可能是不可见字符变化
4. **其他截断** (1次): 似是而非的长文本修改

---

## 🔍 核心问题分析

### 问题1: 被修改的词不在目标产品中

**统计**:
- 70% (11/16) 被修改的词**不在**目标产品中
- 30% (5/16) 被修改的词**在**目标产品中

**影响**:
```
查询: "balloon dies that work with my machine"
目标产品: "My Favorite Things Die-Namics Die, Party Balloons"

✅ 'balloon' 在产品中 → 关键区分信息
✅ 'dies' 在产品中 → 关键区分信息
❌ 'machine' 不在产品中 → 不影响匹配

修改后: "balloon dies that work with my maching"
→ 'maching' 也不在产品中
→ 但查询仍然通过 'balloon' 和 'dies' 匹配成功
```

### 问题2: 修改的词不是关键区分信息

**关键区分信息**:
- 产品名称（"balloon dies", "Spellbinders"）
- 品牌（"My Favorite Things", "Spellbinders"）
- 核心功能（"embossing", "cutting"）

**非关键信息**:
- 通用描述词（"machine", "work with", "projects"）

**当前策略**: 修改了非关键信息 → 不影响检索结果

### 问题3: 词频验证

**在文档中的词频**（前10000个产品）:
- "machine": 951次出现
- "maching": 0次出现 ❌
- "gluing": 9次出现
- "glueing": 0次出现 ❌

**结论**:
- Noisy查询的词 **不存在于文档中**
- 但这些词本身也不是区分查询的关键信息
- 所以Clean和Noisy的检索结果几乎相同

---

## 📈 对检索评估的影响

### 预期影响 vs 实际影响

| 检索器类型 | 预期 | 实际 | 原因 |
|-----------|------|------|------|
| BM25 | Noisy应该下降 | 0%差异 | 修改的词不在产品中 |
| TF-IDF | Noisy应该下降 | +2.17% | 随机波动 |
| Dense (BGE/E5) | Noisy应该下降 | 0%差异 | 语义理解"maching"≈"machine" |
| 所有检索器 | 显著差异 | 几乎相同 | 修改了非关键信息 |

### 异常检索器

**STAR** (+4.35% P@1):
- 在多个k值都显示系统性提升
- 可能是模型对某些错误的偏好
- 需要进一步调查

**TF-IDF** (+2.17% P@1), **ANCE** (+2.18% P@1):
- 边缘案例的随机波动
- 只有1-2个查询排名变化

---

## 💡 改进建议

### 策略1: 修改关键区分信息 ⭐

**当前问题**: 修改 "machine" → "maching"（不重要）

**建议**: 修改关键产品信息
```
查询: "balloon dies for Spellbinders"
修改: "balloon dies for Sizzix"  # 错误品牌
或:  "circle dies for Spellbinders"  # 错误形状
```

### 策略2: 验证词在产品中的存在

```python
# 修改前检查
if 词 in 目标产品.文档:
    修改这个词  # 影响匹配
else:
    选择另一个词  # 当前词不影响匹配
```

### 策略3: 基于检索的重要性加权

```python
# 使用TF-IDF或BM25计算词的重要性
word_importance = calculate_tfidf(word, product_corpus)

# 选择高重要性的词修改
target_word = select_top_k_words(query, k=1, by=word_importance)
```

### 策略4: 多词修改（更强噪声）

```python
# 当前: 每个查询只修改1个词
# 建议: 修改2-3个关键信息词

查询: "Spellbinders balloon dies in A2 size"
修改: "Sizzix circle dies in letter size"
       ↑品牌     ↑形状     ↑规格
```

---

## 📝 总结

### 当前噪声注入策略

✅ **优点**:
- 基于用户真实错误历史（个性化）
- 注入逻辑清晰（拼写错误）
- 不强制注入（无匹配则保持原样）

❌ **缺点**:
- 修改的词大多不在目标产品中（70%）
- 修改的不是关键区分信息
- 导致Clean和Noisy结果几乎相同（无法测试鲁棒性）

### 根本原因

**设计目标** vs **实际效果**:
- 目标: 测试检索器对拼写错误的鲁棒性
- 实际: 修改的词不影响检索（因为它们不是关键区分信息）

### 解决方案

需要修改**关键区分信息**（产品名称、品牌、核心功能），才能真正测试检索器的噪声鲁棒性。

---

## 🔗 相关文件

- Stage 10: `.claude/skills/PersoanlQuery/10_targeted_noisy_query/10_generate_noisy_queries.py`
- Stage 5: `result/personal_query/05_writing_analysis/results/writing_analysis_A13OFOB1394G31.json`
- 输出: `result/personal_query/10_targeted_noisy_query/noisy_queries_A13OFOB1394G31.json`
