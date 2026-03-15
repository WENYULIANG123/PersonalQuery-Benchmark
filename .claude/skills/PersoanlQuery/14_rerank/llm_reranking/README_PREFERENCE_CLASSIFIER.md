# Three-way Preference Classifier

## 概述

`preference_classifier.py` 实现了三分类偏好系统，用于 Stage 14 LLM Reranking，将用户历史偏好分为：

1. **显性偏好 (Explicit)**: 用户明确表达的正面属性偏好
2. **隐性偏好 (Implicit)**: 从负面反馈中推断的改进期望
3. **冲突偏好 (Conflicting)**: 同一维度上的矛盾信号

## 为什么需要三分类？

### 核心思想：匹配查询属性值与历史评价

**关键**：不是比较整个维度内的所有属性，而是**查询要求的具体属性值**在历史中的评价。

### 示例说明

**场景 1：查询要 Spellbinders**

```
查询: {"dimension": "Brand_Preference", "value": "Spellbinders"}

历史偏好:
- Spellbinders: positive ✓  ← 匹配！
- Sizzix: negative         ← 不匹配（查询没要Sizzix）
- Ranger: positive         ← 不匹配

分类结果: Explicit（查询要的"Spellbinders"在历史中是positive）
```

**场景 2：查询要 Sizzix（冲突！）**

```
查询: {"dimension": "Brand_Preference", "value": "Sizzix"}

历史偏好:
- Spellbinders: positive   ← 不匹配
- Sizzix: negative ✗       ← 匹配！但是negative！
- Ranger: positive         ← 不匹配

分类结果: Implicit（查询要的"Sizzix"在历史中是negative → 冲突！）
→ 用户不喜欢Sizzix，但查询却要Sizzix，这是一个需要注意的矛盾
```

**场景 3：查询要的属性在历史中有正有负**

```
查询: {"dimension": "Performance", "value": "clean cutting"}

历史偏好:
- "got a clean cut": positive ✓       ← 匹配
- "requires hand-cutting": negative ✗  ← 也匹配（都是cutting相关）

分类结果: Conflicting（查询要的属性在历史中既有正面也有负面评价）
```

### 三分类定义

| 分类 | 定义 | 示例 |
|------|------|------|
| **Explicit** | 查询属性值在历史中是 positive | 查询要 Spellbinders，历史喜欢 Spellbinders ✅ |
| **Implicit** | 查询属性值在历史中是 negative | 查询要 Sizzix，历史不喜欢 Sizzix ⚠️（矛盾！）|
| **Conflicting** | 查询属性值在历史中有正有负 | 查询要 clean cutting，历史中有好有坏 ⚔️ |

## 数据流

```
Stage 1: Preference Extraction
    ↓ (entity, sentiment, original_text, improvement_wish)
Stage 3: Persona Aggregation
    ↓ (attributes grouped by dimension)
Stage 7/10: Query Generation
    ↓ (selected_attributes: [{dimension, value}])
Stage 14: Three-way Classification ← **新增**
    ↓
    {
      'Brand_Preference': {
        'explicit': [Spellbinders, Ranger],
        'implicit': [Sizzix (avoid), Movers & Shapers (avoid)],
        'conflicting': []
      }
    }
    ↓
LLM Reranking with structured context
```

## 使用方法

### 方法1: 快速使用（推荐）

```python
from preference_classifier import build_three_way_persona_context

# 查询信息
selected_attributes = [
    {"dimension": "Brand_Preference", "value": "Spellbinders"},
    {"dimension": "Performance", "value": "clean cutting"}
]

# 生成格式化上下文（直接用于LLM prompt）
context = build_three_way_persona_context(
    category="Die-Cuts",
    selected_attributes=selected_attributes,
    user_id="A13OFOB1394G31",
    processing_dir="/path/to/result/personal_query/03_processing"
)

print(context)
```

**输出示例**:
```
User Preference Profile (Three-way Classification):
============================================================
### Brand_Preference

**Explicit Preferences (Positive):**
  ✓ Spellbinders
    Evidence: "Spellbinders Grand Calibur machine..."
  ✓ Ranger
    Evidence: "Ranger Stickles Glitter Glue..."

**Implicit Preferences (Inferred from Negative Feedback):**
  ⚠ Dislikes: Sizzix
  ⚠ Dislikes: Movers & Shapers

### Performance

**Explicit Preferences (Positive):**
  ✓ clean cut performance
    Evidence: "got a clean cut..."

**Implicit Preferences (Inferred from Negative Feedback):**
  ⚠ Dislikes: requires hand-cutting
    → Expects: fully die-cut capability
```

### 方法2: 编程访问原始数据

```python
from preference_classifier import PreferenceClassifier

classifier = PreferenceClassifier(
    user_id="A13OFOB1394G31",
    processing_dir="/path/to/result/personal_query/03_processing"
)

# 分类偏好（返回字典）
result = classifier.classify_query_preferences(
    category="Die-Cuts",
    selected_attributes=selected_attributes
)

# 访问原始数据
for dimension, categories in result.items():
    explicit_count = len(categories['explicit'])
    implicit_count = len(categories['implicit'])
    conflict_count = len(categories['conflicting'])
    
    print(f"{dimension}: {explicit_count} explicit, {implicit_count} implicit, {conflict_count} conflicts")
```

### 方法3: 集成到现有 LLM Reranker

修改现有的 reranking 脚本（例如 `13_evaluate_glm_5_both.py`）：

```python
# 旧方法
from persona_utils import build_enhanced_persona_context

persona_context = build_enhanced_persona_context(
    category, selected_attrs, user_id, query_info, processing_dir
)

# 新方法（三分类）
from preference_classifier import build_three_way_persona_context

persona_context = build_three_way_persona_context(
    category, selected_attrs, user_id, processing_dir
)

# 其他代码保持不变
prompt = f"""
[User Profile]
{persona_context}

[Query]
{query}

[Product]
{product_text}

Score the relevance...
"""
```

## 分类逻辑详解

### 核心算法

```python
for each query_attribute in selected_attributes:
    query_value = query_attribute['value']  # e.g., "Sizzix"
    dimension = query_attribute['dimension']  # e.g., "Brand_Preference"
    
    # 1. 找出历史中所有匹配该dimension的属性
    historical_attrs = find_all_attrs(dimension)
    
    # 2. 用模糊匹配找出与query_value匹配的历史属性
    matched_positive = [attr for attr in historical_attrs 
                       if fuzzy_match(attr.value, query_value) 
                       and attr.sentiment == 'positive']
    
    matched_negative = [attr for attr in historical_attrs 
                       if fuzzy_match(attr.value, query_value) 
                       and attr.sentiment == 'negative']
    
    # 3. 分类
    if matched_positive and matched_negative:
        → Conflicting（查询值在历史中有正有负）
    elif matched_positive:
        → Explicit（查询值在历史中是正面）
    elif matched_negative:
        → Implicit（查询值在历史中是负面 → 冲突！）
```

### 1. 显性偏好 (Explicit)

**条件**: 查询属性值在历史中**仅有** `positive` sentiment

**示例**:
```python
查询: {"dimension": "Brand_Preference", "value": "Spellbinders"}

历史匹配:
- "Spellbinders Grand Calibur": positive ✓
- "Spellbinders dies": positive ✓

分类: Explicit
解释: 用户历史喜欢Spellbinders，查询也要Spellbinders → 完美匹配！
```

### 2. 隐性偏好 (Implicit)

**条件**: 查询属性值在历史中**仅有** `negative` sentiment

**示例**:
```python
查询: {"dimension": "Brand_Preference", "value": "Sizzix"}

历史匹配:
- "Sizzix Big Shot": negative ✗
- "Sizzix dies": negative ✗

分类: Implicit（冲突！）
解释: 用户历史不喜欢Sizzix，但查询却要Sizzix → 需要警告LLM这是一个矛盾
```

**改进期望 (improvement_wish)**:
- 如果有：表示用户期望的改进方向（如"更便宜"）
- 如果无：表示用户单纯避免这个属性

### 3. 冲突偏好 (Conflicting)

**条件**: 查询属性值在历史中**同时有** positive 和 negative sentiment

**示例**:
```python
查询: {"dimension": "Performance", "value": "clean cutting"}

历史匹配:
- "got a clean cut": positive ✓
- "requires hand-cutting": negative ✗

分类: Conflicting
解释: 查询要的属性在历史中评价不一致 → LLM需要权衡
```

### 模糊匹配规则

```python
def fuzzy_match(query_value, historical_value):
    # 1. 精确匹配
    if query_value.lower() == historical_value.lower():
        return True
    
    # 2. 包含匹配
    if query_value.lower() in historical_value.lower():
        return True
    if historical_value.lower() in query_value.lower():
        return True
    
    # 3. Token 重叠匹配（>= 50%）
    query_tokens = set(query_value.split())
    hist_tokens = set(historical_value.split())
    overlap = len(query_tokens & hist_tokens)
    if overlap >= max(1, min(len(query_tokens), len(hist_tokens)) * 0.5):
        return True
    
    return False
```

## 输出格式

### JSON 输出

```json
{
  "Brand_Preference": {
    "explicit": [
      {
        "attribute": "Spellbinders",
        "dimension": "Brand_Preference",
        "sentiment": "positive",
        "original_text": "...",
        "improvement_wish": ""
      }
    ],
    "implicit": [
      {
        "attribute": "Sizzix",
        "dimension": "Brand_Preference",
        "sentiment": "negative",
        "original_text": "...",
        "improvement_wish": "More affordable"
      }
    ],
    "conflicting": []
  }
}
```

### 格式化文本输出（用于 LLM）

见上方"方法1"示例。

## 测试

运行测试脚本：

```bash
cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/14_rerank/llm_reranking

# 运行完整测试套件
python3 test_preference_classifier.py

# 或者测试单个用户
python3 preference_classifier.py A13OFOB1394G31 Die-Cuts
```

## 与旧方法的对比

| 特性 | 旧方法 (persona_utils.py) | 新方法 (preference_classifier.py) |
|------|---------------------------|-----------------------------------|
| 偏好分类 | 二分类 (positive/negative) | 三分类 (explicit/implicit/conflicting) |
| 负面反馈 | 混在一起，难以区分 | 明确标记为"implicit"，带改进期望 |
| 冲突检测 | 手动标记 CONFLICTING | 自动检测同维度矛盾信号 |
| LLM 可解释性 | 中等 | 高（清晰的三分类结构）|
| 改进期望 | 未显示 | 显示 `improvement_wish` |

## API 参考

### PreferenceClassifier

```python
classifier = PreferenceClassifier(user_id: str, processing_dir: str)

# 分类单个维度
result = classifier.classify_preferences_for_dimension(
    dimension: str,
    all_attributes: List[Dict]
) -> Dict[str, List[Dict]]

# 分类查询的所有维度
result = classifier.classify_query_preferences(
    category: str,
    selected_attributes: List[Dict]
) -> Dict[str, Dict[str, List[Dict]]]

# 格式化为文本
text = classifier.format_classified_preferences(
    classified: Dict[str, Dict[str, List[Dict]]]
) -> str
```

### 便捷函数

```python
# 一步生成格式化上下文
context = build_three_way_persona_context(
    category: str,
    selected_attributes: List[Dict],
    user_id: str,
    processing_dir: str
) -> str

# 获取原始分类数据
data = classify_preferences(
    category: str,
    selected_attributes: List[Dict],
    user_id: str,
    processing_dir: str
) -> Dict[str, Dict[str, List[Dict]]]
```

## 文件位置

- **核心模块**: `.claude/skills/PersoanlQuery/14_rerank/llm_reranking/preference_classifier.py`
- **测试脚本**: `.claude/skills/PersoanlQuery/14_rerank/llm_reranking/test_preference_classifier.py`
- **输入数据**: `result/personal_query/03_processing/persona_*.json`
- **输出示例**: `result/personal_query/14_rerank/classification_example.json`

## 示例输出文件

运行测试后会生成示例文件：

```bash
result/personal_query/14_rerank/classification_example.json
```

包含完整的分类结果和格式化上下文。

## 常见问题

**Q: 为什么要区分 explicit 和 implicit？**  
A: LLM 需要明确知道用户"想要什么"（explicit）和"不想要什么"（implicit）。混在一起会降低 reranking 精度。

**Q: improvement_wish 为空怎么办？**  
A: 仍然分类为 implicit，表示用户单纯避免这个属性，没有明确的改进期望。

**Q: 冲突偏好如何处理？**  
A: 在 LLM prompt 中明确标注为"CONFLICTING"，并提示"查询要求优先于冲突偏好"。

**Q: 可以用于其他 Stage 吗？**  
A: 可以！任何需要结构化偏好分类的地方都可以使用（如 Stage 11 评估、Stage 7 查询生成等）。

## 下一步

1. **集成到所有 LLM reranker**: 更新 GLM、Minimax、Qwen 脚本
2. **A/B 测试**: 对比三分类 vs 二分类的 reranking 效果
3. **扩展分类逻辑**: 添加更多冲突检测规则（如数值冲突、语义冲突等）
4. **可视化**: 生成偏好分类的可视化报告

## 贡献者

- 初始实现: 2026-03-15
- 基于 Stage 1-3 数据结构设计
