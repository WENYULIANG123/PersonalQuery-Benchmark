# Three-way Preference Classifier

## 概述

`preference_classifier.py` 实现了三分类偏好系统，用于 Stage 14 LLM Reranking，将用户历史偏好分为：

1. **显性偏好 (Explicit)**: 用户明确表达的正面属性偏好
2. **隐性偏好 (Implicit)**: 从负面反馈中推断的改进期望
3. **冲突偏好 (Conflicting)**: 同一维度上的矛盾信号

## 为什么需要三分类？

### 问题：传统二分类的局限性

传统的 positive/negative 二分类无法区分：

```
❌ 传统方法（混乱）:
- Brand: Spellbinders (positive)
- Brand: Sizzix (negative)
- Performance: clean cutting (positive)
- Performance: requires hand-cutting (negative)

→ LLM 难以理解："negative"是用户不喜欢的，还是用户期望改进的？
```

### 解决方案：三分类

```
✅ 三分类方法（清晰）:

显性偏好 (Explicit - 用户喜欢什么):
- Brand: Spellbinders ✓
- Performance: clean cutting ✓

隐性偏好 (Implicit - 用户希望改进什么):
- Brand: Sizzix ⚠ (避免)
- Performance: requires hand-cutting ⚠ → Expects: fully die-cut capability

冲突偏好 (Conflicting - 矛盾信号):
- (此例中无冲突)

→ LLM 清楚知道：用户要什么、不要什么、以及改进方向
```

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

### 1. 显性偏好 (Explicit)

**条件**: `sentiment == 'positive'`

**示例**:
```json
{
  "attribute": "Spellbinders",
  "dimension": "Brand_Preference",
  "sentiment": "positive",
  "original_text": "I love my Spellbinders dies"
}
```

### 2. 隐性偏好 (Implicit)

**条件**: `sentiment == 'negative'`

**示例**:
```json
{
  "attribute": "Sizzix",
  "dimension": "Brand_Preference",
  "sentiment": "negative",
  "original_text": "Sizzix dies are too expensive",
  "improvement_wish": "More affordable pricing"
}
```

**改进期望 (improvement_wish)**:
- 如果有：表示用户期望的改进方向
- 如果无：表示用户简单地避免这个属性

### 3. 冲突偏好 (Conflicting)

**条件**: 同一维度内同时存在 positive 和 negative sentiment

**示例**:
```json
// Brand_Preference 维度
{
  "attribute": "Sizzix",
  "sentiment": "positive",  // 用户A喜欢
  "original_text": "Sizzix Big Shot is great"
}
{
  "attribute": "Sizzix",
  "sentiment": "negative",  // 用户B不喜欢
  "original_text": "Sizzix is overpriced"
}

→ 冲突检测: 同一属性"Sizzix"有不同情感
→ 分类为 conflicting
→ LLM 会收到提示: "查询要求优先于冲突偏好"
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
