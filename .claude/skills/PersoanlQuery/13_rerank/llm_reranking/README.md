# LLM Reranking Module

LLM-based reranking methods for personalized product search.

## Directory Structure

```
llm_reranking/
├── core/                          # 核心功能模块
│   ├── preference_classifier.py   # 三路偏好分类器 (v2)
│   ├── persona_utils.py           # 旧版二元分类器 (legacy)
│   └── __init__.py
│
├── evaluators/                    # 评估脚本 (按模型分组)
│   ├── glm/                       # GLM 系列模型
│   │   ├── 13_evaluate_glm_4_5v_both.py
│   │   ├── 13_evaluate_glm_4_7_both.py
│   │   └── 13_evaluate_glm_5_both.py
│   ├── minimax/                   # Minimax 系列模型
│   │   ├── 13_evaluate_minimax_m2.py
│   │   ├── 13_evaluate_minimax_m2_1.py
│   │   └── 13_evaluate_minimax_m2_5_highspeed.py
│   └── qwen/                      # Qwen 系列模型
│       └── 13_evaluate_qwen_7b.py
│
├── tests/                         # 单元测试
│   ├── test_preference_classifier.py        # 三路分类器测试
│   ├── test_minimax_m2_thinking.py          # Minimax思考链测试
│   ├── test_minimax_glm_thinking.py         # GLM思考链测试
│   └── test_minimax_glm_thinking_improved.py
│
├── examples/                      # 使用示例
│   └── example_usage.py           # 三路分类器集成示例
│
├── docs/                          # 文档
│   ├── README_PREFERENCE_CLASSIFIER.md  # 分类器使用指南
│   ├── CORRECTION_SUMMARY.md            # v1→v2修正详情
│   └── INTEGRATION_SUMMARY.md           # 集成总结
│
├── __init__.py                    # 主导出模块
└── README.md                      # 本文件
```

## Quick Start

### 1. 导入核心功能

```python
# 方式1: 从顶层模块导入
from llm_reranking import (
    PreferenceClassifier,
    build_three_way_persona_context
)

# 方式2: 从 core 子模块导入
from llm_reranking.core import (
    PreferenceClassifier,
    build_three_way_persona_context
)
```

### 2. 构建三路偏好上下文

```python
context = build_three_way_persona_context(
    category='Die-Cuts',
    selected_attributes=[
        {'dimension': 'Brand_Preference', 'value': 'Spellbinders'},
        {'dimension': 'Performance', 'value': 'clean cutting'}
    ],
    user_id='A13OFOB1394G31',
    processing_dir='/path/to/result/personal_query/03_processing'
)

print(context)
```

**输出示例**：
```
User Preference Profile (Query-Value Matched Classification):
======================================================================
NOTE: Classification based on historical sentiment for QUERY VALUES
======================================================================

[Brand_Preference] Query asks for: Spellbinders

✅ Explicit Preferences (User likes these):
  - Spellbinders (positive): "I love Spellbinders dies, they cut so cleanly!"

⚠️ Implicit Preferences (User wants to avoid these):
  (None - query value aligns with user preferences)

⚔️ Conflicting Preferences (Mixed reviews):
  (None)
```

### 3. 运行评估脚本

```bash
# GLM 评估
cd evaluators/glm
python 13_evaluate_glm_5_both.py

# Minimax 评估
cd evaluators/minimax
python 13_evaluate_minimax_m2.py

# Qwen 评估
cd evaluators/qwen
python 13_evaluate_qwen_7b.py
```

## Core Modules

### PreferenceClassifier (v2)

三路偏好分类器，基于查询值匹配历史偏好。

**特点**：
- 只匹配查询值本身（不是整个维度）
- 三路分类：显性/隐性/冲突偏好
- 模糊匹配（支持变体和拼写差异）

**详细文档**: [`docs/README_PREFERENCE_CLASSIFIER.md`](docs/README_PREFERENCE_CLASSIFIER.md)

### persona_utils (legacy)

旧版二元分类器，保留用于兼容性。

**不推荐用于新项目**，请使用 `PreferenceClassifier` (v2)。

## Testing

```bash
# 运行所有测试
cd tests
python test_preference_classifier.py

# 运行特定测试
python test_minimax_m2_thinking.py
```

## Documentation

- **[README_PREFERENCE_CLASSIFIER.md](docs/README_PREFERENCE_CLASSIFIER.md)** - 三路分类器完整文档
- **[CORRECTION_SUMMARY.md](docs/CORRECTION_SUMMARY.md)** - v1→v2修正说明
- **[INTEGRATION_SUMMARY.md](docs/INTEGRATION_SUMMARY.md)** - 集成总结

## Examples

查看 [`examples/example_usage.py`](examples/example_usage.py) 获取5个真实场景的使用示例：

1. 基础使用 - 单查询分类
2. 完整LLM提示构建
3. 批量查询处理
4. 空/冲突偏好处理
5. 与现有重排序系统集成

## Migration Guide

### 从旧版 persona_utils 迁移

**旧代码**:
```python
from persona_utils import build_persona_context

context = build_persona_context(category, selected_attributes, user_id)
```

**新代码**:
```python
from llm_reranking.core import build_three_way_persona_context

context = build_three_way_persona_context(
    category=category,
    selected_attributes=selected_attributes,
    user_id=user_id,
    processing_dir='/path/to/03_processing'
)
```

**主要变化**:
- 函数名从 `build_persona_context` → `build_three_way_persona_context`
- 新增 `processing_dir` 参数（必需）
- 输出格式更结构化（三路分类）
- 只显示与查询值匹配的偏好（更精准）

## Changelog

### 2026-03-15 - Directory Reorganization
- 重组目录结构，按功能分组
- 创建 `core/`, `evaluators/`, `tests/`, `docs/`, `examples/` 子目录
- 更新所有导入路径
- 添加本 README 文档

### 2026-03-15 - Three-way Classifier Integration
- 所有7个评估脚本集成新的三路分类器
- 替换旧的 `_build_persona_context` 方法
- 详见 [`docs/INTEGRATION_SUMMARY.md`](docs/INTEGRATION_SUMMARY.md)

### 2026-03-15 - Preference Classifier v2
- 修正核心逻辑：从匹配整个维度改为只匹配查询值
- 添加三路分类：显性/隐性/冲突
- 详见 [`docs/CORRECTION_SUMMARY.md`](docs/CORRECTION_SUMMARY.md)

## License

Internal project - 仅供课题组内部使用

## Contact

维护者: Sisyphus (Claude Code)  
更新日期: 2026-03-15
