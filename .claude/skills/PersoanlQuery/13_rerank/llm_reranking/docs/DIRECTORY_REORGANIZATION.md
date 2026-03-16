# 目录重组日志

## 重组日期
2026-03-15

## 重组原因
原有目录结构扁平，所有文件（评估脚本、核心模块、测试、文档）混在一起，难以维护和导航。

## 原始结构
```
llm_reranking/
├── 13_evaluate_glm_*.py (3个文件)
├── 13_evaluate_minimax_*.py (3个文件)
├── 13_evaluate_qwen_*.py (1个文件)
├── preference_classifier.py
├── persona_utils.py
├── test_*.py (4个文件)
├── example_usage.py
├── *.md (3个文档)
└── __init__.py
```

**问题**：
- 评估脚本按模型分散，难以快速定位
- 核心模块与测试/示例混在一起
- 文档缺乏组织
- 不清楚哪些是核心功能，哪些是辅助工具

## 新结构
```
llm_reranking/
├── core/                          # 核心功能模块
│   ├── __init__.py
│   ├── preference_classifier.py   # 三路偏好分类器
│   └── persona_utils.py           # 旧版分类器 (legacy)
│
├── evaluators/                    # 评估脚本（按模型分组）
│   ├── glm/                       # GLM系列 (3个脚本)
│   ├── minimax/                   # Minimax系列 (3个脚本)
│   └── qwen/                      # Qwen系列 (1个脚本)
│
├── tests/                         # 单元测试 (4个文件)
├── examples/                      # 使用示例 (1个文件)
├── docs/                          # 文档 (4个文件)
├── __init__.py
└── README.md
```

**优势**：
- **清晰的模块分离**：核心功能、评估器、测试、示例、文档各自独立
- **按模型分组**：GLM/Minimax/Qwen脚本分别存放，易于维护
- **易于导航**：目录名清晰表达用途
- **扩展性好**：新增模型只需在 evaluators/ 下创建子目录

## 变更清单

### 1. 目录创建
```bash
llm_reranking/
├── core/
├── evaluators/glm/
├── evaluators/minimax/
├── evaluators/qwen/
├── tests/
├── examples/
└── docs/
```

### 2. 文件移动

| 原路径 | 新路径 | 数量 |
|--------|--------|------|
| `13_evaluate_glm_*.py` | `evaluators/glm/` | 3 |
| `13_evaluate_minimax_*.py` | `evaluators/minimax/` | 3 |
| `13_evaluate_qwen_*.py` | `evaluators/qwen/` | 1 |
| `preference_classifier.py` | `core/` | 1 |
| `persona_utils.py` | `core/` | 1 |
| `test_*.py` | `tests/` | 4 |
| `example_*.py` | `examples/` | 1 |
| `*.md` | `docs/` | 3 |

**总计移动**：17个文件

### 3. 新建文件

| 文件路径 | 用途 |
|---------|------|
| `core/__init__.py` | 导出核心模块 |
| `README.md` | 顶层文档 |
| `docs/DIRECTORY_REORGANIZATION.md` | 本文档 |

**总计新建**：3个文件

### 4. 导入路径更新

#### 主 `__init__.py`
```python
# 旧
from .preference_classifier import ...
from .persona_utils import ...

# 新
from .core.preference_classifier import ...
from .core.persona_utils import ...
```

#### 评估脚本 (evaluators/*/*.py)
```python
# 旧
from preference_classifier import build_three_way_persona_context_v2

# 新
from ...core.preference_classifier import build_three_way_persona_context_v2
```

#### 测试和示例 (tests/*.py, examples/*.py)
```python
# 旧
from preference_classifier import ...

# 新
from ..core.preference_classifier import ...
```

**总计更新**：10个文件

## 验证结果

### ✅ 语法检查
```bash
$ python3 -m py_compile evaluators/glm/13_evaluate_glm_5_both.py
$ python3 -m py_compile core/preference_classifier.py
# 无错误
```

### ✅ 导入测试
```bash
$ python3 -c "from core.preference_classifier import build_three_way_persona_context_v2; print('✅')"
✅ Core import successful
```

### ✅ 目录统计
- **顶层项目**：8个（6个目录 + 2个文件）
- **总文件数**：20个
- **总目录数**：8个

## 向后兼容性

### 已保持兼容
从顶层模块导入仍然有效：
```python
from llm_reranking import (
    PreferenceClassifier,
    build_three_way_persona_context
)
```

### 不兼容的情况
直接导入子模块的旧代码需要更新：
```python
# ❌ 不再有效
from llm_reranking.preference_classifier import ...

# ✅ 新路径
from llm_reranking.core.preference_classifier import ...
```

**建议**：始终从顶层 `llm_reranking` 导入公共API，避免直接引用子模块路径。

## 回滚方案

如需回滚到原始结构：

```bash
cd .claude/skills/PersoanlQuery/13_rerank/llm_reranking

# 移回所有文件到顶层
mv core/*.py .
mv evaluators/*/*.py .
mv tests/*.py .
mv examples/*.py .
mv docs/*.md .

# 删除子目录
rm -rf core evaluators tests examples docs

# 恢复原始 __init__.py（从git历史）
git checkout HEAD~1 -- __init__.py
```

## 下一步建议

1. **创建 evaluators/__init__.py**：统一导出所有评估器类
2. **添加 tests/__init__.py**：使测试可以作为包导入
3. **文档改进**：在 docs/ 下添加架构图和API参考
4. **CI/CD**：更新测试脚本路径（如有）

## 相关文档

- [`README.md`](../README.md) - 模块总览和快速开始
- [`INTEGRATION_SUMMARY.md`](INTEGRATION_SUMMARY.md) - 三路分类器集成总结
- [`README_PREFERENCE_CLASSIFIER.md`](README_PREFERENCE_CLASSIFIER.md) - 分类器详细文档

---

**执行人**: Sisyphus (Claude Code)  
**审核人**: (待审核)  
**批准日期**: 2026-03-15
