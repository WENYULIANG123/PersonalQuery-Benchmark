# 三路偏好分类器集成总结

## 集成时间
2026-03-15

## 目标
将新开发的三路偏好分类器（v2）集成到所有现有的LLM重排序评估脚本中，替换旧的二元分类逻辑。

## 核心变更

### 1. 导入新分类器
**所有脚本**添加了以下导入：
```python
from preference_classifier import build_three_way_persona_context_v2
```

### 2. 替换 `_build_persona_context` 方法
**旧逻辑（v1）**：
- 按维度（dimension）匹配历史属性
- 显示该维度下的所有属性值
- 问题：查询"Sizzix"时显示所有品牌偏好（包括"Spellbinders"等不相关内容）

**新逻辑（v2）**：
```python
def _build_persona_context(self, category: str, selected_attributes: List[Dict]) -> str:
    """
    使用三路偏好分类器构建用户偏好上下文
    """
    if not selected_attributes:
        return ""
    
    return build_three_way_persona_context_v2(
        category=category,
        selected_attributes=selected_attributes,
        user_id=self.user_id,
        processing_dir=self.processing_dir
    )
```

**优势**：
- 只匹配查询值（value）本身，不显示不相关属性
- 三路分类：显性偏好（用户喜欢的）、隐性偏好（用户不喜欢的）、冲突偏好（矛盾评价）
- 更精准的LLM上下文，减少噪音

### 3. `_score_by_generation` 方法调整
部分脚本（如 `13_evaluate_minimax_m2.py`）的 `_score_by_generation` 方法从使用 `persona_utils` 的旧函数改为直接调用 `_build_persona_context`，统一使用新的三路分类器。

## 已更新的脚本列表

总计 **7个脚本** 已完成集成：

### GLM 系列 (3个)
1. `13_evaluate_glm_5_both.py` - GLM-5模型
2. `13_evaluate_glm_4_5v_both.py` - GLM-4.5V模型
3. `13_evaluate_glm_4_7_both.py` - GLM-4.7模型

### Minimax 系列 (3个)
4. `13_evaluate_minimax_m2.py` - MiniMax M2模型
5. `13_evaluate_minimax_m2_1.py` - MiniMax M2.1模型
6. `13_evaluate_minimax_m2_5_highspeed.py` - MiniMax M2.5高速模型

### Qwen 系列 (1个)
7. `13_evaluate_qwen_7b.py` - Qwen2-7B模型

## 验证结果

✅ **语法检查**：所有脚本通过 `python3 -m py_compile` 验证
✅ **导入测试**：`build_three_way_persona_context_v2` 成功导入
✅ **类型注解**：LSP诊断中的错误均为既有问题（与本次修改无关）

## 向后兼容性

所有修改仅影响 `PersonalizedXXXReRanker` 类的内部实现，外部API保持不变：
- 输入：相同的 `queries` 列表（包含 `selected_attributes`）
- 输出：相同的评分结果格式
- 调用方式：无需修改主函数逻辑

## 下一步建议

1. **A/B测试**：对比新旧分类器对检索指标（Recall@K）的影响
2. **日志分析**：检查生成的persona context是否符合预期
3. **性能监控**：新分类器使用模糊匹配，可能略微增加延迟（预计可忽略）

## 回滚方案

如需回滚到旧逻辑，修改导入为：
```python
from persona_utils import build_enhanced_persona_context as build_three_way_persona_context_v2
```

并还原对应脚本的 `_build_persona_context` 和 `_score_by_generation` 方法（建议先备份当前版本）。

## 相关文档

- `README_PREFERENCE_CLASSIFIER.md` - 新分类器使用指南
- `CORRECTION_SUMMARY.md` - v1 → v2 修正详情
- `example_usage.py` - 集成示例代码
- `test_preference_classifier.py` - 单元测试

---

**修改者**: Sisyphus (Claude Code)  
**修改日期**: 2026-03-15  
**Git提交**: 待创建（建议提交信息："integrate: use three-way preference classifier in LLM rerankers"）
