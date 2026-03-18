# Phase 1-4 改造路线图

## 已完成：Phase 1 ✅

**目标**：添加方面级别的情感分析 + 置信度评分 + 隐式方面检测
**状态**：完成并通过4/4测试
**交付物**：
- ✅ `01_extract_preferences_v2_with_aspects.py` 
- ✅ `test_v2_extraction.py` 
- ✅ `PHASE_1_V2_IMPLEMENTATION.md`

**核心成就**：
- 置信度评分系统（显式 0.8-0.95, 隐式 0.6）
- 隐式方面检测（4种类型）
- 质量检查框架
- 完全向后兼容

---

## 计划中：Phase 2（预计 9-12 天）

### Phase 2a：方面聚类（3 天）

**目标**：使用 DBSCAN + Sentence-Transformers 进行智能聚类

**关键脚本**：`01_consolidate_aspects_v2.py` (~400 行)

**功能**：
```python
# Step 1: Transformer 编码
embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(aspect_texts)

# Step 2: DBSCAN 聚类
clustering = DBSCAN(eps=0.3, min_samples=1).fit_predict(distance_matrix)

# Step 3: 生成规范名称
canonical_name = generate_canonical_name(cluster_members)  # 使用 LLM

# Step 4: 追踪合并历史
merge_history = [{
  "original_aspects": [...],
  "merged_to": "canonical_name",
  "merge_confidence": 0.85
}]
```

**预期收益**：
- 去重率：75-80%
- 聚类精度：>90%
- 规范化比例：100%

**验收标准**：
- [ ] DBSCAN 聚类成功合并重复方面
- [ ] LLM 生成的规范名称可读性 >90%
- [ ] 合并历史完整记录
- [ ] 端到端测试通过

---

### Phase 2b：Stage 10 扩展（2 天）

**目标**：添加方面级别的评估

**修改文件**：`10_evaluate_LLM_score.py`

**新增内容**：

```python
ASPECT_ERROR_CLASSIFICATION = {
    "minor_misrepresentation": {...},  # 方面对，情感错
    "minor_omission": {...},            # 漏 1-2 个次要方面
    "major_omission": {...},            # 漏 >30% 方面
    "exaggeration_understatement": {...} # 严重程度错估
}

def evaluate_aspect_quality(predicted, ground_truth):
    return {
        "aspect_precision": 0.95,
        "aspect_recall": 0.88,
        "aspect_f1": 0.91,
        "error_breakdown": {...},
        "error_severity_score": 0.15
    }
```

**验收标准**：
- [ ] 4 种错误类型自动分类
- [ ] 与人工标注的对齐度 >85%
- [ ] 现有维度评估不受影响（回归测试通过）

---

### Phase 2c：Stage 11 更新（1 天）

**目标**：扩展人工评估界面以支持方面级别的错误分类

**修改文件**：`11_generate_human_eval_tasks.py` + HTML 评估界面

**新增内容**：

```html
<!-- 新标签页：方面错误分类 -->
<div id="tab-aspect-errors">
  <label>
    <input type="checkbox" name="error_type" value="minor_misrepresentation">
    ❌ Minor Misrepresentation
  </label>
  <!-- ... 其他 3 种错误 ... -->
</div>
```

**验收标准**：
- [ ] HTML 界面可正常加载和保存
- [ ] 人工标注数据正确导出
- [ ] 与 LLM 评估的对齐指标计算正确

---

### Phase 2d：集成测试（2-3 天）

**目标**：端到端验证 Stage 0 → 1 → 1.5 → 2+

**测试内容**：

```python
def test_end_to_end():
    # 1. 运行 Stage 1 v2
    result_v2 = run_stage_1_v2(test_user_id)
    assert result_v2['target_user_aspects']  # 有方面输出
    
    # 2. 运行 Stage 1.5
    consolidated = consolidate_aspects(result_v2['target_user_aspects'])
    assert len(consolidated) <= len(result_v2['target_user_aspects'])  # 去重
    
    # 3. 验证与 Stage 2+ 的兼容性
    assert stage_2_can_process(result_v2)  # 能使用维度输出
    assert stage_2_can_process_aspects(result_v2)  # 能使用方面输出
    
    # 4. 运行评估（Stage 10）
    evaluation = stage_10_evaluate(result_v2)
    assert 'aspect_precision' in evaluation  # 新指标存在
    
    # 5. 对比原版和 v2
    result_v1 = run_stage_1_v1(test_user_id)
    compare_results(result_v1, result_v2)
```

**验收标准**：
- [ ] 3 个真实用户的数据处理通过
- [ ] 维度提取精度 ≥90%（vs v1）
- [ ] 方面提取精度 ≥85%
- [ ] 隐式方面检测率 ≥70%
- [ ] LLM 调用成本增长 ≤30%

---

## 后续阶段：Phase 3-4（视情况）

### Phase 3：高级评估（可选，2 周）

**目标**：实现基于错误比较的 LLM 优化（学术前沿）

**参考**：[2025] Error Comparison Optimization for LLMs on ABSA

```python
def learn_from_error_pairs():
    """
    学习：
    - 错误 A（严重）vs 错误 B（轻微）的特征
    - 如何优化 prompt 减少严重错误
    """
```

---

### Phase 4：生产环境优化（可选，1-2 周）

**目标**：性能优化、缓存、并行化

```python
# 缓存管理
@lru_cache(maxsize=1000)
def get_aspect_embeddings(aspect_text):
    return SentenceTransformer(...).encode(aspect_text)

# 批处理
def batch_consolidate_aspects(all_products):
    all_embeddings = encode_batch([...])  # 一次性编码
    clustering = DBSCAN(...).fit_predict(all_embeddings)
```

---

## 📈 整体时间计划

| Phase | 工作量 | 工作日 | 优先级 | 状态 |
|-------|--------|--------|--------|------|
| 1 | 960 行代码 | 3-4 | 🔴 高 | ✅ 完成 |
| 2a | 400 行代码 | 3 | 🔴 高 | ⏳ 计划中 |
| 2b | 200 行代码 | 2 | 🔴 高 | ⏳ 计划中 |
| 2c | 150 行代码 | 1 | 🟡 中 | ⏳ 计划中 |
| 2d | 测试套件 | 2-3 | 🔴 高 | ⏳ 计划中 |
| 3 | 500+ 行代码 | 2 周 | 🟢 低 | 可选 |
| 4 | 优化工作 | 1-2 周 | 🟢 低 | 可选 |

**总计**：12-18 天（Phase 1-2），+2-4 周（可选 Phase 3-4）

---

## 🎯 关键指标

### Phase 2 完成时的目标状态

| 指标 | 目标 | v1 基准 | 提升 |
|------|------|--------|------|
| 方面提取 F1 | 85-90% | 70-75% | +15% |
| 隐式方面检测率 | 70-80% | 0% | +70% |
| 去重率 | 75-80% | 60% | +15% |
| 评估准确度 | 85%+ | 70% | +15% |
| LLM 调用成本 | +25% | 基准 | 可接受 |

---

## 📋 依赖项检查

```
✅ sentence-transformers  (已有)
✅ sklearn.cluster.DBSCAN (需要: pip install scikit-learn)
✅ LLMClient 支持       (已有)
⚠️ GPU 内存 (聚类可能需要 4-8GB)
⚠️ LLM API 速率限制     (需要监控)
```

---

## 🚨 已知风险

| 风险 | 严重程度 | 缓解方案 |
|------|---------|---------|
| LLM prompt 过长导致超时 | 中 | 分离调用或使用函数调用 |
| DBSCAN eps 参数敏感 | 中 | 用验证集优化；支持参数化 |
| 隐式方面规则不完整 | 低 | Value 模式需补充；未来可集成 NER |
| 向后兼容性破坏 | 极低 | 严格遵循 JSON 扩展设计 |

---

## 📚 参考资源

**学术论文**（已收集）：
- [2024] A Systematic Review of Aspect-Based Sentiment Analysis
- [2025] Error Comparison Optimization for LLMs on ABSA (ACL)
- [2022] Amazon - Distantly Supervised Aspect Clustering (NAACL)

**开源工具**（已调研）：
- PyABSA (121 stars) - 基准对比
- InstructABSA (NAACL 2024) - 指令学习
- AutoABSA - 无监督方法

---

## ✅ 下一步行动

### 立即（今天）
- [x] Phase 1 实施完成
- [x] 4/4 测试通过
- [x] 创建路线图

### 本周
- [ ] 准备 Phase 2a 的 DBSCAN 聚类实现
- [ ] 创建聚类测试用例
- [ ] 评估计算成本

### 下周
- [ ] Phase 2a/2b 代码实现
- [ ] 集成测试准备

---

**更新时间**：2026-03-18
**路线图版本**：v1.0
