# Stage 7 改进优先级选择 - 使用指南

## 🎯 改进内容

### 原始版本 (Method 2)

```python
优先级 = 包含用户错误词？
```

**问题**：
- 70% 被修改的词不在目标产品中
- 导致 Stage 10 噪声注入无效

### 改进版本 (v2) ⭐

```python
优先级 = (包含用户错误词) AND (该词在目标产品中)
```

**改进**：
- 确保选择的词既包含用户错误，又在产品中存在
- 提升 Stage 10 噪声注入的有效性
- 真正测试检索器的鲁棒性

---

## 📁 文件说明

### 核心文件

1. **stage7_improved_priority_selection.py**
   - 改进的优先级选择逻辑
   - ProductVocabulary 类（产品词汇库）
   - select_dimensions_with_improved_priority() 函数

2. **07_generate_with_method2.py** (已修改)
   - 导入改进的选择器
   - 使用 ImprovedPrioritySelector 实例
   - 传递给每个查询处理函数

3. **test_improved_selector.py**
   - 测试脚本
   - 验证选择逻辑是否正确

---

## 🚀 使用方法

### 方法1: 测试改进的选择器

```bash
cd /fs04/ar57/wenyu
python test_improved_selector.py
```

**预期输出**：
```
- 加载用户错误模式
- 检查产品词汇
- 使用改进的优先级选择
- 显示选择的5个属性
- 分析每个属性是否包含用户错误词且在产品中
```

### 方法2: 重新生成 Stage 7 查询

```bash
cd /fs04/ar57/wenyu/.claude/skills/PersoanlQuery/07_query

# 使用改进的选择器重新生成查询
python 07_generate_with_method2.py \
    --user-id A13OFOB1394G31 \
    --input-file /fs04/ar57/wenyu/result/personal_query/03_processing/query_A13OFOB1394G31.json \
    --writing-analysis-file /fs04/ar57/wenyu/result/personal_query/05_writing_analysis/results/writing_analysis_A13OFOB1394G31.json \
    --output-dir /fs04/ar57/wenyu/result/personal_query/07_query \
    --meta-file /fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
    --max-workers 5
```

---

## 📊 预期改进效果

### 之前（Method 2）

```
优先级 = 包含用户错误词

查询: "balloon dies that work with my machine"
  ✓ 包含 "machine"（用户错误词）

目标产品: "Party Balloons Die"
  ❌ 不包含 "machine"

修改: "machine" → "maching"
  ❌ 无效（两个词都不在产品中）
```

### 之后（Improved v2）

```
优先级 = (包含用户错误词) AND (在目标产品中)

查询: "balloon dies for Spellbinders"
  ✓ 包含 "Spellbinders"（用户错误词）
  ✓ "Spellbinders" 在目标产品中

目标产品: "Spellbinders Party Balloons Die"
  ✓ 包含 "Spellbinders"

修改: "Spellbinders" → "Spellbinder's"
  ✅ 有效（修改影响检索结果）
```

---

## 🔍 验证改进效果

### 步骤1: 检查生成的查询

运行改进版本后，检查：

```bash
python3 << 'EOF'
import json

with open('result/personal_query/07_query/dual_queries_A13OFOB1394G31.json') as f:
    data = json.load(f)

results = data.get('results', [])

# 统计包含用户错误词的查询
user_errors_in_queries = 0
words_in_product = 0

for result in results:
    tuq = result.get('target_user_query', {})
    query = tuq.get('query', '')

    if 'selected_attributes' in tuq:
        # 检查每个属性
        for attr in tuq['selected_attributes']:
            # 这里应该标记该属性是否既包含错误词又在产品中
            pass

print(f"包含用户错误词的查询: {user_errors_in_queries}/{len(results)}")
print(f"词在产品中的数量: {words_in_product}")
EOF
```

### 步骤2: 运行 Stage 10 噪声注入

```bash
# 使用新生成的 Stage 7 查询
cd /fs04/ar57/wenyu/.claude/skills/PersoanlQuery/10_targeted_noisy_query

python 10_generate_noisy_queries.py
```

### 步骤3: 验证改进效果

```bash
# 运行检索评估
cd /fs04/ar57/wenyu/.claude/skills/PersoanlQuery/13_retrieval

python ./..

# 对比 Clean vs Noisy
python scripts/compare_results.py
```

**预期结果**：
- Noisy 应该**系统性地低于** Clean
- 不应该再有 "Noisy > Clean" 的异常
- 下降幅度应该更明显（例如 5-10%）

---

## 💡 工作原理详解

### 优先级分类

#### 最高优先级 ⭐⭐⭐
```python
条件: (包含用户错误词) AND (在目标产品中)

示例:
  属性: "Works with my machine"
  ✓ "machine" 是用户错误词（machine → maching）
  ✓ "machine" 在目标产品中
  → 优先选择
```

#### 中优先级 ⭐⭐
```python
条件: (包含用户错误词) BUT (不在目标产品中)

示例:
  属性: "machine for crafting"
  ✓ "machine" 是用户错误词
  ✗ "machine" 不在目标产品中
  → 备用选择
```

#### 低优先级 ⭐
```python
条件: (不包含用户错误词)

示例:
  属性: "Easy to use"
  ✗ 不包含用户错误词
  → 最后选择
```

---

## 📈 性能对比

| 版本 | 选择策略 | 预期效果 |
|------|---------|---------|
| Method 2 (旧) | 包含用户错误词 | 87% 查询包含错误词，但70%无效 |
| Improved v2 (新) | 包含错误词 + 在产品中 | 预计 60-70% 查询包含有效错误词 |

---

## ⚠️ 注意事项

### 1. 性能考虑

**产品词汇加载**：需要访问元数据文件
- 首次加载：需要扫描JSON文件
- 后续查询：使用缓存的词汇

**优化**：
- 使用 gzip 压缩的元数据
- 只加载目标产品（46个）的词汇
- 缓存已加载的词汇

### 2. 兼容性

**需要**：
- Stage 5 写作分析文件
- Stage 3 查询结果文件
- 产品元数据文件

**输出格式**：
- 与原始版本完全兼容
- 可以直接用于 Stage 10

### 3. 可复现性

**随机种子**：
- 目前没有设置随机种子
- 每次运行结果可能不同

**建议**：
```python
# 在 main() 函数中添加
random.seed(42)  # 设置固定种子
```

---

## 🎓 总结

### 核心改进

1. ✅ **双重验证**：用户错误词 + 产品存在性
2. ✅ **三级优先级**：最高/中/低优先级
3. ✅ **向后兼容**：输出格式不变
4. ✅ **有效提升**：预计噪声注入有效性提升 3-4倍

### 下一步

1. 运行测试脚本验证逻辑
2. 重新生成 Stage 7 查询
3. 运行 Stage 10 噪声注入
4. 运行 Stage 13 检索评估
5. 对比改进前后的效果

需要我帮你运行这些步骤吗？
