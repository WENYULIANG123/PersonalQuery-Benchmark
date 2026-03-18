# Fail-Fast 设计实现报告

## 设计理念变更

### 【之前】Graceful Degradation (优雅降级)
```
异常值 → 返回默认值 → 继续处理 (隐藏问题)
```
- ❌ 隐藏数据质量问题
- ❌ 难以诊断错误根源
- ❌ 可能产生错误结果

### 【现在】Fail-Fast (快速失败)
```
异常值 → 立即抛出具体错误 → 强制修复数据 (暴露问题)
```
- ✅ 清晰的错误信息
- ✅ 快速定位数据问题
- ✅ 强制数据质量改进

---

## 实现的Fail-Fast函数

### 1. safe_str_len() - 字符串长度
```python
def safe_str_len(value, context):
    if value is None:
        raise TypeError(f"[{context}] Cannot get length of None value")
    
    if isinstance(value, str):
        return len(value)  # ✅ 只有字符串成功
    
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(f"[{context}] Cannot get length of NaN")
        if math.isinf(value):
            raise ValueError(f"[{context}] Cannot get length of Infinity")
        raise TypeError(f"[{context}] Expected str, got float")
    
    raise TypeError(f"[{context}] Expected str, got {type(value).__name__}")
```

**行为**：只接受字符串，其他任何类型都抛出详细错误

### 2. safe_list_len() - 列表长度
```python
def safe_list_len(value, context):
    if value is None:
        raise TypeError(f"[{context}] Cannot get length of None")
    
    if isinstance(value, (list, tuple)):
        return len(value)  # ✅ 只有列表/元组成功
    
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(f"[{context}] Cannot get length of NaN")
        if math.isinf(value):
            raise ValueError(f"[{context}] Cannot get length of Infinity")
        raise TypeError(f"Expected list, got float")
    
    if isinstance(value, dict):
        raise TypeError(f"Expected list, got dict with {len(value)} keys")
    
    raise TypeError(f"Expected list, got {type(value).__name__}")
```

**行为**：只接受列表/元组，其他类型都抛出错误并说明预期类型

### 3. ensure_string() - 字符串转换
```python
def ensure_string(value, context):
    if value is None:
        raise TypeError(f"[{context}] Cannot convert None to string")
    
    if isinstance(value, str):
        return value  # ✅ 只有字符串直接返回
    
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(f"[{context}] Cannot convert NaN to string")
        if math.isinf(value):
            raise ValueError(f"[{context}] Cannot convert Infinity to string")
        raise TypeError(f"Expected str, got float")
    
    raise TypeError(f"Expected str, got {type(value).__name__}")
```

**行为**：不自动转换类型，任何非字符串都抛出错误

### 4. safe_dict_get() - 字典访问
```python
def safe_dict_get(obj, key, context):
    if not isinstance(obj, dict):
        raise TypeError(f"[{context}] Expected dict, got {type(obj).__name__}")
    
    if key not in obj:
        raise KeyError(f"[{context}] Key '{key}' not found")
    
    value = obj[key]
    
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(f"[{context}] Value for '{key}' is NaN")
        if math.isinf(value):
            raise ValueError(f"[{context}] Value for '{key}' is Infinity")
    
    return value
```

**行为**：严格验证字典和key，异常值立即抛出

---

## 错误信息设计

每个错误都包含：

1. **ASIN标识** - `[{asin}]` 快速定位产品
2. **错误类型** - `TypeError`, `ValueError`, `KeyError` 说明问题类别
3. **期望值说明** - "Expected str, got float" 清晰对比
4. **上下文信息** - 哪个字段/维度出错
5. **值预览** - 实际值是什么（截断前100字符）

### 错误示例

```
[B004HS60TG] TypeError: [target_review_str_B004HS60TG] Expected str, got float: nan

[B001DECKGY] ValueError: [entities_B001DECKGY_Product_Attributes_Material] 
Cannot get length of NaN (float('nan'))

[B00VN4V2PO] TypeError: Category 'Quality_Attributes', dimension 'Durability': 
expected list of entities, got float. Value: nan
```

---

## 调用链变更

### 【之前】处理流程
```python
try:
    extraction = extract_preferences_from_review_v2(...)  # 主方法
except Exception:
    extraction = rule_based_extraction(...)  # fallback降级
    # 继续处理，隐藏错误
```

### 【现在】处理流程
```python
# 直接调用，不使用fallback
extraction = extract_preferences_from_review_v2(...)

# 任何错误都会立即抛出给上层处理
# 上层可以记录错误产品并中断处理
```

---

## 统计计算变更

### count_entities()
```python
# 【之前】遇到异常类型 → 返回0
if isinstance(entities, list):
    total += len(entities)
else:
    fixed = fix_entities(entities)  # fallback尝试修复
    total += len(fixed)  # 修复失败返回0

# 【现在】遇到异常类型 → 立即抛出错误
if not isinstance(entities, list):
    raise TypeError(f"expected list of entities, got {type(entities).__name__}")
total += safe_list_len(entities)  # 通过验证后计算
```

### count_categories()
```python
# 【之前】 → 跳过异常值继续
for v in category_data.values():
    if isinstance(v, list):
        # 处理
    else:
        # fallback修复
        continue  # 跳过

# 【现在】 → 异常值立即报错
for v in category_data.values():
    if not isinstance(v, list):
        raise TypeError(f"expected list, got {type(v).__name__}")
    # 处理
```

---

## 优势

### 1. 快速问题定位
```
错误出现立即抛出 → 确切知道哪个产品/字段出错 → 迅速修复
```

### 2. 数据质量保证
```
无法隐藏异常值 → 强制修复数据源 → 长期提升数据质量
```

### 3. 易于调试
```
清晰的错误链 → 直接追踪问题根源 → 无需猜测
```

### 4. 系统可靠性
```
不处理异常数据 → 确保输出数据100%有效 → 下游系统安全
```

---

## 完整的错误处理链

```
用户调用
   ↓
extract_preferences_v2()
   ├─ 解析异常 → 抛出ParseError
   └─ 提取异常 → 抛出ExtractionError
   ↓
process_product()
   ├─ 评论长度计算 → 抛出TypeError/ValueError
   ├─ 数据验证 → 抛出KeyError/TypeError
   └─ 统计计算 → 抛出TypeError
   ↓
main()
   ├─ 捕获所有异常
   ├─ 记录错误产品
   ├─ 记录错误信息
   └─ 中断处理该产品，继续下一个

result日志
   ├─ 成功产品列表
   └─ 失败产品列表 + 具体错误信息
```

---

## 部署注意

✅ **严格模式**：不再隐藏任何错误
- 提取失败 → 产品标记为失败
- 数据错误 → 立即抛出错误
- 类型不匹配 → 拒绝处理

✅ **错误记录**：完整的错误追踪
- 错误产品ASIN
- 具体错误类型
- 错误位置（字段/维度）
- 错误原因描述

✅ **下游兼容**：输出结果100%有效
- 成功提取的产品数据完整准确
- 失败产品清晰标记
- 不会产生部分/错误的数据

---

## 总结

✅ **从容错系统升级到严格验证系统**

**之前**：尽力处理异常数据，可能产生错误结果  
**现在**：拒绝异常数据，确保结果准确或明确失败

这是一个从**best-effort**到**guarantee**的转变，提升了系统的可靠性和可维护性。

---

**设计日期**: 2026-03-18  
**类型**: Architecture Decision  
**状态**: Implemented
