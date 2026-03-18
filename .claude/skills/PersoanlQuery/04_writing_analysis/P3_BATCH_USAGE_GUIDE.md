# P3批量错误分析脚本 - 使用指南

## 📝 脚本概述

**文件**: `05_p3_batch_error_analysis.py`  
**目的**: 使用P3最优模板对多个用户的评论进行批量错误提取  
**论文**: MTSummit 2025 (arXiv:2505.06004)  
**性能**: F1分数 +176% ~ +283%

---

## 🚀 快速开始

### 最简单的使用方法

```bash
# 处理所有选中的用户（使用默认路径）
python3 05_p3_batch_error_analysis.py
```

### 处理特定用户

```bash
# 处理单个用户
python3 05_p3_batch_error_analysis.py --user-ids A13OFOB1394G31

# 处理多个用户
python3 05_p3_batch_error_analysis.py --user-ids A13OFOB1394G31 A2GJX2KCUSR0EI A3RZ23PMNZGQC1
```

### 限制评论数量（用于测试）

```bash
# 只处理每个用户的前50条评论
python3 05_p3_batch_error_analysis.py --max-reviews 50

# 处理特定用户的前10条评论
python3 05_p3_batch_error_analysis.py --user-ids A13OFOB1394G31 --max-reviews 10
```

### 跳过汇总统计（加快速度）

```bash
python3 05_p3_batch_error_analysis.py --skip-summary
```

---

## 📋 完整参数列表

```bash
python3 05_p3_batch_error_analysis.py [OPTIONS]
```

### 输入参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--selected-users-file` | `/home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation/selected_users.json` | 选中用户列表文件 |
| `--reviews-dir` | `/home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation` | 评论文件所在目录 |

### 处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--user-ids [IDs]` | 所有选中用户 | 指定处理的用户ID (可多个) |
| `--max-reviews N` | 无限制 | 每个用户最多处理N条评论 |
| `--max-workers N` | 20 | 并发处理线程数 |

### 输出参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis` | 输出目录 |
| `--skip-summary` | 否 | 是否跳过生成汇总统计 |

---

## 📊 输出文件说明

脚本生成两种输出文件：

### 1. 单用户分析结果
**文件名**: `p3_analysis_{USER_ID}.json`

**结构**:
```json
{
  "user_id": "A2GJX2KCUSR0EI",
  "timestamp": "2026-03-18T10:50:55.123456",
  "method": "P3_optimal_template",
  "template": "Edit the following text...",
  "paper_reference": "MTSummit 2025 - arXiv:2505.06004",
  "processing_stats": {
    "total_reviews": 3,
    "reviews_with_errors": 3,
    "reviews_without_errors": 0,
    "total_words": 592,
    "total_errors_found": 4
  },
  "review_results": [
    {
      "original": "...",
      "corrected": "...",
      "has_errors": true,
      "error_count": 1,
      "extraction_status": "success"
    },
    ...
  ]
}
```

### 2. 批量处理汇总
**文件名**: `p3_batch_summary.json`

**结构**:
```json
{
  "timestamp": "2026-03-18T10:50:55.123456",
  "method": "P3_optimal_template",
  "total_users": 10,
  "processed_users": 10,
  "failed_users": [],
  "aggregate_stats": {
    "total_reviews_analyzed": 1500,
    "total_words": 150000,
    "total_errors_found": 450,
    "overall_error_rate": 0.30,
    "avg_errors_per_review": 0.30,
    "users_with_all_errors": 8,
    "users_with_some_correct": 2
  },
  "user_summaries": {
    "A13OFOB1394G31": {
      "reviews_analyzed": 150,
      "total_errors": 45,
      "error_rate_per_100_words": 0.30,
      "avg_errors_per_review": 0.30
    },
    ...
  }
}
```

---

## 💡 常见使用场景

### 场景1: 快速测试单个用户

```bash
# 验证脚本是否正常工作
python3 05_p3_batch_error_analysis.py \
  --user-ids A13OFOB1394G31 \
  --max-reviews 5
```

**期望输出**:
- P3分析完成日志
- 生成 `p3_analysis_A13OFOB1394G31.json`
- 生成 `p3_batch_summary.json`

---

### 场景2: 批量处理所有用户

```bash
# 处理所有选中的用户
python3 05_p3_batch_error_analysis.py
```

**处理流程**:
1. 加载 `selected_users.json` 中的所有用户
2. 验证 `reviews_{USER_ID}.json` 文件存在
3. 对每个用户调用P3分析
4. 生成汇总统计

**输出**:
- 每个用户一个 `p3_analysis_*.json`
- 一个 `p3_batch_summary.json`

---

### 场景3: 对标对比

```bash
# 只处理前20条评论（快速运行）
python3 05_p3_batch_error_analysis.py --max-reviews 20

# 然后对比与character_level方法的结果
# 使用命令: python3 04_extract_all_user_errors.py (默认是character_level)
```

**对比内容**:
- 错误数量
- 错误率
- 处理时间
- 输出质量

---

### 场景4: 分批处理大量用户

```bash
# 第一批
python3 05_p3_batch_error_analysis.py \
  --user-ids A13OFOB1394G31 A2GJX2KCUSR0EI A3RZ23PMNZGQC1

# 第二批
python3 05_p3_batch_error_analysis.py \
  --user-ids A1GYEGLX3P2Y7P A211W8JLJFDIC0
```

---

## 🔍 输出日志解读

### 处理流程日志

```
[2026-03-18 10:45:48] ================================================================================
[2026-03-18 10:45:48] Stage 4C: P3 Optimal Template Batch Error Analysis
[2026-03-18 10:45:48] ================================================================================
[2026-03-18 10:45:48] Found 2 selected users
[2026-03-18 10:45:48] Validating user review files...
[2026-03-18 10:45:48]   ✓ User A13OFOB1394G31: 200 reviews
[2026-03-18 10:45:48]   ✓ User A2GJX2KCUSR0EI: 150 reviews
[2026-03-18 10:45:48] Processing with P3 template...
```

**含义**:
- `✓` = 用户验证成功
- 括号内的数字 = 该用户的评论数

### P3分析日志

```
[2026-03-18 10:45:50] Running P3 optimal template error analysis...
[2026-03-18 10:45:50] [A13OFOB1394G31] Processing user with P3 template...
[2026-03-18 10:46:15] [A13OFOB1394G31] ✓ P3 analysis completed
```

**含义**:
- P3分析正在进行中
- `✓` = 用户分析成功

### 汇总日志

```
[2026-03-18 10:46:20] ================================================================================
[2026-03-18 10:46:20] P3 BATCH ANALYSIS SUMMARY
[2026-03-18 10:46:20] ================================================================================
[2026-03-18 10:46:20] Processed users: 2/2
[2026-03-18 10:46:20] Total reviews analyzed: 350
[2026-03-18 10:46:20] Total words analyzed: 35000
[2026-03-18 10:46:20] Total errors found: 105
[2026-03-18 10:46:20] Overall error rate: 0.30/100 words
```

**含义**:
- 处理了多少用户
- 分析了多少评论和单词
- 共找到的错误数
- 平均错误率

---

## ⚙️ 高级用法

### 自定义输出目录

```bash
python3 05_p3_batch_error_analysis.py \
  --output-dir /path/to/custom/output
```

### 调整并发数

```bash
# 增加并发数（更快但占用更多资源）
python3 05_p3_batch_error_analysis.py --max-workers 50

# 降低并发数（更慢但占用资源少）
python3 05_p3_batch_error_analysis.py --max-workers 5
```

### 使用自定义的users文件

```bash
python3 05_p3_batch_error_analysis.py \
  --selected-users-file /path/to/custom_users.json
```

---

## 📊 结果分析

### 查看单个用户的详细结果

```python
import json

# 读取单个用户的分析结果
with open('p3_analysis_A13OFOB1394G31.json', 'r') as f:
    result = json.load(f)

# 获取基本统计
print(f"用户: {result['user_id']}")
print(f"分析评论数: {len(result['review_results'])}")
print(f"总错误数: {sum(r['error_count'] for r in result['review_results'])}")
print(f"有错误的评论: {sum(1 for r in result['review_results'] if r['has_errors'])}")
```

### 查看汇总统计

```python
import json

# 读取批量处理汇总
with open('p3_batch_summary.json', 'r') as f:
    summary = json.load(f)

# 获取聚合统计
agg = summary['aggregate_stats']
print(f"总用户: {summary['total_users']}")
print(f"成功: {summary['processed_users']}")
print(f"总评论: {agg['total_reviews_analyzed']}")
print(f"总错误: {agg['total_errors_found']}")
print(f"错误率: {agg['overall_error_rate']}/100词")

# 比较用户
user_errors = {
    uid: data['total_errors']
    for uid, data in summary['user_summaries'].items()
}
print(f"\n用户错误排名:")
for uid, count in sorted(user_errors.items(), key=lambda x: x[1], reverse=True):
    print(f"  {uid}: {count}个错误")
```

---

## 🐛 故障排查

### 问题: "No users to process"

**原因**: 找不到选中的用户

**解决**:
```bash
# 检查selected_users.json是否存在
ls -la result/personal_query/00_data_preparation/selected_users.json

# 指定用户ID
python3 05_p3_batch_error_analysis.py --user-ids A13OFOB1394G31
```

---

### 问题: "reviews_{USER_ID}.json not found"

**原因**: 评论文件不存在

**解决**:
```bash
# 检查评论文件
ls -la result/personal_query/00_data_preparation/reviews_*.json

# 检查文件内容
python3 -c "import json; json.load(open('reviews_A13OFOB1394G31.json'))" && echo "✓ File OK"
```

---

### 问题: "P3 analysis script not found"

**原因**: `04_p3_error_extraction.py` 缺失

**解决**:
```bash
# 检查P3脚本
ls -la 04_p3_error_extraction.py

# 确保在正确的目录
cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/04_writing_analysis/
python3 05_p3_batch_error_analysis.py
```

---

## 📚 与其他脚本的关系

### vs 04_extract_all_user_errors.py

| 特性 | 04_extract_all_user_errors.py | 05_p3_batch_error_analysis.py |
|------|------------------------------|-------------------------------|
| 方法选择 | ✓ 支持 (--method) | ✗ 仅P3 |
| 使用难度 | 较复杂 | 简单 |
| 字符级方法 | ✓ 支持 | ✗ 不支持 |
| P3方法 | ✓ 支持 | ✓ 支持 |
| 最佳用途 | 方法对比 | P3专用处理 |

### vs 04_p3_error_extraction.py

| 特性 | 04_p3_error_extraction.py | 05_p3_batch_error_analysis.py |
|------|--------------------------|-------------------------------|
| 处理数量 | 单用户 | 多用户 |
| 汇总统计 | ✗ 不生成 | ✓ 自动生成 |
| 用户验证 | ✗ 不验证 | ✓ 自动验证 |
| 简单度 | ✓ 更简单 | ✗ 更复杂 |

---

## 🎓 P3模板回顾

```
Edit the following text for spelling and grammar mistakes, 
make minimal changes, and return only the corrected text. 
If the text is already correct, return it without any explanations:.
```

**关键特点**:
1. **明确范围**: 拼写 + 语法
2. **最小化约束**: "make minimal changes"
3. **正确文本处理**: "If already correct, return it"
4. **输出规范**: "return only the corrected text"

**性能**:
- F1分数提升: +176% ~ +283%
- 多语言验证: 英、德、意、瑞
- 最优模型: Gemma 9B

---

## 📞 获取帮助

### 查看脚本帮助

```bash
python3 05_p3_batch_error_analysis.py --help
```

### 查看脚本文档

```bash
head -50 05_p3_batch_error_analysis.py
```

### 查看生成的日志

所有操作都有带时间戳的日志输出，可以直接在终端查看或保存到文件：

```bash
# 保存日志到文件
python3 05_p3_batch_error_analysis.py > processing.log 2>&1

# 查看日志
tail -50 processing.log
```

---

## 🎯 总结

**05_p3_batch_error_analysis.py** 是一个简化的批量处理脚本，特别设计用于：

✓ 直接使用P3最优模板  
✓ 批量处理多个用户  
✓ 自动生成汇总统计  
✓ 最少的参数配置  

**推荐使用场景**:
- 当你确定要使用P3方法时
- 批量处理多个用户
- 需要自动化汇总报告
- 对比性能表现

**相比主控脚本的优势**:
- 更简洁的命令行参数
- 自动生成汇总统计
- 更清晰的输出结构
- 专注于P3方法
