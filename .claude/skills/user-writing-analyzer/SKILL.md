---
name: user-writing-analyzer
description: 分析用户的 Amazon 评论，通过脚本生成分析 Prompt，引导 Agent 手动逐个提取拼写错误和语法错误习惯，并直接写入 error_analysis_USERID.json 文件。⚠️ 严禁批量脚本生成，必须一个一个手动生成。
allowed-tools: run_command, view_file, ask_user_question
---

# User-Writing-Analyzer (Spelling & Grammar)

此技能用于让 Agent (Claude) 模拟特定用户的拼写和语法习惯，通过分析用户的评论历史，**逐个手动**提取其错误模式，直接追加到 `error_analysis_USERID.json` 文件的数组中。

---

## 文件路径规范 (Standard Paths)

- **输入评论文件**: `/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/user_reviews/user_product_reviews.json`
- **默认用户 ID**: `A13OFOB1394G31`
- **输出目录**: `/home/wlia0047/ar57/wenyu/result/writing_analysis`

## 🚀 快速参考（AI必读）

### ⚡ 核心要求（4条必须遵守）

1. **每批最多10条** - 不是101条一次性完成！
2. **直接写入最终文件** - 使用标准 `.json` 数组格式，每条评论分析完立即更新文件，要求带缩进。
3. **全面分析** - 同时识别拼写错误（10种）和语法错误（7种），不要遗漏任何一种错误类型。
4. **🔴 关键要求：每个产品必须独立完成完整的 CoT 推理和质量验证**

### 🎯 执行流程

```
开始 → 分析10条 → 自动质量检测
  ↓                     ↓
通过 → 继续下10条     失败 → 删除重来
  ↓                     ↓
重复直到完成         重新分析
  ↓
全部完成 → 完成
```

---

## ⛔🚨🚨 绝对禁止条款 - 违反即任务失败 🚨🚨⛔

1. ❌ **严禁使用批量处理脚本进行 LLM 分析调用**
2. ❌ **严禁为了凑数而编造错误**
3. ❌ **遗漏任何一种错误类型** - 必须同时检查拼写错误和语法错误

---

## 📂 拼写错误分类 (Spelling Errors - 10 Types)

### 1. Deletion (漏输/缺失字母)
- `colr` -> `color` (缺失 `o`)

### 2. Insertion (多输/多余字母)
- `accross` -> `across` (多了 `c`)

### 3. Transposition (换位/相邻字母颠倒)
- `teh` -> `the`, `thier` -> `their`

### 4. Scramble (复杂混乱/多重错误)
- `definitly` -> `definitely` (既有缺失又有替换)

### 5. Substitution (替换/字母替换)
- `wprk` -> `work` (p 替换了 o)

### 6. Homophone (同音词/音近形异)
- `there` -> `their`, `your` -> `you're` (注意：原词必须是合法单词，但用法错误)

### 7. Suffix (后缀错误/词尾变形错误)
- `runing` -> `running`, `boxs` -> `boxes`

### 8. Hard Word (难词/生僻词错误)
- `fuchsia` -> `fushia`

### 9. Extra Space (多余空格)
- `note book` -> `notebook`

### 10. Extra Hyphen (多余连字符)
- `note-book` -> `notebook`

---

## 📂 语法错误分类 (Grammar Errors - 7 Types)

### 1. Agreement (一致性错误)
主谓不一致、单复数不一致、冠词使用错误等
- `it is` -> `they are` (主语是复数)
- `these kit is` -> `these kits are` (数不一致)
- `a rayon` -> `rayon` (不可数名词不该加冠词)

### 2. Collocation (搭配错误)
词组搭配不自然，固定用法错误
- `between 4 or 5` -> `between 4 and 5` (between应该搭配and)
- `first glance` -> `first attempt` (语境搭配不当)
- `fit in the scheme` -> `fit into the scheme` (介词搭配错误)

### 3. Preposition (介词错误)
介词缺失、错误或多余
- `slots the types` -> `slots for the types` (缺失介词for)
- `excel that set` -> `excel at that set` (缺失介词at)
- `range of light to dark` -> `range from light to dark` (介词错误)

### 4. Pronoun (代词错误)
代词指代不明、使用错误
- `what I consider` -> `which I consider` (关系代词错误)
- `who` -> `that` (非人称代词应用that)
- `these` -> `they` (指代不明)

### 5. Suffix (后缀/词形错误)
比较级、词性后缀、动词形态错误
- `more fine` -> `finer` (比较级应为finer)
- `coarsely` -> `coarse` (应为形容词coarse)
- `to using` -> `to use` (不定式应为use)

### 6. Homophone (同音词-语法类)
动词时态或形式相关的同音词误用
- `lay down` -> `lie down` (动词时态错误，lay是过去式，lie才是原形)

### 7. Hyphenation (连字符错误)
复合形容词缺少或多余连字符
- `good size` -> `good-sized` (复合形容词应加连字符)
- `cross stitch thread` -> `cross-stitch thread` (复合词应加连字符)

---

## 执行步骤

### 1. 生成 Prompt
```bash
python3 /home/wlia0047/ar57/wenyu/.claude/skills/user-writing-analyzer/generate_writing_prompts.py \
    --input /home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/user_reviews/user_product_reviews.json \
    --user_id A13OFOB1394G31 \
    --output /home/wlia0047/ar57/wenyu/result/writing_analysis/prompts.json
```

### 2. 手动分析并写入 (核心步骤 - CRITICAL LOGIC)

分析每个 Prompt，**必须**显式展示思维链推理 (CoT) 和合理性分析：

#### 🔍 CoT 推理过程 (每个 Prompt 必须展示)
1. **识别错误**: 仔细阅读评论，找出所有可能的拼写和语法错误。
2. **分类与纠正**: 根据本技能定义的 10 种拼写错误和 7 种语法错误进行分类，并给出正确形式。
3. **提取片段**: 记录错误所在的文本片段 (context)。

#### ✅ CoT 合理性分析 - 必须对当前推理过程进行验证：
- **检查项 1**: 识别的错误是否真实存在 (非品牌名、非缩写、非标点争议)。
- **检查项 2**: 分类是否准确 (例如：是 Deletion 还是 Scramble)。
- **检查项 3**: 是否遗漏了任何一种类型的错误。
- **检查项 4**: 纠正建议是否符合标准英语规范。
- **⚠️ 只有确认合理后，才执行写入操作。**

#### 💾 写入结果
使用以下输出模板（**仅包含发现错误的类别，若类别为空则不写入**）：

```json
{
  "review_index": 0,
  "spelling_errors": {
    "Deletion": [
      { "original": "colr", "corrected": "color", "fragment": "...pretty colr for...", "reason": "Missing 'o'" }
    ]
  },
  "grammar_errors": {
    "Agreement": [
      { "original": "it is", "corrected": "they are", "fragment": "...gemstones and it is...", "reason": "Subject 'gemstones' is plural" }
    ]
  }
}
```

**写入逻辑示例（手动维护 JSON 数组，注意过滤空类别）：**
```python
import json
import os

# 配置
output_dir = "/home/wlia0047/ar57/wenyu/result/writing_analysis"
user_id = "USER_ID_HERE" # 记得修改ID
analysis_file = os.path.join(output_dir, f"error_analysis_{user_id}.json")
stats_file = os.path.join(output_dir, f"error_stats_{user_id}.json")

# 构建结果时过滤掉空列表 (raw_errors 来自你的手动分析)
# raw_errors = { ... } 

# 过滤空类别
spelling_errors = {k: v for k, v in raw_errors["spelling_errors"].items() if v}
grammar_errors = {k: v for k, v in raw_errors["grammar_errors"].items() if v}

# 1. 写入结果文件 (仅当有错误时)
if spelling_errors or grammar_errors:
    res = {
        "review_index": 0, # 记得修改索引
        "spelling_errors": spelling_errors,
        "grammar_errors": grammar_errors
    }

    if os.path.exists(analysis_file):
        with open(analysis_file, 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append(res)
    with open(analysis_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ 已写入错误分析: {analysis_file}")
else:
    print("ℹ️ 未发现错误，跳过写入结果文件。")

# 2. 更新统计文件 (无论是否有错，都可以记录已分析数量，这里主要统计错误数)
if os.path.exists(stats_file):
    with open(stats_file, 'r') as f:
        stats = json.load(f)
else:
    stats = {"spelling": {}, "grammar": {}, "total_reviews_analyzed": 0}

stats["total_reviews_analyzed"] = stats.get("total_reviews_analyzed", 0) + 1

# 统计此条评论的错误
for category, errors in spelling_errors.items():
    stats["spelling"][category] = stats["spelling"].get(category, 0) + len(errors)

for category, errors in grammar_errors.items():
    stats["grammar"][category] = stats["grammar"].get(category, 0) + len(errors)

# 🔥 重要：添加汇总字段（每次更新时重新计算）
spelling_total = sum(stats.get("spelling", {}).values())
grammar_total = sum(stats.get("grammar", {}).values())
stats["spelling_total"] = spelling_total
stats["grammar_total"] = grammar_total
stats["total_errors"] = spelling_total + grammar_total

# 🆕 计算总单词数和错误率（需要读取所有评论）
# 注意：这里假设你有所有评论的文本，实际使用时需要替换为你的数据源
# stats["total_words"] = calculate_total_words(all_reviews)  # 需要实现这个函数
# stats["errors_per_100_words"] = round((stats["total_errors"] / stats["total_words"]) * 100, 2)

with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(f"📊 已更新统计数据: {stats_file}")
```

**输出格式：**
```json
[
  {
    "review_index": 0,
    "spelling_errors": {
      "Deletion": [],
      "Insertion": []
    },
    "grammar_errors": {
      "Agreement": [],
      "Collocation": []
    }
  },
  ...
]
```

### 📊 统计文件格式要求

统计文件 `error_stats_{USER_ID}.json` **必须**包含以下字段：

```json
{
  "spelling": {
    "Deletion": 2,
    "Insertion": 1,
    ...
  },
  "grammar": {
    "Agreement": 5,
    "Collocation": 3,
    ...
  },
  "total_reviews_analyzed": 101,
  "spelling_total": 10,
  "grammar_total": 31,
  "total_errors": 41,
  "total_words": 7940,
  "errors_per_100_words": 0.52
}
```

**字段说明：**
- `spelling`: 拼写错误按类别统计
- `grammar`: 语法错误按类别统计
- `total_reviews_analyzed`: 已分析的总评论数
- `spelling_total`: 拼写错误总数（所有类别之和）✨ **必填**
- `grammar_total`: 语法错误总数（所有类别之和）✨ **必填**
- `total_errors`: 总错误数（拼写 + 语法）✨ **必填**
- `total_words`: 总单词数（所有评论的单词总和）✨ **必填**
- `errors_per_100_words`: 错误率（每100个单词的错误数）✨ **必填**

⚠️ **重要**：每次更新统计文件时，必须重新计算以下汇总字段：
- `spelling_total`、`grammar_total`、`total_errors`
- `total_words`（需要读取所有评论计算单词数）
- `errors_per_100_words` = `(total_errors / total_words) * 100`
