---
name: product-query-generator
description: 读取 preference_match.json 文件，通过脚本生成 Prompt，引导 Agent 手动逐个生成并润色自然语言查询。必须包含全部3个属性并进行语义转换。⚠️ 严禁批量脚本生成，必须一个一个手动生成。
allowed-tools: run_command, view_file, ask_user_question
---

# Product-Query-Generator

此技能用于让 Agent (Claude) 扮演真实购物者，通过分析 `result/preference_match/preference_match.json` 里的商品属性，**逐个手动生成**高质量的搜索查询。

## 🚨 严禁行为

❌ **严禁使用任何批量处理脚本**
❌ **严禁使用Python循环批量生成query**
❌ **严禁一次性读取所有prompt后批量处理**
❌ **严禁使用模板批量替换属性**

✅ **必须逐个读取、逐个思考、逐个生成、逐个验证**

## ⚠️ 核心原则

1. **必须包含全部3个属性** - 不得遗漏
2. **必须进行语义转换** - 不得复制粘贴原始属性
3. **严格长度控制** - 25-30个单词（不是字符）
4. **逐个手动生成** - 每个query单独处理，独立思考，严禁批量

## 执行流程

### 阶段 1：生成 Prompt 上下文 (Context Generation)

运行脚本，将 `preference_match.json` 中的属性转化为针对每个商品的详细 Prompt。

```bash
mkdir -p /home/wlia0047/ar57/wenyu/result/clean_query

python3 /home/wlia0047/ar57/wenyu/.claude/skills/product-query-generator/generate_query_prompts.py \
    --input /home/wlia0047/ar57/wenyu/result/preference_match/preference_match.json \
    --output /home/wlia0047/ar57/wenyu/result/clean_query/query_prompts.json
```

### 阶段 2：逐个手动生成查询 (Query Generation)

🔴 **必须按照以下步骤，一个一个处理，不得跳过或合并：**

#### 步骤 1：准备输出文件

```bash
# 创建CSV文件，只写入header
echo "id,query,answer_ids_source" > /home/wlia0047/ar57/wenyu/result/clean_query/clean_queries.csv
```

#### 步骤 2：逐个处理每个Prompt

**对于每个query，必须独立执行以下步骤：**

1. **读取单个Prompt** - 使用 `jq` 或 Python 读取指定索引的prompt
2. **分析属性** - 理解每个属性的含义，思考如何语义转换
3. **手动生成Query** - 基于理解，手动组织语言，生成query
4. **验证质量** - 检查是否满足所有要求
5. **追加到CSV** - 将单个query追加到CSV文件
6. **确认完成** - 确认后继续下一个

**具体操作示例：**

```bash
# 读取第0个prompt (索引从0开始)
python3 -c "import json; d=json.load(open('/home/wlia0047/ar57/wenyu/result/clean_query/query_prompts.json')); print(d[0]['prompt'])"
```

然后手动思考并生成query，验证后追加：

```bash
# 追加到CSV (替换为实际生成的query)
echo "0,I need fabric paint with beautiful shimmer effect that lasts through many projects and has rich concentrated colors,B000BGSZFU" >> /home/wlia0047/ar57/wenyu/result/clean_query/clean_queries.csv
```

**重复以上步骤，处理所有102个prompt。**

#### 步骤 3：逐个验证检查清单

每个query生成后，必须验证：

- [ ] **包含全部3个属性** - 没有遗漏
- [ ] **25-30个单词** - 使用 `echo "query" | wc -w` 验证
- [ ] **已进行语义转换** - 没有直接复制原始属性
- [ ] **第一人称自然语气** - 像真实购物者
- [ ] **语法正确** - 没有语法错误
- [ ] **无重复内容** - 属性和词汇不重复

## 📋 逐个处理检查清单

处理每个prompt时，按顺序确认：

- [ ] 1. 读取了单个prompt (不是批量)
- [ ] 2. 理解了商品类别
- [ ] 3. 理解了全部3个属性
- [ ] 4. 对每个属性进行了语义转换思考
- [ ] 5. 手动组织语言生成query
- [ ] 6. 验证了单词数 (25-30)
- [ ] 7. 验证了包含所有3个属性
- [ ] 8. 验证了没有复制原始属性
- [ ] 9. 验证了语法正确性
- [ ] 10. 追加到CSV文件
- [ ] 11. 继续下一个prompt

## 润色指南

### 核心要求
- **必须包含全部3个属性** - 不得遗漏任何属性
- **语义转换** - 将属性转化为自然购物语言，不得复制粘贴
- **长度** - 严格控制在 **25-30 个单词**（English words count）
- **逐个手动生成** - 每个 query 必须单独处理，独立思考

### 语义转换示例

| 原始属性 | ❌ 直接复制 | ✅ 语义转换 |
|---------|-----------|-----------|
| Long-lasting/Durable | "long-lasting durable" | "lasts through many projects" |
| High pigment concentration | "high pigment concentration" | "rich concentrated colors" |
| Compact packaging | "compact packaging" | "fits in my art bag" |
| Color names on pencils | "color names on pencils" | "easy to identify colors" |
| Clean application | "clean application" | "applies smoothly without mess" |

### 生成示例

**输入属性**: ["Pearlescent shimmer", "Long-lasting/Durable", "High pigment concentration"]

❌ **错误示例**:
- "fabric paint pearlescent shimmer long-lasting high pigment" (直接堆砌，不自然)
- "I need fabric decorating product with pearlescent shimmer that is long lasting and durable with high pigment concentration for crafts" (直接复制，超过30词)
- 使用脚本批量生成的query (模板化、重复、语法错误)

✅ **正确示例** (26词):
- "I need fabric paint with a beautiful shimmer effect that lasts through many projects and has rich concentrated colors for my crafts"

### 批量生成的特征 (必须避免)

❌ 如果query出现以下特征，说明是批量生成，必须拒绝：
- 相同的结尾短语 (如 "...for various creative projects" 出现多次)
- "that + 形容词" 的语法错误 (缺少动词)
- "for ... for ..." 的重复结构
- 直接复制原始属性文本
- 包含prompt中的指令文本
- 属性重复使用

## 🔍 质量检查命令

处理每个query后，使用以下命令验证：

```bash
# 检查单词数
query="I need fabric paint with beautiful shimmer effect"
echo "Word count: $(echo $query | wc -w)"

# 检查是否有重复的结尾
tail -20 clean_queries.csv | grep -o "for [a-z]* [a-z]* projects$" | sort | uniq -c
```

## 完成标准

- [ ] 所有102个query都已生成
- [ ] 每个query都是逐个手动生成，不是批量
- [ ] 所有query的单词数都在25-30之间
- [ ] 所有query都包含全部3个属性
- [ ] 所有query都进行了语义转换
- [ ] 没有重复的结尾模式
- [ ] 没有语法错误
- [ ] CSV文件格式正确

## 🚨 违规检测

如果发现以下情况，说明违反了逐个生成原则：
- 使用了Python的for循环批量处理
- 使用了模板批量替换属性
- 多个query使用相同的结尾短语
- 出现了原始prompt文本
- 属性直接复制未转换

**必须重新开始，严格按照逐个处理的方式生成。**
