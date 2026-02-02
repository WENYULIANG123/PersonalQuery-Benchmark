---
name: user-noise-simulator
description: 读取 style_analysis_AG7EF0SVBQOUX.json 文件，通过脚本分析 EPI，引导 Agent 手动逐个对 Clean Query 注入噪声。必须严格符合用户 DNA。⚠️ 严禁批量脚本生成，必须一个一个手动生成。
allowed-tools: run_command, view_file, ask_user_question
---

# User-Noise-Simulator

此技能用于让 Agent (Claude) 模拟特定用户的输入习惯，通过分析 `result/style_analysis_AG7EF0SVBQOUX.json` 里的用户拼写 DNA，**逐个手动**向 Clean Query 中注入噪声。

## 🚨 严禁行为

❌ **严禁使用任何批量处理脚本进行噪声注入**
❌ **严禁使用 Python 循环批量生成 noisy queries**
❌ **严禁一次性读取所有 EPI 分析后批量处理**
❌ **严禁使用模板批量注入噪声**

✅ **必须逐个读取、逐个分析、逐个注入、逐个验证**

## ⚠️ 核心原则

1. **精准注入** - 每个查询仅注入 **1 处**拼写错误
2. **符合用户 DNA** - 必须严格遵循 EPI 推荐的目标词和风格
3. **保持原样** - 除目标词外，查询的其他部分必须与 Clean Query 完全一致
4. **逐个手动注入** - 每个 query 必须单独处理，独立思考，严禁批量

## 执行流程

### 阶段 1：准备阶段

运行脚本批量计算所有查询的 EPI 值，为手动注入提供参考。

```bash
# 确保输出目录存在
mkdir -p /home/wlia0047/ar57/wenyu/result/noisy_query

# 批量计算 EPI (允许使用脚本)
python3 /home/wlia0047/ar57/wenyu/.claude/skills/user-noise-simulator/calculate_epi.py \
    --input /home/wlia0047/ar57/wenyu/result/clean_query/clean_queries.csv \
    --dna /home/wlia0047/ar57/wenyu/result/style_analysis_AG7EF0SVBQOUX.json \
    --output /home/wlia0047/ar57/wenyu/result/noisy_query/epi_analysis_results.json
```

### 阶段 2：逐个手动注入噪声 (Noise Injection)

🔴 **必须按照以下步骤，一个一个处理，不得跳过或合并：**

#### 步骤 1：准备输出文件

```bash
# 创建CSV文件，只写入header
echo "id,query,answer_ids_source" > /home/wlia0047/ar57/wenyu/result/noisy_query/kg_queries_noisy.csv
```

#### 步骤 2：逐个处理每个 Query

**对于每个 query，必须独立执行以下步骤：**

1. **获取单条 EPI 分析** - 从 `epi_analysis_results.json` 读取指定索引的结果
2. **分析注入策略** - 确定目标词和推荐风格（如 `Deletion`, `Suffix`, `Substitution` 等）
3. **手动注入噪声** - 基于理解，手动修改目标词，**仅修改一处**
4. **验证质量** - 检查是否符合用户 DNA，且只有一处错误
5. **追加到 CSV** - 将单个 noisy query 追加到 CSV 文件
6. **确认完成** - 确认后继续下一个

**具体操作示例：**

```bash
# 读取第0个EPI分析
python3 -c "import json; d=json.load(open('/home/wlia0047/ar57/wenyu/result/noisy_query/epi_analysis_results.json')); print(json.dumps(d[0], indent=2))"
```

然后根据推荐手动注入噪声并追加：

```bash
# 追加到CSV (示例：AzureGreen -> Azurereen)
echo "0,\"Looking for Azurereen Assorted Stick Incense\",\"['B005309AFI']\"" >> /home/wlia0047/ar57/wenyu/result/noisy_query/kg_queries_noisy.csv
```

**重复以上步骤，处理所有查询。**

#### 步骤 3：逐个验证检查清单

每个 noisy query 生成后，必须验证：

- [ ] **仅包含 1 处错误** - 没有多处修改
- [ ] **符合推荐风格** - 严格遵循 EPI 分析
- [ ] **其余部分一致** - 没有任何其他变动
- [ ] **格式正确** - CSV 引用和逗号处理正确

## 📋 逐个处理检查清单

处理每个 query 时，按顺序确认：

- [ ] 1. 读取了单条 EPI 分析 (不是批量)
- [ ] 2. 确定了目标单词
- [ ] 3. 确定了注入风格
- [ ] 4. 手动执行了噪声注入
- [ ] 5. 验证了错误数量 (仅限 1 处)
- [ ] 6. 验证了整体语义保留
- [ ] 7. 追加到 CSV 文件
- [ ] 8. 继续下一个 query

## 风格示例 (Injection Guide)

| 风格 | 原始单词 | ❌ 错误注入 | ✅ 正确注入 |
|---------|-----------|-----------|-----------|
| Deletion | "AzureGreen" | "Az Green" | "Azurereen" (删除中间字母) |
| Suffix | "searching" | "searcher" | "searchin" (修改后缀) |
| Substitution | "purchase" | "buy" | "purchace" (替换字母) |
| Extra Space | "Instructions" | "Instruction s" | "Instr uctions" (中间空格) |

## 🔍 质量检查命令

处理每个 query 后，或分批检查时使用：

```bash
# 检查是否只有一处修改 (手动对比或简易脚本)
# 检查 CSV 行数
wc -l /home/wlia0047/ar57/wenyu/result/noisy_query/kg_queries_noisy.csv
```

## 完成标准

- [ ] 所有查询都已处理并注入噪声
- [ ] 每个 noisy query 都是逐个手动生成，不是批量
- [ ] 每个查询严格只有 1 处错误
- [ ] 噪声风格符合用户 DNA
- [ ] CSV 文件格式正确且包含所有字段

## 🚨 违规检测

如果发现以下情况，说明违反了逐个生成原则：
- 使用了 Python 的 for 循环进行 `replace` 或拼接操作
- 自动化生成了整个 CSV 文件
- 注入风格显得机械化或不符合所给出的分析结果
- 原始 Clean Query 文本在大规模处理中被意外篡改

**必须重新开始，严格按照逐个手动注入的方式执行。**
