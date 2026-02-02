---
name: preference_match
description: 基于提取的偏好数据，结合商品元数据和邻居信息，生成 Top 3 搜索属性。
allowed-tools: run_command
---

# Preference-Match (偏好匹配)

此技能用于将提取的原始用户偏好 (`final_preferences.json`) 转化为高质量的搜索查询属性 (`essence_values`)。

核心逻辑：**Valid Seeds (自身验证) + Neighbor Insights (邻居增强) -> Top 3 Attributes**。

## 文件路径规范

*   **输入**: `/home/wlia0047/ar57/wenyu/result/preference_extraction/final_preferences.json` (上一步骤的产出)
*   **Prompt输出**: `/home/wlia0047/ar57/wenyu/result/preference_match/match_prompts.json`
*   **最终结果**: `/home/wlia0047/ar57/wenyu/result/preference_match/preference_match.json`

## 执行流程

### 阶段 1：生成推理上下文 (Context Generation)

运行脚本，根据 `final_preferences.json` 中的数据，自动寻找同类别的邻居商品，**验证目标商品能否解决邻居的痛点**，并生成一个包含"自身验证"和"邻居增强"任务的 Prompt。

**核心逻辑：**
1. 提取邻居商品的 negative 属性的 `improvement_wish`（正面版本）
2. 检查目标商品的用户偏好或元数据是否提到能解决这个痛点
3. 只保留目标商品能解决的 wishes，添加到候选列表

```bash
mkdir -p /home/wlia0047/ar57/wenyu/result/preference_match

python3 /home/wlia0047/ar57/wenyu/.claude/skills/preference_match/preference_match.py \
    --input /home/wlia0047/ar57/wenyu/result/preference_extraction/final_preferences.json \
    --output /home/wlia0047/ar57/wenyu/result/preference_match/match_prompts.json
```

### 阶段 2：AI 手动推理与选择 (Agent Reasoning)

Agent 读取 `match_prompts.json`，遍历每个商品的 Prompt，并执行以下步骤：

**重要：邻居的 wishes 已经验证匹配**，脚本已经在代码层面检查过目标商品能解决这些痛点。

1.  **Step A: Verify Seeds (自身验证)**
    *   检查用户的 Positive/Neutral 偏好是否与该商品的 **Metadata** (Title/Feature) 语义匹配。只有匹配的才算 Valid Seed。
2.  **Step B: Augment from Neighbors (痛点转化)**
    *   查看 Prompt 中提供的 **Neighbor Insights**。
    *   这些 wishes 已经验证：目标商品的用户偏好或元数据中包含相关关键词
    *   **直接添加 these wishes 作为候选属性**（如果它们提供差异化优势）
3.  **Step C: Final Selection (Top 3)**
    *   综合 Step A 和 Step B 的结果，按照以下优先级**选出最重要的 3 个属性**：
        1.  **Priority 1 (High): Resolved Pain Points**. 邻居抱怨过且被本产品解决的痛点（强区分点）。
        2.  **Priority 2 (Medium): Unique/Specific Features**. 具体且独特的特征（如"Waterproof", "Noise-cancelling"）。
        3.  **Priority 3 (Low): Core Specs**. 基础规格（如"Wireless", "X-Large"）。
    *   **筛选规则**: 当候选属性超过 3 个时，**优先保留 Priority 1 和 2**，最先舍弃 Priority 3 (通用/基础) 的属性。

### 阶段 3：输出结果

Agent 将推理结果整理为 JSON 格式保存，**必须包含 category 字段**（从输入文件的 `extraction.Product Category` 中获取）。

```json
[
  {
    "target_asin": "B00XXXXXX",
    "selected_attributes": ["Attr1", "Attr2", "Attr3"],
    "category": "Product Category Name",
    "reasoning": "Selected 'Reinforced Steel' (Priority 1 - Solves Neighbor Pain). Selected 'Waterproof' (Priority 2 - Unique). Dropped 'Nice Design' (Generic)."
  }
]
```

**重要说明：**
- `category` 字段必须从原始输入文件 (`final_preferences.json`) 的 `extraction.Product Category` 中提取
- 每个商品的输出都应包含其对应的类别信息
