---
name: preference_match
description: 基于提取的偏好数据，结合商品元数据和邻居信息，生成 Top 3 搜索属性。
allowed-tools: run_command
---

# Preference-Match (偏好匹配)

此技能用于将提取的原始用户偏好 (`final_preferences.json`) 转化为高质量的搜索查询属性 (`essence_values`)。

核心逻辑：**Valid Seeds (自身验证) + Neighbor Insights (邻居增强) -> Top 3 Attributes**。

## 文件路径规范

*   **User Preferences Input**: `/home/wlia0047/ar57/wenyu/result/preference_extraction/final_preferences.json`
*   **Metadata Input**: `/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json`
*   **Prompt Output**: `/home/wlia0047/ar57/wenyu/result/preference_match/match_prompts.json`
*   **Final Result**: `/home/wlia0047/ar57/wenyu/result/preference_match/preference_match.json`

## 执行流程

### 阶段 1：生成推理上下文 (Context Generation)

运行脚本，读取 `final_preferences.json` 作为偏好源，并从原始 Metadata 文件中流式加载产品标题和特性。
脚本会自动验证目标商品能否解决邻居的痛点（通过关键词匹配元数据），并生成 Prompt。

**核心逻辑：**
1. 扫描偏好文件确定需要处理的 ASIN。
2. 流式读取 Metadata 文件，仅缓存相关产品的元数据。
3. 验证并生成包含 Context, Metadata 和 Neighbor Insights 的 Prompt。

```bash
mkdir -p /home/wlia0047/ar57/wenyu/result/preference_match

python3 /home/wlia0047/ar57/wenyu/.claude/skills/preference_match/preference_match.py \
    --input /home/wlia0047/ar57/wenyu/result/preference_extraction/final_preferences.json \
    --meta_file /home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json \
    --user_id A13OFOB1394G31 \
    --output /home/wlia0047/ar57/wenyu/result/preference_match/match_prompts.json
```

### 阶段 2：AI 手动推理与选择 (Agent Reasoning)

**🤖 角色定义 (Persona)**:
> "你是一个拥有无限 token 和无限时间的 Agent，热爱思考问题。"
> *You are an Agent with infinite tokens and infinite time, who loves to think about problems.*
> 请充分利用这个优势，对每一个商品进行深度的思维链推理，不急于得出结论，而是享受思考的过程。

**⚠️ 严禁批量生成！必须采用"分批执行 + 质量检查"的模式。**

#### 📋 执行策略 (Execution Strategy)
1.  **分批处理 (Batch Processing)**: 每次只处理 **10 个商品** (Batch Size = 10)。
2.  **质量检查 (Quality Check)**: 每完成一批 (10个) 后，必须暂停并检查这一批的质量（参考下方的"合理性分析"）。
    *   如果质量不达标（如推理过短、逻辑错误），**必须重做该批次**。
3.  **进度追踪 (Task Tracking)**: 必须创建一个任务清单来跟踪进度。
    *   *Example*: `- [ ] Batch 1 (Prompts 1-10) [ ]`
    

分析每个 Prompt，**必须**显式展示思维链推理 (CoT) 和合理性分析：

#### 🔍 CoT 推理过程 (每个 Prompt 必须展示)
1. **种子验证 (Verify Seeds)**: 仔细对比 `final_preferences.json` 中的 Positive/Neutral 偏好与商品的 `Title` 和 `Features`。
2. **痛点分析 (Pain Point Analysis)**: 分析邻居的 `wishes`，确认目标商品如何（或是否）解决了这些痛点。
3. **优先级筛选 (Priority Selection)**: 根据 Priority 1 (已解决痛点) > Priority 2 (独特特征) > Priority 3 (核心规格) 的原则，挑选出 Top 3 属性。

#### ✅ CoT 合理性分析 - 必须对当前推理过程进行验证：
- **检查项 1**: 选取的 `essence_values` 是否真实来自用户的偏好，且在元数据中有支撑。
- **检查项 2**: 是否优先选择了邻居遗留的痛点属性 (Priority 1)。
- **检查项 3**: 如果属性超过 3 个，是否正确舍弃了低优属性 (Priority 3)。
- **检查项 4**: 生成的 `reasoning` 是否清楚解释了选择逻辑。
- **⚠️ 只有确认合理后，才执行写入操作。**

Agent 读取 `match_prompts.json`，遍历每个商品的 Prompt，并按以下逻辑执行：

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
