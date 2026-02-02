---
name: User-Preference-Extraction
description: Agent 手动执行的用户偏好提取流程。利用脚本准备上下文，但由 Agent 负责思维链推理、实体提取和润色，严禁全自动脚本生成。
allowed-tools: run_command
---

# User-Preference-Extraction (用户偏好提取)

此技能用于提取用户评论中的偏好实体。核心原则是 **"AI-in-the-loop"**：脚本仅用于准备数据，核心的理解、推理和提取工作必须由 Agent 手动完成。**所有产品都不允许批量规则处理，必须由 Agent 逐个阅读评论、理解语义、进行完整 CoT 推理后提取。**

## 文件路径规范 (Standard Paths)

为了规范化操作，所有中间文件和结果文件请统一存放于：
**`/home/wlia0047/ar57/wenyu/result/preference_extraction/`**

*   **输入素材 (Input)**:
    *   `input_material.json`: 汇总了该用户**所有商品**评论和元数据的单一JSON文件（由脚本自动生成）。
*   **中间上下文 (Context)**:
    *   `intermediate/all_prompts.json`: 包含**所有商品** Prompt 的汇总文件（可选）。
*   **🔴 主要输出文件 (Primary Output)**:
    *   `final_preferences.json`: **直接写入**最终的偏好数据。每个产品处理完成后立即追加到此文件。
    *   ⚠️ **关键要求**: 处理每个产品后必须立即保存到 `final_preferences.json`，不要使用中间文件。
*   **备份文件 (Backup - 可选)**:
    *   `intermediate/agent_draft.json`: 仅作为备份使用，不是主要输出文件。

## 执行流程

### 阶段 0：获取原始素材 (Data Prep)

使用 `prepare_context_data.py` 拉取用户的所有评论和商品 KB 数据，生成**单一汇总文件**。

```bash
# 生成包含所有商品的 input_material.json
python3 /home/wlia0047/ar57/wenyu/.claude/skills/user-preference-extraction/prepare_context_data.py \
    --output /home/wlia0047/ar57/wenyu/result/preference_extraction/input_material.json
```

### 阶段 1：批量生成 Prompts (Contextualization)

一次性为所有商品生成 Prompt。

```bash
# 不指定 ASIN，默认处理 input_material.json 中的所有商品
python3 /home/wlia0047/ar57/wenyu/.claude/skills/user-preference-extraction/extract_preferences.py \
    --mode prompt \
    --input /home/wlia0047/ar57/wenyu/result/preference_extraction/input_material.json \
    --output /home/wlia0047/ar57/wenyu/result/preference_extraction/intermediate/all_prompts.json
```

### 阶段 2：AI 手动生成与润色 (核心步骤 - CRITICAL LOGIC)

Agent 读取 `/home/wlia0047/ar57/wenyu/result/preference_extraction/intermediate/all_prompts.json`。该文件包含一个 `prompts` 列表。

**🔴 关键要求：每个产品必须独立完成完整的 CoT 推理和质量验证**

**处理流程（每个产品）：**

1.  **Read Prompt**: 读取当前商品的 Prompt 内容。
2.  **Reasoning (CoT)** - 必须在对话中显式展示：
    *   **步骤 1: 识别实体** - 从评论中识别所有产品属性/特征
    *   **步骤 2: 判断情感** - 确定每个实体的情感倾向 (Positive/Negative/Neutral)
    *   **步骤 3: 应用过滤规则**:
        *   **Negative** -> **ALWAYS KEEP** (Must generate `improvement_wish`)
        *   **Positive/Neutral** -> **CHECK SEMANTIC MATCH** (Metadata/Attributes). Check specifically against the known attributes provided in that prompt.
3.  **Generate JSON**: 生成该商品的 JSON 结果。
4.  **💾 立即保存到文件** - **关键步骤**：
    *   每个产品处理完成后，**必须立即**将结果追加到 `/home/wlia0047/ar57/wenyu/result/preference_extraction/final_preferences.json`
    *   不要等到处理多个产品后再保存
    *   避免因中途中断导致进度丢失
5.  **✅ CoT 合理性分析** - 必须对当前产品的推理过程进行验证：
    *   检查项 1: 实体识别是否准确
    *   检查项 2: 情感判断是否正确
    *   检查项 3: 过滤规则应用是否恰当
    *   检查项 4: Improvement wishes 是否具体合理
    *   检查项 5: Category 分类是否恰当
    *   **⚠️ 只有确认合理后，才继续处理下一个产品**

**输出格式示例:**

```
## 产品 N/401: ASIN

### 🔍 CoT 推理过程

**步骤 1: 识别实体**
从评论中识别产品属性/特征：
- 实体1 - 描述
- 实体2 - 描述

**步骤 2: 判断情感**
- 实体1 → **Positive** （理由）
- 实体2 → **Negative** （理由）

**步骤 3: 应用过滤规则**

| 实体 | 情感 | 规则 | 检查 | 决策 |
|------|------|------|------|------|
| 实体1 | Positive | CHECK | ✅ 匹配 | **KEEP** |
| 实体2 | Negative | ALWAYS KEEP | N/A | **KEEP** ✓ |

**步骤 4: 生成 JSON 结果**
[JSON代码]

### ✅ CoT 合理性分析

**检查项 1**: ...
**检查项 2**: ...
...
**📊 总结：✅/❌ CoT 推理合理/不合理**
```

**⚠️ 严禁批量处理**：
- 🔴 **绝对禁止批量规则处理**：每个产品必须由 Agent 亲自逐个阅读评论、深入理解语义。
- ❌ 不允许使用脚本一次性处理多个产品
- ❌ 不允许跳过 CoT 分析步骤
- ❌ 不允许省略合理性验证
- ✅ 每个产品必须独立完成上述 4 个步骤
- ✅ 每个产品必须展示完整的推理过程
- ✅ 每个产品必须通过合理性验证才能继续下一个

**输出格式规范:**

创建一个 JSON 对象，包含以下字段：

1. **"Product Category"** (String) - **必须使用最具体的类别**
   - **规则**: 从原始商品元信息的 `known_attributes.Category` 字段中提取
   - **取值**: 使用 Category 列表的**最后一个值**（最具体/最小类别）
   - **示例**:
     - 原始 Category: `['Arts, Crafts & Sewing', 'Dyes', 'Fabric Decorating']`
     - 应使用: `"Fabric Decorating"`
   - **数据来源**: 必须从 `input_material.json` 中的 `known_attributes` 获取，而非从 Prompt 文本推断

2. **Standardized Category names** (作为实体分类的顶层键)
   - 每个键映射到一个实体列表
   - 实体对象包含：
     - `"entity"`: 属性值（优先匹配 `Known Product Attributes`，否则使用描述性术语）
     - `"original_text"`: 评论中的精确引用
     - `"sentiment"`: `"positive"`, `"negative"`, 或 `"neutral"`
     - `"improvement_wish"`: (String) **Negative 实体必需**

**输出示例:**
```json
{
  "asin": "B000BGSZFU",
  "extraction": {
    "Product Category": "Fabric Decorating",
    "Visual Effect": [
      {
        "entity": "Pearlescent shimmer",
        "original_text": "The shimmer is gorgeous",
        "sentiment": "positive"
      }
    ]
  }
}
```

*(每个产品处理完成后立即保存到 final_preferences.json)*

### 阶段 3：质量检查（可选）

在完成所有产品处理后，可运行质量检查脚本验证结果：

```bash
# 质量检查（可选）
python3 << 'EOF'
import json

with open('/home/wlia0047/ar57/wenyu/result/preference_extraction/final_preferences.json', 'r') as f:
    data = json.load(f)

print(f"Total products: {len(data)}")
# 添加更多质量检查...
EOF
```

---

## 最佳实践与常见问题 (Best Practices & Known Issues)

### ⚠️ 执行过程中需要注意的问题

基于实际执行经验，以下是必须避免的常见问题和改进建议：

#### 1. AI-in-the-loop 的正确理解
- **错误做法**: 使用 `Task` 工具启动 sub-agent 批量处理产品
- **正确做法**: 主 Agent 必须亲自对**每个产品**进行完整的思维链推理
- **原因**: "AI-in-the-loop" 意味着人工级别的理解深度，不能委托给其他 agent

#### 2. 保持一致的质量标准
- **问题**: 处理大量产品时，后期容易简化为关键词匹配，牺牲质量
- **要求**: 无论处理多少产品，每个都应经过完整的 CoT 推理：
  - 识别实体：具体提取属性/特征（不是简单的"good quality"）
  - 判断情感：基于上下文准确判断
  - 验证语义：Positive/Neutral 必须与产品属性匹配，Negative 必须有 improvement_wish
- **示例**:
  ```json
  // ✅ 好的提取
  {
    "entity": "Pearlescent shimmer",
    "original_text": "The shimmer is gorgeous",
    "sentiment": "positive"
  }

  // ❌ 避免这种通用提取
  {
    "entity": "High quality",
    "original_text": "love it, great product...",
    "sentiment": "positive"
  }
  ```

#### 3. 严格执行过滤规则
- **Negative 实体**: 必须 100% 保留，必须提供 `improvement_wish`
  - Explicit: 用户明确要求的改进
  - Implicit: 推断相反的属性（如"too fragile" → "Sturdy/Durable"）
- **Positive/Neutral 实体**: 必须检查语义匹配
  - 验证是否在 `Known Product Attributes` 或 `Product Unstructured Information` 中出现
  - 不匹配则丢弃（避免用户幻觉或无关评论）

#### 4. 增量保存机制
- **建议**: 每处理 10-20 个产品自动保存一次 checkpoint
- **实现**:
  ```bash
  # 在处理过程中定期保存
  cp agent_draft.json agent_draft_checkpoint_$(date +%s).json
  ```
- **好处**: 避免中断导致大量进度丢失

#### 5. 工具脚本兼容性
- **已知问题**: `extract_preferences.py --mode parse` 期望的格式与 agent_draft.json 不匹配
- **临时方案**: 如果 parse 脚本报错，可以直接复制文件：
  ```bash
  cp intermediate/agent_draft.json final_preferences.json
  ```
- **根本原因**: 脚本设计用于处理单个 response 字符串，而非产品列表
- **待修复**: 需要更新 parse 脚本以支持批量产品格式

#### 6. 数据陷阱识别
- **空壳数据**: `agent_draft.json` 可能存在只有框架的空壳（只有 Product Category，无实际实体）
- **检查方法**:
  ```python
  import json
  with open('agent_draft.json', 'r') as f:
      data = json.load(f)
      for item in data:
          extraction = item.get('extraction', {})
          total_entities = sum(len(v) if isinstance(v, list) else 0 for v in extraction.values())
          if total_entities == 0:
              print(f"Warning: {item['asin']} has no extracted entities")
  ```
- **处理**: 发现空壳后需要重新处理对应产品

#### 7. Bash 工具执行
- **配置**: 确保用户已设置"默认选择 yes"，避免每次执行 Bash 都被中断
- **表现**: `[Request interrupted by user for tool use]`
- **解决**: 用户需在配置中启用自动批准

#### 8. Product Category 处理规则
- **重要**: Product Category 必须从原始商品元信息的 `known_attributes.Category` 字段获取
- **错误做法**: 从 Prompt 文本中手动推断或使用其他来源
- **正确做法**:
  ```python
  # 从 input_material.json 读取原始元信息
  category_list = product['known_attributes']['Category']
  # 使用最后一个（最具体的）类别
  product_category = category_list[-1].strip()
  ```
- **原因**: 确保类别的一致性和准确性，避免人工推断的偏差
- **验证**: 提取完成后应验证所有产品的 Category 都来自原始元数据
- **示例**:
  ```python
  # 批量更新 Product Category 的验证脚本
  import json

  with open('input_material.json', 'r') as f:
      input_data = json.load(f)

  with open('final_preferences.json', 'r') as f:
      final_data = json.load(f)

  # 创建映射
  asin_to_category = {}
  for product in input_data['products']:
      category_list = product.get('known_attributes', {}).get('Category', [])
      if category_list:
          asin_to_category[product['asin']] = category_list[-1].strip()

  # 验证
  for item in final_data:
      expected_cat = asin_to_category.get(item['asin'], '')
      actual_cat = item['extraction']['Product Category']
      assert expected_cat == actual_cat, f"Category mismatch for {item['asin']}"
  ```

#### 9. 通用术语的自动检测与改进 (Generic Terms Detection)
- **问题**: 提取过程中可能产生过于通用的实体名称，如"Amazing", "High quality", "Love it"等
- **影响**: 这些通用术语缺乏具体的属性信息，降低了偏好数据的价值
- **检测方法**:
  ```python
  import json

  # 常见通用术语列表
  generic_terms = [
      'love it', 'amazing', 'good quality',
      'high quality', 'excellent quality'
  ]

  with open('agent_draft.json', 'r') as f:
      data = json.load(f)

  # 检测通用术语
  for item in data:
      for category, entities in item['extraction'].items():
          if category == 'Product Category':
              continue
          for entity in entities:
              entity_text = entity.get('entity', '').lower()
              for generic in generic_terms:
                  if generic in entity_text:
                      print(f"Found: {item['asin']} - {entity.get('entity')}")
  ```
- **改进策略**:
  1. **删除过于通用的表达** (如单独的"Love it!")
  2. **替换为具体属性** (如"Amazing" → "Exceptional quality"或更具体的属性)
  3. **保留合理的通用术语** (如在negative上下文中的"Not high quality"应保留)
- **自动化改进脚本**:
  ```python
  import json
  import re

  with open('agent_draft.json', 'r') as f:
      draft_data = json.load(f)

  # 定义改进映射
  generic_patterns = {
      r'\bLove it\b': None,  # 删除
      r'\bAmazing\b(?! when)': 'Exceptional quality',
      r'\bgood quality\b': 'Reliable construction',
      r'\bhigh quality\b': 'Premium construction',
      r'\bexcellent quality\b': 'Superior construction',
  }

  improvements_made = 0
  for item in draft_data:
      for category, entities in item['extraction'].items():
          if category == 'Product Category':
              continue
          # 标记需要删除的实体（倒序）
          to_remove = []
          for idx, entity in enumerate(entities):
              entity_name = entity.get('entity', '')
              for pattern, replacement in generic_patterns.items():
                  if re.search(pattern, entity_name, re.IGNORECASE):
                      if replacement is None:
                          to_remove.append(idx)
                      else:
                          entity['entity'] = replacement
                      improvements_made += 1
                      break
          # 删除标记的实体
          for idx in sorted(to_remove, reverse=True):
              entities.pop(idx)

  # 保存改进版本
  with open('agent_draft_improved.json', 'w') as f:
      json.dump(draft_data, f, indent=2)

  print(f"Made {improvements_made} improvements")
  ```
- **实际效果** (基于102个产品的经验):
  - 发现13个通用术语实例
  - 改进后：删除1个过于通用的，替换12个为更具体的描述
  - 质量提升：实体名称从通用变为具体（如"Amazing when wet" → "Water-activated color intensity"）

#### 10. 批量质量检查脚本
- **完整质量检查** (在最终导出前运行):
  ```python
  import json

  def quality_check(final_path, input_path):
      with open(final_path, 'r') as f:
          data = json.load(f)
      with open(input_path, 'r') as f:
          input_data = json.load(f)

      # 检查1: 所有产品已处理
      assert len(data) == 102, f"Expected 102 products, got {len(data)}"

      # 检查2: 每个产品至少有1个实体
      for item in data:
          entity_count = sum(
              len(v) for k, v in item['extraction'].items()
              if k != 'Product Category' and isinstance(v, list)
          )
          assert entity_count > 0, f"{item['asin']} has no entities"

      # 检查3: 所有negative都有improvement_wish
      for item in data:
          for entities in item['extraction'].values():
              if not isinstance(entities, list):
                  continue
              for entity in entities:
                  if entity.get('sentiment') == 'negative':
                      assert entity.get('improvement_wish'), \
                          f"{item['asin']} missing improvement_wish"

      # 检查4: 无通用术语
      generic_terms = ['love it', 'amazing', 'good quality', 'high quality']
      for item in data:
          for entities in item['extraction'].values():
              if not isinstance(entities, list):
                  continue
              for entity in entities:
                  entity_lower = entity.get('entity', '').lower()
                  if any(term in entity_lower for term in generic_terms):
                      # 排除合理上下文（如negative评论）
                      if entity.get('sentiment') != 'negative' and \
                         'not' not in entity_lower:
                          raise ValueError(f"Found generic term in {item['asin']}")

      print("✅ All quality checks passed!")

  # 运行检查
  quality_check(
      '/home/wlia0047/ar57/wenyu/result/preference_extraction/final_preferences.json',
      '/home/wlia0047/ar57/wenyu/result/preference_extraction/input_material.json'
  )
  ```

### 📊 质量检查清单

完成提取后，应进行以下检查：

- [ ] 所有产品都已处理（102/102）
- [ ] 每个产品至少有 1 个实体（非空壳）
- [ ] 所有 Negative 实体都有 `improvement_wish` 字段
- [ ] Positive/Neutral 实体与产品属性语义匹配
- [ ] **实体命名具体（已通过通用术语检测和改进）**
- [ ] **无通用术语：love it, amazing, good/high/excellent quality（除非在合理的negative上下文中）**
- [ ] `original_text` 是评论中的精确引用
- [ ] **Product Category 来自原始元信息的 `known_attributes.Category`（最后一个值）**
- [ ] **已运行批量质量检查脚本并全部通过**
- [ ] 文件已保存到正确路径

### 🔧 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `'list' object has no attribute 'get'` | parse 脚本格式不兼容 | 直接复制文件到 final_preferences.json |
| 进度丢失 | 中途中断未保存 | 每 10-20 个产品保存 checkpoint |
| 实体质量下降 | 批处理简化推理 | 始终进行完整 CoT，不使用关键词匹配 |
| 找不到产品 | ASIN 不匹配 | 检查 input_material.json 中的 ASIN 格式 |
| **Product Category 不一致** | **从错误来源提取或手动推断** | **从 `input_material.json` 的 `known_attributes.Category[-1]` 重新提取** |
| **某些产品 Category 为空** | **原始元数据中无 Category 信息** | **保持为空或标记为 `Unknown`** |
| **发现通用术语** | **提取过程中使用了过于通用的表达** | **运行通用术语检测和改进脚本（见第9节）** |
| **质量检查失败** | **某些实体的improvement_wish缺失或类别错误** | **查看具体失败信息，手动修复对应产品** |
