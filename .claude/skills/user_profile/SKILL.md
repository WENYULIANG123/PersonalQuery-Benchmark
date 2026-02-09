---
name: user-profile-manager
description: 为指定用户提取细粒度偏好并生成用户画像。分为两步：生成 Prompt 文件，然后批量调用 LLM 提取偏好。
---

# User Profile Manager Skill (用户画像管理)

此技能用于对特定用户的历史评论进行深度语义分析，构建详细的用户画像和偏好库。流程采用“以用户为中心”的设计，支持批量处理和高并发提取。

## 核心原则：AI-in-the-loop

核心的理解、推理和提取工作由 AI 亲自完成。AI 必须：
1.  **细粒度阅读**：逐条阅读评论，捕捉用户对产品材质、性能、设计、价格、包装等全方位的反馈。
2.  **强制性改进愿望**：对于任何 Negative（负面）评价，必须推断出用户的“改进愿望”（Improvement Wish）。
3.  **语义对齐**：将口语化表达映射到标准化的产品属性维度。

## 执行流程 (Two-Step Workflow)

### 第一步：生成 Prompt 文件

使用 `generate_user_prompts.py` 脚本生成包含用户所有评论 Prompt 的合并 JSON 文件。支持按用户 ID 或评论数量筛选用户。

```bash
# 1. 指定用户 ID 生成
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/generate_user_prompts.py --user-id [USER_ID]"

# 2. 按评论数批量筛选生成 (例如：找 10 个评论数在 100 左右的用户)
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/generate_user_prompts.py \
     --target-review-count 100 --review-tolerance 10 --max-users 10"
```

**输出：**
*   目录：`/home/wlia0047/ar57/wenyu/result/user_profile/user_prompts/`
*   文件：`prompt_[USER_ID].json` (包含该用户所有商品的 Prompt)

### 第二步：批量提取用户偏好

使用 `generate_user_profile.py` 脚本读取第一步生成的 Prompt 文件，调用 LLM 进行批量解析。脚本会自动扫描目录下的所有 Prompt 文件进行处理。

```bash
# 批量处理所有 Prompt 文件 (支持多线程并发)
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/generate_user_profile.py --max-workers 5"

# 指定特定用户处理 (可选)
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/generate_user_profile.py --user-id [USER_ID]"
```

**输出：**
*   目录：`/home/wlia0047/ar57/wenyu/result/user_profile/user_preferences/`
*   文件：`preferences_[USER_ID].json` (包含该用户所有商品的提取结果)

### 第三步：实体匹配与属性提取

此步骤将提取出的细粒度偏好与商品元数据、邻居信息相结合，生成用于搜索的 Top 3 核心属性。

#### 3.1 生成匹配 Prompt

首先，运行 `generate_match_prompts.py` 脚本生成用于属性匹配的推理上下文。该脚本会验证偏好与元数据的一致性。

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/generate_match_prompts.py \
     --input /home/wlia0047/ar57/wenyu/result/user_profile/user_preferences/preferences_[USER_ID].json \
     --meta-file /home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json \
     --output-dir /home/wlia0047/ar57/wenyu/result/user_profile/preference_match"
```

#### 3.2 批量匹配与推理

读取生成的 `match_prompts_[USER_ID].json`，调用 LLM 进行深度推理并筛选 Top 3 属性。

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/generate_match_results.py \
     --user-id [USER_ID]"
```

**核心逻辑：**
1. **Valid Seeds (自身验证)**：检查用户的 Positive/Neutral 偏好是否与商品的 Metadata (Title/Features) 语义匹配。只有匹配的才算 Valid Seed。
2. **Neighbor Insights (邻居增强)**：分析邻居的 `improvement_wish`，确认目标商品如何解决这些痛点。
3. **优先级筛选**：按以下优先级选出 Top 3 属性：
   - **Priority 1 (High)**: 已解决的邻居痛点 (强区分点)
   - **Priority 2 (Medium)**: 独特/具体的特征
   - **Priority 3 (Low)**: 核心规格 (基础属性)

**输出：**
*   目录：`/home/wlia0047/ar57/wenyu/result/user_profile/preference_match_results/`
*   文件：`match_[USER_ID].json`

**输出结构示例：**
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

**参考实现：**
详细的实体匹配逻辑和推理过程请参考 [`preference_match` Skill](file:///home/wlia0047/ar57/wenyu/.claude/skills/preference_match/SKILL.md)。


最终的 `preferences_[USER_ID].json` 包含：
- 用户 ID
- 处理时间戳
- 商品总数
- `results` 列表：每个元素包含单个商品的 ASIN、标题及提取出的偏好维度（正向/负向/改进建议）。

## 质量检查清单

- [ ] **Prompt 完整性**：生成的 Prompt文件应包含用户的所有评论。
- [ ] **合并输出**：最终偏好结果应合并为一个 JSON 文件，而非散落在多个文件中。
- [ ] **负面建议覆盖**：所有 `sentiment: negative` 的项必须包含非空的 `improvement_wish`。
- [ ] **多线程效率**：在大批量处理时，应观察日志确认并发执行正常。

---
*注：本 Skill 依赖 `/home/wlia0047/ar57/wenyu/stark/code/claude_code_cli.py` (或等效 LLM Client) 进行模型调用。*
