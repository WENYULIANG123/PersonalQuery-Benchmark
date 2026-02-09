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
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/extraction/generate_user_prompts.py --user-id [USER_ID]"

# 2. 按评论数批量筛选生成 (例如：找 10 个评论数在 100 左右的用户)
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/extraction/generate_user_prompts.py \
     --target-review-count 100 --review-tolerance 10 --max-users 10"
```

**输出：**
*   目录：`/home/wlia0047/ar57/wenyu/result/user_profile/user_prompts/`
*   文件：`prompt_[USER_ID].json` (包含该用户所有商品的 Prompt)

### 第二步：批量提取用户偏好

使用 `extraction/generate_user_profile.py` 脚本读取第一步生成的 Prompt 文件，调用 LLM 进行批量解析。脚本会自动扫描目录下的所有 Prompt 文件进行处理。

```bash
# 批量处理所有 Prompt 文件 (支持多线程并发)
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/extraction/generate_user_profile.py --max-workers 5"

# 指定特定用户处理 (可选)
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/extraction/generate_user_profile.py --user-id [USER_ID]"
```

**输出：**
*   目录：`/home/wlia0047/ar57/wenyu/result/user_profile/user_preferences/`
*   文件：`preferences_[USER_ID].json` (包含该用户所有商品的提取结果)

### 第三步：实体匹配与属性提取

此步骤将提取出的细粒度偏好与商品元数据、邻居信息相结合，生成用于搜索的 Top 3 核心属性。

#### 3.1 生成匹配 Prompt

首先，运行 `matching/generate_match_prompts.py` 脚本生成用于属性匹配的推理上下文。该脚本会验证偏好与元数据的一致性。

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/matching/generate_match_prompts.py \
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
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/matching/generate_match_results.py \
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

### 第四步：商品搜索查询生成 (Search Query Generation)

将筛选出的 Top 3 属性转化为符合真实购物者语气的自然语言搜索查询。

#### 4.1 生成查询 Prompt

使用 `query/generate_query_prompts.py` 脚本，将 Step 3 的匹配结果转换为带有语义转换规则的推理 Prompt。

```bash
python3 /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/query/generate_query_prompts.py \
    --input /home/wlia0047/ar57/wenyu/result/user_profile/preference_match_results/match_[USER_ID].json \
    --output-dir /home/wlia0047/ar57/wenyu/result/user_profile/query_prompts
```

#### 4.2 批量查询生成与推理

调用 LLM 批量生成 25-30 个单词的高质量查询。

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/query/generate_query_results.py \
     --user-id [USER_ID] --max-workers 5"
```

### 第五步：用户画像合成 (User Persona Generation)

基于所有已匹配商品的属性优先级，合成一个约 200 词的全局用户画像描述。

#### 5.1 生成画像 Prompt

使用 `persona/generate_persona_prompts.py` 脚本，聚合用户的所有偏好证据并生成综合推理 Prompt。

```bash
python3 /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/persona/generate_persona_prompts.py \
    --input /home/wlia0047/ar57/wenyu/result/user_profile/preference_match_results/match_[USER_ID].json \
    --output-dir /home/wlia0047/ar57/wenyu/result/user_profile/persona_prompts
```

#### 5.2 批量画像生成与推理

调用 LLM 进行画像合成。

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/persona/generate_persona_results.py \
     --max-workers 5"
```

**核心要求：**
1. **全局聚合**：画像应反映用户在多个商品上的共同审美或功能追求。
2. **长度**：目标长度约为 200 词。
3. **结构**：包含兴趣方向、质量标准、实用性偏好及购物意图。

### 第六步：个性化打分 (Personalization Scoring)

评价生成的查询 (Query) 与用户画像 (Persona) 之间的契合程度。

#### 6.1 生成打分 Prompt

使用 `scoring/generate_scoring_prompts.py` 脚本，将 Query 与 Persona 配对生成打分 Prompt。

```bash
python3 /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/scoring/generate_scoring_prompts.py \
    --output-dir /home/wlia0047/ar57/wenyu/result/user_profile/scoring_prompts
```

#### 6.2 批量打分与推理

调用 LLM 进行打分（1-10 分）并提供理由。

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/scoring/generate_scoring_results.py \
     --max-workers 5"
```

**核心要求：**
1. **打分维度**：基于画像中的兴趣倾向、质量标准和实用性偏好对 Query 进行评分。
2. **量化结果**：产出 1-10 的个性化分数值。
3. **解释性**：必须包含简短的打分理由 (Justification)。

## 输出位置 (Output Locations)

- **第一步 Prompt**: `/home/wlia0047/ar57/wenyu/result/user_profile/user_prompts/`
- **第二步 偏好**: `/home/wlia0047/ar57/wenyu/result/user_profile/user_preferences/`
- **第三步 匹配推理**: `/home/wlia0047/ar57/wenyu/result/user_profile/preference_match/`
- **第三步 最终属性**: `/home/wlia0047/ar57/wenyu/result/user_profile/preference_match_results/`
- **第四步 查询 Prompt**: `/home/wlia0047/ar57/wenyu/result/user_profile/query_prompts/`
- **第四步 最终查询**: `/home/wlia0047/ar57/wenyu/result/user_profile/query_results/`
- **第五步 最终画像**: `/home/wlia0047/ar57/wenyu/result/user_profile/persona_results/`
- **第六步 打分 Prompt**: `/home/wlia0047/ar57/wenyu/result/user_profile/scoring_prompts/`
- **第六步 最终得分**: `/home/wlia0047/ar57/wenyu/result/user_profile/scoring_results/`

## 质量检查清单

- [ ] **语义转换**：检查是否出现了直接复制属性文字的情况。
- [ ] **单词计数**：验证生成的 Query 是否在 25-30 个单词之间，画像是否接近 200 词。
- [ ] **唯一性**：确保生成的查询句式多样，画像具有一致性。
- [ ] **多线程效率**：在大批量处理时，应观察日志确认并发执行正常。

---
*注：本 Skill 依赖 `/home/wlia0047/ar57/wenyu/.claude/skills/llm_client.py` 进行模型调用。*
