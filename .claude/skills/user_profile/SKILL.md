---
name: user-profile-manager
description: 提取细粒度用户偏好并生成"接地气"的用户画像与个性化搜索查询。
---

# User Profile Manager (用户画像与个性化查询管理)

此技能实现了一套完整的 **"Grounded" (落地/实操型)** 个性化 pipeline，通过 7 个有序阶段将原始用户评论转化为高度差异化的画像及搜索查询，并进行多维度评估。

## 核心理念：Grounded Strategy
1. **去空洞化**：禁止使用 "High quality", "Dedicated enthusiast" 等泛泛而谈的模板词。
2. **场景实操**：强调用户关注的具体技术规格（如 "Cuttlebug compatibility"）和使用场景。
3. **严格 Holdout**：Persona 生成仅基于训练集，评估则针对完全隔离的 Holdout 集，验证泛化能力。

---

## 执行环节 (Step-by-Step Pipeline)

### Stage 1: 偏好提取与属性验证 (Matching)
从原始评论中提取实体，并与商品元数据（Metadata）进行匹配验证。
- **01_matching/01_extract_attributes.py**: 提取用户偏好实体及上下文。
- **01_matching/02_verify_attributes.py**: 基于元数据验证属性，过滤虚假或泛化的匹配。

### Stage 2: 数据处理与集合划分 (Processing)
筛选高质量用户并进行类目感知的训练集/测试集划分。
- **02_processing/03_select_users.py**: 筛选满足评论数量且类目解析完整的"完美"用户。
- **02_processing/04_split_train_holdout.py**: 智能划分 10 个 Holdout 商品作为测试集，确保其类目在训练集中有覆盖。

### Stage 3: 用户画像生成 (Persona)
基于训练集数据合成差异化的画像。
- **03_persona/05_generate_persona_prompts.py**: 提取训练集属性，计算独特性分数 (Uniqueness Score) 并生成画像生成指令。
- **03_persona/06_execute_persona_generation.py**: 调用 LLM 批量生成 150-200 字的接地气画像。

### Stage 4: 写作风格分析 (Writing Analysis)
分析用户的拼写和语法错误习惯，用于生成更真实的查询。
- **04_writing_analysis/08_generate_writing_prompts.py**: 为每个用户的评论生成错误分析提示词。
- **04_writing_analysis/09_execute_writing_analysis.py**: 调用 LLM 分析拼写错误 (10种) 和语法错误 (7种)，输出统计报告。

### Stage 5: 个性化查询生成 (Query)
针对 Holdout 商品生成对比查询。
- **05_query/07_generate_dual_queries.py**: 生成 **Public (大众版)** 和 **Personalized (基于画像版)** 两种查询词。

### Stage 6: 噪声查询生成 (Noisy Query)
基于写作风格分析结果，为个性化查询注入用户特有的拼写/语法错误。
- **06_noisy_query/10_generate_noisy_queries.py**: 读取 Stage 4 的错误统计，按权重对查询进行单错误注入。

### Stage 7: 多维度评估 (Evaluation)
验证个性化提升效果。
- **07_evaluation/evaluate_with_unique_persona_v2_sbs.py**: LLM 盲测对比 (SBS)。
- **07_evaluation/evaluate_with_unique_persona.py**: 1-10 分关联度评估。
- **07_evaluation/evaluate_semantic_similarity.py**: BERT 语义向量偏移分析。
- **07_evaluation/evaluate_persona_diversity.py**: 跨用户画像差异化（多样性）核查。

---

## 常用运行命令 (SLURM 环境)

所有脚本必须使用 `sbatch_wrapper.py` 提交运行：

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/[STAGE]/[SCRIPT].py [ARGS]"
```

## 目录结构
```text
user_profile/
├── 01_matching/           # Stage 1: 偏好提取与元数据验证
├── 02_processing/         # Stage 2: 用户筛选与训练/Holdout集划分
├── 03_persona/            # Stage 3: 画像提示词生成与执行
├── 04_writing_analysis/   # Stage 4: 写作风格分析
├── 05_query/              # Stage 5: 双重查询词合成
├── 06_noisy_query/        # Stage 6: 噪声查询生成
└── 07_evaluation/         # Stage 7: 多维度质量评估套件
```
