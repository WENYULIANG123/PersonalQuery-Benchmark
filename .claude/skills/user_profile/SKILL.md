---
name: user_profile
description: 提取细粒度用户偏好并生成"接地气"的用户画像与个性化搜索查询。
---

# User Profile Manager (用户画像与个性化查询管理)

此技能实现了一套完整的 **"Grounded" (落地/实操型)** 个性化 pipeline，通过 **9 个有序阶段**将原始用户评论转化为高度差异化的画像及搜索查询，并进行多维度评估。

## 快速概览

| 阶段 | 名称 | 输入 | 输出 |
|------|------|------|------|
| 0 | 数据准备 | 原始评论数据 | 用户偏好 |
| 1 | 属性验证 | 偏好 + 元数据 | 验证后的属性 |
| 2 | 集合划分 | 用户数据 | Train/Holdout 划分 |
| 3 | 画像生成 | 训练集属性 | 用户画像 |
| 4 | 写作分析 | 用户评论 | 错误统计 |
| 5 | 特征提取 | 用户评论 | 风格特征向量 |
| 6 | 双查询生成 | Holdout + 属性 | Public/Personalized 查询 |
| 8 | 迭代优化 | 查询 + 特征 | 风格对齐后查询 |
| 9 | 噪声注入 | 查询 + 错误 | 含噪声查询 |
| 10 | 多维评估 | 所有输出 | 评估报告 |

**最新成果** (2026-02-28):
- 使用 **GLM-5** 模型进行迭代式优化
- 平均风格改进 **12.08%** (比 GLM-4.5-Air 提升 1.07%)
- 无改进样本仅 **7.2%** (比 GLM-4.5-Air 减少 4.6%)

## 核心理念：Grounded Strategy
1. **去空洞化**：禁止使用 "High quality", "Dedicated enthusiast" 等泛泛而谈的模板词。
2. **场景实操**：强调用户关注的具体技术规格（如 "Cuttlebug compatibility"）和使用场景。
3. **严格 Holdout**：Persona 生成仅基于训练集，评估则针对完全隔离的 Holdout 集，验证泛化能力。

---

## 执行环节 (Step-by-Step Pipeline)

---

### Stage 0: 数据准备 (Data Preparation)

**阶段描述**：提取用户偏好并准备基础数据。
- **00_data_preparation/00_extract_user_reviews.py**: 从原始数据中提取用户评论
- **00_data_preparation/00_batch_extract_preferences.py**: 批量提取用户偏好
- **00_data_preparation/00_select_users_with_long_sentences.py**: 筛选包含长句的高质量用户

---

### Stage 1: 偏好提取与属性验证 (Matching)

**阶段描述**：从原始评论中提取实体，并与商品元数据（Metadata）进行匹配验证。
- **01_matching/01_extract_attributes.py**: 提取用户偏好实体及上下文
- **01_matching/01_verify_attributes.py**: 基于元数据验证属性，过滤虚假或泛化的匹配

**运行命令**：
```bash
# 生成匹配任务
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/01_matching/01_extract_attributes.py \
     --input /home/wlia0047/wenyu/result/user_profile/preferences_USER_ID.json \
     --meta-file /home/wlia0047/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
     --output-dir /home/wlia0047/wenyu/result/user_profile/match_tasks"

# 执行属性验证 (需要根据 match_tasks 结果手动执行验证)
```

---

### Stage 2: 数据处理与集合划分 (Processing)

**阶段描述**：筛选高质量用户并进行类目感知的训练集/测试集划分。
- **02_processing/02_select_users.py**: 筛选满足评论数量且类目解析完整的"完美"用户
- **02_processing/02_split_train_holdout.py**: 智能划分 10 个 Holdout 商品作为测试集，确保其类目在训练集中有覆盖

**运行命令**：
```bash
# 1. 筛选用户
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/02_processing/02_select_users.py \
     --meta-file /home/wlia0047/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
     --reviews-file /home/wlia0047/wenyu/data/Amazon-Reviews-2018/processed/user_reviews/user_product_reviews.json \
     --min-reviews 100 \
     --max-reviews 110 \
     --max-users 10"

# 2. 划分训练集/测试集
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/02_processing/02_split_train_holdout.py \
     --input /home/wlia0047/wenyu/result/user_profile/selected_users.json \
     --meta-file /home/wlia0047/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
     --output-dir /home/wlia0047/wenyu/result/user_profile/train_holdout_splits"
```

---

### Stage 3: 用户画像生成 (Persona)

**阶段描述**：基于训练集数据合成差异化的画像。
- **03_persona/03_generate_persona_prompts.py**: 提取训练集属性，计算独特性分数 (Uniqueness Score) 并生成画像生成指令
- **03_persona/03_execute_persona_generation.py**: 调用 LLM 批量生成 150-200 字的接地气画像

**运行命令**：
```bash
# 生成画像提示词
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/03_persona/03_generate_persona_prompts.py \
     --input-dir /home/wlia0047/wenyu/result/user_profile/train_holdout_splits \
     --holdout-dir /home/wlia0047/wenyu/result/user_profile/train_holdout_splits/holdout \
     --meta-file /home/wlia0047/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
     --output-dir /home/wlia0047/wenyu/result/user_profile/persona_prompts"

# 执行画像生成 (需要 LLM 调用，根据生成的 prompts 手动执行)
```

---

### Stage 4: 写作风格分析 (Writing Analysis)

**阶段描述**：分析用户的拼写和语法错误习惯，用于生成更真实的查询。
- **04_writing_analysis/04_generate_writing_prompts.py**: 为每个用户的评论生成错误分析提示词
- **04_writing_analysis/04_execute_writing_analysis.py**: 调用 LLM 分析拼写错误 (10种) 和语法错误 (7种)，输出统计报告

**运行命令**：
```bash
# 生成写作风格分析提示词
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/04_writing_analysis/04_generate_writing_prompts.py \
     --reviews-file /home/wlia0047/wenyu/result/user_profile/user_reviews/all_user_reviews.json \
     --user-ids USER_ID_1 USER_ID_2 \
     --output-dir /home/wlia0047/wenyu/result/user_profile/writing_analysis_prompts \
     --num-reviews 20"

# 执行写作风格分析 (需要 LLM 调用，根据生成的 prompts 手动执行)
```

---

### Stage 5: 语言学特征提取 (Syntactic Analysis)

**阶段描述**：提取语言学特征，用于 Stage 8 风格对齐。
- **05_syntactic_analysis/05_extract_local_features.py**: 使用 spaCy 本地提取 18 维特征

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate ppo_env && \
     python -u 05_extract_local_features.py \
     --reviews-file /home/wlia0047/wenyu/result/user_profile/00_data_preparation/all_user_reviews.json \
     --output-dir /home/wlia0047/wenyu/result/user_profile/05_syntactic_analysis \
     --max-reviews 50"
```

---

### Stage 6: 双查询生成 (Query Generation)

**阶段描述**：针对 Holdout 商品生成对比查询。
- **06_query/06_generate_dual_queries.py**: 生成 **Public (大众版)** 和 **Personalized (基于画像版)** 两种查询词

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u 06_generate_dual_queries.py \
     --match-results-dir /home/wlia0047/wenyu/result/user_profile/01_matching/results \
     --preferences-dir /home/wlia0047/wenyu/result/user_profile/00_data_preparation/preferences \
     --output-dir /home/wlia0047/wenyu/result/user_profile/06_query"
```

---

### Stage 8: 迭代式风格优化 (Iterative Refinement)

**阶段描述**：使用 GLM-5 模型对查询进行迭代式优化，使其风格对齐到用户写作特征。通过计算 16 维风格特征向量的距离判断收敛，阈值设为 0.02。
- **08_multi_candidate_filtering/08_iterative_refinement.py**: 迭代式优化主脚本
- **08_multi_candidate_filtering/extract_sentence_level_features.py**: 句子级别特征提取
- **08_multi_candidate_filtering/run_iterative_refinement.sh**: 运行脚本

**运行命令**：
```bash
# 迭代式优化
bash /home/wlia0047/wenyu/.claude/skills/user_profile/08_multi_candidate_filtering/run_iterative_refinement.sh

# 句子级别特征提取
bash /home/wlia0047/wenyu/.claude/skills/user_profile/08_multi_candidate_filtering/run_sentence_level_extraction.sh
```

---

### Stage 9: 噪声查询生成 (Noisy Query)

**阶段描述**：基于写作风格分析结果，为个性化查询注入用户特有的拼写/语法错误。
- **09_noisy_query/09_generate_noisy_queries.py**: 读取 Stage 4 的错误统计，按权重对查询进行单错误注入

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/09_noisy_query/09_generate_noisy_queries.py \
     --dual-queries-dir /home/wlia0047/wenyu/result/user_profile/06_query \
     --writing-analysis-dir /home/wlia0047/wenyu/result/user_profile/04_writing_analysis/results \
     --output-dir /home/wlia0047/wenyu/result/user_profile/09_noisy_query"
```

---

### Stage 10: 多维度评估 (Evaluation)

**阶段描述**：验证个性化提升效果。
- **10_evaluation/10_evaluate_with_unique_persona.py**: 关联度评估 (1-10 分)
- **10_evaluation/10_evaluate_semantic_similarity.py**: 语义相似度分析
- **10_evaluation/10_evaluate_persona_diversity.py**: 画像多样性评估

**运行命令**：
```bash
# 1. 关联度评估 (1-10 分)
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/10_evaluation/10_evaluate_with_unique_persona.py \
     --dual-queries-dir /home/wlia0047/wenyu/result/user_profile/06_query \
     --persona-dir /home/wlia0047/wenyu/result/user_profile/03_persona/results \
     --output-dir /home/wlia0047/wenyu/result/user_profile/10_evaluation/relevance \
     --user-ids USER_ID_1 USER_ID_2"

# 2. 语义相似度分析
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/10_evaluation/10_evaluate_semantic_similarity.py \
     --dual-queries-dir /home/wlia0047/wenyu/result/user_profile/06_query \
     --output-dir /home/wlia0047/wenyu/result/user_profile/10_evaluation/semantic"

# 3. 画像多样性评估
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/10_evaluation/10_evaluate_persona_diversity.py \
     --persona-dir /home/wlia0047/wenyu/result/user_profile/03_persona/results \
    --output-dir /home/wlia0047/wenyu/result/user_profile/10_evaluation/diversity"
```
