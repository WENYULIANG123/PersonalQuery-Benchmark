---
name: personal_query    
description: 提取细粒度用户偏好并生成"接地气"的用户画像与个性化搜索查询。
---

# Personal Query Manager (个性化查询管理)

此技能实现了一套完整的 **"Grounded" (落地/实操型)** 个性化 pipeline，通过 **11 个有序阶段**将原始用户评论转化为高度差异化的画像及搜索查询，并进行多维度评估。

## 执行环节 (Step-by-Step Pipeline)

---

### Stage 0: 数据准备 (Data Preparation)

**阶段描述**：提取用户偏好并准备基础数据。

**运行命令**：
```bash
# 1. 提取用户评论
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/00_data_preparation/00_extract_user_reviews.py \
     --reviews-file /home/wlia0047/wenyu/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz \
     --users-file /home/wlia0047/wenyu/result/personal_query/selected_users.json \
     --output-dir /home/wlia0047/wenyu/result/personal_query/user_reviews"

# 2. 批量提取用户偏好
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/00_data_preparation/00_batch_extract_preferences.py \
     --reviews-dir /home/wlia0047/wenyu/result/personal_query/user_reviews \
     --output-dir /home/wlia0047/wenyu/result/personal_query/00_data_preparation \
     --meta-file /home/wlia0047/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz"

# 3. 筛选包含长句的用户
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/00_data_preparation/00_select_users_with_long_sentences.py \
     --reviews-file /home/wlia0047/wenyu/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz \
     --output-dir /home/wlia0047/wenyu/result/personal_query/00_data_preparation \
     --min-products 180 \
     --max-products 220 \
     --min-long-review-ratio 0.2"
```

---

### Stage 1: 偏好提取与属性验证 (Matching)

**阶段描述**：从原始评论中提取实体，并与商品元数据（Metadata）进行匹配验证。

**运行命令**：
```bash
# 生成匹配任务
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/01_matching/01_extract_attributes.py \
     --input /home/wlia0047/wenyu/result/personal_query/preferences_USER_ID.json \
     --meta-file /home/wlia0047/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
     --output-dir /home/wlia0047/wenyu/result/personal_query/match_tasks"

# 执行属性验证 (需要根据 match_tasks 结果手动执行验证)
```

---

### Stage 2: 数据处理与集合划分 (Processing)

**阶段描述**：筛选高质量用户并进行类目感知的训练集/测试集划分。

**运行命令**：
```bash
# 1. 筛选用户
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/02_processing/02_select_users.py \
     --meta-file /home/wlia0047/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
     --reviews-file /home/wlia0047/wenyu/data/Amazon-Reviews-2018/processed/user_reviews/user_product_reviews.json \
     --min-reviews 100 \
     --max-reviews 110 \
     --max-users 10"

# 2. 划分训练集/测试集
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/02_processing/02_split_train_holdout.py \
     --input /home/wlia0047/wenyu/result/personal_query/selected_users.json \
     --meta-file /home/wlia0047/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
     --output-dir /home/wlia0047/wenyu/result/personal_query/train_holdout_splits"
```

---

### Stage 3: 用户画像生成 (Persona)

**阶段描述**：基于训练集数据合成差异化的"接地气"画像，支持跨商品类型差异处理。

**运行命令**：
```bash
python .claude/skills/PersoanlQuery/03_persona/03_generate_persona_prompts.py \
    --input-dir /fs04/ar57/wenyu/result/personal_query/01_matching/results \
    --meta-file /fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
    --output-dir /fs04/ar57/wenyu/result/personal_query/persona_prompts_category_aware_v2
```

---

### Stage 4: 写作风格分析 (Writing Analysis)

**阶段描述**：分析用户的拼写和语法错误习惯，用于生成更真实的查询。

**运行命令**：
```bash
# 生成写作风格分析提示词
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/04_writing_analysis/04_generate_writing_prompts.py \
     --reviews-file /home/wlia0047/wenyu/result/personal_query/user_reviews/all_user_reviews.json \
     --user-ids USER_ID_1 USER_ID_2 \
     --output-dir /home/wlia0047/wenyu/result/personal_query/writing_analysis_prompts \
     --num-reviews 20"

# 执行写作风格分析 (需要 LLM 调用，根据生成的 prompts 手动执行)
```

---

### Stage 5: 语言学特征提取 (Syntactic Analysis)

**阶段描述**：提取语言学特征，用于 Stage 8 风格对齐。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate ppo_env && \
     python -u 05_extract_local_features.py \
     --reviews-file /home/wlia0047/wenyu/result/personal_query/00_data_preparation/all_user_reviews.json \
     --output-dir /home/wlia0047/wenyu/result/personal_query/05_syntactic_analysis \
     --max-reviews 50"
```

---

### Stage 6: 双查询生成 (Dual Query Generation)

**阶段描述**：针对 Holdout 商品生成 Public (大众版) 和 Personalized (个性化版) 两种对比查询，用于评估个性化效果。

**运行命令**：
```bash
python .claude/skills/PersoanlQuery/06_query/06_generate_dual_queries.py \
    --match-results-dir /fs04/ar57/wenyu/result/personal_query/01_matching/results \
    --preferences-dir /fs04/ar57/wenyu/result/personal_query/00_data_preparation \
    --holdout-dir /fs04/ar57/wenyu/result/personal_query/02_processing \
    --output-dir /home/wlia0047/wenyu/result/personal_query/06_query
```

---

### Stage 7: 迭代式风格优化 (Iterative Refinement)

**阶段描述**：使用 GLM-5 模型对查询进行迭代式优化，使其风格对齐到用户写作特征。通过计算 16 维风格特征向量的距离判断收敛，阈值设为 0.02。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/07_iterative_refinement/07_iterative_refinement.py \
     --query-dir /home/wlia0047/wenyu/result/personal_query/06_query \
     --linguistic-dir /home/wlia0047/wenyu/result/personal_query/05_syntactic_analysis \
     --output-dir /home/wlia0047/wenyu/result/personal_query/07_iterative_refinement \
     --max-rounds 5 \
     --candidates-per-round 3 \
     --feature-set style_only_16 \
     --max-workers 1"
```

---

### Stage 8: 拼写难度打分模型 (Spelling Difficulty Scorer)

**阶段描述**：基于单词长短、词频、语言学陷阱等特征，与用户独立的多维错误偏好比率联合训练认知拼写难度模型。这为后续真正注入个性化噪声提供精准靶点参考，防止大模型误改简单基础词汇。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/08_spelling_difficulty/08_train_spelling_model.py"
```

---

### Stage 9: 靶向噪声注入 (Targeted Noisy Query)

**阶段描述**：在真实噪声注入阶段，系统提取真实用户偏好计算出的分布权重（如”同音词误用率”、”漏字率”），通过难度模型提前筛选查询中的易错靶点，指导 LLM 严格按照该名用户的错字频率分布进行精确注入。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/09_targeted_noisy_query/09_generate_noisy_queries.py \
     --spelling-model-path /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/08_spelling_difficulty/08_spelling_difficulty_scorer_v1.pt \
     --stage8-results /home/wlia0047/wenyu/result/personal_query/07_iterative_refinement/iterative_refinement_v2/iterative_results.json \
     --writing-analysis-dir /home/wlia0047/wenyu/result/personal_query/04_writing_analysis/results \
     --output-dir /home/wlia0047/wenyu/result/personal_query/09_targeted_noisy_query \
     --user-ids A24FX30B20WLMV"
```

---

### Stage 10: 多维度评估 (Evaluation)

**阶段描述**：验证个性化提升效果，对比 Public Query vs Personalized Query。

**运行命令**：
```bash
# 1. LLM 关联度评分 (1-10 分)
python .claude/skills/PersoanlQuery/10_evaluation/10_evaluate_with_unique_persona.py \
    --dual-queries-dir /home/wlia0047/wenyu/result/personal_query/06_query \
    --persona-dir /fs04/ar57/wenyu/result/personal_query/persona_results_category_aware_v2 \
    --output-dir /home/wlia0047/wenyu/result/personal_query/10_evaluation_v2

# 2. 语义相似度分析 (Sentence-BERT)
python .claude/skills/PersoanlQuery/10_evaluation/10_evaluate_semantic_similarity.py \
    --dual-queries-dir /home/wlia0047/wenyu/result/personal_query/06_query \
    --persona-dir /fs04/ar57/wenyu/result/personal_query/persona_results_category_aware_v2 \
    --output-dir /home/wlia0047/wenyu/result/personal_query/10_evaluation_semantic \
    --method sbert

# 3. 画像多样性评估
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/10_evaluation/10_evaluate_persona_diversity.py \
     --persona-dir /home/wlia0047/wenyu/result/personal_query/03_persona/results \
     --output-dir /home/wlia0047/wenyu/result/personal_query/10_evaluation/diversity"
```

---

### Stage 11: 人类评估 (Human Evaluation)

**阶段描述**：验证 LLM 自动评估与真人评估的对齐度，检测系统性偏见。

**运行命令**：
```bash
# 1. 生成评估任务和 HTML 界面
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/11_human_evaluation/11_generate_human_eval_tasks.py \
     --stage10-dir /home/wlia0047/wenyu/result/personal_query/10_evaluation \
     --stage9-dir /home/wlia0047/wenyu/result/personal_query/09_targeted_noisy_query \
     --persona-dir /home/wlia0047/wenyu/result/personal_query/03_persona/results \
     --output-dir /home/wlia0047/wenyu/result/personal_query/11_human_evaluation/tasks"

# 2. 在浏览器中打开 evaluation_interface.html 完成人工评估

# 3. 计算对齐指标
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/11_human_evaluation/11_compute_alignment_metrics.py \
     --human-results /home/wlia0047/wenyu/result/personal_query/11_human_evaluation/human_eval_results.json \
     --llm-results /home/wlia0047/wenyu/result/personal_query/10_evaluation/evaluation_summary.json \
     --llm-dir /home/wlia0047/wenyu/result/personal_query/10_evaluation \
     --output-dir /home/wlia0047/wenyu/result/personal_query/11_human_evaluation/reports"

# 4. 生成可视化报告
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/11_human_evaluation/11_generate_report.py \
     --metrics-dir /home/wlia0047/wenyu/result/personal_query/11_human_evaluation/reports \
     --output-dir /home/wlia0047/wenyu/result/personal_query/11_human_evaluation/reports"
```
