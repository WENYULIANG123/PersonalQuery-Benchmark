---
name: personal_query    
description: 提取细粒度用户偏好并生成"接地气"的用户画像与个性化搜索查询。
---

# Personal Query Manager

此技能实现了一套完整的 **"Grounded" (落地/实操型)** 个性化 pipeline，通过 **14 个有序阶段**（Stage 0-13）将原始用户评论转化为高度差异化的画像及搜索查询，并进行多维度评估。

---

### Stage 0: 用户筛选与数据准备

**阶段描述**：从大量用户中筛选出高质量用户（有元数据、评论数在范围内），然后加载这些用户的评论数据。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/00_data_preparation/00_batch_prepare_data.py \
     --review-file /fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz \
     --meta-file /fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
     --min-reviews 100 \
     --max-reviews 400 \
     --max-users 10 \
     --output-dir result/personal_query/00_data_preparation"
```

---

### Stage 1: 偏好提取

**阶段描述**：使用LLM从评论中提取偏好，区分目标用户（target）和其他用户（other）。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py  \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/01_preference_extraction/01_batch_extract_preferences.py \
     --output-dir result/personal_query/01_preference_extraction \
     --max-workers 10"
```

### Stage 2: 数据处理与画像生成

**阶段描述**：将 Stage 1 的偏好提取结果处理成结构化的用户画像和查询数据。完整流程包括：
1. **Stage 2a**: 分割画像集和查询集 (02_split_train_holdout.py)
2. **Stage 2b**: 生成个人画像 (02_generate_persona.py)  
3. **Stage 2c**: 生成大众画像 (02_generate_mass_market_data.py)

**输出结构**：
```
02_processing/
├── {user_id}/
│   ├── query.json                          # 画像集和查询集
│   ├── persona/{category}.json             # 个人画像（按类目）
│   └── mass_market/{category}.json         # 大众画像（按类目）
```

**运行命令**：
```bash
# 处理所有用户（默认）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/02_processing/run_stage2_pipeline.py"

```

---

### Stage 3: 画像描述生成

**阶段描述**：读取 Stage 2 的画像数据，为每个维度生成自然语言描述。

**输入文件**：
- `02_processing/{user_id}/persona/{category}.json` - 个人画像（来自 Stage 2b）
- `02_processing/{user_id}/mass_market/{category}.json` - 大众画像（来自 Stage 2c）

**运行命令**：
```bash
# 目标用户画像描述
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/03_persona/03_generate_target_persona.py \
     --user-id A13OFOB1394G31 \
     --input-dir result/personal_query/02_processing \
     --output-dir result/personal_query/03_persona"

# 大众画像描述
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/03_persona/03_generate_mass_market_persona.py \
     --input-dir result/personal_query/02_processing \
     --output-dir result/personal_query/03_persona"
```

---

### Stage 4: 写作风格分析

**阶段描述**：分析用户的字符级拼写错误，进行详细的错误分类和统计，用于生成更真实的查询。

**运行命令**：
```bash
# 处理所有选中用户（默认）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/04_writing_analysis/04_extract_all_user_errors.py"

```


### Stage 5: 语言学特征提取

**阶段描述**：提取 16 维语言学特征，用于 Stage 7 风格对齐。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis/05_extract_local_features.py \
     --reviews-file /fs04/ar57/wenyu/result/personal_query/00_data_preparation/all_user_reviews.json \
     --output-dir /fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis"
```

---

### Stage 6: 双查询生成

**阶段描述**：针对 Holdout 商品生成 Public (大众版) 和 Personalized (个性化版) 两种对比查询。

**运行命令**：
```bash
# 处理所有用户（默认）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/06_query/06_generate_all_user_queries.py"

```

### Stage 7: 迭代式风格优化

**阶段描述**：使用 GLM-5 模型对查询进行迭代式优化，使其风格对齐到用户写作特征。

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

### Stage 8: 拼写难度打分模型

**阶段描述**：训练认知拼写难度模型，为后续噪声注入提供精准靶点参考。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/08_spelling_difficulty/08_train_spelling_model.py"
```

---

### Stage 9: 靶向噪声注入

**阶段描述**：根据用户真实拼写错误历史，精确注入个性化噪声到查询中。支持单用户处理和批量处理。

**运行命令**：

#### 批量处理所有用户（推荐）
```bash
# 批量处理所有完成Stage 6的用户
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/09_targeted_noisy_query/09_generate_all_user_noisy_queries.py"
```

#### 处理单个用户
```bash
# 单个用户处理
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/09_targeted_noisy_query/09_generate_noisy_queries.py \
     --stage7-results result/personal_query/07_iterative_refinement/iterative_refinement_v2/iterative_results.json \
     --writing-analysis result/personal_query/04_writing_analysis/results/writing_analysis_{USER_ID}.json \
     --output-dir result/personal_query/09_targeted_noisy_query \
     --user-ids {USER_ID}"
```

**批量脚本特性**：
- 自动扫描Stage 6的输出，找出所有待处理用户
- 生成每个用户的噪声查询文件：`noisy_queries_{user_id}.json`
- 生成汇总报告：`all_users_summary.json`，包含所有用户的处理统计
- 支持并行处理，显著提高效率

**参数说明**：
- `--stage7-results`: Stage 7输出（仅单用户模式需要）
- `--writing-analysis`: Stage 4写作分析（仅单用户模式需要）
- `--user-ids`: 处理的用户ID列表（仅单用户模式需要）
---

### Stage 10: 多维度评估

**阶段描述**：验证个性化提升效果，对比 Public Query vs Personalized Query。

**运行命令**：
```bash
# LLM 评分
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/10_evaluation/10_evaluate_LLM_score.py \
     --input-file /fs04/ar57/wenyu/result/personal_query/06_query/dual_queries_A13OFOB1394G31.json \
     --persona-dir /fs04/ar57/wenyu/result/personal_query/03_persona \
     --output-dir /fs04/ar57/wenyu/result/personal_query/10_evaluation \
     --workers 10"

# 语义相似度分析
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/10_evaluation/10_evaluate_semantic_similarity.py \
     --dual-queries-dir /fs04/ar57/wenyu/result/personal_query/06_query \
     --persona-dir /fs04/ar57/wenyu/result/personal_query/03_persona \
     --output-dir /fs04/ar57/wenyu/result/personal_query/10_evaluation \
     --method sbert"

# 画像多样性评估
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/10_evaluation/10_evaluate_persona_diversity.py \
     --persona-dir /fs04/ar57/wenyu/result/personal_query/03_persona \
     --output-file /fs04/ar57/wenyu/result/personal_query/10_evaluation/diversity_metrics.json"
```

---

### Stage 11: 人类评估

**阶段描述**：验证 LLM 自动评估与真人评估的对齐度，检测系统性偏见。

**运行命令**：
```bash
# Step 1: 生成人类评估任务
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/11_human_evaluation/11_generate_human_eval_tasks.py \
     --stage10-dir /home/wlia0047/wenyu/result/personal_query/10_evaluation \
     --stage9-dir /home/wlia0047/wenyu/result/personal_query/09_targeted_noisy_query \
     --persona-dir /home/wlia0047/wenyu/result/personal_query/03_persona/results \
     --output-dir /home/wlia0047/wenyu/result/personal_query/11_human_evaluation/tasks"

# Step 2: 计算对齐指标
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/11_human_evaluation/11_compute_alignment_metrics.py \
     --human-results /home/wlia0047/wenyu/result/personal_query/11_human_evaluation/human_eval_results.json \
     --llm-results /home/wlia0047/wenyu/result/personal_query/10_evaluation/evaluation_summary.json \
     --llm-dir /home/wlia0047/wenyu/result/personal_query/10_evaluation \
     --output-dir /home/wlia0047/wenyu/result/personal_query/11_human_evaluation/reports"

# Step 3: 生成报告
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/11_human_evaluation/11_generate_report.py \
     --metrics-dir /home/wlia0047/wenyu/result/personal_query/11_human_evaluation/reports \
     --output-dir /home/wlia0047/wenyu/result/personal_query/11_human_evaluation/reports"
```

---

### Stage 12: 检索评估

**阶段描述**：批量评估所有用户的查询在多种检索模型下的表现，计算 Recall@K, MAP, NDCG, MRR 等指标。

**运行命令**：

```bash
# 批量处理所有用户（自动发现完成Stage 6的用户）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval/evaluators && \
     python 12_evaluate_all_users_retrieval.py"
```

### Stage 13: LLM Reranking

**阶段描述**：使用大语言模型（GLM、Minimax、Qwen）对 Stage 12 的检索结果进行重排序，利用用户画像上下文提升个性化效果。

**主要特性**：
- **Persona Context**: 基于 Stage 1 偏好自动构建用户画像上下文
- **Conflict Resolution**: 自动识别并过滤与查询冲突的偏好
- **Two Modes**: 支持 Standard（无画像）和 Personalized（含画像）双模式对比评估

---

#### 🧠 GLM Reranker（智谱 AI）

**脚本路径**：`.claude/skills/PersoanlQuery/13_rerank/llm_reranking/`

**架构**: BM25 召回 → GLM API 重排

**支持模型**：
- **GLM-4.5V**: 视觉语言模型
- **GLM-4.7**: 高级推理模型
- **GLM-5**: 最新旗舰模型

**运行示例**：

```bash
# GLM-4.5V
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_rerank/llm_reranking/13_evaluate_glm_4_5v_both.py \
     --user-id A13OFOB1394G31"

# GLM-4.7
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_rerank/llm_reranking/13_evaluate_glm_4_7_both.py \
     --user-id A13OFOB1394G31"

# GLM-5
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_rerank/llm_reranking/13_evaluate_glm_5_both.py \
     --user-id A13OFOB1394G31"
```

---

#### ⚡ Minimax Reranker

**脚本路径**：`.claude/skills/PersoanlQuery/13_rerank/llm_reranking/`

**架构**: E5 召回 → Minimax API 重排

**支持模型**：
- **M2.5-highspeed**: 高速推理版本（推荐）
- **M2.1**: 优化版本
- **M2**: 基础版本

**特性**：
- 自动 500 错误重试
- 支持并发请求（提升速度）
- 完整的错误日志

**运行示例**：

```bash
# M2.5-highspeed（推荐）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_rerank/llm_reranking/13_evaluate_minimax_m2_5_highspeed.py \
     --user-id A13OFOB1394G31"

# M2.1
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_rerank/llm_reranking/13_evaluate_minimax_m2_1.py \
     --user-id A13OFOB1394G31"

# M2
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_rerank/llm_reranking/13_evaluate_minimax_m2.py \
     --user-id A13OFOB1394G31"
```

---

#### 🤖 Qwen Reranker

**脚本路径**：`.claude/skills/PersoanlQuery/13_rerank/llm_reranking/13_evaluate_qwen_7b.py`

**架构**: BM25 召回 → Qwen-7B 重排

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_rerank/llm_reranking/13_evaluate_qwen_7b.py \
     --user-id A13OFOB1394G31"
```

---

#### 📊 输出格式

**输出目录**：`result/personal_query/13_rerank/`

**文件命名**：
- Standard 模式：`rerank_{model}_standard_{user_id}.json`
- Personalized 模式：`rerank_{model}_personalized_{user_id}.json`

**指标**：
- Recall@K（K=1,3,5,10）
- MRR（Mean Reciprocal Rank）
- NDCG@K（Normalized Discounted Cumulative Gain）
- MAP（Mean Average Precision）

**详细文档**：`/home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/13_rerank/README.md`

---