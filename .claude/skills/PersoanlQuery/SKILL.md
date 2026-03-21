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
     python -u .claude/skills/PersoanlQuery/01_preference_extraction/01_batch_extract_preferences_all.py \
     --output-dir result/personal_query/01_preference_extraction \
     --max-workers 10"
```

### Stage 2: 数据处理与过滤

**阶段描述**：将 Stage 1 的偏好提取结果进行数据过滤，只保留满足属性阈值的商品作为查询集。该阶段将复杂的3阶段管道简化为单一的数据过滤阶段，移除所有画像生成逻辑。

1. **Stage 2a**: 过滤用户偏好数据 (02_split_train_holdout.py)
   - 只保留 min_attrs >= 5 的商品
   - 删除所有 min_cat_size 限制
   - 生成结构化的查询集

**输出结构**：
```
02_processing/
├── {user_id}/
│   └── query.json                          # 过滤后的查询集（min_attrs >= 5）
```

**运行命令**：
```bash
# 处理所有用户（默认）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/02_processing/run_stage2_pipeline.py"

# 或只处理特定用户
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/02_processing/run_stage2_pipeline.py \
     --user-id A13OFOB1394G31"
```

**参数说明**：
- `--min-attrs`: 查询集商品最少属性数 (默认: 5)
- `--user-id`: 只处理指定用户 (默认: 处理所有用户)
- `--seed`: 随机种子 (默认: 42)

---

### Stage 3: 画像描述生成

**阶段描述**：注：该阶段需要更新以适配新的Stage 2输出结构。原有的Stage 3依赖于Stage 2b和Stage 2c生成的个人画像和大众画像。由于Stage 2已简化为仅输出查询集，Stage 3的功能需要重新设计或重新实现。

**当前状态**：该阶段的脚本尚未适配新的数据流。

**历史说明**：
- 原 Stage 3 读取 Stage 2 的画像数据（persona 和 mass_market），为每个维度生成自然语言描述
- 原输入文件：`02_processing/{user_id}/persona/{category}.json` 和 `02_processing/{user_id}/mass_market/{category}.json`

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
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
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

**阶段描述**：提取构建索引文件。

**运行命令**：

```bash
# 批量处理所有用户（自动发现完成Stage 6的用户）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval/evaluators && \
     python -u 12_build_retriever_indices.py"
```

**阶段描述**：构建查询语句索引文件。

**运行命令**：

```bash
# 批量处理所有用户（自动发现完成Stage 6的用户）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval/evaluators && \
     python -u 12_generate_query_cache.py"
```


**阶段描述**：使用优化的批量评估系统，高效评估所有用户的查询在多种检索模型下的表现。该系统通过共享文档加载和检索器索引，大幅提升评估效率。

**运行命令**：

```bash
# 批量处理所有用户（自动发现完成Stage 6的用户）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval&& \
     python -u 12_evaluate_all_users_fullscale.py"
```

---

### Stage 13: 三路LLM评估 + 属性值选择评估

**阶段描述**：该阶段包含两个核心功能的组合评估系统：


**运行命令**：
```bash
# 批量处理所有查询（包含属性选择评估和商品验证）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/13_rerank && \
     python3 13_batch_llm_rerank_all.py --config 15_config.json \
     --meta-file /fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz"

# 单独运行属性选择脚本（单条查询 JSON 示例）
# query-file 需要是单条查询 JSON，不能直接传 dual_queries_*.json
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python3 /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/13_rerank/13_select_attributes_from_history.py \
     --user-id A2U6VP21H9UVV3 \
     --category Yarn \
     --query-file /fs04/ar57/wenyu/result/personal_query/13_rerank/results/tmp_attr_query_A2U6_yarn.json \
     --meta-file /fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
     --output-file /fs04/ar57/wenyu/result/personal_query/13_rerank/results/attr_select_A2U6_yarn_thr0.7_rerun.json"
```

其中 `tmp_attr_query_A2U6_yarn.json` 的内容示例：

```json
{
  "user_id": "A2U6VP21H9UVV3",
  "category": "Yarn",
  "target_asin": "B00HMXK5DK",
  "query_text": "I am looking for reflective yarn to make a reflective scarf. I need a product in the yarn category that offers high reflective performance for safety and style.",
  "selected_attributes": [
    {"dimension": "Functionality", "value": "reflective"},
    {"dimension": "Product_Category", "value": "reflective yarn"},
    {"dimension": "Product_Category", "value": "yarn"},
    {"dimension": "Performance", "value": "reflective performance"},
    {"dimension": "Style_Design", "value": "reflective scarf"}
  ]
}
```
