---
name: personal_query    
description: 提取细粒度用户偏好并生成"接地气"的用户画像与个性化搜索查询。
---

# Personal Query Manager

此技能实现了一套完整的 **"Grounded" (落地/实操型)** 个性化 pipeline，通过 **14 个有序阶段**将原始用户评论转化为高度差异化的画像及搜索查询，并进行多维度评估。

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
     --max-reviews 150 \
     --max-users 10 \
     --output-dir result/personal_query/00_data_preparation"
```

---

### Stage 1: 偏好提取

**阶段描述**：使用LLM从评论中提取偏好，区分目标用户（target）和其他用户（other）。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/01_preference_extraction/01_batch_extract_preferences.py \
     --input-file result/personal_query/00_data_preparation/reviews_A13OFOB1394G31.json \
     --output-dir result/personal_query/01_preference_extraction \
     --max-workers 50"
---

### Stage 2: 偏好分类

**阶段描述**：区分目标用户偏好（target_attributes）和其他用户偏好（public_attributes），不做元数据匹配验证。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/02_matching/02_verify_with_metadata_v2.py \
     --input-file result/personal_query/01_preference_extraction/preferences_A13OFOB1394G31.json \
     --output-dir result/personal_query/02_matching"
```

---

### Stage 3: 数据处理与画像生成

**阶段描述**：对 Stage 2 输出的匹配结果进行数据分割和画像生成。

**运行命令**：
```bash
# Step 1: 数据分割
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/03_processing/03_split_train_holdout.py \
     --match-dir result/personal_query/02_matching \
     --output-dir result/personal_query/03_processing"

# Step 2: 个人画像生成
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/03_processing/03_generate_persona.py \
     --query-file result/personal_query/03_processing/query_{USER_ID}.json \
     --output-dir result/personal_query/03_processing"

# Step 3: 大众画像生成
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/03_processing/03_generate_mass_market_data.py \
     --input-file result/personal_query/02_matching/match_{USER_ID}.json \
     --output-dir result/personal_query/03_processing"
```

---

### Stage 4: 画像描述生成

**阶段描述**：读取 Stage 3 的画像数据，为每个维度生成自然语言描述。

**运行命令**：
```bash
# 目标用户画像描述
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/04_persona/04_generate_target_persona.py \
     --user-id A13OFOB1394G31 \
     --input-dir result/personal_query/03_processing \
     --output-dir result/personal_query/04_persona"

# 大众画像描述
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/04_persona/04_generate_mass_market_persona.py \
     --input-dir result/personal_query/03_processing \
     --output-dir result/personal_query/04_persona"
```

---

### Stage 5: 写作风格分析

**阶段描述**：分析用户的字符级拼写错误，进行详细的错误分类和统计，用于生成更真实的查询。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/05_writing_analysis/05_character_level_errors.py \
     --reviews-file /fs04/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{USER_ID}.json \
     --user-ids {USER_ID} \
     --output-dir /fs04/ar57/wenyu/result/personal_query/05_writing_analysis/results \
     --max-workers 50"
```

**参数说明**：
- `--reviews-file`: 输入的用户评论文件（Stage 0输出）
- `--user-ids`: 处理的用户ID列表
- `--output-dir`: 输出目录（会保存writing_analysis_{USER_ID}.json）
- `--max-workers`: 并发处理的worker数量（默认：50，最大：50）

### Stage 6: 语言学特征提取

**阶段描述**：提取 16 维语言学特征，用于 Stage 8 风格对齐。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/06_syntactic_analysis/06_extract_local_features.py \
     --reviews-file /fs04/ar57/wenyu/result/personal_query/00_data_preparation/all_user_reviews.json \
     --output-dir /fs04/ar57/wenyu/result/personal_query/06_syntactic_analysis"
```

---

### Stage 7: 双查询生成

**阶段描述**：针对 Holdout 商品生成 Public (大众版) 和 Personalized (个性化版) 两种对比查询。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/07_query/07_generate_dual_queries.py \
     --input-file /fs04/ar57/wenyu/result/personal_query/03_processing/query_{USER_ID}.json \
     --output-dir /fs04/ar57/wenyu/result/personal_query/07_query \
     --workers 10"
```

---

### Stage 8: 迭代式风格优化

**阶段描述**：使用 GLM-5 模型对查询进行迭代式优化，使其风格对齐到用户写作特征。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/08_iterative_refinement/08_iterative_refinement.py \
     --query-dir /home/wlia0047/wenyu/result/personal_query/07_query \
     --linguistic-dir /home/wlia0047/wenyu/result/personal_query/06_syntactic_analysis \
     --output-dir /home/wlia0047/wenyu/result/personal_query/08_iterative_refinement \
     --max-rounds 5 \
     --candidates-per-round 3 \
     --feature-set style_only_16 \
     --max-workers 1"
```

---

### Stage 9: 拼写难度打分模型

**阶段描述**：训练认知拼写难度模型，为后续噪声注入提供精准靶点参考。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/09_spelling_difficulty/08_train_spelling_model.py"
```

---

### Stage 10: 靶向噪声注入

**阶段描述**：根据用户真实拼写错误历史，精确注入个性化噪声到查询中。

**运行命令**：
```bash
# 仅拼写错误模式（默认）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/10_targeted_noisy_query/10_generate_noisy_queries.py \
     --stage7-results result/personal_query/08_iterative_refinement/iterative_refinement_v2/iterative_results.json \
     --writing-analysis result/personal_query/05_writing_analysis/results/writing_analysis_{USER_ID}.json \
     --output-dir result/personal_query/10_targeted_noisy_query \
     --user-ids {USER_ID}"
```

**参数说明**：
- `--stage7-results`: Stage 7输出（必需）
- `--writing-analysis`: Stage 5写作分析（必需，个性化注入必须）
- `--user-ids`: 处理的用户ID列表
---

### Stage 11: 多维度评估

**阶段描述**：验证个性化提升效果，对比 Public Query vs Personalized Query。

**运行命令**：
```bash
# LLM 评分
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/11_evaluation/11_evaluate_LLM_score.py \
     --input-file /fs04/ar57/wenyu/result/personal_query/07_query/dual_queries_A13OFOB1394G31.json \
     --persona-dir /fs04/ar57/wenyu/result/personal_query/04_persona \
     --output-dir /fs04/ar57/wenyu/result/personal_query/11_evaluation \
     --workers 10"

# 语义相似度分析
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/11_evaluation/11_evaluate_semantic_similarity.py \
     --dual-queries-dir /fs04/ar57/wenyu/result/personal_query/07_query \
     --persona-dir /fs04/ar57/wenyu/result/personal_query/04_persona \
     --output-dir /fs04/ar57/wenyu/result/personal_query/11_evaluation \
     --method sbert"

# 画像多样性评估
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/11_evaluation/11_evaluate_persona_diversity.py \
     --persona-dir /fs04/ar57/wenyu/result/personal_query/04_persona \
     --output-file /fs04/ar57/wenyu/result/personal_query/11_evaluation/diversity_metrics.json"
```

---

### Stage 12: 人类评估

**阶段描述**：验证 LLM 自动评估与真人评估的对齐度，检测系统性偏见。

**运行命令**：
```bash
# Step 1: 生成人类评估任务
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_human_evaluation/11_generate_human_eval_tasks.py \
     --stage10-dir /home/wlia0047/wenyu/result/personal_query/11_evaluation \
     --stage9-dir /home/wlia0047/wenyu/result/personal_query/10_targeted_noisy_query \
     --persona-dir /home/wlia0047/wenyu/result/personal_query/04_persona/results \
     --output-dir /home/wlia0047/wenyu/result/personal_query/12_human_evaluation/tasks"

# Step 2: 计算对齐指标
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_human_evaluation/11_compute_alignment_metrics.py \
     --human-results /home/wlia0047/wenyu/result/personal_query/12_human_evaluation/human_eval_results.json \
     --llm-results /home/wlia0047/wenyu/result/personal_query/11_evaluation/evaluation_summary.json \
     --llm-dir /home/wlia0047/wenyu/result/personal_query/11_evaluation \
     --output-dir /home/wlia0047/wenyu/result/personal_query/12_human_evaluation/reports"

# Step 3: 生成报告
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_human_evaluation/11_generate_report.py \
     --metrics-dir /home/wlia0047/wenyu/result/personal_query/12_human_evaluation/reports \
     --output-dir /home/wlia0047/wenyu/result/personal_query/12_human_evaluation/reports"
```

---

### Stage 13: 检索评估

**阶段描述**：使用多种检索模型对查询进行评估，计算 Precision@K, Recall@K, MAP, NDCG, MRR 等指标。

---

#### 🚀 主脚本：批量运行所有检索方法（推荐）

**脚本路径**：`.claude/skills/PersoanlQuery/13_retrieval/evaluators/run_all_retrieval.py`

**包含的检索方法**（11种，排除重排序）：
- **Dense Retrieval** (7个): ANCE, BGE, Dense, E5, MiniLM, MPNet, STAR
- **Late Interaction** (1个): ColBERT
- **Sparse Retrieval** (3个): BM25, Dirichlet, TF-IDF

**基本用法**（运行所有检索，clean + noisy 模式）：

```bash
cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/13_retrieval/evaluators

# 使用 sbatch_wrapper
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/13_retrieval/evaluators && \
     python run_all_retrieval.py --user-id A13OFOB1394G31"
```

**只运行 clean 或 noisy 模式**：

```bash
# 只运行 clean 模式
python run_all_retrieval.py --user-id A13OFOB1394G31 --mode clean

# 只运行 noisy 模式
python run_all_retrieval.py --user-id A13OFOB1394G31 --mode noisy
```

**失败时继续运行**：

```bash
python run_all_retrieval.py --user-id A13OFOB1394G31 --continue-on-error
```

**参数说明**：
- `--user-id`: 用户ID（默认：`A13OFOB1394G31`）
- `--mode`: 查询模式 - `both`, `clean`, 或 `noisy`（默认：`both`）
- `--query-file`: 自定义查询文件路径（默认：自动从 Stage 10 输出生成）
- `--output-dir`: 输出目录（默认：`/fs04/ar57/wenyu/result/personal_query/13_retrieval`）
- `--continue-on-error`: 失败时继续运行（默认：遇到错误停止）

**输出**：
- 每个方法生成一个 JSON 文件：`retrieval_{method}_{mode}_{user_id}.json`
- 示例：`retrieval_bm25_clean_A13OFOB1394G31.json`, `retrieval_e5_noisy_A13OFOB1394G31.json`
- **自动生成影响分析**（仅当 `--mode both` 且全部成功时）：
  - `impact_ranking_{user_id}.txt`：按拼写错误影响排序的检索方法列表
  - 包含：各方法受影响程度排名、关键洞察、最脆弱/最鲁棒方法

**预期运行时间**：
- 单个方法：2-10 分钟
- 全部 22 次评估（11方法 × 2模式）：约 2-4 小时

**详细文档**：`/home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/13_retrieval/evaluators/README_RUN_ALL.md`

---

#### 单独运行各检索方法（高级用法）

**BM25 检索评估**

脚本路径：`.claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_bm25.py`

```bash
# 默认模式（使用噪声查询）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py  \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_bm25.py"

# 使用干净查询（无拼写错误）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py  \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_bm25.py --query-mode clean"

# 显式指定噪声查询
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py  \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_bm25.py --query-mode noisy"
```

**参数说明**：
- `--query-mode`: 查询模式，可选 `noisy`（使用带噪声查询）或 `clean`（使用干净查询），默认 `noisy`
- `--query-file`: 查询文件路径（默认：Stage 10 输出）
- `--output-dir`: 输出目录（默认：`result/personal_query/13_retrieval`）
- `--user-id`: 用户ID（默认：`A13OFOB1394G31`）

**输出文件**：
- Clean 模式：`retrieval_bm25_clean_{USER_ID}.json`
- Noisy 模式：`retrieval_bm25_noisy_{USER_ID}.json`
```

**TF-IDF**

脚本路径：`.claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_tfidf.py`

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py  \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_tfidf.py"
```

**Dirichlet Prior**

脚本路径：`.claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_dirichlet.py`

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_dirichlet.py"
```

---

##### Dense Retrievers

**Dense (ANCE/MiniLM-L6-v2)**

脚本路径：`.claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_dense.py`

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_dense.py"
```

**E5-large-v2**

脚本路径：`.claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_e5.py`

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_e5.py"
```

**BGE-large-en**

脚本路径：`.claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_bge.py`

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_bge.py"
```

---

##### Multi-vector Retriever

**ColBERTv2**

脚本路径：`.claude/skills/PersoanlQuery/13_retrieval/evaluators/late_interaction/13_evaluate_colbert.py`

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/late_interaction/13_evaluate_colbert.py"
```

---

##### SGPT / GritLM (与 STaRK 相同的 VSS 模型)

**GritLM (SGPT-7B-weightedmean-nli-bitfit)**

脚本路径：`.claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_gritlm.py`

- 使用与 STaRK 相同的 embedding 模型
- 模型: `Muennighoff/SGPT-7B-weightedmean-nli-bitfit`
- 这是 STaRK 论文中使用的 VSS 基线

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_gritlm.py"
```

---

### Stage 14: LLM Reranking

**阶段描述**：使用大语言模型（GLM、Minimax、Qwen）对 Stage 13 的检索结果进行重排序，利用用户画像上下文提升个性化效果。

**主要特性**：
- **Persona Context**: 基于 Stage 1 偏好自动构建用户画像上下文
- **Conflict Resolution**: 自动识别并过滤与查询冲突的偏好
- **Two Modes**: 支持 Standard（无画像）和 Personalized（含画像）双模式对比评估

---

#### 🧠 GLM Reranker（智谱 AI）

**脚本路径**：`.claude/skills/PersoanlQuery/14_rerank/llm_reranking/`

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
     python -u .claude/skills/PersoanlQuery/14_rerank/llm_reranking/13_evaluate_glm_4_5v_both.py \
     --user-id A13OFOB1394G31"

# GLM-4.7
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/14_rerank/llm_reranking/13_evaluate_glm_4_7_both.py \
     --user-id A13OFOB1394G31"

# GLM-5
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/14_rerank/llm_reranking/13_evaluate_glm_5_both.py \
     --user-id A13OFOB1394G31"
```

---

#### ⚡ Minimax Reranker

**脚本路径**：`.claude/skills/PersoanlQuery/14_rerank/llm_reranking/`

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
     python -u .claude/skills/PersoanlQuery/14_rerank/llm_reranking/13_evaluate_minimax_m2_5_highspeed.py \
     --user-id A13OFOB1394G31"

# M2.1
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/14_rerank/llm_reranking/13_evaluate_minimax_m2_1.py \
     --user-id A13OFOB1394G31"

# M2
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/14_rerank/llm_reranking/13_evaluate_minimax_m2.py \
     --user-id A13OFOB1394G31"
```

---

#### 🤖 Qwen Reranker

**脚本路径**：`.claude/skills/PersoanlQuery/14_rerank/llm_reranking/13_evaluate_qwen_7b.py`

**架构**: BM25 召回 → Qwen-7B 重排

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/14_rerank/llm_reranking/13_evaluate_qwen_7b.py \
     --user-id A13OFOB1394G31"
```

---

#### 📊 输出格式

**输出目录**：`result/personal_query/14_rerank/`

**文件命名**：
- Standard 模式：`rerank_{model}_standard_{user_id}.json`
- Personalized 模式：`rerank_{model}_personalized_{user_id}.json`

**指标**：
- Recall@K（K=1,3,5,10）
- MRR（Mean Reciprocal Rank）
- NDCG@K（Normalized Discounted Cumulative Gain）
- MAP（Mean Average Precision）

**详细文档**：`/home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/14_rerank/README.md`

---