---
name: personal_query    
description: 提取细粒度用户偏好并生成"接地气"的用户画像与个性化搜索查询。
---

# Personal Query Manager (个性化查询管理)

此技能实现了一套完整的 **"Grounded" (落地/实操型)** 个性化 pipeline，通过 **12 个有序阶段**将原始用户评论转化为高度差异化的画像及搜索查询，并进行多维度评估。

## 执行环节 (Step-by-Step Pipeline)

---

### Stage 0: 用户筛选与数据准备 (Data Preparation)

**阶段描述**：从大量用户中筛选出高质量用户（有元数据、评论数在范围内），然后加载这些用户的评论数据。

**输入**：评论文件、元数据文件
**输出**：
- `reviews_{USER_ID}.json`：每个用户的评论数据（包含 target_review + other_reviews）
- `selected_users.json`：筛选出的用户列表

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
     --max-reviews 110 \
     --max-users 10 \
     --output-dir result/personal_query/00_data_preparation"
```

**筛选条件**：
- 用户评论的商品必须在元数据中（排除 Unknown 类目）
- 每个商品至少有 5 条评论（可调整 `--min-product-reviews`）
- 每个商品的其他用户评论至少有 4 条（可调整 `--min-other-reviews`）
- 用户有效商品数在 [min_reviews, max_reviews] 范围内（默认 100-110）
- 按商品数量排序，选择前 N 个用户（默认 10 个）

---

### Stage 1: 偏好提取 (Preference Extraction)

**阶段描述**：使用LLM从评论中提取偏好，区分目标用户（target）和其他用户（other）。

**输入**：`reviews_{USER_ID}.json` (Stage 0 输出)
**输出**：`preferences_{USER_ID}.json` (包含 target_user_preferences + other_users_preferences)

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/01_preference_extraction/01_extract_preferences.py \
     --input-file result/personal_query/00_data_preparation/reviews_A1BBCMQSEJN0PP.json \
     --output-dir result/personal_query/01_preference_extraction \
     --max-workers 50"
```

---

### Stage 2: 偏好分类 (Preference Classification)

**阶段描述**：区分目标用户偏好（target_attributes）和其他用户偏好（public_attributes），不做元数据匹配验证。

> **注意**：元数据匹配验证已被禁用。原因是用户评论（主观感受）与产品元数据（客观描述）语义空间不同，直接匹配会导致过度过滤。

**输入**：`preferences_{USER_ID}.json` (Stage 1 输出)
**输出**：`match_{USER_ID}.json` (包含 target_attributes + public_attributes)

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

**特性**：
- 区分个性化属性（target_attributes）和大众属性（public_attributes）
- 不做元数据验证，保留所有 Stage 1 提取的偏好
- 去重逻辑在 Stage 3 中进行

---

### Stage 3: 数据集划分 (Data Split)

**阶段描述**：对 Stage 2 输出的匹配结果进行类目感知的训练集/测试集划分。

**划分逻辑**：
- **候选商品**：属性数 ≥ 3 且 public_attributes ≥ 3
- **画像集（Persona Set）**：用于生成用户画像，包含非候选商品 + 每类目最多 4 个候选商品
- **查询集（Query Set）**：用于生成和测试个性化查询，包含超过 4 个的候选商品
- **后处理**：画像集属性去重，包括：
  1. 四元组精确去重
  2. 字符串相似度去重 (70%)
  3. **语义相似度去重** (使用 sentence-transformers + 社区检测，仅当维度属性数 ≥ 6 时触发)
- **语义去重**：按商品类别分组后，在每个类别内部按维度进行语义相似度聚类，使用 `all-MiniLM-L6-v2` 模型，阈值 0.85

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/03_processing/03_split_train_holdout.py \
     --match-dir result/personal_query/02_matching \
     --output-dir result/personal_query/03_processing"
```

---

### Stage 5: 写作风格分析 (Writing Analysis)

**阶段描述**：分析用户的拼写和语法错误习惯，用于生成更真实的查询。

**运行命令**：
```bash
# Step 1: 生成写作风格分析提示词（默认分析全部评论）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/05_writing_analysis/05_generate_writing_prompts.py \
     --reviews-file /fs04/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{USER_ID}.json \
     --user-ids {USER_ID} \
     --output-dir /fs04/ar57/wenyu/result/personal_query/05_writing_analysis/prompts"

# Step 2: 执行写作风格分析（默认 20 并发）
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/05_writing_analysis/05_execute_writing_analysis.py \
     --prompts-dir /fs04/ar57/wenyu/result/personal_query/05_writing_analysis/prompts \
     --output-dir /fs04/ar57/wenyu/result/personal_query/05_writing_analysis/results"
```


### Stage 6: 语言学特征提取 (Syntactic Analysis)

**阶段描述**：提取 16 维语言学特征，用于 Stage 8 风格对齐。默认处理全部评论。

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

### Stage 7: 双查询生成 (Dual Query Generation)

**阶段描述**：针对 Holdout 商品生成 Public (大众版) 和 Personalized (个性化版) 两种对比查询，用于评估个性化效果。

**输入**：`query_{USER_ID}.json` (Stage 3 输出，包含 target_attributes + public_attributes)
**输出**：`dual_queries_{USER_ID}.json` (包含 target_user_query + mass_market_query)

**核心逻辑**：
- 从 Stage 3 的 query 文件中提取 target_attributes 和 public_attributes
- 对每个商品，找到两者的**共享维度**，随机选择 5 个
- 使用相同的维度生成两种查询，确保公平对比
- Target query 反映用户个性化偏好
- Mass market query 反映大众市场偏好

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

**可选参数**：
- `--input-file`: 输入文件路径 (默认: Stage 3 的 query 文件)
- `--output-dir`: 输出目录 (默认: `result/personal_query/07_query`)
- `--workers N`: 并发数 (默认: 5)
- `--seed N`: 随机种子，用于复现结果

---

### Stage 8: 迭代式风格优化 (Iterative Refinement)

**阶段描述**：使用 GLM-5 模型对查询进行迭代式优化，使其风格对齐到用户写作特征。通过计算 16 维风格特征向量的距离判断收敛，阈值设为 0.02。

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

### Stage 9: 拼写难度打分模型 (Spelling Difficulty Scorer)

**阶段描述**：基于单词长短、词频、语言学陷阱等特征，与用户独立的多维错误偏好比率联合训练认知拼写难度模型。这为后续真正注入个性化噪声提供精准靶点参考，防止大模型误改简单基础词汇。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/09_spelling_difficulty/08_train_spelling_model.py"
```

---

### Stage 10: 靶向噪声注入 (Targeted Noisy Query)

**阶段描述**：在真实噪声注入阶段，系统提取真实用户偏好计算出的分布权重（如”同音词误用率”、”漏字率”），通过难度模型提前筛选查询中的易错靶点，指导 LLM 严格按照该名用户的错字频率分布进行精确注入。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/10_targeted_noisy_query/09_generate_noisy_queries.py \
     --spelling-model-path /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/09_spelling_difficulty/09_spelling_difficulty_scorer_v1.pt \
     --stage8-results /home/wlia0047/wenyu/result/personal_query/08_iterative_refinement/iterative_refinement_v2/iterative_results.json \
     --writing-analysis-dir /home/wlia0047/wenyu/result/personal_query/05_writing_analysis/results \
     --output-dir /home/wlia0047/wenyu/result/personal_query/10_targeted_noisy_query \
     --user-ids A24FX30B20WLMV"
```

---

### Stage 11: 多维度评估 (Evaluation)

**阶段描述**：验证个性化提升效果，对比 Public Query vs Personalized Query。

**输入**：
- `dual_queries_{USER_ID}.json` (Stage 7 输出)
- 用户画像文件 (Stage 4 输出)

**输出**：
- `llm_evaluation_{USER_ID}.json` (LLM 评分结果)
- `semantic_similarity_{USER_ID}.json` (语义相似度结果)
- `diversity_metrics.json` (画像多样性指标)

**运行命令**：
```bash
# 1. LLM 评分 (基于用户画像评估查询质量)
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/11_evaluation/11_evaluate_LLM_score.py \
     --input-file /fs04/ar57/wenyu/result/personal_query/07_query/dual_queries_A13OFOB1394G31.json \
     --persona-dir /fs04/ar57/wenyu/result/personal_query/04_persona \
     --output-dir /fs04/ar57/wenyu/result/personal_query/11_evaluation \
     --workers 10"

# 2. 语义相似度分析 (Sentence-BERT 或 OpenAI)
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/11_evaluation/11_evaluate_semantic_similarity.py \
     --dual-queries-dir /fs04/ar57/wenyu/result/personal_query/07_query \
     --persona-dir /fs04/ar57/wenyu/result/personal_query/04_persona \
     --output-dir /fs04/ar57/wenyu/result/personal_query/11_evaluation \
     --method sbert"

# 3. 画像多样性评估
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu && \
     python -u .claude/skills/PersoanlQuery/11_evaluation/11_evaluate_persona_diversity.py \
     --persona-dir /fs04/ar57/wenyu/result/personal_query/04_persona \
     --output-file /fs04/ar57/wenyu/result/personal_query/11_evaluation/diversity_metrics.json"
```

**可选参数**：

LLM 评分：
- `--dual-queries-dir`: 双查询文件目录
- `--persona-dir`: 用户画像目录
- `--output-dir`: 输出目录
- `--user-id`: 用户 ID

语义相似度：
- `--method`: 嵌入方法 (`sbert` 或 `openai`，默认 `sbert`)

画像多样性：
- `--persona-dir`: 用户画像目录

**评估规则说明**：

Stage 11 使用**维度特定的评估规则**，每个维度有自己独立的评估标准：

| 维度类别 | 维度 | 评估问题 |
|----------|------|----------|
| 产品属性 | Product_Category | 查询是否指定了符合用户偏好的产品类型？ |
| | Functionality | 查询是否提到用户想要的产品功能？ |
| | Material_Composition | 查询是否指定了符合用户偏好的材料？ |
| 质量属性 | Quality_Craftsmanship | 查询是否表达符合用户期望的质量要求？ |
| | Performance | 查询是否提到符合用户需求的性能期望？ |
| | Safety | 查询是否解决用户的安全顾虑？ |
| 外观设计 | Appearance_Color | 查询是否指定符合用户审美的颜色/外观？ |
| | Size_Dimensions | 查询是否指定符合用户需求的尺寸规格？ |
| | Style_Design | 查询是否表达符合用户品味的设计风格？ |
| 用户体验 | Comfort | 查询是否提到符合用户舒适度要求？ |
| | Ease_of_Use | 查询是否表达符合用户对简便性的偏好？ |
| | Portability | 查询是否提到符合用户便携性需求？ |
| 使用场景 | Target_User | 查询是否表明适合目标用户类型？ |
| | Usage_Scenario | 查询是否提到符合用户实际使用场景？ |
| | Special_Purpose | 查询是否提到符合用户特殊用途？ |
| 价格价值 | Price | 查询是否表达符合用户预算的价格期望？ |
| | Value | 查询是否表达符合用户价值取向？ |
| | Packaging_Quantity | 查询是否指定符合用户购买习惯的包装数量？ |
| 特殊要求 | Compatibility | 查询是否提到符合用户现有设备的兼容性？ |
| | Special_User_Needs | 查询是否解决用户的特殊需求？ |
| | Brand_Preference | 查询是否提到符合用户品牌忠诚度的品牌？ |

评分：每个通过的维度得 1 分，总分 0-5（取决于使用的维度数量）
- `--output-file`: 输出 JSON 文件路径

---

### Stage 12: 人类评估 (Human Evaluation)

**阶段描述**：验证 LLM 自动评估与真人评估的对齐度，检测系统性偏见。

**运行命令**：
```bash
# 1. 生成评估任务和 HTML 界面
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_human_evaluation/11_generate_human_eval_tasks.py \
     --stage10-dir /home/wlia0047/wenyu/result/personal_query/11_evaluation \
     --stage9-dir /home/wlia0047/wenyu/result/personal_query/10_targeted_noisy_query \
     --persona-dir /home/wlia0047/wenyu/result/personal_query/04_persona/results \
     --output-dir /home/wlia0047/wenyu/result/personal_query/12_human_evaluation/tasks"

# 2. 在浏览器中打开 evaluation_interface.html 完成人工评估

# 3. 计算对齐指标
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_human_evaluation/11_compute_alignment_metrics.py \
     --human-results /home/wlia0047/wenyu/result/personal_query/12_human_evaluation/human_eval_results.json \
     --llm-results /home/wlia0047/wenyu/result/personal_query/11_evaluation/evaluation_summary.json \
     --llm-dir /home/wlia0047/wenyu/result/personal_query/11_evaluation \
     --output-dir /home/wlia0047/wenyu/result/personal_query/12_human_evaluation/reports"

# 4. 生成可视化报告
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_human_evaluation/11_generate_report.py \
     --metrics-dir /home/wlia0047/wenyu/result/personal_query/12_human_evaluation/reports \
     --output-dir /home/wlia0047/wenyu/result/personal_query/12_human_evaluation/reports"
```
