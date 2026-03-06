# Stage 11: Human Evaluation

人类评估阶段，用于评估 LLM 自动评估（Stage 10）与真人评估之间的对齐度。

## 功能特性

- ✅ **生成评估任务** - 自动合并 Stage 9/10/3 的数据生成评估任务
- ✅ **交互式 HTML 评估界面** - 单文件 HTML，支持实时保存和进度跟踪
- ✅ **计算对齐指标** - Spearman, Cohen's Kappa, Recall, MAE, 系统性偏见
- ✅ **生成可视化报告** - Markdown + 专业图表

## 文件结构

```
11_human_evaluation/
├── 11_generate_human_eval_tasks.py    # 生成评估任务和 HTML 界面
├── 11_compute_alignment_metrics.py    # 计算 LLM 与真人评估的对齐指标
├── 11_generate_report.py              # 生成可视化报告
├── evaluation_interface.html           # 中文版评估界面
├── evaluation_interface_en.html        # 英文版评估界面
└── README.md                           # 本文档
```

## 使用流程

### Step 1: 生成评估任务

```bash
python 11_generate_human_eval_tasks.py \
    --stage10-dir /path/to/10_evaluation \
    --stage9-dir /path/to/09_targeted_noisy_query \
    --persona-dir /path/to/03_persona/results \
    --output-dir /path/to/11_human_evaluation/tasks
```

**输出：**
- `tasks/human_eval_tasks.json` - 评估任务定义
- `tasks/evaluation_interface.html` - 人类评估界面
- `tasks/evaluation_interface_en.html` - 英文版评估界面

### Step 2: 人类评估（手动）

1. 在浏览器中打开 `evaluation_interface.html`
2. 完成查询对的评估（会实时保存进度）
3. 点击 "Download Results" 下载 JSON 文件
4. 保存到结果目录

### Step 3: 计算对齐指标

```bash
python 11_compute_alignment_metrics.py \
    --human-results /path/to/human_eval_results.json \
    --llm-results /path/to/evaluation_summary.json \
    --llm-dir /path/to/10_evaluation \
    --output-dir /path/to/reports
```

**输出：**
- `reports/alignment_metrics.json` - 所有指标计算结果

### Step 4: 生成可视化报告

```bash
python 11_generate_report.py \
    --metrics-dir /path/to/reports \
    --output-dir /path/to/reports
```

**输出：**
- `reports/alignment_report.md` - Markdown 报告
- `reports/figures/` - 可视化图表

## 评估指标说明

### 1. Spearman's Rank Correlation (ρ)
- 评估平均分的排名对齐
- ρ > 0.8 强相关，ρ > 0.6 中等相关

### 2. Cohen's Kappa (κ)
- 评估胜率的一致性（排除随机因素）
- κ > 0.8 极高一致性，κ > 0.6 显著一致

### 3. Recall @ Human Preference
- 当真人认为个性化查询更好时，LLM 也选中的概率

### 4. Mean Absolute Error (MAE)
- LLM 与真人评分的平均绝对误差

### 5. Systematic Bias Analysis
- 检测 LLM 是否存在正向偏见

## 依赖包

- scipy
- scikit-learn
- matplotlib
- numpy

---

**实现完成日期**：2026-03-04
**Stage 版本**：v1.0
**状态**：✅ Ready for Use
