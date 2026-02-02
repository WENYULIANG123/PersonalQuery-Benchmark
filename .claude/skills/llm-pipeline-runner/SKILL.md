---
name: LLM-Pipeline-Runner
description: 封装了 Qwen2-1.5B 模型的训练与评估流程。支持三种训练模式（baseline, noise_aug, multitask），并通过 sbatch 提交集群作业。
allowed-tools: run_command, command_status
---

# LLM-Pipeline-Runner

此技能用于启动和监控基于 Qwen2-1.5B 的检索模型训练流水线。该流水线支持多任务学习与噪声增强训练。

## 输入文件配置

训练流水线会自动读取以下输入文件：

- **Clean Query**: `/home/wlia0047/ar57/wenyu/result/clean_query/clean_queries.csv`
- **Noisy Query**: `/home/wlia0047/ar57/wenyu/result/noisy_query/noisy_queries.csv`
- **商品元数据**: `/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz`

确保这些文件存在且格式正确。

## 训练模式说明

1.  **baseline**: 仅使用 Clean Query 进行对比学习。
2.  **noise_aug**: 仅使用注入噪声后的 Noisy Query 进行训练。
3.  **multitask**: 同时使用 Clean 和 Noisy Query，并引入对齐损失（Alignment Loss）来提升鲁棒性。

## 执行流程

### 1. 启动训练作业

**【推荐】同时运行所有三种模式：**

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && conda activate /home/wlia0047/ar57_scratch/wenyu/stark && python -u /home/wlia0047/ar57/wenyu/.claude/skills/llm-pipeline-runner/llm_pipeline.py --run_all --epochs 9"
```

**单模式运行（以 multitask 为例）：**

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && conda activate /home/wlia0047/ar57_scratch/wenyu/stark && python -u /home/wlia0047/ar57/wenyu/.claude/skills/llm-pipeline-runner/llm_pipeline.py --mode multitask --epochs 9"
```

### 2. GPU 参数说明

**重要**：GPU 参数必须传递给 `sbatch_wrapper.py`，而不是传递给 Python 脚本。

```bash
# ✅ 正确方式：--gpu 在 sbatch_wrapper.py 之前
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu "你的命令"

# ❌ 错误方式：--partition gpu 在 Python 命令中
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py "python xxx.py --partition gpu"
```

**`sbatch_wrapper.py` 会自动添加以下 SLURM 配置：**
- `#SBATCH -p gpu`（指定 GPU 分区）
- `#SBATCH --gres=gpu:1`（请求 1 块 GPU）
- `#SBATCH --mem=32G`（请求 32GB 内存）

### 3. 参数说明

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `--run_all` | 同时运行三种模式 | 无需值 | False |
| `--mode` | 单模式运行时指定 | `baseline`, `noise_aug`, `multitask` | `baseline` |
| `--epochs` | 训练轮数 | 整数（推荐 9） | 5 |

### 4. 状态监控
使用 `squeue` 命令跟踪作业进度：

```bash
squeue -u wlia0047  # 查看当前用户的作业
sacct -j <JOB_ID>   # 查看作业详细状态
```

重点关注日志中的 `Epoch Loss`、`MRR`, `Hit@10` 和 `NDCG@10` 指标。

### 5. 查看最终结果

**日志文件位置：**
- 训练日志：`/home/wlia0047/ar57/wenyu/logs/llm_pipeline_<JOB_ID>.log`
- 错误日志：`/home/wlia0047/ar57/wenyu/logs/llm_pipeline_<JOB_ID>.err`
- 持久化日志：`/home/wlia0047/ar57/wenyu/stark/code/train_model/results_qwen/training_qwen.log`

**如果运行了 `--run_all`：**
- 控制台和错误日志末尾会输出最终对比表格
- 表格包含三种模式的 MRR、Hit@1/3/5/10、NDCG@10 指标

## 注意事项

### 显存与 Batch Size
- 当前配置针对 Qwen2-1.5B 模型优化（Batch Size = 4）
- 使用 bfloat16 精度减少显存占用
- 请勿在没有足够显存的情况下随意调大 batch_size

### 集群依赖
- **必须使用 `sbatch_wrapper.py`** 以确保作业正确提交至 SLURM 调度系统
- **必须使用 `--gpu` 参数** 以请求 GPU 资源
- 不要直接运行 Python 脚本，会因为没有 GPU 而失败

### 训练时间参考
- 单模式（9 epochs）：约 5-6 分钟
- 全部三种模式（9 epochs）：约 15-20 分钟
