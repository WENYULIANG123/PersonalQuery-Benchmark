#!/bin/bash
set -e

echo "====================================="
echo "运行修正后的LOOCV实验"
echo "修改：用NOISY query而不是CLEAN query评估"
echo "====================================="
echo ""

python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python3 /fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/unified_loocv_experiment.py"

