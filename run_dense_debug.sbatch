#!/bin/bash
#SBATCH --job-name=dense_debug_batch55
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=/home/wlia0047/ar57/wenyu/logs/dense_debug_%j.log

set -e

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting dense retriever debug test (sparse skipped)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

cd /fs04/ar57/wenyu

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/stark

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running single user (User 1) with dense retrievers only..."
echo "  - Batch size: 2500 docs/batch (reduced from 5000)"
echo "  - Timeout: 1800s (increased from 1200s)"
echo "  - Thread dump: Enabled at timeout"
echo ""

python3 ./.claude/skills/PersoanlQuery/12_retrieval/evaluators/12_evaluate_all_users_fullscale.py --users 1 --mode both

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dense debug test completed!"
