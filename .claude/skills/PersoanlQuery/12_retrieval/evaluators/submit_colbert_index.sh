#!/bin/bash
#SBATCH --job-name=colbert_index
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=m3a120,m3a119,m3a118,m3n104,m3n106,m3n107,m3n112
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/colbert_index_%j.log
#SBATCH --error=logs/colbert_index_%j.err

cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval/evaluators

mkdir -p logs

echo "Starting ColBERT Index Building on A100"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo ""

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/stark

python 12_build_colbert_index.py

echo ""
echo "ColBERT index building complete!"
echo "Finished at: $(date)"
