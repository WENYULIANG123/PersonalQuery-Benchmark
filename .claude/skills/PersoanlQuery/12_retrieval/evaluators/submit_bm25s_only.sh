#!/bin/bash
#SBATCH --job-name=test_bm25s_only
#SBATCH --output=/home/wlia0047/ar57/wenyu/test_bm25s_only_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=comp

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/stark

cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval/evaluators

python3 test_bm25s_only.py
