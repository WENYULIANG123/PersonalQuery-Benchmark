#!/bin/bash
#SBATCH --output=/home/wlia0047/ar57/wenyu/logs/main_%j.log
#SBATCH --error=/home/wlia0047/ar57/wenyu/logs/main_%j.err
#SBATCH --job-name=main_pipeline

set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/stark
cd /home/wlia0047/ar57/wenyu/stark/code/generate_query
python main.py
