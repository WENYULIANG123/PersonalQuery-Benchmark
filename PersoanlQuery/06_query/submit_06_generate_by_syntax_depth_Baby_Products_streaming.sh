#!/usr/bin/env bash
#SBATCH --job-name=submit_06_generate_by_syntax_depth_Baby_Products_streaming
#SBATCH --output=/home/wlia0047/ar57/wenyu/logs/06_generate_by_syntax_depth_Baby_Products_%j.log
#SBATCH --error=/home/wlia0047/ar57/wenyu/logs/06_generate_by_syntax_depth_Baby_Products_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/stark

cd /home/wlia0047/ar57/wenyu

python /home/wlia0047/ar57/wenyu/PersoanlQuery/06_query/06_generate_by_syntax_depth_Baby_Products.py
