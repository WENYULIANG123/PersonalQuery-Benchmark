#!/bin/bash
#SBATCH --output=/home/wlia0047/ar57/wenyu/logs/lingconv_%j.log
#SBATCH --error=/home/wlia0047/ar57/wenyu/logs/lingconv_%j.err
#SBATCH -p m3h
#SBATCH --gres=gpu:H100:1
#SBATCH --qos=m3h
#SBATCH --time=24:00:00
#SBATCH --mem=64G

set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57/wenyu/envs/lingconv
cd /fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis
export PYTHONUNBUFFERED=1

# 删除之前的checkpoint文件
rm -rf /home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints/*

# 使用极简配置测试基线
python -u main.py \
    --do_train \
    --data ling_conversion \
    --ckpt_dir /home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints \
    --model_name google/flan-t5-base \
    --batch_size 8 \
    --gradient_accumulation 1 \
    --epochs 1 \
    --lr 5e-5 \
    --max_length 64 \
    --name '0403_simple_test'
