#!/bin/bash
#SBATCH --job-name=sentence_features
#SBATCH --output=logs/sentence_level_features_%j.out
#SBATCH --error=logs/sentence_level_features_%j.err
#SBATCH --partition=m3j
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

mkdir -p /home/wlia0047/wenyu/logs

export PATH="/home/wlia0047/ar57_scratch/wenyu/stark/bin:$PATH"

cd /home/wlia0047/wenyu/.claude/skills/user_profile/08_multi_candidate_filtering

echo "========================================="
echo "Sentence-Level Feature Extraction"
echo "========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Extracting features from 25-30 word sentences"
echo "Source: user_product_reviews.json"
echo "Output: sentence_level_features/"
echo "========================================="

python extract_sentence_level_features.py

echo "========================================="
echo "Completed at $(date)"
echo "========================================="
