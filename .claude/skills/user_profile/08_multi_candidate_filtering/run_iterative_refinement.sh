#!/bin/bash
#SBATCH --job-name=iter_refine
#SBATCH --output=logs/iterative_refinement_%j.out
#SBATCH --error=logs/iterative_refinement_%j.err
#SBATCH --partition=m3j
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

mkdir -p /home/wlia0047/wenyu/logs

export PATH="/home/wlia0047/ar57_scratch/wenyu/stark/bin:$PATH"

cd /home/wlia0047/wenyu/.claude/skills/user_profile/08_multi_candidate_filtering

echo "========================================="
echo "Iterative Refinement"
echo "========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Model: GLM-5"
echo "Base Query: personalized_query"
echo "Feature Set: style_only_16 (normalized to 0-1)"
echo "User Features: SENTENCE-LEVEL (25-30 word sentences)"
echo "Semantic: Embedding-based similarity (sentence-transformers)"
echo "Max rounds: 1"
echo "Candidates per round: 2"
echo "Max workers: 1 (sequential)"
echo "========================================="

# Run iterative refinement with sentence-level features
python 08_iterative_refinement.py \
    --query-dir /home/wlia0047/wenyu/result/user_profile/06_query \
    --linguistic-dir /home/wlia0047/wenyu/result/user_profile/05_syntactic_analysis \
    --output-dir /home/wlia0047/wenyu/result/user_profile/08_multi_candidate_filtering/iterative_refinement \
    --max-rounds 1 \
    --candidates-per-round 2 \
    --feature-set style_only_16 \
    --max-workers 1 \
    --use-sentence-level \
    --sentence-level-dir /home/wlia0047/wenyu/result/user_profile/08_multi_candidate_filtering/sentence_features

echo "========================================="
echo "Completed at $(date)"
echo "========================================="
