#!/bin/bash

set -e

echo "=========================================="
echo "Testing single-user evaluation (User 1)"
echo "=========================================="
echo ""

# Get the first user ID
FIRST_USER=$(python3 -c "
import json, glob, os
pattern = os.path.join('/home/wlia0047/ar57/wenyu/result/personal_query/06_query', 'dual_queries_*.json')
files = sorted(glob.glob(pattern))
if files:
    filename = os.path.basename(files[0])
    user_id = filename[13:-5]
    print(user_id)
")

echo "Running test on user: $FIRST_USER"
echo ""

# Run the evaluation with sbatch_wrapper
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python3 ./.claude/skills/PersoanlQuery/12_retrieval/evaluators/12_evaluate_all_users_fullscale.py --users 1"

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "=========================================="
