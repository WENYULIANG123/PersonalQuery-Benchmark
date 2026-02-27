#!/bin/bash
#SBATCH --job-name=verify_attrs
#SBATCH --output=/home/wlia0047/ar57/wenyu/logs/verify_attributes_%j.out
#SBATCH --error=/home/wlia0047/ar57/wenyu/logs/verify_attributes_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=3

USER_ID=$1
TASK_FILE="/home/wlia0047/wenyu/result/user_profile/01_matching/tasks/match_tasks_${USER_ID}.json"
OUTPUT_DIR="/home/wlia0047/wenyu/result/user_profile/01_matching/results"

echo "=========================================="
echo "Processing user: ${USER_ID}"
echo "Task file: ${TASK_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=========================================="

python -u /home/wlia0047/ar57/wenyu/.claude/skills/user_profile/01_matching/01_verify_attributes.py \
    --task-file "${TASK_FILE}" \
    --output-dir "${OUTPUT_DIR}"
    --output-dir "${OUTPUT_DIR}" \
    --max-workers 3

exit_code=$?
echo "Job completed with exit code: ${exit_code}"
exit ${exit_code}
