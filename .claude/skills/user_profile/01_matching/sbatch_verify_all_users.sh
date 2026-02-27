#!/bin/bash
#SBATCH --job-name=verify_all_users
#SBATCH --output=/home/wlia0047/ar57/wenyu/logs/verify_all_users_%j.out
#SBATCH --error=/home/wlia0047/ar57/wenyu/logs/verify_all_users_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=3

MATCH_TASKS_DIR="/home/wlia0047/wenyu/result/user_profile/01_matching/tasks"
OUTPUT_DIR="/home/wlia0047/wenyu/result/user_profile/01_matching/results"
PYTHON_SCRIPT="/home/wlia0047/ar57/wenyu/.claude/skills/user_profile/01_matching/01_verify_attributes.py"

# List of all 10 users
USERS=(
    "AKNND79UEP12S"
    "A3TSKADB7Y3BZK"
    "A16KPQPPSZPX7M"
    "A1ZV7FZ1LAIYHR"
    "A34ATJR9KFIXL9"
    "A1A9PD00UVHHVI"
    "A24T8E1AREI9X4"
    "A2PGJP6GV2ZC02"
    "A2GI6MB9RZL72O"
    "AD8YVUDBRMFTH"
)

echo "=========================================="
echo "Starting Stage 1.2: Verify Attributes"
echo "Total users: ${#USERS[@]}"
echo "=========================================="

# Process each user sequentially
for i in "${!USERS[@]}"; do
    USER_ID="${USERS[$i]}"
    TASK_FILE="${MATCH_TASKS_DIR}/match_tasks_${USER_ID}.json"

    if [ ! -f "$TASK_FILE" ]; then
        echo "ERROR: Task file not found: $TASK_FILE"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Processing user [$((i+1))/${#USERS[@]}]: ${USER_ID}"
    echo "=========================================="

    python -u "${PYTHON_SCRIPT}" \
        --task-file "${TASK_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-workers 3

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ User ${USER_ID} completed successfully"
    else
        echo "✗ User ${USER_ID} failed with exit code: ${exit_code}"
    fi
done

echo ""
echo "=========================================="
echo "All users processed!"
echo "=========================================="

exit 0
