#!/bin/bash
# Wrapper script to execute Python commands via sbatch and monitor logs

# Get the original command (all arguments combined)
# shlex.quote wraps commands in single quotes, so we need to handle that
if [ $# -eq 1 ]; then
    # If single argument and it's wrapped in quotes, remove the outer quotes
    if [[ "$1" =~ ^\'.*\'$ ]]; then
        # Remove outer single quotes
        ORIGINAL_COMMAND="${1#\'}"
        ORIGINAL_COMMAND="${ORIGINAL_COMMAND%\'}"
    elif [[ "$1" =~ ^\".*\"$ ]]; then
        # Remove outer double quotes
        ORIGINAL_COMMAND="${1#\"}"
        ORIGINAL_COMMAND="${ORIGINAL_COMMAND%\"}"
    else
        ORIGINAL_COMMAND="$1"
    fi
else
    # Combine all arguments
    ORIGINAL_COMMAND="$*"
fi

LOG_DIR="/home/wlia0047/ar57/wenyu/.cursor/hooks/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S_%N)
LOG_FILE="${LOG_DIR}/sbatch_${TIMESTAMP}.log"
ERR_FILE="${LOG_DIR}/sbatch_${TIMESTAMP}.err"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check if command is provided
if [ -z "$ORIGINAL_COMMAND" ]; then
    echo "[sbatch_wrapper] ❌ 错误: 未提供要执行的命令" >&2
    exit 1
fi

# Create a temporary script file for sbatch
# Use a here-document with proper escaping
SCRIPT_FILE="${LOG_DIR}/sbatch_script_${TIMESTAMP}.sh"
CURRENT_DIR="${PWD:-/home/wlia0047/ar57/wenyu}"

# Write the sbatch script
cat > "$SCRIPT_FILE" << EOF
#!/bin/bash
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${ERR_FILE}
set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/stark
cd "${CURRENT_DIR}"
# Execute the original command
${ORIGINAL_COMMAND}
EOF
chmod +x "$SCRIPT_FILE"

# Submit the job
echo "[sbatch_wrapper] 提交作业到 SLURM..." >&2
echo "[sbatch_wrapper] 日志文件: ${LOG_FILE}" >&2
echo "[sbatch_wrapper] 错误文件: ${ERR_FILE}" >&2

JOB_ID=$(sbatch "$SCRIPT_FILE" 2>&1 | grep -oP 'Submitted batch job \K[0-9]+' || echo "")

if [ -z "$JOB_ID" ]; then
    echo "[sbatch_wrapper] ❌ 提交 sbatch 作业失败" >&2
    exit 1
fi

echo "[sbatch_wrapper] ✅ 作业已提交，Job ID: ${JOB_ID}" >&2
echo "[sbatch_wrapper] 日志文件: ${LOG_FILE}" >&2
echo "[sbatch_wrapper] 错误文件: ${ERR_FILE}" >&2
echo "[sbatch_wrapper] 开始监控日志 (Ctrl+C 停止监控，作业将继续运行)..." >&2

# Function to monitor logs
monitor_logs() {
    # Wait for log files to be created (with timeout)
    MAX_WAIT=30
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        if [ -f "$LOG_FILE" ] || [ -f "$ERR_FILE" ]; then
            break
        fi
        sleep 1
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done
    
    # Start tail -f to monitor both log and error files
    if [ -f "$LOG_FILE" ] || [ -f "$ERR_FILE" ]; then
        tail -f "$LOG_FILE" "$ERR_FILE" 2>/dev/null &
        TAIL_PID=$!
        
        # Monitor until job completes or user interrupts
        while squeue -j "$JOB_ID" &>/dev/null; do
            sleep 1
        done
        
        # Stop tail when job completes
        kill $TAIL_PID 2>/dev/null
        wait $TAIL_PID 2>/dev/null
    else
        echo "[sbatch_wrapper] ⚠️  日志文件未在 ${MAX_WAIT} 秒内创建，继续等待作业完成..." >&2
        # Just wait for job to complete
        while squeue -j "$JOB_ID" &>/dev/null; do
            sleep 1
        done
    fi
}

# Run monitoring in background so we can still handle interrupts
monitor_logs &
MONITOR_PID=$!

# Wait for monitoring to complete
wait $MONITOR_PID 2>/dev/null

echo "[sbatch_wrapper] ✅ 作业已完成 (Job ID: ${JOB_ID})" >&2
if [ -f "$LOG_FILE" ] || [ -f "$ERR_FILE" ]; then
    echo "[sbatch_wrapper] 查看完整日志: tail -f ${LOG_FILE} ${ERR_FILE}" >&2
fi
