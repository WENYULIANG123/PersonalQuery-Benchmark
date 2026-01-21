#!/bin/bash
#
# SBATCH script to run amazon_review_style_analysis.py (user semantic style analysis)
# Minimal, single copy — cleaned to remove duplicated blocks.
#

# Job name and output
#SBATCH --job-name=amazon_generate_grammer_error_query
#SBATCH --output=/home/wlia0047/ar57_scratch/wenyu/logs/amazon_generate_grammer_error_query_%j.out
#SBATCH --error=/home/wlia0047/ar57_scratch/wenyu/logs/amazon_generate_grammer_error_query_%j.err

# Resources (adjust as needed)
#SBATCH --time=04:00:00
#SBATCH --partition=comp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Get current script name (without extension) for dynamic referencing
SCRIPT_NAME=amazon_generate_grammer_error_query  

# Get current SLURM job ID for tracking and logging
JOB_ID=${SLURM_JOB_ID:-"unknown"}

# Clean up previous log files (excluding current job's logs)
# Remove old grammer_aware log files except current job
for log_file in /home/wlia0047/ar57_scratch/wenyu/logs/${SCRIPT_NAME}_*.out; do
    if [[ -f "$log_file" && "$log_file" != "/home/wlia0047/ar57_scratch/wenyu/logs/${SCRIPT_NAME}_${JOB_ID}.out" ]]; then
        rm -f "$log_file"
        echo "Removed old log file: $log_file"
    fi
done  

for log_file in /home/wlia0047/ar57_scratch/wenyu/logs/${SCRIPT_NAME}_*.err; do
    if [[ -f "$log_file" && "$log_file" != "/home/wlia0047/ar57_scratch/wenyu/logs/${SCRIPT_NAME}_${JOB_ID}.err" ]]; then
        rm -f "$log_file"
        echo "Removed old error file: $log_file"
    fi
done

# Display job information
echo "=========================================="
echo "SLURM Job Information:"
echo "Script Name: $SCRIPT_NAME"
echo "Job ID: $JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

set -euo pipefail

# Move to the project directory
cd /home/wlia0047/ar57_scratch/wenyu

# Ensure logs directory exists
mkdir -p logs
# Initialize and activate conda environment (robust for non-interactive shells)
if [ -f /etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
module load cuda || true

# Activate conda environment (fail loudly if activation fails so logs show issues)
#conda activate /home/wlia0047/ar57/wenyu/wenyu-vllm
#which python3 || true
conda activate /home/wlia0047/ar57_scratch/wenyu/stark

# Run the simplified analysis script using SiliconFlow API
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_cache
# Enable unbuffered output for real-time logging
export PYTHONUNBUFFERED=1
# Use stdbuf to ensure unbuffered output from Python
stdbuf -o0 -e0 python3 /home/wlia0047/ar57/wenyu/stark/code/query_generation/generate_grammer_query.py

# Job completion information
echo ""
echo "=========================================="
echo "Job Completed Successfully!"
echo "Job ID: $JOB_ID"
echo "End Time: $(date)"
echo "=========================================="


