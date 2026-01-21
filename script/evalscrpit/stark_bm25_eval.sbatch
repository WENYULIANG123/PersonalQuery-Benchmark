#!/bin/bash
#
# SBATCH script to run STaRK BM25 evaluation on Amazon dataset
# Minimal, single copy — cleaned to remove duplicated blocks.
#
# Job name and output
#SBATCH --job-name=stark_bm25_eval
#SBATCH --output=/home/wlia0047/ar57_scratch/wenyu/logs/%x_%j.out
#SBATCH --error=/home/wlia0047/ar57_scratch/wenyu/logs/%x_%j.err

# Resources (optimized for multi-CPU parallel processing)
#SBATCH --time=04:00:00
#SBATCH --partition=comp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
SCRIPT_NAME=stark_bm25_eval  

# Get current SLURM job ID for tracking and logging
JOB_ID=${SLURM_JOB_ID:-"unknown"}


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
# conda activate /home/wlia0047/ar57_scratch/wenyu/stark

# # Move to the project directory
# cd /home/wlia0047/ar57_scratch/wenyu

# # Ensure logs directory exists
# mkdir -p logs
# Check if conda is available

# Initialize and activate conda environment (robust for non-interactive shells)
echo "Initializing conda environment..."
if [ -f /etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  echo "Sourcing /etc/profile.d/conda.sh"
  source /etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
  echo "Using conda shell hook"
  eval "$(conda shell.bash hook)"
fi

conda activate /home/wlia0047/ar57_scratch/wenyu/stark


# Load CUDA module (commented out for CPU usage)
echo "CUDA module loading skipped for CPU usage..."
# module load cuda || echo "Warning: CUDA module load failed"

# Activate conda environment (fail loudly if activation fails so logs show issues)
# echo "Activating conda environment: /home/wlia0047/ar57_scratch/wenyu/stark"
conda activate /home/wlia0047/ar57_scratch/wenyu/stark

# Verify environment activation
echo "Python executable: $(which python3)"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-not_activated}"

# Check if script exists and set executable permissions
# Set HuggingFace cache directory
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_cache
echo "HF_HOME set to: $HF_HOME"

# echo "=== STARTING PYTHON SCRIPT ==="
# echo "Current directory before Python execution: $(pwd)"
# echo "Python3 executable: $(which python3)"
# echo "Python3 version: $(python3 --version 2>&1)"
# echo "PATH: $PATH"
# echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not_set}"

# # Test if we can import basic modules
# echo "Testing basic Python import..."
# python3 -c "import sys; print(f'Python executable: {sys.executable}'); import os; print(f'Current dir: {os.getcwd()}')"

# # Run the evaluation script using the script's internal hard-coded configuration.
# No command-line arguments or environment exports for INPUT/OUTPUT/DEVICE will be used.
echo "Launching Python script..."
# Force unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

cd /home/wlia0047/ar57/wenyu/stark/code
exec python3 /home/wlia0047/ar57/wenyu/stark/code/retrieval/eval/run_stark_bm25_eval.py
#exec python3 /home/wlia0047/ar57/wenyu/stark/code/retrieval/eval/run_stark_bm25_eval.py  --strategy grammar_aware

