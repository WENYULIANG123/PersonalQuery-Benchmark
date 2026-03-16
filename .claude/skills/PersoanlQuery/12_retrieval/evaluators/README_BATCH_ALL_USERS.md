# Stage 12 Batch Evaluation Script

## Overview

The `12_evaluate_all_users_retrieval.py` script automatically runs all retrieval evaluations for all users who have completed Stage 6 (query generation).

## Features

- **Automatic User Discovery**: Finds all users with Stage 6 queries
- **Comprehensive Evaluation**: Runs all 11 retrieval methods (excluding reranking)
- **Dual Mode Support**: Evaluates both clean and noisy queries
- **Error Handling**: Option to continue processing even if some users fail
- **Progress Tracking**: Real-time logging with detailed progress information
- **Summary Report**: Generates a comprehensive summary of all evaluations

## Usage

### Basic Usage (Recommended)

Run all retrieval evaluations for all users:

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval/evaluators && \
     python 12_evaluate_all_users_retrieval.py"
```

### Advanced Options

#### Run Only Clean Mode
```bash
python 12_evaluate_all_users_retrieval.py --mode clean
```

#### Run Only Noisy Mode
```bash
python 12_evaluate_all_users_retrieval.py --mode noisy
```

#### Continue on Error
Continue processing remaining users even if some fail:
```bash
python 12_evaluate_all_users_retrieval.py --continue-on-error
```

#### Process Specific Users
```bash
python 12_evaluate_all_users_retrieval.py --user-ids A13OFOB1394G31 A1GYEGLX3P2Y7P
```

## What Gets Evaluated

For each user, the script runs:

### Dense Retrieval Methods (7)
- ANCE
- BGE
- Dense (MiniLM)
- E5
- MiniLM
- MPNet  
- STAR

### Late Interaction (1)
- ColBERT

### Sparse Retrieval (3)
- BM25
- Dirichlet
- TF-IDF

**Total**: 11 methods × 2 modes (clean/noisy) = 22 evaluations per user

## Output

### Directory Structure
```
/fs04/ar57/wenyu/result/personal_query/12_retrieval/
├── retrieval_bm25_clean_A13OFOB1394G31.json
├── retrieval_bm25_noisy_A13OFOB1394G31.json
├── retrieval_e5_clean_A13OFOB1394G31.json
├── retrieval_e5_noisy_A13OFOB1394G31.json
├── ... (more evaluation results)
├── impact_ranking_A13OFOB1394G31.txt        # Per-user impact analysis
└── all_users_evaluation_summary.json         # Overall summary
```

### Summary Report
The `all_users_evaluation_summary.json` contains:
- List of successfully processed users
- List of failed users (if any)
- List of skipped users (missing Stage 9 data)
- Total processing time

### Log File
Detailed logs are saved to: `/home/wlia0047/ar57/wenyu/stage12_all_users_batch.log`

## Prerequisites

Users must have completed:
- **Stage 6**: Query generation (`dual_queries_*.json`)
- **Stage 9**: Noisy query generation (`noisy_queries_*.json`)

Users without Stage 9 data will be automatically skipped.

## Expected Runtime

- **Per user**: 2-4 hours (22 evaluations)
- **11 users**: ~22-44 hours total

Consider using `tmux` or `screen` for long-running evaluations:

```bash
tmux new -s stage12_eval
# Run the command inside tmux
# Detach with Ctrl-B then D
# Reattach with: tmux attach -t stage12_eval
```

## Error Recovery

If the script fails:

1. Check the log file for error details
2. The summary report shows which users completed successfully
3. Use `--user-ids` to re-run only the failed users
4. Use `--continue-on-error` to skip problematic users

## Example Commands

### Process All Users (Both Modes)
```bash
cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval/evaluators
python 12_evaluate_all_users_retrieval.py
```

### Process Failed Users Only
```bash
python 12_evaluate_all_users_retrieval.py --user-ids A1GYEGLX3P2Y7P A2MNB77YGJ3CN0 --continue-on-error
```

### Quick Test with One User
```bash
python 12_evaluate_all_users_retrieval.py --user-ids A13OFOB1394G31 --mode clean
```