# AGENTS.md - Python Execution Rules

## ⚠️ CRITICAL: ALL Python Scripts Must Use sbatch_wrapper

**Never run Python scripts directly. Always use sbatch_wrapper.py:**

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && python3 script.py"
```

This ensures:
- ✅ Correct conda environment
- ✅ GPU resource allocation via SLURM
- ✅ Job tracking and logging
- ✅ Proper timeout handling
- ✅ Full integration with cluster infrastructure

---

## Critical Execution Rules

### Rule 1: ALWAYS Use sbatch_wrapper for Python Execution
- ❌ **NEVER**: `python3 script.py`
- ❌ **NEVER**: Direct Python execution
- ✅ **ALWAYS**: Use sbatch_wrapper.py wrapper

### Rule 2: Required Environment Every Time
Every sbatch_wrapper call MUST include:
1. Conda activation: `source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh`
2. Environment: `conda activate /home/wlia0047/ar57_scratch/wenyu/stark`
3. Working directory: `cd /fs04/ar57/wenyu` (or appropriate path)

### Rule 3: When to Use sbatch_wrapper
- ✅ All training/fine-tuning (Stage 14)
- ✅ All evaluation scripts (Stage 12, 13)
- ✅ All model inference
- ✅ All GPU-dependent operations
- ✅ All tests that use ML models

### Rule 4: Task Completion & Summary
- ❌ **NEVER**: Create README files or documentation after task completion
- ❌ **NEVER**: Generate documentation files
- ✅ **ALWAYS**: Print task summary directly to output
- ✅ **ALWAYS**: Summarize what was accomplished in the response

When a task is complete, provide a concise summary of:
- What was done
- Results and outcomes
- Any important information
- Print it directly without creating files

### Rule 5: Final Completion Message
- ✅ **ALWAYS**: End every completed task with the final line:
  ```
  当前任务已完成，请做下一个任务的指示。
  ```
- This must be the last sentence in your response after task completion
- Use this in all task completion summaries

---

## Example Commands

### Run a Single Test
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu/.claude/skills/PersoanlQuery/13_rerank/llm_reranking/tests && \
     python3 test_preference_classifier.py"
```

### Run Model Fine-tuning (Stage 14)
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning && \
     python3 finetune_e5_gpu.py"
```

### Run Evaluation Scripts
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python3 ./.claude/skills/PersoanlQuery/12_retrieval/scripts/verify_all_retrievers.py"
```

---

Last Updated: 2026-03-16
