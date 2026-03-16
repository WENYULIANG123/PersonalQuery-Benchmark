# Master Retrieval Evaluation Script

## Overview

`run_all_retrieval.py` is the master orchestration script that runs **all retrieval evaluations** (excluding reranking methods) for a given user in both clean and noisy query modes.

## What It Runs

### ✅ Included (Retrieval Methods)

**Dense Retrieval** (7 methods):
- ANCE
- BGE  
- Dense (baseline)
- E5
- MiniLM
- MPNet
- STAR

**Late Interaction** (1 method):
- ColBERT

**Sparse Retrieval** (3 methods):
- BM25
- Dirichlet
- TF-IDF

**Total**: 11 retrieval methods × 2 modes (clean/noisy) = **22 evaluations**

### ❌ Excluded (Reranking Methods)

- `llm_reranking/*` - All LLM-based rerankers (GLM, Minimax, Qwen)
- `traditional_reranking/13_evaluate_bert_reranker.py` - BERT reranker

## Usage

### Basic Usage

Run all retrieval methods for user `A13OFOB1394G31` in both clean and noisy modes:

```bash
cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/13_retrieval/evaluators
python run_all_retrieval.py --user-id A13OFOB1394G31
```

### Run Only Clean Mode

```bash
python run_all_retrieval.py --user-id A13OFOB1394G31 --mode clean
```

### Run Only Noisy Mode

```bash
python run_all_retrieval.py --user-id A13OFOB1394G31 --mode noisy
```

### Custom Query File

```bash
python run_all_retrieval.py \
  --user-id A13OFOB1394G31 \
  --query-file /path/to/custom_queries.json
```

### Custom Output Directory

```bash
python run_all_retrieval.py \
  --user-id A13OFOB1394G31 \
  --output-dir /fs04/ar57/wenyu/result/custom_output
```

### Continue on Error

By default, the script stops on the first failure. To continue running even if some scripts fail:

```bash
python run_all_retrieval.py \
  --user-id A13OFOB1394G31 \
  --continue-on-error
```

## Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--user-id` | No | `A13OFOB1394G31` | User ID to evaluate |
| `--mode` | No | `both` | Query mode: `clean`, `noisy`, or `both` |
| `--query-file` | No | Auto-generated | Path to Stage 10 query file |
| `--output-dir` | No | `/fs04/ar57/wenyu/result/personal_query/13_retrieval` | Output directory |
| `--continue-on-error` | No | `False` | Continue on script failures |

## Input Requirements

### Query File Format

The script expects a Stage 10 query file (JSON format) containing personalized queries.

**Default path**:
```
/fs04/ar57/wenyu/result/personal_query/10_targeted_noisy_query/noisy_queries_{user_id}.json
```

### Prerequisites

1. Stage 10 query file must exist
2. Product metadata, reviews, and QA data must be accessible
3. Python environment with required dependencies (PyTorch, transformers, etc.)

## Output

### File Naming Convention

Each retrieval method produces one JSON file per mode:

```
{output_dir}/retrieval_{method}_{mode}_{user_id}.json
```

**Examples**:
- `retrieval_bm25_clean_A13OFOB1394G31.json`
- `retrieval_bm25_noisy_A13OFOB1394G31.json`
- `retrieval_e5_clean_A13OFOB1394G31.json`
- `retrieval_colbert_noisy_A13OFOB1394G31.json`

### Execution Summary

The script prints a summary at the end:

```
==================================================================================
SUMMARY
==================================================================================
Total evaluations: 22/22
Succeeded: 20
Failed: 2

✓ Successful evaluations:
  - 13_evaluate_bm25 (clean)
  - 13_evaluate_bm25 (noisy)
  ...

✗ Failed evaluations:
  - 13_evaluate_colbert (noisy)
  - 13_evaluate_mpnet (clean)

==================================================================================
Results written to: /fs04/ar57/wenyu/result/personal_query/13_retrieval
==================================================================================
```

## Resource Management

### Execution Time

- **Per script**: ~2-10 minutes (depending on method and data size)
- **Total runtime**: ~2-4 hours for all 22 evaluations (if run sequentially)

### GPU Usage

- Dense retrieval and ColBERT methods use GPU if available
- Sparse methods (BM25, TF-IDF, Dirichlet) run on CPU
- Scripts automatically detect CUDA availability

### Memory Requirements

- Sparse methods: ~2-4 GB RAM
- Dense methods: ~8-16 GB GPU memory (or 16-32 GB RAM if CPU-only)

## Troubleshooting

### Query File Not Found

**Error**:
```
✗ Query file not found: /fs04/ar57/wenyu/result/personal_query/10_targeted_noisy_query/noisy_queries_A13OFOB1394G31.json
```

**Solution**:
Run Stage 10 first to generate the noisy query file, or provide a custom path with `--query-file`.

### Script Timeout

**Error**:
```
✗ 13_evaluate_colbert timed out after 1 hour (mode=noisy)
```

**Solution**:
The script has a 1-hour timeout per evaluation. If a script consistently times out, check for:
- Data loading issues
- GPU availability
- Model download failures

### Missing Dependencies

**Error**:
```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solution**:
Install required dependencies:
```bash
pip install torch transformers sentence-transformers faiss-gpu rank_bm25
```

## Adding New Retrieval Methods

To add a new retrieval method:

1. Create the evaluation script in the appropriate subdirectory:
   - `dense_retrieval/` for dense methods
   - `sparse_retrieval/` for sparse methods
   - `late_interaction/` for ColBERT-style methods

2. Add the script path to `RETRIEVAL_SCRIPTS` list in `run_all_retrieval.py`:

```python
RETRIEVAL_SCRIPTS = [
    # ... existing scripts ...
    "dense_retrieval/13_evaluate_new_method.py",  # Add here
]
```

3. Ensure the new script follows the standard CLI interface:
   - `--query-mode` (choices: clean, noisy)
   - `--query-file` (path to query JSON)
   - `--output-dir` (output directory)
   - `--user-id` (user identifier)

## Notes

- **Reranking scripts are excluded**: This script only runs retrieval methods. For reranking, use separate scripts.
- **Sequential execution**: Scripts run one at a time to avoid resource conflicts.
- **Logging**: All output is timestamped for easy debugging.
- **Exit codes**: The script returns 0 on success, 1 if any evaluations failed.
