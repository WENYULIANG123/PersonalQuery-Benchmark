# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Personal Query** - A "Grounded" personalized query generation system that extracts fine-grained user preferences from Amazon reviews and generates realistic user personas with personalized search queries.

The system implements an 11-stage pipeline that transforms raw user reviews into highly differentiated personas and search queries, with multi-dimensional evaluation.

## Environment Setup

### Conda Environment
```bash
# Environment is auto-activated via direnv (.envrc)
# Manual activation:
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/stark
```

### Python Path
```bash
export PYTHONPATH="/home/wlia0047/ar57/wenyu/stark/code:$PYTHONPATH"
```

## Running Code on the Cluster

**All Python scripts must be run through `sbatch_wrapper.py`** to submit jobs to the SLURM cluster:

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u /path/to/script.py \
     --arg1 value1 \
     --arg2 value2"
```

The `sbatch_wrapper.py` hook prevents accidental local execution of long-running scripts.

## LLM Client

A shared LLM client is available at `.claude/skills/llm_client.py`:

```python
from .claude.skills.llm_client import LLMClient

client = LLMClient()
response = client.call(prompt, max_tokens=4096, temperature=0.7)
```

- Uses Anthropic SDK with GLM-4.5-Air (can switch to GLM-5)
- Base URL: `https://api.z.ai/api/anthropic`
- Includes automatic retry with exponential backoff for rate limiting (429 errors)

## Core Pipeline: PersoanlQuery Skill

The main pipeline is implemented as a Claude Code skill at `.claude/skills/PersoanlQuery/` with 11 stages:

| Stage | Name | Purpose |
|-------|------|---------|
| 0 | Data Preparation | Extract user reviews and preferences |
| 1 | Matching | Extract entities and validate against metadata |
| 2 | Processing | Select users and split train/holdout sets |
| 3 | Persona | Generate grounded user personas with category-aware analysis |
| 4 | Writing Analysis | Analyze spelling/grammar error patterns |
| 5 | Syntactic Analysis | Extract linguistic features (spaCy, 18-dim) |
| 6 | Dual Query | Generate Public vs Personalized queries |
| 7 | Iterative Refinement | Align query style to user writing (GLM-5) |
| 8 | Spelling Difficulty | Train joint spelling difficulty model |
| 9 | Noisy Query | Inject targeted noise based on user error patterns |
| 10 | Evaluation | Multi-dimensional evaluation (LLM + semantic + diversity) |
| 11 | Human Evaluation | Human-LLM alignment metrics |

**Key Principle**: "Grounded Strategy" - avoid generic terms like "high quality", focus on specific technical specs and use cases. Strict holdout separation ensures generalization.

### Running Pipeline Stages

Each stage script is in `.claude/skills/PersoanlQuery/{XX_stage_name}/`. Use the `sbatch_wrapper.py` pattern shown above.

Example:
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u .claude/skills/PersoanlQuery/03_persona/03_generate_persona_prompts.py \
     --input-dir /fs04/ar57/wenyu/result/personal_query/01_matching/results \
     --meta-file /fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
     --output-dir /fs04/ar57/wenyu/result/personal_query/persona_prompts"
```

## Data Structure

### Input Data
- `/data/Amazon-Reviews-2018/raw/` - Raw Amazon review data
- `/data/Amazon-Reviews-2018/raw/meta_*.json.gz` - Product metadata

### Output Data
- `/result/personal_query/` - All pipeline outputs organized by stage
  - `00_data_preparation/` - Extracted preferences
  - `01_matching/results/` - Validated attributes
  - `02_processing/` - Train/holdout splits
  - `03_persona/results/` - Generated personas
  - `06_query/` - Dual queries (public/personalized)
  - `07_iterative_refinement/` - Style-aligned queries
  - `10_evaluation/` - Evaluation results

## Code Architecture

### `stark/code/` Directory Structure
- `analysis/` - Analysis utilities
- `entity_extraction/` - Entity extraction from reviews
- `entity_matching/` - Match entities to product metadata
- `model.py` - Core model definitions
- `query_generation/` - Query generation logic
- `retrieval/` - Retrieval components
- `train_model/` - Model training scripts
- `user_profile/` - User profile utilities

### Legacy Skills
- `.claude/skills/user_profile/` - 9-stage version (deprecated, use PersoanlQuery)
- Various other skills for product/user analysis tasks

## Common Tasks

### Monitor Running Jobs
```bash
squeue -u $USER
```

### Check Job Logs
Logs are written to `logs/` directory with timestamp and job ID.

### Debug Hook Issues
The `sbatch_wrapper.py` hook outputs debug info to stderr:
```
[sbatch_wrapper] Hook被调用 - is_hook_mode: True, stdin_isatty: False
```

If hook is not triggered, use the wrapper explicitly as shown above.
