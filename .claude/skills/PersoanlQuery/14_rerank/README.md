# Stage 14: LLM Reranking Pipeline

## Overview

This stage implements LLM-based reranking methods to improve retrieval results from Stage 13 using persona-aware context.

## Directory Structure

```
14_rerank/
└── llm_reranking/            # LLM-based reranking
    ├── persona_utils.py      # Persona context builder with conflict resolution
    ├── 13_evaluate_glm_*.py  # GLM rerankers (GLM-4-5v, GLM-4-7, GLM-5)
    ├── 13_evaluate_minimax_*.py  # Minimax rerankers (M2, M2-1, M2-5)
    ├── 13_evaluate_qwen_*.py # Qwen rerankers
    └── test_*.py             # Testing scripts for thinking/reasoning modes
```

## LLM Reranking Methods

### GLM Models
- **GLM-4-5v**: Vision-language model for reranking
- **GLM-4-7**: Advanced language model with reasoning
- **GLM-5**: Latest GLM model with improved context understanding
- **GLM-5 Two-Stage**: Two-stage reranking pipeline

### Minimax Models
- **Minimax M2**: Base model for reranking
- **Minimax M2-1**: Optimized variant
- **Minimax M2-5 High-Speed**: Fast inference variant

### Qwen Models
- **Qwen-7B**: Efficient reranking model

All LLM rerankers use persona context built from user preferences (Stage 1) to improve personalization. The `persona_utils.py` module provides:
- Preference relevance classification (REQUIRED, RELEVANT, CONFLICTING, IRRELEVANT)
- Persona context building with conflict resolution
- Attribute-aware context generation

## Input

- Stage 13 retrieval candidates (e.g., BM25 candidates from `13_retrieval/result/bm25_candidates_*.json`)
- User preferences from Stage 1 (`01_preference_extraction/preferences_*.json`)
- Product metadata from Stage 2 (`02_matching/match_*.json`)
- Dual queries from Stage 7 or noisy queries from Stage 10

## Output

- Reranked results with improved metrics (Recall@K, MRR, NDCG)
- Saved to `result/personal_query/14_rerank/`
- Output format: `rerank_{model}_{query_type}_{user_id}.json`

## Usage

Each reranking method has its own evaluation script. Run them individually or use batch scripts.

Example:
```bash
# GLM-5 reranking
python llm_reranking/13_evaluate_glm_5_both.py --user-id A13OFOB1394G31

# Minimax M2-5 high-speed reranking
python llm_reranking/13_evaluate_minimax_m2_5_highspeed.py --user-id A13OFOB1394G31
```

## Persona Context

The persona context builder (`persona_utils.py`) analyzes user preferences and classifies them into:
- **REQUIRED**: Preferences directly mentioned in query
- **RELEVANT**: Related preferences from same category
- **CONFLICTING**: Preferences contradicting query intent
- **IRRELEVANT**: Unrelated preferences

This classification helps LLMs make better reranking decisions by focusing on relevant context and avoiding conflicts.
