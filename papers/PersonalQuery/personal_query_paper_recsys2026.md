# PersonalQuery: Grounded Personalized Search Query Generation via Language Style Alignment

## Abstract

Personalized search systems traditionally rely on user profiles or collaborative filtering, often failing to capture fine-grained preferences and communication patterns. We present PersonalQuery, a 12-stage pipeline that generates "grounded" personalized search queries by extracting fine-grained user preferences from historical reviews and aligning query language with individual writing patterns. Our approach integrates LLM-based preference extraction, per-dimension persona generation, 16-dimensional linguistic style analysis, iterative feature-aware refinement, and CNN-based spelling difficulty prediction for targeted noise injection. We also introduce PersonalQuery-Bench, the first large-scale dataset containing explicit user personas paired with personalized search queries. Experiments on Amazon product reviews demonstrate significant improvements in both semantic relevance and stylistic authenticity compared to baseline methods. Human evaluation confirms strong alignment between LLM and human judgments (Spearman's $\rho$ = 0.82). Our key methodological contribution is a bias-aware evaluation framework using Difference-in-Differences (DiD) scoring, revealing that traditional metrics systematically underestimate personalization gains by up to 48% for users with high population overlap.

**CCS Concepts**: • Information systems → Personalization; Query representation; • Computing methodologies → Natural language generation

**Keywords**: personalized search, query generation, user modeling, style transfer, large language models

---

## 1. Introduction

E-commerce search systems serve as the primary interface between users and vast product catalogs. Traditional search interfaces treat all users identically, returning the same results for the same queries regardless of individual preferences, expertise levels, or communication patterns. This one-size-fits-all approach fails to capture the rich heterogeneity in how users express needs and evaluate products.

A professional crafter searching for "3mm German glass glitter, 80 grit texture" communicates differently from a novice asking for "shiny craft glitter." Users' search queries reflect not only their information needs but also their communication style, technical sophistication, and personal preferences.

### 1.1 Research Challenges

We define **grounded personalization** as generating search queries that are:

1. **Semantically faithful** to user preferences extracted from behavioral data
2. **Stylistically authentic** to individual writing patterns (including spelling and grammar habits)
3. **Behaviorally realistic** including natural variations such as user-specific error patterns

Achieving grounded personalization requires addressing:
- **Fine-grained preference extraction**: Moving beyond coarse categories to attribute-sentiment pairs
- **Writing style analysis**: Capturing users' spelling and grammar error patterns
- **Style modeling**: Capturing linguistic patterns beyond vocabulary
- **Controllable generation**: Balancing semantic accuracy with style alignment

### 1.2 Contributions

We present PersonalQuery, addressing these challenges through:

1. **12-stage processing pipeline** transforming raw user reviews into personalized search queries with controllable style attributes

2. **Per-dimension persona generation** capturing user preferences across multiple dimensions (Quality, Use Case, Design, Price, etc.)

3. **Writing style analysis** extracting users' spelling and grammar error patterns for realistic noise injection

4. **Iterative feature-aware refinement** progressively aligning generated queries with target style profiles using 16-dimensional linguistic features

5. **Bias-aware evaluation framework** using Difference-in-Differences (DiD) scoring to isolate true personalization value from baseline similarity effects

6. **PersonalQuery-Bench dataset** — the first large-scale dataset containing explicit user personas paired with personalized search queries (~120K query pairs)

---

## 2. Related Work

**Personalized Search.** Traditional approaches focus on click-through prediction and result re-ranking [1]. Recent methods employ deep learning for user preference modeling but typically operate at the result level rather than query generation. Existing datasets (JDsearch, KuaiSearch, AOL4PS) lack explicit user persona information.

**User Preference Extraction.** Aspect-based sentiment analysis extracts preferences from text. Our approach uses prompt engineering with LLMs to extract structured attribute-sentiment pairs.

**Language Style Transfer.** Style transfer has been applied to formality adjustment and author imitation [5]. Our work uses explicit 16-dimensional feature profiles to guide iterative LLM refinement via ProfilingUD [3].

**Query Generation.** Most approaches focus on semantic expansion or clarification questions. Our work generates queries matching users' specific linguistic styles while maintaining semantic intent.

---

## 3. Method

### 3.1 System Overview

PersonalQuery implements a 12-stage modular pipeline:

```
Stage 0-1:   Data Preparation → LLM Preference Extraction
Stage 2-3:   Target vs Public Classification → Train/Test Split
Stage 4:     Per-Dimension Persona Generation
Stage 5:     Writing Style Analysis (spelling/grammar patterns)
Stage 6:     16-Dimensional Linguistic Feature Extraction
Stage 7:     Dual Query Generation (Target + Mass Market)
Stage 8:     Iterative Style Refinement
Stage 9-10:  CNN Spelling Model → Targeted Noise Injection
Stage 11-12: Multi-dimensional Evaluation → Human Alignment
```

### 3.2 Stage 0-3: Preference Extraction and Classification

**User Selection.** We select users with 100-110 product reviews, valid metadata, and minimum 5 reviews per product.

**LLM-based Extraction.** For each user-product pair, GLM-5 extracts structured preferences:

```json
{
  "target_user_preferences": {
    "Quality": [{"entity": "adhesive strength", "sentiment": "positive"}]
  },
  "public_attributes": [
    {"attribute": "price range", "sentiment": "neutral"}
  ]
}
```

**Data Split.** Products are divided into:
- **Persona Set**: For persona generation
- **Query Set**: For query generation and evaluation

### 3.3 Stage 4: Per-Dimension Persona Generation

Unlike traditional single unified personas, we generate **separate persona descriptions for each dimension**:

```
For each user:
    For each category:
        For each dimension (Quality, Use Case, Design, Price, ...):
            → LLM generates 50-80 word description
            → Output: dimension_personas = {dim1: p1, dim2: p2, ...}
```

**Key Insight**: Per-dimension personas capture nuanced preferences within each aspect, enabling more precise query generation.

### 3.4 Stage 5: Writing Style Analysis

We analyze users' historical reviews to extract **spelling and grammar error patterns**:

| Error Type | Description |
|------------|-------------|
| Homophone Confusion | 35% (e.g., "their" → "there") |
| Letter Omission | 25% |
| Letter Substitution | 20% |
| Letter Insertion | 15% |
| Position Swap | 5% |

This analysis enables **personalized noise injection** that reflects individual error patterns.

### 3.5 Stage 6: Linguistic Feature Extraction

We extract 16-dimensional features using ProfilingUD:

| Category | Features |
|----------|----------|
| Length | tokens_per_sent, char_per_tok, n_tokens |
| Lexical | ttr_lemma_chunks_100, lexical_density |
| POS Distribution | NOUN, VERB, ADJ, ADV, PRON, DET, AUX, PART, SCONJ, CCONJ, ADP |

### 3.6 Stage 7: Dual Query Generation

We generate **paired queries** for fair comparison:

**Critical Design**: Both queries use the **same shared dimensions**.

```
1. Find dimensions shared by target and public attributes
2. Randomly select 3 shared dimensions
3. Generate Target Query using target user's attribute values
4. Generate Mass Market Query using public attribute values
```

**Prompt Difference**:
- Target: "Generate a first-person search query for an Amazon shopper..."
- Mass Market: "Generate a first-person search query for a TYPICAL Amazon shopper..."

**Constraints**: 25-30 words, first-person perspective, natural language.

### 3.7 Stage 8: Iterative Style Refinement

**Round 0**: Base generation with style description
**Round 1+**: Gap-driven refinement

For each round:
1. Extract 16-dimensional features from current query
2. Compute style distance to user feature vector
3. Generate targeted instructions for top-5 largest gaps
4. Create refinement prompt with specific adjustments

**Candidate Selection**:
$$\text{score} = 0.7 \cdot d_{\text{style}} + 0.3 \cdot d_{\text{semantic}}$$

**Convergence**: Style distance improvement < 0.02, or max 5 rounds.

### 3.8 Stage 9-10: Targeted Noise Injection

**CNN Spelling Difficulty Model**:
```
Input: [char_embeddings (64-dim), handcrafted_features (50-dim), user_features (9-dim)]
  → Conv1D (128→512) → Adaptive Max Pool → FC (256→64) → Sigmoid
Output: Word-level difficulty score [0,1]
```

**Injection Process**:
1. Score difficulty for each word in query
2. Select top-K highest difficulty words
3. Inject errors according to user's error distribution (from Stage 5)
4. LLM verifies semantic preservation

---

## 4. Experiments

### 4.1 Experimental Setup

**Dataset**: Amazon Arts, Crafts & Sewing
- Users: 50 high-quality users (100-110 product reviews each)
- Products: ~5,000 with metadata
- Reviews: ~150,000

**Implementation**:
- LLM: GLM-5
- Embedding: sentence-transformers/all-MiniLM-L6-v2
- Linguistic Features: ProfilingUD with spaCy
- Spelling Model: PyTorch CNN, 50 epochs, Adam (lr=0.001)

**Baselines**:
1. Generic Query: Category + generic attributes
2. Attribute-Only: Personalized attributes without style alignment
3. Single-Turn LLM: Single generation without iteration
4. Template-Based: Fill-in-the-blank templates

### 4.2 Evaluation Framework

#### 4.2.1 LLM 5-Rule Assessment

Each query rated 1-5 on: Preference Alignment, Persona Consistency, Semantic Completeness, Naturalness, Specificity.

#### 4.2.2 Bias-Aware Evaluation: Difference-in-Differences

**Problem**: Target user profile ($P_u$) inherently overlaps with population profile ($P_g$), giving public queries a "floor score."

**Solution**: DiD scoring isolates true personalization value:

$$\text{Score}_{\text{DiD}} = [S(Q_u, P_u) - S(Q_u, P_g)] - [S(Q_g, P_u) - S(Q_g, P_g)]$$

**Properties**:
- Zero-bias: When $Q_u = Q_g$, Score = 0
- Monotonicity: If $Q_u$ captures more unique dimensions, Score > 0

### 4.3 Results

#### 4.3.1 Style Alignment

| Method | Style Distance ↓ | Semantic Sim ↑ | Convergence |
|--------|------------------|----------------|-------------|
| Generic | 0.412 | 0.65 | - |
| Attribute-Only | 0.298 | 0.78 | - |
| Single-Turn | 0.215 | 0.82 | - |
| **PersonalQuery** | **0.142** | **0.89** | **87%** |

#### 4.3.2 Iterative Refinement

| Round | Style Distance | Improvement | Convergence |
|-------|----------------|-------------|-------------|
| 0 | 0.287 | - | 0% |
| 1 | 0.198 | 0.089 | 42% |
| 2 | 0.156 | 0.042 | 71% |
| 3 | **0.142** | 0.014 | **87%** |

Most queries converge by Round 3.

#### 4.3.3 LLM 5-Rule Assessment

| Method | Preference | Persona | Semantic | Natural | Specific | Total |
|--------|------------|---------|----------|---------|----------|-------|
| Generic | 2.1 | 1.8 | 2.4 | 4.2 | 1.5 | 12.0 |
| Single-Turn | 4.1 | 3.5 | 4.0 | 3.8 | 3.6 | 19.0 |
| **PersonalQuery** | **4.6** | **4.3** | **4.4** | **4.1** | **4.2** | **21.6** |

#### 4.3.4 Bias Correction Impact

| Method | Raw Score | DiD Score | Improvement |
|--------|-----------|-----------|-------------|
| Generic | 12.0 | 0.0 | Baseline |
| Single-Turn | 19.0 | 6.8 | +6.8 |
| **PersonalQuery** | **21.6** | **11.2** | **+11.2** |

**Key Finding**: DiD increases differentiation ratio from 1.8× (raw) to 2.7×.

**User-Population Similarity Analysis**:

| Similarity | Users | Raw Gain | DiD Gain | Amplification |
|------------|-------|----------|----------|---------------|
| [0.9, 1.0] | 12 | +8.2 | +12.1 | **1.48×** |
| [0.8, 0.9) | 18 | +9.1 | +11.5 | 1.26× |

Users with high population similarity (>0.9) benefit most from bias correction.

#### 4.3.5 Human-LLM Alignment

| Metric | Value |
|--------|-------|
| Spearman's $\rho$ | 0.82 |
| Cohen's $\kappa$ | 0.71 |
| Recall@Human | 0.89 |
| MAE | 0.42 |

Strong alignment validates LLM-based evaluation.

#### 4.3.6 Noise Injection

| Metric | Targeted | Random | None |
|--------|----------|--------|------|
| Realism | 4.2 | 2.8 | 3.5 |
| Semantic Preservation | 0.94 | 0.71 | 1.0 |
| User Pattern Match | 0.87 | 0.42 | - |

### 4.4 Ablation Study

| Configuration | Style Distance | Rounds | Convergence |
|---------------|----------------|--------|-------------|
| Full Pipeline | 0.142 | 2.8 | 87% |
| Without Gap Analysis | 0.178 | 4.2 | 62% |
| Without Semantic Weight | 0.156 | 3.1 | 79% |

All components contribute to performance.

---

## 5. Discussion

### 5.1 Key Findings

1. **Per-dimension personas improve quality**: Separate personas for each dimension capture nuanced preferences better than unified descriptions.

2. **Writing style analysis is essential**: Analyzing spelling/grammar patterns enables personalized noise injection.

3. **Iterative refinement is necessary**: Single-pass generation cannot capture fine-grained style features.

4. **16 features are sufficient**: Carefully curated style features capture necessary variation without high-dimensional noise.

5. **Bias correction reveals true value**: DiD scoring shows traditional metrics underestimate personalization gains by up to 48%.

### 5.2 Limitations

1. **Domain specificity**: Evaluated only on Arts & Crafts
2. **Cold start**: Requires substantial review history
3. **LLM dependency**: Quality bounded by underlying LLM
4. **Computational cost**: 12-stage pipeline with iteration

### 5.3 Future Directions

1. Cross-domain profile transfer
2. Few-shot adaptation for users with limited history
3. Real-time deployment optimization
4. Multimodal personalization (voice, visual)

---

## 6. Conclusion

We presented PersonalQuery, a 12-stage pipeline for grounded personalized search query generation. By extracting fine-grained preferences, analyzing writing style patterns, modeling 16-dimensional linguistic features, and employing iterative feature-aware refinement, PersonalQuery generates queries that are both semantically faithful and stylistically authentic. We also released PersonalQuery-Bench, the first large-scale dataset with explicit user personas paired with personalized queries. Our bias-aware evaluation framework using DiD scoring ensures such systems are properly valued in research literature.

---

## References

[1] Bennett, P. N., & White, R. W. (2012). Modeling the large-scale dynamics of search. WWW.

[2] Brunato, D., & Dell'Orletta, F. (2017). Profiling-UD: A language-independent tool for linguistic profiling. NLPCS.

[3] Fu, Z., et al. (2018). Style transfer in text: Exploration and evaluation. AAAI.

[4] Reimers, N., & Gurevych, I. (2019). Sentence-BERT. EMNLP.

[5] Guo, Q., et al. (2021). AOL4PS: A large-scale data set for personalized search. Data Intelligence.

---

## Appendix A: PersonalQuery-Bench Dataset

### A.1 Dataset Overview

| Statistic | Value |
|-----------|-------|
| Users | 1,000+ |
| Products | 10,000+ |
| Reviews | ~150,000 |
| Query Pairs | ~120,000 |

### A.2 Data Format

```json
{
  "user_id": "u_12345",
  "category": "Arts & Crafts > Glitter",
  "dimension_personas": {
    "Quality": "This user prioritizes durability...",
    "Use Case": "She mainly purchases glitter for card making..."
  },
  "writing_style": {
    "spelling_errors": {"homophone": 0.35, "omission": 0.25, ...}
  },
  "linguistic_features": {"tokens_per_sent": 15.2, ...},
  "personalized_query": "I need 3mm German glass glitter...",
  "mass_market_query": "German glass glitter 3mm craft supplies",
  "query_with_noise": "I need 3mm German glass gliter...",
  "evaluation": {"did_score": 11.2, ...}
}
```

### A.3 Data Availability

- **Platform**: HuggingFace / GitHub
- **License**: CC BY-NC 4.0
- **Link**: [Anonymous for review]

---

## Appendix B: 16-Dimensional Linguistic Features

| Feature | Definition | Range |
|---------|------------|-------|
| tokens_per_sent | Mean tokens per sentence | [5, 50] |
| char_per_tok | Mean characters per token | [3, 10] |
| ttr_lemma_chunks_100 | Type-token ratio (100-word chunks) | [0, 1] |
| lexical_density | Content word ratio | [0, 1] |
| upos_dist_* | POS distribution (11 features) | [0, 1] |
| n_tokens | Total tokens | [25, 30] |

---

## Appendix C: DiD Evaluation Algorithm

```python
def compute_did_score(Q_u, Q_g, P_u, P_g):
    s_uu = llm_evaluate(Q_u, P_u)
    s_ug = llm_evaluate(Q_u, P_g)
    s_gu = llm_evaluate(Q_g, P_u)
    s_gg = llm_evaluate(Q_g, P_g)
    
    delta_u = s_uu - s_ug  # Personalized query's user preference
    delta_g = s_gu - s_gg  # Public query's baseline
    
    return delta_u - delta_g  # Isolated personalization value
```
