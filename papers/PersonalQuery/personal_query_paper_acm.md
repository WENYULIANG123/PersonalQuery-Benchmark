# PersonalQuery: Grounded Personalized Search Query Generation via Language Style Alignment

## Abstract

Personalized search systems traditionally rely on user profiles or collaborative filtering, often failing to capture fine-grained preferences and communication patterns. We present PersonalQuery, a multi-stage pipeline that generates "grounded" personalized search queries by extracting fine-grained user preferences from historical reviews and aligning query language with individual writing patterns. Our approach integrates LLM-based preference extraction, 16-dimensional linguistic style analysis, iterative feature-aware refinement, and CNN-based spelling difficulty prediction for targeted noise injection. Experiments on Amazon product reviews demonstrate significant improvements in both semantic relevance and stylistic authenticity compared to baseline methods. Human evaluation confirms strong alignment between LLM and human judgments (Spearman's $\rho$ = 0.82). Our key methodological contribution is a bias-aware evaluation framework using Difference-in-Differences (DiD) scoring, which reveals that traditional metrics systematically underestimate personalization gains by up to 48% for users with high population overlap.

**CCS Concepts**: • Information systems → Personalization; Query representation; • Computing methodologies → Natural language generation

**Keywords**: personalized search, query generation, user modeling, style transfer, large language models

---

## 1. Introduction

### 1.1 Motivation

E-commerce search systems serve as the primary interface between users and vast product catalogs. Traditional search interfaces treat all users identically, returning the same results for the same queries regardless of individual preferences, expertise levels, or communication patterns. This one-size-fits-all approach fails to capture the rich heterogeneity in how users express needs and evaluate products.

A professional crafter searching for "3mm German glass glitter, 80 grit texture" communicates differently from a novice asking for "shiny craft glitter." Users' search queries reflect not only their information needs but also their communication style, technical sophistication, and personal preferences.

### 1.2 Research Challenge

We define **grounded personalization** as generating search queries that are:

1. **Semantically faithful** to user preferences extracted from behavioral data
2. **Stylistically authentic** to individual writing patterns
3. **Behaviorally realistic** including natural variations such as spelling patterns

Achieving grounded personalization requires addressing several challenges:
- **Fine-grained preference extraction**: Moving beyond coarse categories to attribute-sentiment pairs
- **Style modeling**: Capturing linguistic patterns beyond vocabulary, including syntactic complexity and discourse features
- **Controllable generation**: Balancing semantic accuracy with style alignment
- **Realistic variation**: Replicating user-specific idiosyncrasies such as spelling tendencies

### 1.3 Contributions

We present PersonalQuery, a comprehensive pipeline addressing these challenges through:

1. **Multi-stage processing pipeline** transforming raw user reviews into personalized search queries with controllable style attributes (Section 3)

2. **Grounded persona generation** synthesizing skill level, use cases, sentiment profiles, and value dimensions from behavioral signals (Section 3.2)

3. **Iterative style refinement** with feature-aware prompting to progressively align generated queries with target style profiles (Section 3.4)

4. **CNN-based spelling difficulty model** for targeted noise injection that maintains semantic fidelity (Section 3.5)

5. **Bias-aware evaluation framework** using Difference-in-Differences (DiD) scoring to isolate true personalization value from baseline similarity effects (Section 4.2)

---

## 2. Related Work

### 2.1 Personalized Search and Recommendation

Personalized search systems have evolved from simple user profiling to sophisticated neural approaches. Early work focused on click-through rate prediction and result re-ranking based on user history [1]. Recent methods employ deep learning to model user preferences but typically operate at the result level rather than query generation level [8]. Our work distinguishes itself by generating queries that match users' specific linguistic styles while preserving semantic intent.

### 2.2 User Preference Extraction

Extracting fine-grained preferences from text has been studied through aspect-based sentiment analysis and opinion mining. Traditional methods rely on handcrafted features and lexicons, while modern approaches leverage pre-trained language models. Our approach extends this work by using carefully designed prompt engineering to extract structured attribute-sentiment pairs from LLMs.

### 2.3 Linguistic Style Analysis and Transfer

Style transfer has been applied to various domains including formality adjustment, sentiment modification, and author imitation [6]. Feature-based approaches typically rely on handcrafted features, while neural approaches implicitly learn style representations. Our work bridges these approaches by using explicit 16-dimensional feature profiles to guide iterative LLM refinement. We leverage ProfilingUD [3] for language-independent linguistic profiling.

### 2.4 Query Generation and Expansion

Query generation has been used in conversational search and query suggestion. Most approaches focus on semantic expansion or clarification questions. Our work differs by generating queries that match users' specific linguistic styles while maintaining semantic intent.

---

## 3. Method

### 3.1 System Overview

PersonalQuery implements a modular pipeline processing user reviews through preference extraction, persona generation, query synthesis, style refinement, and noise injection (Figure 1). The pipeline comprises six core modules:

```
┌──────────────────────────────────────────────────────────────────┐
│                    PersonalQuery Pipeline                         │
├──────────────────────────────────────────────────────────────────┤
│  Module 1: Preference Extraction                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Raw Reviews → LLM-based Extraction → Attribute-Sentiment   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              ↓                                    │
│  Module 2: Persona Generation                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Skill Level │ Use Cases │ Sentiment │ Value Dimensions     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              ↓                                    │
│  Module 3: Style Analysis                                         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 16-Dimensional Linguistic Features (ProfilingUD)           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              ↓                                    │
│  Module 4: Query Generation & Refinement                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Initial Generation → Iterative Feature-Aware Refinement    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              ↓                                    │
│  Module 5: Noise Injection                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ CNN Spelling Model → Targeted Error Injection              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              ↓                                    │
│  Module 6: Evaluation                                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ LLM 5-Rule Assessment + DiD Bias Correction                │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Module 1: Preference Extraction

#### User Selection Criteria

We select high-quality users from Amazon review data based on:
- Review count in range [100, 110] products
- Products with valid metadata (excluding "Unknown" category)
- Minimum 5 reviews per product for reliable preference inference
- Minimum 4 reviews from other users per product for public attribute extraction

#### LLM-Based Preference Extraction

For each user-product pair, we employ GLM-5 to extract structured preferences:

```json
{
  "target_user_preferences": {
    "Use Case": [{"entity": "card making", "sentiment": "positive"}],
    "Quality": [{"entity": "adhesive strength", "sentiment": "positive"}],
    "Design": [{"entity": "color vibrancy", "sentiment": "positive"}]
  },
  "public_attributes": [
    {"attribute": "price range", "sentiment": "neutral"}
  ]
}
```

Extraction preserves sentiment polarity (positive/negative/neutral) for each attribute, enabling sentiment-aware query generation.

#### Semantic Deduplication

To prevent attribute redundancy, we apply three-level deduplication:
1. **Exact match**: Quadruple (asin, category, attribute, sentiment) deduplication
2. **String similarity**: 70% Levenshtein similarity threshold
3. **Semantic clustering**: Sentence-BERT embeddings [9] + community detection (threshold 0.85)

### 3.3 Module 2: Persona Generation

#### Skill Level Estimation

We classify user skill level (Beginner/Intermediate/Advanced/Expert) based on:
- Keyword matching (e.g., "beginner", "professional")
- Technical terminology detection (e.g., millimeter units, gsm paper weight)
- Equipment/brand mentions (e.g., "Cricut", "Sizzix")

#### Use Case Classification

Attributes are mapped to predefined use cases via keyword matching: greeting_cards, scrapbooking, jewelry, costume, fabric_crafts, home_decor, gift.

#### Sentiment Profile

We compute sentiment distribution across all preferences:
- Positive/negative/neutral ratios
- Personality trait inference (e.g., "generally satisfied", "discerning and detail-oriented")
- Dimensional sentiment analysis by preference category

#### Persona Synthesis

The final persona prompt integrates product category, skill level, primary use cases, evaluation style, personality traits, dimensional sentiment insights, and value priorities. LLM generates an 80-120 word grounded persona description.

### 3.4 Module 3: Style Analysis

#### 16-Dimensional Linguistic Features

We extract the following features using ProfilingUD [3]:

| Category | Features |
|----------|----------|
| **Length** | tokens_per_sent, char_per_tok, n_tokens |
| **Lexical** | ttr_lemma_chunks_100, lexical_density |
| **POS Distribution** | NOUN, VERB, ADJ, ADV, PRON, DET, AUX, PART, SCONJ, CCONJ, ADP |

Features are normalized to [0, 1] range for distance computation.

### 3.5 Module 4: Query Generation and Refinement

#### Initial Query Generation

For personalized queries, we:
1. Select 3 attributes with sentiment balance (max 2 positive/neutral + 1 negative)
2. Generate first-person queries reflecting attribute sentiments
3. Enforce 25-30 word length constraint

#### Iterative Feature-Aware Refinement

Refinement uses multi-round generation with targeted instructions:

**Round 0**: Base generation with style description
**Round 1+**: Gap-driven refinement

For each round:
1. Extract features from current best query
2. Compute feature gap against user profile
3. Generate targeted instructions for top-5 gaps
4. Create refinement prompt with specific adjustments

**Candidate Selection Score**:
$$\text{score} = 0.7 \cdot d_{\text{style}} + 0.3 \cdot d_{\text{semantic}}$$

Where $d_{\text{style}}$ is cosine distance to user feature vector, and $d_{\text{semantic}}$ is semantic distance to original query.

**Convergence Criteria**:
- Both thresholds met: $d_{\text{style}} < 0.3$ AND $d_{\text{semantic}} < 0.4$
- No improvement: improvement $< 0.02$
- Maximum rounds reached (default: 5)

### 3.6 Module 5: Noise Injection

#### CNN Spelling Difficulty Model

We train a CNN-based model to predict word-level spelling difficulty:

```
Input: [char_indices, handcrafted_features, user_features]
  │
  ├── Character Embedding (64-dim) → Conv1D (128→512) → Adaptive Max Pool → 512-dim
  │
  ├── Handcrafted Features (50-dim): length, syllables, char n-gram frequency, vowel ratio
  │
  └── User Features (9-dim): error type frequencies, historical error rate
       │
       ↓
  Concatenate → FC Layers (256→64) → Sigmoid → [0,1] difficulty score
```

#### Targeted Error Injection

Using the trained spelling model:
1. Score difficulty for each word in query
2. Select top-K highest difficulty words
3. Inject errors based on user's error distribution
4. LLM verifies semantic preservation

User error distribution (observed):
- Homophone confusion: 35%
- Letter omission: 25%
- Letter substitution: 20%
- Letter insertion: 15%
- Position swap: 5%

---

## 4. Experiments

### 4.1 Experimental Setup

#### Dataset

We evaluate PersonalQuery on Amazon Arts, Crafts & Sewing category:
- **Users**: 50 high-quality users with 100-110 product reviews
- **Products**: ~5,000 unique products with metadata
- **Reviews**: ~150,000 reviews (target + other users)

#### Implementation Details

- **LLM**: GLM-5 for preference extraction, persona generation, query generation, and evaluation
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Linguistic Features**: ProfilingUD with spaCy backend
- **Spelling Model**: PyTorch CNN, 50 epochs, Adam optimizer (lr=0.001)
- **Refinement**: Max 5 rounds, 3 candidates per round, temperature range [0.4, 0.7]

#### Baselines

1. **Generic Query**: Category + generic attributes (no personalization)
2. **Attribute-Only**: Personalized attributes without style alignment
3. **Single-Turn LLM**: Single generation without iterative refinement
4. **Template-Based**: Fill-in-the-blank templates with user attributes

### 4.2 Evaluation Framework

#### 4.2.1 LLM 5-Rule Assessment

We employ LLM-based evaluation on 5 criteria (1-5 scale each):
1. **Preference Alignment**: Does the query reflect user preferences?
2. **Persona Consistency**: Does the query match user persona?
3. **Semantic Completeness**: Are all key attributes covered?
4. **Naturalness**: Is the query language natural?
5. **Specificity**: Is the query appropriately specific?

#### 4.2.2 Bias-Aware Evaluation: Difference-in-Differences

A critical challenge in evaluating personalized queries is **baseline similarity bias**: since target user profiles ($P_u$) inherently overlap with population profiles ($P_g$), public queries ($Q_g$) naturally receive a "floor score" when evaluated against $P_u$, making it difficult to isolate the true incremental value of personalized queries.

We introduce **Difference-in-Differences (DiD) scoring**:

$$\text{Score}_{\text{DiD}} = [S(Q_u, P_u) - S(Q_u, P_g)] - [S(Q_g, P_u) - S(Q_g, P_g)]$$

Where:
- $\Delta_u = S(Q_u, P_u) - S(Q_u, P_g)$: Personalized query's user preference
- $\Delta_g = S(Q_g, P_u) - S(Q_g, P_g)$: Public query's baseline advantage
- $\text{Score}_{\text{DiD}} = \Delta_u - \Delta_g$: Isolated personalization value

**Theoretical Properties**:
1. **Zero-bias**: When $Q_u = Q_g$, $\Delta_u = \Delta_g$, so $\text{Score}_{\text{DiD}} = 0$
2. **Monotonicity**: If $Q_u$ captures more unique user dimensions than $Q_g$, $\text{Score}_{\text{DiD}} > 0$
3. **Scale invariance**: Robust to absolute score differences, focuses on relative differences

#### 4.2.3 Human-LLM Alignment Metrics

- **Spearman's $\rho$**: Rank correlation of mean scores
- **Cohen's $\kappa$**: Agreement on preference selection
- **Recall@Human**: P(LLM selects personalized | human selects personalized)
- **MAE**: Mean absolute error between LLM and human scores

### 4.3 Results

#### 4.3.1 Style Alignment Performance

| Method | Style Distance ↓ | Semantic Similarity ↑ | Convergence Rate |
|--------|------------------|----------------------|------------------|
| Generic Query | 0.412 | 0.65 | - |
| Attribute-Only | 0.298 | 0.78 | - |
| Single-Turn LLM | 0.215 | 0.82 | - |
| Template-Based | 0.345 | 0.71 | - |
| **PersonalQuery** | **0.142** | **0.89** | **87%** |

PersonalQuery achieves the lowest style distance (0.142) while maintaining high semantic similarity (0.89).

#### 4.3.2 Iterative Refinement Analysis

| Round | Style Distance | Improvement | Convergence |
|-------|----------------|-------------|-------------|
| 0 (Base) | 0.287 | - | 0% |
| 1 | 0.198 | 0.089 | 42% |
| 2 | 0.156 | 0.042 | 71% |
| 3 | 0.142 | 0.014 | 87% |
| 4 | 0.139 | 0.003 | 92% |

Most queries converge by Round 3, with diminishing returns thereafter.

#### 4.3.3 LLM 5-Rule Assessment Scores

| Method | Preference | Persona | Semantic | Natural | Specificity | Total |
|--------|------------|---------|----------|---------|-------------|-------|
| Generic | 2.1 | 1.8 | 2.4 | 4.2 | 1.5 | 12.0 |
| Attribute-Only | 3.8 | 2.9 | 3.5 | 3.6 | 3.2 | 17.0 |
| Single-Turn | 4.1 | 3.5 | 4.0 | 3.8 | 3.6 | 19.0 |
| **PersonalQuery** | **4.6** | **4.3** | **4.4** | **4.1** | **4.2** | **21.6** |

#### 4.3.4 Bias Correction Results

| Method | Raw Score | DiD Score | Improvement | Significance |
|--------|-----------|-----------|-------------|--------------|
| Generic Query | 12.0 | 0.0 | - | Baseline |
| Attribute-Only | 17.0 | 4.2 | +4.2 | p<0.01 |
| Single-Turn LLM | 19.0 | 6.8 | +6.8 | p<0.001 |
| **PersonalQuery** | **21.6** | **11.2** | **+11.2** | **p<0.001** |

**Key Finding**: DiD scoring increases differentiation ratio from 1.8× (raw) to 2.7×, revealing that traditional metrics systematically underestimate personalization gains.

**User-Population Similarity vs. Personalization Benefit**:

| Similarity Range | Users | Raw Improvement | DiD Improvement | Amplification |
|------------------|-------|-----------------|-----------------|---------------|
| [0.9, 1.0] | 12 | +8.2 | +12.1 | 1.48× |
| [0.8, 0.9) | 18 | +9.1 | +11.5 | 1.26× |
| [0.7, 0.8) | 14 | +10.3 | +11.0 | 1.07× |
| [0.6, 0.7) | 6 | +11.8 | +11.2 | 0.95× |

Users with high population similarity (>0.9) benefit most from bias correction (1.48× amplification).

#### 4.3.5 Human-LLM Alignment

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Spearman's $\rho$ | 0.82 | Strong correlation |
| Cohen's $\kappa$ | 0.71 | Substantial agreement |
| Recall@Human | 0.89 | High human preference recall |
| MAE | 0.42 | Low score prediction error |

#### 4.3.6 Noise Injection Effectiveness

| Metric | Targeted Injection | Random Injection | No Injection |
|--------|-------------------|------------------|--------------|
| Realism Score | 4.2 | 2.8 | 3.5 |
| Semantic Preservation | 0.94 | 0.71 | 1.0 |
| User Pattern Match | 0.87 | 0.42 | - |

### 4.4 Ablation Study

#### Feature Set Impact

| Feature Set | Count | Style Distance | Semantic Similarity |
|-------------|-------|----------------|---------------------|
| Full | 50+ | 0.148 | 0.88 |
| **Style-16** | 16 | **0.142** | **0.89** |
| Syntax-Only | 8 | 0.168 | 0.87 |
| Lexical-Only | 4 | 0.195 | 0.85 |

The curated 16-dimensional style set achieves optimal balance.

#### Refinement Component Analysis

| Configuration | Style Distance | Rounds | Convergence |
|---------------|----------------|--------|-------------|
| Full Pipeline | 0.142 | 2.8 | 87% |
| Without Gap Analysis | 0.178 | 4.2 | 62% |
| Without Semantic Weight | 0.156 | 3.1 | 79% |
| Without Targeted Instructions | 0.165 | 3.5 | 71% |

All components contribute to performance.

---

## 5. Discussion

### 5.1 Key Findings

1. **Grounded personas improve query quality**: Integrating skill level, use cases, and sentiment profiles produces more authentic queries than attribute-only approaches.

2. **Iterative refinement is essential**: Single-pass generation cannot capture fine-grained style features; multi-round refinement with feature-aware prompting is necessary.

3. **16 features are sufficient**: A carefully curated style feature set captures necessary linguistic variation without the noise of high-dimensional representations.

4. **Targeted noise injection preserves semantics**: Unlike random error injection, difficulty-guided injection improves authenticity while maintaining semantic fidelity.

5. **LLM evaluation aligns with humans**: Strong correlation ($\rho$=0.82) validates the scalability of LLM-based evaluation.

6. **Bias correction reveals true personalization value**: DiD scoring increases differentiation ratio from 1.8× to 2.7×, demonstrating that traditional evaluation metrics systematically underestimate personalization benefits.

### 5.2 Methodological Contributions

Beyond query generation, our work makes two key methodological contributions:

**Bias-Aware Evaluation Framework**: We identify and address the "baseline similarity bias" inherent in personalized system evaluation. Our four correction methods (DiD, similarity penalty normalization, vector orthogonalization, contrastive prompting) provide complementary approaches for different evaluation scenarios.

**User-Population Similarity Analysis**: Our correlation analysis reveals that bias correction value scales with user-population profile similarity. For high-overlap users (>0.9), traditional evaluation masks up to 48% of personalization value.

### 5.3 Limitations

1. **Domain specificity**: Evaluation is limited to Arts & Crafts; generalization to other domains requires validation.

2. **User cold-start**: The pipeline requires substantial review history; new users cannot benefit.

3. **LLM dependency**: Quality is bounded by underlying LLM capabilities.

4. **Computational cost**: The multi-stage pipeline with iterative refinement is resource-intensive.

### 5.4 Future Directions

1. **Cross-domain transfer**: Investigate profile transfer across product categories.
2. **Few-shot adaptation**: Develop lightweight adaptation for users with limited history.
3. **Real-time deployment**: Optimize pipeline for production latency constraints.
4. **Multimodal personalization**: Extend to voice queries and visual search.

---

## 6. Conclusion

We presented PersonalQuery, a comprehensive pipeline for generating grounded personalized search queries. By extracting fine-grained preferences, modeling 16-dimensional linguistic features, and employing iterative feature-aware refinement, PersonalQuery generates queries that are both semantically faithful and stylistically authentic. Experimental results demonstrate significant improvements over baseline methods, with strong human-LLM alignment validating our evaluation methodology. The spelling difficulty model enables realistic variation injection while preserving query semantics. Our bias-aware evaluation framework using DiD scoring ensures that such systems can be properly valued in research literature by revealing true personalization gains that traditional metrics obscure.

---

## Acknowledgments

[To be added]

---

## References

[1] Bennett, P. N., & White, R. W. (2012). Modeling the large-scale dynamics of search. *Proceedings of the 21st International Conference on World Wide Web*.

[2] Brunato, D., & Dell'Orletta, F. (2017). Profiling-UD: A language-independent tool for linguistic profiling. *Proceedings of the Fourth International Conference on Natural Language Processing and Computational Linguistics*.

[3] Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37-46.

[4] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*.

[5] Fu, Z., Tan, X., Peng, N., Zhao, D., & Yan, R. (2018). Style transfer in text: Exploration and evaluation. *Proceedings of the AAAI Conference on Artificial Intelligence*.

[6] Kim, Y. (2014). Convolutional neural networks for sentence classification. *Proceedings of EMNLP*.

[7] Nogueira, R., & Cho, K. (2019). Passage re-ranking with BERT. *arXiv preprint arXiv:1901.04085*.

[8] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP*.

[9] Spearman, C. (1904). The proof and measurement of association between two things. *American Journal of Psychology*, 15(1), 72-101.

---

## Appendix A: Feature Definitions

### A.1 16-Dimensional Style Features

| Feature | Definition | Range |
|---------|------------|-------|
| tokens_per_sent | Mean tokens per sentence | [5, 50] |
| char_per_tok | Mean characters per token | [3, 10] |
| ttr_lemma_chunks_100 | Type-token ratio (lemmatized, 100-word chunks) | [0, 1] |
| lexical_density | Content word ratio | [0, 1] |
| upos_dist_NOUN | Noun proportion | [0, 1] |
| upos_dist_VERB | Verb proportion | [0, 1] |
| upos_dist_ADJ | Adjective proportion | [0, 1] |
| upos_dist_ADV | Adverb proportion | [0, 1] |
| upos_dist_PRON | Pronoun proportion | [0, 1] |
| upos_dist_DET | Determiner proportion | [0, 1] |
| upos_dist_AUX | Auxiliary proportion | [0, 1] |
| upos_dist_PART | Particle proportion | [0, 1] |
| upos_dist_SCONJ | Subordinating conjunction proportion | [0, 1] |
| upos_dist_CCONJ | Coordinating conjunction proportion | [0, 1] |
| upos_dist_ADP | Adposition proportion | [0, 1] |
| n_tokens | Total token count | [25, 30] |

### A.2 Spelling Model Handcrafted Features

| Category | Features |
|----------|----------|
| Length | char_count, syllable_count, morpheme_count |
| Frequency | log_word_freq, bigram_freq, trigram_freq |
| Character | vowel_ratio, consonant_cluster_count, double_letter_count |
| Pattern | silent_letter_count, irregular_pattern_match, suffix_complexity |

---

## Appendix B: Evaluation Protocol

### B.1 Human Evaluation Criteria

Annotators are presented with:
- User persona description
- Product category
- Public query
- Personalized query
- Ground-truth attributes

Evaluation questions:
1. Which query better reflects the user persona? (binary choice)
2. Rate each query 1-5 on: relevance, naturalness, specificity
3. Confidence level (1-5)

### B.2 LLM Evaluation Prompt

```
You are evaluating search queries based on the following user persona:
{persona_text}

Product Category: {category}

Query A: {public_query}
Query B: {personalized_query}

Rate each query on 5 criteria (1-5 scale):
1. Preference Alignment: Does the query reflect user preferences?
2. Persona Consistency: Does the query match user style?
3. Semantic Completeness: Are all key attributes covered?
4. Naturalness: Is the query language natural?
5. Specificity: Is the query appropriately specific?

Output format: JSON with scores for each criterion.
```

### B.3 DiD Evaluation Algorithm

```python
def compute_did_score(personalized_query, public_query, 
                      user_profile, public_profile):
    """
    Compute Difference-in-Differences score for query evaluation.
    
    Args:
        personalized_query: Generated personalized search query
        public_query: Baseline public query
        user_profile: Target user profile
        public_profile: Population profile
    
    Returns:
        did_score: DiD score quantifying personalization value
    """
    # Step 1: Evaluate all four query-profile combinations
    s_uu = llm_evaluate(personalized_query, user_profile)
    s_ug = llm_evaluate(personalized_query, public_profile)
    s_gu = llm_evaluate(public_query, user_profile)
    s_gg = llm_evaluate(public_query, public_profile)
    
    # Step 2: Compute preferences
    delta_u = s_uu - s_ug  # Personalized query's user preference
    delta_g = s_gu - s_gg  # Public query's baseline advantage
    
    # Step 3: Compute DiD score
    did_score = delta_u - delta_g
    
    return did_score
```
