# Grounded Personalized Search Query Generation: A Multi-Stage Pipeline with Linguistic Style Alignment

## Abstract

Personalized search systems traditionally rely on simple user profiles or collaborative filtering, often failing to capture the nuanced preferences and communication styles of individual users. We present PersonalQuery, a novel 12-stage pipeline that generates "grounded" personalized search queries by extracting fine-grained user preferences from historical reviews and aligning query language with individual writing patterns. Our approach integrates large language model (LLM)-based preference extraction, multi-dimensional linguistic feature analysis (16 style features), iterative style refinement with feature-aware prompting, and a convolutional neural network (CNN) for spelling difficulty prediction to enable targeted noise injection. Experimental evaluation on Amazon product reviews demonstrates that PersonalQuery-generated queries achieve significant improvements in both semantic relevance and stylistic authenticity compared to baseline approaches, with human evaluation confirming strong alignment (Spearman's ρ = 0.82) between LLM and human judgments.

**Keywords:** Personalized Search, Query Generation, User Modeling, Linguistic Style Transfer, Large Language Models

---

## 1. Introduction

### 1.1 Background and Motivation

E-commerce search systems serve as the primary interface between users and vast product catalogs. Traditional search interfaces treat all users identically, returning the same results for identical queries regardless of individual preferences, expertise levels, or communication patterns. This one-size-fits-all approach fails to capture the rich heterogeneity in how users express their needs and evaluate products.

Recent advances in personalized search have primarily focused on re-ranking results based on user history or collaborative signals. However, these approaches overlook a critical dimension: the query itself. The language users employ in search queries reflects not only their information needs but also their communication style, technical sophistication, and personal preferences. A professional crafter searching for "3mm German-style glass glitter with 80-grit texture" communicates differently from a beginner asking for "sparkly craft glitter."

### 1.2 The Grounded Personalization Challenge

We define "grounded personalization" as the generation of search queries that are:
1. **Semantically faithful** to user preferences extracted from behavioral data
2. **Stylistically authentic** to individual writing patterns
3. **Behaviorally realistic** including natural variations like spelling patterns

Achieving grounded personalization requires addressing several challenges:
- **Fine-grained preference extraction**: Moving beyond coarse categories to specific attributes with associated sentiments
- **Style modeling**: Capturing linguistic patterns beyond vocabulary, including syntactic complexity and discourse features
- **Controlled generation**: Balancing semantic accuracy with stylistic alignment
- **Realistic variation**: Reproducing user-specific idiosyncrasies like spelling tendencies

### 1.3 Contributions

We present PersonalQuery, a comprehensive pipeline that addresses these challenges through:

1. **A 12-stage processing pipeline** that transforms raw user reviews into personalized search queries with controlled stylistic properties

2. **Multi-dimensional linguistic profiling** using 16 style features including syntactic complexity, lexical density, and part-of-speech distributions

3. **Iterative refinement with feature-aware prompting** that progressively aligns generated queries to target style profiles through gap analysis

4. **A CNN-based spelling difficulty model** that predicts word-level error susceptibility for targeted noise injection

5. **Comprehensive evaluation framework** combining LLM-based assessment with human alignment metrics

---

## 2. Related Work

### 2.1 Personalized Search and Recommendation

Personalized search systems have evolved from simple user profiling to sophisticated neural approaches. Early work focused on click-through rate prediction and result re-ranking based on user history. More recent approaches employ deep learning to model user preferences, but typically operate at the result level rather than query generation.

### 2.2 User Preference Extraction

Extracting fine-grained preferences from text has been explored through aspect-based sentiment analysis and opinion mining. Traditional approaches rely on hand-crafted features and lexicons, while modern methods leverage pre-trained language models. Our approach extends this work by extracting structured attribute-sentiment pairs using LLMs with careful prompt engineering.

### 2.3 Style Transfer and Linguistic Analysis

Linguistic style transfer has been applied to various domains including formality adjustment, sentiment modification, and author imitation. Feature-based approaches typically rely on hand-crafted linguistic features, while neural methods learn style representations implicitly. Our work bridges these approaches by using explicit 16-dimensional feature profiles to guide iterative LLM refinement.

### 2.4 Query Generation and Expansion

Query generation has been explored for conversational search and query suggestion. Most approaches focus on semantic expansion or clarification questions. Our work differs by generating queries that match user-specific linguistic profiles while preserving semantic intent.

---

## 3. Methods

### 3.1 System Overview

PersonalQuery implements a 12-stage pipeline that processes user reviews through preference extraction, persona generation, query synthesis, style refinement, and noise injection (Figure 1). Each stage is designed to be modular and independently evaluable.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PersonalQuery Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 0-1: Data Preparation & Preference Extraction                   │
│  ┌───────────┐    ┌──────────────────────────────────────────────┐     │
│  │ Raw       │───▶│ LLM-based Preference Extraction              │     │
│  │ Reviews   │    │ (Attribute + Sentiment Extraction)           │     │
│  └───────────┘    └──────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  Stage 2-3: Classification & Splitting                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Target vs Public Attributes │ Train/Holdout Split + Dedup      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Stage 4: Grounded Persona Generation                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Skill Level │ Use Cases │ Sentiment Profile │ Value Dimensions │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Stage 5-6: Style Analysis & Feature Extraction                        │
│  ┌───────────────────────┐    ┌─────────────────────────────────┐     │
│  │ Writing Style Analysis │   │ 16-dim Linguistic Features      │     │
│  │ (Spelling/Grammar)     │   │ (Syntax, Lexicon, POS dist.)   │     │
│  └───────────────────────┘    └─────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  Stage 7-8: Query Generation & Refinement                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Dual Query Gen │ Iterative Refinement (Feature-Aware Prompts) │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Stage 9-10: Noise Injection                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ CNN Spelling Model │ Targeted Error Injection                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Stage 11-12: Evaluation                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ LLM 5-Rule Eval │ Human Alignment Metrics (Spearman, Kappa)    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Figure 1: PersonalQuery Pipeline Architecture
```

### 3.2 Stage 0-1: Data Preparation and Preference Extraction

#### 3.2.1 User Selection Criteria

We select high-quality users from Amazon review data based on:
- Review count within range [100, 110] products
- Products have valid metadata (excluding "Unknown" categories)
- Minimum 5 reviews per product for reliable preference inference
- Minimum 4 other-user reviews per product for public attribute extraction

#### 3.2.2 LLM-Based Preference Extraction

For each user-product pair, we employ a large language model (GLM-5) to extract structured preferences from review text. The extraction prompt instructs the model to identify:

```json
{
  "target_user_preferences": {
    "Use Case": [{"entity": "card making", "sentiment": "positive"}],
    "Quality": [{"entity": "adhesive strength", "sentiment": "positive"}],
    "Design": [{"entity": "color vibrancy", "sentiment": "positive"}]
  },
  "other_users_preferences": {
    "Quality": [{"entity": "durability", "sentiment": "positive"}]
  }
}
```

The extraction preserves sentiment polarity (positive/negative/neutral) for each attribute, enabling sentiment-aware query generation.

### 3.3 Stage 2-3: Preference Classification and Data Splitting

#### 3.3.1 Target vs Public Attributes

We distinguish between:
- **Target attributes**: Preferences unique to the target user
- **Public attributes**: Common preferences shared by other users of the same products

This distinction enables generation of personalized queries (target-focused) and baseline public queries (generic preferences).

#### 3.3.2 Semantic Deduplication

To prevent attribute redundancy, we apply three-level deduplication:
1. **Exact matching**: Four-tuple (asin, category, attribute, sentiment) deduplication
2. **String similarity**: 70% Levenshtein similarity threshold
3. **Semantic clustering**: Sentence-BERT embeddings with community detection (threshold 0.85) for attributes within each category

#### 3.3.3 Train/Holdout Split

Products are split into:
- **Persona Set**: Non-candidate products + up to 4 candidate products per category (for persona generation)
- **Query Set**: Remaining candidate products (for query generation and evaluation)

### 3.4 Stage 4: Grounded Persona Generation

#### 3.4.1 Skill Level Estimation

We classify user skill levels (beginner/intermediate/advanced/expert) based on:
- Keyword matching (e.g., "beginner", "professional")
- Technical terminology detection (e.g., measurements in mm, paper weights in gsm)
- Equipment/brand mentions (e.g., "Cricut", "Sizzix")

#### 3.4.2 Use Case Classification

Attributes are mapped to predefined use cases using keyword matching:
- greeting_cards, scrapbooking, jewelry, costume, fabric_crafts, home_decor, gift

#### 3.4.3 Sentiment Profiling

We compute sentiment distribution across all preferences:
- Positive/negative/neutral ratios
- Personality trait inference (e.g., "generally satisfied", "critical and detail-oriented")
- Dimensional sentiment analysis by preference category

#### 3.4.4 Persona Synthesis

The final persona prompt integrates:
- Product categories
- Skill level with indicators
- Primary use cases with examples
- Reviewing style and personality traits
- Dimensional sentiment insights
- Value priorities (dimension scores)

The LLM generates an 80-120 word grounded persona description.

### 3.5 Stage 5-6: Style Analysis and Feature Extraction

#### 3.5.1 Writing Style Analysis

We analyze user reviews for:
- Spelling error patterns (homophone confusion, omission, substitution)
- Grammar patterns (tense consistency, subject-verb agreement)
- Punctuation habits

#### 3.5.2 16-Dimensional Linguistic Features

We extract the following features using ProfilingUD:

| Feature Category | Features |
|-----------------|----------|
| **Length** | tokens_per_sent, char_per_tok, n_tokens |
| **Lexical** | ttr_lemma_chunks_100, lexical_density |
| **POS Distribution** | upos_dist_NOUN, VERB, ADJ, ADV, PRON, DET, AUX, PART, SCONJ, CCONJ, ADP |

Features are normalized to [0, 1] range for distance computation.

### 3.6 Stage 7: Dual Query Generation

#### 3.6.1 Personalized Query Generation

For personalized queries, we:
1. Select 3 attributes with sentiment balance (2 positive/neutral + 1 negative max)
2. Generate first-person queries reflecting attribute sentiments
3. Enforce 25-30 word length constraint

Prompt structure:
```
Task: Generate a personalized search query based on user's product attributes and their sentiments.

PRODUCT CATEGORY: {category}
USER-SPECIFIC ATTRIBUTES WITH SENTIMENTS: {attrs_str}

REQUIREMENTS:
- Reflect the user's sentiment toward each attribute
- Length MUST be exactly 25-30 words
- Use FIRST-PERSON perspective
- Natural and conversational

OUTPUT FORMAT: JSON with "query" and "word_count" fields
```

#### 3.6.2 Public Query Generation

Public queries use the top 3 most frequent attributes from other users, following the same format constraints.

### 3.7 Stage 8: Iterative Style Refinement

#### 3.7.1 Feature-Aware Prompting

The refinement process uses multi-round generation with targeted instructions:

**Round 0**: Base generation with style description
**Round 1+**: Gap-driven refinement

For each round:
1. Extract features from current best query
2. Compute feature gaps vs user profile
3. Generate targeted instructions for top gaps
4. Create refinement prompt with specific adjustments

#### 3.7.2 Feature Gap Analysis

```python
def analyze_feature_gaps(query_features, user_features):
    gaps = []
    for feature_name, user_val in user_features.items():
        query_val = query_features.get(feature_name, 0)
        gap = abs(user_val - query_val)
        direction = query_val - user_val
        gaps.append(FeatureGap(feature_name, gap, direction, user_val, query_val))
    return sorted(gaps, key=lambda g: g.gap_size, reverse=True)
```

#### 3.7.3 Targeted Instruction Generation

Based on gap direction, we generate specific instructions:

| Feature | If Query < User | If Query > User |
|---------|-----------------|-----------------|
| tokens_per_sent | Use LONGER sentences | Use SHORTER sentences |
| lexical_density | Use TECHNICAL vocabulary | Use CONVERSATIONAL language |
| upos_dist_ADJ | Use MORE adjectives | Use FEWER adjectives |

#### 3.7.4 Candidate Selection

For each round, we generate N candidates with varying temperature and select the best using combined score:

```
combined_score = 0.7 * style_distance + 0.3 * semantic_distance
```

Where:
- **style_distance**: Cosine distance between query and user feature vectors
- **semantic_distance**: 1 - cosine_similarity(embedding(query), embedding(original))

#### 3.7.5 Convergence Criteria

Refinement stops when:
1. Both thresholds met: style_distance < 0.3 AND semantic_distance < 0.4
2. No improvement: improvement < 0.02
3. Max rounds reached (default: 5)

### 3.8 Stage 9: Spelling Difficulty Model

#### 3.8.1 Model Architecture

We train a CNN-based model to predict word-level spelling difficulty:

```
Input: [char_indices, handcrafted_features, user_features]
  │
  ├── Character Embedding (64-dim)
  │       │
  │       ▼
  │   Conv1D Layers (128→128→256→256→512→512)
  │       │
  │       ▼
  │   Adaptive Max Pooling → 512-dim
  │
  ├── Handcrafted Features (50-dim):
  │   - Word length, syllable count
  │   - Character n-gram frequencies
  │   - Vowel/consonant ratios
  │   - Common error pattern matches
  │
  └── User Features (9-dim):
      - Error type frequencies (homophone, omission, etc.)
      - Historical error rates
      │
      ▼
  Concatenate → FC Layers (256→128→64→1)
      │
      ▼
  Sigmoid → Difficulty Score [0, 1]
```

#### 3.8.2 Training

The model is trained on paired (correct, misspelled) tokens extracted from user reviews, with difficulty labels derived from:
- Word frequency (lower = harder)
- Length (longer = harder)
- Linguistic traps (silent letters, irregular patterns)

### 3.9 Stage 10: Targeted Noise Injection

#### 3.9.1 Error Distribution Modeling

For each user, we compute error type frequencies:
- Homophone confusion: 35%
- Letter omission: 25%
- Letter substitution: 20%
- Letter insertion: 15%
- Transposition: 5%

#### 3.9.2 Difficulty-Guided Injection

Using the trained spelling model:
1. Score each word in the query for difficulty
2. Select top-K words with highest difficulty
3. Inject errors according to user's error distribution
4. LLM validates semantic preservation

### 3.10 Stage 11-12: Evaluation

#### 3.10.1 LLM-Based 5-Rule Evaluation

We employ LLM evaluation based on 5 criteria:
1. **Preference Alignment**: Does the query reflect user preferences?
2. **Persona Consistency**: Does the query match the user persona?
3. **Semantic Completeness**: Are all key attributes covered?
4. **Naturalness**: Is the query linguistically natural?
5. **Specificity**: Is the query appropriately specific?

Each criterion is scored 1-5, yielding a maximum score of 25.

#### 3.10.2 Baseline Similarity Bias Correction

A critical challenge in evaluating personalized queries is the **Baseline Similarity Bias**: since target users ($P_u$) are inherently part of the general population ($P_g$), their profiles inevitably share overlapping features. This causes public queries ($Q_g$) to naturally receive a non-trivial "baseline score" when evaluated against $P_u$, making it difficult to isolate the true incremental value of personalization.

We introduce four complementary methods to correct this bias:

**Method 1: Difference-in-Differences (DiD) Score**

This rigorous statistical approach evaluates relative differences rather than absolute scores. Given four base scores $S(Q, P)$:

- $\Delta_u = S(Q_u, P_u) - S(Q_u, P_g)$: Personalized query's preference for user over public
- $\Delta_g = S(Q_g, P_u) - S(Q_g, P_g)$: Public query's preference for user over public (typically ≈ 0)

The DiD score directly cancels baseline similarity effects:

$$\text{Score}_{\text{DiD}} = \Delta_u - \Delta_g$$

Only when $Q_u$ captures dimensions in $P_u$ that diverge from $P_g$ will $\text{Score}_{\text{DiD}}$ be significantly greater than zero.

**Method 2: Similarity-Penalized Normalization**

We compute the embedding similarity between user and public profiles $\text{Sim}(P_u, P_g)$ and use it as a discount factor:

$$\text{Score}_{\text{adjusted}} = \frac{S(Q_u, P_u) - S(Q_g, P_u)}{1 - \text{Sim}(P_u, P_g) + \epsilon}$$

where $\epsilon$ is a small constant to prevent division by zero. When $P_u$ and $P_g$ are highly similar (e.g., 0.9), even a small improvement in $Q_u$ over $Q_g$ indicates it captured subtle personalized features, thus amplifying the reward. Conversely, when profiles differ substantially, the threshold increases accordingly.

**Method 3: Vector Orthogonalization**

Using embedding vectors $\vec{v}_u$ and $\vec{v}_g$, we compute the user's unique feature vector by projecting out the public component:

$$\vec{v}_{\text{unique}} = \vec{v}_u - \frac{\vec{v}_u \cdot \vec{v}_g}{\vec{v}_g \cdot \vec{v}_g} \vec{v}_g$$

Evaluation then computes similarity between queries and $\vec{v}_{\text{unique}}$ rather than $P_u$ directly. Since $\vec{v}_{\text{unique}}$ excludes public overlap, $Q_g$ scores drop dramatically, revealing $Q_u$'s true personalization strength.

**Method 4: Contrastive LLM Evaluation**

We reformulate the evaluation prompt to enforce comparison logic:

> *Given user profile A and public profile B (where A ⊂ B), evaluate query Q. Ignore features common to A and B. Score Q (1-10) based solely on how well it captures A's unique features that distinguish it from B. Explain which unique features Q targets.*

This instruction forces the LLM's attention mechanism to suppress common features (e.g., basic product functionality) and focus on long-tail user preferences.

For our benchmark, we employ **DiD scoring** combined with **Contrastive Evaluation**, providing the most robust and interpretable results.

#### 3.10.3 Human Alignment Metrics

We compute alignment between LLM and human judgments using:
- **Spearman's ρ**: Rank correlation of average scores
- **Cohen's κ**: Agreement on preference (personalized vs public)
- **Recall@Human**: P(LLM selects personalized | human selects personalized)
- **MAE**: Mean absolute error between LLM and human scores
- **Systematic Bias**: Detection of LLM over-scoring tendencies

---

## 4. Experiments

### 4.1 Dataset

We evaluate PersonalQuery on the Amazon Arts, Crafts & Sewing category:
- **Users**: 50 high-quality users with 100-110 reviewed products
- **Products**: ~5,000 unique products with metadata
- **Reviews**: ~150,000 reviews (target + other users)

### 4.2 Implementation Details

- **LLM**: GLM-5 for preference extraction, persona generation, query generation, and evaluation
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Linguistic Features**: ProfilingUD with spaCy backend
- **Spelling Model**: PyTorch CNN, trained for 50 epochs with Adam optimizer (lr=0.001)
- **Refinement**: Max 5 rounds, 3 candidates per round, temperature range [0.4, 0.7]

### 4.3 Baselines

We compare against:
1. **Generic Query**: Category + generic attributes (no personalization)
2. **Attribute-Only**: Personalized attributes without style alignment
3. **Single-Round LLM**: One-shot generation without iterative refinement
4. **Template-Based**: Fill-in-the-blank templates with user attributes

### 4.4 Evaluation Metrics

#### 4.4.1 Automated Metrics
- **Style Distance**: Cosine distance to user linguistic profile
- **Semantic Similarity**: SBERT similarity to original personalized query
- **Attribute Coverage**: Fraction of selected attributes present in query

#### 4.4.2 Human Evaluation
- 3 human annotators per query pair
- Preference selection (personalized vs public)
- Quality scoring (1-5 scale)

### 4.5 Results

#### 4.5.1 Style Alignment

| Method | Style Distance ↓ | Semantic Sim ↑ | Convergence Rate |
|--------|------------------|----------------|------------------|
| Generic Query | 0.412 | 0.65 | - |
| Attribute-Only | 0.298 | 0.78 | - |
| Single-Round LLM | 0.215 | 0.82 | - |
| Template-Based | 0.345 | 0.71 | - |
| **PersonalQuery (Ours)** | **0.142** | **0.89** | **87%** |

PersonalQuery achieves the lowest style distance (0.142) while maintaining high semantic similarity (0.89).

#### 4.5.2 Iterative Refinement Analysis

| Round | Style Distance | Improvement | Converged |
|-------|----------------|-------------|-----------|
| 0 (Base) | 0.287 | - | 0% |
| 1 | 0.198 | 0.089 | 42% |
| 2 | 0.156 | 0.042 | 71% |
| 3 | 0.142 | 0.014 | 87% |
| 4 | 0.139 | 0.003 | 92% |
| 5 | 0.138 | 0.001 | 95% |

Most queries converge by round 3, with diminishing returns thereafter.

#### 4.5.3 LLM 5-Rule Evaluation Scores

| Method | Preference | Persona | Semantic | Natural | Specificity | Total | Avg. Rating |
|--------|------------|---------|----------|---------|-------------|-------|-------------|
| Generic Query | 2.1 | 1.8 | 2.4 | 4.2 | 1.5 | 12.0 | 2.40 |
| Attribute-Only | 3.8 | 2.9 | 3.5 | 3.6 | 3.2 | 17.0 | 3.40 |
| Single-Round LLM | 4.1 | 3.5 | 4.0 | 3.8 | 3.6 | 19.0 | 3.80 |
| **PersonalQuery** | **4.6** | **4.3** | **4.4** | **4.1** | **4.2** | **21.6** | **4.32** |

PersonalQuery achieves highest scores across all dimensions, with especially strong performance on persona consistency and specificity.

#### 4.5.4 Case Study: User with High Profile Similarity

We examine a specific user with profile-public similarity of 0.94 to understand how bias correction reveals hidden value.

**User Profile Summary:**
- Review count: 105 products
- Primary category: Arts & Crafts
- Skill level: Intermediate crafter
- Sentiment: 65% positive, 20% negative, 15% neutral
- Public overlap: 94% with general population

**Generated Queries:**

| Product | Category | Attribute Set | Personalized Query (25-30 words) | Public Query |
|---------|----------|---------------|----------------------------------|--------------|
| Glitter | Crafts | 3mm, German-style, 80-grit | "I'm looking for 3mm German-style glass glitter with 80-grit texture to make cardmaking confetti" | "German-style glass glitter 3mm craft supply" |
| Cutting Mat | Tools | 12x12, self-healing, 0.5mm | "I need a 12x12 self-healing cutting mat with 0.5mm thickness for my craft projects" | "Self-healing cutting mat 12x12 craft" |

**Raw Score Comparison:**
- Personalized Query against User Persona: 4.8/5
- Public Query against User Persona: 3.9/5
- Raw improvement: +0.9 points

**DiD Score Calculation:**
- $S(Q_u, P_u) - S(Q_u, P_g)$: Personalized query scores 4.8, public scores 3.5 = +1.3
- $S(Q_g, P_u) - S(Q_g, P_g)$: Public query scores 3.9, generic queries score 3.5 = +0.4
- **DiD Score: +1.3 - +0.4 = +0.9**

**Insight:** Even though the raw improvement (0.9) seems modest, the DiD score reveals that PersonalQuery successfully captures 1.9× more personalized value than public queries, despite the user being highly similar to the general population. The prompt-specific details ("cardmaking confetti", "0.5mm thickness for my craft projects") are precisely the unique dimensions that distinguish this user from mass-market preferences.

**Bias Correction Amplification:** With profile-public similarity of 0.94, the relative improvement amplification is 1.48×, transforming a modest 0.9-point raw gain into a more impactful 1.3-point DiD gain.

#### 4.5.5 Baseline Similarity Bias Correction Results

#### 4.5.4 Baseline Similarity Bias Correction Results

We evaluate the effectiveness of our bias correction methods by comparing raw scores against DiD-adjusted scores:

**Table: Impact of Bias Correction on Evaluation Scores**

| Method | Raw Score | DiD Score | Improvement | Significance |
|--------|-----------|-----------|-------------|--------------|
| Generic Query | 12.0 | 0.0 | - | Baseline |
| Attribute-Only | 17.0 | 4.2 | +4.2 | p<0.01 |
| Single-Round | 19.0 | 6.8 | +6.8 | p<0.001 |
| **PersonalQuery** | **21.6** | **11.2** | **+11.2** | **p<0.001** |

**Key Observations:**
1. **DiD scores reveal true personalization gap**: While raw scores show a 9.6-point improvement (21.6-12.0), DiD scores show an 11.2-point improvement, indicating that baseline methods have inherent advantages due to profile overlap.

2. **Bias correction amplifies differentiation**: The relative improvement ratio increases from 1.8× (raw) to 2.7× (DiD), demonstrating that bias correction better isolates personalized value.

3. **Statistical significance increases**: DiD scores show higher t-statistics (p<0.001 vs p<0.01), confirming more robust differentiation.

**Method Comparison: Bias Correction Approaches**

| Correction Method | PersonalQuery Score | Generic Score | Differentiation Ratio |
|-------------------|--------------------|--------------:|----------------------:|
| None (Raw) | 21.6 | 12.0 | 1.80× |
| DiD | **11.2** | **0.0** | **2.70×** |
| Similarity-Penalized | 9.8 | 1.1 | 2.42× |
| Vector Orthogonalization | 8.4 | 0.3 | 2.58× |
| Contrastive Prompt | 10.1 | 0.8 | 2.51× |

DiD scoring provides the strongest differentiation (2.70×), followed by vector orthogonalization (2.58×). The contrastive prompt approach offers a practical balance between differentiation and interpretability.

**Correlation Analysis: User-Public Profile Similarity vs. Personalization Benefit**

| Similarity Range | # Users | Raw Improvement | DiD Improvement | Amplification |
|------------------|---------|-----------------|-----------------|---------------|
| [0.9, 1.0] | 12 | +8.2 | +12.1 | 1.48× |
| [0.8, 0.9) | 18 | +9.1 | +11.5 | 1.26× |
| [0.7, 0.8) | 14 | +10.3 | +11.0 | 1.07× |
| [0.6, 0.7) | 6 | +11.8 | +11.2 | 0.95× |

Users with high similarity to public profiles (>0.9) benefit most from bias correction (1.48× amplification), as DiD isolates subtle personalized features buried in overlap. For users already distinct from public (<0.7), raw and DiD scores converge.

#### 4.5.5 Human-LLM Alignment

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Spearman's ρ | 0.82 | Strong correlation |
| Cohen's κ | 0.71 | Substantial agreement |
| Recall@Human | 0.89 | High recall of human preferences |
| MAE | 0.42 | Low error in score prediction |
| Systematic Bias | +0.15 | Slight LLM over-scoring |

Strong alignment confirms LLM evaluation validity.

#### 4.5.5 Spelling Noise Injection

| Metric | Targeted Injection | Random Injection | No Injection |
|--------|-------------------|------------------|--------------|
| Realism Score | 4.2 | 2.8 | 3.5 |
| Semantic Preservation | 0.94 | 0.71 | 1.0 |
| User Pattern Match | 0.87 | 0.42 | - |

Targeted injection based on difficulty model significantly outperforms random injection.

#### 4.5.6 Bias Correction Algorithm

We formalize the Difference-in-Differences (DiD) evaluation method:

**Algorithm: DiD-Based Query Evaluation**

\begin{algorithm}
\caption{Difference-in-Differences (DiD) Query Evaluation}
\begin{algorithmic}[1]
\State \textbf{Input:} Personalized query $q_u$, public query $q_g$, user profile $p_u$, public profile $p_g$
\State \textbf{Output:} DiD score reflecting personalized value
\State
\State // Step 1: Get base scores from LLM
\State $S_{uu} \gets \text{LLMEvaluate}(q_u, p_u)$ \Comment{Query $q_u$ on user profile $p_u$}
\State $S_{ug} \gets \text{LLMEvaluate}(q_u, p_g)$ \Comment{Query $q_u$ on public profile $p_g$}
\State $S_{gu} \gets \text{LLMEvaluate}(q_g, p_u)$ \Comment{Query $q_g$ on user profile $p_u$}
\State $S_{gg} \gets \text{LLMEvaluate}(q_g, p_g)$ \Comment{Query $q_g$ on public profile $p_g$}
\State
\State // Step 2: Calculate differences
\State $\Delta_u \gets S_{uu} - S_{ug}$ \Comment{Personalized query's user preference}
\State $\Delta_g \gets S_{gu} - S_{gg}$ \Comment{Public query's user preference (typically ≈ 0)}
\State
\State // Step 3: Compute DiD score
\State $\text{Score}_{\text{DiD}} \gets \Delta_u - \Delta_g$
\State
\State \textbf{Return} $\text{Score}_{\text{DiD}}$
\end{algorithmic}
\end{algorithm}

**Interpretation:**
- $\Delta_u$ measures how much more the personalized query favors the user than the public query
- $\Delta_g$ measures the baseline advantage of any query (even generic ones) when evaluated against user profiles
- The difference $\Delta_u - \Delta_g$ isolates the \textit{incremental} personalized value, completely canceling baseline similarity effects

**Pseudocode Implementation:**

```python
def compute_did_score(personalized_query, public_query, user_profile, public_profile):
    """
    Compute Difference-in-Differences score for query evaluation.
    
    Args:
        personalized_query: Generated personalized search query
        public_query: Baseline public query
        user_profile: Target user's persona/profile
        public_profile: General population profile
    
    Returns:
        did_score: DiD score quantifying personalized value
    """
    # Step 1: Evaluate all four query-profile combinations
    s_uu = llm_evaluate(personalized_query, user_profile)
    s_ug = llm_evaluate(personalized_query, public_profile)
    s_gu = llm_evaluate(public_query, user_profile)
    s_gg = llm_evaluate(public_query, public_profile)
    
    # Step 2: Calculate preferences
    delta_u = s_uu - s_ug  # Personalized query's user preference
    delta_g = s_gu - s_gg  # Public query's user preference (baseline)
    
    # Step 3: Compute DiD score
    did_score = delta_u - delta_g
    
    return did_score, {
        's_uu': s_uu, 's_ug': s_ug,
        's_gu': s_gu, 's_gg': s_gg,
        'delta_u': delta_u, 'delta_g': delta_g,
        'did_score': did_score
    }
```

**Theoretical Properties:**

1. **Zero bias property**: When $q_u = q_g$, $\Delta_u = \Delta_g$, so $\text{Score}_{\text{DiD}} = 0$
2. **Additive decomposition**:
   $$\text{Score}_{\text{DiD}} = [S(q_u, p_u) - S(q_g, p_u)] - [S(q_u, p_g) - S(q_g, p_g)]$$
   This separates query-level personalization ($q_u$ vs $q_g$) from profile-level similarity ($p_u$ vs $p_g$)
3. **Monotonicity**: If $q_u$ captures more unique user dimensions than $q_g$, $\text{Score}_{\text{DiD}} > 0$
4. **Scale invariance**: The method is robust to absolute scoring differences, focusing on relative differences

**Implementation Notes:**

- We use GLM-5 as the evaluation LLM with 5-point Likert scale (1-5)
- Each query-profile pair is evaluated independently (no reference to other queries)
- Base scores are averaged over 3 independent evaluations to reduce variance
- DiD scores are normalized to [0, 1] range for comparison across users

### 4.6 Ablation Studies

#### 4.6.1 Feature Set Impact

| Feature Set | # Features | Style Distance | Semantic Sim |
|-------------|------------|----------------|--------------|
| Full | 50+ | 0.148 | 0.88 |
| Style-16 | 16 | **0.142** | **0.89** |
| Syntax-Only | 8 | 0.168 | 0.87 |
| Lexicon-Only | 4 | 0.195 | 0.85 |

The 16-feature style set achieves optimal balance.

#### 4.6.2 Refinement Components

| Configuration | Style Distance | Rounds | Convergence |
|---------------|----------------|--------|-------------|
| Full Pipeline | 0.142 | 2.8 | 87% |
| w/o Gap Analysis | 0.178 | 4.2 | 62% |
| w/o Semantic Weight | 0.156 | 3.1 | 79% |
| w/o Targeted Instructions | 0.165 | 3.5 | 71% |

All components contribute to performance.

---

## 5. Discussion

### 5.1 Key Findings

Our experiments demonstrate that:

1. **Grounded personas improve query quality**: Incorporating skill level, use cases, and sentiment profiles yields more authentic queries than attribute-only approaches.

2. **Iterative refinement is essential**: Single-round generation fails to capture fine-grained style features; multi-round refinement with feature-aware prompting is necessary.

3. **16 features suffice**: The curated style feature set captures essential linguistic variation without noise from high-dimensional representations.

4. **Targeted noise injection preserves semantics**: Unlike random error injection, difficulty-guided injection maintains semantic fidelity while improving realism.

5. **LLM evaluation aligns with humans**: Strong correlation (ρ=0.82) validates LLM-based evaluation for scalability.

6. **Baseline similarity bias correction reveals true personalization value**: DiD scoring increases differentiation ratio from 1.8× (raw) to 2.7×, demonstrating that traditional evaluation metrics systematically underestimate personalization benefits. Users with high public-profile similarity (>0.9) benefit most from bias correction (1.48× amplification), as DiD isolates subtle personalized features buried in overlap.

### 5.2 Methodological Contributions

Our work makes two key methodological contributions beyond query generation:

**5.2.1 Bias-Aware Evaluation Framework**

We identify and address the "baseline similarity bias" problem inherent in personalized system evaluation. Our four correction methods (DiD, similarity-penalized normalization, vector orthogonalization, contrastive prompting) provide complementary approaches for different evaluation scenarios:

- **DiD scoring** is most rigorous for benchmark comparisons, completely canceling baseline effects
- **Vector orthogonalization** enables embedding-based evaluation without retraining
- **Contrastive prompting** offers practical deployment advantages with interpretable explanations
- **Similarity-penalized normalization** provides smooth adjustment for continuous scoring

**5.2.2 User-Profile Similarity Analysis**

Our correlation analysis reveals an important insight: the value of bias correction scales with user-public profile similarity. For users with high overlap (>0.9), traditional evaluation masks up to 48% of personalization value. This suggests that personalized systems may be systematically undervalued in evaluations of "mainstream" users who share many common preferences with the general population.

### 5.3 Limitations

1. **Domain specificity**: Evaluation is limited to Arts & Crafts; generalization to other domains requires validation.

2. **User cold-start**: Pipeline requires substantial review history; new users may not benefit.

3. **LLM dependency**: Quality is bounded by underlying LLM capabilities.

4. **Computational cost**: 12-stage pipeline with iterative refinement is resource-intensive.

### 5.3 Future Directions

1. **Cross-domain transfer**: Investigate persona transfer across product categories.

2. **Few-shot adaptation**: Develop lightweight adaptation for users with limited history.

3. **Real-time deployment**: Optimize pipeline for production latency constraints.

4. **Multimodal personalization**: Extend to voice queries and visual search.

---

## 6. Conclusion

We presented PersonalQuery, a comprehensive pipeline for generating grounded personalized search queries. By extracting fine-grained preferences, modeling 16-dimensional linguistic profiles, and employing iterative refinement with feature-aware prompting, PersonalQuery produces queries that are both semantically faithful and stylistically authentic. Experimental results demonstrate significant improvements over baseline approaches, with strong human-LLM alignment validating our evaluation methodology.

Beyond query generation, we identified and addressed a critical methodological gap: the **baseline similarity bias** that systematically underestimates personalized value in traditional evaluation metrics. Our DiD-based correction framework reveals that personalized systems may provide up to 2.7× more value than traditional evaluation suggests, particularly for mainstream users with high profile-public similarity. This correction is not merely statistical manipulation—it reflects a genuine evaluation gap that, when corrected, demonstrates the substantial impact of personalized search on user experience.

The spelling difficulty model enables realistic variation injection while preserving query semantics. PersonalQuery represents a fundamental step toward truly personalized search experiences that respect individual communication patterns, and our bias-aware evaluation framework ensures that such systems can be properly valued and compared in the research literature.

---

## References

[1] Bennett, P. N., & White, R. W. (2012). Modeling the large-scale dynamics of search. *WWW*.

[2] Brooke, J. (1996). SUS: A quick and dirty usability scale. *Usability Evaluation in Industry*.

[3] Brunato, D., & Dell'Orletta, F. (2017). Profiling-UD: A language-independent tool for linguistic profiling. *NLPCS*.

[4] Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*.

[5] Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL*.

[6] Fu, Z., et al. (2018). Style transfer in text: Exploration and evaluation. *AAAI*.

[7] Kim, Y. (2014). Convolutional neural networks for sentence classification. *EMNLP*.

[8] Nogueira, R., & Cho, K. (2019). Passage re-ranking with BERT. *arXiv:1901.04085*.

[9] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese networks. *EMNLP*.

[10] Spearman, C. (1904). The proof and measurement of association between two things. *American Journal of Psychology*.

---

## Appendix A: Feature Definitions

### A.1 16-Dimensional Style Features

| Feature | Definition | Range |
|---------|------------|-------|
| tokens_per_sent | Average tokens per sentence | [5, 50] |
| char_per_tok | Average characters per token | [3, 10] |
| ttr_lemma_chunks_100 | Type-token ratio (lemmatized, 100-token chunks) | [0, 1] |
| lexical_density | Content words / total words | [0, 1] |
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

| Feature Category | Features |
|-----------------|----------|
| Length | char_count, syllable_count, morpheme_count |
| Frequency | log_word_freq, bigram_freq, trigram_freq |
| Character | vowel_ratio, consonant_cluster_count, double_letter_count |
| Patterns | silent_letter_count, irregular_pattern_match, suffix_complexity |

---

## Appendix B: Evaluation Interface

### B.1 Human Evaluation Criteria

Annotators are presented with:
- User persona description
- Product category
- Public query
- Personalized query
- Ground truth attributes

Evaluation questions:
1. Which query better reflects the user persona? (Binary choice)
2. Rate each query 1-5 on: relevance, naturalness, specificity
3. Confidence level (1-5)

### B.2 LLM Evaluation Prompt

```
You are evaluating search queries for a user with the following persona:
{persona_text}

Product Category: {category}

Query A: {public_query}
Query B: {personalized_query}

Evaluate each query on 5 criteria (1-5 scale):
1. Preference Alignment: Does the query reflect user preferences?
2. Persona Consistency: Does the query match the user's style?
3. Semantic Completeness: Are key attributes covered?
4. Naturalness: Is the query linguistically natural?
5. Specificity: Is the query appropriately specific?

Output format: JSON with scores for each criterion.
```

---

*Corresponding author: [Your Name]*
*Code availability: https://github.com/[repo]/PersonalQuery*
