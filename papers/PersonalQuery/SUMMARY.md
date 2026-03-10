# PersonalQuery Paper Summary

## Paper Location
- **Markdown Version**: `/fs04/ar57/wenyu/papers/PersonalQuery/personal_query_paper.md`
- **LaTeX Version**: `/fs04/ar57/wenyu/papers/PersonalQuery/personal_query_paper.tex`
- **References**: `/fs04/ar57/wenyu/papers/PersonalQuery/references.bib`

## Title
**Grounded Personalized Search Query Generation: A Multi-Stage Pipeline with Linguistic Style Alignment**

## Abstract Summary
A 12-stage pipeline that generates personalized search queries by:
1. Extracting fine-grained user preferences from reviews
2. Modeling 16-dimensional linguistic style features
3. Iteratively refining queries with feature-aware prompting
4. Injecting realistic spelling variations via CNN-based difficulty prediction

## Key Contributions

### 1. 12-Stage Processing Pipeline
| Stage | Function | Output |
|-------|----------|--------|
| 0-1 | Data prep + Preference extraction | Structured attribute-sentiment pairs |
| 2-3 | Classification + Splitting | Target/Public attributes, Train/Holdout split |
| 4 | Persona generation | Grounded user personas |
| 5-6 | Style analysis | 16-dim linguistic features |
| 7-8 | Query generation + Refinement | Style-aligned queries |
| 9-10 | Noise injection | Realistic spelling variations |
| 11-12 | Evaluation | LLM + Human alignment metrics |

### 2. 16-Dimensional Linguistic Features
- **Length**: tokens_per_sent, char_per_tok, n_tokens
- **Lexical**: ttr_lemma_chunks_100, lexical_density
- **POS Distribution**: NOUN, VERB, ADJ, ADV, PRON, DET, AUX, PART, SCONJ, CCONJ, ADP

### 3. Iterative Refinement Algorithm
```
Input: Base query q0, user features fu, max rounds R
q* ← q0
for r = 1 to R:
    fq ← ExtractFeatures(q*)
    g ← AnalyzeGaps(fq, fu)
    I ← GenerateInstructions(g)
    C ← GenerateCandidates(q*, I)
    q* ← SelectBest(C, fu)
    if Converged(q*, fu):
        break
Output: Refined query q*
```

### 4. CNN Spelling Difficulty Model
- Character embedding + Conv1D layers
- Handcrafted features: word length, syllable count, frequency
- User features: error type frequencies
- Output: Difficulty score [0, 1]

### 5. Evaluation Results

| Metric | Value |
|--------|-------|
| Style Distance | 0.142 (vs 0.412 baseline) |
| Semantic Similarity | 0.89 (vs 0.65 baseline) |
| Spearman's ρ (Human-LLM) | 0.82 |
| Cohen's κ | 0.71 |
| Convergence Rate | 87% |

## Key Findings

1. **Grounded personas improve query quality** - Skill level, use cases, sentiment profiles yield authentic queries
2. **Iterative refinement is essential** - Single-round generation misses fine-grained style features
3. **16 features suffice** - Curated set captures essential variation
4. **Targeted noise injection preserves semantics** - Difficulty-guided injection maintains fidelity
5. **LLM evaluation aligns with humans** - Strong correlation validates evaluation approach

## Limitations

1. Domain specificity (Arts & Crafts only)
2. User cold-start (requires review history)
3. LLM dependency (quality bounded by model)
4. Computational cost (12-stage pipeline)

## Future Directions

1. Cross-domain transfer
2. Few-shot adaptation for new users
3. Real-time deployment optimization
4. Multimodal personalization (voice, visual)

## Files in `/fs04/ar57/wenyu/papers/PersonalQuery/`

```
PersonalQuery/
├── personal_query_paper.md      # Full paper in Markdown
├── personal_query_paper.tex     # Full paper in LaTeX
├── references.bib               # BibTeX references
└── SUMMARY.md                   # This file
```

## Compilation Instructions

For LaTeX version:
```bash
cd /fs04/ar57/wenyu/papers/PersonalQuery
pdflatex personal_query_paper.tex
bibtex personal_query_paper
pdflatex personal_query_paper.tex
pdflatex personal_query_paper.tex
```

## Citation

```bibtex
@inproceedings{personalquery2024,
  title={Grounded Personalized Search Query Generation: A Multi-Stage Pipeline with Linguistic Style Alignment},
  author={Anonymous Authors},
  booktitle={Proceedings of the Conference},
  year={2024}
}
```
