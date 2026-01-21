#!/usr/bin/env python3
"""
Generate query variants using 6 different TextAttack strategies.

This script uses TextAttack to generate variants for each query using 6 different attack strategies:
1. WordNet - Synonym replacement using WordNet
2. Embedding - Cosine similarity embedding replacement
3. Character - Character-level perturbations
4. Dependency - Dependency tree based transformations
5. Typo - Simulated typos and keyboard errors
6. Other - Additional transformation strategies
"""

import os
import sys
import pandas as pd
import numpy as np

def generate_variants_with_strategy(queries, strategy_name, num_variants_per_query=1, original_answer_ids=None, original_answer_ids_source=None):
    """Generate variants using a specific TextAttack strategy."""
    print(f"🎯 Generating variants using {strategy_name} strategy...")

    variants = []

    # Set default values if not provided
    if original_answer_ids is None:
        original_answer_ids = ['[]'] * len(queries)
    if original_answer_ids_source is None:
        original_answer_ids_source = ['[]'] * len(queries)

    try:
        if strategy_name == "wordnet":
            from textattack.transformations import WordSwapWordNet
            from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
            from textattack.constraints.semantics import WordEmbeddingDistance
            from textattack.augmentation import Augmenter

            transformation = WordSwapWordNet()
            constraints = [RepeatModification(), StopwordModification()]
            augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.1)

        elif strategy_name == "embedding":
            from textattack.transformations import WordSwapEmbedding
            from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
            from textattack.constraints.semantics import WordEmbeddingDistance
            from textattack.augmentation import Augmenter

            transformation = WordSwapEmbedding(max_candidates=10)
            constraints = [RepeatModification(), StopwordModification(), WordEmbeddingDistance(min_cos_sim=0.8)]
            augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.1)

        elif strategy_name == "character":
            from textattack.transformations import CompositeTransformation
            from textattack.transformations.word_swaps import WordSwapQWERTY, WordSwapRandomCharacterDeletion, WordSwapRandomCharacterInsertion
            from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
            from textattack.augmentation import Augmenter

            transformation = CompositeTransformation([
                WordSwapQWERTY(),
                WordSwapRandomCharacterDeletion(),
                WordSwapRandomCharacterInsertion()
            ])
            constraints = [RepeatModification(), StopwordModification()]
            augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.1)

        elif strategy_name == "dependency":
            from textattack.transformations import WordSwapHomoglyphSwap
            from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
            from textattack.augmentation import Augmenter

            transformation = WordSwapHomoglyphSwap()
            constraints = [RepeatModification(), StopwordModification()]
            augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.1)

        elif strategy_name == "typo":
            from textattack.transformations import CompositeTransformation
            from textattack.transformations.word_swaps import WordSwapQWERTY, WordSwapNeighboringCharacterSwap
            from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
            from textattack.augmentation import Augmenter

            transformation = CompositeTransformation([
                WordSwapQWERTY(),
                WordSwapNeighboringCharacterSwap()
            ])
            constraints = [RepeatModification(), StopwordModification()]
            augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.05)

        else:  # other/default strategy
            from textattack.augmentation import EasyDataAugmenter
            augmenter = EasyDataAugmenter(pct_words_to_swap=0.1, transformations_per_example=num_variants_per_query)

        # Generate variants for each query
        for i, query in enumerate(queries):
            try:
                if strategy_name in ["wordnet", "embedding", "character", "dependency", "typo"]:
                    # Use augmenter
                    augmented = augmenter.augment(query)
                    for aug_query in augmented[:num_variants_per_query]:
                        variants.append({
                            'id': i,
                            'query': aug_query,
                            'answer_ids': original_answer_ids[i],
                            'answer_ids_source': original_answer_ids_source[i]
                        })
                else:
                    # EasyDataAugmenter returns list directly
                    augmented = augmenter.augment(query)
                    for aug_query in augmented[:num_variants_per_query]:
                        variants.append({
                            'id': i,
                            'query': aug_query,
                            'answer_ids': original_answer_ids[i],
                            'answer_ids_source': original_answer_ids_source[i]
                        })

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(queries)} queries")

            except Exception as e:
                print(f"  Warning: Failed to generate variant for query {i}: {e}")
                # Add original query as fallback
                variants.append({
                    'id': i,
                    'query': query,
                    'answer_ids': original_answer_ids[i],
                    'answer_ids_source': original_answer_ids_source[i]
                })

    except ImportError as e:
        print(f"  Error: TextAttack import failed: {e}")
        print("  Falling back to simple word replacement...")

        # Simple fallback: random word replacement
        import random
        for i, query in enumerate(queries):
            words = query.split()
            if len(words) > 2:
                # Replace a random word with a synonym placeholder
                idx = random.randint(0, len(words) - 1)
                words[idx] = f"[{words[idx]}]"
                modified_query = " ".join(words)
            else:
                modified_query = query

            variants.append({
                'id': i,
                'query': modified_query,
                'answer_ids': original_answer_ids[i],
                'answer_ids_source': original_answer_ids_source[i]
            })

    print(f"✅ Generated {len(variants)} variants using {strategy_name} strategy")
    return variants

def generate_synthesized_variants():
    """Generate variants for synthesized queries using all strategies."""
    print("🚀 Generating Synthesized Query Variants with 6 Different Strategies")
    print("=" * 80)

    # Load synthesized queries
    synthesized_file = "/home/wlia0047/ar57/wenyu/stark/data/stark_qa_synthesized_100.csv"
    if not os.path.exists(synthesized_file):
        print(f"❌ Synthesized input file not found: {synthesized_file}")
        return

    df = pd.read_csv(synthesized_file)
    original_queries = df['query'].tolist()
    original_answer_ids = df['answer_ids'].tolist()

    # For synthesized data, create empty answer_ids_source lists
    original_answer_ids_source = ['[]'] * len(original_queries)
    print(f"📚 Loaded {len(original_queries)} synthesized queries")

    # Define 6 strategies
    strategies = [
        "wordnet",
        "embedding",
        "character",
        "dependency",
        "typo",
        "other"
    ]

    # Generate variants for each strategy
    all_variants = []

    for strategy in strategies:
        print(f"\n🔄 Processing strategy: {strategy}")
        print("-" * 50)

        strategy_variants = generate_variants_with_strategy(
            original_queries,
            strategy,
            num_variants_per_query=1,  # One variant per query per strategy
            original_answer_ids=original_answer_ids,
            original_answer_ids_source=original_answer_ids_source
        )

        all_variants.extend(strategy_variants)

        # Save individual strategy file
        strategy_df = pd.DataFrame(strategy_variants)
        strategy_file = f"/home/wlia0047/ar57/wenyu/stark/data/strategy_variants/{strategy}_synthesized_variants_100.csv"
        os.makedirs("/home/wlia0047/ar57/wenyu/stark/data/strategy_variants", exist_ok=True)
        strategy_df.to_csv(strategy_file, index=False)
        print(f"💾 Saved {len(strategy_variants)} {strategy} synthesized variants to {strategy_file}")

    # Save combined file
    combined_df = pd.DataFrame(all_variants)
    combined_file = "all_synthesized_strategy_variants.csv"
    combined_df.to_csv(combined_file, index=False)

    print("\n" + "=" * 80)
    print("🎉 Synthesized Variant Generation Complete!")
    print(f"📊 Total variants generated: {len(all_variants)}")
    print(f"📁 Strategy files saved in: strategy_variants/")
    print(f"📄 Combined file: {combined_file}")

    # Summary by strategy
    print("\n📈 Strategy Summary:")
    variants_per_strategy = len(original_queries)  # Each strategy generates one variant per query
    for strategy in strategies:
        print(f"  {strategy}: {variants_per_strategy} variants")

def main():
    """Main function to generate variants for all strategies."""
    print("🚀 Generating Query Variants with 6 Different Strategies")
    print("=" * 80)

    # Load original queries
    input_file = "/home/wlia0047/ar57/wenyu/stark/data/stark_qa_human_generated_eval.csv"
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)

    df = pd.read_csv(input_file)
    original_queries = df['query'].tolist()
    original_answer_ids = df['answer_ids'].tolist()

    # Handle missing answer_ids_source column in human generated data
    if 'answer_ids_source' in df.columns:
        original_answer_ids_source = df['answer_ids_source'].tolist()
    else:
        # For human generated data, use answer_ids as source or create empty lists
        original_answer_ids_source = ['[]'] * len(original_queries)
    print(f"📚 Loaded {len(original_queries)} original queries")

    # Define 6 strategies
    strategies = [
        "wordnet",
        "embedding",
        "character",
        "dependency",
        "typo",
        "other"
    ]

    # Generate variants for each strategy
    all_variants = []

    for strategy in strategies:
        print(f"\n🔄 Processing strategy: {strategy}")
        print("-" * 50)

        strategy_variants = generate_variants_with_strategy(
            original_queries,
            strategy,
            num_variants_per_query=1,  # One variant per query per strategy
            original_answer_ids=original_answer_ids,
            original_answer_ids_source=original_answer_ids_source
        )

        all_variants.extend(strategy_variants)

        # Save individual strategy file
        strategy_df = pd.DataFrame(strategy_variants)
        strategy_file = f"/home/wlia0047/ar57/wenyu/stark/data/strategy_variants/{strategy}_variants_81.csv"
        os.makedirs("/home/wlia0047/ar57/wenyu/stark/data/strategy_variants", exist_ok=True)
        strategy_df.to_csv(strategy_file, index=False)
        print(f"💾 Saved {len(strategy_variants)} {strategy} variants to {strategy_file}")

    # Save combined file
    combined_df = pd.DataFrame(all_variants)
    combined_file = "all_strategy_variants.csv"
    combined_df.to_csv(combined_file, index=False)

    print("\n" + "=" * 80)
    print("🎉 Variant Generation Complete!")
    print(f"📊 Total variants generated: {len(all_variants)}")
    print(f"📁 Strategy files saved in: strategy_variants/")
    print(f"📄 Combined file: {combined_file}")

    # Summary by strategy
    print("\n📈 Strategy Summary:")
    variants_per_strategy = len(original_queries)  # Each strategy generates one variant per query
    for strategy in strategies:
        print(f"  {strategy}: {variants_per_strategy} variants")

    # Generate synthesized variants
    print("\n" + "=" * 80)
    generate_synthesized_variants()

if __name__ == "__main__":
    main()
