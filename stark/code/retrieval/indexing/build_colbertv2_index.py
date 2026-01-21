#!/usr/bin/env python3
"""
Run ColBERTv2 Evaluation with Query Variants - Complete Pipeline

This script runs the complete ColBERTv2 evaluation pipeline for all query variants
(original + adversarial strategies) using pre-built index.
"""

import os
import sys
import torch
import argparse
import pandas as pd
from datetime import datetime
from stark_qa import load_qa, load_skb

# Add project root to path
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/stark')

def create_strategy_dataset(strategy_name):
    """Create STaRK-compatible dataset for a specific strategy."""
    if strategy_name == 'original':
        return None

    variants_file = f"/home/wlia0047/ar57/wenyu/stark/data/strategy_variants/{strategy_name}_variants_81.csv"
    stark_base_dir = f"/home/wlia0047/ar57/wenyu/stark/data/stark_strategy_{strategy_name}_dataset"

    os.makedirs(stark_base_dir, exist_ok=True)

    if not os.path.exists(variants_file):
        raise FileNotFoundError(f"Variants file not found: {variants_file}")

    df = pd.read_csv(variants_file)

    # All variant files now use the new format: id, query, answer_ids, answer_ids_source
    stark_df = pd.DataFrame({
        'id': range(len(df)),
        'query': df['query'],
        'answer_ids': df['answer_ids'],
        'query_type': ['variant'] * len(df)
    })

    # Create STaRK directory structure
    qa_dir = os.path.join(stark_base_dir, "qa", "amazon")
    split_dir = os.path.join(qa_dir, "split")
    stark_qa_dir = os.path.join(qa_dir, "stark_qa")

    os.makedirs(stark_qa_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    stark_qa_file = os.path.join(stark_qa_dir, "stark_qa.csv")
    stark_df.to_csv(stark_qa_file, index=False)

    # Create split file
    split_file = os.path.join(split_dir, "variants.index")
    with open(split_file, 'w') as f:
        for i in range(len(stark_df)):
            f.write(f"{i}\n")

    print(f"Created STaRK dataset for strategy '{strategy_name}' at: {stark_base_dir}")
    return stark_base_dir

def find_file_path_by_name(name: str, path: str) -> str:
    """Find the file path by its name in a given directory."""
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return None

def evaluate_strategy(strategy, dataset, experiments_dir):
    """Evaluate a single strategy using ColBERTv2."""
    print(f"\n{'='*60}")
    print(f"🔍 EVALUATING STRATEGY: {strategy.upper()} (FORCED, NO CACHING)")
    print("=" * 60)

    strategy_start_time = datetime.now()

    # Set up strategy-specific parameters
    if strategy == 'original':
        strategy_split = "human_generated_eval"
        query_tsv_path = "/fs04/ar57/wenyu/stark/experiments/query_hg.tsv"
    else:
        strategy_dataset_root = create_strategy_dataset(strategy)
        strategy_split = "variants"
        # Create strategy-specific query file
        variants_file = f"/home/wlia0047/ar57/wenyu/stark/data/strategy_variants/{strategy}_variants_81.csv"
        if os.path.exists(variants_file):
            df = pd.read_csv(variants_file)
            query_tsv_path = f"/home/wlia0047/ar57/wenyu/stark/experiments/query_{strategy}_variants.tsv"
            with open(query_tsv_path, 'w') as f:
                for i, row in enumerate(df.itertuples()):
                    f.write(f"{i}\t{row.query}\n")
        else:
            print(f"Warning: Variants file not found for strategy {strategy}, skipping...")
            return False

    # Always run evaluation for this strategy - USE SHARED INDEX
    try:
        from colbert.infra import Run, RunConfig, ColBERTConfig
        from colbert.data import Queries
        from colbert import Searcher

        nranks = torch.cuda.device_count()
        # Use the SAME index for all strategies (amazon_hg)
        exp_name = f"{dataset}_hg"

        with Run().context(RunConfig(nranks=nranks, experiment=exp_name)):
            config = ColBERTConfig(root=experiments_dir)
            searcher = Searcher(index=f"{dataset}_hg.nbits=2", config=config)
            queries = Queries(query_tsv_path)
            ranking = searcher.search_all(queries, k=100)
            # Save with strategy-specific name to avoid conflicts
            ranking_filename = f'ranking_{strategy}.tsv'
            ranking.save(ranking_filename)

        # Find the ranking file
        ranking_path = find_file_path_by_name(ranking_filename, experiments_dir)

        if ranking_path:
            print(f"✅ Strategy {strategy} evaluation completed! Ranking saved to: {ranking_path}")
            strategy_end_time = datetime.now()
            strategy_duration = strategy_end_time - strategy_start_time
            print(f"⏱️  Strategy {strategy} time: {strategy_duration}")
            return True
        else:
            print(f"❌ Strategy {strategy} evaluation failed - ranking file not found")
            return False

    except Exception as e:
        print(f"❌ Strategy {strategy} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_colbertv2_evaluation(strategy='all'):
    """Run ColBERTv2 evaluation for all query variants using pre-built index."""
    print("🚀 Running ColBERTv2 Evaluation with Query Variants (Pre-built Index)")
    print("=" * 80)

    # Configuration
    dataset = "amazon"
    experiments_dir = "/home/wlia0047/ar57/wenyu/stark/experiments"

    # Determine strategies to evaluate
    if strategy == 'all':
        strategies = ['original', 'character', 'embedding', 'other', 'typo', 'wordnet', 'error_aware']
    else:
        strategies = [strategy]

    print(f"🔍 Will evaluate {len(strategies)} strategies: {', '.join(strategies)}")

    try:
        # Load datasets
        print("📚 Loading datasets...")
        skb = load_skb(dataset)
        qa_dataset = load_qa(dataset, human_generated_eval=True)

        print(f"📊 Dataset loaded: {len(skb.candidate_ids)} documents, {len(qa_dataset)} queries")

        # Import ColBERTv2
        print("🔧 Importing ColBERTv2...")
        from stark_qa.models.colbertv2 import Colbertv2

        # Use pre-built index for evaluation
        print("📚 Using pre-built ColBERTv2 index...")
        print("📝 This will:")
        print("   1. Load ColBERTv2 model")
        print("   2. Load pre-built index")
        print("   3. Evaluate all query variants")
        print("   4. Save evaluation results")

        # Use the specified strategy for both index building and evaluation
        model = Colbertv2(
            skb=skb,
            dataset_name=dataset,
            human_generated_eval=True,
            add_rel=False,
            download_dir='/fs04/ar57/wenyu/stark/experiments',
            save_dir='/fs04/ar57/wenyu/stark/experiments',
            nbits=2,
            k=100,
            strategy=strategy  # Use the strategy parameter
        )

        print("✅ ColBERTv2 evaluation completed successfully!")
        print("📊 All strategies have been evaluated using the pre-built index.")

        return True

    except Exception as e:
        print(f"❌ Process failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build ColBERTv2 Index and Evaluate with Query Variants')
    parser.add_argument('--strategy', type=str, default='all',
                       choices=['original', 'character', 'embedding', 'other', 'typo', 'wordnet', 'error_aware', 'all'],
                       help='Strategy for query variants (default: all)')
    args = parser.parse_args()

    success = run_colbertv2_evaluation(strategy=args.strategy)
    if success:
        print("\n🎉 ColBERTv2 evaluation completed successfully!")
        print("📊 All query variants have been evaluated using the pre-built index.")
        print("🚀 Ready for analysis and comparison of different strategies.")
    else:
        print("\n💥 ColBERTv2 evaluation failed.")
        print("🔍 Check the error messages above.")
