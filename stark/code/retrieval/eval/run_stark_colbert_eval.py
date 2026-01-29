#!/usr/bin/env python3
"""
Simple STaRK ColBERT Evaluation Script
======================================

This script evaluates ColBERT models on the STaRK dataset with different query types.
Supports evaluating human-generated queries, variants, and standard splits.

Default: Evaluates human query dataset class generated queries (--split human_generated_eval)

Usage examples:
- Evaluate human-generated queries: python run_stark_colbert_eval.py
- Evaluate variants with strategy: python run_stark_colbert_eval.py --split variants --strategy some_strategy
- Evaluate standard test split: python run_stark_colbert_eval.py --split test
"""

import os
import subprocess
import argparse
from datetime import datetime


def create_strategy_dataset(strategy_name):
    """Create STaRK-compatible dataset for a specific strategy."""
    import pandas as pd
    import os.path as osp

    if strategy_name == 'original':
        return None

    if strategy_name == 'kg_query':
        # For KG-generated queries, use the specific CSV file
        query_file = "/home/wlia0047/ar57/wenyu/result/generated_kg_queries.csv"
        stark_base_dir = "/home/wlia0047/ar57/wenyu/stark/data/stark_strategy_kg_query_dataset"

        # Create STaRK directory structure
        qa_dir = os.path.join(stark_base_dir, "qa", "amazon")
        split_dir = os.path.join(qa_dir, "split")
        stark_qa_dir = os.path.join(qa_dir, "stark_qa")

        os.makedirs(stark_qa_dir, exist_ok=True)
        os.makedirs(split_dir, exist_ok=True)

        # Load queries
        if not os.path.exists(query_file):
            raise FileNotFoundError(f"Query file not found: {query_file}")

        df = pd.read_csv(query_file)

        # Handle answer_ids_source column name if present
        if 'answer_ids_source' in df.columns and 'answer_ids' not in df.columns:
            df['answer_ids'] = df['answer_ids_source']

        # Ensure required columns exist
        if 'id' not in df.columns:
            df['id'] = range(len(df))

        # Filter and save STaRK format file
        stark_df = pd.DataFrame({
            'id': df['id'],
            'query': df['query'],
            'answer_ids': df['answer_ids'],
            'query_type': ['kg_query'] * len(df)
        })

        stark_qa_file = os.path.join(stark_qa_dir, "stark_qa.csv")
        stark_df.to_csv(stark_qa_file, index=False)

        # Create split file
        split_file = os.path.join(split_dir, "variants.index")
        with open(split_file, 'w') as f:
            for idx in stark_df['id']:
                f.write(f"{idx}\n")

        return stark_base_dir


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

    split_file = os.path.join(split_dir, "variants.index")
    with open(split_file, 'w') as f:
        for idx in stark_df['id']:
            f.write(f"{idx}\n")

    return stark_base_dir


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='STaRK ColBERT Variants Evaluation Script')
    parser.add_argument('--dataset', type=str, default='amazon',
                       help='Dataset to evaluate (default: amazon)')
    parser.add_argument('--model', type=str, default='Colbertv2',
                       help='Model to use (default: Colbertv2)')
    parser.add_argument('--strategy', type=str, default='original',
                       choices=['original', 'character', 'embedding', 'other', 'typo', 'wordnet', 'error_aware', 'grammar_aware', 'kg_query', 'all'],
                       help='Attack strategy to evaluate (default: original, use "all" for all strategies)')
    parser.add_argument('--save_pred', action='store_true',
                       help='Save predictions (default: False)')
    parser.add_argument('--force_rerun', action='store_true',
                       help='Force rerun even if results exist (default: False)')

    args = parser.parse_args()

    # Change to STaRK root directory
    stark_root = "/home/wlia0047/ar57/wenyu/stark"
    os.chdir(stark_root)

    # Determine strategies to evaluate
    strategies = ['original', 'character', 'embedding', 'other', 'typo', 'wordnet', 'error_aware', 'grammar_aware'] if args.strategy == 'all' else [args.strategy]

    print(f"STaRK ColBERT Variants Evaluation - {len(strategies)} strategies")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Save predictions: {args.save_pred}")
    print("=" * 80)

    total_start_time = datetime.now()
    success_count = 0

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy.upper()}")
        print(f"{'='*60}")

        try:
            start_time = datetime.now()

            # Set evaluation parameters based on strategy
            if strategy == 'original':
                split = "human_generated_eval"
                dataset_root = None
            elif strategy == 'kg_query':
                split = "variants"
                # Create dataset structure
                _ = create_strategy_dataset(strategy)
                # Use SKB path
                dataset_root = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/attribute_kb"
            else:
                split = "variants"
                dataset_root = create_strategy_dataset(strategy)

            print(f"Dataset: {args.dataset}")
            print(f"Model: {args.model}")
            print(f"Strategy: {strategy}")
            print(f"Split: {split}")
            print(f"Dataset root: {dataset_root}")
            print("-" * 40)

            # Build evaluation command
            cmd = [
                "python", "eval.py",
                "--dataset", args.dataset,
                "--model", args.model,
                "--split", split,
                "--save_pred"
            ]

            if args.force_rerun:
                cmd.append("--force_rerun")  # Force rerun even if results exist

            if dataset_root:
                cmd.extend(["--dataset_root", dataset_root])

            cmd.extend(["--strategy", strategy])

            if strategy == 'kg_query':
                cmd.extend(["--csv_file", "/home/wlia0047/ar57/wenyu/stark/data/stark_strategy_kg_query_dataset/qa/amazon/stark_qa/stark_qa.csv"])

            print(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(cmd, cwd=stark_root, text=True)

            end_time = datetime.now()
            duration = end_time - start_time

            if result.returncode == 0:
                print("=" * 60)
                print(f"✓ {strategy.upper()} EVALUATION COMPLETED!")
                print(f"Duration: {duration}")
                print("=" * 60)
                success_count += 1
            else:
                print(f"✗ {strategy.upper()} EVALUATION FAILED!")
                print(f"Return code: {result.returncode}")
                print(f"Duration: {duration}")

        except Exception as e:
            print(f"✗ {strategy.upper()} EVALUATION FAILED!")
            print(f"Error: {e}")

    total_end_time = datetime.now()
    total_duration = total_end_time - total_start_time

    print(f"\n{'='*80}")
    print(f"🎉 COLBERT EVALUATION SUMMARY")
    print(f"Total strategies: {len(strategies)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(strategies) - success_count}")
    print(f"Total duration: {total_duration}")
    print(f"Strategies evaluated: {', '.join(strategies)}")
    print(f"{'='*80}")

    return 0 if success_count == len(strategies) else 1


if __name__ == "__main__":
    main()