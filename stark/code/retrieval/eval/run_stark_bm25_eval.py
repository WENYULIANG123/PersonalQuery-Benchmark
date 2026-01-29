#!/usr/bin/env python3
"""
STaRK BM25 Evaluation Script for Amazon Dataset
==============================================

This script runs BM25 evaluation on the STaRK Amazon dataset.
Optimized for multi-CPU parallel processing.
"""

import os
import select
import sys
import subprocess
from datetime import datetime
import multiprocessing
import argparse


def create_strategy_dataset(strategy_name, input_csv=None):
    """Create STaRK-compatible dataset for a specific strategy."""
    import pandas as pd
    import hashlib

    if strategy_name == 'original':
        # For original queries, use the standard STaRK dataset path
        # Return None to indicate using standard dataset loading
        return None, None

    if strategy_name == 'synthesized':
        # For synthesized queries, use the original synthesized data file
        synthesized_file = "/home/wlia0047/ar57/wenyu/stark/data/stark_qa_synthesized_100.csv"
        stark_base_dir = "/home/wlia0047/ar57/wenyu/stark/data/stark_strategy_synthesized_dataset"

        # Create STaRK directory structure
        qa_dir = os.path.join(stark_base_dir, "qa", "amazon")
        split_dir = os.path.join(qa_dir, "split")
        stark_qa_dir = os.path.join(qa_dir, "stark_qa")

        os.makedirs(stark_qa_dir, exist_ok=True)
        os.makedirs(split_dir, exist_ok=True)

        # Load synthesized data
        if not os.path.exists(synthesized_file):
            raise FileNotFoundError(f"Synthesized data file not found: {synthesized_file}")

        df = pd.read_csv(synthesized_file)

        # Synthesized data format: id, query, answer_ids
        stark_df = pd.DataFrame({
            'id': df['id'],  # Keep original IDs
            'query': df['query'],
            'answer_ids': df['answer_ids'],
            'query_type': ['synthesized'] * len(df)
        })

        # Save STaRK format file
        stark_qa_file = os.path.join(stark_qa_dir, "stark_qa.csv")
        stark_df.to_csv(stark_qa_file, index=False)

        # Create split file (using test split for synthesized data)
        split_file = os.path.join(split_dir, "test.index")
        with open(split_file, 'w') as f:
            for idx in stark_df['id']:
                f.write(f"{idx}\n")

        return stark_base_dir, stark_qa_file

    if strategy_name == 'kg_query':
        # For KG-generated queries, use the provided CSV or default
        if input_csv:
            query_file = input_csv
            # Create unique directory for custom CSV to avoid conflicts
            csv_hash = hashlib.md5(input_csv.encode()).hexdigest()[:8]
            stark_base_dir = f"/home/wlia0047/ar57/wenyu/stark/data/stark_strategy_kg_query_{csv_hash}_dataset"
        else:
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

        return stark_base_dir, stark_qa_file

    # Paths for variant strategies
    variants_file = f"/home/wlia0047/ar57/wenyu/stark/data/strategy_variants/{strategy_name}_variants_81.csv"
    stark_base_dir = f"/home/wlia0047/ar57/wenyu/stark/data/stark_strategy_{strategy_name}_dataset"

    # Create STaRK directory structure
    qa_dir = os.path.join(stark_base_dir, "qa", "amazon")
    split_dir = os.path.join(qa_dir, "split")
    stark_qa_dir = os.path.join(qa_dir, "stark_qa")

    os.makedirs(stark_qa_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    # Load variants data
    if not os.path.exists(variants_file):
        raise FileNotFoundError(f"Variants file not found: {variants_file}")

    df = pd.read_csv(variants_file)

    # All variant files now use the new format: id, query, answer_ids, answer_ids_source
    stark_df = pd.DataFrame({
        'id': range(len(df)),  # Use sequential IDs starting from 0
        'query': df['query'],
        'answer_ids': df['answer_ids'],
        'query_type': ['variant'] * len(df)
    })

    # Save STaRK format file
    stark_qa_file = os.path.join(stark_qa_dir, "stark_qa.csv")
    stark_df.to_csv(stark_qa_file, index=False)

    # Create split file
    split_file = os.path.join(split_dir, "variants.index")
    with open(split_file, 'w') as f:
        for idx in stark_df['id']:
            f.write(f"{idx}\n")

    return stark_base_dir, stark_qa_file


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='STaRK BM25 Evaluation for Query Variants')
    parser.add_argument('--strategy', type=str, default='kg_query',
                       choices=['original', 'character', 'dependency', 'embedding', 'other', 'typo', 'wordnet', 'synthesized', 'grammar_aware', 'kg_query', 'all'],
                       help='Attack strategy to evaluate (default: kg_query)')
    parser.add_argument('--categories', type=str, default='all',
                       help='Categories to include (default: all)')
    parser.add_argument('--force_rerun', action='store_true',
                       help='Force rerun even if results exist')
    parser.add_argument('--input_csv', type=str, default=None,
                       help='Path to custom input CSV file (overrides default for kg_query)')
    args = parser.parse_args()

    # STaRK project root directory (where stark_qa package is located)
    stark_root = "/home/wlia0047/ar57/wenyu/stark"

    # Change to the STaRK directory
    os.chdir(stark_root)

    # Add current directory to Python path to ensure stark_qa can be imported
    sys.path.insert(0, stark_root)

    # Define strategies to evaluate (excluding original to avoid duplicates with human_generated_eval)
    if args.strategy == 'all':
        strategies = ['character', 'dependency', 'embedding', 'other', 'typo', 'wordnet', 'synthesized', 'grammar_aware']
    else:
        strategies = [args.strategy]

    # Set up common evaluation parameters
    dataset = "amazon"
    categories = [args.categories] if args.categories != 'all' else ['all']
    model = "BM25"
    split = "variants"  # Use variants split for our custom datasets
    save_pred = True

    # Sequential processing settings
    num_workers = 1  # Sequential processing (no parallelism)
    parallel_processing = False  # Disable parallel processing

    print(f"EVALUATING {len(strategies)} STRATEGIES USING EVAL.PY: {', '.join(strategies)}")
    print("=" * 80)

    total_start_time = datetime.now()

    print("Using single-threaded evaluation for all strategies (calling eval.py)")
    print("=" * 80)

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy.upper()} (USING EVAL.PY)")
        print(f"{'='*60}")

        try:
            start_time = datetime.now()  # Define start_time at the beginning of try block

            if strategy == 'original':
                # For original queries, use standard STaRK dataset
                print(f"Using standard STaRK dataset for strategy: {strategy}")
                dataset_root = None  # Will use standard dataset loading
                output_dir = "BM25eval"  # Unified BM25 evaluation directory
                split = "human_generated_eval"  # Use original split for original queries
            elif strategy == 'synthesized':
                # For synthesized queries, create custom dataset from synthesized data (same as test split)
                print(f"Creating dataset for synthesized queries: {strategy}")
                dataset_root, _ = create_strategy_dataset(strategy)
                output_dir = "BM25eval"  # Unified BM25 evaluation directory
                split = "test"  # Use test split for synthesized queries (same evaluation method as test data)
            elif strategy == 'kg_query':
                # For KG queries, use the specific SKB path
                print(f"Setting up for KG queries with custom SKB: {strategy}")
                # First create the QA dataset structure
                dataset_path, stark_qa_path = create_strategy_dataset(strategy, args.input_csv)
                # Then set the dataset_root to the specific SKB path requested by user
                dataset_root = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/attribute_kb"
                output_dir = "BM25eval"
                split = "variants"
                print(f"Using SKB from: {dataset_root}")
            else:
                # Create dataset for variant strategies
                print(f"Creating dataset for strategy: {strategy}")
                dataset_root, _ = create_strategy_dataset(strategy)
                output_dir = "BM25eval"  # Unified BM25 evaluation directory
                split = "variants"  # Use variants split for variant strategies

            # Store variables for exception handling
            current_output_dir = output_dir
            current_split = split

            print(f"QUERY VARIANTS EVALUATION PARAMETERS - {strategy}")
            print("-" * 40)
            print(f"Dataset: {dataset}")
            print(f"Model: {model}")
            print(f"Split: {split}")
            print(f"Dataset root: {dataset_root}")
            print(f"Output directory: {output_dir}")
            print(f"Save predictions: {save_pred}")
            print("-" * 40)
            print("PERFORMANCE NOTES:")
            print("- Using STaRK optimized evaluation engine")
            print("- 🔧 OPTIMIZATION: Reusing shared BM25 model across strategies")
            if strategy == 'original':
                print("- 81 original queries - STANDARD STaRK LOADING")
            elif strategy == 'synthesized':
                print("- 102 synthesized queries - FORCED LOCAL LOADING")
            else:
                print("- 81 variant queries per strategy - FORCED LOCAL LOADING")
            print("- Sequential processing (no parallelism)")
            print(f"- Number of workers: {num_workers}")
            print("- Expected runtime per strategy: 5-10 minutes (calling eval.py)")
            print("-" * 40)

            # Run the evaluation with multi-CPU optimization
            print(f"Starting STaRK BM25 evaluation for {strategy} strategy...")
            print("FORCED LOCAL DATASET LOADING: Using custom dataset with query variants")
            print("SEQUENTIAL PROCESSING: Processing queries one by one")
            print("=" * 60)

            # Set environment variables for optimal multi-threading
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['OMP_NUM_THREADS'] = str(num_workers)  # OpenMP threads
            env['MKL_NUM_THREADS'] = str(num_workers)  # MKL threads
            env['NUMEXPR_NUM_THREADS'] = str(num_workers)  # NumExpr threads
            env['OPENBLAS_NUM_THREADS'] = '1'  # Avoid oversubscription

            # For variant strategies, allow loading from local paths but prevent HF Hub downloads
            # For original strategy, allow full HF access
            if strategy != 'original':
                env['HF_HUB_OFFLINE'] = '1'  # Prevent downloads but allow local cache usage

            # Use single-threaded evaluation for all models (including BM25)
            print("Using single-threaded evaluation (calling eval.py)...")
            
            # For kg_query, we need to pass the CSV file path explicitly because dataset_root points to SKB
            csv_file = None
            if strategy == 'kg_query':
                csv_file = stark_qa_path
            
            result = run_single_threaded_evaluation(
                dataset, model, split, dataset_root, output_dir, env, strategy, csv_file=csv_file, categories=categories, force_rerun=args.force_rerun
            )

            end_time = datetime.now()
            duration = end_time - start_time

            print("=" * 60)
            print(f"✓ {strategy.upper()} EVALUATION COMPLETED SUCCESSFULLY!")
            print(f"Duration: {duration}")
            print("=" * 60)

            # Print output file locations
            output_base_dir = os.path.join(stark_root, current_output_dir, "eval", dataset, model)

            # Generate filename with strategy info (consistent with eval.py)
            split_name = current_split
            if current_split == "variants" and strategy != "original":
                split_name = f"{current_split}_{strategy}"
            elif current_split == "test" and strategy == "synthesized":
                split_name = f"{current_split}_{strategy}"

            eval_csv_path = os.path.join(output_base_dir, f"eval_results_{split_name}.csv")
            eval_metrics_path = os.path.join(output_base_dir, f"eval_metrics_{split_name}.json")
            args_path = os.path.join(output_base_dir, "args.json")

            print(f"{strategy.upper()} EVALUATION RESULTS SAVED TO:")
            print(f"📁 Output directory: {output_base_dir}")
            print(f"📊 Detailed results (CSV): {eval_csv_path}")
            print(f"📈 Summary metrics (JSON): {eval_metrics_path}")
            print(f"⚙️  Configuration (JSON): {args_path}")
            print("=" * 60)

        except subprocess.CalledProcessError as e:
            end_time = datetime.now()
            duration = end_time - start_time

            print("=" * 60)
            print(f"✗ {strategy.upper()} EVALUATION FAILED!")
            print(f"Duration: {duration}")
            print(f"Return code: {e.returncode}")
            print("=" * 60)

        except Exception as e:
            end_time = datetime.now()
            duration = end_time - start_time

            print("=" * 60)
            print(f"✗ {strategy.upper()} UNEXPECTED ERROR!")
            print(f"Duration: {duration}")
            print(f"Error: {e}")
            print("=" * 60)

    # Overall completion
    total_end_time = datetime.now()
    total_duration = total_end_time - total_start_time

    print(f"\n{'='*80}")
    print(f"🎉 ALL {len(strategies)} STRATEGIES EVALUATION COMPLETED!")
    print(f"Total Duration: {total_duration}")
    print(f"Strategies evaluated: {', '.join(strategies)}")
    print(f"{'='*80}")


def load_custom_qa_dataset(csv_path):
    """Load QA dataset directly from CSV file."""
    import pandas as pd
    import ast

    df = pd.read_csv(csv_path)

    class CustomQADataset:
        def __init__(self, dataframe):
            self.data = []
            for idx, row in dataframe.iterrows():
                # Parse answers (stored as string representation of list)
                answers = ast.literal_eval(row['answer_ids']) if isinstance(row['answer_ids'], str) else row['answer_ids']
                # Create tuple: (query, query_id, answers, meta)
                self.data.append((row['query'], int(row['id']), answers, None))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        def get_idx_split(self, test_ratio=1.0):
            """Mock split method for compatibility."""
            import torch
            total = len(self.data)
            indices = torch.tensor(list(range(total)))
            return {'variants': indices, 'test': indices, 'val': indices, 'train': indices}

    return CustomQADataset(df)


def run_single_threaded_evaluation(dataset, model, split, dataset_root, output_dir, env, strategy=None, csv_file=None, categories=None, force_rerun=False):
    """Fallback to original single-threaded evaluation."""
    import shutil

    # Construct the evaluation command with unbuffered output
    base_cmd = ["python", "eval.py",
                "--dataset", dataset,
                "--model", model,
                "--split", split,
                "--output_dir", output_dir,
                "--save_pred",
                "--batch_size", "1",
                "--device", "cpu"]

    # Add dataset_root only if it's not None
    if dataset_root is not None:
        base_cmd.extend(["--dataset_root", dataset_root])

    # Add csv_file if provided
    if csv_file is not None:
        base_cmd.extend(["--csv_file", csv_file])

    # Add strategy if provided
    if strategy:
        base_cmd.extend(["--strategy", strategy])
        
    # Add categories if provided
    if "categories" in locals() and categories:
        if isinstance(categories, list):
            for cat in categories:
                base_cmd.extend(["--categories", cat])
        else:
            base_cmd.extend(["--categories", categories])

    if force_rerun:
        base_cmd.append("--force_rerun")

    if shutil.which('stdbuf'):
        cmd = ["stdbuf", "-oL", "-eL"] + base_cmd  # Line-buffered stdout and stderr
    else:
        cmd = base_cmd

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
        universal_newlines=True,
        env=env
    )

    # Read output character by character for true real-time streaming
    while True:
        ready, _, _ = select.select([process.stdout], [], [], 0.1)
        if ready:
            char = process.stdout.read(1)
            if char:
                sys.stdout.write(char)
                sys.stdout.flush()
            else:
                if process.poll() is not None:
                    break
        else:
            if process.poll() is not None:
                remaining = process.stdout.read()
                if remaining:
                    sys.stdout.write(remaining)
                    sys.stdout.flush()
                break

    return_code = process.poll()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    return type('Result', (), {'returncode': return_code})()


if __name__ == "__main__":
    main()
