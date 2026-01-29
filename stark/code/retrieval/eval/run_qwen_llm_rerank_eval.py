#!/usr/bin/env python3
"""
Qwen-based LLM Reranking Evaluation Script

This script performs intelligent two-stage evaluation:
1. First stage: Qwen-based retrieval (VSS with qwen embeddings) - SKIPPED if results exist
2. Second stage: LLM reranking (using FREE CodeLlama-7B for intelligent reranking)

Smart Features:
- ✅ Automatically detects existing VSS results and skips redundant computation
- 💰 Completely FREE (no API costs)
- 🖥️ Runs locally on CPU/GPU
- 🤖 Open source CodeLlama model for advanced reranking
- 🚀 Optimized workflow: Retrieval → Smart Skip → Reranking

Usage:
    python qwen_llm_rerank_eval.py --strategy all          # Evaluate all strategies
    python qwen_llm_rerank_eval.py --strategy character     # Evaluate specific strategy
    python qwen_llm_rerank_eval.py --skip_qwen              # Force skip VSS step
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime


def create_strategy_dataset_structure(strategy_name):
    """Helper to ensure directory structure exists for kg_query (simplified version)."""
    import pandas as pd
    import os.path as osp

    if strategy_name != 'kg_query':
        return

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
        print(f"Warning: Query file not found: {query_file}")
        return

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


def run_qwen_retrieval(strategy, output_base="output"):
    """Run Qwen-based retrieval for a specific strategy."""
    import sys
    print(f"🔍 Running Qwen-based retrieval for strategy: {strategy}")
    sys.stdout.flush()

    cmd = [
        "python", "eval.py",
        "--dataset", "amazon",
        "--model", "VSS",
        "--emb_model", "alibaba-nlp/gte-base-en-v1.5",
        "--output_dir", output_base,
        "--split", "variants" if strategy != "original" else "human_generated_eval",
        "--save_pred",
        "--batch_size", "256",
        "--device", "cpu"
    ]

    # Add strategy parameter and dataset_root for variants
    if strategy != "original":
        cmd.extend(["--strategy", strategy])
        if strategy == "kg_query":
             # Ensure dataset structure exists
            create_strategy_dataset_structure(strategy)
            cmd.extend(["--dataset_root", "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/attribute_kb"])
            # Pass csv_file explicitly if eval.py supports it or relies on dataset_structure
            cmd.extend(["--csv_file", "/home/wlia0047/ar57/wenyu/stark/data/stark_strategy_kg_query_dataset/qa/amazon/stark_qa/stark_qa.csv"])
        else:
            cmd.extend(["--dataset_root", "/home/wlia0047/ar57/wenyu/stark/data/stark_variants_dataset"])

    # Convert all to strings
    cmd = [str(x) for x in cmd]

    cmd_str = ' '.join(cmd)
    print(f"Command: {cmd_str}")
    print(f"📝 Executing Qwen retrieval for {strategy}...")
    sys.stdout.flush()

    # Run command using os.system for better output visibility
    import os
    exit_code = os.system(cmd_str)

    if exit_code != 0:
        print(f"❌ Qwen retrieval failed for {strategy} (exit code: {exit_code})")
        sys.stdout.flush()
        return False

    print(f"✅ Qwen retrieval completed for {strategy}")
    sys.stdout.flush()
    return True

def check_vss_results_exist(strategy, output_base="output"):
    """Check if VSS results already exist for a strategy."""
    import os.path as osp

    # The eval.py script saves results to LLMbasedeval/{dataset}/{model}/{emb_model}/
    # For VSS model with alibaba-nlp/gte-base-en-v1.5 embeddings
    base_path = osp.join("LLMbasedeval", "amazon", "VSS", "alibaba-nlp", "gte-base-en-v1.5")

    # Determine the expected VSS results file path
    if strategy == "original":
        results_file = osp.join(base_path, "eval_results_human_generated_eval.csv")
    else:
        results_file = osp.join(base_path, f"eval_results_variants_{strategy}.csv")

    return osp.exists(results_file), results_file

def run_llm_reranking(strategy, output_base="output"):
    """Run LLM reranking with CodeLlama for a specific strategy."""
    import sys
    print(f"🤖 Running LLM reranking with CodeLlama for strategy: {strategy}")
    sys.stdout.flush()

    # Check if VSS results exist
    vss_exists, vss_file = check_vss_results_exist(strategy, output_base)
    if vss_exists:
        print(f"✅ Found existing VSS results: {vss_file}")
        print("🔄 Proceeding with LLM reranking...")
        sys.stdout.flush()
    else:
        print(f"⚠️  VSS results not found: {vss_file}")
        print("❌ Cannot proceed with LLM reranking without VSS results")
        sys.stdout.flush()
        return False

    cmd = [
        "python", "eval.py",
        "--dataset", "amazon",
        "--model", "LLMReranker",
        "--emb_model", "alibaba-nlp/gte-base-en-v1.5",
        "--output_dir", output_base,
        "--emb_dir", "emb/",
        "--llm_model", "huggingface/codellama/CodeLlama-7b-Instruct-hf",  # Keep instruction-tuned but smaller
        "--split", "variants" if strategy != "original" else "human_generated_eval",
        "--save_pred",
        "--llm_topk", "10",  # Reduce to 10 to strictly avoid OOM
        "--max_retry", "3",  # Original STaRK uses max_cnt=3
        "--device", "cuda",
        "--force_rerun"  # Always force rerun regardless of existing results
    ]

    # Add strategy parameter and dataset_root for variants
    if strategy != "original":
        cmd.extend(["--strategy", strategy])
        if strategy == "kg_query":
             cmd.extend(["--dataset_root", "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/attribute_kb"])
             cmd.extend(["--csv_file", "/home/wlia0047/ar57/wenyu/stark/data/stark_strategy_kg_query_dataset/qa/amazon/stark_qa/stark_qa.csv"])
        else:
            cmd.extend(["--dataset_root", "/home/wlia0047/ar57/wenyu/stark/data/stark_variants_dataset"])

    # Convert all to strings
    cmd = [str(x) for x in cmd]

    cmd_str = ' '.join(cmd)
    print(f"Command: {cmd_str}")
    print(f"🤖 Executing LLM reranking for {strategy}...")
    sys.stdout.flush()

    # Run command using os.system for better output visibility
    import os
    exit_code = os.system(cmd_str)

    if exit_code != 0:
        print(f"❌ LLM reranking failed for {strategy} (exit code: {exit_code})")
        sys.stdout.flush()
        return False

    print(f"✅ LLM reranking completed for {strategy}")
    sys.stdout.flush()
    return True

def main():
    """Main function to run Qwen-based LLM reranking evaluation."""
    import sys
    import os

    # Ensure output is not buffered
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("🚀 Starting Qwen LLM Reranking Script...")
    sys.stdout.flush()

    # Simple test execution
    print("Testing script execution...")
    sys.stdout.flush()
    print("Script loaded successfully!")
    sys.stdout.flush()

    parser = argparse.ArgumentParser(description='Qwen-based LLM Reranking Evaluation')
    parser.add_argument('--strategy', type=str, default='all',
                       choices=['original', 'character', 'embedding', 'other', 'typo', 'wordnet', 'dependency', 'error_aware', 'kg_query', 'all'],
                       help='Strategy to evaluate (default: all)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Base output directory')
    parser.add_argument('--skip_qwen', action='store_true',
                       help='Skip Qwen retrieval step (use existing VSS results if available)')

    args = parser.parse_args()
    print(f"📋 Parsed arguments: strategy={args.strategy}, output_dir={args.output_dir}, skip_qwen={args.skip_qwen}")
    sys.stdout.flush()

    # Change to STaRK directory (script is in code/ subdirectory)
    stark_root = "/home/wlia0047/ar57/wenyu/stark"
    os.chdir(stark_root)
    print(f"📂 Changed to STaRK directory: {stark_root}")

    # Set optimal PyTorch memory management environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking CUDA launches

    # Aggressive GPU memory cleanup at program start
    import torch
    if torch.cuda.is_available():
        print("🧹 Performing initial GPU memory cleanup...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"📊 Initial GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total, "
              f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB allocated")
    sys.stdout.flush()

    # Set custom output directory for LLM reranking results
    llm_output_dir = "/home/wlia0047/ar57/wenyu/stark/LLMRankereval"

    # Define strategies
    if args.strategy == 'all':
        strategies = ['original', 'character', 'embedding', 'other', 'typo', 'wordnet', 'dependency', 'error_aware']
    else:
        strategies = [args.strategy]

    print(f"📊 Will evaluate {len(strategies)} strategies: {', '.join(strategies)}")
    print(f"💾 LLM results will be saved to: {llm_output_dir}")
    sys.stdout.flush()

    total_start_time = datetime.now()
    results_summary = []

    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'='*60}")
        print(f"🎯 STRATEGY {i}/{len(strategies)}: {strategy.upper()}")
        print('='*60)
        sys.stdout.flush()

        strategy_start_time = datetime.now()

        # Check if VSS results already exist
        vss_exists, vss_file = check_vss_results_exist(strategy, args.output_dir)
        # FORCE RETRIEVAL: Ignore vss_exists to ensure we get fresh embeddings
        skip_vss = args.skip_qwen # or vss_exists

        # Step 1: Qwen-based retrieval (skip if results exist or explicitly requested)
        if not skip_vss:
            print(f"🔍 STEP 1: Qwen-based retrieval for {strategy}")
            sys.stdout.flush()
            qwen_success = run_qwen_retrieval(strategy, args.output_dir)
            if not qwen_success:
                print(f"⚠️  Skipping {strategy} due to Qwen retrieval failure")
                sys.stdout.flush()
                continue
        else:
            if vss_exists:
                print(f"⏭️  Skipping Qwen retrieval for {strategy} (existing VSS results found)")
                print(f"📁 VSS results: {vss_file}")
            else:
                print(f"⏭️  Skipping Qwen retrieval for {strategy} (explicitly requested)")
            sys.stdout.flush()

        # Step 2: LLM reranking
        print(f"🤖 STEP 2: LLM reranking for {strategy}")
        sys.stdout.flush()
        llm_success = run_llm_reranking(strategy, llm_output_dir)

        strategy_end_time = datetime.now()
        strategy_duration = strategy_end_time - strategy_start_time

        if llm_success:
            results_summary.append({
                'strategy': strategy,
                'status': 'completed',
                'duration': str(strategy_duration)
            })
            print(f"✅ Strategy {strategy} completed successfully in {strategy_duration}")
            sys.stdout.flush()
        else:
            results_summary.append({
                'strategy': strategy,
                'status': 'failed',
                'duration': str(strategy_duration)
            })
            print(f"❌ Strategy {strategy} failed")
            print(f"🛑 Fatal error: Stopping evaluation due to LLM reranking failure for strategy '{strategy}'")
            sys.stdout.flush()
            exit(1)  # Exit the entire program on any LLM failure

    # Final summary
    total_end_time = datetime.now()
    total_duration = total_end_time - total_start_time

    print(f"\n{'='*80}")
    print("🎉 QWEN-BASED LLM RERANKING EVALUATION COMPLETED")
    print('='*80)
    print(f"⏱️  Total time: {total_duration}")
    print(f"📊 Strategies processed: {len(results_summary)}")
    print(f"✅ Successful: {len([r for r in results_summary if r['status'] == 'completed'])}")
    print(f"❌ Failed: {len([r for r in results_summary if r['status'] == 'failed'])}")
    sys.stdout.flush()

    print("\n📋 Detailed Results:")
    for result in results_summary:
        status_icon = "✅" if result['status'] == 'completed' else "❌"
        print(f"  {status_icon} {result['strategy']}: {result['status']} ({result['duration']})")
    sys.stdout.flush()

    print("\n💾 Results saved in:")
    print(f"  {llm_output_dir}/eval/amazon/LLMReranker/meta-llama/CodeLlama-7b-hf/")
    print("  ├── eval_metrics_variants_*.json (for each strategy)")
    print("  └── eval_results_variants_*.csv (for each strategy)")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
