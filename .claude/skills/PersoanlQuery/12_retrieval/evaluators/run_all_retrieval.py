#!/usr/bin/env python3
"""
Master Script: Run All Retrieval Evaluations (Excluding Reranking)

Runs all retrieval methods (BM25, TF-IDF, Dense, ColBERT, etc.) in both clean and noisy modes.
Excludes all reranking scripts (llm_reranking, traditional_reranking).

Usage:
    python run_all_retrieval.py --user-id A13OFOB1394G31 [--mode both|clean|noisy] [--output-dir PATH]
"""

import argparse
import subprocess
import sys
import os
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Get absolute path to evaluators directory
EVALUATORS_DIR = Path(__file__).parent.absolute()
BASE_DIR = EVALUATORS_DIR.parent

# Default configuration
DEFAULT_USER_ID = "A13OFOB1394G31"
DEFAULT_QUERY_FILE = "/fs04/ar57/wenyu/result/personal_query/09_targeted_noisy_query/noisy_queries_{user_id}.json"
DEFAULT_OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/12_retrieval"

# Retrieval scripts to run (EXCLUDING reranking)
RETRIEVAL_SCRIPTS = [
    # Dense Retrieval (7 scripts)
    "dense_retrieval/12_evaluate_ance.py",
    "dense_retrieval/12_evaluate_bge.py",
    "dense_retrieval/12_evaluate_dense.py",
    "dense_retrieval/12_evaluate_e5.py",
    "dense_retrieval/12_evaluate_minilm.py",
    "dense_retrieval/12_evaluate_mpnet.py",
    "dense_retrieval/12_evaluate_star.py",
    
    # Late Interaction (1 script)
    "late_interaction/12_evaluate_colbert.py",
    
    # Sparse Retrieval (3 scripts)
    "sparse_retrieval/12_evaluate_bm25.py",
    "sparse_retrieval/12_evaluate_dirichlet.py",
    "sparse_retrieval/12_evaluate_tfidf.py",
]


def log_with_timestamp(message: str):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def clean_old_results(output_dir: str, user_id: str):
    log_with_timestamp("=" * 80)
    log_with_timestamp("CLEANING OLD RESULT FILES")
    log_with_timestamp("=" * 80)
    
    patterns = [
        f"retrieval_*_{user_id}.json",
        f"*_candidates_*_{user_id}_*.json",
        f"*_candidates_{user_id}_*.json",
        f"impact_ranking_{user_id}.txt",
        f"comparison_summary_{user_id}.txt"
    ]
    
    deleted_count = 0
    for pattern in patterns:
        file_pattern = os.path.join(output_dir, pattern)
        matching_files = glob.glob(file_pattern)
        
        for file_path in matching_files:
            try:
                os.remove(file_path)
                log_with_timestamp(f"  ✓ Deleted: {os.path.basename(file_path)}")
                deleted_count += 1
            except Exception as e:
                log_with_timestamp(f"  ✗ Failed to delete {os.path.basename(file_path)}: {e}")
    
    if deleted_count == 0:
        log_with_timestamp("  No old result files found")
    else:
        log_with_timestamp(f"\n✓ Deleted {deleted_count} old result file(s)")
    
    log_with_timestamp("")


def run_evaluation(script_path: Path, 
                   query_mode: str, 
                   query_file: str, 
                   output_dir: str, 
                   user_id: str) -> Tuple[int, str]:
    """
    Run a single evaluation script
    
    Returns:
        (exit_code, script_name)
    """
    script_name = script_path.stem
    cmd = [
        sys.executable,
        str(script_path),
        "--query-mode", query_mode,
        "--query-file", query_file,
        "--output-dir", output_dir,
        "--user-id", user_id
    ]
    
    log_with_timestamp(f"Running {script_name} (mode={query_mode})...")
    log_with_timestamp(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per script
        )
        
        if result.returncode == 0:
            log_with_timestamp(f"✓ {script_name} completed successfully (mode={query_mode})")
            return (0, script_name)
        else:
            log_with_timestamp(f"✗ {script_name} failed (mode={query_mode})")
            log_with_timestamp(f"  STDERR: {result.stderr[:500]}")
            return (result.returncode, script_name)
            
    except subprocess.TimeoutExpired:
        log_with_timestamp(f"✗ {script_name} timed out after 1 hour (mode={query_mode})")
        return (124, script_name)
    except Exception as e:
        log_with_timestamp(f"✗ {script_name} raised exception: {e}")
        return (1, script_name)


def generate_impact_analysis(output_dir: str, user_id: str):
    """
    Generate impact analysis comparing clean vs noisy query performance
    """
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("GENERATING IMPACT ANALYSIS...")
    log_with_timestamp("=" * 80)
    
    # Load all results
    results = {}
    pattern = os.path.join(output_dir, f"retrieval_*_{user_id}.json")
    result_files = glob.glob(pattern)
    
    if len(result_files) == 0:
        log_with_timestamp("⚠️  No result files found for analysis")
        return
    
    for file in result_files:
        try:
            with open(file) as f:
                data = json.load(f)
                basename = os.path.basename(file)
                parts = basename.replace('.json', '').split('_')
                method = parts[1]
                mode = parts[2]
                key = f"{method}_{mode}"
                results[key] = data['metrics']
        except Exception as e:
            log_with_timestamp(f"⚠️  Error loading {file}: {e}")
            continue
    
    # Check if we have both clean and noisy results
    has_clean = any('_clean' in k for k in results.keys())
    has_noisy = any('_noisy' in k for k in results.keys())
    
    if not (has_clean and has_noisy):
        log_with_timestamp("⚠️  Need both clean and noisy results for impact analysis")
        log_with_timestamp(f"   Found: clean={has_clean}, noisy={has_noisy}")
        return
    
    # Get unique methods
    methods = list(set([k.split('_')[0] for k in results.keys()]))
    methods.sort()
    
    # Calculate impacts
    impacts = []
    for method in methods:
        clean = results.get(f"{method}_clean", {})
        noisy = results.get(f"{method}_noisy", {})
        
        if not clean or not noisy:
            continue
        
        p1_delta = noisy.get('P@1', 0) - clean.get('P@1', 0)
        map10_delta = noisy.get('MAP@10', 0) - clean.get('MAP@10', 0)
        ndcg10_delta = noisy.get('NDCG@10', 0) - clean.get('NDCG@10', 0)
        mrr10_delta = noisy.get('MRR@10', 0) - clean.get('MRR@10', 0)
        
        avg_impact = (p1_delta + map10_delta + ndcg10_delta + mrr10_delta) / 4
        
        impacts.append({
            'method': method,
            'P@1': p1_delta,
            'MAP@10': map10_delta,
            'NDCG@10': ndcg10_delta,
            'MRR@10': mrr10_delta,
            'avg_impact': avg_impact,
            'clean_ndcg10': clean.get('NDCG@10', 0),
            'noisy_ndcg10': noisy.get('NDCG@10', 0)
        })
    
    if not impacts:
        log_with_timestamp("⚠️  No impact data to analyze")
        return
    
    # Sort by average impact (most negative = most affected)
    impacts.sort(key=lambda x: x['avg_impact'])
    
    # Print ranking
    log_with_timestamp("\n" + "=" * 120)
    log_with_timestamp("RETRIEVAL METHODS RANKED BY SPELLING ERROR IMPACT (Most Affected → Most Robust)")
    log_with_timestamp("=" * 120)
    log_with_timestamp(f"{'Rank':<6} {'Method':<12} {'Avg Impact':<13} {'P@1 Δ':<10} {'MAP@10 Δ':<11} {'NDCG@10 Δ':<12} {'MRR@10 Δ':<11} {'Impact Level':<20}")
    log_with_timestamp("-" * 120)
    
    for i, impact in enumerate(impacts, 1):
        if impact['method'] in ['bm25', 'tfidf', 'dirichlet']:
            category = "Sparse"
        elif impact['method'] == 'colbert':
            category = "Late"
        else:
            category = "Dense"
        
        if impact['avg_impact'] < -0.01:
            indicator = "High Impact"
        elif impact['avg_impact'] < -0.005:
            indicator = "Medium Impact"
        elif impact['avg_impact'] < 0.005:
            indicator = "Low Impact"
        else:
            indicator = "Improved"
        
        log_with_timestamp(
            f"{i:<6} {impact['method']:<12} {impact['avg_impact']:>+12.4f} "
            f"{impact['P@1']:>+9.4f} {impact['MAP@10']:>+10.4f} "
            f"{impact['NDCG@10']:>+11.4f} {impact['MRR@10']:>+10.4f} "
            f"{indicator} ({category})"
        )
    
    # Print key insights
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("KEY INSIGHTS")
    log_with_timestamp("=" * 80)
    
    most_affected = impacts[0]
    least_affected = impacts[-1]
    
    log_with_timestamp(f"\n🔴 MOST AFFECTED METHOD: {most_affected['method'].upper()}")
    log_with_timestamp(f"   Average impact: {most_affected['avg_impact']:+.4f}")
    log_with_timestamp(f"   NDCG@10: {most_affected['clean_ndcg10']:.4f} → {most_affected['noisy_ndcg10']:.4f} ({most_affected['NDCG@10']:+.4f})")
    
    log_with_timestamp(f"\n🟢 MOST ROBUST METHOD: {least_affected['method'].upper()}")
    log_with_timestamp(f"   Average impact: {least_affected['avg_impact']:+.4f}")
    log_with_timestamp(f"   NDCG@10: {least_affected['clean_ndcg10']:.4f} → {least_affected['noisy_ndcg10']:.4f} ({least_affected['NDCG@10']:+.4f})")
    
    # Save detailed analysis to file
    analysis_file = os.path.join(output_dir, f"impact_ranking_{user_id}.txt")
    try:
        with open(analysis_file, 'w') as f:
            f.write("\n" + "="*120 + "\n")
            f.write("RETRIEVAL METHODS RANKED BY SPELLING ERROR IMPACT\n")
            f.write(f"User ID: {user_id}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("="*120 + "\n\n")
            
            f.write("RANKING BY AVERAGE IMPACT:\n")
            f.write("-"*120 + "\n")
            for i, impact in enumerate(impacts, 1):
                if impact['method'] in ['bm25', 'tfidf', 'dirichlet']:
                    cat = "Sparse"
                elif impact['method'] == 'colbert':
                    cat = "Late"
                else:
                    cat = "Dense"
                
                f.write(
                    f"{i:2}. {impact['method']:<12} [{cat:<7}] "
                    f"Avg={impact['avg_impact']:>+7.4f}  "
                    f"P@1={impact['P@1']:>+7.4f}  "
                    f"MAP@10={impact['MAP@10']:>+7.4f}  "
                    f"NDCG@10={impact['NDCG@10']:>+7.4f}  "
                    f"({impact['clean_ndcg10']:.4f} → {impact['noisy_ndcg10']:.4f})\n"
                )
            
            f.write("\n" + "="*120 + "\n")
        
        log_with_timestamp(f"\n✅ Saved detailed impact analysis to: {analysis_file}")
    except Exception as e:
        log_with_timestamp(f"⚠️  Error saving analysis file: {e}")
    
    log_with_timestamp("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run all retrieval evaluations (excluding reranking)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all retrievals for user A13OFOB1394G31 in both modes
  python run_all_retrieval.py --user-id A13OFOB1394G31
  
  # Run only clean mode
  python run_all_retrieval.py --user-id A13OFOB1394G31 --mode clean
  
  # Run only noisy mode
  python run_all_retrieval.py --user-id A13OFOB1394G31 --mode noisy
"""
    )
    
    parser.add_argument("--user-id", 
                       default=DEFAULT_USER_ID,
                       help=f"User ID (default: {DEFAULT_USER_ID})")
    parser.add_argument("--mode",
                       choices=["both", "clean", "noisy"],
                       default="both",
                       help="Query mode: both, clean, or noisy (default: both)")
    parser.add_argument("--output-dir",
                       default=DEFAULT_OUTPUT_DIR,
                       help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--query-file",
                       help="Override query file path (default: auto-generated from user-id)")
    parser.add_argument("--continue-on-error",
                       action="store_true",
                       help="Continue running even if some scripts fail")
    
    args = parser.parse_args()
    
    # Determine query file
    if args.query_file:
        query_file = args.query_file
    else:
        query_file = DEFAULT_QUERY_FILE.format(user_id=args.user_id)
    
    # Verify query file exists
    if not os.path.exists(query_file):
        log_with_timestamp(f"✗ Query file not found: {query_file}")
        log_with_timestamp(f"  Please check the path or provide --query-file")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    clean_old_results(args.output_dir, args.user_id)
    
    if args.mode == "both":
        modes = ["clean", "noisy"]
    else:
        modes = [args.mode]
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("Master Retrieval Evaluation Script")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"User ID: {args.user_id}")
    log_with_timestamp(f"Query file: {query_file}")
    log_with_timestamp(f"Output directory: {args.output_dir}")
    log_with_timestamp(f"Modes to run: {', '.join(modes)}")
    log_with_timestamp(f"Total scripts: {len(RETRIEVAL_SCRIPTS)}")
    log_with_timestamp(f"Total evaluations: {len(RETRIEVAL_SCRIPTS) * len(modes)}")
    log_with_timestamp(f"Continue on error: {args.continue_on_error}")
    log_with_timestamp("")
    
    results = {
        "succeeded": [],
        "failed": []
    }
    
    total_runs = len(RETRIEVAL_SCRIPTS) * len(modes)
    current_run = 0
    
    total_runs = len(RETRIEVAL_SCRIPTS) * len(modes)
    current_run = 0
    
    for script_rel_path in RETRIEVAL_SCRIPTS:
        script_path = EVALUATORS_DIR / script_rel_path
        
        if not script_path.exists():
            log_with_timestamp(f"⚠️  Script not found: {script_path}")
            continue
        
        for mode in modes:
            current_run += 1
            log_with_timestamp(f"\n[{current_run}/{total_runs}] Processing: {script_rel_path} (mode={mode})")
            
            exit_code, script_name = run_evaluation(
                script_path=script_path,
                query_mode=mode,
                query_file=query_file,
                output_dir=args.output_dir,
                user_id=args.user_id
            )
            
            if exit_code == 0:
                results["succeeded"].append(f"{script_name} ({mode})")
            else:
                results["failed"].append(f"{script_name} ({mode})")
                
                if not args.continue_on_error:
                    log_with_timestamp(f"\n✗ Stopping due to failure in {script_name}")
                    log_with_timestamp(f"  Use --continue-on-error to skip failures")
                    break
        
        # If not continuing on error and there was a failure, break outer loop
        if results["failed"] and not args.continue_on_error:
            break
    
    # Print summary
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("SUMMARY")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Total evaluations: {current_run}/{total_runs}")
    log_with_timestamp(f"Succeeded: {len(results['succeeded'])}")
    log_with_timestamp(f"Failed: {len(results['failed'])}")
    
    if results["succeeded"]:
        log_with_timestamp("\n✓ Successful evaluations:")
        for item in results["succeeded"]:
            log_with_timestamp(f"  - {item}")
    
    if results["failed"]:
        log_with_timestamp("\n✗ Failed evaluations:")
        for item in results["failed"]:
            log_with_timestamp(f"  - {item}")
    
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp(f"Results written to: {args.output_dir}")
    log_with_timestamp("=" * 80)
    
    # Generate impact analysis if we ran both modes successfully
    if args.mode == "both" and len(results["succeeded"]) == total_runs:
        try:
            generate_impact_analysis(args.output_dir, args.user_id)
        except Exception as e:
            log_with_timestamp(f"\n⚠️  Error generating impact analysis: {e}")
            log_with_timestamp("   Continuing anyway...")
    
    # Exit with appropriate code
    if results["failed"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
