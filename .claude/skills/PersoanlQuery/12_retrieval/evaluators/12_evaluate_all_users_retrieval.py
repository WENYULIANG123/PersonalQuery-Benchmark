#!/usr/bin/env python3
"""
Master Script: Run All Retrieval Evaluations for All Users

This script automatically discovers all users who have completed Stage 6 (query generation)
and runs all retrieval evaluations for each user.

Usage:
    python 12_evaluate_all_users_retrieval.py [--mode both|clean|noisy] [--continue-on-error]
    
Example:
    # Run all retrieval methods for all users
    python 12_evaluate_all_users_retrieval.py
    
    # Run only clean mode for all users
    python 12_evaluate_all_users_retrieval.py --mode clean
    
    # Continue even if some users fail
    python 12_evaluate_all_users_retrieval.py --continue-on-error
"""

import argparse
import subprocess
import sys
import os
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

# Get absolute path
SCRIPT_DIR = Path(__file__).parent.absolute()
RUN_ALL_SCRIPT = SCRIPT_DIR / "run_all_retrieval.py"

# Default paths
STAGE6_DIR = "/fs04/ar57/wenyu/result/personal_query/06_query"
STAGE9_DIR = "/fs04/ar57/wenyu/result/personal_query/09_targeted_noisy_query"
OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/12_retrieval"

# Log file
LOG_FILE = "/home/wlia0047/ar57/wenyu/stage12_all_users_batch.log"


def setup_logging():
    """Setup logging to both file and console"""
    import logging
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def find_users_with_queries() -> List[str]:
    """Find all users who have completed Stage 6 query generation"""
    users = []
    
    # Look for dual_queries files in stage 6 directory
    pattern = os.path.join(STAGE6_DIR, "dual_queries_*.json")
    query_files = glob.glob(pattern)
    
    for file_path in query_files:
        filename = os.path.basename(file_path)
        # Extract user ID from filename: dual_queries_USERID.json
        if filename.startswith("dual_queries_") and filename.endswith(".json"):
            user_id = filename[13:-5]  # Remove "dual_queries_" prefix and ".json" suffix
            users.append(user_id)
    
    return sorted(users)


def check_user_has_noisy_queries(user_id: str) -> bool:
    """Check if user has Stage 9 noisy queries"""
    noisy_query_file = os.path.join(STAGE9_DIR, f"noisy_queries_{user_id}.json")
    return os.path.exists(noisy_query_file)


def run_user_evaluation(user_id: str, mode: str, continue_on_error: bool, logger) -> Tuple[int, str]:
    """Run retrieval evaluation for a single user"""
    
    # Check if user has noisy queries (required for evaluation)
    if not check_user_has_noisy_queries(user_id):
        logger.warning(f"User {user_id} missing Stage 9 noisy queries - skipping")
        return 1, f"Missing noisy queries for {user_id}"
    
    # Build command
    cmd = [
        sys.executable,
        str(RUN_ALL_SCRIPT),
        "--user-id", user_id,
        "--mode", mode,
        "--output-dir", OUTPUT_DIR
    ]
    
    if continue_on_error:
        cmd.append("--continue-on-error")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        if process.stdout:
            for line in process.stdout:
                logger.info(f"[{user_id}] {line.rstrip()}")
        
        # Wait for completion
        exit_code = process.wait()
        
        if exit_code == 0:
            return 0, f"Successfully completed {user_id}"
        else:
            return exit_code, f"Failed to complete {user_id} (exit code: {exit_code})"
            
    except Exception as e:
        return 1, f"Error running evaluation for {user_id}: {str(e)}"


def generate_summary_report(results: Dict[str, List[str]], total_time: float, logger):
    """Generate final summary report"""
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_users": len(results["succeeded"]) + len(results["failed"]) + len(results["skipped"]),
        "succeeded": results["succeeded"],
        "failed": results["failed"],
        "skipped": results["skipped"],
        "total_time_minutes": round(total_time / 60, 2)
    }
    
    # Save to JSON
    summary_file = os.path.join(OUTPUT_DIR, "all_users_evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Total users processed: {summary['total_users']}")
    logger.info(f"Succeeded: {len(results['succeeded'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    logger.info(f"Skipped: {len(results['skipped'])}")
    logger.info(f"Total time: {summary['total_time_minutes']} minutes")
    
    if results["succeeded"]:
        logger.info("\nSuccessful users:")
        for user_id in results["succeeded"]:
            logger.info(f"  ✓ {user_id}")
    
    if results["failed"]:
        logger.info("\nFailed users:")
        for user_id in results["failed"]:
            logger.info(f"  ✗ {user_id}")
    
    if results["skipped"]:
        logger.info("\nSkipped users (missing noisy queries):")
        for user_id in results["skipped"]:
            logger.info(f"  ⚠ {user_id}")
    
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info(f"Full log saved to: {LOG_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all retrieval evaluations for all users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all retrievals for all users in both modes
  python 12_evaluate_all_users_retrieval.py
  
  # Run only clean mode for all users
  python 12_evaluate_all_users_retrieval.py --mode clean
  
  # Continue even if some users fail
  python 12_evaluate_all_users_retrieval.py --continue-on-error
  
  # Process specific users only
  python 12_evaluate_all_users_retrieval.py --user-ids A13OFOB1394G31 A1GYEGLX3P2Y7P
"""
    )
    
    parser.add_argument("--mode",
                       choices=["both", "clean", "noisy"],
                       default="both",
                       help="Query mode: both, clean, or noisy (default: both)")
    parser.add_argument("--continue-on-error",
                       action="store_true",
                       help="Continue processing users even if some fail")
    parser.add_argument("--user-ids",
                       nargs="+",
                       help="Specific user IDs to process (default: all users)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Start time
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info("STAGE 12: BATCH RETRIEVAL EVALUATION FOR ALL USERS")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Continue on error: {args.continue_on_error}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Log file: {LOG_FILE}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find users to process
    if args.user_ids:
        users_to_process = args.user_ids
        logger.info(f"Processing specific users: {', '.join(users_to_process)}")
    else:
        users_to_process = find_users_with_queries()
        logger.info(f"Found {len(users_to_process)} users with Stage 6 queries")
    
    if not users_to_process:
        logger.error("No users found to process!")
        sys.exit(1)
    
    logger.info(f"Users to process: {', '.join(users_to_process)}")
    logger.info("")
    
    # Process each user
    results = {
        "succeeded": [],
        "failed": [],
        "skipped": []
    }
    
    for i, user_id in enumerate(users_to_process, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING USER {i}/{len(users_to_process)}: {user_id}")
        logger.info(f"{'='*80}")
        
        # Check if user has noisy queries first
        if not check_user_has_noisy_queries(user_id):
            logger.warning(f"User {user_id} missing Stage 9 noisy queries - skipping")
            results["skipped"].append(user_id)
            continue
        
        exit_code, message = run_user_evaluation(
            user_id=user_id,
            mode=args.mode,
            continue_on_error=args.continue_on_error,
            logger=logger
        )
        
        if exit_code == 0:
            results["succeeded"].append(user_id)
            logger.info(f"✓ {message}")
        else:
            results["failed"].append(user_id)
            logger.error(f"✗ {message}")
            
            if not args.continue_on_error:
                logger.error("Stopping due to error (use --continue-on-error to continue)")
                break
    
    # Calculate total time
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Generate final report
    generate_summary_report(results, total_time, logger)
    
    # Exit with appropriate code
    if results["failed"] and not args.continue_on_error:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()