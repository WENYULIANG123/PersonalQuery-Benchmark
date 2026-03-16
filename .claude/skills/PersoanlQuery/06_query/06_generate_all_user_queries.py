#!/usr/bin/env python3
"""
Stage 6: Generate Dual Queries for All Users

主脚本：自动处理所有用户的双重查询生成。
根据 02_processing 目录下的 query.json 文件，批量生成 Target User 和 Mass Market 查询。

Input: 
  - /home/wlia0047/ar57/wenyu/result/personal_query/02_processing/{USER_ID}/query.json (per user)
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/writing_analysis_{USER_ID}.json (per user)
  - Product metadata file (for vocabulary checking)

Output: 
  - /home/wlia0047/ar57/wenyu/result/personal_query/06_query/dual_queries_{USER_ID}.json (per user)
  - /home/wlia0047/ar57/wenyu/result/personal_query/06_query/all_users_summary.json (summary)

Usage:
  # Process all users (default)
  python 06_generate_all_user_queries.py
  
  # Process specific users only
  python 06_generate_all_user_queries.py --user-ids A13OFOB1394G31 A2GJX2KCUSR0EI
  
  # Use custom parameters
  python 06_generate_all_user_queries.py --workers 10 --seed 42
"""

import json
import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def find_users_with_query_files(processing_dir: str) -> List[str]:
    log_with_timestamp(f"Scanning for users in {processing_dir}...")
    
    users_found = []
    
    try:
        for entry in os.listdir(processing_dir):
            user_dir = os.path.join(processing_dir, entry)
            
            if not os.path.isdir(user_dir):
                continue
            
            if entry.startswith('.') or entry == '_archived':
                continue
            
            query_file = os.path.join(user_dir, 'query.json')
            
            if os.path.exists(query_file):
                users_found.append(entry)
                log_with_timestamp(f"  ✓ Found user: {entry}")
            else:
                log_with_timestamp(f"  ✗ Skipped {entry}: no query.json")
    
    except Exception as e:
        log_with_timestamp(f"ERROR scanning directory: {e}")
        return []
    
    log_with_timestamp(f"Found {len(users_found)} users with query.json files")
    return sorted(users_found)

def validate_user_files(
    user_ids: List[str],
    processing_dir: str,
    writing_analysis_dir: str
) -> Dict[str, Dict[str, str]]:
    log_with_timestamp("Validating required files for each user...")
    
    validated_users = {}
    
    for user_id in user_ids:
        query_file = os.path.join(processing_dir, user_id, 'query.json')
        writing_file = os.path.join(writing_analysis_dir, f'writing_analysis_{user_id}.json')
        
        if not os.path.exists(query_file):
            log_with_timestamp(f"  ✗ User {user_id}: query.json NOT FOUND")
            continue
        
        if not os.path.exists(writing_file):
            log_with_timestamp(f"  ✗ User {user_id}: writing_analysis_{user_id}.json NOT FOUND")
            continue
        
        validated_users[user_id] = {
            'query_file': query_file,
            'writing_file': writing_file
        }
        log_with_timestamp(f"  ✓ User {user_id}: all files validated")
    
    log_with_timestamp(f"Validated {len(validated_users)}/{len(user_ids)} users")
    return validated_users

def run_query_generation(
    user_id: str,
    query_file: str,
    writing_file: str,
    output_dir: str,
    meta_file: str,
    workers: int,
    seed: Optional[int] = None
) -> Dict:
    script_path = os.path.join(
        os.path.dirname(__file__),
        '06_generate_dual_queries.py'
    )
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    log_with_timestamp(f"\n[{user_id}] Starting dual query generation...")
    log_with_timestamp(f"[{user_id}]   Input: {query_file}")
    log_with_timestamp(f"[{user_id}]   Writing analysis: {writing_file}")
    
    cmd = [
        sys.executable,
        script_path,
        '--input-file', query_file,
        '--writing-analysis-file', writing_file,
        '--output-dir', output_dir,
        '--meta-file', meta_file,
        '--workers', str(workers)
    ]
    
    if seed is not None:
        cmd.extend(['--seed', str(seed)])
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        
        log_with_timestamp(f"[{user_id}] ✓ Completed successfully")
        return {'success': True, 'user_id': user_id}
        
    except subprocess.CalledProcessError as e:
        log_with_timestamp(f"[{user_id}] ✗ FAILED with return code {e.returncode}")
        return {'success': False, 'user_id': user_id, 'error': str(e)}

def generate_summary(output_dir: str, user_ids: List[str]) -> Dict:
    log_with_timestamp("="*80)
    log_with_timestamp("Generating summary statistics...")
    log_with_timestamp("="*80)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_users': len(user_ids),
        'processed_users': 0,
        'failed_users': [],
        'user_summaries': {},
        'aggregate_stats': {
            'total_queries': 0,
            'total_target_queries': 0,
            'total_mass_market_queries': 0,
            'total_valid_target_error_words': 0,
            'total_valid_mass_market_error_words': 0,
            'target_validation_rate': 0.0,
            'mass_market_validation_rate': 0.0
        }
    }
    
    for user_id in user_ids:
        output_file = os.path.join(output_dir, f'dual_queries_{user_id}.json')
        
        if not os.path.exists(output_file):
            log_with_timestamp(f"  ✗ User {user_id}: output file not found")
            summary['failed_users'].append(user_id)
            continue
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            summary['processed_users'] += 1
            
            user_summary = {
                'user_id': user_id,
                'total_queries': user_data.get('total_queries', 0),
                'successful_target_queries': user_data.get('successful_target_queries', 0),
                'successful_mass_market_queries': user_data.get('successful_mass_market_queries', 0),
                'valid_target_error_words': user_data.get('valid_target_error_words', 0),
                'valid_mass_market_error_words': user_data.get('valid_mass_market_error_words', 0)
            }
            
            summary['aggregate_stats']['total_queries'] += user_summary['total_queries']
            summary['aggregate_stats']['total_target_queries'] += user_summary['successful_target_queries']
            summary['aggregate_stats']['total_mass_market_queries'] += user_summary['successful_mass_market_queries']
            summary['aggregate_stats']['total_valid_target_error_words'] += user_summary['valid_target_error_words']
            summary['aggregate_stats']['total_valid_mass_market_error_words'] += user_summary['valid_mass_market_error_words']
            
            if user_summary['successful_target_queries'] > 0:
                user_summary['target_validation_rate'] = round(
                    user_summary['valid_target_error_words'] / user_summary['successful_target_queries'] * 100,
                    1
                )
            else:
                user_summary['target_validation_rate'] = 0.0
            
            if user_summary['successful_mass_market_queries'] > 0:
                user_summary['mass_market_validation_rate'] = round(
                    user_summary['valid_mass_market_error_words'] / user_summary['successful_mass_market_queries'] * 100,
                    1
                )
            else:
                user_summary['mass_market_validation_rate'] = 0.0
            
            summary['user_summaries'][user_id] = user_summary
            
            log_with_timestamp(
                f"  ✓ User {user_id}: {user_summary['total_queries']} queries, "
                f"TU validation: {user_summary['target_validation_rate']}%, "
                f"MM validation: {user_summary['mass_market_validation_rate']}%"
            )
            
        except Exception as e:
            log_with_timestamp(f"  ✗ User {user_id}: error reading results - {e}")
            summary['failed_users'].append(user_id)
    
    if summary['aggregate_stats']['total_target_queries'] > 0:
        summary['aggregate_stats']['target_validation_rate'] = round(
            summary['aggregate_stats']['total_valid_target_error_words'] / 
            summary['aggregate_stats']['total_target_queries'] * 100,
            1
        )
    
    if summary['aggregate_stats']['total_mass_market_queries'] > 0:
        summary['aggregate_stats']['mass_market_validation_rate'] = round(
            summary['aggregate_stats']['total_valid_mass_market_error_words'] / 
            summary['aggregate_stats']['total_mass_market_queries'] * 100,
            1
        )
    
    summary_file = os.path.join(output_dir, 'all_users_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"Summary saved to {summary_file}")
    
    log_with_timestamp("="*80)
    log_with_timestamp("AGGREGATE STATISTICS")
    log_with_timestamp("="*80)
    log_with_timestamp(f"Processed users: {summary['processed_users']}/{summary['total_users']}")
    log_with_timestamp(f"Total queries: {summary['aggregate_stats']['total_queries']}")
    log_with_timestamp(f"Total target user queries: {summary['aggregate_stats']['total_target_queries']}")
    log_with_timestamp(f"Total mass market queries: {summary['aggregate_stats']['total_mass_market_queries']}")
    log_with_timestamp(f"")
    log_with_timestamp(f"Error Word Validation:")
    log_with_timestamp(f"  Target queries with all error words: {summary['aggregate_stats']['total_valid_target_error_words']}/{summary['aggregate_stats']['total_target_queries']} ({summary['aggregate_stats']['target_validation_rate']}%)")
    log_with_timestamp(f"  Mass market queries with all error words: {summary['aggregate_stats']['total_valid_mass_market_error_words']}/{summary['aggregate_stats']['total_mass_market_queries']} ({summary['aggregate_stats']['mass_market_validation_rate']}%)")
    
    if summary['failed_users']:
        log_with_timestamp(f"\nFailed users: {', '.join(summary['failed_users'])}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(
        description="Generate dual queries for all users with query.json files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--processing-dir',
        default='/home/wlia0047/ar57/wenyu/result/personal_query/02_processing',
        help='Directory containing user subdirectories with query.json (default: %(default)s)'
    )
    parser.add_argument(
        '--writing-analysis-dir',
        default='/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis',
        help='Directory containing writing_analysis_{USER_ID}.json files (default: %(default)s)'
    )
    parser.add_argument(
        '--output-dir',
        default='/home/wlia0047/ar57/wenyu/result/personal_query/06_query',
        help='Output directory (default: %(default)s)'
    )
    parser.add_argument(
        '--meta-file',
        default='/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz',
        help='Product metadata file (default: %(default)s)'
    )
    parser.add_argument(
        '--user-ids',
        nargs='+',
        help='Specific user IDs to process (default: all users with query.json)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of concurrent workers per user (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility (optional)'
    )
    parser.add_argument(
        '--skip-summary',
        action='store_true',
        help='Skip generating summary statistics'
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_with_timestamp("="*80)
    log_with_timestamp("Stage 6: Generate Dual Queries for All Users")
    log_with_timestamp("="*80)
    
    if args.user_ids:
        user_ids = args.user_ids
        log_with_timestamp(f"Processing {len(user_ids)} user(s) specified by --user-ids")
    else:
        user_ids = find_users_with_query_files(args.processing_dir)
    
    if not user_ids:
        log_with_timestamp("ERROR: No users to process!")
        sys.exit(1)
    
    validated_users = validate_user_files(
        user_ids,
        args.processing_dir,
        args.writing_analysis_dir
    )
    
    if not validated_users:
        log_with_timestamp("ERROR: No valid users found!")
        sys.exit(1)
    
    log_with_timestamp("="*80)
    log_with_timestamp("Starting query generation...")
    log_with_timestamp("="*80)
    
    failed_users = []
    
    for user_id, files in validated_users.items():
        result = run_query_generation(
            user_id=user_id,
            query_file=files['query_file'],
            writing_file=files['writing_file'],
            output_dir=args.output_dir,
            meta_file=args.meta_file,
            workers=args.workers,
            seed=args.seed
        )
        
        if not result['success']:
            failed_users.append(user_id)
    
    log_with_timestamp("="*80)
    if failed_users:
        log_with_timestamp(f"WARNING: {len(failed_users)} users failed: {', '.join(failed_users)}")
    else:
        log_with_timestamp("All users completed successfully!")
    
    if not args.skip_summary:
        summary = generate_summary(args.output_dir, list(validated_users.keys()))
        
        if summary['processed_users'] == 0:
            log_with_timestamp("ERROR: No users were successfully processed!")
            sys.exit(1)
    
    log_with_timestamp("="*80)
    log_with_timestamp("ALL PROCESSING COMPLETE!")
    log_with_timestamp("="*80)

if __name__ == '__main__':
    main()
