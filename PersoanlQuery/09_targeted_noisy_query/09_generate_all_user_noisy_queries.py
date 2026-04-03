#!/usr/bin/env python3
"""
Stage 9: Generate Noisy Queries for All Users

主脚本：自动处理所有用户的噪声查询生成。
根据 06_query 目录下的 queries_{USER_ID}.json 文件，批量生成带噪声的查询。

Input: 
  - /home/wlia0047/ar57/wenyu/result/personal_query/06_query/queries_{USER_ID}.json (per user)
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/writing_analysis_{USER_ID}.json (per user)

Output: 
  - /home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query/noisy_queries_{USER_ID}.json (per user)
  - /home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query/all_users_summary.json (summary)

Usage:
  # Process all users (default)
  python 09_generate_all_user_noisy_queries.py
  
  # Process specific users only
  python 09_generate_all_user_noisy_queries.py --user-ids A13OFOB1394G31 A2GJX2KCUSR0EI
  
  # Use custom parameters
  python 09_generate_all_user_noisy_queries.py --seed 42 --skip-summary
"""

import json
import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional
import tempfile
import shutil

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def find_users_with_query_files(query_dir: str, source: str) -> List[str]:
    log_with_timestamp(f"Scanning for users in {query_dir}...")
    
    users_found = []
    
    try:
        for filename in os.listdir(query_dir):
            if source == 'stage7':
                if not filename.endswith('_interative_query.json'):
                    continue
                user_id = filename.replace('_interative_query.json', '')
            else:
                if not filename.startswith('queries_') or not filename.endswith('.json'):
                    continue
                user_id = filename.replace('queries_', '').replace('.json', '')
            
            if user_id and user_id != 'summary':
                users_found.append(user_id)
                log_with_timestamp(f"  ✓ Found user: {user_id}")
    
    except Exception as e:
        log_with_timestamp(f"ERROR scanning directory: {e}")
        return []
    
    log_with_timestamp(f"Found {len(users_found)} users with query files")
    return sorted(users_found)

def validate_user_files(
    user_ids: List[str],
    query_dir: str,
    writing_analysis_dir: str,
    source: str
) -> Dict[str, Dict[str, str]]:
    """Validate that all required files exist for each user"""
    log_with_timestamp("Validating required files for each user...")
    
    validated_users = {}
    
    for user_id in user_ids:
        if source == 'stage7':
            query_file = os.path.join(query_dir, f'{user_id}_interative_query.json')
        else:
            query_file = os.path.join(query_dir, f'queries_{user_id}.json')
        writing_file = os.path.join(writing_analysis_dir, f'writing_analysis_{user_id}.json')
        
        if not os.path.exists(query_file):
            log_with_timestamp(f"  ✗ User {user_id}: query file NOT FOUND ({query_file})")
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

def run_noisy_query_generation(
    user_id: str,
    query_file: str,
    writing_file: str,
    output_dir: str,
    source: str,
    seed: Optional[int] = None
) -> Dict:
    """Run the noisy query generation for a single user using a modified script"""
    script_path = os.path.join(
        os.path.dirname(__file__),
        '09_generate_noisy_queries_single.py'
    )
    
    log_with_timestamp(f"\n[{user_id}] Starting noisy query generation...")
    log_with_timestamp(f"[{user_id}]   Input: {query_file}")
    log_with_timestamp(f"[{user_id}]   Writing analysis: {writing_file}")
    
    try:
        # Run the main script with --user argument
        original_script_path = os.path.join(os.path.dirname(__file__), '09_generate_noisy_queries.py')
        cmd = [
            sys.executable,
            original_script_path,
            '--user', user_id,
            '--source', source,
            '--output-dir', output_dir
        ]
        
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
    except Exception as e:
        log_with_timestamp(f"[{user_id}] ✗ ERROR: {str(e)}")
        return {'success': False, 'user_id': user_id, 'error': str(e)}

def generate_summary(output_dir: str, user_ids: List[str], source: str) -> Dict:
    """Generate summary statistics for all processed users"""
    log_with_timestamp("Generating summary statistics...")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_users': len(user_ids),
        'users': []
    }
    
    total_queries = 0
    total_modified = 0
    total_personalized = 0
    
    for user_id in user_ids:
        if source == 'stage7':
            user_file = os.path.join(output_dir, f'iterative_noisy_query_{user_id}.json')
        else:
            user_file = os.path.join(output_dir, f'noisy_queries_{user_id}.json')
        
        if not os.path.exists(user_file):
            log_with_timestamp(f"  ✗ User {user_id}: output file not found")
            continue
        
        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            # The data is at the top level, not under 'summary'
            user_summary = {
                'user_id': user_id,
                'total_queries': user_data.get('total_queries', 0),
                'modified_queries': user_data.get('modified_queries', 0),
                'unmodified_queries': user_data.get('unmodified_queries', 0),
                'modification_rate': user_data.get('modification_rate', 0.0),
                'personalized_injections': user_data.get('personalized_injections', 0)
            }
            
            summary['users'].append(user_summary)
            
            total_queries += user_summary['total_queries']
            total_modified += user_summary['modified_queries']
            total_personalized += user_summary['personalized_injections']
            
            log_with_timestamp(f"  ✓ User {user_id}: {user_summary['total_queries']} queries, "
                             f"modification rate: {user_summary['modification_rate']:.1f}%")
        
        except Exception as e:
            log_with_timestamp(f"  ✗ User {user_id}: error reading results: {e}")
    
    summary['aggregate'] = {
        'total_queries': total_queries,
        'total_modified': total_modified,
        'total_personalized': total_personalized,
        'overall_modification_rate': total_modified / total_queries if total_queries > 0 else 0.0
    }
    
    if source == 'stage7':
        summary_file = os.path.join(output_dir, 'iterative_noisy_query_all_users_summary.json')
    else:
        summary_file = os.path.join(output_dir, 'all_users_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"Summary saved to {summary_file}")
    return summary

def main():
    parser = argparse.ArgumentParser(
        description='Generate noisy queries for all users',
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--query-dir',
        type=str,
        default=None,
        help='Directory containing input query files'
    )
    
    parser.add_argument(
        '--writing-analysis-dir',
        type=str,
        default='/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis',
        help='Directory containing writing_analysis_{USER_ID}.json files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for noisy queries'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='stage6',
        choices=['stage6', 'stage7'],
        help='Query source: stage6 dual queries or stage7 iterative queries'
    )
    
    parser.add_argument(
        '--user-ids',
        nargs='+',
        type=str,
        help='Specific user IDs to process (default: all users with query files)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--skip-summary',
        action='store_true',
        help='Skip generating summary statistics'
    )
    
    args = parser.parse_args()

    if args.query_dir is None:
        if args.source == 'stage7':
            args.query_dir = '/home/wlia0047/ar57/wenyu/result/personal_query/07_iterative_refinement'
        else:
            args.query_dir = '/home/wlia0047/ar57/wenyu/result/personal_query/06_query'

    if args.output_dir is None:
        args.output_dir = '/home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query'
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 9: Generate Noisy Queries for All Users")
    log_with_timestamp("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find or validate users
    if args.user_ids:
        log_with_timestamp(f"Processing {len(args.user_ids)} user(s) specified by --user-ids")
        user_ids = args.user_ids
    else:
        user_ids = find_users_with_query_files(args.query_dir, args.source)
        if not user_ids:
            log_with_timestamp("No users found with query files!")
            return 1
    
    # Validate files
    validated_users = validate_user_files(
        user_ids,
        args.query_dir,
        args.writing_analysis_dir,
        args.source
    )
    
    if not validated_users:
        log_with_timestamp("No users passed validation!")
        return 1
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("Starting noisy query generation...")
    log_with_timestamp("=" * 80)
    
    # Process each user
    results = []
    for user_id, files in validated_users.items():
        result = run_noisy_query_generation(
            user_id=user_id,
            query_file=files['query_file'],
            writing_file=files['writing_file'],
            output_dir=args.output_dir,
            source=args.source,
            seed=args.seed
        )
        results.append(result)
    
    # Summary statistics
    successful_users = [r['user_id'] for r in results if r['success']]
    failed_users = [r['user_id'] for r in results if not r['success']]
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("All users completed!")
    log_with_timestamp("=" * 80)
    
    if not args.skip_summary and successful_users:
        summary = generate_summary(args.output_dir, successful_users, args.source)
        
        log_with_timestamp("=" * 80)
        log_with_timestamp("AGGREGATE STATISTICS")
        log_with_timestamp("=" * 80)
        log_with_timestamp(f"Processed users: {len(successful_users)}/{len(validated_users)}")
        
        if 'aggregate' in summary:
            agg = summary['aggregate']
            log_with_timestamp(f"Total queries: {agg['total_queries']}")
            log_with_timestamp(f"Total modified: {agg['total_modified']}")
            log_with_timestamp(f"Total personalized: {agg['total_personalized']}")
            log_with_timestamp(f"")
            log_with_timestamp(f"Overall modification rate: {agg['overall_modification_rate']*100:.1f}%")
    
    if failed_users:
        log_with_timestamp(f"\n⚠️  Failed users: {', '.join(failed_users)}")
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("ALL PROCESSING COMPLETE!")
    log_with_timestamp("=" * 80)
    
    return 0 if not failed_users else 1

if __name__ == "__main__":
    exit(main())
