#!/usr/bin/env python3
"""Stage 6: Stage5->Template 批量生成（无 LLM）。"""

import json
import os
import sys
import importlib.util
from datetime import datetime
from typing import Dict, List, Optional

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def find_users_with_profiles(profile_dir: str) -> List[str]:
    log_with_timestamp(f"Scanning for users in {profile_dir}...")
    
    users_found = []
    
    try:
        for entry in os.listdir(profile_dir):
            if not entry.startswith('linguistic_profile_') or not entry.endswith('.json'):
                continue
            user_id = entry.replace('linguistic_profile_', '').replace('.json', '')
            users_found.append(user_id)
            log_with_timestamp(f"  ✓ Found user: {user_id}")
    
    except Exception as e:
        log_with_timestamp(f"ERROR scanning directory: {e}")
        return []
    
    log_with_timestamp(f"Found {len(users_found)} users with linguistic profiles")
    return sorted(users_found)

def validate_user_files(user_ids: List[str], profile_dir: str) -> Dict[str, Dict[str, str]]:
    log_with_timestamp("Validating required profile files for each user...")
    
    validated_users = {}
    
    for user_id in user_ids:
        profile_file = os.path.join(profile_dir, f'linguistic_profile_{user_id}.json')
        if not os.path.exists(profile_file):
            log_with_timestamp(f"  ✗ User {user_id}: linguistic_profile_{user_id}.json NOT FOUND")
            continue
        
        validated_users[user_id] = {
            'profile_file': profile_file
        }
        log_with_timestamp(f"  ✓ User {user_id}: all files validated")
    
    log_with_timestamp(f"Validated {len(validated_users)}/{len(user_ids)} users")
    return validated_users

def _load_user_complexity_score(profile_file: str) -> float:
    with open(profile_file, 'r', encoding='utf-8') as f:
        profile = json.load(f)

    axis = profile.get('complexity_axis_features', {})
    if isinstance(axis, dict) and axis:
        subordination = float(axis.get('subordination', 0.0))
        negation = float(axis.get('negation', 0.0))
        coordination = float(axis.get('coordination', 0.0))
        length_depth = float(axis.get('length_depth', 0.0))
        return (1.6 * subordination) + (1.2 * negation) + (1.0 * coordination) + (0.6 * length_depth)

    counts = profile.get('complexity_rule_based', {}).get('sentence_counts', {})
    low = float(counts.get('low', 0))
    medium = float(counts.get('medium', 0))
    high = float(counts.get('high', 0))
    total = low + medium + high
    if total <= 0:
        return 0.0
    return (2.0 * high + 1.0 * medium + 0.0 * low) / total


def assign_levels_normal(validated_users: Dict[str, Dict[str, str]], ratios: Dict[str, float]) -> Dict[str, str]:
    scored = []
    for user_id, files in validated_users.items():
        score = _load_user_complexity_score(files['profile_file'])
        scored.append((user_id, score))

    scored.sort(key=lambda x: (x[1], x[0]))
    n = len(scored)
    if n == 0:
        return {}

    low_n = int(round(n * float(ratios.get('low', 0.16))))
    high_n = int(round(n * float(ratios.get('high', 0.16))))
    low_n = max(0, min(low_n, n))
    high_n = max(0, min(high_n, n - low_n))
    medium_n = n - low_n - high_n

    level_map: Dict[str, str] = {}
    for user_id, _ in scored[:low_n]:
        level_map[user_id] = 'low'
    for user_id, _ in scored[low_n:low_n + medium_n]:
        level_map[user_id] = 'medium'
    for user_id, _ in scored[low_n + medium_n:]:
        level_map[user_id] = 'high'

    return level_map


def run_query_generation(template_module, user_id: str, profile_file: str, output_dir: str, seed: Optional[int] = None, forced_level: Optional[str] = None) -> Dict:
    log_with_timestamp(f"\n[{user_id}] Starting template query generation...")
    log_with_timestamp(f"[{user_id}]   Profile: {profile_file}")
    
    try:
        out_fp = template_module.run_generation(
            linguistic_profile_file=profile_file,
            output_dir=output_dir,
            seed=seed,
            forced_level=forced_level,
        )
        
        log_with_timestamp(f"[{user_id}] ✓ Completed successfully: {out_fp}")
        return {'success': True, 'user_id': user_id}
        
    except Exception as e:
        log_with_timestamp(f"[{user_id}] ✗ FAILED: {e}")
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
            'total_valid_target_error_words': 0,
            'target_validation_rate': 0.0
        }
    }
    
    for user_id in user_ids:
        output_file = os.path.join(output_dir, f'queries_{user_id}.json')
        
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
                'valid_target_error_words': user_data.get('successful_target_queries', 0)
            }
            
            summary['aggregate_stats']['total_queries'] += user_summary['total_queries']
            summary['aggregate_stats']['total_target_queries'] += user_summary['successful_target_queries']
            summary['aggregate_stats']['total_valid_target_error_words'] += user_summary['valid_target_error_words']
            
            if user_summary['successful_target_queries'] > 0:
                user_summary['target_validation_rate'] = round(
                    user_summary['valid_target_error_words'] / user_summary['successful_target_queries'] * 100,
                    1
                )
            else:
                user_summary['target_validation_rate'] = 0.0
            
            summary['user_summaries'][user_id] = user_summary
            
            log_with_timestamp(
                f"  ✓ User {user_id}: {user_summary['total_queries']} queries, "
                f"TU validation: {user_summary['target_validation_rate']}%"
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
    log_with_timestamp(f"")
    log_with_timestamp(f"Error Word Validation:")
    log_with_timestamp(f"  Target queries with all error words: {summary['aggregate_stats']['total_valid_target_error_words']}/{summary['aggregate_stats']['total_target_queries']} ({summary['aggregate_stats']['target_validation_rate']}%)")
    
    if summary['failed_users']:
        log_with_timestamp(f"\nFailed users: {', '.join(summary['failed_users'])}")
    
    return summary

def main():
    config = {
        'profile_dir': '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis',
        'output_dir': '/fs04/ar57/wenyu/result/personal_query/06_query',
        'user_ids': None,
        'seed': None,
        'skip_summary': False,
        'level_distribution': {
            'low': 0.16,
            'medium': 0.68,
            'high': 0.16,
        },
    }

    script_path = os.path.join(os.path.dirname(__file__), '06_generate_template_queries.py')
    spec = importlib.util.spec_from_file_location('stage6_template_queries', script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load module from {script_path}')
    template_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template_module)

    os.makedirs(config['output_dir'], exist_ok=True)
    
    log_with_timestamp("="*80)
    log_with_timestamp("Stage 6: Generate Template Queries from Stage5 (No LLM)")
    log_with_timestamp("="*80)
    
    if config['user_ids']:
        user_ids = config['user_ids']
        log_with_timestamp(f"Processing {len(user_ids)} user(s) specified by --user-ids")
    else:
        user_ids = find_users_with_profiles(config['profile_dir'])
    
    if not user_ids:
        log_with_timestamp("ERROR: No users to process!")
        sys.exit(1)
    
    validated_users = validate_user_files(
        user_ids,
        config['profile_dir']
    )
    
    if not validated_users:
        log_with_timestamp("ERROR: No valid users found!")
        sys.exit(1)

    forced_levels = assign_levels_normal(validated_users, config['level_distribution'])
    low_count = sum(1 for v in forced_levels.values() if v == 'low')
    medium_count = sum(1 for v in forced_levels.values() if v == 'medium')
    high_count = sum(1 for v in forced_levels.values() if v == 'high')
    log_with_timestamp(f"Assigned complexity levels -> low={low_count}, medium={medium_count}, high={high_count}")
    
    log_with_timestamp("="*80)
    log_with_timestamp("Starting query generation...")
    log_with_timestamp("="*80)
    
    failed_users = []
    
    for user_id, files in validated_users.items():
        result = run_query_generation(
            template_module=template_module,
            user_id=user_id,
            profile_file=files['profile_file'],
            output_dir=config['output_dir'],
            seed=config['seed'],
            forced_level=forced_levels.get(user_id),
        )
        
        if not result['success']:
            failed_users.append(user_id)
    
    log_with_timestamp("="*80)
    if failed_users:
        log_with_timestamp(f"WARNING: {len(failed_users)} users failed: {', '.join(failed_users)}")
    else:
        log_with_timestamp("All users completed successfully!")
    
    if not config['skip_summary']:
        summary = generate_summary(config['output_dir'], list(validated_users.keys()))
        
        if summary['processed_users'] == 0:
            log_with_timestamp("ERROR: No users were successfully processed!")
            sys.exit(1)
    
    log_with_timestamp("="*80)
    log_with_timestamp("ALL PROCESSING COMPLETE!")
    log_with_timestamp("="*80)

if __name__ == '__main__':
    main()
