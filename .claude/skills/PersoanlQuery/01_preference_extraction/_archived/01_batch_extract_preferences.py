#!/usr/bin/env python3
"""
Stage 1 Batch: Extract preferences for all selected users serially

Automatically reads selected_users.json and processes each user sequentially.
"""
import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Batch: Extract preferences for all users")
    parser.add_argument("--selected-users-file", 
                        default="/fs04/ar57/wenyu/result/personal_query/00_data_preparation/selected_users.json",
                        help="Path to selected_users.json")
    parser.add_argument("--reviews-dir",
                        default="/fs04/ar57/wenyu/result/personal_query/00_data_preparation",
                        help="Directory containing reviews_{USER_ID}.json files")
    parser.add_argument("--output-dir",
                        default="/fs04/ar57/wenyu/result/personal_query/01_preference_extraction",
                        help="Output directory for preferences")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Number of products to process concurrently per user")
    parser.add_argument("--script-path",
                        default="/home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/01_preference_extraction/01_extract_preferences.py",
                        help="Path to single-user extraction script")
    args = parser.parse_args()

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 1 Batch: Preference Extraction for All Users")
    log_with_timestamp("=" * 80)

    if not os.path.exists(args.selected_users_file):
        log_with_timestamp(f"Error: {args.selected_users_file} not found")
        sys.exit(1)

    with open(args.selected_users_file, 'r') as f:
        data = json.load(f)

    # 支持 'users' 或 'selected_users' 两种字段名
    users_data = data.get('users') or data.get('selected_users', [])

    if not users_data:
        log_with_timestamp("Error: No users found in selected_users.json")
        sys.exit(1)

    # 提取user_id（支持字典列表或字符串列表）
    users = []
    for user in users_data:
        if isinstance(user, dict):
            users.append(user.get('user_id'))
        elif isinstance(user, str):
            users.append(user)

    if not users:
        log_with_timestamp("Error: No valid user IDs found")
        sys.exit(1)

    log_with_timestamp(f"Found {len(users)} users to process")
    log_with_timestamp(f"Users: {', '.join(users)}")
    log_with_timestamp("")

    os.makedirs(args.output_dir, exist_ok=True)

    success_count = 0
    failed_users = []

    for idx, user_id in enumerate(users, 1):
        log_with_timestamp("=" * 80)
        log_with_timestamp(f"[{idx}/{len(users)}] Processing user: {user_id}")
        log_with_timestamp("=" * 80)
        
        input_file = os.path.join(args.reviews_dir, f"reviews_{user_id}.json")
        
        if not os.path.exists(input_file):
            log_with_timestamp(f"  Error: Input file not found: {input_file}")
            failed_users.append(user_id)
            continue
        
        cmd = [
            sys.executable,
            "-u",
            args.script_path,
            "--input-file", input_file,
            "--output-dir", args.output_dir,
            "--max-workers", str(args.max_workers)
        ]
        
        log_with_timestamp(f"  Running: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(f"    {line.rstrip()}", flush=True)
            
            stderr_output = process.stderr.read()
            returncode = process.wait()
            
            if returncode == 0:
                log_with_timestamp(f"  ✓ User {user_id} completed successfully")
                success_count += 1
            else:
                log_with_timestamp(f"  ✗ User {user_id} failed with exit code {returncode}")
                if stderr_output:
                    log_with_timestamp(f"  STDERR: {stderr_output[:500]}")
                failed_users.append(user_id)
        
        except Exception as e:
            log_with_timestamp(f"  ✗ User {user_id} error: {e}")
            failed_users.append(user_id)
        
        log_with_timestamp("")

    log_with_timestamp("=" * 80)
    log_with_timestamp("SUMMARY")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Total users: {len(users)}")
    log_with_timestamp(f"Successful: {success_count}")
    log_with_timestamp(f"Failed: {len(failed_users)}")
    
    if failed_users:
        log_with_timestamp(f"\nFailed users: {', '.join(failed_users)}")
    
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("Stage 1 Batch Complete!")
    log_with_timestamp("=" * 80)
    
    if failed_users:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
