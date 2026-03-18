#!/usr/bin/env python3
"""
Stage 4: Extract Character-Level Errors for All Selected Users

主脚本：自动处理所有选中用户的评论，提取拼写和语法错误。
根据 selected_users.json 和 reviews_{USER_ID}.json 文件，批量提取所有用户的评论错误。

支持两种分析方法：
  1. character_level (默认) - 传统字符级错误检测
  2. p3_optimal - P3最优prompt模板 (基于MTSummit 2025论文 arXiv:2505.06004)

Input: 
  - /home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation/selected_users.json
  - /home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{USER_ID}.json (per user)
  - Product metadata file (for brand name extraction, character_level method only)

Output: 
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/writing_analysis_{user_id}.json (per user)
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/p3_analysis_{user_id}.json (p3_optimal method)
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/all_users_summary.json (summary)

Usage:
  # Process all selected users with character-level method (default)
  python 04_extract_all_user_errors.py
  
  # Process all selected users with P3 optimal template method
  python 04_extract_all_user_errors.py --method p3_optimal
  
  # Process specific users only
  python 04_extract_all_user_errors.py --user-ids A13OFOB1394G31 A2GJX2KCUSR0EI
  
  # Limit reviews per user
  python 04_extract_all_user_errors.py --max-reviews 50
  
  # Use custom metadata file (character-level method only)
  python 04_extract_all_user_errors.py --metadata-file /path/to/metadata.json
  
  # P3 method with specific users
  python 04_extract_all_user_errors.py --method p3_optimal --user-ids A13OFOB1394G31 --max-reviews 20
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
    """打印带时间戳的日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def load_selected_users(selected_users_file: str) -> List[str]:
    """
    加载选中的用户列表
    
    Args:
        selected_users_file: selected_users.json 文件路径
        
    Returns:
        用户ID列表
    """
    log_with_timestamp(f"Loading selected users from {selected_users_file}...")
    
    with open(selected_users_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    users = data.get('users', [])
    log_with_timestamp(f"Found {len(users)} selected users: {', '.join(users)}")
    
    return users

def validate_user_review_files(reviews_dir: str, user_ids: List[str]) -> Set[str]:
    """
    验证所有用户的 reviews_{USER_ID}.json 文件存在
    
    Args:
        reviews_dir: 评论文件目录
        user_ids: 要验证的用户ID列表
        
    Returns:
        存在的用户ID集合
    """
    log_with_timestamp(f"Validating user review files in {reviews_dir}...")
    
    existing_users = set()
    missing_users = []
    
    for user_id in user_ids:
        review_file = os.path.join(reviews_dir, f"reviews_{user_id}.json")
        
        if os.path.exists(review_file):
            try:
                with open(review_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                review_count = len(user_data.get('results', user_data.get('reviews', [])))
                existing_users.add(user_id)
                log_with_timestamp(f"  ✓ User {user_id}: {review_count} reviews (file: reviews_{user_id}.json)")
            except Exception as e:
                missing_users.append(user_id)
                log_with_timestamp(f"  ✗ User {user_id}: ERROR reading file - {e}")
        else:
            missing_users.append(user_id)
            log_with_timestamp(f"  ✗ User {user_id}: FILE NOT FOUND (reviews_{user_id}.json)")
    
    if missing_users:
        log_with_timestamp(f"WARNING: {len(missing_users)} users missing or have invalid review files")
    
    return existing_users

def run_p3_analysis(
    user_ids: List[str],
    reviews_dir: str,
    output_dir: str,
    max_reviews: Optional[int] = None,
    max_workers: int = 20
) -> Dict:
    """Run P3 optimal template comprehensive analysis"""
    script_path = os.path.join(os.path.dirname(__file__), "04_p3_comprehensive_analysis.py")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"P3 analysis script not found: {script_path}")
    
    log_with_timestamp("="*80)
    log_with_timestamp("Running P3 optimal template comprehensive error analysis...")
    log_with_timestamp("(Method: MTSummit 2025 - arXiv:2505.06004)")
    log_with_timestamp("="*80)
    
    cmd = [
        sys.executable, 
        script_path,
        "--reviews-dir", reviews_dir,
        "--analysis-dir", output_dir,
        "--user-ids"] + user_ids + [
        "--max-workers", str(max_workers)
    ]
    
    if max_reviews:
        cmd.extend(["--max-reviews", str(max_reviews)])
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        log_with_timestamp("All users completed P3 analysis successfully!")
        return {"success": True, "failed_users": []}
        
    except subprocess.CalledProcessError as e:
        log_with_timestamp(f"P3 analysis FAILED with return code {e.returncode}")
        return {"success": False, "failed_users": user_ids}

def run_character_level_analysis(
    user_ids: List[str],
    reviews_dir: str,
    metadata_file: str,
    output_dir: str,
    max_reviews: Optional[int] = None,
    max_workers: int = 50
) -> Dict:
    script_path = os.path.join(os.path.dirname(__file__), "04_character_level_errors.py")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Character-level analysis script not found: {script_path}")
    
    log_with_timestamp("="*80)
    log_with_timestamp("Running character-level error analysis...")
    log_with_timestamp("="*80)
    
    failed_users = []
    
    for user_id in user_ids:
        reviews_file = os.path.join(reviews_dir, f"reviews_{user_id}.json")
        
        log_with_timestamp(f"\n[{user_id}] Processing user...")
        log_with_timestamp(f"[{user_id}] Reviews file: {reviews_file}")
        
        cmd = [
            sys.executable, 
            script_path,
            "--reviews-file", reviews_file,
            "--user-ids", user_id,
            "--output-dir", output_dir,
            "--metadata-file", metadata_file,
            "--max-workers", str(max_workers)
        ]
        
        if max_reviews:
            cmd.extend(["--max-reviews", str(max_reviews)])
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            log_with_timestamp(f"[{user_id}] ✓ Completed successfully")
            
        except subprocess.CalledProcessError as e:
            log_with_timestamp(f"[{user_id}] ✗ FAILED with return code {e.returncode}")
            failed_users.append(user_id)
    
    log_with_timestamp("="*80)
    if failed_users:
        log_with_timestamp(f"WARNING: {len(failed_users)} users failed: {', '.join(failed_users)}")
        return {"success": False, "failed_users": failed_users}
    else:
        log_with_timestamp("All users completed successfully!")
        return {"success": True, "failed_users": []}

def generate_summary(output_dir: str, user_ids: List[str]) -> Dict:
    """
    生成所有用户的汇总统计
    
    Args:
        output_dir: 输出目录
        user_ids: 用户ID列表
        
    Returns:
        汇总统计字典
    """
    log_with_timestamp("="*80)
    log_with_timestamp("Generating summary statistics...")
    log_with_timestamp("="*80)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_users": len(user_ids),
        "processed_users": 0,
        "failed_users": [],
        "user_summaries": {},
        "aggregate_stats": {
            "total_reviews_analyzed": 0,
            "total_words_analyzed": 0,
            "total_character_errors": 0,
            "overall_error_rate": 0.0,
            "error_type_distribution": {},
            "severity_distribution": {
                "low": 0,
                "medium": 0,
                "high": 0
            }
        }
    }
    
    error_type_counter = {}
    
    for user_id in user_ids:
        output_file = os.path.join(output_dir, f"writing_analysis_{user_id}.json")
        
        if not os.path.exists(output_file):
            log_with_timestamp(f"  ✗ User {user_id}: output file not found")
            summary["failed_users"].append(user_id)
            continue
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            summary["processed_users"] += 1
            
            # 提取关键指标
            user_summary = {
                "user_id": user_id,
                "reviews_analyzed": user_data.get("reviews_analyzed", 0),
                "total_words": user_data.get("total_words", 0),
                "total_character_errors": user_data.get("total_character_errors", 0),
                "character_error_rate": user_data.get("character_error_rate", 0.0),
                "average_severity": user_data.get("average_severity", 0.0),
                "top_error_types": []
            }
            
            # 汇总到总体统计
            summary["aggregate_stats"]["total_reviews_analyzed"] += user_summary["reviews_analyzed"]
            summary["aggregate_stats"]["total_words_analyzed"] += user_summary["total_words"]
            summary["aggregate_stats"]["total_character_errors"] += user_summary["total_character_errors"]
            
            # 错误类型分布
            error_types = user_data.get("error_type_distribution", {})
            for error_type, count in error_types.items():
                error_type_counter[error_type] = error_type_counter.get(error_type, 0) + count
            
            # Top 3 错误类型
            top_types = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:3]
            user_summary["top_error_types"] = [{"type": t, "count": c} for t, c in top_types]
            
            # 严重度分布
            severity_dist = user_data.get("severity_distribution", {})
            for severity, count in severity_dist.items():
                if severity in summary["aggregate_stats"]["severity_distribution"]:
                    summary["aggregate_stats"]["severity_distribution"][severity] += count
            
            summary["user_summaries"][user_id] = user_summary
            
            log_with_timestamp(
                f"  ✓ User {user_id}: {user_summary['total_character_errors']} errors "
                f"in {user_summary['reviews_analyzed']} reviews "
                f"(rate: {user_summary['character_error_rate']:.2f}/100 words)"
            )
            
        except Exception as e:
            log_with_timestamp(f"  ✗ User {user_id}: error reading results - {e}")
            summary["failed_users"].append(user_id)
    
    # 计算总体错误率
    if summary["aggregate_stats"]["total_words_analyzed"] > 0:
        summary["aggregate_stats"]["overall_error_rate"] = round(
            summary["aggregate_stats"]["total_character_errors"] / 
            summary["aggregate_stats"]["total_words_analyzed"] * 100,
            2
        )
    
    # 错误类型分布
    summary["aggregate_stats"]["error_type_distribution"] = dict(
        sorted(error_type_counter.items(), key=lambda x: x[1], reverse=True)
    )
    
    # 保存汇总
    summary_file = os.path.join(output_dir, "all_users_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"Summary saved to {summary_file}")
    
    # 打印汇总统计
    log_with_timestamp("="*80)
    log_with_timestamp("AGGREGATE STATISTICS")
    log_with_timestamp("="*80)
    log_with_timestamp(f"Processed users: {summary['processed_users']}/{summary['total_users']}")
    log_with_timestamp(f"Total reviews analyzed: {summary['aggregate_stats']['total_reviews_analyzed']}")
    log_with_timestamp(f"Total words analyzed: {summary['aggregate_stats']['total_words_analyzed']}")
    log_with_timestamp(f"Total character errors: {summary['aggregate_stats']['total_character_errors']}")
    log_with_timestamp(f"Overall error rate: {summary['aggregate_stats']['overall_error_rate']}/100 words")
    
    log_with_timestamp("\nTop Error Types:")
    top_error_types = list(summary['aggregate_stats']['error_type_distribution'].items())[:5]
    for i, (error_type, count) in enumerate(top_error_types, 1):
        log_with_timestamp(f"  {i}. {error_type}: {count}")
    
    log_with_timestamp("\nSeverity Distribution:")
    for severity, count in summary['aggregate_stats']['severity_distribution'].items():
        log_with_timestamp(f"  {severity}: {count}")
    
    if summary["failed_users"]:
        log_with_timestamp(f"\nFailed users: {', '.join(summary['failed_users'])}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(
        description="Extract character-level errors for all selected users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 输入文件
    parser.add_argument(
        "--selected-users-file",
        default="/home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation/selected_users.json",
        help="Path to selected_users.json (default: %(default)s)"
    )
    parser.add_argument(
        "--reviews-dir",
        default="/home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation",
        help="Directory containing reviews_{USER_ID}.json files (default: %(default)s)"
    )
    parser.add_argument(
        "--metadata-file",
        default="/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json",
        help="Path to product metadata file (default: %(default)s)"
    )
    
    # 输出目录
    parser.add_argument(
        "--output-dir",
        default="/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis",
        help="Output directory (default: %(default)s)"
    )
    
    # 用户选择
    parser.add_argument(
        "--user-ids",
        nargs="+",
        help="Specific user IDs to process (default: all users from selected_users.json)"
    )
    
    parser.add_argument(
        "--method",
        choices=["character_level", "p3_optimal"],
        default="character_level",
        help="Analysis method: character_level (default) or p3_optimal (MTSummit 2025 template)"
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        help="Maximum number of reviews to analyze per user (default: all)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=50,
        help="Maximum concurrent workers (default: 50)"
    )
    
    # 跳过汇总
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip generating summary statistics"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_with_timestamp("="*80)
    log_with_timestamp("Stage 4: Extract Character-Level Errors for All Users")
    log_with_timestamp("="*80)
    
    # 确定要处理的用户
    if args.user_ids:
        # 用户指定了具体用户ID
        user_ids = args.user_ids
        log_with_timestamp(f"Processing {len(user_ids)} user(s) specified by --user-ids")
    else:
        # 从 selected_users.json 读取
        user_ids = load_selected_users(args.selected_users_file)
    
    if not user_ids:
        log_with_timestamp("ERROR: No users to process!")
        sys.exit(1)
    
    # 验证用户存在
    existing_users = validate_user_review_files(args.reviews_dir, user_ids)
    
    if not existing_users:
        log_with_timestamp("ERROR: No valid users found with review files!")
        sys.exit(1)
    
    user_ids_to_process = sorted(list(existing_users))
    
    log_with_timestamp(f"Selected analysis method: {args.method}")
    
    if args.method == "p3_optimal":
        result = run_p3_analysis(
            user_ids=user_ids_to_process,
            reviews_dir=args.reviews_dir,
            output_dir=args.output_dir,
            max_reviews=args.max_reviews,
            max_workers=args.max_workers
        )
        
        if not result["success"]:
            log_with_timestamp("ERROR: P3 optimal template analysis failed!")
            sys.exit(1)
    else:
        result = run_character_level_analysis(
            user_ids=user_ids_to_process,
            reviews_dir=args.reviews_dir,
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            max_reviews=args.max_reviews,
            max_workers=args.max_workers
        )
        
        if not result["success"]:
            log_with_timestamp("ERROR: Character-level analysis failed!")
            sys.exit(1)
    
    # 生成汇总统计
    if not args.skip_summary:
        summary = generate_summary(args.output_dir, user_ids_to_process)
        
        if summary["processed_users"] == 0:
            log_with_timestamp("ERROR: No users were successfully processed!")
            sys.exit(1)
    
    log_with_timestamp("="*80)
    log_with_timestamp("ALL PROCESSING COMPLETE!")
    log_with_timestamp("="*80)

if __name__ == "__main__":
    main()
