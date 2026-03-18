#!/usr/bin/env python3
"""
Stage 4C: P3最优模板批量错误提取

专门用P3最优模板进行批量错误提取的脚本。
基于MTSummit 2025论文(arXiv:2505.06004)的发现，对所有选中用户的评论进行高质量错误提取。

P3模板特点:
  • 性能: F1分数 +176% ~ +283% (vs P1)
  • 特点: 明确约束、正确文本保持、最小改动
  • 多语言: 英文、德文、意大利文、瑞典文都验证有效

Input:
  - /home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation/selected_users.json
  - /home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{USER_ID}.json

Output:
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/p3_analysis_{user_id}.json
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/p3_batch_summary.json

Usage:
  # 处理所有选中用户
  python 05_p3_batch_error_analysis.py
  
  # 处理特定用户
  python 05_p3_batch_error_analysis.py --user-ids A13OFOB1394G31 A2GJX2KCUSR0EI
  
  # 限制每个用户的评论数
  python 05_p3_batch_error_analysis.py --max-reviews 50
  
  # 自定义输出目录
  python 05_p3_batch_error_analysis.py --output-dir /path/to/output
  
  # 跳过汇总统计
  python 05_p3_batch_error_analysis.py --skip-summary
"""

import json
import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")


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
    log_with_timestamp(f"Found {len(users)} selected users")
    
    return users


def validate_user_review_files(reviews_dir: str, user_ids: List[str]) -> Set[str]:
    """验证用户的reviews文件存在"""
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
                log_with_timestamp(f"  ✓ User {user_id}: {review_count} reviews")
            except Exception as e:
                missing_users.append(user_id)
                log_with_timestamp(f"  ✗ User {user_id}: ERROR reading file - {e}")
        else:
            missing_users.append(user_id)
            log_with_timestamp(f"  ✗ User {user_id}: FILE NOT FOUND")
    
    if missing_users:
        log_with_timestamp(f"WARNING: {len(missing_users)} users have missing files")
    
    return existing_users


def run_p3_analysis(
    user_ids: List[str],
    reviews_dir: str,
    output_dir: str,
    max_reviews: Optional[int] = None,
    max_workers: int = 20
) -> Dict:
    """运行P3最优prompt模板分析"""
    script_path = os.path.join(os.path.dirname(__file__), "04_p3_error_extraction.py")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"P3 analysis script not found: {script_path}")
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("Running P3 optimal template error analysis (MTSummit 2025)")
    log_with_timestamp("=" * 80)
    
    failed_users = []
    successful_users = []
    
    for user_id in user_ids:
        reviews_file = os.path.join(reviews_dir, f"reviews_{user_id}.json")
        
        log_with_timestamp(f"\n[{user_id}] Processing with P3 template...")
        
        cmd = [
            sys.executable, 
            script_path,
            "--reviews-file", reviews_file,
            "--user-ids", user_id,
            "--output-dir", output_dir,
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
            log_with_timestamp(f"[{user_id}] ✓ P3 analysis completed")
            successful_users.append(user_id)
            
        except subprocess.CalledProcessError as e:
            log_with_timestamp(f"[{user_id}] ✗ FAILED with return code {e.returncode}")
            failed_users.append(user_id)
    
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Processing complete: {len(successful_users)} successful, {len(failed_users)} failed")
    log_with_timestamp("=" * 80)
    
    return {
        "successful_users": successful_users,
        "failed_users": failed_users,
        "success": len(failed_users) == 0
    }


def generate_batch_summary(output_dir: str, user_ids: List[str]) -> Dict:
    """生成批量处理的汇总统计"""
    log_with_timestamp("=" * 80)
    log_with_timestamp("Generating P3 batch analysis summary...")
    log_with_timestamp("=" * 80)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "method": "P3_optimal_template",
        "paper_reference": "MTSummit 2025 - arXiv:2505.06004",
        "total_users": len(user_ids),
        "processed_users": 0,
        "failed_users": [],
        "aggregate_stats": {
            "total_reviews_analyzed": 0,
            "total_words": 0,
            "total_errors_found": 0,
            "overall_error_rate": 0.0,
            "avg_errors_per_review": 0.0,
            "users_with_all_errors": 0,
            "users_with_some_correct": 0
        },
        "user_summaries": {}
    }
    
    for user_id in user_ids:
        output_file = os.path.join(output_dir, f"p3_analysis_{user_id}.json")
        
        if not os.path.exists(output_file):
            log_with_timestamp(f"  ✗ User {user_id}: output file not found")
            summary["failed_users"].append(user_id)
            continue
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            summary["processed_users"] += 1
            
            # 提取关键指标
            review_results = user_data.get('review_results', [])
            processing_stats = user_data.get('processing_stats', {})
            
            user_summary = {
                "user_id": user_id,
                "timestamp": user_data.get('timestamp'),
                "reviews_analyzed": len(review_results),
                "total_errors": processing_stats.get('total_errors_found', 0),
                "total_words": processing_stats.get('total_words', 0),
                "error_rate_per_100_words": 0.0,
                "reviews_with_errors": processing_stats.get('reviews_with_errors', 0),
                "reviews_without_errors": processing_stats.get('reviews_without_errors', 0),
                "avg_errors_per_review": 0.0,
                "status": user_data.get('processing_stats', {}).get('status', 'unknown')
            }
            
            # 计算错误率
            if user_summary["total_words"] > 0:
                user_summary["error_rate_per_100_words"] = round(
                    user_summary["total_errors"] / user_summary["total_words"] * 100, 2
                )
            
            if user_summary["reviews_analyzed"] > 0:
                user_summary["avg_errors_per_review"] = round(
                    user_summary["total_errors"] / user_summary["reviews_analyzed"], 2
                )
            
            # 汇总到总体统计
            summary["aggregate_stats"]["total_reviews_analyzed"] += user_summary["reviews_analyzed"]
            summary["aggregate_stats"]["total_words"] += user_summary["total_words"]
            summary["aggregate_stats"]["total_errors_found"] += user_summary["total_errors"]
            
            if user_summary["reviews_without_errors"] == 0 and user_summary["reviews_analyzed"] > 0:
                summary["aggregate_stats"]["users_with_all_errors"] += 1
            
            if user_summary["reviews_without_errors"] > 0:
                summary["aggregate_stats"]["users_with_some_correct"] += 1
            
            summary["user_summaries"][user_id] = user_summary
            
            log_with_timestamp(
                f"  ✓ User {user_id}: {user_summary['reviews_analyzed']} reviews, "
                f"{user_summary['total_errors']} errors, "
                f"{user_summary['error_rate_per_100_words']}/100 words"
            )
            
        except Exception as e:
            log_with_timestamp(f"  ✗ User {user_id}: error reading results - {e}")
            summary["failed_users"].append(user_id)
    
    # 计算总体统计
    if summary["aggregate_stats"]["total_words"] > 0:
        summary["aggregate_stats"]["overall_error_rate"] = round(
            summary["aggregate_stats"]["total_errors_found"] / 
            summary["aggregate_stats"]["total_words"] * 100, 2
        )
    
    if summary["aggregate_stats"]["total_reviews_analyzed"] > 0:
        summary["aggregate_stats"]["avg_errors_per_review"] = round(
            summary["aggregate_stats"]["total_errors_found"] / 
            summary["aggregate_stats"]["total_reviews_analyzed"], 2
        )
    
    # 保存汇总
    summary_file = os.path.join(output_dir, "p3_batch_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"\nSummary saved to {summary_file}")
    
    # 打印汇总统计
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("P3 BATCH ANALYSIS SUMMARY")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Processed users: {summary['processed_users']}/{summary['total_users']}")
    log_with_timestamp(f"Total reviews analyzed: {summary['aggregate_stats']['total_reviews_analyzed']}")
    log_with_timestamp(f"Total words analyzed: {summary['aggregate_stats']['total_words']}")
    log_with_timestamp(f"Total errors found: {summary['aggregate_stats']['total_errors_found']}")
    log_with_timestamp(f"Overall error rate: {summary['aggregate_stats']['overall_error_rate']}/100 words")
    log_with_timestamp(f"Avg errors per review: {summary['aggregate_stats']['avg_errors_per_review']}")
    log_with_timestamp(f"Users with all reviews having errors: {summary['aggregate_stats']['users_with_all_errors']}")
    log_with_timestamp(f"Users with some correct reviews: {summary['aggregate_stats']['users_with_some_correct']}")
    
    if summary["failed_users"]:
        log_with_timestamp(f"\nFailed users: {', '.join(summary['failed_users'])}")
    
    log_with_timestamp("=" * 80)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="P3最优模板批量错误提取 (MTSummit 2025)",
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
    
    # 处理参数
    parser.add_argument(
        "--max-reviews",
        type=int,
        help="Maximum number of reviews to analyze per user (default: all)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Maximum concurrent workers (default: 20)"
    )
    
    # 跳过汇总
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip generating batch summary statistics"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 4C: P3 Optimal Template Batch Error Analysis")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Paper: MTSummit 2025 - arXiv:2505.06004")
    log_with_timestamp(f"Method: P3 Optimal Template (F1: +176% ~ +283%)")
    
    # 确定要处理的用户
    if args.user_ids:
        user_ids = args.user_ids
        log_with_timestamp(f"Processing {len(user_ids)} user(s) specified by --user-ids")
    else:
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
    
    # 运行P3分析
    result = run_p3_analysis(
        user_ids=user_ids_to_process,
        reviews_dir=args.reviews_dir,
        output_dir=args.output_dir,
        max_reviews=args.max_reviews,
        max_workers=args.max_workers
    )
    
    if not result["success"]:
        log_with_timestamp("WARNING: Some users failed during analysis")
    
    # 生成汇总统计
    if not args.skip_summary:
        summary = generate_batch_summary(args.output_dir, user_ids_to_process)
        
        if summary["processed_users"] == 0:
            log_with_timestamp("ERROR: No users were successfully processed!")
            sys.exit(1)
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("ALL PROCESSING COMPLETE!")
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    main()
