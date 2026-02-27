#!/usr/bin/env python3
"""
Stage 0: 从原始 Amazon 评论中提取用户偏好

这个脚本从原始评论数据中提取指定用户的评论，然后使用 LLM 提取偏好实体。

步骤：
1. 读取选中的用户列表
2. 从原始评论文件中提取这些用户的评论
3. 调用 LLM 提取偏好实体
4. 生成 preferences_[USER_ID].json 文件
"""

import json
import os
import argparse
from datetime import datetime
from collections import defaultdict

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def extract_user_reviews(reviews_file, target_users):
    """从原始评论文件中提取目标用户的评论"""
    user_reviews = defaultdict(list)

    log_with_timestamp(f"从 {reviews_file} 提取用户评论...")

    with open(reviews_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"  已处理 {i:,} 条评论...", end='\r')

            try:
                review = json.loads(line)
                user_id = review.get('reviewerID')

                if user_id in target_users:
                    user_reviews[user_id].append(review)
            except:
                continue

    print(f"\n  完成！")
    for user_id, reviews in user_reviews.items():
        log_with_timestamp(f"    用户 {user_id}: {len(reviews)} 条评论")

    return user_reviews

def main():
    parser = argparse.ArgumentParser(description="Extract user reviews from Amazon review data")
    parser.add_argument("--reviews-file", required=True, help="Path to Amazon reviews JSON file")
    parser.add_argument("--users-file", required=True, help="Path to selected users JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for user reviews")

    args = parser.parse_args()

    # 加载选中的用户列表
    log_with_timestamp(f"加载用户列表从 {args.users_file}")
    with open(args.users_file) as f:
        data = json.load(f)

    target_users = set(user['user_id'] for user in data['selected_users'])
    log_with_timestamp(f"  找到 {len(target_users)} 个目标用户")

    # 提取用户评论
    user_reviews = extract_user_reviews(args.reviews_file, target_users)

    # 保存用户评论
    os.makedirs(args.output_dir, exist_ok=True)

    for user_id, reviews in user_reviews.items():
        output_file = os.path.join(args.output_dir, f"reviews_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'total_reviews': len(reviews),
                'reviews': reviews
            }, f, indent=2, ensure_ascii=False)

        log_with_timestamp(f"  保存到 {output_file}")

    log_with_timestamp(f"\n完成！共保存 {len(user_reviews)} 个用户的评论")

if __name__ == "__main__":
    main()
