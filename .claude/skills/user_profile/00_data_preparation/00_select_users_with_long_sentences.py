#!/usr/bin/env python3
"""
Stage 0: 选择高质量用户

从原始 Amazon 评论数据中选择满足以下条件的用户：
- 用户评论的商品中，至少有 min_qualified_ratio (20%) 的商品满足：
  该商品有 >= min_total_users (4) 个用户评论
- 新增：至少有 min_long_review_ratio (20%) 的评论是长评论（≥100词）

Usage:
    python select_users_with_long_sentences.py \
        --reviews-file /path/to/reviews.json \
        --output-dir /path/to/output \
        --min-products 180 \
        --max-products 220 \
        --min-long-review-ratio 0.2
"""

import json
import os
import argparse
import re
from collections import defaultdict
from datetime import datetime


def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def count_long_review(review_text, min_words=100):
    """
    判断评论是否为长评论（≥min_words词）
    返回: (是否为长评论, 词数)
    """
    if not review_text or not isinstance(review_text, str):
        return False, 0

    # 移除HTML标签
    text = re.sub(r'<[^>]+>', ' ', review_text)
    text = text.strip()

    if not text:
        return False, 0

    # 统计词数
    words = text.split()
    word_count = len(words)

    return word_count >= min_words, word_count


def analyze_and_select_users(
    reviews_file,
    output_dir,
    min_qualified_ratio=0.2,
    min_total_users=4,
    target_user_count=10,
    min_products=180,
    max_products=220,
    min_long_review_ratio=0.2
):
    """分析原始评论数据，选择高质量用户"""

    # ==================== 第一遍：统计每个商品的用户数 ====================
    log_with_timestamp("=" * 60)
    log_with_timestamp("第一遍扫描：统计每个商品的用户数...")
    log_with_timestamp("=" * 60)

    asin_to_users = defaultdict(set)

    with open(reviews_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 500000 == 0:
                print(f"  已处理 {i:,} 条评论...", end='\r')

            try:
                review = json.loads(line)
                asin = review.get('asin')
                user_id = review.get('reviewerID')

                if asin and user_id:
                    asin_to_users[asin].add(user_id)
            except:
                continue

    print(f"\n  完成！共 {len(asin_to_users):,} 个商品")

    qualified_asins = {
        asin for asin, users in asin_to_users.items()
        if len(users) >= min_total_users
    }

    log_with_timestamp(f"  有 {len(qualified_asins):,} 个商品有 >= {min_total_users} 个用户")

    # ==================== 第二遍：统计每个用户的指标 ====================
    log_with_timestamp("\n" + "=" * 60)
    log_with_timestamp("第二遍扫描：统计每个用户的指标...")
    log_with_timestamp("=" * 60)

    user_stats = defaultdict(lambda: {
        'review_count': 0,
        'qualified_products': set(),
        'all_products': set(),
        'total_reviews': 0,
        'long_reviews': 0
    })

    with open(reviews_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 500000 == 0:
                print(f"  已处理 {i:,} 条评论...", end='\r')

            try:
                review = json.loads(line)
                user_id = review.get('reviewerID')
                asin = review.get('asin')
                review_text = review.get('reviewText', '')

                if user_id and asin:
                    user_stats[user_id]['review_count'] += 1
                    user_stats[user_id]['all_products'].add(asin)

                    # 判断是否为长评论（只统计有评论文本的评论）
                    if review_text:
                        user_stats[user_id]['total_reviews'] += 1
                        is_long, _ = count_long_review(review_text, min_words=100)
                        if is_long:
                            user_stats[user_id]['long_reviews'] += 1

                    if asin in qualified_asins:
                        user_stats[user_id]['qualified_products'].add(asin)
            except:
                continue

    print(f"\n  完成！共 {len(user_stats):,} 个用户")

    # ==================== 筛选高质量用户 ====================
    log_with_timestamp("\n" + "=" * 60)
    log_with_timestamp("筛选高质量用户...")
    log_with_timestamp("=" * 60)

    log_with_timestamp(f"\n筛选条件:")
    log_with_timestamp(f"  1. 商品数在 [{min_products}, {max_products}] 范围内")
    log_with_timestamp(f"  2. 符合条件商品比例 >= {min_qualified_ratio*100:.0f}%")
    log_with_timestamp(f"  3. 长评论比例 >= {min_long_review_ratio*100:.0f}% (≥100词)")

    qualified_users = []

    for user_id, stats in user_stats.items():
        total_products = len(stats['all_products'])
        qualified_count = len(stats['qualified_products'])
        qualified_ratio = qualified_count / total_products if total_products > 0 else 0

        total_reviews = stats['total_reviews']
        long_reviews = stats['long_reviews']
        long_review_ratio = long_reviews / total_reviews if total_reviews > 0 else 0

        if (min_products <= total_products <= max_products and
            qualified_ratio >= min_qualified_ratio and
            long_review_ratio >= min_long_review_ratio):
            qualified_users.append({
                'user_id': user_id,
                'review_count': stats['review_count'],
                'product_count': total_products,
                'qualified_product_count': qualified_count,
                'qualified_ratio': round(qualified_ratio * 100, 1),
                'long_review_ratio': round(long_review_ratio * 100, 1),
                'total_reviews': total_reviews,
                'long_reviews': long_reviews,
                'qualified_products': list(stats['qualified_products'])[:50]
            })

    qualified_users.sort(key=lambda x: (x['qualified_ratio'], x['long_review_ratio'], x['product_count']), reverse=True)

    log_with_timestamp(f"\n筛选结果:")
    log_with_timestamp(f"  总用户数: {len(user_stats):,}")
    log_with_timestamp(f"  通过所有条件: {len(qualified_users):,}")

    selected_users = qualified_users[:target_user_count]

    if len(selected_users) < target_user_count:
        log_with_timestamp(f"\n⚠️  警告: 只找到 {len(selected_users)} 个符合条件的用户")

    # ==================== 保存结果 ====================
    os.makedirs(output_dir, exist_ok=True)

    all_qualified_file = os.path.join(output_dir, "all_qualified_users.json")
    with open(all_qualified_file, 'w', encoding='utf-8') as f:
        json.dump({
            'selection_criteria': {
                'min_qualified_ratio': min_qualified_ratio,
                'min_total_users': min_total_users,
                'min_products': min_products,
                'max_products': max_products,
                'min_long_review_ratio': min_long_review_ratio,
                'target_user_count': target_user_count
            },
            'total_qualified_users': len(qualified_users),
            'qualified_users': qualified_users
        }, f, indent=2)

    selected_file = os.path.join(output_dir, "selected_users.json")
    with open(selected_file, 'w', encoding='utf-8') as f:
        json.dump({
            'selection_criteria': {
                'min_qualified_ratio': min_qualified_ratio,
                'min_total_users': min_total_users,
                'min_products': min_products,
                'max_products': max_products,
                'min_long_review_ratio': min_long_review_ratio,
                'target_user_count': target_user_count
            },
            'total_qualified_users': len(qualified_users),
            'selected_users': selected_users
        }, f, indent=2)

    log_with_timestamp(f"\n保存了结果:")
    log_with_timestamp(f"  - 所有符合条件的用户: {all_qualified_file}")
    log_with_timestamp(f"  - 选中的用户: {selected_file}")

    print("\n" + "=" * 80)
    print(f"选中的 {len(selected_users)} 个用户:")
    print("=" * 80)
    for i, user in enumerate(selected_users, 1):
        print(f"\n{i}. {user['user_id']}")
        print(f"   评论数: {user['review_count']} | 商品数: {user['product_count']} | 共享商品: {user['qualified_product_count']} ({user['qualified_ratio']}%)")
        print(f"   长评论比例: {user['long_review_ratio']}% ({user['long_reviews']}/{user['total_reviews']})")

    return selected_users


def main():
    parser = argparse.ArgumentParser(description="Select users with long reviews")
    parser.add_argument("--reviews-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-qualified-ratio", type=float, default=0.2)
    parser.add_argument("--min-total-users", type=int, default=4)
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--min-products", type=int, default=180)
    parser.add_argument("--max-products", type=int, default=220)
    parser.add_argument("--min-long-review-ratio", type=float, default=0.2)

    args = parser.parse_args()

    analyze_and_select_users(
        reviews_file=args.reviews_file,
        output_dir=args.output_dir,
        min_qualified_ratio=args.min_qualified_ratio,
        min_total_users=args.min_total_users,
        target_user_count=args.target_count,
        min_products=args.min_products,
        max_products=args.max_products,
        min_long_review_ratio=args.min_long_review_ratio
    )


if __name__ == "__main__":
    main()
