#!/usr/bin/env python3
"""
Stage 0: Batch Data Preparation with User Selection
从大量用户中筛选出高质量用户，然后加载这些用户的评论数据

功能：
1. 筛选符合条件的用户（有元数据、评论数在范围内）
2. 对筛选出的每个用户加载评论数据（target + other reviews）
3. 输出所有用户的 reviews_{USER_ID}.json 和 selected_users.json

输入：
- review-file: 原始评论文件
- meta-file: 产品元数据文件

输出：
- reviews_{USER_ID}.json: 每个用户的评论数据
- selected_users.json: 筛选出的用户列表
"""
import os
import sys
import json
import gzip
import argparse
from datetime import datetime
from typing import Dict, List, Set
from collections import defaultdict

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def load_metadata_asins(meta_file: str) -> tuple:
    """加载所有有元数据的商品 ASIN 和类目映射"""
    log_with_timestamp(f"Loading metadata from {meta_file}...")
    valid_asins = set()
    asin_to_category = {}

    try:
        open_func = gzip.open if meta_file.endswith('.gz') else open
        with open_func(meta_file, 'rt', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
                for item in data:
                    asin = item.get('asin')
                    cat = item.get('category')
                    if asin and cat and isinstance(cat, list) and len(cat) > 0:
                        valid_asins.add(asin)
                        # 取倒数第二级类目（更具体）
                        if len(cat) >= 2:
                            main_category = cat[-2]
                        else:
                            main_category = cat[-1]
                        asin_to_category[asin] = main_category
            else:
                for line in f:
                    try:
                        item = json.loads(line)
                        asin = item.get('asin')
                        cat = item.get('category')
                        if asin and cat and isinstance(cat, list) and len(cat) > 0:
                            valid_asins.add(asin)
                            # 取倒数第二级类目（更具体）
                            if len(cat) >= 2:
                                main_category = cat[-2]
                            else:
                                main_category = cat[-1]
                            asin_to_category[asin] = main_category
                    except:
                        continue
    except Exception as e:
        log_with_timestamp(f"Error loading metadata: {e}")

    log_with_timestamp(f"Total valid products in metadata: {len(valid_asins)}")
    log_with_timestamp(f"Total products with category info: {len(asin_to_category)}")
    return valid_asins, asin_to_category

def count_product_reviews(review_file: str, valid_asins: Set[str], min_total_reviews: int = 5, min_other_reviews: int = 4) -> Dict[str, Dict]:
    """
    统计每个商品的评论数

    返回: {asin: {'total_reviews': N, 'reviewers': {user_id, ...}}}
    只保留满足条件的商品：
    - 总评论数 >= min_total_reviews
    - 其他用户评论数 >= min_other_reviews (即总评论数 - 1 >= min_other_reviews)
    """
    log_with_timestamp(f"Counting reviews per product...")
    log_with_timestamp(f"Criteria: >= {min_total_reviews} total reviews, >= {min_other_reviews} other reviews")

    product_stats = {}  # {asin: {'total_reviews': N, 'reviewers': set()}}

    with gzip.open(review_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                review = json.loads(line)
                asin = review.get('asin')
                user_id = review.get('reviewerID')

                if asin and asin in valid_asins and user_id:
                    if asin not in product_stats:
                        product_stats[asin] = {'total_reviews': 0, 'reviewers': set()}
                    product_stats[asin]['total_reviews'] += 1
                    product_stats[asin]['reviewers'].add(user_id)
            except:
                continue

    # 筛选符合条件的商品
    valid_products = {}
    for asin, stats in product_stats.items():
        total = stats['total_reviews']
        # 其他用户评论数 = 总评论数 - 1 (假设目标用户只有1条评论)
        # 实际上我们需要确保即使去掉目标用户，还有至少 min_other_reviews 条
        # DISABLED: min_other_reviews requirement - all products are now qualified
        # other_reviews = total - 1

        # if total >= min_total_reviews and other_reviews >= min_other_reviews:
        #     valid_products[asin] = stats
        
        # 只检查总评论数要求
        if total >= min_total_reviews:
            valid_products[asin] = stats

    log_with_timestamp(f"Total products found: {len(product_stats):,}")
    log_with_timestamp(f"Products meeting review criteria: {len(valid_products):,} (>= {min_total_reviews} total, min_other_reviews disabled)")

    return valid_products

def count_words(text: str) -> int:
    """计算文本中的词数"""
    if not text:
        return 0
    return len(text.split())

def scan_users_and_collect_reviews(review_file: str, valid_asins: Set[str], asin_to_category: Dict[str, str],
                                     min_reviews: int, max_reviews: int,
                                     min_product_reviews: int = 5, min_other_reviews: int = 4,
                                     max_users: int = 10, min_target_words: int = 100) -> tuple:
    """
    扫描评论文件，筛选符合条件的用户，并收集这些用户的评论数据

    返回: (found_users, all_reviews_data)
    - found_users: 符合条件的用户列表
    - all_reviews_data: {user_id: {asin: {'target_reviews': [...], 'other_reviews': [...]}}}

    条件：
    - 用户评论的商品必须在元数据中（valid_asins）
    - 商品至少有一条目标用户评论 >= min_target_words 词（保存所有符合条件的评论）
    - 有效评论数在 [min_reviews, max_reviews] 范围内
    """
    log_with_timestamp(f"Scanning users from {review_file}...")
    log_with_timestamp(f"User criteria: {min_reviews}-{max_reviews} products")
    log_with_timestamp(f"Product criteria: Product must have metadata (no min_product_reviews requirement)")
    log_with_timestamp(f"Target review criteria: >= {min_target_words} words")

    valid_product_asins = valid_asins

    # 第二遍扫描：同时做两件事
    # 1. 统计每个用户的有效商品
    # 2. 收集所有评论数据（用于后续筛选）
    user_products = defaultdict(set)  # {user_id: {asin1, asin2, ...}}
    user_product_titles = defaultdict(dict)  # {user_id: {asin: title}}

    # 存储所有评论数据：{asin: [review1, review2, ...]}
    # 为了内存效率，我们只存储必要的字段
    asin_reviews = defaultdict(list)  # {asin: [review, ...]}

    log_with_timestamp("Processing user-product associations and collecting reviews...")
    with gzip.open(review_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                review = json.loads(line)
                user_id = review.get('reviewerID')
                asin = review.get('asin')
                title = review.get('title', '')

                if asin and asin in valid_asins and user_id:
                    asin_reviews[asin].append(review)
                    user_products[user_id].add(asin)
                    if asin not in user_product_titles[user_id]:
                        user_product_titles[user_id][asin] = title
            except:
                continue

    log_with_timestamp(f"Total unique users found: {len(user_products):,}")

    # 为所有用户整理评论数据，并筛选目标评论长度>=min_target_words的商品
    all_reviews_data = {}
    user_valid_products = {}  # {user_id: {asin1, asin2, ...}} - 只包含符合条件的商品
    user_categories = {}  # {user_id: {category1, category2, ...}} - 用户涉及的类目

    for user_id in user_products.keys():
        user_asins = user_products[user_id]
        all_reviews_data[user_id] = {}
        valid_asins_for_user = set()
        categories_for_user = set()

        for asin in user_asins:
            reviews = asin_reviews.get(asin, [])
            target_reviews = []
            other_reviews = []

            for review in reviews:
                reviewer_id = review.get('reviewerID', '')
                if reviewer_id == user_id:
                    target_reviews.append(review)
                else:
                    other_reviews.append(review)

            # 计算所有目标评论的总词数
            total_target_words = sum(count_words(r.get('reviewText', '')) for r in target_reviews)

            # 如果总词数 >= min_target_words，保留所有目标评论
            if total_target_words >= min_target_words:
                other_reviews = other_reviews[:10]

                all_reviews_data[user_id][asin] = {
                    'target_reviews': target_reviews,
                    'other_reviews': other_reviews
                }
                valid_asins_for_user.add(asin)
                
                # 收集用户涉及的类目
                if asin in asin_to_category:
                    categories_for_user.add(asin_to_category[asin])

        user_valid_products[user_id] = valid_asins_for_user
        user_categories[user_id] = categories_for_user

    # 根据符合条件的商品数量筛选用户
    found_users = []
    for user_id, valid_asins in user_valid_products.items():
        count = len(valid_asins)
        category_count = len(user_categories.get(user_id, set()))
        if min_reviews <= count <= max_reviews:
            found_users.append({
                'user_id': user_id,
                'product_count': count,
                'category_count': category_count,
                'categories': list(user_categories.get(user_id, set()))[:20],  # 最多显示20个类目
                'products': [{'asin': asin, 'title': user_product_titles[user_id][asin]}
                            for asin in valid_asins]
            })

    # 修改排序逻辑：优先选择类目少的用户（类目集中 = 更好的候选用户）
    # 排序优先级：1) category_count升序（类目少优先） 2) product_count降序
    found_users.sort(key=lambda x: (x['category_count'], -x['product_count']))

    # 只保留前 N 个用户的评论数据
    selected_users = found_users[:max_users]
    selected_user_ids = set(u['user_id'] for u in selected_users)

    log_with_timestamp(f"Found {len(found_users)} users matching criteria")
    log_with_timestamp(f"Selected top {len(selected_users)} users for review data collection")

    return found_users, all_reviews_data

def prepare_user_data_from_collected(user_info: Dict, all_reviews_data: Dict, output_dir: str, max_other_reviews: int = 10) -> bool:
    """从已收集的数据为单个用户准备评论数据"""
    user_id = user_info['user_id']
    products = user_info['products']

    results = []
    for i, product in enumerate(products):
        asin = product['asin']

        reviews_data = all_reviews_data[user_id][asin]

        target_reviews_texts = [r.get('reviewText', '') for r in reviews_data['target_reviews']]
        other_reviews_texts = [r.get('reviewText', '') for r in reviews_data['other_reviews']]

        result = {
            'asin': asin,
            'product_title': product['title'],
            'target_user_id': user_id,
            'target_reviews_count': len(target_reviews_texts),
            'target_reviews': target_reviews_texts,
            'other_reviews_count': len(other_reviews_texts),
            'other_reviews': other_reviews_texts
        }
        results.append(result)

    # 保存该用户的数据
    output_data = {
        'user_id': user_id,
        'timestamp': datetime.now().isoformat(),
        'total_products': len(results),
        'results': results
    }

    output_file = os.path.join(output_dir, f'reviews_{user_id}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return True

def main():
    parser = argparse.ArgumentParser(description="Stage 0: Batch Data Preparation with User Selection")
    parser.add_argument("--review-file", required=True, help="Path to review JSON file")
    parser.add_argument("--meta-file", required=True, help="Path to metadata JSON file")
    parser.add_argument("--min-reviews", type=int, default=100, help="Minimum user products (default: 100)")
    parser.add_argument("--max-reviews", type=int, default=300, help="Maximum user products (default: 300)")
    parser.add_argument("--max-users", type=int, default=10, help="Maximum users to select (default: 10)")
    parser.add_argument("--min-product-reviews", type=int, default=5, help="Minimum total reviews per product (default: 5)")
    parser.add_argument("--min-other-reviews", type=int, default=4, help="Minimum other reviews per product (default: 4, DISABLED)")
    parser.add_argument("--min-target-words", type=int, default=50, help="Minimum word count for target user review (default: 50)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 0: Batch Data Preparation with User Selection")
    log_with_timestamp("=" * 80)

    # Step 1: 加载元数据
    valid_asins, asin_to_category = load_metadata_asins(args.meta_file)

    # Step 2: 扫描并筛选用户，同时收集评论数据
    found_users, all_reviews_data = scan_users_and_collect_reviews(
        args.review_file,
        valid_asins,
        asin_to_category,
        args.min_reviews,
        args.max_reviews,
        args.min_product_reviews,
        args.min_other_reviews,
        args.max_users,
        args.min_target_words
    )

    if not found_users:
        log_with_timestamp("No users found matching criteria!")
        return

    # Step 3: 选择前 N 个用户（已在 scan_users_and_collect_reviews 中完成）
    selected_users = found_users[:args.max_users]

    log_with_timestamp(f"\nSelected Top {len(selected_users)} Users (sorted by category count):")
    for i, user in enumerate(selected_users):
        category_info = f", {user['category_count']} categories" if 'category_count' in user else ""
        log_with_timestamp(f"  {i+1}. {user['user_id']}: {user['product_count']} products{category_info}")

    # Step 4: 从已收集的数据为每个用户准备评论数据
    log_with_timestamp(f"\nPreparing review data for {len(selected_users)} users...")
    for i, user_info in enumerate(selected_users):
        user_id = user_info['user_id']
        log_with_timestamp(f"[{i+1}/{len(selected_users)}] Processing user {user_id}...")

        prepare_user_data_from_collected(user_info, all_reviews_data, args.output_dir)

    # Step 5: 保存选中的用户列表
    selected_users_list = [u['user_id'] for u in selected_users]
    summary_file = os.path.join(args.output_dir, 'selected_users.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'selection_criteria': {
                'min_user_products': args.min_reviews,
                'max_user_products': args.max_reviews,
                'max_users': args.max_users,
                'min_product_reviews': args.min_product_reviews,
                'min_other_reviews': args.min_other_reviews,
                'min_target_words': args.min_target_words
            },
            'total_found': len(found_users),
            'total_selected': len(selected_users),
            'users': selected_users_list
        }, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"\n{'=' * 80}")
    log_with_timestamp(f"Stage 0 Complete!")
    log_with_timestamp(f"Selected {len(selected_users)} users")
    log_with_timestamp(f"Output directory: {args.output_dir}")
    log_with_timestamp(f"User list: {summary_file}")
    log_with_timestamp(f"{'=' * 80}")

if __name__ == "__main__":
    main()
