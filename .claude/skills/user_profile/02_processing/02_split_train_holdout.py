import json
import os
import argparse
import random
from datetime import datetime
from collections import defaultdict

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_match_results(match_file):
    if not os.path.exists(match_file):
        return None
    with open(match_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('results', [])

def load_all_users_preferences(prefs_dir):
    """
    加载所有用户的偏好数据，统计每个商品有多少个用户评论

    注意：此函数已弃用，仅用于向后兼容
    推荐使用 load_reviews_from_file() 从原始评论文件统计全网数据
    """
    asin_to_users = defaultdict(list)

    for filename in os.listdir(prefs_dir):
        if not filename.startswith('preferences_') or not filename.endswith('.json'):
            continue

        user_id = filename.replace('preferences_', '').replace('.json', '')
        pref_file = os.path.join(prefs_dir, filename)

        try:
            with open(pref_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data.get('results', []):
                asin = item.get('asin')
                if asin:
                    asin_to_users[asin].append(user_id)
        except Exception as e:
            log_with_timestamp(f"  Warning: Failed to load {filename}: {e}")
            continue

    return asin_to_users


def load_reviews_from_file(reviews_file):
    """
    从原始评论文件加载所有评论，统计每个商品的全网用户数

    参数:
        reviews_file: 原始评论 JSON 文件路径（每行一个 JSON 对象）

    返回:
        asin_to_users: 字典 {asin: [user_id1, user_id2, ...]}
    """
    asin_to_users = defaultdict(set)
    total_reviews = 0

    log_with_timestamp(f"Loading reviews from {reviews_file}...")

    with open(reviews_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 500000 == 0:
                log_with_timestamp(f"  Processed {i:,} reviews...")

            try:
                review = json.loads(line.strip())
                asin = review.get('asin')
                user_id = review.get('reviewerID')

                if asin and user_id:
                    asin_to_users[asin].add(user_id)
                    total_reviews += 1
            except:
                continue

    # 转换 set 为 list 以便后续使用
    asin_to_users_list = {asin: list(users) for asin, users in asin_to_users.items()}

    log_with_timestamp(f"  Total reviews processed: {total_reviews:,}")
    log_with_timestamp(f"  Total unique products: {len(asin_to_users_list):,}")

    # 统计用户数分布
    user_counts = [len(users) for users in asin_to_users_list.values()]
    if user_counts:
        avg_users = sum(user_counts) / len(user_counts)
        min_users = min(user_counts)
        max_users = max(user_counts)
        log_with_timestamp(f"  Users per product: avg={avg_users:.1f}, min={min_users}, max={max_users}")

    return asin_to_users_list

def split_user_data(results, user_id, asin_to_users, target_query_count=10, min_attrs=3, min_cat_size=4, min_other_users=3):
    """
    划分训练集和测试集

    参数:
        results: 匹配结果列表
        user_id: 目标用户 ID
        asin_to_users: 商品到用户列表的映射
        target_query_count: 目标查询数量
        min_attrs: 最小属性数
        min_cat_size: 最小类目规模
        min_other_users: 最少其他用户数（用于大众属性）
    """
    # 按类目分组商品
    category_to_items = defaultdict(list)
    for item in results:
        cat = item.get('category', 'Unknown')
        category_to_items[cat].append(item)

    persona_items = []
    query_items = []

    # 1. 识别候选 Query 商品
    # 条件：
    #   1. 属性数 >= min_attrs
    #   2. 该类目下的总商品数 >= min_cat_size
    #   3. 其他用户评论数 >= min_other_users (NEW!)
    candidates_by_cat = defaultdict(list)
    non_candidates_by_cat = defaultdict(list)

    for cat, items in category_to_items.items():
        cat_total_count = len(items)
        for item in items:
            asin = item.get('asin')

            # 检查属性数量
            selected_attrs = item.get('selected_attributes', [])
            if not selected_attrs:
                final_match = item.get('final_match')
                if final_match:
                    selected_attrs = final_match.get('selected_attributes', [])

            # 检查其他用户数
            all_users = asin_to_users.get(asin, [])
            other_users_count = len([u for u in all_users if u != user_id])

            # 同时满足三个条件
            meets_attr_req = len(selected_attrs) >= min_attrs
            meets_cat_req = cat_total_count >= min_cat_size
            meets_users_req = other_users_count >= min_other_users

            if meets_attr_req and meets_cat_req and meets_users_req:
                candidates_by_cat[cat].append(item)
            else:
                non_candidates_by_cat[cat].append(item)
                # 记录被过滤的原因
                if not meets_users_req:
                    item['_filter_reason'] = f"其他用户数不足 ({other_users_count} < {min_other_users})"

    # 2. 从候选商品中挑选 Query 集，确保类别覆盖
    # 策略：如果 target_query_count 为 None，则包含所有候选商品；否则保证每个有候选商品的类目至少有一个进入 query，直到达到 target_query_count
    sorted_cats = sorted(candidates_by_cat.keys(), key=lambda x: len(candidates_by_cat[x]), reverse=True)

    # 如果没有设置限制（target_query_count=None），则包含所有候选商品
    if target_query_count is None:
        # 将所有候选商品加入 query_items
        for cat in sorted_cats:
            query_items.extend(candidates_by_cat[cat])
    else:
        # 第一轮：每类拿一个
        for cat in sorted_cats:
            if len(query_items) < target_query_count and candidates_by_cat[cat]:
                item = candidates_by_cat[cat].pop(random.randrange(len(candidates_by_cat[cat])))
                query_items.append(item)

        # 第二轮：如果还没够，从候选商品最多的类目再拿
        while len(query_items) < target_query_count:
            # Check if we have any candidates left
            valid_cats = [c for c in candidates_by_cat if candidates_by_cat[c]]
            if not valid_cats:
                break

            # 寻找目前候选商品最多的类目
            next_cat = max(valid_cats, key=lambda x: len(candidates_by_cat[x]))
            item = candidates_by_cat[next_cat].pop(random.randrange(len(candidates_by_cat[next_cat])))
            query_items.append(item)

    # 3. 剩下的所有商品（候选商品中没被选中的 + 非候选商品）全部进入 Persona 集
    for cat, items in candidates_by_cat.items():
        persona_items.extend(items)
    for cat, items in non_candidates_by_cat.items():
        persona_items.extend(items)

    return persona_items, query_items

def main():
    parser = argparse.ArgumentParser(description="Category-aware data split with attribute count constraint.")
    parser.add_argument("--match-dir", required=True, help="Directory containing match_USERID.json files")
    parser.add_argument("--preferences-dir", required=True, help="Directory containing preferences_USERID.json files")
    parser.add_argument("--output-dir", required=True, help="Output directory for split data")
    parser.add_argument("--reviews-file", help="Original reviews JSON file for counting global user statistics (recommended)")
    parser.add_argument("--user-id", help="Single user ID to process")
    parser.add_argument("--target-query", type=int, default=None, help="Target number of queries (default: no limit, generate all qualifying products)")
    parser.add_argument("--min-attrs", type=int, default=3, help="Minimum attributes for query items")
    parser.add_argument("--min-other-users", type=int, default=3, help="Minimum other users for public attributes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载商品的用户统计数据
    if args.reviews_file:
        # 优先使用原始评论文件统计全网数据（推荐）
        log_with_timestamp("=" * 60)
        log_with_timestamp("Loading GLOBAL review statistics from original reviews file...")
        log_with_timestamp("This ensures min-other-users reflects the true global user count")
        log_with_timestamp("=" * 60)
        asin_to_users = load_reviews_from_file(args.reviews_file)
    else:
        # 向后兼容：从10个用户的偏好数据统计
        log_with_timestamp("=" * 60)
        log_with_timestamp("WARNING: Using only 10 selected users for user count statistics")
        log_with_timestamp("This may underestimate global user counts!")
        log_with_timestamp("Recommendation: Use --reviews-file for accurate global statistics")
        log_with_timestamp("=" * 60)
        asin_to_users = load_all_users_preferences(args.preferences_dir)

    # 找出待处理的用户
    if args.user_id:
        user_files = [f"match_{args.user_id}.json"]
    else:
        user_files = [f for f in os.listdir(args.match_dir) if f.startswith("match_") and f.endswith(".json")]

    summary = []

    for f_name in user_files:
        user_id = f_name.replace("match_", "").replace(".json", "")
        match_file = os.path.join(args.match_dir, f_name)

        log_with_timestamp(f"Processing user {user_id}...")
        results = load_match_results(match_file)
        if not results:
            log_with_timestamp(f"  Warning: No results found for user {user_id}")
            continue

        persona_items, query_items = split_user_data(
            results, user_id, asin_to_users,
            args.target_query, args.min_attrs, min_cat_size=4, min_other_users=args.min_other_users
        )

        log_with_timestamp(f"  Query items: {len(query_items)}, Persona items: {len(persona_items)}")

        # 保存结果
        output_file = os.path.join(args.output_dir, f"query_{user_id}.json")
        data_to_save = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "split_strategy": "attribute_aware_category_split_with_public_attrs",
            "min_attrs": args.min_attrs,
            "min_other_users": args.min_other_users,
            "target_query_count": args.target_query,
            "total_products": len(results),
            "persona_count": len(persona_items),
            "query_count": len(query_items),
            "query_results": query_items
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        log_with_timestamp(f"  Split: Persona={len(persona_items)}, Query={len(query_items)}")
        summary.append({
            "user_id": user_id,
            "persona_count": len(persona_items),
            "query_count": len(query_items)
        })

    # 保存汇总信息
    with open(os.path.join(args.output_dir, "split_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
