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

def split_user_data(results, target_query_count=10, min_attrs=3, min_cat_size=4):
    # 按类目分组商品
    category_to_items = defaultdict(list)
    for item in results:
        cat = item.get('category', 'Unknown')
        category_to_items[cat].append(item)

    persona_items = []
    query_items = []

    # 1. 识别候选 Query 商品
    # 条件：1. 属性数 >= min_attrs; 2. 该类目下的总商品数 >= min_cat_size
    candidates_by_cat = defaultdict(list)
    non_candidates_by_cat = defaultdict(list)
    
    for cat, items in category_to_items.items():
        cat_total_count = len(items)
        for item in items:
            # 检查属性数量
            selected_attrs = item.get('selected_attributes', [])
            if not selected_attrs:
                final_match = item.get('final_match')
                if final_match:
                    selected_attrs = final_match.get('selected_attributes', [])
            
            # 同时满足：属性数达到门槛 且 类目规模达到门槛
            if len(selected_attrs) >= min_attrs and cat_total_count >= min_cat_size:
                candidates_by_cat[cat].append(item)
            else:
                non_candidates_by_cat[cat].append(item)

    # 2. 从候选商品中挑选 Query 集，确保类别覆盖
    # 策略：保证每个有候选商品的类目至少有一个进入 query，直到达到 target_query_count
    sorted_cats = sorted(candidates_by_cat.keys(), key=lambda x: len(candidates_by_cat[x]), reverse=True)
    
    # 第一轮：每类拿一个
    for cat in sorted_cats:
        if len(query_items) < target_query_count:
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
    parser.add_argument("--output-dir", required=True, help="Output directory for split data")
    parser.add_argument("--user-id", help="Single user ID to process")
    parser.add_argument("--target-query", type=int, default=10, help="Target number of queries")
    parser.add_argument("--min-attrs", type=int, default=3, help="Minimum attributes for query items")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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

        persona_items, query_items = split_user_data(results, args.target_query, args.min_attrs)

        # 保存结果
        output_file = os.path.join(args.output_dir, f"query_{user_id}.json")
        data_to_save = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "split_strategy": "attribute_aware_category_split",
            "min_attrs": args.min_attrs,
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
