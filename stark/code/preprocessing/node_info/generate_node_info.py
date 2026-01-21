#!/usr/bin/env python3
"""
Generate node info for Amazon SKB knowledge base.

This script processes Amazon review, metadata, and QA data to create
a structured knowledge base using the AmazonSKB class.
"""

import os
import sys
import argparse
from pathlib import Path

# 强制使用本地stark代码，而不是conda环境中的版本
print("=== FORCING LOCAL STARK CODE ===")
local_stark_path = Path("/home/wlia0047/ar57/wenyu/stark")
sys.path.insert(0, str(local_stark_path))
print(f"Added local stark path: {local_stark_path}")

# 验证我们使用的是本地版本
try:
    import stark_qa.skb.amazon
    amazon_file_path = stark_qa.skb.amazon.__file__
    print(f"Using AmazonSKB from: {amazon_file_path}")
    if "site-packages" in amazon_file_path:
        print("⚠️  WARNING: Still using conda installed version!")
    else:
        print("✅ SUCCESS: Using local version!")
except ImportError as e:
    print(f"Import error: {e}")

from stark_qa.skb.amazon import AmazonSKB


def main():
    """
    Main function to generate node info for Amazon SKB.
    """
    parser = argparse.ArgumentParser(description='Generate Amazon SKB knowledge base')
    parser.add_argument(
        '--data_root',
        type=str,
        default='/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018',
        help='Root directory containing the raw data'
    )
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        default=['Arts_Crafts_and_Sewing'],
        help='Product categories to process'
    )
    parser.add_argument(
        '--meta_link_types',
        type=str,
        nargs='+',
        default=['brand', 'category', 'color'],
        help='Meta link types to add (brand, category, color)'
    )
    parser.add_argument(
        '--max_entries',
        type=int,
        default=25,
        help='Maximum number of review & QA entries to show in description'
    )
    parser.add_argument(
        '--download_processed',
        action='store_true',
        default=False,
        help='Whether to download processed data (default: False, process from raw)'
    )
    parser.add_argument(
        '--quick_info',
        action='store_true',
        default=False,
        help='Show only basic information without loading large files (default: False)'
    )
    parser.add_argument(
        '--generate_user_reviews',
        action='store_true',
        default=False,
        help='Generate user-to-product reviews mapping file (default: False)'
    )

    args = parser.parse_args()

    # Set the raw data directory
    raw_data_dir = os.path.join(args.data_root, 'raw')

    # Verify data exists
    print("Checking data availability...")
    for category in args.categories:
        review_file = os.path.join(raw_data_dir, f'{category}.json.gz')
        meta_file = os.path.join(raw_data_dir, f'meta_{category}.json.gz')
        qa_file = os.path.join(raw_data_dir, f'qa_{category}.json.gz')

        if not os.path.exists(review_file):
            print(f"Warning: Review file not found: {review_file}")
        else:
            print(f"Found review file: {review_file}")

        if not os.path.exists(meta_file):
            print(f"Warning: Meta file not found: {meta_file}")
        else:
            print(f"Found meta file: {meta_file}")

        if not os.path.exists(qa_file):
            print(f"Warning: QA file not found: {qa_file}")
        else:
            print(f"Found QA file: {qa_file}")

    # Create output directory
    output_dir = os.path.join(args.data_root, 'processed')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nInitializing AmazonSKB with:")
    print(f"  Data root: {args.data_root}")
    print(f"  Categories: {args.categories}")
    print(f"  Meta link types: {args.meta_link_types}")
    print(f"  Max entries: {args.max_entries}")
    print(f"  Download processed: {args.download_processed}")

    try:
        # Initialize the Amazon SKB
        skb = AmazonSKB(
            root=args.data_root,
            categories=args.categories,
            meta_link_types=args.meta_link_types,
            max_entries=args.max_entries,
            download_processed=args.download_processed
        )

        print("\nSKB initialized successfully!")
        print(f"Number of nodes: {len(skb.node_info)}")
        print(f"Number of edges: {skb.edge_index.shape[1] if hasattr(skb, 'edge_index') else 'N/A'}")
        print(f"Node types: {skb.node_type_dict}")
        print(f"Edge types: {skb.edge_type_dict}")

        # Save the processed data
        print(f"\nProcessed data saved to: {output_dir}")

        # Print detailed information of the first product using SKB interface functions
        print("\n" + "="*60)
        print("DETAILED INFORMATION OF FIRST PRODUCT (NODE 0) USING SKB INTERFACES")
        print("="*60)

        if args.quick_info:
            print("\n使用快速模式，只显示基本统计信息:")
            print(f"  节点总数: {len(skb.node_info)}")
            print(f"  边总数: {skb.edge_index.shape[1] if hasattr(skb, 'edge_index') else 'N/A'}")
            print(f"  节点类型: {skb.node_type_dict}")
            print(f"  边类型: {skb.edge_type_dict}")
            print("\n快速模式完成。如需详细信息，请去掉 --quick_info 参数。")
            return

        try:
            print("\n1. 节点0的基本信息:")
            node_0 = skb[0]
            print(f"  标题: {getattr(node_0, 'title', 'N/A')}")
            print(f"  品牌: {getattr(node_0, 'brand', 'N/A')}")
            print(f"  评论数量: {len(getattr(node_0, 'review', []))}")
            print(f"  QA数量: {len(getattr(node_0, 'qa', []))}")

            print("\n2. 节点0的完整文档信息:")
            doc_info = skb.get_doc_info(0, add_rel=True)
            print(doc_info)

            print("\n3. 节点0的关系信息:")
            rel_info = skb.get_rel_info(0)
            if rel_info:
                print(rel_info)
            else:
                print("  无关系信息")

            print("\n4. 图结构统计:")
            print(f"  节点总数: {len(skb.node_info)}")
            print(f"  边总数: {skb.edge_index.shape[1] if hasattr(skb, 'edge_index') else 'N/A'}")
            print(f"  节点类型分布: {skb.node_type_dict}")
            print(f"  边类型分布: {skb.edge_type_dict}")

            # Show detailed information of nodes connected to node 0
            print("\n5. 节点0连接的其他节点详细信息:")

            # Get different types of neighbors
            also_buy_neighbors = skb.get_neighbor_nodes(0, 'also_buy')
            also_view_neighbors = skb.get_neighbor_nodes(0, 'also_view')
            brand_neighbors = skb.get_neighbor_nodes(0, 'has_brand')
            category_neighbors = skb.get_neighbor_nodes(0, 'has_category')
            color_neighbors = skb.get_neighbor_nodes(0, 'has_color')

            # Display also_buy neighbors
            if also_buy_neighbors:
                print(f"\n  也购买的产品 ({len(also_buy_neighbors)} 个):")
                for i, neighbor_id in enumerate(also_buy_neighbors[:5]):  # Show first 5
                    try:
                        neighbor = skb[neighbor_id]
                        title = getattr(neighbor, 'title', 'N/A')
                        brand = getattr(neighbor, 'brand', 'N/A')
                        print(f"    {i+1}. 节点{neighbor_id}:")
                        print(f"        标题: {title[:60]}{'...' if len(title) > 60 else ''}")
                        print(f"        品牌: {brand}")
                    except Exception as e:
                        print(f"    {i+1}. 节点{neighbor_id}: 获取信息失败 ({e})")

            # Display also_view neighbors
            if also_view_neighbors:
                print(f"\n  也浏览的产品 ({len(also_view_neighbors)} 个):")
                for i, neighbor_id in enumerate(also_view_neighbors[:5]):  # Show first 5
                    try:
                        neighbor = skb[neighbor_id]
                        title = getattr(neighbor, 'title', 'N/A')
                        brand = getattr(neighbor, 'brand', 'N/A')
                        print(f"    {i+1}. 节点{neighbor_id}:")
                        print(f"        标题: {title[:60]}{'...' if len(title) > 60 else ''}")
                        print(f"        品牌: {brand}")
                    except Exception as e:
                        print(f"    {i+1}. 节点{neighbor_id}: 获取信息失败 ({e})")

            # Display brand neighbor
            if brand_neighbors:
                print(f"\n  品牌信息:")
                try:
                    brand_node = skb[brand_neighbors[0]]
                    brand_name = getattr(brand_node, 'brand_name', 'N/A')
                    print(f"    品牌名称: {brand_name}")
                    print(f"    品牌节点ID: {brand_neighbors[0]}")
                except Exception as e:
                    print(f"    获取品牌信息失败: {e}")

            # Display category neighbors
            if category_neighbors:
                print(f"\n  类别信息 ({len(category_neighbors)} 个):")
                for i, neighbor_id in enumerate(category_neighbors[:3]):  # Show first 3
                    try:
                        category_node = skb[neighbor_id]
                        category_name = getattr(category_node, 'category_name', 'N/A')
                        print(f"    {i+1}. 类别节点{neighbor_id}: {category_name}")
                    except Exception as e:
                        print(f"    {i+1}. 类别节点{neighbor_id}: 获取信息失败 ({e})")

            # Display color neighbors
            if color_neighbors:
                print(f"\n  颜色信息 ({len(color_neighbors)} 个):")
                for i, neighbor_id in enumerate(color_neighbors[:3]):  # Show first 3
                    try:
                        color_node = skb[neighbor_id]
                        color_name = getattr(color_node, 'color_name', 'N/A')
                        print(f"    {i+1}. 颜色节点{neighbor_id}: {color_name}")
                    except Exception as e:
                        print(f"    {i+1}. 颜色节点{neighbor_id}: 获取信息失败 ({e})")

            # Summary
            total_connections = len(also_buy_neighbors) + len(also_view_neighbors) + len(brand_neighbors) + len(category_neighbors) + len(color_neighbors)
            print(f"\n  总连接数: {total_connections}")
            print(f"    - also_buy: {len(also_buy_neighbors)}")
            print(f"    - also_view: {len(also_view_neighbors)}")
            print(f"    - has_brand: {len(brand_neighbors)}")
            print(f"    - has_category: {len(category_neighbors)}")
            print(f"    - has_color: {len(color_neighbors)}")

        except Exception as e:
            print(f"使用SKB接口获取信息时出错: {e}")
            import traceback
            traceback.print_exc()

        # Generate ID to ASIN mapping file
        print("\n正在生成ID-ASIN映射文件...")
        try:
            import json

            # Create mappings directory if it doesn't exist
            mappings_dir = os.path.join(output_dir, 'mappings')
            os.makedirs(mappings_dir, exist_ok=True)

            # Generate ID to ASIN mapping
            id_to_asin = {}
            asin_to_id = {}

            # Use asin2id and id2asin if available, otherwise extract from nodes
            if hasattr(skb, 'asin2id') and hasattr(skb, 'id2asin'):
                print("  使用SKB内置映射...")
                id_to_asin = dict(skb.id2asin.items())  # 已经是 {ID: ASIN} 格式
                asin_to_id = {v: k for k, v in skb.asin2id.items()}  # ASIN是字符串，ID是整数
            else:
                print("  从节点信息中提取映射...")
                for node_id in range(len(skb.node_info)):
                    try:
                        node = skb[node_id]
                        asin = getattr(node, 'asin', None)
                        if asin:
                            id_to_asin[node_id] = asin
                            asin_to_id[asin] = node_id
                    except Exception as e:
                        print(f"  警告: 处理节点{node_id}时出错: {e}")
                        continue

            # Save mappings to files
            id_to_asin_file = os.path.join(mappings_dir, 'id_to_asin.json')
            asin_to_id_file = os.path.join(mappings_dir, 'asin_to_id.json')

            with open(id_to_asin_file, 'w', encoding='utf-8') as f:
                json.dump(id_to_asin, f, indent=2, ensure_ascii=False)

            with open(asin_to_id_file, 'w', encoding='utf-8') as f:
                json.dump(asin_to_id, f, indent=2, ensure_ascii=False)

            print(f"  映射文件已生成:")
            print(f"    ID->ASIN: {id_to_asin_file} ({len(id_to_asin)} 条映射)")
            print(f"    ASIN->ID: {asin_to_id_file} ({len(asin_to_id)} 条映射)")

            # Show sample mappings
            print(f"  示例映射 (前5个):")
            for i, (node_id, asin) in enumerate(list(id_to_asin.items())[:5]):
                print(f"    节点{node_id} -> ASIN: {asin}")

        except Exception as e:
            print(f"生成映射文件时出错: {e}")
            import traceback
            traceback.print_exc()

        # Generate user-product review mapping file
        if args.generate_user_reviews:
            print("\n正在生成用户-商品评论映射文件...")
            try:
                import gzip
                from collections import defaultdict

                # Create user reviews directory if it doesn't exist
                user_reviews_dir = os.path.join(output_dir, 'user_reviews')
                os.makedirs(user_reviews_dir, exist_ok=True)

                # Dictionary to store user -> list of (product_asin, review_info) mappings
                # Use a nested structure to handle duplicates: user -> asin -> list of reviews
                user_reviews_temp = defaultdict(lambda: defaultdict(list))

                # Process each category
                for category in args.categories:
                    review_file = os.path.join(raw_data_dir, f'{category}.json.gz')
                    print(f"  处理类别: {category}")

                    if not os.path.exists(review_file):
                        print(f"  警告: 评论文件不存在: {review_file}")
                        continue

                    try:
                        # Read and process review file
                        with gzip.open(review_file, 'rt', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                try:
                                    review_data = json.loads(line.strip())

                                    # Extract reviewer information
                                    reviewer_id = review_data.get('reviewerID', '')
                                    asin = review_data.get('asin', '')
                                    review_text = review_data.get('reviewText', '')
                                    rating = review_data.get('overall', 0)
                                    review_time = review_data.get('reviewTime', '')
                                    summary = review_data.get('summary', '')

                                    if reviewer_id and asin:
                                        # Create a unique key for deduplication based on content
                                        content_key = (summary, review_text)

                                        # Store review information with deduplication
                                        review_info = {
                                            'asin': asin,
                                            'rating': rating,
                                            'review_time': review_time,
                                            'summary': summary,
                                            'review_text': review_text  # Keep full review text without truncation
                                        }
                                        user_reviews_temp[reviewer_id][content_key].append(review_info)

                                    # Progress indicator every 10000 lines
                                    if line_num % 10000 == 0:
                                        print(f"    已处理 {line_num} 条评论记录...")

                                except json.JSONDecodeError as e:
                                    print(f"    警告: 第{line_num}行JSON解析错误: {e}")
                                    continue
                                except Exception as e:
                                    print(f"    警告: 处理第{line_num}行时出错: {e}")
                                    continue

                    except Exception as e:
                        print(f"  处理评论文件{review_file}时出错: {e}")
                        continue

                # Convert to final format with deduplication
                # For each user, flatten the nested structure and keep only one review per unique content
                user_reviews = defaultdict(list)
                for user_id, content_groups in user_reviews_temp.items():
                    for content_key, reviews in content_groups.items():
                        # Keep only the first review for each unique content (summary + review_text)
                        user_reviews[user_id].append(reviews[0])

                # Save user reviews mapping to file
                user_reviews_file = os.path.join(user_reviews_dir, 'user_product_reviews.json')

                # Convert defaultdict to regular dict and sort users and their reviews by time
                user_reviews_dict = {}
                for user_id, reviews in user_reviews.items():
                    # Sort reviews by time (assuming format like "MM DD, YYYY")
                    try:
                        reviews_sorted = sorted(reviews, key=lambda x: x['review_time'], reverse=True)
                    except:
                        # If sorting fails, keep original order
                        reviews_sorted = reviews

                    # Calculate unique products count
                    unique_products = set(review['asin'] for review in reviews_sorted)

                    user_reviews_dict[user_id] = {
                        'review_count': len(reviews_sorted),
                        'unique_products_count': len(unique_products),
                        'reviews': reviews_sorted
                    }

                # Sort users by number of reviews (descending)
                user_reviews_sorted = dict(sorted(user_reviews_dict.items(),
                                                key=lambda x: x[1]['review_count'], reverse=True))

                with open(user_reviews_file, 'w', encoding='utf-8') as f:
                    json.dump(user_reviews_sorted, f, indent=2, ensure_ascii=False)

                print(f"  用户-商品评论映射文件已生成:")
                print(f"    文件路径: {user_reviews_file}")
                print(f"    用户数量: {len(user_reviews_sorted)}")
                total_reviews = sum(user_info['review_count'] for user_info in user_reviews_sorted.values())
                print(f"    评论总数: {total_reviews}")

                # Show statistics
                print(f"  评论统计:")
                review_counts = [user_info['review_count'] for user_info in user_reviews_sorted.values()]
                if review_counts:
                    print(f"    平均每用户评论数: {sum(review_counts)/len(review_counts):.2f}")
                    print(f"    最多评论用户: {max(review_counts)} 条评论")
                    print(f"    最少评论用户: {min(review_counts)} 条评论")

                # Show sample user reviews
                print(f"  示例用户评论 (前3个多评论用户):")
                sample_users = list(user_reviews_sorted.items())[:3]
                for i, (user_id, user_info) in enumerate(sample_users, 1):
                    print(f"    用户{i}: {user_id} ({user_info['review_count']} 条评论)")
                    for j, review in enumerate(user_info['reviews'][:2]):  # Show first 2 reviews
                        print(f"      评论{j+1}: 商品{review['asin']}, 评分{review['rating']}, 时间{review['review_time']}")
                        if review['summary']:
                            print(f"        摘要: {review['summary'][:50]}{'...' if len(review['summary']) > 50 else ''}")

            except Exception as e:
                print(f"生成用户评论映射文件时出错: {e}")
                import traceback
                traceback.print_exc()

        print("\nSKB generation completed successfully!")

    except Exception as e:
        print(f"Error during SKB generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
