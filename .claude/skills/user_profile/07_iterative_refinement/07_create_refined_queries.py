#!/usr/bin/env python3
"""
Create Refined Dual Query Files from Stage 7 Iterative Results

This script reads the Stage 7 iterative refinement results and creates
new dual query files with the refined personalized queries.

Input:
  - Stage 6 dual query files (original personalized + public queries)
  - Stage 7 iterative refinement results (refined personalized queries)

Output:
  - New dual query files with refined personalized queries

Usage:
    python create_refined_queries.py \
        --dual-queries-dir 06_query_final \
        --iterative-results-dir 07_neural_proxy/iterative_refinement_v2 \
        --output-dir 06_query_refined
"""

import json
import os
import sys
from typing import Dict, List
from collections import defaultdict


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def load_iterative_results(iterative_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Load Stage 7 iterative refinement results.

    Returns:
        Dict mapping (user_id, asin) -> refined_query
    """
    results_file = os.path.join(iterative_dir, "iterative_results.json")
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        return {}

    with open(results_file, 'r', encoding='utf-8') as f:
        iterative_data = json.load(f)

    # Map (user_id, asin) -> final_query
    refined_queries = {}
    for item in iterative_data:
        key = (item['user_id'], item['asin'])
        refined_queries[key] = item['final_query']

    print(f"Loaded {len(refined_queries)} refined queries from {results_file}")
    return refined_queries


def create_refined_dual_queries(
    dual_queries_dir: str,
    refined_queries: Dict[str, Dict[str, str]],
    output_dir: str
):
    """
    Create refined dual query files.

    For each user:
    - Load original dual query file
    - Replace personalized_query with refined version
    - Keep public_query unchanged
    - Save to new directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all dual query files
    dual_files = [f for f in os.listdir(dual_queries_dir) if f.startswith("dual_queries_") and f.endswith(".json")]

    stats = {
        "total_users": len(dual_files),
        "total_queries": 0,
        "refined_queries": 0,
        "unchanged_queries": 0,
        "missing_refinements": 0
    }

    for filename in dual_files:
        user_id = filename.replace("dual_queries_", "").replace(".json", "")
        input_path = os.path.join(dual_queries_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Load original dual queries
        dual_data = load_json(input_path)
        if not dual_data:
            continue

        # Handle both dict and list formats
        if isinstance(dual_data, dict):
            queries = dual_data.get("queries", [])
        else:
            queries = dual_data

        # Process each query
        refined_count = 0
        unchanged_count = 0
        missing_count = 0

        for query_item in queries:
            stats["total_queries"] += 1
            asin = query_item.get("asin")
            key = (user_id, asin)

            if key in refined_queries:
                refined_query = refined_queries[key]
                original_query = query_item.get("personalized_query", "")

                # Update with refined query
                query_item["original_personalized_query"] = original_query  # Keep backup
                query_item["personalized_query"] = refined_query
                query_item["query_refined"] = True

                refined_count += 1
                stats["refined_queries"] += 1
            else:
                # No refinement found for this query
                query_item["query_refined"] = False
                missing_count += 1
                stats["missing_refinements"] += 1

        # Save refined dual queries
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dual_data, f, indent=2, ensure_ascii=False)

        print(f"  {user_id}: {refined_count} refined, {missing_count} missing")

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create refined dual query files")
    parser.add_argument("--dual-queries-dir", required=True,
                       help="Directory containing original dual query files")
    parser.add_argument("--iterative-results-dir", required=True,
                       help="Directory containing Stage 7 iterative refinement results")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for refined dual query files")

    args = parser.parse_args()

    print("=" * 60)
    print("Creating Refined Dual Query Files")
    print("=" * 60)
    print(f"Dual queries dir: {args.dual_queries_dir}")
    print(f"Iterative results dir: {args.iterative_results_dir}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # Load refined queries from Stage 7
    print("\nLoading Stage 7 iterative refinement results...")
    refined_queries = load_iterative_results(args.iterative_results_dir)

    if not refined_queries:
        print("Error: No refined queries found. Exiting.")
        sys.exit(1)

    # Create refined dual query files
    print("\nCreating refined dual query files...")
    stats = create_refined_dual_queries(
        args.dual_queries_dir,
        refined_queries,
        args.output_dir
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total users: {stats['total_users']}")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Refined queries: {stats['refined_queries']} ({stats['refined_queries']/stats['total_queries']*100:.1f}%)")
    print(f"Unchanged queries: {stats['unchanged_queries']}")
    print(f"Missing refinements: {stats['missing_refinements']}")
    print("=" * 60)

    # Save stats
    stats_file = os.path.join(args.output_dir, "refinement_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {stats_file}")


if __name__ == "__main__":
    main()
