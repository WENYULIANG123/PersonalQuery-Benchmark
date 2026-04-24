"""
Compare P@10 values across 3 domains (categories) and all retrievers.

Reads retrieval_all_summary.json from each domain and prints a comparison table.
"""

import json
import os
from pathlib import Path

BASE_DIR = Path("/home/wlia0047/ar57/wenyu/result/personal_query/08_retrieval")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]


def load_p10_data(category: str) -> dict:
    """Load P@10 data for all retrievers from a category."""
    summary_path = BASE_DIR / category / "retrieval_all_summary.json"
    with open(summary_path, "r") as f:
        data = json.load(f)

    results = {}
    rbct = data["results_by_category_and_type"]

    # Get 'acl' or first available query category with 'correct' query type
    key = None
    for k in rbct.keys():
        if "correct" in k:
            key = k
            break

    if key is None:
        return results

    for item in rbct[key]:
        retriever = item["retriever"]
        group_metrics = item["group_metrics"]

        results[retriever] = {}
        for level, metrics in group_metrics.items():
            results[retriever][level] = metrics.get("P@10", 0.0)

    return results


def main():
    # Load data from all categories
    all_data = {}
    for cat in CATEGORIES:
        try:
            all_data[cat] = load_p10_data(cat)
            print(f"Loaded {cat}: {len(all_data[cat])} retrievers")
        except Exception as e:
            print(f"Error loading {cat}: {e}")

    # Collect all retrievers across categories
    all_retrievers = set()
    for cat_data in all_data.values():
        all_retrievers.update(cat_data.keys())
    all_retrievers = sorted(all_retrievers)

    # Collect all levels
    all_levels = set()
    for cat_data in all_data.values():
        for retriever_data in cat_data.values():
            all_levels.update(retriever_data.keys())
    all_levels = sorted(all_levels, key=lambda x: int(x))

    # Print table
    print("\n" + "=" * 120)
    print("P@10 Comparison Across Domains")
    print("=" * 120)

    # Header
    header = f"{'Retriever':<12}"
    for cat in CATEGORIES:
        short_name = cat.replace("_", "\n")
        header += f" {cat:^30}"
    header += f" {'Average':>12}"
    print(header)
    print("-" * 120)

    # Column sub-header
    sub_header = f"{'':12}"
    for _ in CATEGORIES:
        for lvl in all_levels:
            sub_header += f" L{lvl:>5}"
        sub_header += f" {'Avg':>5}"
    print(sub_header)
    print("-" * 120)

    # Data rows
    for retriever in all_retrievers:
        row = f"{retriever:<12}"

        cat_avgs = []
        for cat in CATEGORIES:
            cat_data = all_data.get(cat, {})
            retriever_data = cat_data.get(retriever, {})

            vals = []
            for lvl in all_levels:
                val = retriever_data.get(lvl, None)
                if val is not None:
                    vals.append(val)
                else:
                    vals.append(None)

            # Print per-level values
            for v in vals:
                if v is not None:
                    row += f" {v:>5.4f}"
                else:
                    row += f" {'N/A':>5}"

            # Calculate category average (excluding None)
            valid_vals = [v for v in vals if v is not None]
            cat_avg = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0
            cat_avgs.append(cat_avg)
            row += f" {cat_avg:>5.4f}"

        # Overall average across categories
        overall_avg = sum(cat_avgs) / len(cat_avgs) if cat_avgs else 0.0
        row += f" {overall_avg:>12.4f}"

        print(row)

    print("-" * 120)

    # Summary: best retriever per category
    print("\nBest Retriever by P@10 (Highest Average Across Levels):")
    for cat in CATEGORIES:
        cat_data = all_data.get(cat, {})
        if not cat_data:
            continue
        best_retriever = None
        best_avg = -1
        for retriever, levels in cat_data.items():
            vals = [v for v in levels.values() if v is not None]
            avg = sum(vals) / len(vals) if vals else 0.0
            if avg > best_avg:
                best_avg = avg
                best_retriever = retriever
        print(f"  {cat}: {best_retriever} (P@10 = {best_avg:.4f})")


if __name__ == "__main__":
    main()
