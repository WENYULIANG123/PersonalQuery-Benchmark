"""
Compare P@10 values across 3 domains (categories) and all retrievers.
Includes both:
1. Clean query results (from 08_retrieval/retrieval_all_summary.json)
2. Correct vs Noisy query results (from 09_noisy_retrieval/correct_vs_noisy_results.json)

Reads retrieval_all_summary.json from each domain and prints a comparison table.
"""

import json
import os
from pathlib import Path

BASE_DIR_08 = Path("/home/wlia0047/ar57/wenyu/result/personal_query/08_retrieval")
BASE_DIR_09 = Path("/home/wlia0047/ar57/wenyu/result/personal_query/09_noisy_retrieval")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
CATEGORY_SHORT_NAMES = {
    "Baby_Products": "Baby",
    "Grocery_and_Gourmet_Food": "Grocery",
    "Pet_Supplies": "Pet",
}


def load_clean_data(category: str) -> dict:
    """Load P@10 data for all retrievers from a category (08_retrieval clean results)."""
    summary_path = BASE_DIR_08 / category / "retrieval_all_summary.json"
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


def load_correct_vs_noisy_data(category: str) -> dict:
    """Load correct and noisy P@10 data from 09_noisy_retrieval."""
    results_path = BASE_DIR_09 / category / "correct_vs_noisy_results.json"
    with open(results_path, "r") as f:
        data = json.load(f)

    results = {"correct": {}, "noisy": {}}

    # Parse correct results
    for item in data.get("correct_results", []):
        retriever = item["retriever"]
        p10 = item["metrics"].get("P@10", 0.0)
        results["correct"][retriever] = p10

    # Parse noisy results
    for item in data.get("noisy_results", []):
        retriever = item["retriever"]
        p10 = item["metrics"].get("P@10", 0.0)
        results["noisy"][retriever] = p10

    return results


def print_clean_comparison(all_data: dict):
    """Print clean query P@10 comparison table."""
    all_retrievers = set()
    for cat_data in all_data.values():
        all_retrievers.update(cat_data.keys())
    all_retrievers = sorted(all_retrievers)

    print("\n" + "=" * 100)
    print("Clean Query P@10 Comparison Across Domains (from 08_retrieval)")
    print("=" * 100)

    header = f"{'Retriever':<12}"
    for cat in CATEGORIES:
        short = CATEGORY_SHORT_NAMES.get(cat, cat[:15])
        header += f" {short:^20}"
    header += f" {'Average':>12}"
    print(header)
    print("-" * 100)

    retriever_avgs = []
    for retriever in all_retrievers:
        row = f"{retriever:<12}"
        cat_vals = []
        for cat in CATEGORIES:
            cat_data = all_data.get(cat, {})
            val = cat_data.get(retriever, {})
            if isinstance(val, dict):
                # Average across levels
                vals = [v for v in val.values() if v is not None]
                avg = sum(vals) / len(vals) if vals else 0.0
            else:
                avg = val if val else 0.0
            cat_vals.append(avg)
            row += f" {avg:>20.4f}"

        overall_avg = sum(cat_vals) / len(cat_vals) if cat_vals else 0.0
        retriever_avgs.append((retriever, overall_avg))
        row += f" {overall_avg:>12.4f}"
        print(row)

    print("-" * 100)
    # Best per category
    print("\nBest Retriever by Clean P@10:")
    for cat in CATEGORIES:
        cat_data = all_data.get(cat, {})
        if not cat_data:
            continue
        best_retriever = None
        best_avg = -1
        for retriever, val in cat_data.items():
            if isinstance(val, dict):
                vals = [v for v in val.values() if v is not None]
                avg = sum(vals) / len(vals) if vals else 0.0
            else:
                avg = val if val else 0.0
            if avg > best_avg:
                best_avg = avg
                best_retriever = retriever
        print(f"  {cat}: {best_retriever} (P@10 = {best_avg:.4f})")


def print_correct_vs_noisy_by_domain(all_data: dict):
    """Print correct vs noisy P@10 comparison table, one table per domain."""
    # Collect all retrievers
    all_retrievers = set()
    for cat_data in all_data.values():
        if "correct" in cat_data:
            all_retrievers.update(cat_data["correct"].keys())
        if "noisy" in cat_data:
            all_retrievers.update(cat_data["noisy"].keys())
    all_retrievers = sorted(all_retrievers)

    for cat in CATEGORIES:
        short = CATEGORY_SHORT_NAMES.get(cat, cat[:10])
        print("\n" + "=" * 80)
        print(f"Correct vs Noisy P@10 - {cat} (from 09_noisy_retrieval)")
        print("=" * 80)

        header = f"{'Retriever':<12} {'CORR':>10} {'NOISY':>10} {'DIFF':>10}"
        print(header)
        print("-" * 80)

        cat_data = all_data.get(cat, {})
        corr_data = cat_data.get("correct", {})
        noisy_data = cat_data.get("noisy", {})

        for retriever in all_retrievers:
            corr = corr_data.get(retriever, None)
            noi = noisy_data.get(retriever, None)

            if corr is not None and noi is not None:
                diff = noi - corr
                corr_str = f"{corr:.4f}"
                noi_str = f"{noi:.4f}"
                diff_str = f"{diff:+.4f}"
            else:
                corr_str = "N/A"
                noi_str = "N/A"
                diff_str = "N/A"

            print(f"{retriever:<12} {corr_str:>10} {noi_str:>10} {diff_str:>10}")

        print("-" * 80)


def print_correct_vs_noisy_summary(all_data: dict):
    """Print summary across all domains."""
    # Collect all retrievers
    all_retrievers = set()
    for cat_data in all_data.values():
        if "correct" in cat_data:
            all_retrievers.update(cat_data["correct"].keys())
        if "noisy" in cat_data:
            all_retrievers.update(cat_data["noisy"].keys())
    all_retrievers = sorted(all_retrievers)

    print("\n" + "=" * 100)
    print("Correct vs Noisy P@10 Summary (All Domains)")
    print("=" * 100)

    header = f"{'Retriever':<12}"
    for cat in CATEGORIES:
        short = CATEGORY_SHORT_NAMES.get(cat, cat[:6])
        header += f" {'CORR':^8} {'NOISY':^8} {'DIFF':^8}"
    header += f" {'Avg Corr':>10} {'Avg Noi':>10} {'Avg Diff':>10}"
    print(header)
    print("-" * 100)

    retriever_corr_avgs = []
    retriever_noi_avgs = []
    retriever_diff_avgs = []

    for retriever in all_retrievers:
        row = f"{retriever:<12}"
        corr_vals = []
        noi_vals = []
        diff_vals = []

        for cat in CATEGORIES:
            cat_data = all_data.get(cat, {})

            corr = cat_data.get("correct", {}).get(retriever, None)
            noi = cat_data.get("noisy", {}).get(retriever, None)

            if corr is not None and noi is not None:
                diff = noi - corr
            else:
                diff = None

            corr_vals.append(corr)
            noi_vals.append(noi)
            diff_vals.append(diff)

            corr_str = f"{corr:.4f}" if corr is not None else "N/A"
            noi_str = f"{noi:.4f}" if noi is not None else "N/A"
            diff_str = f"{diff:+.4f}" if diff is not None else "N/A"

            row += f" {corr_str:^8} {noi_str:^8} {diff_str:^8}"

        # Calculate averages (only for retrievers with data in at least one category)
        valid_corr = [v for v in corr_vals if v is not None]
        valid_noi = [v for v in noi_vals if v is not None]
        valid_diff = [v for v in diff_vals if v is not None]

        avg_corr = sum(valid_corr) / len(valid_corr) if valid_corr else 0.0
        avg_noi = sum(valid_noi) / len(valid_noi) if valid_noi else 0.0
        avg_diff = sum(valid_diff) / len(valid_diff) if valid_diff else 0.0

        retriever_corr_avgs.append((retriever, avg_corr))
        retriever_noi_avgs.append((retriever, avg_noi))
        retriever_diff_avgs.append((retriever, avg_diff))

        row += f" {avg_corr:>10.4f} {avg_noi:>10.4f} {avg_diff:>+10.4f}"
        print(row)

    print("-" * 100)

    # Summary: best retriever per category for correct and noisy
    print("\nBest Retriever by Correct P@10:")
    for cat in CATEGORIES:
        cat_data = all_data.get(cat, {}).get("correct", {})
        if not cat_data:
            continue
        best_retriever = max(cat_data.items(), key=lambda x: x[1] if x[1] else 0)
        print(f"  {cat}: {best_retriever[0]} (P@10 = {best_retriever[1]:.4f})")

    print("\nBest Retriever by Noisy P@10:")
    for cat in CATEGORIES:
        cat_data = all_data.get(cat, {}).get("noisy", {})
        if not cat_data:
            continue
        best_retriever = max(cat_data.items(), key=lambda x: x[1] if x[1] else 0)
        print(f"  {cat}: {best_retriever[0]} (P@10 = {best_retriever[1]:.4f})")

    print("\nMost Robust to Noise (Smallest Avg Diff):")
    best_robust = min(retriever_diff_avgs, key=lambda x: x[1] if x[1] is not None else float('inf'))
    print(f"  {best_robust[0]} (Avg Diff = {best_robust[1]:+.4f})")


# ============ Trend Analysis Functions ============

def load_segmented_data(category: str) -> dict:
    """加载 CORRECT 查询的分段 P@10 数据"""
    summary_path = BASE_DIR_08 / category / "retrieval_all_summary.json"
    with open(summary_path, "r") as f:
        data = json.load(f)

    rbct = data["results_by_category_and_type"]

    results = {}  # {retriever: {query_type: {level: p10}}}

    for key in rbct.keys():
        # key 是字符串形式的元组: "('acl', 'correct')"
        key_tuple = eval(key) if isinstance(key, str) else key
        query_type, query_category = key_tuple  # e.g., ('acl', 'correct')

        for item in rbct[key]:
            retriever = item["retriever"]
            group_metrics = item.get("group_metrics", {})

            if retriever not in results:
                results[retriever] = {}

            if query_type not in results[retriever]:
                results[retriever][query_type] = {}

            for level, metrics in group_metrics.items():
                results[retriever][query_type][level] = metrics.get("P@10", 0.0)

    return results


import numpy as np


def ols_trend(values: list) -> tuple:
    """
    使用 OLS 线性回归分析趋势
    返回 (斜率, 趋势描述)
    """
    if len(values) < 2:
        return 0.0, "数据不足"

    x = np.array([0, 1, 2, 3])  # level 0,1,2,3
    y = np.array(values)

    # OLS: 斜率 = sum((x-mean_x)(y-mean_y)) / sum((x-mean_x)^2)
    x_mean = x.mean()
    y_mean = y.mean()
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

    # 判断趋势
    if abs(slope) < 0.001:
        return slope, "持平"
    elif slope > 0:
        return slope, "上升"
    else:
        return slope, "下降"


def get_step_trend(v1: float, v2: float) -> str:
    """获取两个值之间的趋势"""
    if v2 > v1:
        return "上升"
    elif v2 < v1:
        return "下降"
    else:
        return "持平"


def analyze_level_full(domain_data: dict) -> dict:
    """完整分析单个域中 ACL 和 CCOMP 的趋势"""
    result = {}

    for query_cat in ["acl", "ccomp"]:
        if query_cat not in domain_data:
            continue

        cat_data = domain_data[query_cat]
        levels = ["0", "1", "2", "3"]
        values = [cat_data.get(l, None) for l in levels]

        if all(v is not None for v in values):
            slope, trend = ols_trend(values)
            # L0->L1, L1->L2, L2->L3
            steps = [
                get_step_trend(values[0], values[1]),
                get_step_trend(values[1], values[2]),
                get_step_trend(values[2], values[3]),
            ]
            result[query_cat] = {
                "values": values,
                "slope": slope,
                "trend": trend,
                "steps": steps
            }

    return result


def print_trend_analysis(all_data: dict):
    """打印 CORRECT 查询的 ACL/CCOMP Level 趋势分析 (OLS + Step)"""
    # 收集所有检索器
    all_retrievers = set()
    for cat_data in all_data.values():
        all_retrievers.update(cat_data.keys())
    all_retrievers = sorted(all_retrievers)

    print("\n" + "=" * 100)
    print("CORRECT Query - ACL/CCOMP Level Trend Analysis (P@10)")
    print("OLS整体趋势 + 每个Step的统一性判断")
    print("=" * 100)

    for retriever in all_retrievers:
        print(f"\n{'='*80}")
        print(f"Retriever: {retriever.upper()}")
        print(f"{'='*80}")

        # 收集三个域的完整趋势
        domain_info = {}
        for cat in CATEGORIES:
            short = CATEGORY_SHORT_NAMES.get(cat, cat[:6])
            if cat not in all_data or retriever not in all_data[cat]:
                continue

            cat_data = all_data[cat][retriever]
            info = analyze_level_full(cat_data)
            domain_info[short] = info

        # ACL 分析
        print("\n[ACL]")
        acl_values = {short: t.get("acl", {}).get("values", [])
                      for short, t in domain_info.items()}
        acl_slopes = {short: t.get("acl", {}).get("slope", 0.0)
                      for short, t in domain_info.items()}
        acl_trends = {short: t.get("acl", {}).get("trend", "N/A")
                      for short, t in domain_info.items()}
        acl_steps = {short: t.get("acl", {}).get("steps", [])
                     for short, t in domain_info.items()}

        # 打印各域值、OLS斜率
        for short in ["Baby", "Grocery", "Pet"]:
            vals = acl_values.get(short, [])
            slope = acl_slopes.get(short, 0.0)
            trend = acl_trends.get(short, "N/A")
            vals_str = " | ".join([f"L{i}={v:.4f}" for i, v in enumerate(vals)]) if vals else "N/A"
            print(f"  {short:8}: {vals_str}  (OLS slope={slope:+.4f}, {trend})")

        # Step 统一性判断
        step_names = ["L0->L1", "L1->L2", "L2->L3"]
        print("  Step统一性:")
        for i, step_name in enumerate(step_names):
            step_trends = {short: s[i] if i < len(s) else "N/A"
                          for short, s in acl_steps.items()}
            if not step_trends:
                continue
            all_same = len(set(step_trends.values())) == 1
            unified = list(step_trends.values())[0]
            if all_same:
                print(f"    {step_name}: ★ 统一 = {unified}")
            else:
                print(f"    {step_name}: 不统一 ({step_trends})")

        # OLS 整体统一性
        print("  OLS整体:")
        if acl_trends:
            all_same = len(set(acl_trends.values())) == 1
            unified = list(acl_trends.values())[0]
            if all_same and unified in ["上升", "下降"]:
                avg_slope = sum(acl_slopes.values()) / len(acl_slopes)
                print(f"    ★ 统一趋势 = {unified} (avg slope = {avg_slope:+.4f})")
            elif all_same:
                print(f"    {unified}")
            else:
                print(f"    不统一 ({acl_trends})")

        # CCOMP 分析
        print("\n[CCOMP]")
        ccomp_values = {short: t.get("ccomp", {}).get("values", [])
                        for short, t in domain_info.items()}
        ccomp_slopes = {short: t.get("ccomp", {}).get("slope", 0.0)
                        for short, t in domain_info.items()}
        ccomp_trends = {short: t.get("ccomp", {}).get("trend", "N/A")
                        for short, t in domain_info.items()}
        ccomp_steps = {short: t.get("ccomp", {}).get("steps", [])
                       for short, t in domain_info.items()}

        # 打印各域值、OLS斜率
        for short in ["Baby", "Grocery", "Pet"]:
            vals = ccomp_values.get(short, [])
            slope = ccomp_slopes.get(short, 0.0)
            trend = ccomp_trends.get(short, "N/A")
            vals_str = " | ".join([f"L{i}={v:.4f}" for i, v in enumerate(vals)]) if vals else "N/A"
            print(f"  {short:8}: {vals_str}  (OLS slope={slope:+.4f}, {trend})")

        # Step 统一性判断
        print("  Step统一性:")
        for i, step_name in enumerate(step_names):
            step_trends = {short: s[i] if i < len(s) else "N/A"
                          for short, s in ccomp_steps.items()}
            if not step_trends:
                continue
            all_same = len(set(step_trends.values())) == 1
            unified = list(step_trends.values())[0]
            if all_same:
                print(f"    {step_name}: ★ 统一 = {unified}")
            else:
                print(f"    {step_name}: 不统一 ({step_trends})")

        # OLS 整体统一性
        print("  OLS整体:")
        if ccomp_trends:
            all_same = len(set(ccomp_trends.values())) == 1
            unified = list(ccomp_trends.values())[0]
            if all_same and unified in ["上升", "下降"]:
                avg_slope = sum(ccomp_slopes.values()) / len(ccomp_slopes)
                print(f"    ★ 统一趋势 = {unified} (avg slope = {avg_slope:+.4f})")
            elif all_same:
                print(f"    {unified}")
            else:
                print(f"    不统一 ({ccomp_trends})")


def main():
    # Load clean data from 08_retrieval
    print("Loading clean query data from 08_retrieval...")
    clean_data = {}
    for cat in CATEGORIES:
        try:
            clean_data[cat] = load_clean_data(cat)
            print(f"  Loaded {cat}: {len(clean_data[cat])} retrievers")
        except Exception as e:
            print(f"  Error loading {cat}: {e}")

    # Load correct vs noisy data from 09_noisy_retrieval
    print("\nLoading correct vs noisy data from 09_noisy_retrieval...")
    noisy_data = {}
    for cat in CATEGORIES:
        try:
            noisy_data[cat] = load_correct_vs_noisy_data(cat)
            n_correct = len(noisy_data[cat].get("correct", {}))
            n_noisy = len(noisy_data[cat].get("noisy", {}))
            print(f"  Loaded {cat}: {n_correct} correct retrievers, {n_noisy} noisy retrievers")
        except Exception as e:
            print(f"  Error loading {cat}: {e}")

    # Load segmented data for trend analysis
    print("\nLoading segmented data for trend analysis...")
    trend_data = {}
    for cat in CATEGORIES:
        try:
            trend_data[cat] = load_segmented_data(cat)
            print(f"  Loaded {cat}: {len(trend_data[cat])} retrievers")
        except Exception as e:
            print(f"  Error loading {cat}: {e}")

    # Print clean comparison
    print_clean_comparison(clean_data)

    # Print correct vs noisy comparison (one table per domain)
    print_correct_vs_noisy_by_domain(noisy_data)

    # Print summary
    print_correct_vs_noisy_summary(noisy_data)

    # Print trend analysis
    print_trend_analysis(trend_data)

    print("\n" + "=" * 100)
    print("Comparison Complete")
    print("=" * 100)


if __name__ == "__main__":
    main()