"""
Compare P@10 values across 3 domains (categories) and all retrievers.
Includes both:
1. Clean query results (from 08_retrieval/retrieval_all_summary.json)
2. Correct vs Noisy query results (from 09_noisy_retrieval/correct_vs_noisy_results.json)

Reads retrieval_all_summary.json from each domain and prints a comparison table.
"""

import json
import csv
from pathlib import Path

BASE_DIR_08 = Path("/home/wlia0047/ar57/wenyu/result/personal_query/08_retrieval")
BASE_DIR_09 = Path("/home/wlia0047/ar57/wenyu/result/personal_query/09_noisy_retrieval")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
CATEGORY_SHORT_NAMES = {
    "Baby_Products": "Baby",
    "Grocery_and_Gourmet_Food": "Grocery",
    "Pet_Supplies": "Pet",
}

# 对比脚本中禁用的检索器
DISABLED_RETRIEVERS = set()
RETRIEVER_ORDER = [
    "bm25",
    "splade",
    "bge",
    "e5",
    "minilm",
    "star",
    "ance",
    "colbertv2",
    "gritlm",
]


def round4(x: float) -> float:
    """统一保留 4 位小数用于趋势比较。"""
    return round(float(x), 4)


def sort_retrievers(retrievers) -> list:
    """按论文表格常用顺序排序，新增检索器排在已知顺序之后。"""
    order = {name: idx for idx, name in enumerate(RETRIEVER_ORDER)}
    return sorted(retrievers, key=lambda name: (order.get(name, len(order)), name))


def load_clean_data(category: str) -> dict:
    """Load clean P@10 data for all retrievers from ACL/CCOMP correct results."""
    summary_path = BASE_DIR_08 / category / "retrieval_all_summary.json"
    with open(summary_path, "r") as f:
        data = json.load(f)

    rbct = data["results_by_category_and_type"]
    correct_keys = ["('acl', 'correct')", "('ccomp', 'correct')"]
    missing_keys = [key for key in correct_keys if key not in rbct]
    if missing_keys:
        raise KeyError(f"{category} missing clean correct result keys: {missing_keys}")

    grouped_values = {}
    for key in correct_keys:
        for item in rbct[key]:
            retriever = item["retriever"]
            if retriever in DISABLED_RETRIEVERS:
                continue
            group_metrics = item["group_metrics"]

            if retriever not in grouped_values:
                grouped_values[retriever] = {}
            for level, metrics in group_metrics.items():
                if "P@10" not in metrics:
                    raise KeyError(f"{category} {key} {retriever} level {level} missing P@10")
                if level not in grouped_values[retriever]:
                    grouped_values[retriever][level] = []
                grouped_values[retriever][level].append(metrics["P@10"])

    results = {}
    for retriever, level_values in grouped_values.items():
        results[retriever] = {}
        for level, values in level_values.items():
            if not values:
                raise ValueError(f"{category} {retriever} level {level} has no P@10 values")
            results[retriever][level] = sum(values) / len(values)

    return results


def load_clean_overall_data(category: str) -> dict:
    """Load overall clean P@10 by query family from all_results_combined."""
    summary_path = BASE_DIR_08 / category / "retrieval_all_summary.json"
    with open(summary_path, "r") as f:
        data = json.load(f)

    results = {}
    for item in data["all_results_combined"]:
        retriever = item["retriever"]
        if retriever in DISABLED_RETRIEVERS:
            continue
        query_category = item["query_category"]
        query_type = item["query_type"]
        if query_type != "correct" or query_category not in ("acl", "ccomp"):
            continue
        metrics = item["metrics"]
        if "P@10" not in metrics:
            raise KeyError(f"{category} {query_category}/{query_type} {retriever} missing P@10")
        if retriever not in results:
            results[retriever] = {}
        results[retriever][query_category] = metrics["P@10"]

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
        if retriever in DISABLED_RETRIEVERS:
            continue
        p10 = item["metrics"].get("P@10", 0.0)
        results["correct"][retriever] = p10

    # Parse noisy results
    for item in data.get("noisy_results", []):
        retriever = item["retriever"]
        if retriever in DISABLED_RETRIEVERS:
            continue
        p10 = item["metrics"].get("P@10", 0.0)
        results["noisy"][retriever] = p10

    return results


def print_clean_comparison(all_data: dict):
    """Print clean query P@10 comparison table."""
    all_retrievers = set()
    for cat_data in all_data.values():
        all_retrievers.update(cat_data.keys())
    all_retrievers = sort_retrievers(r for r in all_retrievers if r not in DISABLED_RETRIEVERS)

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
            if retriever in DISABLED_RETRIEVERS:
                continue
            if isinstance(val, dict):
                vals = [v for v in val.values() if v is not None]
                avg = sum(vals) / len(vals) if vals else 0.0
            else:
                avg = val if val else 0.0
            if avg > best_avg:
                best_avg = avg
                best_retriever = retriever
        print(f"  {cat}: {best_retriever} (P@10 = {best_avg:.4f})")


def print_clean_family_overall_comparison(all_data: dict):
    """Print ACL/CCOMP overall clean P@10 comparison across domains."""
    all_retrievers = set()
    for cat_data in all_data.values():
        all_retrievers.update(cat_data.keys())
    all_retrievers = sort_retrievers(r for r in all_retrievers if r not in DISABLED_RETRIEVERS)

    print("\n" + "=" * 130)
    print("Clean Query Overall P@10 by Family Across Domains (from 08_retrieval)")
    print("=" * 130)
    print(
        f"{'Retriever':<12} "
        f"{'Baby ACL':>10} {'Baby CC':>10} "
        f"{'Grocery ACL':>12} {'Grocery CC':>12} "
        f"{'Pet ACL':>10} {'Pet CC':>10} "
        f"{'Avg ACL':>10} {'Avg CC':>10}"
    )
    print("-" * 130)

    for retriever in all_retrievers:
        row = f"{retriever:<12}"
        acl_vals = []
        ccomp_vals = []
        for cat in CATEGORIES:
            cat_data = all_data.get(cat, {}).get(retriever, {})
            acl = cat_data.get("acl")
            ccomp = cat_data.get("ccomp")
            if acl is not None:
                acl_vals.append(acl)
            if ccomp is not None:
                ccomp_vals.append(ccomp)
            row += f" {fmt4(acl):>10} {fmt4(ccomp):>10}"

        avg_acl = sum(acl_vals) / len(acl_vals) if acl_vals else None
        avg_ccomp = sum(ccomp_vals) / len(ccomp_vals) if ccomp_vals else None
        row += f" {fmt4(avg_acl):>10} {fmt4(avg_ccomp):>10}"
        print(row)

    print("-" * 130)


def print_correct_vs_noisy_by_domain(all_data: dict):
    """Print correct vs noisy P@10 comparison table, one table per domain."""
    # Collect all retrievers
    all_retrievers = set()
    for cat_data in all_data.values():
        if "correct" in cat_data:
            all_retrievers.update(cat_data["correct"].keys())
        if "noisy" in cat_data:
            all_retrievers.update(cat_data["noisy"].keys())
    all_retrievers = sort_retrievers(r for r in all_retrievers if r not in DISABLED_RETRIEVERS)

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
    all_retrievers = sort_retrievers(r for r in all_retrievers if r not in DISABLED_RETRIEVERS)

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

def sign_from_delta(delta: float) -> str:
    """根据 OLS 差值方向判定趋势（按 4 位小数比较）。"""
    d = round4(delta)
    if d > 0:
        return "上升"
    if d < 0:
        return "下降"
    return "持平"


def load_segmented_data(category: str) -> dict:
    """加载 within_family_ols_results.csv，并提取 OLS 后的 step 差值。"""
    csv_path = BASE_DIR_08 / category / "within_family_ols_results.csv"
    results = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            retriever = row.get("retriever", "").strip()
            if not retriever or retriever in DISABLED_RETRIEVERS:
                continue

            family = row.get("family", "").strip().lower()
            if family not in ("acl", "ccomp"):
                continue

            d01 = float(row["delta_l1_vs_l0"])
            d12 = float(row["delta_l2_vs_l1"])
            d23 = float(row["delta_l3_vs_l2"])
            p01 = float(row["p_l1_vs_l0"])
            p12 = float(row["p_l2_vs_l1"])
            p23 = float(row["p_l3_vs_l2"])

            if retriever not in results:
                results[retriever] = {}

            results[retriever][family] = {
                "deltas": {
                    "L0->L1": d01,
                    "L1->L2": d12,
                    "L2->L3": d23,
                },
                "pvals": {
                    "L0->L1": p01,
                    "L1->L2": p12,
                    "L2->L3": p23,
                },
                "steps": {
                    "L0->L1": sign_from_delta(d01),
                    "L1->L2": sign_from_delta(d12),
                    "L2->L3": sign_from_delta(d23),
                },
            }

    return results


def load_ols_adjusted_hit10(category: str) -> dict:
    """加载 within_family_ols_results.csv，并恢复每个 family 各 level 的 OLS-adjusted hit@10。"""
    csv_path = BASE_DIR_08 / category / "within_family_ols_results.csv"
    results = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            retriever = row.get("retriever", "").strip()
            if not retriever or retriever in DISABLED_RETRIEVERS:
                continue

            family = row.get("family", "").strip().lower()
            if family not in ("acl", "ccomp"):
                continue

            l0 = float(row["intercept_l0_at_mean_covariates"])
            d10 = float(row["delta_l1_vs_l0"])
            d20 = float(row["delta_l2_vs_l0"])
            d30 = float(row["delta_l3_vs_l0"])

            if retriever not in results:
                results[retriever] = {}

            results[retriever][family] = {
                "L0": l0,
                "L1": l0 + d10,
                "L2": l0 + d20,
                "L3": l0 + d30,
            }

    return results


def print_ols_adjusted_hit10(all_data: dict):
    """按域拆分打印 3 张表，并补一个 Avg 小表。"""
    all_retrievers = set()
    for cat_data in all_data.values():
        all_retrievers.update(cat_data.keys())
    all_retrievers = sort_retrievers(all_retrievers)

    level_order = ["L0", "L1", "L2", "L3"]
    domain_order = [
        ("Baby_Products", "Baby"),
        ("Grocery_and_Gourmet_Food", "Grocery"),
        ("Pet_Supplies", "Pet"),
    ]

    print("\n" + "=" * 100)
    print("CORRECT Query - OLS-adjusted hit@10 Across 3 Domains")
    print("分成 3 个域表显示，最后附 Avg 小表")
    print("=" * 100)

    for family in ["acl", "ccomp"]:
        print(f"\n[{family.upper()} Across Domains]")
        print("-" * 136)
        print(
            f"{'':<12} "
            f"{'Baby':^35} "
            f"{'Grocery':^35} "
            f"{'Pet':^35}"
        )
        print(
            f"{'Retriever':<12} "
            f"{'L0':>8} {'L1':>8} {'L2':>8} {'L3':>8} | "
            f"{'L0':>8} {'L1':>8} {'L2':>8} {'L3':>8} | "
            f"{'L0':>8} {'L1':>8} {'L2':>8} {'L3':>8}"
        )
        print("-" * 136)
        for retriever in all_retrievers:
            baby_data = all_data.get("Baby_Products", {}).get(retriever, {}).get(family, {})
            grocery_data = all_data.get("Grocery_and_Gourmet_Food", {}).get(retriever, {}).get(family, {})
            pet_data = all_data.get("Pet_Supplies", {}).get(retriever, {}).get(family, {})
            if not any([baby_data, grocery_data, pet_data]):
                continue
            print(
                f"{retriever.upper():<12} "
                f"{pct_str(baby_data.get('L0')):>8} {pct_str(baby_data.get('L1')):>8} {pct_str(baby_data.get('L2')):>8} {pct_str(baby_data.get('L3')):>8} | "
                f"{pct_str(grocery_data.get('L0')):>8} {pct_str(grocery_data.get('L1')):>8} {pct_str(grocery_data.get('L2')):>8} {pct_str(grocery_data.get('L3')):>8} | "
                f"{pct_str(pet_data.get('L0')):>8} {pct_str(pet_data.get('L1')):>8} {pct_str(pet_data.get('L2')):>8} {pct_str(pet_data.get('L3')):>8}"
            )
        print("-" * 136)

    print("\n[Avg Across 3 Domains]")
    print("-" * 132)
    print(
        f"{'':<12} {'ACL':^48}    {'CCOMP':^48}"
    )
    print(
        f"{'Retriever':<12} "
        f"{'L0':>10} {'L1':>10} {'L2':>10} {'L3':>10}    "
        f"{'L0':>10} {'L1':>10} {'L2':>10} {'L3':>10}"
    )
    print("-" * 132)
    for retriever in all_retrievers:
        family_avgs = {}
        has_any = False
        for family in ["acl", "ccomp"]:
            avg_vals = []
            has_data = False
            for level in level_order:
                _, _, _, avg = get_ols_domain_values(all_data, retriever, family, level)
                avg_vals.append(avg)
                if avg is not None:
                    has_data = True
            family_avgs[family] = avg_vals if has_data else [None, None, None, None]
            has_any = has_any or has_data
        if not has_any:
            continue
        acl_vals = family_avgs["acl"]
        ccomp_vals = family_avgs["ccomp"]
        print(
            f"{retriever.upper():<12} "
            f"{pct_str(acl_vals[0]):>10} {pct_str(acl_vals[1]):>10} {pct_str(acl_vals[2]):>10} {pct_str(acl_vals[3]):>10}    "
            f"{pct_str(ccomp_vals[0]):>10} {pct_str(ccomp_vals[1]):>10} {pct_str(ccomp_vals[2]):>10} {pct_str(ccomp_vals[3]):>10}"
        )
    print("-" * 132)


def get_ols_domain_values(all_data: dict, retriever: str, family: str, level: str):
    """返回指定 retriever/family/level 在三个域上的值及平均值。"""
    baby = all_data.get("Baby_Products", {}).get(retriever, {}).get(family, {}).get(level)
    grocery = all_data.get("Grocery_and_Gourmet_Food", {}).get(retriever, {}).get(family, {}).get(level)
    pet = all_data.get("Pet_Supplies", {}).get(retriever, {}).get(family, {}).get(level)
    vals = [v for v in [baby, grocery, pet] if v is not None]
    avg = sum(vals) / len(vals) if vals else None
    return baby, grocery, pet, avg


def pct_str(value):
    """百分比字符串。"""
    return f"{value * 100:.2f}%" if value is not None else "N/A"


def fmt4(value):
    """4 位小数字符串。"""
    return f"{value:.4f}" if value is not None else "N/A"


def print_ols_adjusted_hit10_avg_wide(all_data: dict):
    """方案 1: 每个 retriever 一行，只显示 Avg，ACL/CCOMP 横向展开。"""
    all_retrievers = sort_retrievers({r for cat_data in all_data.values() for r in cat_data.keys()})
    levels = ["L0", "L1", "L2", "L3"]

    print("\n" + "=" * 120)
    print("OLS-adjusted hit@10 Layout 1 - Avg-only Wide Table")
    print("每行一个 Retriever，仅保留 Avg；列为 ACL/CCOMP 的 L0-L3")
    print("=" * 120)
    print(
        f"{'Retriever':<12} "
        f"{'ACL-L0':>10} {'ACL-L1':>10} {'ACL-L2':>10} {'ACL-L3':>10} "
        f"{'CC-L0':>10} {'CC-L1':>10} {'CC-L2':>10} {'CC-L3':>10}"
    )
    print("-" * 120)

    for retriever in all_retrievers:
        row = f"{retriever:<12}"
        for family in ["acl", "ccomp"]:
            for level in levels:
                _, _, _, avg = get_ols_domain_values(all_data, retriever, family, level)
                row += f" {pct_str(avg):>10}"
        print(row)


def print_ols_adjusted_hit10_compact_cells(all_data: dict):
    """方案 2: 每个 retriever/family 一行，每个单元格为 B/G/P/A。"""
    all_retrievers = sort_retrievers({r for cat_data in all_data.values() for r in cat_data.keys()})
    levels = ["L0", "L1", "L2", "L3"]

    print("\n" + "=" * 160)
    print("OLS-adjusted hit@10 Layout 2 - Compact Cells")
    print("每个 cell = Baby/Grocery/Pet/Avg (%)")
    print("=" * 160)
    print(f"{'Retriever':<12} {'Family':<8} {'L0':>30} {'L1':>30} {'L2':>30} {'L3':>30}")
    print("-" * 160)

    for retriever in all_retrievers:
        for family in ["acl", "ccomp"]:
            quads = []
            has_data = False
            for level in levels:
                baby, grocery, pet, avg = get_ols_domain_values(all_data, retriever, family, level)
                if any(v is not None for v in [baby, grocery, pet, avg]):
                    has_data = True
                quad = f"{pct_str(baby)}/{pct_str(grocery)}/{pct_str(pet)}/{pct_str(avg)}"
                quads.append(quad)
            if not has_data:
                continue
            print(f"{retriever:<12} {family.upper():<8} {quads[0]:>30} {quads[1]:>30} {quads[2]:>30} {quads[3]:>30}")


def print_ols_adjusted_hit10_trend_strings(all_data: dict):
    """方案 3: 每个 retriever/family 一行，仅显示 Avg 的趋势串。"""
    all_retrievers = sort_retrievers({r for cat_data in all_data.values() for r in cat_data.keys()})
    levels = ["L0", "L1", "L2", "L3"]

    print("\n" + "=" * 100)
    print("OLS-adjusted hit@10 Layout 3 - Avg Trend Strings")
    print("每行一个 Retriever / Family，仅显示 Avg 趋势")
    print("=" * 100)

    for retriever in all_retrievers:
        for family in ["acl", "ccomp"]:
            avg_vals = []
            for level in levels:
                _, _, _, avg = get_ols_domain_values(all_data, retriever, family, level)
                avg_vals.append(avg)
            if not any(v is not None for v in avg_vals):
                continue
            trend = " -> ".join(pct_str(v) for v in avg_vals)
            print(f"{retriever.upper():<10} {family.upper():<6}: {trend}")


def print_trend_analysis(all_data: dict):
    """打印基于 Within-Family OLS 差值的 ACL/CCOMP Step 趋势统一性。"""
    # 收集所有检索器
    all_retrievers = set()
    for cat_data in all_data.values():
        all_retrievers.update(cat_data.keys())
    all_retrievers = sort_retrievers(all_retrievers)

    print("\n" + "=" * 100)
    print("CORRECT Query - ACL/CCOMP Step Trend Analysis (Within-Family OLS)")
    print("基于 OLS 差值: L1vsL0, L2vsL1, L3vsL2")
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

            domain_info[short] = all_data[cat][retriever]

        step_names = ["L0->L1", "L1->L2", "L2->L3"]
        for family in ["acl", "ccomp"]:
            print(f"\n[{family.upper()}]")

            for short in ["Baby", "Grocery", "Pet"]:
                fam_data = domain_info.get(short, {}).get(family, {})
                deltas = fam_data.get("deltas", {})
                pvals = fam_data.get("pvals", {})
                steps = fam_data.get("steps", {})
                if not deltas:
                    print(f"  {short:8}: N/A")
                    continue
                d01 = deltas["L0->L1"]
                d12 = deltas["L1->L2"]
                d23 = deltas["L2->L3"]
                p01 = pvals["L0->L1"]
                p12 = pvals["L1->L2"]
                p23 = pvals["L2->L3"]
                s01 = steps["L0->L1"]
                s12 = steps["L1->L2"]
                s23 = steps["L2->L3"]
                print(
                    f"  {short:8}: "
                    f"L0->L1={d01:+.4f} (p={p01:.4g}, {s01}) | "
                    f"L1->L2={d12:+.4f} (p={p12:.4g}, {s12}) | "
                    f"L2->L3={d23:+.4f} (p={p23:.4g}, {s23})"
                )

            print("  Step统一性:")
            for step_name in step_names:
                step_trends = {}
                for short in ["Baby", "Grocery", "Pet"]:
                    trend = domain_info.get(short, {}).get(family, {}).get("steps", {}).get(step_name)
                    if trend is not None:
                        step_trends[short] = trend

                if len(step_trends) < 2:
                    print(f"    {step_name}: 数据不足 ({step_trends})")
                    continue

                all_same = len(set(step_trends.values())) == 1
                if all_same:
                    unified = list(step_trends.values())[0]
                    print(f"    {step_name}: ★ 统一 = {unified}")
                else:
                    print(f"    {step_name}: 不统一 ({step_trends})")


def main():
    # Load clean data from 08_retrieval
    print("Loading clean query data from 08_retrieval...")
    clean_data = {}
    for cat in CATEGORIES:
        clean_data[cat] = load_clean_data(cat)
        print(f"  Loaded {cat}: {len(clean_data[cat])} retrievers")

    print("\nLoading clean query family overall data from 08_retrieval...")
    clean_family_overall_data = {}
    for cat in CATEGORIES:
        clean_family_overall_data[cat] = load_clean_overall_data(cat)
        print(f"  Loaded {cat}: {len(clean_family_overall_data[cat])} retrievers")

    # Load correct vs noisy data from 09_noisy_retrieval
    print("\nLoading correct vs noisy data from 09_noisy_retrieval...")
    noisy_data = {}
    for cat in CATEGORIES:
        noisy_data[cat] = load_correct_vs_noisy_data(cat)
        n_correct = len(noisy_data[cat].get("correct", {}))
        n_noisy = len(noisy_data[cat].get("noisy", {}))
        print(f"  Loaded {cat}: {n_correct} correct retrievers, {n_noisy} noisy retrievers")

    # Load within-family OLS data for trend analysis
    print("\nLoading within-family OLS data for trend analysis...")
    trend_data = {}
    for cat in CATEGORIES:
        trend_data[cat] = load_segmented_data(cat)
        print(f"  Loaded {cat}: {len(trend_data[cat])} retrievers")

    print("\nLoading OLS-adjusted hit@10 data...")
    ols_hit10_data = {}
    for cat in CATEGORIES:
        ols_hit10_data[cat] = load_ols_adjusted_hit10(cat)
        print(f"  Loaded {cat}: {len(ols_hit10_data[cat])} retrievers")

    # Print clean comparison
    print_clean_comparison(clean_data)
    print_clean_family_overall_comparison(clean_family_overall_data)

    # Print correct vs noisy comparison (one table per domain)
    print_correct_vs_noisy_by_domain(noisy_data)

    # Print summary
    print_correct_vs_noisy_summary(noisy_data)

    # Print OLS-adjusted hit@10
    print_ols_adjusted_hit10(ols_hit10_data)

    # Print trend analysis
    print_trend_analysis(trend_data)

    print("\n" + "=" * 100)
    print("Comparison Complete")
    print("=" * 100)


if __name__ == "__main__":
    main()
