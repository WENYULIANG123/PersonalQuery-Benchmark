"""
比较三个域上的 syntax-depth hit@10。

08 侧读取 retrieval_syntax_depth_summary.json 中的低/中/深分组 H@10。
09 侧读取 syntax_depth_correct_vs_noisy_results.json 中的 correct/noisy 低/中分组 H@10。
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

BASE_DIR_08 = Path("/home/wlia0047/ar57/wenyu/result/personal_query/08_retrieval")
BASE_DIR_09 = Path("/home/wlia0047/ar57/wenyu/result/personal_query/09_noisy_retrieval")
SYNTAX_DEPTH_CLEAN_SUMMARY_FILE = "retrieval_syntax_depth_summary.json"
SYNTAX_DEPTH_NOISY_RESULTS_FILE = "syntax_depth_correct_vs_noisy_results.json"

CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]

SYNTAX_DEPTH_GROUP_ORDER_CLEAN = ["low_complexity", "medium_complexity", "high_complexity"]
SYNTAX_DEPTH_GROUP_ORDER_NOISY = ["low_complexity", "medium_complexity"]

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


def sort_retrievers(retrievers) -> list:
    order = {name: idx for idx, name in enumerate(RETRIEVER_ORDER)}
    return sorted(retrievers, key=lambda name: (order.get(name, len(order)), name))


def fmt4(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def fmt_signed(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.4f}"


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return f"{p_value:.3e}"
    return f"{p_value:.4f}"


def significance_star(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def average_domain_values(values: list) -> float:
    if len(values) != len(CATEGORIES):
        raise ValueError(f"Expected {len(CATEGORIES)} domain values, got {len(values)}")
    return float(sum(values) / len(values))


def get_group_hit10(group_data: dict, context: str) -> float:
    if not isinstance(group_data, dict):
        raise TypeError(f"{context} group data must be dict, got {type(group_data).__name__}")
    metrics = group_data.get("metrics", group_data)
    if not isinstance(metrics, dict):
        raise TypeError(f"{context} metrics must be dict, got {type(metrics).__name__}")
    value = metrics.get("H@10")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{context} missing numeric H@10")
    return float(value)


def get_record_hit10(record: dict, context: str) -> float:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError(f"{context} metrics must be dict")
    value = metrics.get("H@10")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{context} missing numeric metrics['H@10']")
    return float(value)


def load_08_group_hit10(category: str) -> dict:
    summary_path = BASE_DIR_08 / category / SYNTAX_DEPTH_CLEAN_SUMMARY_FILE
    data = load_json(summary_path)
    results = {}

    for index, item in enumerate(data["all_results_combined"]):
        retriever = item.get("retriever")
        if retriever in DISABLED_RETRIEVERS:
            continue
        if retriever in results:
            raise ValueError(f"{category} duplicate 08 retriever: {retriever}")
        if item.get("query_category") != "syntax_depth":
            raise ValueError(f"{category} 08 item {index} unexpected query_category={item.get('query_category')}")
        if item.get("query_type") != "correct":
            raise ValueError(f"{category} 08 item {index} unexpected query_type={item.get('query_type')}")

        group_metrics = item.get("group_metrics")
        if not isinstance(group_metrics, dict):
            raise TypeError(f"{category} {retriever} group_metrics must be dict")

        results[retriever] = {}
        for group_name in SYNTAX_DEPTH_GROUP_ORDER_CLEAN:
            if group_name not in group_metrics:
                raise KeyError(f"{category} {retriever} missing 08 group {group_name}")
            results[retriever][group_name] = get_group_hit10(
                group_metrics[group_name],
                f"{category} {retriever} 08 {group_name}",
            )

    if not results:
        raise ValueError(f"{category} 08 group hit@10 results are empty")
    return results


def load_09_group_hit10(category: str) -> dict:
    results_path = BASE_DIR_09 / category / SYNTAX_DEPTH_NOISY_RESULTS_FILE
    data = load_json(results_path)
    results = {"correct": {}, "noisy": {}}

    for source_name, output_key, expected_query_type in (
        ("correct_results", "correct", "correct"),
        ("noisy_results", "noisy", "noisy"),
    ):
        for index, item in enumerate(data.get(source_name, [])):
            retriever = item.get("retriever")
            if retriever in DISABLED_RETRIEVERS:
                continue
            if retriever in results[output_key]:
                raise ValueError(f"{category} duplicate 09 {source_name} retriever: {retriever}")
            if item.get("query_category") != "syntax_depth":
                raise ValueError(
                    f"{category} 09 {source_name} item {index} "
                    f"unexpected query_category={item.get('query_category')}"
                )
            if item.get("query_type") != expected_query_type:
                raise ValueError(
                    f"{category} 09 {source_name} item {index} "
                    f"unexpected query_type={item.get('query_type')}"
                )

            group_metrics = item.get("metrics_by_depth_group")
            if not isinstance(group_metrics, dict):
                raise TypeError(f"{category} {retriever} 09 {source_name} metrics_by_depth_group must be dict")

            results[output_key][retriever] = {}
            for group_name in SYNTAX_DEPTH_GROUP_ORDER_NOISY:
                if group_name not in group_metrics:
                    raise KeyError(f"{category} {retriever} missing 09 group {group_name}")
                results[output_key][retriever][group_name] = get_group_hit10(
                    group_metrics[group_name],
                    f"{category} {retriever} 09 {source_name} {group_name}",
                )

    if set(results["correct"]) != set(results["noisy"]):
        raise KeyError(
            f"{category} 09 correct/noisy retriever sets differ: "
            f"correct={sorted(results['correct'])}, noisy={sorted(results['noisy'])}"
        )
    return results


def build_08_results() -> dict:
    return {category: load_08_group_hit10(category) for category in CATEGORIES}


def build_09_results() -> dict:
    return {category: load_09_group_hit10(category) for category in CATEGORIES}


def get_all_retrievers_from_08(all_data: dict) -> list:
    retrievers = set()
    for cat_data in all_data.values():
        retrievers.update(cat_data.keys())
    return sort_retrievers(retrievers)


def get_all_retrievers_from_09(all_data: dict) -> list:
    retrievers = set()
    for cat_data in all_data.values():
        retrievers.update(cat_data["correct"].keys())
        retrievers.update(cat_data["noisy"].keys())
    return sort_retrievers(retrievers)


def print_08_hit10_table(all_data: dict) -> None:
    retrievers = get_all_retrievers_from_08(all_data)

    print("\n" + "=" * 132)
    print("08 clean syntax-depth hit@10")
    print("Mean 低/中/深为三域均值，Mean = (|Mean 中 - Mean 低| + |Mean 深 - Mean 中|) / 2")
    print("=" * 132)
    print(
        f"{'Retriever':<12} "
        f"{'Baby 低':>9} {'Baby 中':>9} {'Baby 深':>9} "
        f"{'Grocery 低':>11} {'Grocery 中':>11} {'Grocery 深':>11} "
        f"{'Pet 低':>9} {'Pet 中':>9} {'Pet 深':>9} "
        f"{'Mean 低':>9} {'Mean 中':>9} {'Mean 深':>9} {'Mean':>9}"
    )
    print("-" * 132)

    for retriever in retrievers:
        row = f"{retriever:<12}"
        mean_by_group = {group: [] for group in SYNTAX_DEPTH_GROUP_ORDER_CLEAN}
        for category in CATEGORIES:
            if retriever not in all_data[category]:
                raise KeyError(f"{category} missing 08 result for {retriever}")
            group_values = all_data[category][retriever]
            for group_name in SYNTAX_DEPTH_GROUP_ORDER_CLEAN:
                value = group_values[group_name]
                mean_by_group[group_name].append(value)
                row += f" {fmt4(value):>9}"

        group_means = []
        for group_name in SYNTAX_DEPTH_GROUP_ORDER_CLEAN:
            group_mean = average_domain_values(mean_by_group[group_name])
            group_means.append(group_mean)
            row += f" {fmt4(group_mean):>9}"
        low_mid_delta = group_means[1] - group_means[0]
        mid_high_delta = group_means[2] - group_means[1]
        adjacent_delta_mean = float((abs(low_mid_delta) + abs(mid_high_delta)) / 2.0)
        row += f" {fmt4(adjacent_delta_mean):>9}"
        print(row)
    print("-" * 132)


def print_08_adjacent_trend(all_data: dict) -> None:
    retrievers = get_all_retrievers_from_08(all_data)

    print("\n" + "=" * 110)
    print("08 clean syntax-depth adjacent trend")
    print("相邻差值基于原始分组 hit@10：中-低、深-中；mean 为三域差值均值")
    print("=" * 110)
    print(
        f"{'Retriever':<12} "
        f"{'Baby 中-低':>12} {'Baby 深-中':>12} "
        f"{'Grocery 中-低':>14} {'Grocery 深-中':>14} "
        f"{'Pet 中-低':>12} {'Pet 深-中':>12} "
        f"{'Mean 中-低':>12} {'Mean 深-中':>12}"
    )
    print("-" * 110)

    for retriever in retrievers:
        row = f"{retriever:<12}"
        low_to_mid = []
        mid_to_high = []
        for category in CATEGORIES:
            group_values = all_data[category][retriever]
            d_low_mid = group_values["medium_complexity"] - group_values["low_complexity"]
            d_mid_high = group_values["high_complexity"] - group_values["medium_complexity"]
            low_to_mid.append(d_low_mid)
            mid_to_high.append(d_mid_high)
            row += f" {fmt_signed(d_low_mid):>12} {fmt_signed(d_mid_high):>12}"
        row += f" {fmt_signed(average_domain_values(low_to_mid)):>12}"
        row += f" {fmt_signed(average_domain_values(mid_to_high)):>12}"
        print(row)
    print("-" * 110)


def print_08_low_to_high_trend(all_data: dict) -> None:
    retrievers = get_all_retrievers_from_08(all_data)

    print("\n" + "=" * 86)
    print("08 clean syntax-depth low-to-high trend")
    print("低到高差值基于原始分组 hit@10：深-低；mean 为三域差值均值，保留正负号")
    print("=" * 86)
    print(
        f"{'Retriever':<12} "
        f"{'Baby 深-低':>12} "
        f"{'Grocery 深-低':>14} "
        f"{'Pet 深-低':>12} "
        f"{'Mean 深-低':>12}"
    )
    print("-" * 86)

    for retriever in retrievers:
        row = f"{retriever:<12}"
        low_to_high = []
        for category in CATEGORIES:
            group_values = all_data[category][retriever]
            d_low_high = group_values["high_complexity"] - group_values["low_complexity"]
            low_to_high.append(d_low_high)
            row += f" {fmt_signed(d_low_high):>12}"
        row += f" {fmt_signed(average_domain_values(low_to_high)):>12}"
        print(row)
    print("-" * 86)


def print_09_correct_vs_noisy(all_data: dict) -> None:
    retrievers = get_all_retrievers_from_09(all_data)

    print("\n" + "=" * 132)
    print("09 syntax-depth correct vs noisy hit@10")
    print("09 数据只包含低/中复杂度；Mean Diff = Mean Noisy - Mean Corr")
    print("=" * 132)
    print(
        f"{'Retriever':<12} "
        f"{'Corr 低':>9} {'Noisy 低':>9} {'Diff 低':>9} "
        f"{'Corr 中':>9} {'Noisy 中':>9} {'Diff 中':>9} "
        f"{'Mean Corr':>10} {'Mean Noisy':>11} {'Mean Diff':>10}"
    )
    print("-" * 132)

    for retriever in retrievers:
        correct_group_values = {group: [] for group in SYNTAX_DEPTH_GROUP_ORDER_NOISY}
        noisy_group_values = {group: [] for group in SYNTAX_DEPTH_GROUP_ORDER_NOISY}
        for category in CATEGORIES:
            if retriever not in all_data[category]["correct"]:
                raise KeyError(f"{category} missing 09 correct result for {retriever}")
            if retriever not in all_data[category]["noisy"]:
                raise KeyError(f"{category} missing 09 noisy result for {retriever}")

            for group_name in SYNTAX_DEPTH_GROUP_ORDER_NOISY:
                correct_group_values[group_name].append(all_data[category]["correct"][retriever][group_name])
                noisy_group_values[group_name].append(all_data[category]["noisy"][retriever][group_name])

        group_summary = {}
        for group_name in SYNTAX_DEPTH_GROUP_ORDER_NOISY:
            correct_mean = average_domain_values(correct_group_values[group_name])
            noisy_mean = average_domain_values(noisy_group_values[group_name])
            group_summary[group_name] = {
                "correct": correct_mean,
                "noisy": noisy_mean,
                "diff": noisy_mean - correct_mean,
            }

        mean_correct = float(np.mean([group_summary[g]["correct"] for g in SYNTAX_DEPTH_GROUP_ORDER_NOISY]))
        mean_noisy = float(np.mean([group_summary[g]["noisy"] for g in SYNTAX_DEPTH_GROUP_ORDER_NOISY]))
        print(
            f"{retriever:<12} "
            f"{fmt4(group_summary['low_complexity']['correct']):>9} "
            f"{fmt4(group_summary['low_complexity']['noisy']):>9} "
            f"{fmt_signed(group_summary['low_complexity']['diff']):>9} "
            f"{fmt4(group_summary['medium_complexity']['correct']):>9} "
            f"{fmt4(group_summary['medium_complexity']['noisy']):>9} "
            f"{fmt_signed(group_summary['medium_complexity']['diff']):>9} "
            f"{fmt4(mean_correct):>10} {fmt4(mean_noisy):>11} {fmt_signed(mean_noisy - mean_correct):>10}"
        )
    print("-" * 132)


def load_09_paired_diffs(category: str) -> dict:
    results_path = BASE_DIR_09 / category / SYNTAX_DEPTH_NOISY_RESULTS_FILE
    data = load_json(results_path)
    correct_items = {}
    noisy_items = {}

    for source_name, target, expected_query_type in (
        ("correct_results", correct_items, "correct"),
        ("noisy_results", noisy_items, "noisy"),
    ):
        for item in data.get(source_name, []):
            retriever = item["retriever"]
            if retriever in DISABLED_RETRIEVERS:
                continue
            if item.get("query_category") != "syntax_depth":
                raise ValueError(f"{category} {source_name} {retriever} unexpected query_category")
            if item.get("query_type") != expected_query_type:
                raise ValueError(f"{category} {source_name} {retriever} unexpected query_type")
            if retriever in target:
                raise ValueError(f"{category} duplicate retriever in {source_name}: {retriever}")
            target[retriever] = item

    if set(correct_items) != set(noisy_items):
        raise KeyError(
            f"{category} 09 correct/noisy retriever sets differ: "
            f"correct={sorted(correct_items)}, noisy={sorted(noisy_items)}"
        )

    result = {}
    for retriever in sort_retrievers(correct_items):
        correct_records = {
            record["pair_id"]: record
            for record in correct_items[retriever].get("all_query_records", [])
        }
        noisy_records = {
            record["pair_id"]: record
            for record in noisy_items[retriever].get("all_query_records", [])
        }
        if set(correct_records) != set(noisy_records):
            raise KeyError(f"{category} {retriever} 09 pair_id sets differ")

        result[retriever] = {"all": [], "low_complexity": [], "medium_complexity": []}
        for pair_id in sorted(correct_records):
            correct_record = correct_records[pair_id]
            noisy_record = noisy_records[pair_id]
            group_name = correct_record.get("syntax_depth_group")
            if group_name != noisy_record.get("syntax_depth_group"):
                raise ValueError(f"{category} {retriever} pair_id={pair_id} syntax_depth_group mismatch")
            if group_name not in SYNTAX_DEPTH_GROUP_ORDER_NOISY:
                raise KeyError(f"{category} {retriever} pair_id={pair_id} unsupported group {group_name}")
            correct_hit10 = get_record_hit10(correct_record, f"{category} {retriever} correct pair {pair_id}")
            noisy_hit10 = get_record_hit10(noisy_record, f"{category} {retriever} noisy pair {pair_id}")
            diff = noisy_hit10 - correct_hit10
            result[retriever]["all"].append(diff)
            result[retriever][group_name].append(diff)
    return result


def build_09_paired_diffs() -> dict:
    return {category: load_09_paired_diffs(category) for category in CATEGORIES}


def bootstrap_mean_ci(values, n_boot: int = 2000, confidence: float = 0.95, seed: int = 42):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError(f"Bootstrap requires at least 2 values, got {values.size}")

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_means[idx] = float(np.mean(sample))

    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1.0 - alpha))


def print_09_paired_significance(all_data: dict) -> None:
    retrievers = sort_retrievers({r for cat_data in all_data.values() for r in cat_data})
    groups = [("all", "ALL"), ("low_complexity", "低"), ("medium_complexity", "中")]

    print("\n" + "=" * 118)
    print("09 paired significance: noisy hit@10 - correct hit@10")
    print("基于 pair_id 配对的逐查询 hit@10 差值")
    print("=" * 118)

    for group_name, group_label in groups:
        print(f"\n[{group_label}]")
        print(f"{'Retriever':<12} {'N_pairs':>10} {'Mean_Diff':>12} {'95% CI':>22} {'p-value':>12} {'Sig':>6}")
        print("-" * 92)
        for retriever in retrievers:
            diffs = []
            for category in CATEGORIES:
                if retriever not in all_data[category]:
                    raise KeyError(f"{category} missing paired diffs for {retriever}")
                if group_name not in all_data[category][retriever]:
                    raise KeyError(f"{category} {retriever} missing paired group {group_name}")
                diffs.extend(all_data[category][retriever][group_name])
            if len(diffs) < 2:
                raise ValueError(f"{retriever} {group_name} has insufficient paired diffs: {len(diffs)}")

            diffs_array = np.asarray(diffs, dtype=float)
            mean_diff = float(np.mean(diffs_array))
            ci_lower, ci_upper = bootstrap_mean_ci(diffs_array)
            t_result = stats.ttest_1samp(diffs_array, popmean=0.0, nan_policy="raise")
            p_value = float(t_result.pvalue)
            print(
                f"{retriever:<12} {len(diffs):>10} {fmt_signed(mean_diff):>12} "
                f"{f'[{ci_lower:+.4f}, {ci_upper:+.4f}]':>22} "
                f"{format_p_value(p_value):>12} {significance_star(p_value):>6}"
            )
        print("-" * 92)


def main():
    print("Loading 08 syntax-depth grouped hit@10...")
    data_08 = build_08_results()
    for category in CATEGORIES:
        print(f"  {category}: {len(data_08[category])} retrievers")

    print("\nLoading 09 syntax-depth correct/noisy grouped hit@10...")
    data_09 = build_09_results()
    for category in CATEGORIES:
        print(
            f"  {category}: correct={len(data_09[category]['correct'])}, "
            f"noisy={len(data_09[category]['noisy'])}"
        )

    paired_09 = build_09_paired_diffs()

    print_08_hit10_table(data_08)
    print_08_low_to_high_trend(data_08)
    print_08_adjacent_trend(data_08)
    print_09_correct_vs_noisy(data_09)
    print_09_paired_significance(paired_09)

    print("\n" + "=" * 100)
    print("Comparison Complete")
    print("=" * 100)


if __name__ == "__main__":
    main()
