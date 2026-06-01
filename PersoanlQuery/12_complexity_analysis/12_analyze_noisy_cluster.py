#!/usr/bin/env python3
"""分析 noisy query 在不同 cluster 的检索下降情况"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu")


def load_cluster_mapping(category: str) -> dict[str, int]:
    """加载 user_id -> cluster_index 映射"""
    mapping_file = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / category / "strict5550_query_gmm_user_profiles.jsonl"
    user_to_cluster = {}
    with open(mapping_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            user_to_cluster[row["user_id"]] = row["cluster_index"]
    return user_to_cluster


def analyze_category(category: str) -> dict:
    """分析单个类别的 cluster 下降情况"""
    # 加载 noisy 评估结果
    noisy_file = REPO_ROOT / "result" / "personal_query" / "09_noisy_retrieval" / category / "syntax_depth_correct_vs_noisy_results.json"
    with open(noisy_file, encoding="utf-8") as f:
        noisy_data = json.load(f)

    # 加载 cluster 映射
    cluster_mapping = load_cluster_mapping(category)

    # 提取 correct 和 noisy 结果
    correct_rows = noisy_data.get("raw_correct_results", [])
    noisy_rows = noisy_data.get("raw_noisy_results", [])

    results = {}
    for row in correct_rows:
        retriever = row["retriever"]
        correct_records = row.get("all_query_records", [])
        # 找对应的 noisy records
        noisy_record_list = next((r for r in noisy_rows if r["retriever"] == retriever), None)
        if not noisy_record_list:
            continue
        noisy_records = noisy_record_list.get("all_query_records", [])

        # 按 cluster 分组
        cluster_correct_h10 = defaultdict(list)
        cluster_noisy_h10 = defaultdict(list)
        cluster_count = defaultdict(int)

        for crec, nrec in zip(correct_records, noisy_records):
            user_id = crec.get("user_id")
            cluster_idx = cluster_mapping.get(user_id)
            if cluster_idx is None:
                continue

            c_metrics = crec.get("metrics", {})
            n_metrics = nrec.get("metrics", {})
            c_h10 = c_metrics.get("H@10", 0)
            n_h10 = n_metrics.get("H@10", 0)

            cluster_correct_h10[cluster_idx].append(c_h10)
            cluster_noisy_h10[cluster_idx].append(n_h10)
            cluster_count[cluster_idx] += 1

        # 计算每个 cluster 的均值和下降
        cluster_stats = {}
        for cid in sorted(set(cluster_correct_h10.keys())):
            c_vals = cluster_correct_h10[cid]
            n_vals = cluster_noisy_h10[cid]
            c_mean = sum(c_vals) / len(c_vals) if c_vals else 0
            n_mean = sum(n_vals) / len(n_vals) if n_vals else 0
            diff = n_mean - c_mean
            cluster_stats[f"cluster_{cid}"] = {
                "N": len(c_vals),
                "correct_H@10": c_mean,
                "noisy_H@10": n_mean,
                "diff": diff,
                "diff_pct": diff * 100,
            }

        # 找最重和最轻下降（只考虑下降的 cluster，即 diff < 0）
        diffs = {k: v["diff_pct"] for k, v in cluster_stats.items() if v["diff_pct"] < 0}
        if not diffs:
            diffs = {k: v["diff_pct"] for k, v in cluster_stats.items()}
        worst = min(diffs, key=diffs.get)
        lightest = max(diffs, key=diffs.get)
        gap = diffs[worst] - diffs[lightest]  # worst - lightest，gap 为负

        results[retriever] = {
            "cluster_stats": cluster_stats,
            "worst_cluster": worst,
            "worst_delta": diffs[worst],
            "lightest_cluster": lightest,
            "lightest_delta": diffs[lightest],
            "gap": gap,
            "total_noisy_samples": sum(cluster_count.values()),
        }

    return results


def main():
    categories = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]

    print("=" * 80)
    print("Noisy Query Cluster Analysis")
    print("=" * 80)

    all_results = {}
    for cat in categories:
        print(f"\n处理 {cat}...")
        try:
            results = analyze_category(cat)
            all_results[cat] = results
            print(f"  完成，共 {len(results)} 个检索器")
        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 打印汇总表
    print("\n" + "=" * 80)
    print("各检索器 Cluster 下降汇总")
    print("=" * 80)

    for cat, retriever_results in all_results.items():
        print(f"\n=== {cat} ===")
        print(f"{'Retriever':<12} {'N':>6} {'Worst':>8} {'Best':>8} {'Gap':>8} | {'Worst C':>10} {'Best C':>10}")
        print("-" * 80)
        for retriever, data in sorted(retriever_results.items()):
            n = data["total_noisy_samples"]
            worst = f"{data['worst_delta']:+.2f}%"
            lightest = f"{data['lightest_delta']:+.2f}%"
            gap = f"{data['gap']:.2f}%"
            wc = f"{data['worst_cluster']}"
            bc = f"{data['lightest_cluster']}"
            print(f"{retriever:<12} {n:>6} {worst:>8} {lightest:>8} {gap:>8} | {wc:<10} {bc:<10}")

    # 打印详细 per-cluster 数据
    print("\n" + "=" * 80)
    print("详细 Cluster 下降 (H@10_diff %)")
    print("=" * 80)

    for cat, retriever_results in all_results.items():
        print(f"\n=== {cat} ===")
        # 获取所有 cluster ID
        if not retriever_results:
            continue
        sample = next(iter(retriever_results.values()))
        cluster_ids = sorted(sample["cluster_stats"].keys(), key=lambda x: int(x.split("_")[1]))

        # 第一行表头：Retriever + cluster名称
        header1 = f"{'Retriever':<12}"
        for cid in cluster_ids:
            header1 += f" {cid:>10}"
        header1 += f" {'Gap':>8}"
        # 第二行表头：N
        header2 = f"{'':12}"
        for cid in cluster_ids:
            n = sample["cluster_stats"][cid]["N"]
            header2 += f" {'N='+str(n):>10}"
        header2 += f" {'':>8}"
        print(header1)
        print(header2)
        print("-" * len(header1))

        for retriever, data in sorted(retriever_results.items()):
            row = f"{retriever:<12}"
            for cid in cluster_ids:
                diff = data["cluster_stats"][cid]["diff_pct"]
                row += f" {diff:>+10.2f}%"
            row += f" {data['gap']:>+8.2f}%"
            print(row)

    # 汇总：各检索器在三个域的平均 Gap
    print("\n" + "=" * 80)
    print("各检索器跨域平均 Gap")
    print("=" * 80)
    retrievers = set()
    for cat_results in all_results.values():
        retrievers.update(cat_results.keys())
    retrievers = sorted(retrievers)

    print(f"{'Retriever':<12} {'Baby':>10} {'Grocery':>10} {'Pet':>10} {'Avg Gap':>10}")
    print("-" * 55)
    for retriever in retrievers:
        gaps = []
        for cat in categories:
            if retriever in all_results[cat]:
                gaps.append(all_results[cat][retriever]["gap"])
            else:
                gaps.append(None)
        gap_strs = []
        for g in gaps:
            if g is not None:
                gap_strs.append(f"{g:>10.2f}%")
            else:
                gap_strs.append(f"{'N/A':>10}")
        valid_gaps = [g for g in gaps if g is not None]
        avg_gap = sum(valid_gaps) / len(valid_gaps) if valid_gaps else 0
        print(f"{retriever:<12} {''.join(gap_strs)} {avg_gap:>+10.2f}%")

    # 保存结果
    output_file = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis" / "noisy_cluster_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
