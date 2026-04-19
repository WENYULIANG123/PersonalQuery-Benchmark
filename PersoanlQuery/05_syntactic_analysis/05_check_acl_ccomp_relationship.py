#!/usr/bin/env python3
"""
检查 ACL 和 CCOMP 等级关系
==========================
分析每个用户内 ACL 和 CCOMP 等级之间的关联性。

例如：
- ACL0 用户的 CCOMP 等级分布是怎样的？
- ACL 等级和 CCOMP 等级是否存在负相关？
- 是否存在 ACL0 但 CCOMP3 的用户？
"""

import json
import os
from collections import defaultdict
from datetime import datetime

# ========================================
# 硬编码参数
# ========================================
CATEGORY = "Pet_Supplies"  # 可选: Arts_Crafts_and_Sewing, Grocery_and_Gourmet_Food, Pet_Supplies

# Stage 6 查询文件
ACL_QUERY_FILE = f'/fs04/ar57/wenyu/result/personal_query/06_query/{CATEGORY}/acl_query.json'
CCOMP_QUERY_FILE = f'/fs04/ar57/wenyu/result/personal_query/06_query/{CATEGORY}/ccomp_query.json'


# ========================================
# 日志
# ========================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========================================
# 主函数
# ========================================
def main():
    log(f"=== 检查 ACL 和 CCOMP 等级关系 (Category: {CATEGORY}) ===")

    # 加载 ACL 查询
    log(f"加载 ACL 查询 from {ACL_QUERY_FILE}...")
    with open(ACL_QUERY_FILE, 'r', encoding='utf-8') as f:
        acl_data = json.load(f)

    # 加载 CCOMP 查询
    log(f"加载 CCOMP 查询 from {CCOMP_QUERY_FILE}...")
    with open(CCOMP_QUERY_FILE, 'r', encoding='utf-8') as f:
        ccomp_data = json.load(f)

    log(f"ACL 查询用户数: {len(acl_data)}")
    log(f"CCOMP 查询用户数: {len(ccomp_data)}")

    # 构建用户等级映射
    acl_ground_truth = {}  # uid -> ground_truth_acl
    ccomp_ground_truth = {}  # uid -> ground_truth_ccomp

    for user in acl_data:
        uid = user['user_id']
        acl_ground_truth[uid] = user.get('ground_truth_acl', None)

    for user in ccomp_data:
        uid = user['user_id']
        ccomp_ground_truth[uid] = user.get('ground_truth_ccomp', None)

    # 找到同时有 ACL 和 CCOMP 数据的用户
    common_users = set(acl_ground_truth.keys()) & set(ccomp_ground_truth.keys())
    log(f"同时有 ACL 和 CCOMP 数据的用户数: {len(common_users)}")

    # ACL 分布统计
    acl_dist = defaultdict(int)
    for uid, level in acl_ground_truth.items():
        if level is not None:
            acl_dist[level] += 1
    log(f"\nACL 等级分布: {dict(acl_dist)}")

    # CCOMP 分布统计
    ccomp_dist = defaultdict(int)
    for uid, level in ccomp_ground_truth.items():
        if level is not None:
            ccomp_dist[level] += 1
    log(f"CCOMP 等级分布: {dict(ccomp_dist)}")

    # 过滤后 ACL > 3 的用户数
    acl_gt3 = sum(1 for level in acl_ground_truth.values() if level is not None and level > 3)
    log(f"\nACL 等级 > 3 的用户数: {acl_gt3}")

    # 过滤后 CCOMP > 3 的用户数
    ccomp_gt3 = sum(1 for level in ccomp_ground_truth.values() if level is not None and level > 3)
    log(f"CCOMP 等级 > 3 的用户数: {ccomp_gt3}")

    # ========================================
    # 分析 ACL 等级和 CCOMP 等级的交叉关系
    # ========================================
    log("\n" + "=" * 60)
    log("ACL 等级 vs CCOMP 等级 交叉表")
    log("=" * 60)

    # 创建交叉表
    cross_table = defaultdict(lambda: defaultdict(int))
    acl_level_counts = defaultdict(int)
    ccomp_level_counts = defaultdict(int)

    for uid in common_users:
        acl_level = acl_ground_truth[uid]
        ccomp_level = ccomp_ground_truth[uid]

        if acl_level is not None and ccomp_level is not None:
            cross_table[acl_level][ccomp_level] += 1
            acl_level_counts[acl_level] += 1
            ccomp_level_counts[ccomp_level] += 1

    # 打印交叉表
    header = "ACL\\CCOMP\t" + "\t".join([f"CCOMP{l}" for l in range(4)]) + "\t合计"
    log(header)
    log("-" * len(header))

    for acl_level in range(4):
        row = f"ACL{acl_level}\t\t"
        row_values = []
        for ccomp_level in range(4):
            row_values.append(str(cross_table[acl_level][ccomp_level]))
        row += "\t".join(row_values) + f"\t\t{acl_level_counts[acl_level]}"
        log(row)

    log("-" * len(header))
    footer = "合计\t\t"
    footer_values = []
    for ccomp_level in range(4):
        footer_values.append(str(sum(cross_table[acl][ccomp_level] for acl in range(4))))
    footer += "\t".join(footer_values) + f"\t\t{len(common_users)}"
    log(footer)

    # ========================================
    # 分析具体关系
    # ========================================
    log("\n" + "=" * 60)
    log("等级关系分析")
    log("=" * 60)

    # 检查是否 ACL 低等级用户的 CCOMP 等级偏高
    for acl_level in range(4):
        total_for_acl = acl_level_counts[acl_level]
        if total_for_acl == 0:
            continue

        log(f"\nACL{acl_level} 用户 (共 {total_for_acl} 人) 的 CCOMP 等级分布:")
        for ccomp_level in range(4):
            count = cross_table[acl_level][ccomp_level]
            pct = count / total_for_acl * 100
            log(f"  CCOMP{ccomp_level}: {count} 人 ({pct:.1f}%)")

    # 检查特殊情况：ACL0 但 CCOMP3
    log("\n" + "=" * 60)
    log("特殊情况检查")
    log("=" * 60)

    acl0_ccomp3_users = []
    acl3_ccomp0_users = []

    for uid in common_users:
        acl_level = acl_ground_truth[uid]
        ccomp_level = ccomp_ground_truth[uid]

        if acl_level == 0 and ccomp_level == 3:
            acl0_ccomp3_users.append(uid)
        if acl_level == 3 and ccomp_level == 0:
            acl3_ccomp0_users.append(uid)

    log(f"\nACL0 但 CCOMP3 的用户数: {len(acl0_ccomp3_users)}")
    if acl0_ccomp3_users:
        log(f"  用户ID示例: {acl0_ccomp3_users[:5]}")

    log(f"\nACL3 但 CCOMP0 的用户数: {len(acl3_ccomp0_users)}")
    if acl3_ccomp0_users:
        log(f"  用户ID示例: {acl3_ccomp0_users[:5]}")

    # ========================================
    # 计算相关性
    # ========================================
    log("\n" + "=" * 60)
    log("相关性分析")
    log("=" * 60)

    # 计算皮尔逊相关系数
    n = len(common_users)
    acl_levels = []
    ccomp_levels = []

    for uid in common_users:
        acl_level = acl_ground_truth[uid]
        ccomp_level = ccomp_ground_truth[uid]
        if acl_level is not None and ccomp_level is not None:
            acl_levels.append(acl_level)
            ccomp_levels.append(ccomp_level)

    mean_acl = sum(acl_levels) / len(acl_levels)
    mean_ccomp = sum(ccomp_levels) / len(ccomp_levels)

    numerator = sum((acl_levels[i] - mean_acl) * (ccomp_levels[i] - mean_ccomp) for i in range(len(acl_levels)))
    denom_acl = sum((x - mean_acl) ** 2 for x in acl_levels) ** 0.5
    denom_ccomp = sum((x - mean_ccomp) ** 2 for x in ccomp_levels) ** 0.5

    if denom_acl > 0 and denom_ccomp > 0:
        correlation = numerator / (denom_acl * denom_ccomp)
        log(f"皮尔逊相关系数 (ACL vs CCOMP): {correlation:.4f}")
    else:
        log("无法计算相关系数")

    # ACL 和 CCOMP 等级的平均值
    log(f"\nACL 等级平均值: {mean_acl:.2f}")
    log(f"CCOMP 等级平均值: {mean_ccomp:.2f}")

    # ========================================
    # 总结
    # ========================================
    log("\n" + "=" * 60)
    log("总结")
    log("=" * 60)

    # 检查是否 ACL 和 CCOMP 等级存在负相关
    if correlation < -0.3:
        log("ACL 和 CCOMP 等级呈现负相关（用户倾向于一种类型强，另一种类型弱）")
    elif correlation > 0.3:
        log("ACL 和 CCOMP 等级呈现正相关（用户两种类型都强或都弱）")
    else:
        log("ACL 和 CCOMP 等级相关性不强（两种类型独立）")

    # 检查是否有 ACL0 但 CCOMP3 的极端情况
    if len(acl0_ccomp3_users) > 0:
        log(f"警告: 发现 {len(acl0_ccomp3_users)} 个用户 ACL0 但 CCOMP3（极端差异）")

    log("\n分析完成!")


if __name__ == '__main__':
    main()
