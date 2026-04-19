#!/usr/bin/env python3
"""
检查 ACL 和 CCOMP 等级关系
==========================
分析每个用户内 ACL 和 CCOMP 等级之间的关联性。

输入：Stage 5 的用户画像文件
- acl_user_profiles.json
- ccomp_user_profiles.json
- attr_density_user_profiles.json
"""

import json
import os
import math
from collections import defaultdict
from datetime import datetime

# ========================================
# 硬编码参数
# ========================================
CATEGORY = "Grocery_and_Gourmet_Food"

# Stage 5 用户画像文件
ACL_PROFILES_FILE = f'/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/{CATEGORY}/acl_user_profiles.json'
CCOMP_PROFILES_FILE = f'/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/{CATEGORY}/ccomp_user_profiles.json'
ATTR_DENSITY_PROFILES_FILE = f'/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/{CATEGORY}/attr_density_user_profiles.json'


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

    # 加载 ACL 用户画像
    log(f"加载 ACL 用户画像 from {ACL_PROFILES_FILE}...")
    with open(ACL_PROFILES_FILE, 'r', encoding='utf-8') as f:
        acl_profiles = json.load(f)

    # 加载 CCOMP 用户画像
    log(f"加载 CCOMP 用户画像 from {CCOMP_PROFILES_FILE}...")
    with open(CCOMP_PROFILES_FILE, 'r', encoding='utf-8') as f:
        ccomp_profiles = json.load(f)

    # 加载 Attr Density 用户画像
    log(f"加载 Attr Density 用户画像 from {ATTR_DENSITY_PROFILES_FILE}...")
    with open(ATTR_DENSITY_PROFILES_FILE, 'r', encoding='utf-8') as f:
        attr_density_profiles = json.load(f)

    log(f"ACL 用户画像数: {len(acl_profiles)}")
    log(f"CCOMP 用户画像数: {len(ccomp_profiles)}")
    log(f"Attr Density 用户画像数: {len(attr_density_profiles)}")

    # 构建用户画像映射
    acl_profile_map = {p['user_id']: p for p in acl_profiles}
    ccomp_profile_map = {p['user_id']: p for p in ccomp_profiles}
    attr_density_map = {p['user_id']: p for p in attr_density_profiles}

    # 找到同时存在于三个数据集的用户
    all_user_ids = set(acl_profile_map.keys()) & set(ccomp_profile_map.keys()) & set(attr_density_map.keys())
    log(f"同时存在于 ACL、CCOMP、AttrDensity 的用户数: {len(all_user_ids)}")

    # 计算每个用户的 ground_truth 等级
    acl_ground_truth = {}  # uid -> ground_truth_acl
    ccomp_ground_truth = {}  # uid -> ground_truth_ccomp

    for uid in all_user_ids:
        acl_profile = acl_profile_map[uid]
        ccomp_profile = ccomp_profile_map[uid]
        attr_profile = attr_density_map[uid]

        words_per_attribute = attr_profile.get('words_per_attribute') or 10.0
        words_per_acl = acl_profile.get('words_per_acl') or 100.0
        words_per_ccomp = ccomp_profile.get('words_per_ccomp') or 100.0

        target_length = math.ceil(words_per_attribute) * 5

        # 计算 ACL ground_truth
        if words_per_acl and words_per_acl > 0:
            ground_truth_acl = int(target_length / words_per_acl)
            ground_truth_acl = max(0, min(5, ground_truth_acl))
        else:
            ground_truth_acl = 0

        # 计算 CCOMP ground_truth
        if words_per_ccomp and words_per_ccomp > 0:
            ground_truth_ccomp = int(target_length / words_per_ccomp)
            ground_truth_ccomp = max(0, min(5, ground_truth_ccomp))
        else:
            ground_truth_ccomp = 0

        acl_ground_truth[uid] = ground_truth_acl
        ccomp_ground_truth[uid] = ground_truth_ccomp

    log(f"已计算 {len(acl_ground_truth)} 个用户的 ground_truth 等级")

    # ========================================
    # 等级分布统计
    # ========================================
    acl_dist = defaultdict(int)
    ccomp_dist = defaultdict(int)

    for uid in all_user_ids:
        acl_dist[acl_ground_truth[uid]] += 1
        ccomp_dist[ccomp_ground_truth[uid]] += 1

    log(f"\nACL 等级分布: {dict(sorted(acl_dist.items()))}")
    log(f"CCOMP 等级分布: {dict(sorted(ccomp_dist.items()))}")

    # ACL > 3 和 CCOMP > 3 的用户数
    acl_gt3 = sum(1 for level in acl_ground_truth.values() if level > 3)
    ccomp_gt3 = sum(1 for level in ccomp_ground_truth.values() if level > 3)
    log(f"\nACL 等级 > 3 的用户数: {acl_gt3}")
    log(f"CCOMP 等级 > 3 的用户数: {ccomp_gt3}")

    # ========================================
    # ACL 等级 vs CCOMP 等级 交叉表
    # ========================================
    log("\n" + "=" * 60)
    log("ACL 等级 vs CCOMP 等级 交叉表")
    log("=" * 60)

    cross_table = defaultdict(lambda: defaultdict(int))

    for uid in all_user_ids:
        acl_level = acl_ground_truth[uid]
        ccomp_level = ccomp_ground_truth[uid]
        cross_table[acl_level][ccomp_level] += 1

    header = "ACL\\CCOMP\t" + "\t".join([f"CCOMP{l}" for l in range(6)]) + "\t合计"
    log(header)
    log("-" * len(header))

    for acl_level in range(6):
        row = f"ACL{acl_level}\t\t"
        row_values = []
        for ccomp_level in range(6):
            row_values.append(str(cross_table[acl_level][ccomp_level]))
        row += "\t".join(row_values) + f"\t\t{sum(cross_table[acl_level].values())}"
        log(row)

    log("-" * len(header))
    footer = "合计\t\t"
    footer_values = []
    for ccomp_level in range(6):
        footer_values.append(str(sum(cross_table[acl][ccomp_level] for acl in range(6))))
    footer += "\t".join(footer_values) + f"\t\t{len(all_user_ids)}"
    log(footer)

    # ========================================
    # 分析具体关系
    # ========================================
    log("\n" + "=" * 60)
    log("等级关系分析")
    log("=" * 60)

    for acl_level in range(6):
        total_for_acl = sum(cross_table[acl_level].values())
        if total_for_acl == 0:
            continue

        log(f"\nACL{acl_level} 用户 (共 {total_for_acl} 人) 的 CCOMP 等级分布:")
        for ccomp_level in range(6):
            count = cross_table[acl_level][ccomp_level]
            if count > 0:
                pct = count / total_for_acl * 100
                log(f"  CCOMP{ccomp_level}: {count} 人 ({pct:.1f}%)")

    # ========================================
    # 特殊情况检查
    # ========================================
    log("\n" + "=" * 60)
    log("特殊情况检查")
    log("=" * 60)

    acl0_ccomp3_users = [uid for uid in all_user_ids if acl_ground_truth[uid] == 0 and ccomp_ground_truth[uid] == 3]
    acl3_ccomp0_users = [uid for uid in all_user_ids if acl_ground_truth[uid] == 3 and ccomp_ground_truth[uid] == 0]

    log(f"\nACL0 但 CCOMP3 的用户数: {len(acl0_ccomp3_users)}")
    if acl0_ccomp3_users:
        log(f"  用户ID示例: {acl0_ccomp3_users[:5]}")

    log(f"\nACL3 但 CCOMP0 的用户数: {len(acl3_ccomp0_users)}")
    if acl3_ccomp0_users:
        log(f"  用户ID示例: {acl3_ccomp0_users[:5]}")

    # ========================================
    # 相关性分析
    # ========================================
    log("\n" + "=" * 60)
    log("相关性分析")
    log("=" * 60)

    acl_levels = [acl_ground_truth[uid] for uid in all_user_ids]
    ccomp_levels = [ccomp_ground_truth[uid] for uid in all_user_ids]

    mean_acl = sum(acl_levels) / len(acl_levels)
    mean_ccomp = sum(ccomp_levels) / len(ccomp_levels)

    numerator = sum((acl_levels[i] - mean_acl) * (ccomp_levels[i] - mean_ccomp) for i in range(len(acl_levels)))
    denom_acl = sum((x - mean_acl) ** 2 for x in acl_levels) ** 0.5
    denom_ccomp = sum((x - mean_ccomp) ** 2 for x in ccomp_levels) ** 0.5

    if denom_acl > 0 and denom_ccomp > 0:
        correlation = numerator / (denom_acl * denom_ccomp)
        log(f"皮尔逊相关系数 (ACL vs CCOMP): {correlation:.4f}")

    log(f"\nACL 等级平均值: {mean_acl:.2f}")
    log(f"CCOMP 等级平均值: {mean_ccomp:.2f}")

    # ========================================
    # 总结
    # ========================================
    log("\n" + "=" * 60)
    log("总结")
    log("=" * 60)

    if correlation < -0.3:
        log("ACL 和 CCOMP 等级呈现负相关")
    elif correlation > 0.3:
        log("ACL 和 CCOMP 等级呈现正相关")
    else:
        log("ACL 和 CCOMP 等级相关性不强（两种类型独立）")

    if len(acl0_ccomp3_users) > 0:
        log(f"警告: 发现 {len(acl0_ccomp3_users)} 个用户 ACL0 但 CCOMP3")

    log("\n分析完成!")


if __name__ == '__main__':
    main()
