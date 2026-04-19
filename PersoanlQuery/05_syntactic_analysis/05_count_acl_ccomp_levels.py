#!/usr/bin/env python3
"""
统计 ACL 和 CCOMP 各等级（0, 1, 2, 3）的用户数量
==========================================

分析三个类别领域中：
- ACL0, ACL1, ACL2, ACL3 各有多少用户
- CCOMP0, CCOMP1, CCOMP2, CCOMP3 各有多少用户

输入：Stage 5 句法分析输出文件
- acl_user_profiles.json    -> words_per_acl
- ccomp_user_profiles.json  -> words_per_ccomp
- attr_density_user_profiles.json -> words_per_attribute

等级计算公式：
- target_length = ceil(words_per_attribute) * 5
- words_per_acl = total_token_count / acl_count (仅 acl，不含 relcl)
- ground_truth_acl = int(target_length / words_per_acl)
- ground_truth_acl = max(0, min(3, ground_truth_acl))
- ground_truth_ccomp = int(target_length / words_per_ccomp)
- ground_truth_ccomp = max(0, min(3, ground_truth_ccomp))

注意：words_per_acl 仅统计 acl（形容词性从句），不包含 relcl（关系从句）
"""

import json
import os
from collections import defaultdict
from datetime import datetime

# ========================================
# 配置
# ========================================
CATEGORIES = [
    "Arts_Crafts_and_Sewing",
    "Grocery_and_Gourmet_Food",
    "Pet_Supplies"
]

BASE_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis"

# ========================================
# 日志
# ========================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========================================
# 主函数
# ========================================
def main():
    log("=" * 70)
    log("ACL / CCOMP 各等级用户数量统计（使用 Stage 5 输出文件）")
    log("=" * 70)

    category_stats = {}

    for category in CATEGORIES:
        log(f"\n{'=' * 50}")
        log(f"类别: {category}")
        log(f"{'=' * 50}")

        acl_file = f"{BASE_DIR}/{category}/acl_user_profiles.json"
        ccomp_file = f"{BASE_DIR}/{category}/ccomp_user_profiles.json"
        attr_file = f"{BASE_DIR}/{category}/attr_density_user_profiles.json"

        # 加载 ACL 数据
        log(f"加载 ACL 用户画像: {acl_file}")
        with open(acl_file, 'r', encoding='utf-8') as f:
            acl_users = json.load(f)

        # 加载 CCOMP 数据
        log(f"加载 CCOMP 用户画像: {ccomp_file}")
        with open(ccomp_file, 'r', encoding='utf-8') as f:
            ccomp_users = json.load(f)

        # 加载 AttrDensity 数据（用于 words_per_attribute）
        log(f"加载 AttrDensity 用户画像: {attr_file}")
        with open(attr_file, 'r', encoding='utf-8') as f:
            attr_users = json.load(f)

        # 构建 user_id -> words_per_attribute 映射
        user_wpa_map = {}
        for user in attr_users:
            uid = user.get('user_id')
            wpa = user.get('words_per_attribute')
            if uid and wpa:
                user_wpa_map[uid] = wpa

        # 构建 user_id -> acl_profile 映射
        user_acl_map = {}
        for user in acl_users:
            uid = user.get('user_id')
            if uid:
                user_acl_map[uid] = user

        # 构建 user_id -> ccomp_profile 映射
        user_ccomp_map = {}
        for user in ccomp_users:
            uid = user.get('user_id')
            if uid:
                user_ccomp_map[uid] = user

        # 计算每个用户的等级
        acl_level_counts = defaultdict(int)
        ccomp_level_counts = defaultdict(int)
        user_levels = []  # 保存每个用户的等级

        valid_user_count = 0
        skip_count = 0

        for uid, acl_profile in user_acl_map.items():
            # 获取 words_per_attribute
            words_per_attribute = user_wpa_map.get(uid)
            if words_per_attribute is None:
                skip_count += 1
                continue

            # 获取句子数
            total_sentences = acl_profile.get('total_sentences', 1)
            if total_sentences <= 0:
                total_sentences = 1

            # ACL: level = int((acl + relcl) / 句子数)
            acl_type_dist = acl_profile.get('acl_type_distribution', {})
            acl_count = acl_type_dist.get('acl', 0) + acl_type_dist.get('relcl_reference', 0)
            ground_truth_acl = max(0, min(3, int(acl_count / total_sentences)))

            # CCOMP: level = int(ccomp / 句子数)
            ccomp_profile = user_ccomp_map.get(uid, {})
            ccomp_type_dist = ccomp_profile.get('ccomp_type_distribution', {})
            ccomp_count = ccomp_type_dist.get('ccomp', 0)
            ground_truth_ccomp = max(0, min(3, int(ccomp_count / total_sentences)))

            acl_level_counts[ground_truth_acl] += 1
            ccomp_level_counts[ground_truth_ccomp] += 1
            valid_user_count += 1

            # 保存用户等级
            user_levels.append({
                'user_id': uid,
                'acl_level': ground_truth_acl,
                'ccomp_level': ground_truth_ccomp
            })

        if skip_count > 0:
            log(f"  跳过 {skip_count} 个用户（缺少 words_per_attribute 数据）")

        # 打印 ACL 统计
        log(f"\n  ACL 等级分布（有效用户: {valid_user_count}）:")
        log(f"  {'等级':<10} {'用户数':<10} {'占比':<10}")
        log(f"  {'-' * 30}")
        total_acl_users = sum(acl_level_counts.values())
        for level in range(4):
            count = acl_level_counts.get(level, 0)
            pct = count / total_acl_users * 100 if total_acl_users > 0 else 0
            log(f"  ACL{level:<8} {count:<10} {pct:.1f}%")

        # 打印 CCOMP 统计
        log(f"\n  CCOMP 等级分布:")
        log(f"  {'等级':<10} {'用户数':<10} {'占比':<10}")
        log(f"  {'-' * 30}")
        total_ccomp_users = sum(ccomp_level_counts.values())
        for level in range(4):
            count = ccomp_level_counts.get(level, 0)
            pct = count / total_ccomp_users * 100 if total_ccomp_users > 0 else 0
            log(f"  CCOMP{level:<6} {count:<10} {pct:.1f}%")

        # 保存用户等级到 JSON 文件
        output_file = f"{BASE_DIR}/{category}/level.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(user_levels, f, indent=2, ensure_ascii=False)
        log(f"保存 {category} 用户等级到: {output_file}")

        # 保存该类别的统计数据
        category_stats[category] = {
            'acl_level_counts': dict(acl_level_counts),
            'ccomp_level_counts': dict(ccomp_level_counts),
            'total_acl_users': total_acl_users,
            'total_ccomp_users': total_ccomp_users,
            'skip_count': skip_count
        }

    # ========================================
    # 汇总统计
    # ========================================
    log("\n" + "=" * 70)
    log("汇总统计（三个类别）")
    log("=" * 70)

    total_acl_by_level = defaultdict(int)
    total_ccomp_by_level = defaultdict(int)
    grand_total_acl = 0
    grand_total_ccomp = 0

    for category, stats in category_stats.items():
        for level, count in stats['acl_level_counts'].items():
            total_acl_by_level[level] += count
            grand_total_acl += count
        for level, count in stats['ccomp_level_counts'].items():
            total_ccomp_by_level[level] += count
            grand_total_ccomp += count

    log(f"\n  ACL 等级汇总（总用户: {grand_total_acl}）:")
    log(f"  {'等级':<10} {'用户数':<10} {'占比':<10}")
    log(f"  {'-' * 30}")
    for level in range(4):
        count = total_acl_by_level.get(level, 0)
        pct = count / grand_total_acl * 100 if grand_total_acl > 0 else 0
        log(f"  ACL{level:<8} {count:<10} {pct:.1f}%")

    log(f"\n  CCOMP 等级汇总（总用户: {grand_total_ccomp}）:")
    log(f"  {'等级':<10} {'用户数':<10} {'占比':<10}")
    log(f"  {'-' * 30}")
    for level in range(4):
        count = total_ccomp_by_level.get(level, 0)
        pct = count / grand_total_ccomp * 100 if grand_total_ccomp > 0 else 0
        log(f"  CCOMP{level:<6} {count:<10} {pct:.1f}%")

    # 保存汇总到 level_distribution 目录
    output_dir = f"{BASE_DIR}/level_distribution"
    os.makedirs(output_dir, exist_ok=True)
    summary_file = f"{output_dir}/summary_level_distribution.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_acl_by_level': dict(total_acl_by_level),
            'total_ccomp_by_level': dict(total_ccomp_by_level),
            'grand_total_acl': grand_total_acl,
            'grand_total_ccomp': grand_total_ccomp,
            'categories': list(category_stats.keys())
        }, f, indent=2, ensure_ascii=False)
    log(f"保存汇总到: {summary_file}")

    log("\n" + "=" * 70)
    log("统计完成！")
    log("=" * 70)


if __name__ == '__main__':
    main()
