#!/usr/bin/env python3
"""
计算回译前后句子的14维特征的L2距离
"""
import json
import os
import numpy as np
from datetime import datetime


def log_with_timestamp(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def compute_l2_distance(features1: dict, features2: dict) -> float:
    """计算两个14维特征向量的L2距离"""
    # 14维特征键（排除word_count和avg_sentence_length）
    feature_keys = [
        'subordinate_clause_freq', 'dep_distance', 'modifier_density',
        'coord_chain', 'negation_scope', 'voice_ratio', 'branching_direction',
        'advcl_freq', 'comp_clause_freq', 'fanout', 'parataxis_freq',
        'prep_density', 'appos_freq'
    ]

    vec1 = np.array([features1.get(k, 0.0) for k in feature_keys])
    vec2 = np.array([features2.get(k, 0.0) for k in feature_keys])

    return float(np.linalg.norm(vec1 - vec2))


def main() -> None:
    INPUT_FILE = "/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/user_sentences/all_users_merged.json"

    log_with_timestamp("=" * 80)
    log_with_timestamp("Compute L2 Distance: Original vs Back-translated")
    log_with_timestamp("=" * 80)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    users = data.get("users", [])
    log_with_timestamp(f"Loaded {len(users)} users")

    all_distances = []
    user_distances = {}

    for user in users:
        user_id = user["user_id"]
        sentences = user.get("sentences", [])

        user_dists = []
        for sent in sentences:
            if "backtrans_features_14d" not in sent:
                continue

            orig_features = sent.get("features_14d", {})
            back_features = sent.get("backtrans_features_14d", {})

            if not orig_features or not back_features:
                continue

            dist = compute_l2_distance(orig_features, back_features)
            all_distances.append(dist)
            user_dists.append(dist)

        if user_dists:
            user_distances[user_id] = {
                "mean": np.mean(user_dists),
                "std": np.std(user_dists),
                "min": np.min(user_dists),
                "max": np.max(user_dists),
                "count": len(user_dists),
            }

    log_with_timestamp("=" * 80)
    log_with_timestamp("Overall Statistics")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Total sentence pairs: {len(all_distances)}")
    log_with_timestamp(f"Mean L2 distance: {np.mean(all_distances):.6f}")
    log_with_timestamp(f"Std L2 distance: {np.std(all_distances):.6f}")
    log_with_timestamp(f"Min L2 distance: {np.min(all_distances):.6f}")
    log_with_timestamp(f"Max L2 distance: {np.max(all_distances):.6f}")
    log_with_timestamp(f"Median L2 distance: {np.median(all_distances):.6f}")

    # 按用户统计
    log_with_timestamp("=" * 80)
    log_with_timestamp("Per-User Statistics")
    log_with_timestamp("=" * 80)
    for user_id, stats in sorted(user_distances.items(), key=lambda x: x[1]["mean"]):
        log_with_timestamp(
            f"User {user_id}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
            f"min={stats['min']:.6f}, max={stats['max']:.6f}, n={stats['count']}"
        )

    # 保存结果
    output_file = INPUT_FILE.replace(".json", "_l2_stats.json")
    result = {
        "timestamp": datetime.now().isoformat(),
        "overall": {
            "total_pairs": len(all_distances),
            "mean": float(np.mean(all_distances)),
            "std": float(np.std(all_distances)),
            "min": float(np.min(all_distances)),
            "max": float(np.max(all_distances)),
            "median": float(np.median(all_distances)),
        },
        "per_user": user_distances,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Results saved to: {output_file}")
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    main()
