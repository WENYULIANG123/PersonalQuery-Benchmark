#!/usr/bin/env python3
"""
Stage 7: Batch Iterative Refinement for All Users

This script scans dual query files from Stage 6, validates users with available
linguistic profiles, and then runs Stage 7 iterative refinement in one batch.
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


def log_with_timestamp(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def load_stage7_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("stage7_iterative", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Stage 7 script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_users_with_dual_queries(query_dir: Path) -> List[str]:
    users = []
    for query_file in sorted(query_dir.glob("queries_*.json")):
        user_id = query_file.stem.replace("queries_", "")
        if user_id and user_id != "summary":
            users.append(user_id)
    return users


def load_sentence_level_users(sentence_level_dir: Path) -> Set[str]:
    combined_file = sentence_level_dir / "sentence_level_features_all_users.json"
    if combined_file.exists():
        with open(combined_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.keys()) if isinstance(data, dict) else set()

    users = set()
    for file_path in sentence_level_dir.glob("sentence_level_features_*.json"):
        user_id = file_path.stem.replace("sentence_level_features_", "")
        if user_id:
            users.add(user_id)
    return users


def validate_users(
    user_ids: List[str],
    linguistic_dir: Path,
    use_sentence_level: bool,
    sentence_level_dir: Path,
) -> Dict[str, str]:
    valid_users: Dict[str, str] = {}

    if use_sentence_level:
        available_users = load_sentence_level_users(sentence_level_dir)
        for user_id in user_ids:
            if user_id in available_users:
                valid_users[user_id] = "sentence_level"
            else:
                log_with_timestamp(f"  - Skip {user_id}: sentence-level profile not found")
        return valid_users

    for user_id in user_ids:
        profile_file = linguistic_dir / f"linguistic_profile_{user_id}.json"
        if profile_file.exists():
            valid_users[user_id] = str(profile_file)
        else:
            log_with_timestamp(f"  - Skip {user_id}: linguistic profile not found")

    return valid_users


def copy_user_query_files(user_ids: List[str], src_query_dir: Path, dst_query_dir: Path) -> int:
    copied = 0
    for user_id in user_ids:
        src_file = src_query_dir / f"queries_{user_id}.json"
        if not src_file.exists():
            log_with_timestamp(f"  - Missing query file for {user_id}: {src_file}")
            continue
        shutil.copy2(src_file, dst_query_dir / src_file.name)
        copied += 1
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 7 iterative refinement for all users in Stage 6 query directory."
    )
    parser.add_argument(
        "--query-dir",
        default="/home/wlia0047/ar57/wenyu/result/personal_query/06_query",
        help="Directory containing queries_{USER_ID}.json files",
    )
    parser.add_argument(
        "--linguistic-dir",
        default="/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis",
        help="Directory containing linguistic_profile_{USER_ID}.json files",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/wlia0047/ar57/wenyu/result/personal_query/07_iterative_refinement",
        help="Output directory for iterative refinement results",
    )
    parser.add_argument(
        "--user-ids",
        nargs="+",
        help="Optional user ID list. If omitted, process all users in --query-dir.",
    )
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--candidates-per-round", type=int, default=2)
    parser.add_argument(
        "--feature-set",
        type=str,
        default="style_only_16",
        choices=["emnlp_16", "short_query_18", "short_query_13", "style_only_16", "full"],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--query-delay", type=float, default=0.0)
    parser.add_argument("--style-strength", type=str, default="medium", choices=["weak", "medium", "strong"])
    parser.add_argument("--style-shot-k", type=int, default=5)
    parser.add_argument("--use-sentence-level", action="store_true")
    parser.add_argument(
        "--sentence-level-dir",
        type=str,
        default="/home/wlia0047/ar57/wenyu/result/user_profile/sentence_level_features",
        help="Directory containing sentence-level features",
    )
    parser.add_argument(
        "--keep-temp-dir",
        action="store_true",
        help="Keep temporary filtered query directory for debugging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).parent
    stage7_script = script_dir / "07_iterative_refinement.py"
    query_dir = Path(args.query_dir)
    linguistic_dir = Path(args.linguistic_dir)
    sentence_level_dir = Path(args.sentence_level_dir)
    output_dir = Path(args.output_dir)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 7: Batch Iterative Refinement for All Users")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Query dir: {query_dir}")
    log_with_timestamp(f"Linguistic dir: {linguistic_dir}")
    log_with_timestamp(f"Output dir: {output_dir}")

    if not query_dir.exists():
        log_with_timestamp(f"ERROR: query dir not found: {query_dir}")
        return 1
    if not stage7_script.exists():
        log_with_timestamp(f"ERROR: stage7 script not found: {stage7_script}")
        return 1

    all_users = find_users_with_dual_queries(query_dir)
    if not all_users:
        log_with_timestamp("ERROR: No queries_*.json found in query dir")
        return 1

    target_users = sorted(set(args.user_ids)) if args.user_ids else all_users
    if args.user_ids:
        discovered = set(all_users)
        missing = [u for u in target_users if u not in discovered]
        for user_id in missing:
            log_with_timestamp(f"  - Skip {user_id}: dual query file not found")
        target_users = [u for u in target_users if u in discovered]

    if not target_users:
        log_with_timestamp("ERROR: No valid users to process after filtering")
        return 1

    log_with_timestamp(f"Discovered users in query dir: {len(all_users)}")
    log_with_timestamp(f"Target users before profile validation: {len(target_users)}")

    valid_map = validate_users(
        user_ids=target_users,
        linguistic_dir=linguistic_dir,
        use_sentence_level=args.use_sentence_level,
        sentence_level_dir=sentence_level_dir,
    )
    valid_users = sorted(valid_map.keys())

    if not valid_users:
        log_with_timestamp("ERROR: No users passed profile validation")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="stage7_query_subset_") as tmp_dir:
        tmp_query_dir = Path(tmp_dir)
        copied_count = copy_user_query_files(valid_users, query_dir, tmp_query_dir)

        if copied_count == 0:
            log_with_timestamp("ERROR: No query files copied to temporary query directory")
            return 1

        if args.keep_temp_dir:
            persist_dir = output_dir / f"_tmp_query_subset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(tmp_query_dir, persist_dir, dirs_exist_ok=True)
            log_with_timestamp(f"Temporary query subset saved to: {persist_dir}")

        log_with_timestamp(f"Users passed validation: {len(valid_users)}")
        log_with_timestamp(f"Query files copied for refinement: {copied_count}")
        log_with_timestamp("Starting Stage 7 refinement...")

        stage7_module = load_stage7_module(stage7_script)
        stage7_module.run_iterative_refinement(
            query_dir=str(tmp_query_dir),
            linguistic_dir=str(linguistic_dir),
            output_dir=str(output_dir),
            max_rounds=args.max_rounds,
            candidates_per_round=args.candidates_per_round,
            feature_set=args.feature_set,
            max_samples_per_user=args.max_samples,
            max_workers=args.max_workers,
            query_delay=args.query_delay,
            style_strength=args.style_strength,
            style_shot_k=args.style_shot_k,
            use_sentence_level=args.use_sentence_level,
            sentence_level_dir=str(sentence_level_dir),
        )

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "query_dir": str(query_dir),
        "linguistic_dir": str(linguistic_dir),
        "output_dir": str(output_dir),
        "use_sentence_level": bool(args.use_sentence_level),
        "sentence_level_dir": str(sentence_level_dir),
        "total_discovered_users": len(all_users),
        "total_target_users": len(target_users),
        "total_valid_users": len(valid_users),
        "valid_users": valid_users,
    }

    manifest_file = output_dir / "batch_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 7 batch refinement completed")
    log_with_timestamp(f"Manifest saved: {manifest_file}")
    log_with_timestamp("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
