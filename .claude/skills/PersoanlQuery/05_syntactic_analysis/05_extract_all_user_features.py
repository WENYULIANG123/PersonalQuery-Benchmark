#!/usr/bin/env python3

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def log_with_timestamp(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_stage5_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("stage5_local_features", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_users_from_query_dir(query_dir: Path) -> List[str]:
    users = []
    for query_file in sorted(query_dir.glob("dual_queries_*.json")):
        user_id = query_file.stem.replace("dual_queries_", "")
        if user_id and user_id != "summary":
            users.append(user_id)
    return users


def load_reviews_from_stage0_user_file(user_file: Path) -> List[Dict]:
    if not user_file.exists():
        return []

    try:
        with open(user_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    results = data.get("results", []) if isinstance(data, dict) else []
    reviews: List[Dict] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        target_reviews = item.get("target_reviews", [])
        if not isinstance(target_reviews, list):
            continue
        for text in target_reviews:
            if isinstance(text, str) and text.strip():
                reviews.append({"reviewText": text.strip()})
    return reviews


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch extract Stage5 linguistic profiles for all Stage6 users")
    parser.add_argument(
        "--query-dir",
        default="/fs04/ar57/wenyu/result/personal_query/06_query",
        help="Directory containing dual_queries_{USER_ID}.json files",
    )
    parser.add_argument(
        "--reviews-file",
        default="/fs04/ar57/wenyu/result/personal_query/00_data_preparation/all_user_reviews.json",
        help="Path to all_user_reviews.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis",
        help="Output directory for linguistic_profile_{USER_ID}.json",
    )
    parser.add_argument(
        "--user-ids",
        nargs="+",
        default=None,
        help="Optional explicit user IDs; default discovers from --query-dir",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help="Maximum reviews per user",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    script_dir = Path(__file__).parent
    stage5_script = script_dir / "05_extract_local_features.py"
    query_dir = Path(args.query_dir)
    reviews_file = Path(args.reviews_file)
    output_dir = Path(args.output_dir)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 5 Batch Linguistic Feature Extraction")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Query dir: {query_dir}")
    log_with_timestamp(f"Reviews file: {reviews_file}")
    log_with_timestamp(f"Output dir: {output_dir}")

    if not stage5_script.exists():
        log_with_timestamp(f"ERROR: Stage5 script not found: {stage5_script}")
        return 1
    if not reviews_file.exists():
        log_with_timestamp(f"ERROR: reviews file not found: {reviews_file}")
        return 1
    if args.user_ids is None and not query_dir.exists():
        log_with_timestamp(f"ERROR: query dir not found: {query_dir}")
        return 1

    stage5 = load_stage5_module(stage5_script)
    user_reviews: Dict[str, List[Dict]] = stage5.load_user_reviews(str(reviews_file))
    stage0_user_reviews_dir = reviews_file.parent

    discovered_users = discover_users_from_query_dir(query_dir) if args.user_ids is None else []
    target_users = sorted(set(args.user_ids)) if args.user_ids else discovered_users

    if not target_users:
        log_with_timestamp("ERROR: No target users found")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = stage5.LocalFeatureExtractor()

    stats = {
        "timestamp": datetime.now().isoformat(),
        "query_dir": str(query_dir),
        "reviews_file": str(reviews_file),
        "output_dir": str(output_dir),
        "total_target_users": len(target_users),
        "processed_users": 0,
        "failed_users": 0,
        "missing_reviews_users": [],
        "empty_profile_users": [],
        "failed_user_reasons": {},
        "fallback_reviews_users": [],
    }

    for idx, user_id in enumerate(target_users, start=1):
        reviews = user_reviews.get(user_id)
        if not reviews:
            fallback_file = stage0_user_reviews_dir / f"reviews_{user_id}.json"
            reviews = load_reviews_from_stage0_user_file(fallback_file)
            if reviews:
                stats["fallback_reviews_users"].append(user_id)
            else:
                stats["missing_reviews_users"].append(user_id)
                stats["failed_users"] += 1
                stats["failed_user_reasons"][user_id] = "reviews_not_found"
                log_with_timestamp(f"[{idx}/{len(target_users)}] Skip {user_id}: reviews not found")
                continue

        try:
            profile = stage5.extract_user_profile(user_id, reviews, extractor, args.max_reviews)
            if not profile:
                stats["empty_profile_users"].append(user_id)
                stats["failed_users"] += 1
                stats["failed_user_reasons"][user_id] = "empty_profile"
                log_with_timestamp(f"[{idx}/{len(target_users)}] Skip {user_id}: empty profile")
                continue

            stage5.save_profile(profile, str(output_dir))
            stats["processed_users"] += 1
            log_with_timestamp(
                f"[{idx}/{len(target_users)}] Done {user_id}: "
                f"reviews={profile.get('num_reviews_processed', 0)}, "
                f"features={profile.get('feature_count', 0)}"
            )
        except Exception as e:
            stats["failed_users"] += 1
            stats["failed_user_reasons"][user_id] = str(e)
            log_with_timestamp(f"[{idx}/{len(target_users)}] Failed {user_id}: {e}")

    summary_file = output_dir / "batch_extraction_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 5 batch extraction completed")
    log_with_timestamp(f"Processed users: {stats['processed_users']}/{stats['total_target_users']}")
    log_with_timestamp(f"Failed users: {stats['failed_users']}")
    log_with_timestamp(f"Summary: {summary_file}")
    log_with_timestamp("=" * 80)

    return 0 if stats["processed_users"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
