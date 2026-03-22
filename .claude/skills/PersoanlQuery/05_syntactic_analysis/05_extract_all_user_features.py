#!/usr/bin/env python3

import importlib.util
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    for query_file in sorted(query_dir.glob("queries_*.json")):
        user_id = query_file.stem.replace("queries_", "")
        if user_id and user_id != "summary":
            users.append(user_id)
    return users


def discover_users_from_stage0_dir(stage0_dir: Path) -> List[str]:
    users = []
    for review_file in sorted(stage0_dir.glob("reviews_*.json")):
        user_id = review_file.stem.replace("reviews_", "")
        if user_id:
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


_WORKER_STAGE5 = None
_WORKER_EXTRACTOR = None
_WORKER_OUTPUT_DIR = None
_WORKER_MAX_REVIEWS = None
_WORKER_SPACY_N_PROCESS = 1
_WORKER_SPACY_BATCH_SIZE = 32


def init_worker(
    stage5_script_path: str,
    output_dir: str,
    max_reviews,
    spacy_n_process: int,
    spacy_batch_size: int,
) -> None:
    global _WORKER_STAGE5, _WORKER_EXTRACTOR, _WORKER_OUTPUT_DIR
    global _WORKER_MAX_REVIEWS, _WORKER_SPACY_N_PROCESS, _WORKER_SPACY_BATCH_SIZE

    _WORKER_STAGE5 = load_stage5_module(Path(stage5_script_path))
    _WORKER_EXTRACTOR = _WORKER_STAGE5.LocalFeatureExtractor()
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_MAX_REVIEWS = max_reviews
    _WORKER_SPACY_N_PROCESS = spacy_n_process
    _WORKER_SPACY_BATCH_SIZE = spacy_batch_size


def process_one_user(job: Dict) -> Dict:
    user_id = job["user_id"]
    reviews = job["reviews"]

    if _WORKER_STAGE5 is None or _WORKER_EXTRACTOR is None or _WORKER_OUTPUT_DIR is None:
        return {"status": "failed", "user_id": user_id, "error": "worker_not_initialized"}

    try:
        profile = _WORKER_STAGE5.extract_user_profile(
            user_id,
            reviews,
            _WORKER_EXTRACTOR,
            _WORKER_MAX_REVIEWS,
            _WORKER_SPACY_N_PROCESS,
            _WORKER_SPACY_BATCH_SIZE,
        )
        if not profile:
            return {"status": "empty_profile", "user_id": user_id}

        _WORKER_STAGE5.save_profile(profile, _WORKER_OUTPUT_DIR)
        return {
            "status": "ok",
            "user_id": user_id,
            "num_reviews_processed": profile.get("num_reviews_processed", 0),
            "feature_count": profile.get("feature_count", 0),
        }
    except Exception as e:
        return {"status": "failed", "user_id": user_id, "error": str(e)}


def main() -> int:
    config = {
        "user_source": "stage0",
        "query_dir": "/fs04/ar57/wenyu/result/personal_query/06_query",
        "stage0_reviews_dir": "/fs04/ar57/wenyu/result/personal_query/00_data_preparation",
        "reviews_file": "/fs04/ar57/wenyu/result/personal_query/00_data_preparation/all_user_reviews.json",
        "output_dir": "/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis",
        "user_ids": None,
        "max_reviews": None,
        "spacy_n_process": 4,
        "spacy_batch_size": 64,
        "user_workers": 10,
    }

    script_dir = Path(__file__).parent
    stage5_script = script_dir / "05_extract_local_features.py"
    user_source = config["user_source"]
    query_dir = Path(config["query_dir"])
    stage0_reviews_dir = Path(config["stage0_reviews_dir"])
    reviews_file = Path(config["reviews_file"])
    output_dir = Path(config["output_dir"])
    reviews_source = reviews_file if reviews_file.exists() else stage0_reviews_dir

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 5 Batch Linguistic Feature Extraction")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"User source: {user_source}")
    log_with_timestamp(f"Query dir: {query_dir}")
    log_with_timestamp(f"Stage0 dir: {stage0_reviews_dir}")
    log_with_timestamp(f"Reviews file: {reviews_file}")
    log_with_timestamp(f"Reviews source used: {reviews_source}")
    log_with_timestamp(f"Output dir: {output_dir}")
    log_with_timestamp(f"spaCy n_process: {config['spacy_n_process']}")
    log_with_timestamp(f"spaCy batch_size: {config['spacy_batch_size']}")
    log_with_timestamp(f"user_workers: {config['user_workers']}")

    if not stage5_script.exists():
        log_with_timestamp(f"ERROR: Stage5 script not found: {stage5_script}")
        return 1
    if not reviews_source.exists():
        log_with_timestamp(f"ERROR: reviews source not found: {reviews_source}")
        return 1
    if user_source == "stage6" and config["user_ids"] is None and not query_dir.exists():
        log_with_timestamp(f"ERROR: query dir not found: {query_dir}")
        return 1
    if user_source == "stage0" and config["user_ids"] is None and not stage0_reviews_dir.exists():
        log_with_timestamp(f"ERROR: stage0 reviews dir not found: {stage0_reviews_dir}")
        return 1

    stage5 = load_stage5_module(stage5_script)
    user_reviews: Dict[str, List[Dict]] = stage5.load_user_reviews(str(reviews_source))

    if config["user_ids"] is None:
        if user_source == "stage6":
            discovered_users = discover_users_from_query_dir(query_dir)
        else:
            discovered_users = discover_users_from_stage0_dir(stage0_reviews_dir)
    else:
        discovered_users = []
    target_users = sorted(set(config["user_ids"])) if config["user_ids"] else discovered_users

    if not target_users:
        log_with_timestamp("ERROR: No target users found")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    user_workers = int(config["user_workers"]) if config["user_workers"] else 1
    effective_user_workers = max(1, user_workers)
    effective_spacy_n_process = int(config["spacy_n_process"]) if config["spacy_n_process"] else 1
    if effective_user_workers > 1 and effective_spacy_n_process > 1:
        effective_spacy_n_process = 1
        log_with_timestamp("Adjust spaCy n_process to 1 because user_workers > 1")

    extractor = None
    if effective_user_workers == 1:
        extractor = stage5.LocalFeatureExtractor()

    stats = {
        "timestamp": datetime.now().isoformat(),
        "user_source": user_source,
        "query_dir": str(query_dir),
        "stage0_reviews_dir": str(stage0_reviews_dir),
        "reviews_file": str(reviews_file),
        "reviews_source": str(reviews_source),
        "output_dir": str(output_dir),
        "total_target_users": len(target_users),
        "processed_users": 0,
        "failed_users": 0,
        "missing_reviews_users": [],
        "empty_profile_users": [],
        "failed_user_reasons": {},
        "fallback_reviews_users": [],
    }

    jobs = []
    for idx, user_id in enumerate(target_users, start=1):
        reviews = user_reviews.get(user_id)
        if not reviews:
            fallback_file = stage0_reviews_dir / f"reviews_{user_id}.json"
            reviews = load_reviews_from_stage0_user_file(fallback_file)
            if reviews:
                stats["fallback_reviews_users"].append(user_id)
            else:
                stats["missing_reviews_users"].append(user_id)
                stats["failed_users"] += 1
                stats["failed_user_reasons"][user_id] = "reviews_not_found"
                log_with_timestamp(f"[{idx}/{len(target_users)}] Skip {user_id}: reviews not found")
                continue
        jobs.append({"user_id": user_id, "reviews": reviews})

    total_jobs = len(jobs)
    if total_jobs == 0:
        log_with_timestamp("ERROR: No runnable users found after review resolution")
        return 1

    if effective_user_workers == 1:
        if extractor is None:
            log_with_timestamp("ERROR: extractor is not initialized in single-worker mode")
            return 1
        for idx, job in enumerate(jobs, start=1):
            user_id = job["user_id"]
            reviews = job["reviews"]
            try:
                profile = stage5.extract_user_profile(
                    user_id,
                    reviews,
                    extractor,
                    config["max_reviews"],
                    effective_spacy_n_process,
                    config["spacy_batch_size"],
                )
                if not profile:
                    stats["empty_profile_users"].append(user_id)
                    stats["failed_users"] += 1
                    stats["failed_user_reasons"][user_id] = "empty_profile"
                    log_with_timestamp(f"[{idx}/{total_jobs}] Skip {user_id}: empty profile")
                    continue

                stage5.save_profile(profile, str(output_dir))
                stats["processed_users"] += 1
                log_with_timestamp(
                    f"[{idx}/{total_jobs}] Done {user_id}: "
                    f"reviews={profile.get('num_reviews_processed', 0)}, "
                    f"features={profile.get('feature_count', 0)}"
                )
            except Exception as e:
                stats["failed_users"] += 1
                stats["failed_user_reasons"][user_id] = str(e)
                log_with_timestamp(f"[{idx}/{total_jobs}] Failed {user_id}: {e}")
    else:
        max_workers = min(effective_user_workers, total_jobs, max(1, os.cpu_count() or 1))
        log_with_timestamp(f"Run user-level parallel with {max_workers} workers")
        completed = 0
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(
                str(stage5_script),
                str(output_dir),
                config["max_reviews"],
                effective_spacy_n_process,
                config["spacy_batch_size"],
            ),
        ) as executor:
            future_to_user = {executor.submit(process_one_user, job): job["user_id"] for job in jobs}
            for future in as_completed(future_to_user):
                completed += 1
                result = future.result()
                user_id = result.get("user_id", future_to_user[future])
                status = result.get("status")
                if status == "ok":
                    stats["processed_users"] += 1
                    log_with_timestamp(
                        f"[{completed}/{total_jobs}] Done {user_id}: "
                        f"reviews={result.get('num_reviews_processed', 0)}, "
                        f"features={result.get('feature_count', 0)}"
                    )
                elif status == "empty_profile":
                    stats["empty_profile_users"].append(user_id)
                    stats["failed_users"] += 1
                    stats["failed_user_reasons"][user_id] = "empty_profile"
                    log_with_timestamp(f"[{completed}/{total_jobs}] Skip {user_id}: empty profile")
                else:
                    stats["failed_users"] += 1
                    stats["failed_user_reasons"][user_id] = result.get("error", "unknown_error")
                    log_with_timestamp(f"[{completed}/{total_jobs}] Failed {user_id}: {result.get('error', 'unknown_error')}")

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
