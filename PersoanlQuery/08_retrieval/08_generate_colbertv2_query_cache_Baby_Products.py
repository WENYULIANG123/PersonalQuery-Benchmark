#!/usr/bin/env python3
"""Generate Baby_Products ColBERTv2 query token embedding caches."""

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_QUERY_CACHE_SCRIPT = SCRIPT_DIR / "08_generate_query_cache_Baby_Products.py"


def load_main_query_cache_module():
    if not MAIN_QUERY_CACHE_SCRIPT.exists():
        raise FileNotFoundError(f"Required query cache script not found: {MAIN_QUERY_CACHE_SCRIPT}")

    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("generate_query_cache_baby_products", MAIN_QUERY_CACHE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load query cache script: {MAIN_QUERY_CACHE_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_main_query_cache_module()
    module.log_with_timestamp("=" * 80)
    module.log_with_timestamp("BUILD BABY_PRODUCTS COLBERTV2 QUERY TOKEN EMBEDDING CACHE - STARTING")
    module.log_with_timestamp("=" * 80)

    acl_correct, _ = module.load_acl_queries()
    ccomp_correct, _ = module.load_ccomp_queries()
    if not acl_correct and not ccomp_correct:
        raise ValueError("No ACL or CCOMP correct queries found for ColBERTv2 cache generation")

    module.initialize_cache_dir()
    query_types = [
        ("ACL", acl_correct, module._build_queries_by_user(acl_correct), "acl_correct"),
        ("CCOMP", ccomp_correct, module._build_queries_by_user(ccomp_correct), "ccomp_correct"),
    ]
    stats = module.generate_colbertv2_cache_from_query_types(query_types)

    module.log_with_timestamp("=" * 80)
    module.log_with_timestamp("BUILD BABY_PRODUCTS COLBERTV2 QUERY TOKEN EMBEDDING CACHE - COMPLETE")
    module.log_with_timestamp(f"Total cached query embeddings: {stats['total_cached']}")
    for summary in stats["summaries"]:
        module.log_with_timestamp(
            f"{summary['mode']}: query_count={summary['query_count']}, "
            f"unique_query_count={summary['unique_query_count']}, "
            f"elapsed={summary['elapsed_seconds']:.1f}s"
        )
    module.log_with_timestamp("当前任务已完成，请做下一个任务的指示。")


if __name__ == "__main__":
    main()
