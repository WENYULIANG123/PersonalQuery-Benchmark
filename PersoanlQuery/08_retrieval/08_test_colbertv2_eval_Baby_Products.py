#!/usr/bin/env python3
"""Smoke test for Baby_Products ColBERTv2 cached-result evaluation."""

import argparse
import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "08_fast_fullscale_eval_Baby_Products.py"


def load_eval_module():
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"Evaluation script not found: {EVAL_SCRIPT}")

    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("baby_products_fullscale_eval", EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluation script: {EVAL_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idf-sample-size", type=int, default=1000)
    parser.add_argument("--query-categories", nargs="+", choices=["acl", "ccomp"], default=["acl", "ccomp"])
    return parser.parse_args()


def print_table(title, headers, rows):
    if not rows:
        raise ValueError(f"No rows to print for table: {title}")

    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), max(len(row[i]) for row in str_rows))
        for i, header in enumerate(headers)
    ]

    print(f"\n{title}")
    print("| " + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)) + " |")
    print("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
    for row in str_rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")


def main():
    args = parse_args()
    module = load_eval_module()
    k_values = [1, 3, 5, 10]

    print("Building test IDF dictionary...")
    word_idf = module.build_word_idf_dict(module.META_FILE, sample_size=args.idf_sample_size)

    overall_rows = []
    group_rows = []

    for query_category in args.query_categories:
        print(f"\nTesting ColBERTv2 cached evaluation: {query_category}/correct")
        user_queries, user_to_group, _ = module.load_user_queries("correct", query_category)
        result = module.evaluate_cached_result_retriever(
            "colbertv2",
            user_queries,
            user_to_group,
            k_values,
            word_idf,
            "correct",
            query_category,
        )
        metrics = result["metrics"]
        overall_rows.append(
            [
                "colbertv2",
                query_category,
                "correct",
                result["num_users"],
                result["num_queries"],
                f"{metrics['P@1']:.6f}",
                f"{metrics['P@3']:.6f}",
                f"{metrics['P@5']:.6f}",
                f"{metrics['P@10']:.6f}",
                f"{metrics['N@10']:.6f}",
                f"{metrics['MR@10']:.6f}",
                f"{metrics['H@10']:.6f}",
            ]
        )

        for group, count in sorted(result["group_counts"].items()):
            group_metrics = result["group_metrics"][group]
            group_rows.append(
                [
                    "colbertv2",
                    query_category,
                    group,
                    count,
                    f"{group_metrics['P@1']:.6f}",
                    f"{group_metrics['P@3']:.6f}",
                    f"{group_metrics['P@5']:.6f}",
                    f"{group_metrics['P@10']:.6f}",
                    f"{group_metrics['N@10']:.6f}",
                    f"{group_metrics['MR@10']:.6f}",
                    f"{group_metrics['H@10']:.6f}",
                ]
            )

    print_table(
        "OVERALL METRICS",
        ["retriever", "category", "query_type", "users", "queries", "P@1", "P@3", "P@5", "P@10", "N@10", "MR@10", "H@10"],
        overall_rows,
    )
    print_table(
        "GROUP METRICS",
        ["retriever", "category", "level", "count", "P@1", "P@3", "P@5", "P@10", "N@10", "MR@10", "H@10"],
        group_rows,
    )

    print("\nColBERTv2 cached evaluation smoke test passed.")


if __name__ == "__main__":
    main()
