#!/usr/bin/env python3
"""
Stage 7 结果审计脚本。

校验项：
1. `ground_truth_query` / `noisy_query` 的 `that` / `which` 复杂度是否符合 `level`
2. `injected_errors.correct` 是否出现在 `ground_truth_query`
3. `injected_errors.error` 是否出现在 `noisy_query`
4. 注入对是否属于该用户真实错误模式
5. `noisy_query` 是否真的和 `ground_truth_query` 不同
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "noisy_query_config.json"

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_concatenated_json_objects(text: str):
    decoder = json.JSONDecoder()
    index = 0
    text_len = len(text)
    while index < text_len:
        while index < text_len and text[index].isspace():
            index += 1
        if index >= text_len:
            break
        obj, next_index = decoder.raw_decode(text, index)
        yield obj
        index = next_index


def load_output_objects(path: Path):
    return list(iter_concatenated_json_objects(path.read_text(encoding="utf-8")))


def count_keyword(query: str, keyword: str) -> int:
    token_strip_chars = " \t\n\r\f\v.,;:!?\"“”‘’()[]{}<>"
    keyword_lower = keyword.lower()
    count = 0
    for raw_token in query.split():
        token = raw_token.strip(token_strip_chars)
        if token.lower() == keyword_lower:
            count += 1
    return count


def query_contains_exact_anchor(query: str, text: str) -> bool:
    if not isinstance(query, str) or not isinstance(text, str) or not text:
        return False
    escaped = re.escape(text)
    if re.fullmatch(r"[A-Za-z0-9']+", text):
        return re.search(rf"\b{escaped}\b", query, flags=re.IGNORECASE) is not None
    return re.search(escaped, query, flags=re.IGNORECASE) is not None


def expected_complexity_ok(query: str, query_category: str, level: int) -> bool:
    which_count = count_keyword(query, "which")
    that_count = count_keyword(query, "that")
    if query_category == "acl":
        return which_count == level and that_count == 0
    if query_category == "ccomp":
        return that_count == level and which_count == 0
    raise ValueError(f"未知 query_category: {query_category}")


def load_user_errors(error_file: Path):
    data = load_json(error_file)
    users_list = data if isinstance(data, list) else data.get("user_results", [])
    user_errors = {}
    for user in users_list:
        uid = user["user_id"]
        if user.get("total_errors", 0) == 0 or not user.get("detailed_results"):
            continue
        grouped = {"acl": [], "ccomp": []}
        for detail in user["detailed_results"]:
            category = detail.get("error_category", "")
            if category not in grouped:
                continue
            seen = set()
            for err in detail.get("errors", []):
                orig = err.get("original", "")
                corr = err.get("corrected", "")
                pair = (orig, corr)
                if pair in seen:
                    continue
                seen.add(pair)
                grouped[category].append(
                    {
                        "original": orig,
                        "corrected": corr,
                        "error_type": err.get("error_type", "unknown"),
                    }
                )
        user_errors[uid] = grouped
    return user_errors


def build_real_error_pairs(error_patterns):
    pairs = set()
    for pattern in error_patterns:
        corrected = pattern.get("corrected", "")
        original = pattern.get("original", "")
        if corrected and original:
            pairs.add((corrected, original))
            pairs.add((corrected.lower(), original.lower()))
    return pairs


def safe_pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def build_category_map(config):
    return {
        category: {
            "output_file": Path(category_config["noisy_output_file"]),
            "error_file": Path(category_config["user_error_file"]),
        }
        for category, category_config in config["categories"].items()
    }


def validate_objects(category: str, objects: list, user_errors: dict):
    issues = Counter()
    issue_examples = defaultdict(list)

    for item in objects:
        query_category = item.get("query_category")
        ground_truth_query = item.get("ground_truth_query", "")
        noisy_query = item.get("noisy_query", "")
        injected_errors = item.get("injected_errors", [])
        level = item.get("original_query_info", {}).get("level", 0)
        uid = item.get("user_id", "")

        if not expected_complexity_ok(ground_truth_query, query_category, level):
            issues["ground_truth_complexity_mismatch"] += 1
            if len(issue_examples["ground_truth_complexity_mismatch"]) < 5:
                issue_examples["ground_truth_complexity_mismatch"].append(
                    {"user_id": uid, "query_category": query_category, "level": level, "query": ground_truth_query}
                )

        if not expected_complexity_ok(noisy_query, query_category, level):
            issues["noisy_query_complexity_mismatch"] += 1
            if len(issue_examples["noisy_query_complexity_mismatch"]) < 5:
                issue_examples["noisy_query_complexity_mismatch"].append(
                    {"user_id": uid, "query_category": query_category, "level": level, "query": noisy_query}
                )

        if noisy_query == ground_truth_query:
            issues["noisy_equals_ground_truth"] += 1
            if len(issue_examples["noisy_equals_ground_truth"]) < 5:
                issue_examples["noisy_equals_ground_truth"].append({"user_id": uid, "query": noisy_query})

        if not isinstance(injected_errors, list) or not injected_errors:
            issues["missing_injected_errors"] += 1
            if len(issue_examples["missing_injected_errors"]) < 5:
                issue_examples["missing_injected_errors"].append({"user_id": uid, "query": ground_truth_query})
            continue

        real_pairs = build_real_error_pairs(user_errors.get(uid, {}).get(query_category, []))
        for injected_error in injected_errors:
            correct = injected_error.get("correct", "")
            error = injected_error.get("error", "")
            if not query_contains_exact_anchor(ground_truth_query, correct):
                issues["correct_not_in_ground_truth"] += 1
                if len(issue_examples["correct_not_in_ground_truth"]) < 5:
                    issue_examples["correct_not_in_ground_truth"].append(
                        {"user_id": uid, "correct": correct, "ground_truth_query": ground_truth_query}
                    )
            if not query_contains_exact_anchor(noisy_query, error):
                issues["error_not_in_noisy"] += 1
                if len(issue_examples["error_not_in_noisy"]) < 5:
                    issue_examples["error_not_in_noisy"].append(
                        {"user_id": uid, "error": error, "noisy_query": noisy_query}
                    )
            if (correct, error) not in real_pairs and (correct.lower(), error.lower()) not in real_pairs:
                issues["pattern_mismatch"] += 1
                if len(issue_examples["pattern_mismatch"]) < 5:
                    issue_examples["pattern_mismatch"].append(
                        {"user_id": uid, "pair": [correct, error], "query_category": query_category}
                    )

    total = len(objects)
    return {
        "category_name": category,
        "total": total,
        "issue_counts": dict(issues),
        "issue_rates": {key: safe_pct(value, total) for key, value in issues.items()},
        "issue_examples": dict(issue_examples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        choices=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies", "all"],
        default="all",
    )
    args = parser.parse_args()

    config = load_json(CONFIG_PATH)
    category_map = build_category_map(config)
    selected_categories = category_map.keys() if args.category == "all" else [args.category]

    report = {}
    for category in selected_categories:
        output_file = category_map[category]["output_file"]
        error_file = category_map[category]["error_file"]
        objects = load_output_objects(output_file)
        user_errors = load_user_errors(error_file)
        report[category] = validate_objects(category, objects, user_errors)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
