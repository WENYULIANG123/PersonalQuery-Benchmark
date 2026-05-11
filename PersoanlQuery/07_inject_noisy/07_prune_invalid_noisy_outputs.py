#!/usr/bin/env python3
"""
删除 Stage 7 结果里不合格的 user/category 记录，供定向重跑使用。

输入文件采用当前 Stage 7 的“多个 JSON 对象串接”格式。
输出文件保留其余合格对象，之后原脚本会基于“已完成 user_id”跳过保留项，
只重跑被删掉的 user/category。
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "noisy_query_config.json"


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


def write_output_objects(path: Path, objects: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj, ensure_ascii=False, indent=2))
            f.write("\n")


def count_keyword(query: str, keyword: str) -> int:
    token_strip_chars = " \t\n\r\f\v.,;:!?\"“”‘’()[]{}<>"
    keyword_lower = keyword.lower()
    count = 0
    for raw_token in query.split():
        token = raw_token.strip(token_strip_chars)
        if token.lower() == keyword_lower:
            count += 1
    return count


def expected_complexity_ok(query: str, query_category: str, level: int) -> bool:
    which_count = count_keyword(query, "which")
    that_count = count_keyword(query, "that")
    if query_category == "acl":
        return which_count == level and that_count == 0
    if query_category == "ccomp":
        return that_count == level and which_count == 0
    raise ValueError(f"未知 query_category: {query_category}")


def query_contains_exact_anchor(query: str, text: str) -> bool:
    if not isinstance(query, str) or not isinstance(text, str) or not text:
        return False
    escaped = re.escape(text)
    if re.fullmatch(r"[A-Za-z0-9']+", text):
        return re.search(rf"\b{escaped}\b", query, flags=re.IGNORECASE) is not None
    return re.search(escaped, query, flags=re.IGNORECASE) is not None


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


def object_is_invalid(item: dict, user_errors: dict) -> bool:
    query_category = item.get("query_category")
    ground_truth_query = item.get("ground_truth_query", "")
    noisy_query = item.get("noisy_query", "")
    injected_errors = item.get("injected_errors", [])
    level = item.get("original_query_info", {}).get("level", 0)
    uid = item.get("user_id", "")

    if not expected_complexity_ok(ground_truth_query, query_category, level):
        return True
    if not expected_complexity_ok(noisy_query, query_category, level):
        return True
    if noisy_query == ground_truth_query:
        return True
    if not isinstance(injected_errors, list) or not injected_errors:
        return True

    real_pairs = build_real_error_pairs(user_errors.get(uid, {}).get(query_category, []))
    for injected_error in injected_errors:
        correct = injected_error.get("correct", "")
        error = injected_error.get("error", "")
        if not query_contains_exact_anchor(ground_truth_query, correct):
            return True
        if not query_contains_exact_anchor(noisy_query, error):
            return True
        if (correct, error) not in real_pairs and (correct.lower(), error.lower()) not in real_pairs:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_json(CONFIG_PATH)
    category_config = config["categories"][args.category]
    user_errors = load_user_errors(Path(category_config["user_error_file"]))
    input_path = Path(args.input)
    output_path = Path(args.output)

    objects = load_output_objects(input_path)
    invalid_pairs = set()
    invalid_counter = Counter()
    for obj in objects:
        pair = (obj.get("user_id", ""), obj.get("query_category", ""))
        if object_is_invalid(obj, user_errors):
            invalid_pairs.add(pair)
            invalid_counter[obj.get("query_category", "unknown")] += 1

    filtered_objects = [
        obj for obj in objects if (obj.get("user_id", ""), obj.get("query_category", "")) not in invalid_pairs
    ]

    print(f"category={args.category}")
    print(f"input_total={len(objects)}")
    print(f"invalid_pairs={len(invalid_pairs)}")
    print(f"invalid_objects={dict(invalid_counter)}")
    print(f"output_total={len(filtered_objects)}")

    if not args.dry_run:
        write_output_objects(output_path, filtered_objects)
        print(f"written={output_path}")


if __name__ == "__main__":
    main()
