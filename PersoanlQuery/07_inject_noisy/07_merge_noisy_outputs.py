#!/usr/bin/env python3
"""
把旧完整结果与当前重跑的部分结果合并。

规则：
1. 以 `user_id + query_category` 作为主键
2. 当前重跑结果优先，覆盖旧结果中的同键对象
3. 输出保持 Stage 7 当前使用的“多个 JSON 对象串接”格式
"""

import argparse
import json
from pathlib import Path


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


def load_objects(path: Path):
    if not path.exists():
        return []
    return list(iter_concatenated_json_objects(path.read_text(encoding="utf-8")))


def write_objects(path: Path, objects: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj, ensure_ascii=False, indent=2))
            f.write("\n")


def make_key(obj: dict):
    return (obj.get("user_id", ""), obj.get("query_category", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="旧完整结果文件")
    parser.add_argument("--overlay", required=True, help="当前重跑的部分结果文件")
    parser.add_argument("--output", required=True, help="合并输出文件")
    args = parser.parse_args()

    base_path = Path(args.base)
    overlay_path = Path(args.overlay)
    output_path = Path(args.output)

    base_objects = load_objects(base_path)
    overlay_objects = load_objects(overlay_path)

    merged = {}
    for obj in base_objects:
        merged[make_key(obj)] = obj
    for obj in overlay_objects:
        merged[make_key(obj)] = obj

    # 保持尽量稳定的顺序：先按 base 原顺序输出，再补 overlay 中新增键
    output_objects = []
    seen = set()
    for obj in base_objects:
        key = make_key(obj)
        if key in seen:
            continue
        output_objects.append(merged[key])
        seen.add(key)
    for obj in overlay_objects:
        key = make_key(obj)
        if key in seen:
            continue
        output_objects.append(merged[key])
        seen.add(key)

    write_objects(output_path, output_objects)
    print(f"base={len(base_objects)} overlay={len(overlay_objects)} output={len(output_objects)}")


if __name__ == "__main__":
    main()
