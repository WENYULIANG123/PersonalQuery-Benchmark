#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)

    if not input_file.exists():
        print(f"ERROR: input file not found: {input_file}")
        return 1

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("ERROR: input JSON must be a list of records")
        return 1

    per_user = defaultdict(list)
    for row in data:
        if not isinstance(row, dict):
            continue
        user_id = row.get("user_id")
        if not user_id:
            continue
        per_user[user_id].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for user_id in sorted(per_user.keys()):
        user_file = output_dir / f"{user_id}_interative_query.json"
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(per_user[user_id], f, indent=2, ensure_ascii=False)
        written += 1

    print(f"Done. users={written}, input_records={len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
