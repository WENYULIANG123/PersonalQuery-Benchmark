#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path("/fs04/ar57/wenyu")
QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / "Grocery_and_Gourmet_Food" / "query.json"
AUDIT_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / "strict_attr_usage_audit.json"


def main() -> None:
    rows = json.loads(QUERY_FILE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    bad_items = audit["categories"]["Grocery_and_Gourmet_Food"]["items"]
    bad_users = {item["user_id"] for item in bad_items}
    row_users = {row["user_id"] for row in rows}
    intersection = sorted(row_users & bad_users)

    print(f"rows={len(rows)}")
    print(f"row_users={len(row_users)}")
    print(f"bad_items={len(bad_items)}")
    print(f"bad_users={len(bad_users)}")
    print(f"intersection={len(intersection)}")
    if intersection:
        print(f"sample={intersection[:10]}")


if __name__ == "__main__":
    main()
