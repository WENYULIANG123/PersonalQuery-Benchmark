#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/fs04/ar57/wenyu")
QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / "Grocery_and_Gourmet_Food" / "query.json"
AUDIT_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / "strict_attr_usage_audit.json"


def main() -> None:
    rows = json.loads(QUERY_FILE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    bad_items = audit["categories"]["Grocery_and_Gourmet_Food"]["items"]
    bad_users = {item["user_id"] for item in bad_items}

    backup_file = QUERY_FILE.with_name(
        f"query.json.backup_before_bad_user_prune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    backup_file.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    filtered_rows = [row for row in rows if row["user_id"] not in bad_users]
    QUERY_FILE.write_text(
        json.dumps(filtered_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"backup={backup_file}")
    print(f"rows_before={len(rows)}")
    print(f"bad_items={len(bad_items)}")
    print(f"bad_users={len(bad_users)}")
    print(f"rows_after={len(filtered_rows)}")


if __name__ == "__main__":
    main()
