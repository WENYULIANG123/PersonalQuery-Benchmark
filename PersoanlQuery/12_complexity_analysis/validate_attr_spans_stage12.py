#!/usr/bin/env python3
"""Validate Stage 12 attribute spans against Stage 6 attr usage rules."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path("/fs04/ar57/wenyu")
STAGE12_PATH = REPO_ROOT / "PersoanlQuery" / "12_complexity_analysis" / "12_complexity_analysis.py"
QUERY_ROOT = REPO_ROOT / "result" / "personal_query" / "06_query"
CATEGORIES = ("Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies")
QUERY_TYPES = ("acl_query", "ccomp_query")


def load_stage12_module():
    spec = importlib.util.spec_from_file_location("stage12_complexity_analysis", STAGE12_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 stage12 模块: {STAGE12_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_stage12_module()
    nlp = module._load_spacy_model()

    total_checked = 0
    for category in CATEGORIES:
        rows = json.loads((QUERY_ROOT / category / "query.json").read_text(encoding="utf-8"))
        category_checked = 0
        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"{category} rows[{row_idx}] 必须是对象")
            for query_type in QUERY_TYPES:
                payload = row.get(query_type)
                if not isinstance(payload, dict):
                    raise TypeError(f"{category} rows[{row_idx}].{query_type} 必须是对象")
                query = payload.get("query")
                attrs_used = payload.get("attrs_used")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError(f"{category} rows[{row_idx}].{query_type}.query 为空")
                if not isinstance(attrs_used, dict):
                    raise TypeError(f"{category} rows[{row_idx}].{query_type}.attrs_used 必须是对象")
                doc = nlp(query.strip())
                spans = module._attribute_spans(doc, attrs_used)
                if len(spans) != 5:
                    raise ValueError(
                        f"{category} rows[{row_idx}].{query_type} span_count={len(spans)} "
                        f"user_id={row.get('user_id')} asin={row.get('asin')}"
                    )
                category_checked += 1
                total_checked += 1
        print(f"PASS {category} checked={category_checked}", flush=True)

    print(f"ALL_PASS total_checked={total_checked}", flush=True)


if __name__ == "__main__":
    main()
