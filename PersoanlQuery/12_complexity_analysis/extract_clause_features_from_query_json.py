#!/usr/bin/env python3
"""Batch extract 20 clause-related features from Stage 06 query_by_syntax_depth.json."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path("/fs04/ar57/wenyu")
QUERY_ROOT = REPO_ROOT / "result" / "personal_query" / "06_query"
OUTPUT_ROOT = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
SINGLE_QUERY_SCRIPT = REPO_ROOT / "PersoanlQuery" / "12_complexity_analysis" / "extract_clause_features_single_query.py"
SOURCE_QUERY_FILE = "query_by_syntax_depth.json"
QUERY_PAYLOAD_KEY = "syntax_depth_query"


def load_single_query_module():
    spec = importlib.util.spec_from_file_location("extract_clause_features_single_query", SINGLE_QUERY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载单 query 特征脚本: {SINGLE_QUERY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch extract 20 clause-related features from Stage 06 query_by_syntax_depth.json")
    parser.add_argument("--category", required=True, help="类别名，例如 Baby_Products")
    parser.add_argument("--max-rows", type=int, default=None, help="仅处理前 N 行，默认全量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category = args.category
    query_file = QUERY_ROOT / category / SOURCE_QUERY_FILE
    if not query_file.is_file():
        raise FileNotFoundError(f"{SOURCE_QUERY_FILE} 不存在: {query_file}")

    rows = json.loads(query_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{query_file} 顶层必须是列表")

    selected_rows = rows if args.max_rows is None else rows[: args.max_rows]
    extractor = load_single_query_module()
    nlp = extractor.load_spacy_model()

    query_texts = []
    metadata = []
    for row_idx, row in enumerate(selected_rows):
        if not isinstance(row, dict):
            raise TypeError(f"rows[{row_idx}] 必须是对象")
        payload = row.get(QUERY_PAYLOAD_KEY)
        if not isinstance(payload, dict):
            raise TypeError(f"rows[{row_idx}].{QUERY_PAYLOAD_KEY} 必须是对象")
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"rows[{row_idx}].{QUERY_PAYLOAD_KEY}.query 必须是非空字符串")
        query_texts.append(query.strip())
        metadata.append((row, payload))

    docs = list(nlp.pipe(query_texts, batch_size=64))
    if len(docs) != len(query_texts):
        raise RuntimeError("spaCy 文档数量与输入 query 数量不一致")

    output_rows = []
    for (row, payload), doc, query_text in zip(metadata, docs, query_texts):
        result = extractor.extract_clause_features_from_doc(doc, query_text)
        output_rows.append(
            {
                "user_id": row.get("user_id"),
                "asin": row.get("asin"),
                "query_type": QUERY_PAYLOAD_KEY,
                "target_depth": payload.get("target_depth"),
                "actual_depth": payload.get("actual_depth"),
                "user_avg_depth": payload.get("user_avg_depth"),
                "query": result["query"],
                "word_count": result["word_count"],
                "features": result["features"],
            }
        )

    output_dir = OUTPUT_ROOT / category
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "single_query_clause_features.jsonl"
    with output_file.open("w", encoding="utf-8") as f:
        for item in output_rows:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")

    summary = {
        "category": category,
        "source_query_file": str(query_file),
        "output_file": str(output_file),
        "num_input_rows": len(selected_rows),
        "num_output_rows": len(output_rows),
        "query_types": [QUERY_PAYLOAD_KEY],
        "feature_names": list(output_rows[0]["features"].keys()) if output_rows else [],
    }
    summary_file = output_dir / "single_query_clause_features_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
