#!/usr/bin/env python3
from __future__ import annotations

import json
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from statistics import mean


REPO_ROOT = Path("/fs04/ar57/wenyu")
QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / "Baby_Products" / "query.json"
OUTPUT_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / "Baby_Products" / "sentence_syntax_tree_depth.json"
QUERY_TYPES = ("acl_query", "ccomp_query")


@lru_cache(maxsize=1)
def _load_spacy_model():
    """加载 spaCy 英文句法模型，只初始化一次。"""
    import spacy

    return spacy.load("en_core_web_sm")


def _token_dependency_depth(token) -> int:
    """计算单个 token 到依存树根节点的深度，根节点深度记为 1。"""
    if token.head == token:
        return 1
    return _token_dependency_depth(token.head) + 1


def compute_sentence_syntax_tree_depth(sentence: str) -> int:
    """
    计算一句话的依存句法树深度。

    规则：
    - 使用 spaCy 的依存句法分析结果；
    - 根节点深度记为 1；
    - 句子深度取所有非空白、非标点 token 的最大深度。
    """
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")

    sentence = sentence.strip()
    if not sentence:
        raise ValueError("sentence must be a non-empty string")

    nlp = _load_spacy_model()
    doc = nlp(sentence)
    if len(doc) == 0:
        raise ValueError("sentence produced an empty document")

    depths = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        depths.append(_token_dependency_depth(token))

    if not depths:
        raise ValueError("sentence contains no valid tokens for depth computation")

    return max(depths)


def load_query_rows() -> list[dict]:
    """读取 Baby_Products 的 query.json。"""
    if not QUERY_FILE.exists():
        raise FileNotFoundError(f"query file not found: {QUERY_FILE}")

    rows = json.loads(QUERY_FILE.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"query file must contain a list: {QUERY_FILE}")
    return rows


def compute_all_query_depths(rows: list[dict]) -> list[dict]:
    """计算每条 ACL/CCOMP 查询语句的句法树深度。"""
    records: list[dict] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"row #{row_index} must be an object")

        if "user_id" not in row:
            raise KeyError(f"row #{row_index} missing user_id")
        if "asin" not in row:
            raise KeyError(f"row #{row_index} missing asin")

        for query_type in QUERY_TYPES:
            if query_type not in row:
                raise KeyError(f"row #{row_index} missing {query_type}")

            query_item = row[query_type]
            if not isinstance(query_item, dict):
                raise TypeError(f"row #{row_index} {query_type} must be an object")
            if "query" not in query_item:
                raise KeyError(f"row #{row_index} {query_type} missing query")
            if "level" not in query_item:
                raise KeyError(f"row #{row_index} {query_type} missing level")

            query_text = query_item["query"]
            if not isinstance(query_text, str) or not query_text.strip():
                raise ValueError(f"row #{row_index} {query_type}.query must be a non-empty string")

            depth = compute_sentence_syntax_tree_depth(query_text)
            records.append(
                {
                    "user_id": row["user_id"],
                    "asin": row["asin"],
                    "query_type": query_type,
                    "level": query_item["level"],
                    "query": query_text,
                    "word_count": query_item.get("word_count"),
                    "syntax_tree_depth": depth,
                }
            )

    return records


def build_summary(records: list[dict]) -> dict:
    """汇总深度统计。"""
    if not records:
        raise ValueError("no query depth records were generated")

    depths = [int(item["syntax_tree_depth"]) for item in records]
    summary = {
        "total_queries": len(records),
        "min_depth": min(depths),
        "max_depth": max(depths),
        "avg_depth": round(mean(depths), 4),
        "by_query_type": {},
    }

    for query_type in QUERY_TYPES:
        typed_depths = [int(item["syntax_tree_depth"]) for item in records if item["query_type"] == query_type]
        if not typed_depths:
            raise ValueError(f"no records found for {query_type}")
        summary["by_query_type"][query_type] = {
            "count": len(typed_depths),
            "min_depth": min(typed_depths),
            "max_depth": max(typed_depths),
            "avg_depth": round(mean(typed_depths), 4),
        }

    return summary


def main() -> None:
    rows = load_query_rows()
    records = compute_all_query_depths(rows)
    summary = build_summary(records)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "source_query_file": str(QUERY_FILE),
        "category": "Baby_Products",
        "summary": summary,
        "records": records,
    }

    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"source={QUERY_FILE}")
    print(f"output={OUTPUT_FILE}")
    print(f"total_queries={summary['total_queries']}")
    print(f"min_depth={summary['min_depth']}")
    print(f"max_depth={summary['max_depth']}")
    print(f"avg_depth={summary['avg_depth']}")
    for query_type, stats in summary["by_query_type"].items():
        print(
            f"{query_type}: count={stats['count']} min={stats['min_depth']} "
            f"max={stats['max_depth']} avg={stats['avg_depth']}"
        )


if __name__ == "__main__":
    main()
