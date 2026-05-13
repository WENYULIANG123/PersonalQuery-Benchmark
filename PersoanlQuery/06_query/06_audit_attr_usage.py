#!/usr/bin/env python3
"""Audit Stage 6 query outputs under the strict 5-attribute usage rule."""

from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


REPO_ROOT = Path("/fs04/ar57/wenyu")
QUERY_ROOT = REPO_ROOT / "result" / "personal_query" / "06_query"
CATEGORIES = ("Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies")
OUTPUT_FILE = QUERY_ROOT / "strict_attr_usage_audit.json"
REQUIRED_ATTR_COUNT = 5
VALID_ATTR_KEYS = {f'A{i}' for i in range(1, 19)}


def _canonicalize_attr_text(raw_value: str) -> str:
    raw_value = re.sub(r'\s+', ' ', raw_value).strip()
    if '(' in raw_value:
        prefix = raw_value.split('(', 1)[0].strip()
        if prefix:
            raw_value = prefix
    return raw_value.strip()


def count_attr_value_occurrences(query: str, attr_value: str) -> int:
    counts = count_attr_value_occurrences_map(query, {"attr": attr_value})
    return counts["attr"]


def _normalize_variant_token(token: str) -> str:
    token = token.casefold()
    if len(token) > 4 and token.endswith('ies'):
        token = token[:-3] + 'y'
    elif len(token) > 3 and token.endswith('es') and not token.endswith(('aes', 'ees', 'oes')):
        token = token[:-2]
    elif len(token) > 3 and token.endswith('s') and not token.endswith('ss'):
        token = token[:-1]

    for suffix in ('ingly', 'edly', 'ing', 'ed', 'en', 'ly', 'ness', 'ment', 'er', 'est', 'al', 'ic', 'ish'):
        if len(token) - len(suffix) >= 3 and token.endswith(suffix):
            token = token[:-len(suffix)]
            break
    return token


def _build_attr_value_pattern(attr_value: str) -> str | None:
    if not isinstance(attr_value, str) or not attr_value.strip():
        return None
    attr_value = _canonicalize_attr_text(attr_value)
    if not attr_value:
        return None
    try:
        numeric_value = Decimal(attr_value)
    except InvalidOperation:
        numeric_value = None
    if numeric_value is not None:
        normalized = format(numeric_value.normalize(), 'f')
        if '.' in normalized:
            integer_part, fractional_part = normalized.split('.', 1)
            return rf"(?<![A-Za-z0-9])\$?{re.escape(integer_part)}\.{re.escape(fractional_part)}(?:0+)?(?![A-Za-z0-9])"
        return rf"(?<![A-Za-z0-9])\$?{re.escape(normalized)}(?:\.0+)?(?![A-Za-z0-9])"
    if re.fullmatch(r"[A-Za-z0-9' ]+", attr_value):
        return rf"\b{re.escape(attr_value)}\b"
    return re.escape(attr_value)


def _find_variant_token_spans(query: str, attr_value: str) -> list[tuple[int, int]]:
    attr_value = _canonicalize_attr_text(attr_value)
    attr_tokens = [match.group(0) for match in re.finditer(r"[A-Za-z0-9']+", attr_value)]
    if not attr_tokens:
        return []
    query_tokens = list(re.finditer(r"[A-Za-z0-9']+", query))
    if len(query_tokens) < len(attr_tokens):
        return []

    normalized_attr_tokens = [_normalize_variant_token(token) for token in attr_tokens]
    spans = []
    window = len(attr_tokens)
    for start_index in range(len(query_tokens) - window + 1):
        token_window = query_tokens[start_index:start_index + window]
        normalized_query_tokens = [_normalize_variant_token(token.group(0)) for token in token_window]
        if normalized_query_tokens == normalized_attr_tokens:
            spans.append((token_window[0].start(), token_window[-1].end()))
    return spans


def count_attr_value_occurrences_map(query: str, attrs_used: dict) -> dict[str, int]:
    matches_by_key = {}
    counts = {}
    for key, value in attrs_used.items():
        pattern = _build_attr_value_pattern(value)
        if pattern is None:
            matches = []
        else:
            matches = [match.span() for match in re.finditer(pattern, query, re.IGNORECASE)]
        if isinstance(value, str):
            matches.extend(_find_variant_token_spans(query, value))
        if not matches:
            matches_by_key[key] = []
            counts[key] = 0
            continue
        matches_by_key[key] = sorted(set(matches))
        counts[key] = 0

    occupied_spans: list[tuple[int, int]] = []
    ordered_keys = sorted(
        attrs_used,
        key=lambda key: (-len(str(attrs_used[key]).strip()), key),
    )
    for key in ordered_keys:
        for span in matches_by_key[key]:
            if any(not (span[1] <= used[0] or span[0] >= used[1]) for used in occupied_spans):
                continue
            occupied_spans.append(span)
            counts[key] += 1
    return counts


def normalize_attr_value_for_duplicate_check(attr_value: str) -> str:
    if not isinstance(attr_value, str):
        return str(attr_value)
    attr_value = _canonicalize_attr_text(attr_value)
    try:
        numeric_value = Decimal(attr_value)
    except InvalidOperation:
        numeric_value = None
    if numeric_value is not None:
        return f"NUM::{format(numeric_value.normalize(), 'f')}"
    return f"TEXT::{attr_value.lower()}"


def audit_query(query: str, attrs_used: dict) -> tuple[list[str], dict, list[list[str]]]:
    reasons = []
    occurrence_map = {}

    invalid_keys = sorted(key for key in attrs_used if key not in VALID_ATTR_KEYS)
    for key in invalid_keys:
        reasons.append(f"invalid_attr_key={key}")

    attr_count = len(attrs_used)
    if attr_count != REQUIRED_ATTR_COUNT:
        reasons.append(f"attr_count={attr_count}")

    normalized_value_to_keys: dict[str, list[str]] = {}
    for key, value in attrs_used.items():
        normalized_value = normalize_attr_value_for_duplicate_check(value)
        normalized_value_to_keys.setdefault(normalized_value, []).append(key)

    duplicate_key_groups = [
        sorted(keys)
        for keys in normalized_value_to_keys.values()
        if len(keys) > 1
    ]
    for keys in duplicate_key_groups:
        reasons.append(f"duplicate_attr_keys={'+'.join(keys)}")

    occurrence_counts = count_attr_value_occurrences_map(query, attrs_used)
    for key, value in attrs_used.items():
        count = occurrence_counts[key]
        occurrence_map[key] = {
            "value": value,
            "count": count,
        }
        if count != 1:
            reasons.append(f"{key}_count={count}")

    return reasons, occurrence_map, duplicate_key_groups


def main() -> None:
    report = {
        "categories": {},
        "overall": {},
    }

    overall_reason_counts: Counter[str] = Counter()
    overall_total_queries = 0
    overall_bad_queries = 0

    for category in CATEGORIES:
        category_file = QUERY_ROOT / category / "query.json"
        rows = json.loads(category_file.read_text(encoding="utf-8"))

        items = []
        reason_counts: Counter[str] = Counter()
        bad_by_query_type: Counter[str] = Counter()
        total_queries = 0

        for row in rows:
            for query_type in ("acl_query", "ccomp_query"):
                total_queries += 1
                query_item = row[query_type]
                reasons, occurrence_map, duplicate_key_groups = audit_query(
                    query_item["query"], query_item["attrs_used"]
                )
                if not reasons:
                    continue

                bad_by_query_type[query_type] += 1
                reason_counts.update(reasons)
                items.append(
                    {
                        "user_id": row["user_id"],
                        "asin": row["asin"],
                        "query_type": query_type,
                        "level": query_item["level"],
                        "query": query_item["query"],
                        "attr_count": len(query_item["attrs_used"]),
                        "attrs_used": query_item["attrs_used"],
                        "reasons": reasons,
                        "occurrence_map": occurrence_map,
                        "duplicate_attr_key_groups": duplicate_key_groups,
                    }
                )

        bad_queries = len(items)
        good_queries = total_queries - bad_queries
        report["categories"][category] = {
            "total_queries": total_queries,
            "bad_queries": bad_queries,
            "good_queries": good_queries,
            "bad_by_query_type": dict(sorted(bad_by_query_type.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "items": items,
        }

        overall_total_queries += total_queries
        overall_bad_queries += bad_queries
        overall_reason_counts.update(reason_counts)

    report["overall"] = {
        "total_queries": overall_total_queries,
        "bad_queries": overall_bad_queries,
        "good_queries": overall_total_queries - overall_bad_queries,
        "reason_counts": dict(sorted(overall_reason_counts.items())),
    }

    OUTPUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"report={OUTPUT_FILE}")
    print(f"overall_total_queries={overall_total_queries}")
    print(f"overall_bad_queries={overall_bad_queries}")
    for category in CATEGORIES:
        info = report["categories"][category]
        print(
            f"{category}\ttotal={info['total_queries']}\tbad={info['bad_queries']}"
            f"\tacl_bad={info['bad_by_query_type'].get('acl_query', 0)}"
            f"\tccomp_bad={info['bad_by_query_type'].get('ccomp_query', 0)}"
        )


if __name__ == "__main__":
    main()
