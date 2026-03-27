#!/usr/bin/env python3
import glob
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List


CLAUSE_CONNECTOR_PATTERN = r"\b(that|which|who|whose|where|when|if|unless|because|although|though|while|whereas|provided that|in which|so that)\b"
RELATIVE_MARKER_PATTERN = r"\b(that|which|who|whose|where|in which)\b"
NEGATION_PATTERN = r"\b(no|not|never|none|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't)\b"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_words(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def count_hits(text_lower: str, patterns: List[str]) -> int:
    hits = 0
    for p in patterns:
        if re.search(p, text_lower):
            hits += 1
    return hits


def count_clause_connectors(text_lower: str) -> int:
    return len(re.findall(CLAUSE_CONNECTOR_PATTERN, text_lower))


def count_relative_markers(text_lower: str) -> int:
    return len(re.findall(RELATIVE_MARKER_PATTERN, text_lower))


def count_nested_markers(text_lower: str) -> int:
    nested_cues = ["in which", "whose", "where", "under agreements", "using", "verified", "guaranteed"]
    return sum(1 for cue in nested_cues if cue in text_lower)


def count_negations(text_lower: str) -> int:
    return len(re.findall(NEGATION_PATTERN, text_lower))


def evaluate_item(query: str, level: str, rules: Dict[str, Any]) -> Dict[str, Any]:
    q = (query or "").strip()
    ql = q.lower()
    wc = count_words(q)

    level_rule = rules["levels"].get(level)
    if not level_rule:
        return {
            "pass": False,
            "reason": f"unknown_level:{level}",
            "word_count": wc,
        }

    word_cfg = rules["global"]["word_range"].get(level, [24, 31])
    min_w, max_w = int(word_cfg[0]), int(word_cfg[1])

    must_hits = count_hits(ql, level_rule.get("must_patterns_any", []))
    min_must_hits = int(level_rule.get("min_must_hits", 1))
    clause_connectors = count_clause_connectors(ql)
    rel_markers = count_relative_markers(ql)
    nested_markers = count_nested_markers(ql)
    negations = count_negations(ql)

    checks = {
        "word_ok": min_w <= wc <= max_w,
        "must_ok": must_hits >= min_must_hits,
        "min_clause_ok": clause_connectors >= int(level_rule.get("min_clause_connectors", 0)),
        "max_clause_ok": clause_connectors <= int(level_rule.get("max_clause_connectors", 10**9)),
        "min_rel_ok": rel_markers >= int(level_rule.get("min_relative_markers", 0)),
        "max_rel_ok": rel_markers <= int(level_rule.get("max_relative_markers", 10**9)),
        "min_nested_ok": nested_markers >= int(level_rule.get("min_nested_markers", 0)),
        "max_neg_ok": negations <= int(level_rule.get("max_negations", 10**9)),
    }

    passed = all(checks.values())
    return {
        "pass": passed,
        "reason": "ok" if passed else "rule_not_met",
        "word_count": wc,
        "must_hits": must_hits,
        "clause_connectors": clause_connectors,
        "relative_markers": rel_markers,
        "nested_markers": nested_markers,
        "negations": negations,
        **checks,
    }


def main() -> None:
    base_dir = os.path.dirname(__file__)
    rules_file = os.path.join(base_dir, "complexity_template_rules_v1.json")
    rules = load_json(rules_file)

    input_glob = rules["global"]["input_glob"]
    output_file = rules["global"]["output_file"]
    files = sorted(glob.glob(input_glob))

    rows: List[Dict[str, Any]] = []
    by_level: Dict[str, Dict[str, Any]] = {
        "low": {"count": 0, "pass_count": 0},
        "medium": {"count": 0, "pass_count": 0},
        "high": {"count": 0, "pass_count": 0},
    }

    for fp in files:
        data = load_json(fp)
        user_id = data.get("user_id", "unknown")
        top_level = str(data.get("selected_subtype", "HIGH-1")).lower()
        for idx, item in enumerate(data.get("results", [])):
            tu = item.get("target_user_query", {})
            query = (tu.get("query") or "").strip()
            if not query:
                continue
            level = str(item.get("skeleton_level", top_level)).lower()
            ev = evaluate_item(query, level, rules)
            row = {
                "file": fp,
                "user_id": user_id,
                "index": idx,
                "asin": item.get("asin", ""),
                "category": item.get("category", ""),
                "level": level,
                "query": query,
                **ev,
            }
            rows.append(row)
            if level in by_level:
                by_level[level]["count"] += 1
                by_level[level]["pass_count"] += 1 if ev["pass"] else 0

    total = len(rows)
    pass_count = sum(1 for r in rows if r["pass"])
    for lv, agg in by_level.items():
        c = agg["count"]
        agg["pass_rate"] = round((agg["pass_count"] / c), 4) if c else 0.0

    report = {
        "timestamp": datetime.now().isoformat(),
        "rules_version": rules.get("version", "unknown"),
        "input_glob": input_glob,
        "files_scanned": len(files),
        "items_evaluated": total,
        "overall": {
            "pass_count": pass_count,
            "pass_rate": round((pass_count / total), 4) if total else 0.0,
        },
        "by_level": by_level,
        "failed_examples": [
            {
                "user_id": r["user_id"],
                "level": r["level"],
                "reason": r["reason"],
                "word_count": r["word_count"],
                "must_hits": r["must_hits"],
                "clause_connectors": r["clause_connectors"],
                "relative_markers": r["relative_markers"],
                "query": r["query"],
            }
            for r in rows if not r["pass"]
        ][:100],
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[Complexity QA] files={len(files)} items={total} pass={pass_count}")
    print(f"[Complexity QA] report={output_file}")


if __name__ == "__main__":
    main()
