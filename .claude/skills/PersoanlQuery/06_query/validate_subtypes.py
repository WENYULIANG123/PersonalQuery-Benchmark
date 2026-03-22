#!/usr/bin/env python3
import glob
import json
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any


LOW_PRIORITY_TOKENS = {
    "really", "very", "quite", "simply", "especially", "particularly",
    "generally", "basically", "overall", "mostly", "usually"
}


def count_words(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def trim_to_max_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    kept = []
    for w in words:
        key = re.sub(r"[^A-Za-z]", "", w).lower()
        if key in LOW_PRIORITY_TOKENS and len(words) - len(kept) > max_words:
            continue
        kept.append(w)
        if len(kept) == max_words:
            break

    if len(kept) < max_words:
        kept = words[:max_words]

    out = " ".join(kept).rstrip(".,;:!?") + "."
    return normalize_spaces(out)


def expand_to_min_words(text: str, min_words: int, safe_tail_pool: List[str]) -> str:
    if count_words(text) >= min_words:
        return text

    out = text.rstrip(".,;:!?")
    pool = safe_tail_pool[:] if safe_tail_pool else ["for consistent daily use"]
    random.shuffle(pool)

    i = 0
    while count_words(out) < min_words:
        tail = pool[i % len(pool)]
        out = normalize_spaces(f"{out} {tail}")
        i += 1
        if i > 10:
            break

    if out and out[-1] not in ".!?":
        out += "."
    return normalize_spaces(out)


def normalize_length(text: str, length_cfg: Dict[str, Any]) -> Tuple[str, int, int]:
    before = count_words(text)
    min_words = int(length_cfg.get("min_words", 25))
    max_words = int(length_cfg.get("max_words", 28))
    safe_tail_pool = list(length_cfg.get("safe_tail_pool", []))

    out = normalize_spaces(text)
    if count_words(out) > max_words:
        out = trim_to_max_words(out, max_words)
    if count_words(out) < min_words:
        out = expand_to_min_words(out, min_words, safe_tail_pool)

    after = count_words(out)
    return out, before, after


def count_pattern_hits(text_lower: str, patterns: List[str]) -> int:
    hits = 0
    for p in patterns:
        if re.search(p, text_lower):
            hits += 1
    return hits


def count_keyword_hits(text_lower: str, keywords: List[str]) -> int:
    hits = 0
    for k in keywords:
        if k.lower() in text_lower:
            hits += 1
    return hits


def evaluate_structure(text: str, subtype: str, rule: Dict[str, Any]) -> bool:
    t = text.lower()

    if subtype == "Interrogative" and rule.get("must_end_with_question_mark", False):
        return t.strip().endswith("?")

    if subtype == "Constraint_List":
        items = 1 + len(re.findall(r",|\band\b", t))
        min_items = int(rule.get("min_constraint_items", 3))
        max_items = int(rule.get("max_constraint_items", 4))
        return min_items <= items <= max_items

    if subtype == "Elliptical_Telegraphic":
        finite_verb_proxy = len(re.findall(r"\b(is|are|was|were|am|do|does|did|has|have|had|will|would|can|could)\b", t))
        return finite_verb_proxy <= int(rule.get("max_finite_verbs_proxy", 1))

    if "max_clause_connectors" in rule:
        connectors = len(re.findall(r"\b(if|unless|provided that|because|although|though|while|so that|in order to)\b", t))
        return connectors <= int(rule["max_clause_connectors"])

    return True


def score_item(length_ok: bool, trigger_ok: bool, forbidden_ok: bool, structure_ok: bool, cfg: Dict[str, Any]) -> float:
    w = cfg["global"]["scoring"]["weights"]
    score = 0.0
    score += float(w.get("length_ok", 0.35)) * (1.0 if length_ok else 0.0)
    score += float(w.get("trigger_hit", 0.4)) * (1.0 if trigger_ok else 0.0)
    score += float(w.get("forbidden_ok", 0.15)) * (1.0 if forbidden_ok else 0.0)
    score += float(w.get("structure_ok", 0.1)) * (1.0 if structure_ok else 0.0)
    return round(score, 4)


def evaluate_query(query: str, subtype: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    length_cfg = cfg["global"]["length"]
    hard_min = int(length_cfg.get("hard_min_words", 20))
    hard_max = int(length_cfg.get("hard_max_words", 35))
    soft_min = int(length_cfg.get("min_words", 25))
    soft_max = int(length_cfg.get("max_words", 28))

    normalized, wc_before, wc_after = normalize_length(query, length_cfg)
    length_ok = soft_min <= wc_after <= soft_max
    hard_ok = hard_min <= wc_after <= hard_max

    subtype_rules = cfg.get("subtypes", {}).get(subtype)
    if not subtype_rules:
        return {
            "subtype": subtype,
            "length_before": wc_before,
            "length_after": wc_after,
            "length_normalized": normalized,
            "length_ok": length_ok,
            "hard_length_ok": hard_ok,
            "trigger_hits": 0,
            "keyword_hits": 0,
            "forbidden_hits": 0,
            "trigger_ok": False,
            "forbidden_ok": True,
            "structure_ok": False,
            "score": 0.0,
            "pass": False,
            "reason": "unknown_subtype"
        }

    text_lower = normalized.lower()
    trigger_hits = count_pattern_hits(text_lower, subtype_rules.get("must_patterns_any", []))
    keyword_hits = count_keyword_hits(text_lower, subtype_rules.get("must_keywords_any", []))
    forbidden_hits = count_pattern_hits(text_lower, subtype_rules.get("forbidden_patterns", []))

    min_trigger = int(subtype_rules.get("min_trigger_hits", 1))
    trigger_ok = (trigger_hits >= min_trigger) or (keyword_hits >= min_trigger)
    forbidden_ok = forbidden_hits == 0
    structure_ok = evaluate_structure(normalized, subtype, subtype_rules)

    score = score_item(length_ok, trigger_ok, forbidden_ok, structure_ok, cfg)
    pass_threshold = float(cfg["global"]["scoring"].get("pass_threshold", 0.8))
    passed = hard_ok and (score >= pass_threshold)

    return {
        "subtype": subtype,
        "length_before": wc_before,
        "length_after": wc_after,
        "length_normalized": normalized,
        "length_ok": length_ok,
        "hard_length_ok": hard_ok,
        "trigger_hits": trigger_hits,
        "keyword_hits": keyword_hits,
        "forbidden_hits": forbidden_hits,
        "trigger_ok": trigger_ok,
        "forbidden_ok": forbidden_ok,
        "structure_ok": structure_ok,
        "score": score,
        "pass": passed,
        "reason": "ok" if passed else "rule_not_met"
    }


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_input_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    return [p for p in files if os.path.isfile(p)]


def main() -> None:
    config = {
        'input_glob': '/fs04/ar57/wenyu/result/personal_query/06_query/queries_*.json',
        'rules_file': os.path.join(os.path.dirname(__file__), 'subtype_qa_rules_v1.json'),
        'output_file': '/fs04/ar57/wenyu/result/personal_query/06_query/subtype_qa_report.json',
        'default_subtype': 'Constraint_List',
        'seed': 42,
    }

    random.seed(config['seed'])

    cfg = load_json(config['rules_file'])
    files = collect_input_files(config['input_glob'])

    all_items: List[Dict[str, Any]] = []
    by_subtype: Dict[str, Dict[str, Any]] = {}

    for fp in files:
        data = load_json(fp)
        user_id = data.get("user_id", "unknown")
        results = data.get("results", [])

        for i, item in enumerate(results):
            tu = item.get("target_user_query", {})
            query = (tu.get("query") or "").strip()
            if not query:
                continue

            subtype = (
                tu.get("subtype")
                or item.get("target_subtype")
                or config['default_subtype']
            )

            ev = evaluate_query(query, subtype, cfg)
            row = {
                "file": fp,
                "user_id": user_id,
                "index": i,
                "asin": item.get("asin", ""),
                "category": item.get("category", ""),
                "query_original": query,
                **ev
            }
            all_items.append(row)

            st = ev["subtype"]
            if st not in by_subtype:
                by_subtype[st] = {
                    "count": 0,
                    "pass_count": 0,
                    "avg_score": 0.0,
                    "length_ok_count": 0,
                    "trigger_ok_count": 0,
                    "forbidden_ok_count": 0,
                    "structure_ok_count": 0
                }
            by_subtype[st]["count"] += 1
            by_subtype[st]["pass_count"] += 1 if ev["pass"] else 0
            by_subtype[st]["avg_score"] += ev["score"]
            by_subtype[st]["length_ok_count"] += 1 if ev["length_ok"] else 0
            by_subtype[st]["trigger_ok_count"] += 1 if ev["trigger_ok"] else 0
            by_subtype[st]["forbidden_ok_count"] += 1 if ev["forbidden_ok"] else 0
            by_subtype[st]["structure_ok_count"] += 1 if ev["structure_ok"] else 0

    total = len(all_items)
    pass_count = sum(1 for x in all_items if x["pass"])
    avg_score = round(sum(x["score"] for x in all_items) / total, 4) if total else 0.0

    for st, agg in by_subtype.items():
        c = max(1, agg["count"])
        agg["pass_rate"] = round(agg["pass_count"] / c, 4)
        agg["avg_score"] = round(agg["avg_score"] / c, 4)
        agg["length_ok_rate"] = round(agg["length_ok_count"] / c, 4)
        agg["trigger_ok_rate"] = round(agg["trigger_ok_count"] / c, 4)
        agg["forbidden_ok_rate"] = round(agg["forbidden_ok_count"] / c, 4)
        agg["structure_ok_rate"] = round(agg["structure_ok_count"] / c, 4)

    failed_examples = [
        {
            "user_id": x["user_id"],
            "asin": x["asin"],
            "subtype": x["subtype"],
            "score": x["score"],
            "reason": x["reason"],
            "query_original": x["query_original"],
            "query_normalized": x["length_normalized"]
        }
        for x in all_items if not x["pass"]
    ][:50]

    report = {
        "timestamp": datetime.now().isoformat(),
        "rules_version": cfg.get("version", "unknown"),
        "input_glob": config['input_glob'],
        "files_scanned": len(files),
        "items_evaluated": total,
        "overall": {
            "pass_count": pass_count,
            "pass_rate": round(pass_count / total, 4) if total else 0.0,
            "avg_score": avg_score
        },
        "by_subtype": by_subtype,
        "failed_examples": failed_examples
    }

    os.makedirs(os.path.dirname(config['output_file']), exist_ok=True)
    with open(config['output_file'], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[Subtype QA] files={len(files)} items={total} pass={pass_count}")
    print(f"[Subtype QA] report={config['output_file']}")


if __name__ == "__main__":
    main()
