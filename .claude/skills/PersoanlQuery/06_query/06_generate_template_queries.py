#!/usr/bin/env python3
import json
import importlib.util
import hashlib
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_tpl_path = os.path.join(CURRENT_DIR, "query_templates.py")
_tpl_spec = importlib.util.spec_from_file_location("stage6_query_templates", _tpl_path)
if _tpl_spec is None or _tpl_spec.loader is None:
    raise RuntimeError(f"Failed to load template module from {_tpl_path}")
_tpl_mod = importlib.util.module_from_spec(_tpl_spec)
_tpl_spec.loader.exec_module(_tpl_mod)
generate_query_from_attributes = _tpl_mod.generate_query_from_attributes


SUBTYPE_BY_LEVEL = {
    "low": "Elliptical_Telegraphic",
    "medium": "Constraint_List",
    "high": "Comparative",
}

SUBTYPES = [
    "Conditional",
    "Causal",
    "Concessive",
    "Comparative",
    "Purpose",
    "Passive",
    "Apposition_Parenthetical",
    "Interrogative",
    "Elliptical_Telegraphic",
    "Constraint_List",
]

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "you", "are", "was",
    "were", "have", "has", "had", "but", "not", "all", "can", "will", "would", "just", "very",
    "about", "then", "than", "when", "where", "while", "they", "them", "their", "there", "also",
    "what", "which", "much", "many", "more", "most", "some", "such", "only", "over", "under",
    "out", "off", "our", "its", "it's", "it's", "it's", "it's", "too", "few", "lot", "use",
    "used", "using", "like", "made", "make", "made", "still", "after", "before", "being", "been",
}

COLOR_WORDS = {
    "black", "white", "blue", "red", "green", "pink", "purple", "gray", "grey", "brown",
    "beige", "navy", "silver", "gold", "yellow", "orange", "clear", "transparent"
}

MATERIAL_WORDS = {
    "cotton", "wool", "silicone", "leather", "metal", "plastic", "polyester", "linen", "nylon",
    "canvas", "wood", "bamboo", "paper", "steel", "rubber", "ceramic", "glass", "acrylic"
}


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _word_count(text: str) -> int:
    return len([w for w in (text or "").split() if w])


def _pick_level(profile: Dict) -> str:
    counts = profile.get("complexity_rule_based", {}).get("sentence_counts")
    if not isinstance(counts, dict):
        counts = profile.get("sentence_counts", {})
    if not isinstance(counts, dict) or not counts:
        return "medium"

    ranked = sorted(
        [(k, int(v)) for k, v in counts.items() if k in ("low", "medium", "high")],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[0][0] if ranked else "medium"


def _extract_keywords(text: str, top_k: int = 18) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", text or "")
    seen = set()
    out = []
    for w in words:
        key = w.lower()
        if key in STOPWORDS:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
        if len(out) >= top_k:
            break
    return out


def _safe_phrase(words: List[str], start: int, fallback: str) -> str:
    if start < len(words):
        if start + 1 < len(words):
            return f"{words[start]} {words[start + 1]}"
        return words[start]
    return fallback


def _extract_price_phrase(text: str) -> str:
    m = re.search(r"\$(\d{1,4})", text or "")
    if m:
        return f"${m.group(1)}"
    m = re.search(r"\b(\d{2,4})\b", text or "")
    if m:
        return f"${m.group(1)}"
    return "$50"


def _extract_color_phrase(words: List[str]) -> str:
    for w in words:
        k = w.lower()
        if k in COLOR_WORDS:
            return k
    return "black"


def _extract_material_phrase(words: List[str]) -> str:
    for w in words:
        k = w.lower()
        if k in MATERIAL_WORDS:
            return k
    return "cotton"


def _build_attributes(category: str, review_text: str) -> List[Tuple[str, str]]:
    kws = _extract_keywords(review_text, top_k=24)
    c = category or "Craft Supplies"
    product_phrase = _safe_phrase(kws, 0, c.lower())
    brand_phrase = _safe_phrase(kws, 2, "trusted")
    price_phrase = _extract_price_phrase(review_text)
    color_phrase = _extract_color_phrase(kws)
    material_phrase = _extract_material_phrase(kws)
    delivery_days = "5"

    return [
        ("Product_Keyword", product_phrase),
        ("Brand_Preference", brand_phrase),
        ("Price_Range", price_phrase),
        ("Color_Style", color_phrase),
        ("Material_Composition", material_phrase),
        ("Delivery_Days", delivery_days),
        ("Product_Category", c),
        ("Quality_Craftsmanship", f"{_safe_phrase(kws, 8, 'consistent quality')}"),
        ("Target_User", _safe_phrase(kws, 14, "craft users")),
    ]


def _default_subtype_scores(selected: str) -> Dict[str, float]:
    scores = {name: 0.12 for name in SUBTYPES}
    if selected in scores:
        scores[selected] = 1.25
    return scores


def _load_existing_context(user_id: str, output_dir: str) -> Tuple[str, str]:
    fp = os.path.join(output_dir, f"queries_{user_id}.json")
    if not os.path.exists(fp):
        return "", "Craft Supplies"
    try:
        old = _read_json(fp)
        asin = old.get("reviewed_asin", "")
        cat = "Craft Supplies"
        results = old.get("results", [])
        if results and isinstance(results[0], dict):
            cat = results[0].get("category", cat) or cat
            asin = results[0].get("asin", asin) or asin
        return asin, cat
    except Exception:
        return "", "Craft Supplies"


def _build_rng(user_id: str, seed: Optional[int]) -> random.Random:
    if seed is None:
        return random.Random()
    uid_hash = int(hashlib.md5(user_id.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(int(seed) + uid_hash)


def run_generation(
    linguistic_profile_file: str,
    output_dir: str,
    seed: Optional[int] = None,
    forced_level: Optional[str] = None,
) -> str:
    profile = _read_json(linguistic_profile_file)
    user_id = profile.get("user_id")
    if not user_id:
        raise ValueError("Missing user_id in linguistic profile")

    rng = _build_rng(user_id, seed)

    if isinstance(forced_level, str) and forced_level.lower() in {"low", "medium", "high"}:
        level = forced_level.lower()
    else:
        level = _pick_level(profile)
    subtype = SUBTYPE_BY_LEVEL.get(level, "Constraint_List")
    subtype_scores = _default_subtype_scores(subtype)
    sentence_counts = profile.get("complexity_rule_based", {}).get("sentence_counts", {})
    complexity_templates = profile.get("complexity_templates", {})
    level_block = complexity_templates.get(level, {}) if isinstance(complexity_templates, dict) else {}
    skeleton_template = level_block.get("skeleton_template", "")
    review_text = level_block.get("review_text", "")

    reviewed_asin, category = _load_existing_context(user_id, output_dir)
    attrs = _build_attributes(category, review_text)

    query_text, template_id = generate_query_from_attributes(category, attrs, subtype, rng=rng)

    result = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "method": "template_library_only_no_llm",
        "reviewed_asin": reviewed_asin,
        "selected_skeleton_level": level,
        "selected_subtype": subtype,
        "selected_template_id": template_id,
        "selected_subtype_scores": subtype_scores,
        "sentence_counts": sentence_counts,
        "skeleton_template": skeleton_template,
        "total_queries": 1,
        "successful_target_queries": 1,
        "results": [
            {
                "asin": reviewed_asin,
                "category": category,
                "user_id": user_id,
                "target_subtype": subtype,
                "skeleton_level": level,
                "shared_dimensions": [d for d, _ in attrs],
                "target_user_query": {
                    "query": query_text,
                    "subtype": subtype,
                    "template_id": template_id,
                    "subtype_scores": subtype_scores,
                    "word_count": _word_count(query_text),
                    "attempts": 1,
                    "error_words_valid": True,
                    "missing_error_words": [],
                    "selected_attributes": [
                        {"dimension": d, "value": v} for d, v in attrs
                    ],
                    "attribute_priority_tracking": [
                        {
                            "dimension": d,
                            "attribute": v,
                            "priority_level": "medium",
                            "reason": "模板链路默认优先级"
                        }
                        for d, v in attrs
                    ]
                },
                "skeleton_template": skeleton_template,
            }
        ],
    }

    os.makedirs(output_dir, exist_ok=True)
    out_fp = os.path.join(output_dir, f"queries_{user_id}.json")
    _write_json(out_fp, result)
    return out_fp
