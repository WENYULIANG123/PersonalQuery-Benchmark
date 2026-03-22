#!/usr/bin/env python3
import re
import random
from typing import Dict, List, Optional, Tuple


TEMPLATE_POOLS: Dict[str, List[str]] = {
    "Elliptical_Telegraphic": [
        "looking for {A0}, {A1}, under {A2}, {A3}, {A4}, free shipping preferred.",
    ],
    "Constraint_List": [
        "find {A0} that works for {A1}, with {A2} ratings, offering {A3} returns, with {A4} material, and within quick shipping windows.",
    ],
    "Comparative": [
        "looking for {A0} listings in which sellers whose {A1} status is verified have maintained {A2} scores, using {A3} delivery partners under terms where {A4} compensation is guaranteed.",
    ],
}


FALLBACK_TEMPLATES: Dict[str, str] = {
    "Interrogative": "Can you help me find {A0} products from {A1} brand, priced under {A2}, available in {A3} color, made of {A4} material, and ready for quick delivery?",
    "Causal": "Because buyers need reliable performance, retrieve {A0} listings in which sellers whose {A1} status is verified maintain {A2} scores while using {A3} logistics under terms where {A4} compensation is guaranteed.",
    "default": "Find {A0} items that fit for {A1} use, with {A2} ratings, offering {A3} return terms, and delivered within {A4} business days.",
}


def _clean(text: str) -> str:
    out = re.sub(r"\s+", " ", (text or "").strip())
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out


def _safe_get(values: List[str], idx: int, fallback: str) -> str:
    if 0 <= idx < len(values) and values[idx]:
        return values[idx]
    return fallback


def _compact_value(value: str, max_tokens: int = 2) -> str:
    tokens = [t for t in re.split(r"\s+", (value or "").strip()) if t]
    bad = {
        "who", "whose", "which", "that", "where", "when", "why", "how",
        "dont", "don't", "cant", "can't", "wont", "won't", "didnt", "didn't", "isnt", "aren't", "arent", "wasnt", "weren't", "werent", "not", "none",
        "this", "these", "those", "there", "here", "then", "really", "very",
    }
    filtered = []
    for t in tokens:
        k = re.sub(r"[^a-z0-9%$-]", "", t.lower())
        if not k or k in bad:
            continue
        filtered.append(t)
    if filtered:
        tokens = filtered
    if not tokens:
        return value
    if len(tokens) <= max_tokens:
        return " ".join(tokens)
    return " ".join(tokens[:max_tokens])


def generate_query_from_attributes(
    category: str,
    selected_attrs: List[Tuple[str, str]],
    subtype: str,
    rng: Optional[random.Random] = None,
) -> Tuple[str, str]:
    values = [v for _, v in selected_attrs if isinstance(v, str) and v.strip()]
    c = category if category else "product"

    a0 = _compact_value(_safe_get(values, 0, f"reliable {c}"), max_tokens=2)
    a1 = _compact_value(_safe_get(values, 1, "good overall performance"), max_tokens=2)
    a2 = _compact_value(_safe_get(values, 2, "easy daily use"), max_tokens=2)
    a3 = _compact_value(_safe_get(values, 3, "regular crafting tasks"), max_tokens=2)
    a4 = _compact_value(_safe_get(values, 4, c), max_tokens=2)

    fmt_map = {
        "A0": a0,
        "A1": a1,
        "A2": a2,
        "A3": a3,
        "A4": a4,
    }

    chooser = rng if rng is not None else random
    pool = TEMPLATE_POOLS.get(subtype, [])
    if pool:
        idx = chooser.randrange(len(pool))
        template_text = pool[idx]
        template_id = f"{subtype}#{idx+1}"
    elif subtype in FALLBACK_TEMPLATES:
        template_text = FALLBACK_TEMPLATES[subtype]
        template_id = f"{subtype}#fallback"
    else:
        template_text = FALLBACK_TEMPLATES["default"]
        template_id = "default#fallback"

    q = template_text.format(**fmt_map)

    q = _clean(q)
    if q and q[-1] not in ".!?":
        q += "."
    return q, template_id
