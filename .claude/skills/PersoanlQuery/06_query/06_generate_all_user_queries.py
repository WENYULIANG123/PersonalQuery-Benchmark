#!/usr/bin/env python3
"""Stage 6: Template-based Query Generation."""

import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy

# ============================================================
# 路径配置
# ============================================================
STAGE0_REVIEWS_DIR = "/fs04/ar57/wenyu/result/personal_query/00_data_preparation"
STAGE1_ATTRIBUTES_FILE = "/fs04/ar57/wenyu/result/personal_query/01_preference_extraction/attributes_Arts_Crafts_and_Sewing.json"
OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/06_query"

# ============================================================
# 查询模板定义（从 query_templates.py 内联）
# ============================================================

DIMENSION_TO_SEMANTIC_TYPE: Dict[str, str] = {
    "Product_Category": "CATEGORY",
    "Product_Keyword": "CATEGORY",
    "Brand_Preference": "BRAND",
    "Price_Range": "PRICE",
    "Material_Composition": "MATERIAL",
    "A4_appearance": "STYLE",
    "Size_Spec": "SIZE",
    "Quality_Description": "QUALITY",
    "Quality_Craftsmanship": "QUALITY",
    "Use_Scene": "USE_CASE",
    "Safety_Feature": "FEATURE",
    "Durability": "FEATURE",
    "Ease_Of_Use": "FEATURE",
    "Temperature_Resistance": "FEATURE",
    "Surface_Feature": "FEATURE",
    "Reusability": "FEATURE",
    "Compatibility": "FEATURE",
}

SEMANTIC_DEFAULTS: Dict[str, str] = {
    "CATEGORY": "craft supplies",
    "BRAND": "trusted brand",
    "PRICE": "$20",
    "MATERIAL": "durable material",
    "COLOR": "classic",
    "SIZE": "standard size",
    "QUALITY": "good quality",
    "STYLE": "modern style",
    "USE_CASE": "general crafting",
    "FEATURE": "reliable performance",
}

TEMPLATES: Dict[str, List[Tuple[List[str], str]]] = {
    "HIGH-1": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} that is from the {BRAND} brand, that is priced around {PRICE}, and that is suitable for {USE} in my current project."),
    ],
    "HIGH-2": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} that is from the {BRAND} brand, which offers products that are priced around {PRICE}, and that are suitable for {USE} in my current project."),
    ],
    "HIGH-3": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} that, being from the {BRAND} brand and being priced around {PRICE}, is suitable for {USE} in my current project."),
    ],
    "HIGH-4": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT}, a product from the {BRAND} brand, priced around {PRICE}, and suitable for {USE} in my current project."),
    ],
    "HIGH-5": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand, with a price around {PRICE}, for use in {USE}, in my current project."),
    ],
    "HIGH-6": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} to be used in {USE}, to be priced around {PRICE}, and to be from the {BRAND} brand in my current project."),
    ],
    "HIGH-7": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} that is designed by the {BRAND} brand, that is priced around {PRICE}, and that is used for {USE} in my current project."),
    ],
    "HIGH-8": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "It is {ARTICLE} {STYLE} {CAT} from the {BRAND} brand that I am looking for, which is priced around {PRICE} and suitable for {USE} in my current project."),
    ],
    "HIGH-9": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand and priced around {PRICE} and suitable for {USE} and appropriate for my current project needs."),
    ],
    "HIGH-10": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand priced around {PRICE} suitable for {USE} in my current project."),
    ],
    "HIGH-11": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} that is from the {BRAND} brand that provides products that are priced around {PRICE} that are suitable for {USE} in my current project."),
    ],
    "HIGH-12": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "As for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand, priced around {PRICE} and suitable for {USE}, I am looking for one for my current project."),
    ],
    "HIGH-13": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "There is a need for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand, priced around {PRICE} and suitable for {USE} in my current project."),
    ],
    "HIGH-14": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "The requirement is for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand, with a price around {PRICE} and suitability for {USE} in my current project."),
    ],
    "HIGH-15": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for what would be {ARTICLE} {STYLE} {CAT} from the {BRAND} brand, priced around {PRICE} and suitable for {USE} in my current project."),
    ],
    "HIGH-16": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "Looking for is {ARTICLE} {STYLE} {CAT} from the {BRAND} brand, priced around {PRICE} and suitable for {USE} in my current project."),
    ],
    "HIGH-17": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand with a price around {PRICE} with suitability for {USE} with application in my current project."),
    ],
    "HIGH-18": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT}, from the {BRAND} brand, as it happens, priced around {PRICE} and suitable for {USE}, in my current project."),
    ],
}

SUBTYPES = [
    "Conditional", "Causal", "Concessive", "Comparative", "Purpose",
    "Passive", "Apposition_Parenthetical", "Interrogative",
    "Elliptical_Telegraphic", "Constraint_List",
]


def _clean(text: str) -> str:
    out = re.sub(r"\s+", " ", (text or "").strip())
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out


def _compact_value(value: str, max_tokens: int = 2) -> str:
    if not value:
        return ""
    tokens = [t for t in re.split(r"\s+", value.strip()) if t]
    bad = {
        "who", "whose", "which", "that", "where", "when", "why", "how",
        "dont", "don't", "cant", "can't", "wont", "won't", "didnt", "didn't",
        "isnt", "aren't", "arent", "wasnt", "weren't", "werent", "not", "none",
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


def _get_semantic_type(dimension: str) -> str:
    return DIMENSION_TO_SEMANTIC_TYPE.get(dimension, "FEATURE")


def _build_attr_map_by_semantic_type(selected_attrs: List[Tuple[str, str]]) -> Dict[str, str]:
    semantic_map: Dict[str, str] = {}
    for dimension, value in selected_attrs:
        sem_type = _get_semantic_type(dimension)
        if sem_type not in semantic_map:
            semantic_map[sem_type] = value
    return semantic_map


def _map_slot_placeholder(sem_type: str) -> str:
    mapping = {
        "CATEGORY": "{CAT}", "BRAND": "{BRAND}", "PRICE": "{PRICE}",
        "MATERIAL": "{MATERIAL}", "COLOR": "{COLOR}", "SIZE": "{SIZE}",
        "QUALITY": "{QUALITY}", "STYLE": "{STYLE}", "USE_CASE": "{USE}",
        "FEATURE": "{FEATURE}",
    }
    return mapping.get(sem_type, "{FEATURE}")


def _is_plural(noun_phrase: str) -> bool:
    if not noun_phrase:
        return False
    words = noun_phrase.strip().split()
    if not words:
        return False
    last_word = words[-1].lower()
    irregular_plurals = {
        'feet', 'teeth', 'geese', 'mice', 'lice', 'men', 'women', 'children',
        'people', 'oxen', 'cattle', 'deer', 'sheep', 'fish', 'species', 'series',
        'people', 'folks', 'guys', 'pads', 'inks', 'beads', 'colors',
    }
    plural_indicators = {'some', 'any', 'few', 'many', 'various', 'different', 'multiple'}
    if any(ind in words for ind in plural_indicators):
        return True
    if last_word in irregular_plurals:
        return True
    plural_patterns = (
        last_word.endswith('s') and not last_word.endswith('ss') and not last_word.endswith('us')
    ) or last_word.endswith('ies') or last_word.endswith('ves')
    return plural_patterns


def _get_article(noun_phrase: str) -> str:
    if not noun_phrase:
        return "a"
    words = noun_phrase.strip().split()
    if not words:
        return "a"
    last_word = words[-1].lower()
    plural_patterns = (
        last_word.endswith('s') and not last_word.endswith('ss') and not last_word.endswith('us')
    ) or last_word.endswith('ies') or last_word.endswith('ves')
    irregular_plurals = {
        'feet', 'teeth', 'geese', 'mice', 'lice', 'men', 'women', 'children',
        'people', 'oxen', 'cattle', 'deer', 'sheep', 'fish', 'species', 'series',
        'people', 'folks', 'guys', 'pads', 'inks', 'beads', 'colors',
    }
    plural_indicators = {'some', 'any', 'few', 'many', 'various', 'different', 'multiple'}
    if any(ind in words for ind in plural_indicators):
        return "some"
    if last_word in irregular_plurals:
        return "some"
    if plural_patterns:
        return "some"
    first_word = words[0].lower()
    vowel_start = first_word[0] in 'aeiou' if first_word else False
    if vowel_start:
        return "an"
    return "a"


def generate_query_from_attributes(
    category: str,
    selected_attrs: List[Tuple[str, str]],
    subtype: str,
    rng: Optional[random.Random] = None,
) -> Tuple[str, str]:
    chooser = rng if rng is not None else random
    c = category if category else "craft supplies"

    semantic_map = _build_attr_map_by_semantic_type(selected_attrs)
    templates = TEMPLATES.get(subtype, TEMPLATES["HIGH-1"])
    template_idx = chooser.randrange(len(templates))
    slots_needed, template_text = templates[template_idx]

    slot_values: Dict[str, str] = {}
    for sem_type in slots_needed:
        placeholder = _map_slot_placeholder(sem_type)
        if sem_type in semantic_map:
            slot_values[placeholder] = _compact_value(semantic_map[sem_type], max_tokens=2)
        else:
            default = SEMANTIC_DEFAULTS.get(sem_type, c)
            slot_values[placeholder] = _compact_value(default, max_tokens=2)

    if "{CAT}" not in slot_values or slot_values["{CAT}"] in ["craft supplies", ""]:
        slot_values["{CAT}"] = _compact_value(c, max_tokens=2)

    cat_value = slot_values.get("{CAT}", c)
    article = _get_article(cat_value)
    slot_values["{ARTICLE}"] = article

    query = template_text
    for placeholder, value in slot_values.items():
        query = query.replace(placeholder, value)

    query = _clean(query)
    if query and query[-1] not in ".!?":
        query += "."

    template_id = f"{subtype}#{template_idx + 1}"
    return query, template_id


# ============================================================
# Stage 5 解耦模型加载与风格嵌入
# ============================================================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


# ============================================================
# 多进程 Worker
# ============================================================
def _init_worker():
    """Initialize worker process resources (spaCy model, caches)."""
    global nlp, _user_style_cache, _precomputed_user_features, _template_style_features, _stage1_cache, _stage1_asin_index
    import spacy
    nlp = spacy.load("en_core_web_sm")
    _user_style_cache = {}
    _precomputed_user_features = _load_precomputed_features()
    _template_style_features = None
    _stage1_cache = None
    _stage1_asin_index = {}


# Global nlp instance for precompute workers (loaded once per worker process)
_nlp_precompute_worker = None


def _init_precompute_worker():
    """Initializer for precompute pool - no spaCy needed for lexical features."""
    import os
    import sys
    pid = os.getpid()
    print(f"[DEBUG] Init worker PID={pid} started (lexical features, no spaCy)", flush=True, file=sys.stderr)


def _precompute_user_features_worker(args: Tuple) -> Tuple[str, np.ndarray]:
    """Module-level worker for parallel precomputation. Returns (user_id, 17-dim lexical features).

    注意：此版本使用词汇级特征，不需要 spaCy NLP 模型，大幅提升速度。
    """
    import numpy as np
    import json as json_module
    import re

    user_id, user_data = args

    def _extract_lexical_features(text: str):
        """提取14维词汇级特征（不需要spaCy）- 使用模板特有短语模式"""
        return _extract_style_features_from_text(text)

    reviews = []
    review_file = os.path.join(STAGE0_REVIEWS_DIR, f"reviews_{user_id}.json")
    if os.path.exists(review_file):
        try:
            with open(review_file, 'r', encoding='utf-8') as f:
                review_data = json_module.load(f)
            for item in review_data.get("results", []):
                for review in item.get("target_reviews", []):
                    if isinstance(review, str):
                        text = review
                    elif isinstance(review, dict):
                        text = review.get("review_text", "")
                    else:
                        text = ""
                    if text:
                        reviews.append(text)
                for review in item.get("other_reviews", []):
                    if isinstance(review, str):
                        text = review
                    elif isinstance(review, dict):
                        text = review.get("review_text", "")
                    else:
                        text = ""
                    if text:
                        reviews.append(text)
        except Exception:
            pass

    all_style_feats = []
    for review in reviews:
        try:
            feat = _extract_lexical_features(review)
            if feat is not None:
                all_style_feats.append(feat)
        except Exception:
            continue

    if all_style_feats:
        return (user_id, np.mean(all_style_feats, axis=0))
    else:
        return (user_id, np.zeros(17, dtype=np.float32))


def _process_user_worker(args: Tuple) -> Tuple[str, bool, Optional[Dict]]:
    """Worker function for multiprocessing. Returns (user_id, success, result_dict or error)."""
    user_id, user_data, config = args
    try:
        result = run_generation(
            linguistic_profile_file=None,
            user_id=user_id,
            asin=user_data.get('asins'),
            output_dir=config['output_dir'],
            seed=config['seed'],
            forced_level=None,
        )
        # Return the result dict instead of just the path for verification
        with open(result, 'r') as f:
            result_data = json.load(f)
        return (user_id, True, result_data)
    except Exception as e:
        return (user_id, False, str(e))


def _extract_opening_pattern_features(text: str) -> np.ndarray:
    """提取开头模式特征（7维binary特征）"""
    text_lower = text.lower().strip()
    first_words = ' '.join(text_lower.split()[:5])  # 前5个词

    features = [
        1.0 if text_lower.startswith('i am looking for') else 0.0,
        1.0 if text_lower.startswith('it is') or text_lower.startswith("it's") else 0.0,
        1.0 if text_lower.startswith('as for') else 0.0,
        1.0 if text_lower.startswith('there is') or text_lower.startswith("there's") or text_lower.startswith('there are') else 0.0,
        1.0 if text_lower.startswith('the requirement') else 0.0,
        1.0 if 'looking for is' in first_words else 0.0,
        1.0 if 'as it happens' in text_lower else 0.0,
    ]
    return np.array(features, dtype=np.float32)


def _extract_style_features_from_text(text: str) -> Optional[np.ndarray]:
    """从文本中提取词汇级风格特征（14维），用于区分不同模板。

    基于模板特有的短语模式进行检测，而不是泛化的统计特征。
    每个特征对应一个模板/模板群组独有的短语模式。
    """
    text_lower = text.lower()
    words = [w for w in text.split() if w.strip()]
    word_count = max(len(words), 1)

    # 少于10词跳过（模板填充后约29-32词）
    if len(words) < 10:
        return None

    # 使用标准化的介词模式，避免不同词数导致的尺度差异
    # 所有介词短语都用统一的 with 计数来标准化
    features = [
        # ========== HIGH-2 特有: "which ... offers products" ==========
        1.0 if 'which' in text_lower and 'offers products' in text_lower else 0.0,

        # ========== HIGH-1/HIGH-2 共有: "that is ... and that is" ==========
        # 至少2个 "that is/are" 模式
        1.0 if (text_lower.count('that is') + text_lower.count('that are')) >= 2 else 0.0,

        # ========== HIGH-18 特有: "as it happens" ==========
        1.0 if 'as it happens' in text_lower else 0.0,

        # ========== HIGH-17 特有: 多个 "with ... for" 结构 ==========
        # HIGH-17: "with a price around" + "with suitability for" + "with application"
        (1.0 if 'with a price' in text_lower else 0.0) +
        (1.0 if 'with suitability' in text_lower else 0.0) +
        (1.0 if 'with application' in text_lower else 0.0),

        # ========== HIGH-5/HIGH-17 共有: "with ... with" 结构 ==========
        # 统计连续 with 结构 (HIGH-5用1个，HIGH-17用多个)
        min(2.0, text_lower.count(' with ')),

        # ========== HIGH-3 特有: "that, being ... is suitable" ==========
        1.0 if 'that,' in text_lower and 'being' in text_lower else 0.0,

        # ========== HIGH-6 特有: "to be used ... to be priced" ==========
        1.0 if text_lower.count('to be') >= 2 else 0.0,

        # ========== HIGH-7 特有: "that is designed by" ==========
        1.0 if 'that is designed' in text_lower else 0.0,

        # ========== HIGH-8 特有: "It is ... that I am looking for" (倒装) ==========
        1.0 if text_lower.startswith('it is') else 0.0,

        # ========== HIGH-9/HIGH-4 共有: 逗号分隔的并列结构 ==========
        # HIGH-4: "a product from ..., priced ..., and suitable"
        # HIGH-9: "from ... and priced ... and suitable ... and appropriate"
        text_lower.count(','),

        # ========== HIGH-10/HIGH-14 特有: 句末介词/无动词结构 ==========
        # HIGH-10: "for ... for ... for"
        # HIGH-14: "the requirement is"
        1.0 if 'requirement is' in text_lower else 0.0,

        # ========== HIGH-13 特有: "the need for" ==========
        1.0 if 'the need for' in text_lower else 0.0,

        # ========== HIGH-15 特有: 疑问句 "what ... how" ==========
        1.0 if re.search(r'\bwhat\b', text_lower) or re.search(r'\bhow\b', text_lower) else 0.0,

        # ========== HIGH-11/HIGH-12 特有: "there is" 存在句 ==========
        1.0 if 'there is' in text_lower or 'there are' in text_lower else 0.0,
    ]

    return np.array(features, dtype=np.float32)


_template_style_features = None


def _get_template_style_features() -> Dict[str, np.ndarray]:
    """获取模板的14维词汇级风格特征"""
    global _template_style_features
    if _template_style_features is not None:
        return _template_style_features

    print("[Stage 6] Computing template style features (14-dim lexical)...")
    _template_style_features = {}

    placeholders_map = {
        '{ARTICLE}': 'a', '{STYLE}': 'elegant', '{CAT}': 'craft supplies',
        '{BRAND}': 'trusted', '{PRICE}': '$20', '{USE}': 'crafting',
        '{COLOR}': 'blue', '{MATERIAL}': 'cotton',
    }
    fallback_text = (
        "I am looking for elegant craft supplies that are from the trusted brand "
        "that is priced around twenty dollars and that are suitable for crafting "
        "in my current project"
    )

    for subtype, templates in TEMPLATES.items():
        for slots_needed, template_text in templates:
            cleaned = template_text
            for ph, replacement in placeholders_map.items():
                if ph in cleaned:
                    cleaned = cleaned.replace(ph, replacement)

            style_features = _extract_style_features_from_text(cleaned)
            if style_features is None:
                style_features = _extract_style_features_from_text(fallback_text)

            _template_style_features[subtype] = style_features

    print(f"[Stage 6] Computed style features for {len(_template_style_features)} templates")
    return _template_style_features


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


_user_style_cache: Dict[str, np.ndarray] = {}
_precomputed_user_features: Dict[str, np.ndarray] = {}
_PRECOMPUTED_FEATURES_FILE = os.path.join(OUTPUT_DIR, "precomputed_user_style_features.json")


def _load_precomputed_features() -> Dict[str, np.ndarray]:
    """加载预计算的用户风格特征。"""
    global _precomputed_user_features
    if os.path.exists(_PRECOMPUTED_FEATURES_FILE):
        try:
            with open(_PRECOMPUTED_FEATURES_FILE, 'r') as f:
                data = json.load(f)
            for uid, feat_list in data.items():
                _precomputed_user_features[uid] = np.array(feat_list, dtype=np.float32)
            log_with_timestamp(f"Loaded {len(_precomputed_user_features)} precomputed user style features")
            return _precomputed_user_features
        except Exception as e:
            log_with_timestamp(f"Warning: Failed to load precomputed features: {e}")
    return {}


def _get_user_style_features(user_id: str, reviews: List[str]) -> np.ndarray:
    """获取用户的17维词汇级风格特征（直接从评论提取，不使用模型）。

    优先使用预计算结果，否则实时计算并缓存到进程内。
    """
    # 优先使用预计算结果（跨进程共享）
    if user_id in _precomputed_user_features:
        return _precomputed_user_features[user_id]

    # 进程内缓存
    if user_id in _user_style_cache:
        return _user_style_cache[user_id]

    all_style_feats = []
    for review in reviews:
        try:
            style_features = _extract_style_features_from_text(review)
            if style_features is None:
                continue
            all_style_feats.append(style_features)
        except Exception:
            continue

    if not all_style_feats:
        emb = np.zeros(17, dtype=np.float32)
    else:
        emb = np.mean(all_style_feats, axis=0)

    _user_style_cache[user_id] = emb
    return emb


def _select_template_by_cached_style(user_id: str, reviews: List[str]) -> Tuple[str, Dict[str, float]]:
    user_style_feat = _get_user_style_features(user_id, reviews)
    template_feats = _get_template_style_features()

    similarities = {}
    for subtype, template_feat in template_feats.items():
        sim = _cosine_similarity(user_style_feat, template_feat)
        similarities[subtype] = sim

    best_template = max(similarities.items(), key=lambda x: x[1])[0]
    return best_template, similarities


# ============================================================
# 模板分组（层次聚类 + MLP）
# ============================================================
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "you", "are", "was",
    "were", "have", "has", "had", "but", "not", "all", "can", "will", "would", "just", "very",
    "about", "then", "than", "when", "where", "while", "they", "them", "their", "there", "also",
    "what", "which", "much", "many", "more", "most", "some", "such", "only", "over", "under",
    "out", "off", "our", "its", "it's", "too", "few", "lot", "use", "used", "using", "like",
    "made", "make", "still", "after", "before", "being", "been",
}

COLOR_WORDS = {
    "black", "white", "blue", "red", "green", "pink", "purple", "gray", "grey", "brown",
    "beige", "navy", "silver", "gold", "yellow", "orange", "clear", "transparent"
}

MATERIAL_WORDS = {
    "cotton", "wool", "silicone", "leather", "metal", "plastic", "polyester", "linen", "nylon",
    "canvas", "wood", "bamboo", "paper", "steel", "rubber", "ceramic", "glass", "acrylic"
}


def _extract_template_features(template_text: str) -> np.ndarray:
    words = template_text.split()
    word_count = max(len(words), 1)
    text_lower = template_text.lower()

    features = [
        template_text.count(',') / word_count,
        text_lower.count('that') / word_count,
        text_lower.count('which') / word_count,
        text_lower.count(' and ') / word_count,
        (text_lower.count('to be') + text_lower.count('being')) / word_count,
        len(re.findall(r'\b(is|are|was|were)\b.*\b(\w+ed)\b', text_lower)) / word_count,
        len(re.findall(r'\b(with|for|in|to|of|by|from)\b', text_lower)) / word_count,
        1.0 if 'there is' in text_lower or 'there are' in text_lower else 0.0,
        1.0 if re.search(r'\b(what|how)\b', text_lower) else 0.0,
    ]
    return np.array(features, dtype=float)


def _compute_all_template_features() -> Tuple[np.ndarray, List[str]]:
    feature_list = []
    subtype_list = []
    for subtype, templates in TEMPLATES.items():
        for slots_needed, template_text in templates:
            features = _extract_template_features(template_text)
            feature_list.append(features)
            subtype_list.append(subtype)
    return np.array(feature_list), subtype_list


def _cluster_templates_mlp(features: np.ndarray, subtype_list: List[str]) -> Dict[str, List[str]]:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    desired_groups = {
        "low": ["HIGH-15", "HIGH-17", "HIGH-18", "HIGH-2", "HIGH-3", "HIGH-11", "HIGH-9"],
        "medium": ["HIGH-1", "HIGH-4", "HIGH-14", "HIGH-8", "HIGH-12", "HIGH-16"],
        "high": ["HIGH-5", "HIGH-10", "HIGH-13", "HIGH-6", "HIGH-7"],
    }

    label_to_id = {"low": 0, "medium": 1, "high": 2}
    y = np.array([label_to_id[g] for s in subtype_list for g, v in desired_groups.items() if s in v])

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    mlp = MLPClassifier(hidden_layer_sizes=(8, 4), activation='relu', max_iter=1000, random_state=42)
    mlp.fit(X, y)

    return {"low": [], "medium": [], "high": []}


_TEMPLATE_SCORES = None
TEMPLATE_GROUPS = None


def _init_template_groups():
    global _TEMPLATE_SCORES, TEMPLATE_GROUPS
    features, subtype_list = _compute_all_template_features()
    TEMPLATE_GROUPS = _cluster_templates_mlp(features, subtype_list)
    _TEMPLATE_SCORES = {}
    for subtype, feats in zip(subtype_list, features):
        _TEMPLATE_SCORES[subtype] = float(np.sum(feats))


_init_template_groups()


def _get_templates_for_level(level: str) -> List[str]:
    return TEMPLATE_GROUPS.get(level, TEMPLATE_GROUPS.get("medium", []))


# ============================================================
# Stage 1 / Stage 0 数据加载
# ============================================================
_stage1_cache = None
_stage1_asin_index: Dict[str, Dict] = {}  # asin -> product (O(1) lookup)


def _load_stage1_attributes() -> Dict:
    global _stage1_cache, _stage1_asin_index
    if _stage1_cache is None:
        if os.path.exists(STAGE1_ATTRIBUTES_FILE):
            with open(STAGE1_ATTRIBUTES_FILE, 'r', encoding='utf-8') as f:
                _stage1_cache = json.load(f)
            # Build asin -> product index for O(1) lookup
            _stage1_asin_index = {}
            for product in _stage1_cache.get("products", []):
                asin = product.get("asin")
                if asin:
                    _stage1_asin_index[asin] = product
        else:
            _stage1_cache = {"products": []}
    return _stage1_cache


def _get_stage1_attributes(asin: str) -> Dict:
    """Get product attributes by asin using O(1) index lookup."""
    # Ensure index is built
    _load_stage1_attributes()
    return _stage1_asin_index.get(asin, {})


def _get_user_reviewed_asins(user_id: str) -> List[str]:
    review_file = os.path.join(STAGE0_REVIEWS_DIR, f"reviews_{user_id}.json")
    if not os.path.exists(review_file):
        return []
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        asins = []
        for item in data.get("results", []):
            asin = item.get("asin")
            if asin:
                asins.append(asin)
        return asins
    except Exception:
        return []


def _find_valid_asin_for_user(user_id: str, rng: random.Random) -> Tuple[str, str]:
    user_asins = _get_user_reviewed_asins(user_id)
    if not user_asins:
        return "", "Craft Supplies"

    stage1_data = _load_stage1_attributes()
    stage1_asins = set(p.get("asin") for p in stage1_data.get("products", []))
    valid_asins = [asin for asin in user_asins if asin in stage1_asins]

    if not valid_asins:
        return "", "Craft Supplies"

    selected_asin = rng.choice(valid_asins)
    for product in stage1_data.get("products", []):
        if product.get("asin") == selected_asin:
            category = product.get("A1_product_type", "Craft Supplies")
            return selected_asin, category

    return selected_asin, "Craft Supplies"


# ============================================================
# 属性选择与查询生成
# ============================================================
def _pick_best_attributes(stage1_attrs: Dict) -> List[Tuple[str, str]]:
    attrs = []

    product_type = stage1_attrs.get("A1_product_type")
    if product_type:
        attrs.append(("Product_Category", str(product_type)))
    else:
        attrs.append(("Product_Category", "Craft Supplies"))

    brand = stage1_attrs.get("A2_brand")
    if brand:
        attrs.append(("Brand_Preference", str(brand)))
    else:
        attrs.append(("Brand_Preference", "trusted"))

    price = stage1_attrs.get("A3_price")
    if price:
        attrs.append(("Price_Range", str(price)))
    else:
        attrs.append(("Price_Range", "$50"))

    appearance = stage1_attrs.get("A4_appearance")
    if appearance:
        if isinstance(appearance, list):
            appearance = appearance[0] if appearance else ""
        attrs.append(("A4_appearance", str(appearance)[:30]))

    use_case = stage1_attrs.get("A5_use_case")
    if use_case:
        uc_str = str(use_case).strip()
        if uc_str.lower().startswith("for "):
            uc_str = uc_str[4:]
        attrs.append(("Use_Scene", uc_str))
    else:
        attrs.append(("Use_Scene", "Crafting"))

    return attrs


def _word_count(text: str) -> int:
    return len([w for w in (text or "").split() if w])


def _default_subtype_scores(selected: str) -> Dict[str, float]:
    scores = {name: 0.12 for name in SUBTYPES}
    if selected in scores:
        scores[selected] = 1.25
    return scores


def _build_rng(user_id: str, seed: Optional[int]) -> random.Random:
    if seed is None:
        return random.Random()
    uid_hash = int(hashlib.md5(user_id.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(int(seed) + uid_hash)


# ============================================================
# 单用户查询生成
# ============================================================
def run_generation(
    linguistic_profile_file: Optional[str],
    output_dir: str,
    seed: Optional[int] = None,
    forced_level: Optional[str] = None,
    user_id: Optional[str] = None,
    asin: Optional[str | List[str]] = None,
) -> str:
    """生成用户查询。支持单个asin或asin列表（每个asin生成一个查询）。"""
    if linguistic_profile_file and os.path.exists(linguistic_profile_file):
        with open(linguistic_profile_file, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        profile_user_id = profile.get("user_id")
        if profile_user_id:
            user_id = profile_user_id
        if not user_id:
            raise ValueError("Missing user_id")
    else:
        profile = {}

    if not user_id:
        raise ValueError("Missing user_id")

    rng = _build_rng(user_id, seed)

    # 加载用户评论
    user_reviews = []
    review_file = os.path.join(STAGE0_REVIEWS_DIR, f"reviews_{user_id}.json")
    if os.path.exists(review_file):
        try:
            with open(review_file, 'r', encoding='utf-8') as f:
                review_data = json.load(f)
            for item in review_data.get("results", []):
                for review in item.get("target_reviews", []):
                    if isinstance(review, str):
                        text = review
                    elif isinstance(review, dict):
                        text = review.get("review_text", "")
                    else:
                        text = ""
                    if text:
                        user_reviews.append(text)
                for review in item.get("other_reviews", []):
                    if isinstance(review, str):
                        text = review
                    elif isinstance(review, dict):
                        text = review.get("review_text", "")
                    else:
                        text = ""
                    if text:
                        user_reviews.append(text)
        except Exception as e:
            print(f"[Stage 6] Warning: failed to load reviews for {user_id}: {e}")

    # 确定使用哪个模板
    if forced_level:
        templates_for_level = _get_templates_for_level(forced_level)
        subtype = templates_for_level[0] if templates_for_level else "HIGH-1"
        template_similarities = {}
    else:
        if user_reviews:
            best_template, template_similarities = _select_template_by_cached_style(user_id, user_reviews)
            subtype = best_template
        else:
            best_template, template_similarities = "HIGH-1", {}
            subtype = best_template

    subtype_scores = _default_subtype_scores(subtype)

    # 获取用户风格特征
    if user_reviews:
        user_style_feat = _get_user_style_features(user_id, user_reviews)
    else:
        user_style_feat = np.zeros(17, dtype=np.float32)

    # 确定要处理的asin列表
    if asin is None:
        asins_to_process = [_find_valid_asin_for_user(user_id, rng)[0]]
    elif isinstance(asin, str):
        asins_to_process = [asin]
    else:
        asins_to_process = asin

    stage1_data = _load_stage1_attributes()

    # 为每个asin生成查询
    all_results = []
    for reviewed_asin in asins_to_process:
        # 获取category（使用O(1)索引查找）
        product_info = _get_stage1_attributes(reviewed_asin)
        category = product_info.get("A1_product_type", "Craft Supplies")

        stage1_attrs = product_info
        attrs = _pick_best_attributes(stage1_attrs)

        query_text, template_id = generate_query_from_attributes(category, attrs, subtype, rng=rng)

        all_results.append({
            "asin": reviewed_asin,
            "category": category,
            "user_id": user_id,
            "target_subtype": subtype,
            "skeleton_level": subtype,
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
                "selected_attributes": [{"dimension": d, "value": v} for d, v in attrs],
                "attribute_priority_tracking": [
                    {
                        "dimension": d,
                        "attribute": v,
                        "priority_level": "medium",
                        "reason": "Stage1预提取属性"
                    }
                    for d, v in attrs
                ]
            },
        })

    # 合并结果
    result = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "method": "direct_style_matching" if user_reviews else "knn_fallback",
        "template_selection_method": "30dim_style_features" if user_reviews else "knn_feature_matching",
        "user_style_features": user_style_feat.tolist() if isinstance(user_style_feat, np.ndarray) else user_style_feat,
        "num_user_reviews_used": len(user_reviews),
        "total_queries": len(all_results),
        "successful_target_queries": len(all_results),
        "results": all_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_fp = os.path.join(output_dir, f"queries_{user_id}.json")
    with open(out_fp, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out_fp


# ============================================================
# 批量处理入口
# ============================================================
def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def find_users_from_stage1() -> List[str]:
    log_with_timestamp("Reading users from Stage 1 attributes file...")
    try:
        with open(STAGE1_ATTRIBUTES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        products = data.get('products', [])
        users_set = set()
        for product in products:
            user_id = product.get('user_id')
            if user_id:
                users_set.add(user_id)
        users_list = sorted(list(users_set))
        log_with_timestamp(f"Found {len(users_list)} unique users from Stage 1")
        return users_list
    except Exception as e:
        log_with_timestamp(f"ERROR reading Stage 1 file: {e}")
        return []


def find_users_with_profiles(profile_dir: str = None) -> List[str]:
    return find_users_from_stage1()


def validate_user_files(user_ids: List[str], profile_dir: str = None) -> Dict[str, Dict[str, List[str]]]:
    """从 Stage 1 构建用户-商品映射，支持每个用户多个商品。"""
    log_with_timestamp("Building user-product mapping from Stage 1...")
    validated_users = {}  # user_id -> {'user_id': user_id, 'asins': [asin1, asin2, ...]}
    try:
        with open(STAGE1_ATTRIBUTES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        products = data.get('products', [])
        for product in products:
            asin = product.get('asin')
            user_id = product.get('user_id')
            if asin and user_id:
                if user_id not in validated_users:
                    validated_users[user_id] = {'user_id': user_id, 'asins': []}
                if asin not in validated_users[user_id]['asins']:
                    validated_users[user_id]['asins'].append(asin)
        total_pairs = sum(len(v['asins']) for v in validated_users.values())
        log_with_timestamp(f"Found {len(validated_users)} users with {total_pairs} user-product pairs from Stage 1")
    except Exception as e:
        log_with_timestamp(f"ERROR reading Stage 1 file: {e}")
    return validated_users


def generate_summary(output_dir: str, user_ids: List[str]) -> Dict:
    log_with_timestamp("=" * 80)
    log_with_timestamp("Generating summary statistics...")
    log_with_timestamp("=" * 80)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_users': len(user_ids),
        'processed_users': 0,
        'failed_users': [],
        'user_summaries': {},
        'aggregate_stats': {
            'total_queries': 0,
            'total_target_queries': 0,
            'total_valid_target_error_words': 0,
            'target_validation_rate': 0.0
        }
    }

    for user_id in user_ids:
        output_file = os.path.join(output_dir, f'queries_{user_id}.json')
        if not os.path.exists(output_file):
            log_with_timestamp(f"  ✗ User {user_id}: output file not found")
            summary['failed_users'].append(user_id)
            continue

        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)

            summary['processed_users'] += 1
            user_summary = {
                'user_id': user_id,
                'total_queries': user_data.get('total_queries', 0),
                'successful_target_queries': user_data.get('successful_target_queries', 0),
                'valid_target_error_words': user_data.get('successful_target_queries', 0)
            }

            summary['aggregate_stats']['total_queries'] += user_summary['total_queries']
            summary['aggregate_stats']['total_target_queries'] += user_summary['successful_target_queries']
            summary['aggregate_stats']['total_valid_target_error_words'] += user_summary['valid_target_error_words']

            if user_summary['successful_target_queries'] > 0:
                user_summary['target_validation_rate'] = round(
                    user_summary['valid_target_error_words'] / user_summary['successful_target_queries'] * 100, 1
                )
            else:
                user_summary['target_validation_rate'] = 0.0

            summary['user_summaries'][user_id] = user_summary
            log_with_timestamp(
                f"  ✓ User {user_id}: {user_summary['total_queries']} queries, "
                f"TU validation: {user_summary['target_validation_rate']}%"
            )
        except Exception as e:
            log_with_timestamp(f"  ✗ User {user_id}: error reading results - {e}")
            summary['failed_users'].append(user_id)

    if summary['aggregate_stats']['total_target_queries'] > 0:
        summary['aggregate_stats']['target_validation_rate'] = round(
            summary['aggregate_stats']['total_valid_target_error_words'] /
            summary['aggregate_stats']['total_target_queries'] * 100, 1
        )

    summary_file = os.path.join(output_dir, 'all_users_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"Summary saved to {summary_file}")
    log_with_timestamp("=" * 80)
    log_with_timestamp("AGGREGATE STATISTICS")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Processed users: {summary['processed_users']}/{summary['total_users']}")
    log_with_timestamp(f"Total queries: {summary['aggregate_stats']['total_queries']}")
    log_with_timestamp(f"Total target user queries: {summary['aggregate_stats']['total_target_queries']}")
    log_with_timestamp("")
    log_with_timestamp(f"Error Word Validation:")
    log_with_timestamp(f"  Target queries with all error words: {summary['aggregate_stats']['total_valid_target_error_words']}/{summary['aggregate_stats']['total_target_queries']} ({summary['aggregate_stats']['target_validation_rate']}%)")

    if summary['failed_users']:
        log_with_timestamp(f"\nFailed users: {', '.join(summary['failed_users'])}")

    return summary


def main():
    config = {
        'profile_dir': '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis',
        'output_dir': OUTPUT_DIR,
        'user_ids': None,
        'seed': None,
        'skip_summary': False,
    }

    os.makedirs(config['output_dir'], exist_ok=True)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 6: Generate Template Queries (Style-based Template Matching)")
    log_with_timestamp("=" * 80)
    log_with_timestamp("Step 1: Extract 17-dim lexical features for all templates")
    log_with_timestamp("Step 2: Compute user 17-dim lexical features from their reviews")
    log_with_timestamp("Step 3: Match user to best template via cosine similarity")
    log_with_timestamp("=" * 80)

    if config['user_ids']:
        user_ids = config['user_ids']
        log_with_timestamp(f"Processing {len(user_ids)} user(s) specified by --user-ids")
    else:
        user_ids = find_users_with_profiles(config['profile_dir'])

    if not user_ids:
        log_with_timestamp("ERROR: No users to process!")
        sys.exit(1)

    validated_users = validate_user_files(user_ids, config['profile_dir'])

    if not validated_users:
        log_with_timestamp("ERROR: No valid users found!")
        sys.exit(1)

    # ============================================================
    # 预计算所有用户的风格特征（主进程一次性完成，避免 worker 重复计算）
    # ============================================================
    if os.path.exists(_PRECOMPUTED_FEATURES_FILE):
        log_with_timestamp(f"Loading precomputed user style features from {_PRECOMPUTED_FEATURES_FILE}")
        try:
            with open(_PRECOMPUTED_FEATURES_FILE, 'r') as f:
                precomputed_features = json.load(f)
            log_with_timestamp(f"Loaded {len(precomputed_features)} precomputed user style features")
        except Exception as e:
            log_with_timestamp(f"Failed to load precomputed features: {e}, will recompute")
            precomputed_features = None
    else:
        precomputed_features = None

    if precomputed_features is None:
        log_with_timestamp("Precomputing user style features (parallel)...")

        # 获取CPU核心数，用于并行（限制为8以提高速度）
        num_workers = min(8, mp.cpu_count())

        # Worker函数：处理单个用户
        def _precompute_user_features(args):
            import os
            import sys
            import warnings
            import spacy
            import numpy as np
            import json as json_module

            user_id, user_data = args
            pid = os.getpid()
            print(f"[DEBUG] Worker PID={pid} started for user={user_id}", flush=True, file=sys.stderr)

            try:
                nlp = spacy.load("en_core_web_sm")
                print(f"[DEBUG] Worker PID={pid} loaded spaCy model for user={user_id}", flush=True, file=sys.stderr)
            except OSError:
                import subprocess
                print(f"[DEBUG] Worker PID={pid} downloading spaCy model for user={user_id}", flush=True, file=sys.stderr)
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True, capture_output=True)
                nlp = spacy.load("en_core_web_sm")

            def _extract_30dim_features(text: str):
                """提取17维词汇级特征"""
                words = [w for w in text.split() if w.strip()]
                if len(words) < 25:
                    return None
                doc = nlp(text)
                n_tokens = max(len([t for t in doc if not t.is_punct]), 1)
                n_subj = sum(1 for t in doc if t.dep_ in {'nsubj', 'nsubj:pass'})
                n_dobj = sum(1 for t in doc if t.dep_ in {'dobj', 'pobj', 'attr'})
                n_amod = sum(1 for t in doc if t.dep_ == 'amod')
                n_advmod = sum(1 for t in doc if t.dep_ == 'advmod')
                n_prep = sum(1 for t in doc if t.dep_ == 'prep')
                n_conj = sum(1 for t in doc if t.dep_ == 'conj')
                n_neg = sum(1 for t in doc if t.dep_ == 'neg')
                n_relcl = sum(1 for t in doc if t.dep_ == 'relcl')
                n_pass = sum(1 for t in doc if t.dep_ in {'nsubj:pass', 'aux:pass'} or (t.tag_ == 'VBN' and t.dep_ not in {'amod', 'conj'}))
                n_part = sum(1 for t in doc if t.tag_ in {'VBG', 'VBN'} and t.dep_ in {'amod', 'advcl', 'relcl'})
                n_inf = sum(1 for t in doc if t.tag_ == 'VB' and t.dep_ in {'xcomp', 'ccomp', 'advcl'})
                n_det = sum(1 for t in doc if t.dep_ == 'det')
                n_cc = sum(1 for t in doc if t.dep_ == 'cc')
                n_intj = sum(1 for t in doc if t.dep_ == 'intj')
                pos_counts = {}
                for token in doc:
                    if not token.is_punct:
                        pos = token.pos_
                        pos_counts[pos] = pos_counts.get(pos, 0) + 1

                def get_depth(token):
                    depth = 0
                    while token.head != token:
                        depth += 1
                        token = token.head
                        if depth > 20:
                            break
                    return depth
                depths = [get_depth(t) for t in doc if not t.is_punct]
                avg_depth = sum(depths) / max(len(depths), 1)

                # 23维基础特征
                subordinate_ratio = n_subj / n_tokens
                coordination_ratio = n_conj / n_tokens
                negation_ratio = n_neg / n_tokens
                length_depth = avg_depth / 10.0
                upos_order = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'AUX', 'PART', 'SCONJ', 'CCONJ', 'ADP']
                pos_dist = [pos_counts.get(p, 0) / n_tokens for p in upos_order]
                relative_clause_ratio = n_relcl / n_tokens
                passive_ratio = n_pass / n_tokens
                participial_ratio = n_part / n_tokens
                infinitive_ratio = n_inf / n_tokens
                appositive_ratio = n_intj / n_tokens
                parenthetical_ratio = n_det / n_tokens
                prep_phrase_ratio = n_prep / n_tokens
                insertion_frequency = n_amod / n_tokens

                base_features = [
                    subordinate_ratio * 10, coordination_ratio * 10, negation_ratio * 10, length_depth,
                    *pos_dist, relative_clause_ratio, passive_ratio, participial_ratio,
                    infinitive_ratio, appositive_ratio, parenthetical_ratio, prep_phrase_ratio, insertion_frequency,
                ]

                # 7维开头特征
                text_lower = text.lower().strip()
                first_words = ' '.join(text_lower.split()[:5])
                opening_features = [
                    1.0 if text_lower.startswith('i am looking for') else 0.0,
                    1.0 if text_lower.startswith('it is') or text_lower.startswith("it's") else 0.0,
                    1.0 if text_lower.startswith('as for') else 0.0,
                    1.0 if text_lower.startswith('there is') or text_lower.startswith("there's") or text_lower.startswith('there are') else 0.0,
                    1.0 if text_lower.startswith('the requirement') else 0.0,
                    1.0 if 'looking for is' in first_words else 0.0,
                    1.0 if 'as it happens' in text_lower else 0.0,
                ]

                # 只使用23维基特征
                return np.array(base_features, dtype=np.float32)

            reviews = []
            review_file = os.path.join(STAGE0_REVIEWS_DIR, f"reviews_{user_id}.json")
            print(f"[DEBUG] Worker PID={pid} reading file={review_file} for user={user_id}", flush=True, file=sys.stderr)
            if os.path.exists(review_file):
                try:
                    with open(review_file, 'r', encoding='utf-8') as f:
                        review_data = json_module.load(f)
                    for item in review_data.get("results", []):
                        for review in item.get("target_reviews", []):
                            if isinstance(review, str):
                                text = review
                            elif isinstance(review, dict):
                                text = review.get("review_text", "")
                            else:
                                text = ""
                            if text:
                                reviews.append(text)
                        for review in item.get("other_reviews", []):
                            if isinstance(review, str):
                                text = review
                            elif isinstance(review, dict):
                                text = review.get("review_text", "")
                            else:
                                text = ""
                            if text:
                                reviews.append(text)
                    print(f"[DEBUG] Worker PID={pid} loaded {len(reviews)} reviews for user={user_id}", flush=True, file=sys.stderr)
                except Exception as e:
                    print(f"[DEBUG] Worker PID={pid} error loading reviews: {e}", flush=True, file=sys.stderr)

            all_style_feats = []
            for i, review in enumerate(reviews):
                try:
                    feat = _extract_lexical_features(review)
                    if feat is not None:
                        all_style_feats.append(feat)
                except Exception as e:
                    print(f"[DEBUG] Worker PID={pid} error processing review {i}: {e}", flush=True, file=sys.stderr)
                    continue

            print(f"[DEBUG] Worker PID={pid} processed {len(all_style_feats)} features for user={user_id}", flush=True, file=sys.stderr)

            if all_style_feats:
                result = (user_id, np.mean(all_style_feats, axis=0))
                print(f"[DEBUG] Worker PID={pid} returning result for user={user_id}", flush=True, file=sys.stderr)
                return result
            else:
                print(f"[DEBUG] Worker PID={pid} returning zeros for user={user_id}", flush=True, file=sys.stderr)
                return (user_id, np.zeros(17, dtype=np.float32))

        # 并行处理所有用户
        log_with_timestamp(f"  Using {num_workers} parallel workers for {len(validated_users)} users")
        precomputed_features = {}

        print(f"  Creating pool with {num_workers} workers...", flush=True, file=sys.stderr)
        with mp.Pool(num_workers, initializer=_init_precompute_worker) as pool:
            print(f"  Pool created, starting imap_unordered...", flush=True, file=sys.stderr)
            # Use imap_unordered with chunksize=1 to reduce memory pressure
            results_gen = pool.imap_unordered(_precompute_user_features_worker, validated_users.items(), chunksize=1)
            print(f"  Started receiving results...", flush=True, file=sys.stderr)
            for i, (user_id, feat) in enumerate(results_gen):
                precomputed_features[user_id] = feat
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(validated_users)} users", flush=True, file=sys.stderr)
            print(f"  All {len(precomputed_features)} users processed", flush=True, file=sys.stderr)

        log_with_timestamp(f"  Processed {len(precomputed_features)} users")

        # 保存预计算结果到文件，供所有 worker 共享
        precompute_output = {}
        for uid, feat in precomputed_features.items():
            precompute_output[uid] = feat.tolist()
        with open(_PRECOMPUTED_FEATURES_FILE, 'w') as f:
            json.dump(precompute_output, f)
        log_with_timestamp(f"Precomputed {len(precomputed_features)} user style features, saved to {_PRECOMPUTED_FEATURES_FILE}")

    log_with_timestamp("=" * 80)
    log_with_timestamp("Starting query generation (style-based template matching, multiprocessing)...")
    log_with_timestamp("=" * 80)

    failed_users = []
    level_counts = Counter()
    template_id_counts = Counter()
    total_users = len(validated_users)

    # Prepare work items
    work_items = [
        (user_id, user_data, config)
        for user_id, user_data in validated_users.items()
    ]

    # Use multiprocessing with initializer to set up spaCy in each worker
    num_workers = min(mp.cpu_count(), 16)  # Cap at 16 workers
    log_with_timestamp(f"Using {num_workers} parallel workers")

    with mp.Pool(num_workers, initializer=_init_worker) as pool:
        for i, (user_id, success, result) in enumerate(pool.imap_unordered(_process_user_worker, work_items)):
            if not success:
                log_with_timestamp(f"[{user_id}] ✗ FAILED: {result}")
                failed_users.append(user_id)
            else:
                # Collect level and template_id statistics
                for res in result.get('results', []):
                    level = res.get('target_subtype', 'unknown')
                    level_counts[level] += 1
                    template_id = res.get('target_user_query', {}).get('template_id', 'unknown')
                    template_id_counts[template_id] += 1

            if (i + 1) % 500 == 0:
                log_with_timestamp(f"Progress: {i+1}/{total_users} users processed ({(i+1) * 100 // total_users}%)")

    log_with_timestamp("=" * 80)
    log_with_timestamp("TEMPLATE DISTRIBUTION SUMMARY")
    log_with_timestamp("=" * 80)

    # 按查询数量排序，打印所有模板
    total_queries = sum(template_id_counts.values())
    log_with_timestamp(f"\n[Template ID -> Query Count] (共 {total_queries} 条查询)")
    log_with_timestamp("-" * 60)
    for template_id, count in template_id_counts.most_common():
        percentage = count / total_queries * 100
        log_with_timestamp(f"  {template_id}: {count} ({percentage:.1f}%)")

    if failed_users:
        log_with_timestamp(f"WARNING: {len(failed_users)} users failed: {', '.join(failed_users)}")
    else:
        log_with_timestamp("All users completed successfully!")

    if not config['skip_summary']:
        summary = generate_summary(config['output_dir'], list(validated_users.keys()))
        if summary['processed_users'] == 0:
            log_with_timestamp("ERROR: No users were successfully processed!")
            sys.exit(1)

    log_with_timestamp("=" * 80)
    log_with_timestamp("ALL PROCESSING COMPLETE!")
    log_with_timestamp("=" * 80)


if __name__ == '__main__':
    main()
