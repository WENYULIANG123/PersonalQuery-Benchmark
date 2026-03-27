#!/usr/bin/env python3
import re
import random
from typing import Dict, List, Optional, Tuple


# 维度到语义类型的映射
DIMENSION_TO_SEMANTIC_TYPE: Dict[str, str] = {
    "Product_Category": "CATEGORY",
    "Product_Keyword": "CATEGORY",
    "Brand_Preference": "BRAND",
    "Price_Range": "PRICE",
    "Material_Composition": "MATERIAL",
    "A4_appearance": "STYLE",  # Appearance (颜色+风格)
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

# 语义类型到默认值的映射
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


# 完整模板定义：每个模板包含结构信息和文本
# 每个模板是 (slots_needed, template_text)
# slots_needed: 该模板需要的语义类型列表
# 只使用 A1-A5 对应的语义类型: CATEGORY, BRAND, PRICE, COLOR, USE_CASE
TEMPLATES: Dict[str, List[Tuple[List[str], str]]] = {
    # HIGH-1: Relative Clause（平行从句）
    "HIGH-1": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} that is from the {BRAND} brand, that is priced around {PRICE}, and that is suitable for {USE} in my current project."),
    ],
    # HIGH-2: Nested Clause（嵌套结构）
    "HIGH-2": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} that is from the {BRAND} brand, which offers products that are priced around {PRICE}, and that are suitable for {USE} in my current project."),
    ],
    # HIGH-3: Participial Structure（分散结构）
    "HIGH-3": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} that, being from the {BRAND} brand and being priced around {PRICE}, is suitable for {USE} in my current project."),
    ],
    # HIGH-4: Appositive Structure（插入结构）
    "HIGH-4": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT}, a product from the {BRAND} brand, priced around {PRICE}, and suitable for {USE} in my current project."),
    ],
    # HIGH-5: Prepositional Stacking（介词堆叠）
    "HIGH-5": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand, with a price around {PRICE}, for use in {USE}, in my current project."),
    ],
    # HIGH-6: Infinitival Structure（不定式结构）
    "HIGH-6": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} to be used in {USE}, to be priced around {PRICE}, and to be from the {BRAND} brand in my current project."),
    ],
    # HIGH-7: Passive Structure（被动结构）
    "HIGH-7": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} that is designed by the {BRAND} brand, that is priced around {PRICE}, and that is used for {USE} in my current project."),
    ],
    # HIGH-8: Cleft Sentence（强调句）
    "HIGH-8": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "It is {ARTICLE} {STYLE} {CAT} from the {BRAND} brand that I am looking for, which is priced around {PRICE} and suitable for {USE} in my current project."),
    ],
    # HIGH-9: Coordination-heavy（重并列）
    "HIGH-9": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand and priced around {PRICE} and suitable for {USE} and appropriate for my current project needs."),
    ],
    # HIGH-10: Reduced Relative（省略从句）
    "HIGH-10": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand priced around {PRICE} suitable for {USE} in my current project."),
    ],
    # HIGH-11: Right-branching（链式结构）
    "HIGH-11": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} that is from the {BRAND} brand that provides products that are priced around {PRICE} that are suitable for {USE} in my current project."),
    ],
    # HIGH-12: Left-dislocation（前置结构）
    "HIGH-12": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "As for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand, priced around {PRICE} and suitable for {USE}, I am looking for one for my current project."),
    ],
    # HIGH-13: Existential Sentence（存在句）
    "HIGH-13": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "There is a need for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand, priced around {PRICE} and suitable for {USE} in my current project."),
    ],
    # HIGH-14: Nominalization（名词化）
    "HIGH-14": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "The requirement is for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand, with a price around {PRICE} and suitability for {USE} in my current project."),
    ],
    # HIGH-15: Wh-clause（疑问嵌入）
    "HIGH-15": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for what would be {ARTICLE} {STYLE} {CAT} from the {BRAND} brand, priced around {PRICE} and suitable for {USE} in my current project."),
    ],
    # HIGH-16: Inversion（倒装结构）
    "HIGH-16": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "Looking for is {ARTICLE} {STYLE} {CAT} from the {BRAND} brand, priced around {PRICE} and suitable for {USE} in my current project."),
    ],
    # HIGH-17: Modifier Stacking（修饰堆叠）
    "HIGH-17": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT} from the {BRAND} brand with a price around {PRICE} with suitability for {USE} with application in my current project."),
    ],
    # HIGH-18: Parenthetical（插入干扰）
    "HIGH-18": [
        (["CATEGORY", "BRAND", "PRICE", "USE_CASE", "STYLE"],
         "I am looking for {ARTICLE} {STYLE} {CAT}, from the {BRAND} brand, as it happens, priced around {PRICE} and suitable for {USE}, in my current project."),
    ],
}


def _clean(text: str) -> str:
    out = re.sub(r"\s+", " ", (text or "").strip())
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out


def _compact_value(value: str, max_tokens: int = 2) -> str:
    """压缩属性值为指定最大词数"""
    if not value:
        return ""
    tokens = [t for t in re.split(r"\s+", value.strip()) if t]
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


def _get_semantic_type(dimension: str) -> str:
    """获取维度的语义类型"""
    return DIMENSION_TO_SEMANTIC_TYPE.get(dimension, "FEATURE")


def _build_attr_map_by_semantic_type(
    selected_attrs: List[Tuple[str, str]]
) -> Dict[str, str]:
    """
    将属性列表按语义类型分组
    返回: {语义类型: 属性值}
    """
    semantic_map: Dict[str, str] = {}
    for dimension, value in selected_attrs:
        sem_type = _get_semantic_type(dimension)
        if sem_type not in semantic_map:  # 只保留第一个匹配的类型
            semantic_map[sem_type] = value
    return semantic_map


def _map_slot_placeholder(sem_type: str) -> str:
    """语义类型到槽位占位符的映射"""
    mapping = {
        "CATEGORY": "{CAT}",
        "BRAND": "{BRAND}",
        "PRICE": "{PRICE}",
        "MATERIAL": "{MATERIAL}",
        "COLOR": "{COLOR}",
        "SIZE": "{SIZE}",
        "QUALITY": "{QUALITY}",
        "STYLE": "{STYLE}",
        "USE_CASE": "{USE}",
        "FEATURE": "{FEATURE}",
    }
    return mapping.get(sem_type, "{FEATURE}")


def _is_plural(noun_phrase: str) -> bool:
    """判断名词短语是否为复数"""
    if not noun_phrase:
        return False

    words = noun_phrase.strip().split()
    if not words:
        return False

    last_word = words[-1].lower()

    # 不规则复数词（foot->feet, tooth->teeth, mouse->mice 等）
    irregular_plurals = {
        'feet', 'teeth', 'geese', 'mice', 'lice', 'men', 'women', 'children',
        'people', 'oxen', 'cattle', 'deer', 'sheep', 'fish', 'species', 'series',
        'people', 'folks', 'guys', 'pads', 'inks', 'beads', 'feet', 'colors',
    }

    # 常见复数名词特征词
    plural_indicators = {'some', 'any', 'few', 'many', 'various', 'different', 'multiple'}
    if any(ind in words for ind in plural_indicators):
        return True

    # 检查最后一个词是否是不规则复数
    if last_word in irregular_plurals:
        return True

    # 检查最后一个词是否以 s 结尾（复数）
    plural_patterns = (
        last_word.endswith('s') and
        not last_word.endswith('ss') and
        not last_word.endswith('us')  # us 不是复数
    ) or last_word.endswith('ies') or last_word.endswith('ves')

    return plural_patterns


def _get_article(noun_phrase: str) -> str:
    """根据名词短语返回合适的冠词（a/an/some）"""
    if not noun_phrase:
        return "a"

    # 清理短语，取最后一个词来判断复数形式
    words = noun_phrase.strip().split()
    if not words:
        return "a"

    # 取最后一个词来判断是否复数
    last_word = words[-1].lower()

    # 检查最后一个词是否以 s 结尾（复数）
    plural_patterns = (
        last_word.endswith('s') and
        not last_word.endswith('ss') and
        not last_word.endswith('us')  # us 不是复数
    ) or last_word.endswith('ies') or last_word.endswith('ves')

    # 不规则复数词（foot->feet, tooth->teeth, mouse->mice 等）
    irregular_plurals = {
        'feet', 'teeth', 'geese', 'mice', 'lice', 'men', 'women', 'children',
        'people', 'oxen', 'cattle', 'deer', 'sheep', 'fish', 'species', 'series',
        'people', 'folks', 'guys', 'pads', 'inks', 'beads', 'colors',
    }

    # 常见的复数名词特征词
    plural_indicators = {'some', 'any', 'few', 'many', 'various', 'different', 'multiple'}
    if any(ind in words for ind in plural_indicators):
        return "some"

    # 检查是否是不规则复数
    if last_word in irregular_plurals:
        return "some"

    # 复数名词
    if plural_patterns:
        return "some"

    # 取第一个词来判断元音开头
    first_word = words[0].lower()
    vowel_start = first_word[0] in 'aeiou' if first_word else False

    # 元音开头用 an
    if vowel_start:
        return "an"

    # 其他情况用 a
    return "a"


def generate_query_from_attributes(
    category: str,
    selected_attrs: List[Tuple[str, str]],
    subtype: str,
    rng: Optional[random.Random] = None,
) -> Tuple[str, str]:
    """
    按语义类型映射生成查询
    """
    chooser = rng if rng is not None else random
    c = category if category else "craft supplies"

    # 按语义类型组织属性
    semantic_map = _build_attr_map_by_semantic_type(selected_attrs)

    # 获取模板池
    templates = TEMPLATES.get(subtype, TEMPLATES["HIGH-1"])

    # 随机选择一个模板
    template_idx = chooser.randrange(len(templates))
    slots_needed, template_text = templates[template_idx]

    # 准备槽位填充值
    slot_values: Dict[str, str] = {}
    for sem_type in slots_needed:
        placeholder = _map_slot_placeholder(sem_type)
        if sem_type in semantic_map:
            slot_values[placeholder] = _compact_value(semantic_map[sem_type], max_tokens=2)
        else:
            # 使用默认值
            default = SEMANTIC_DEFAULTS.get(sem_type, c)
            slot_values[placeholder] = _compact_value(default, max_tokens=2)

    # 确保类别槽位有合理值
    if "{CAT}" not in slot_values or slot_values["{CAT}"] in ["craft supplies", ""]:
        slot_values["{CAT}"] = _compact_value(c, max_tokens=2)

    # 计算合适的冠词
    cat_value = slot_values.get("{CAT}", c)
    article = _get_article(cat_value)

    # 添加冠词槽位
    slot_values["{ARTICLE}"] = article

    # 替换所有槽位
    query = template_text
    for placeholder, value in slot_values.items():
        query = query.replace(placeholder, value)

    # 清理
    query = _clean(query)
    if query and query[-1] not in ".!?":
        query += "."

    template_id = f"{subtype}#{template_idx + 1}"
    return query, template_id
