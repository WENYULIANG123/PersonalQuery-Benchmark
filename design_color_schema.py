#!/usr/bin/env python3
"""
设计统一的颜色语义命名系统

方案：层次化颜色命名体系
1. 基础色（Base Color）- 12种标准色
2. 修饰符（Modifier）- 深浅、饱和度、色调
3. 特殊类型（Special）- 金属色、特殊效果
4. 映射规则 - 将现有颜色映射到新系统
"""

import json
import re
import argparse
from collections import defaultdict

# 1. 基础颜色体系（12色环 + 中性色）
BASE_COLORS = {
    # 主色
    'red': ['red', 'carmine', 'scarlet', 'carnival', 'wine', 'geranium', 'sanguine'],
    'orange': ['orange', 'burnt orange', 'vermillion'],
    'yellow': ['yellow', 'lemon', 'bright yellow', 'cadmium yellow', 'hansa yellow'],
    'green': ['green', 'emerald', 'grass', 'olive', 'hooker', 'sap', 'marine', 'may', 'pale', 'turquoise', 'undersea', 'spring'],
    'blue': ['blue', 'cobalt', 'ultramarine', 'french ultramarine', 'peacock', 'persian', 'true', 'haze', 'corn flour', 'deep', 'dull', 'light'],
    'purple': ['purple', 'violet', 'carbazole violet', 'deep violet', 'light violet', 'lavender', 'misty lavender', 'english lavender', 'lilac'],
    'pink': ['pink', 'flamingo', 'peach', 'pale', 'rose', 'tea rose', 'pale rose', 'blush', 'almond pink', 'dark pink', 'light pink'],
    'brown': ['brown', 'mid brown', 'dark brown', 'light brown', 'deep brown', 'gray brown', 'ochre', 'sepia', 'mud', 'oatmeal', 'dark oatmeal'],
    'black': ['black', 'gloss black', 'polished black', 'black coarse', 'black fine'],
    'white': ['white', 'pearlwhite', 'white coarse', 'white fine'],
    'gray': ['gray', 'grey', 'blue gray', 'green gray', 'mid gray', 'light gray', 'dark gray', 'pale gray', 'pale dawn gray', 'natural gray', 'gray tint', 'cool gray', 'warm gray'],
    'beige': ['beige', 'brick beige'],
    
    # 特殊基础色
    'aqua': ['aqua', 'teal', 'turquoise'],
    'clear': ['clear'],
}

# 基础色集合（用于“只保留基础色”输出）
BASE_COLOR_SET = set(BASE_COLORS.keys())

# 2. 修饰符系统
MODIFIERS = {
    'light': ['light', 'pale', 'bright'],
    'dark': ['dark', 'deep'],
    'vibrant': ['vibrant', 'brilliant', 'gorgeous', 'beautiful'],
    'muted': ['muted', 'dull', 'shadow'],
}

# 3. 特殊类型
SPECIAL_TYPES = {
    'metallic': ['metallic', 'gold', 'silver', 'copper', 'bronze', 'antique gold', 'antique silver', 'antique bronze', 'antique copper', 'aztec gold', 'brilliant gold', 'light gold', 'rich gold', 'sparkle gold', 'pewter', '14k', 'white gold', 'sparkling copper', 'super-copper'],
    'pearl': ['pearl', 'macropearl', 'micropearl'],
    'glow': ['glow in the dark', 'luminous', 'sparkle'],
    'frost': ['frost'],
    'multicolor': ['multicolored', 'assorted', 'any color', 'variegated', 'disney designs'],
}

# 一些常见非标准词到基础色的补充映射（优先级高于规则匹配）
EXTRA_BASE_SYNONYMS = {
    'magenta': 'pink',
    'mauve': 'purple',
    'mint': 'green',
    'flesh': 'beige',
    'skin': 'beige',
}

def _canonicalize_text(s: str) -> str:
    """规范化字符串，便于稳健匹配（去标点、统一空白、保留字母数字）。"""
    s = (s or "").lower().strip()
    # 保留“/”用于检测多色，但匹配时也会用空格版
    s = s.replace("&", " and ")
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"[^a-z0-9/\s\-]+", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize(s: str) -> list[str]:
    s = _canonicalize_text(s)
    # 将斜杠也视为分隔符，避免 “red/blue” 被当成一个 token
    s = s.replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split(" ") if t]

def _has_token(tokens: list[str], token: str) -> bool:
    return token in set(tokens)

def _has_phrase(text: str, phrase: str) -> bool:
    """用词边界匹配短语，避免子串误判（如 marine 命中 ultramarine）。"""
    text = f" { _canonicalize_text(text) } "
    phrase = f" { _canonicalize_text(phrase) } "
    return phrase in text

def normalize_color_to_base(color_str: str) -> str | None:
    """
    只保留基础色（base color）：
    - 丢弃 light/dark/vibrant/muted 等修饰
    - 将 metallic/gold/silver/pearl/glow/multicolor 等特殊类型映射到基础色或 unknown
    """
    if not color_str or not str(color_str).strip():
        return None

    raw = str(color_str)
    text = _canonicalize_text(raw)
    tokens = _tokenize(raw)

    # 移除常见后缀
    text = re.sub(r"\s+colors?$", "", text)
    text = re.sub(r"\s+color$", "", text)

    # 0) 多色/杂色：不属于基础色，统一 unknown（你也可以改成返回 'multicolor'）
    if "/" in raw or _has_phrase(text, "multicolored") or _has_phrase(text, "assorted") or _has_phrase(text, "variegated"):
        return "unknown"

    # 1) 先处理特殊类型与金属色（映射到基础色）
    # 金属色 → 近似基础色
    if any(k in text for k in ["metallic", "14k", "gold", "silver", "copper", "bronze", "pewter", "white gold"]):
        if "gold" in text:
            return "yellow"
        if "silver" in text or "pewter" in text:
            return "gray"
        if "copper" in text or "bronze" in text:
            return "brown"
        # 泛 metallic
        return "gray"

    # 珍珠/珠光倾向白色
    if any(k in text for k in ["pearl", "micropearl", "macropearl"]):
        return "white"

    # 发光/霜冻/透明等不是基础色
    if any(k in text for k in ["glow", "luminous", "sparkle", "frost", "clear"]):
        return "unknown"

    # 2) 补充词优先（magenta/mint/mauve/flesh...)
    for k, v in EXTRA_BASE_SYNONYMS.items():
        if _has_phrase(text, k) or _has_token(tokens, k):
            return v

    # 3) 去掉修饰符 token（只保留基础色）
    modifier_tokens = set(sum(MODIFIERS.values(), []))
    core_tokens = [t for t in tokens if t not in modifier_tokens]
    core_text = " ".join(core_tokens)

    # 4) 匹配基础色（先短语/再 token），避免子串误判
    # 4.1 明确把 aqua/teal/turquoise 归到 blue（若你想归 green，改这里即可）
    if any(_has_phrase(core_text, w) or _has_token(core_tokens, w) for w in ["aqua", "teal", "turquoise"]):
        return "blue"

    # 4.2 直接基础色 token
    for base in BASE_COLOR_SET:
        if _has_token(core_tokens, base):
            return base

    # 4.3 别名/变体匹配（按词边界短语优先）
    for base, keywords in BASE_COLORS.items():
        for kw in sorted(keywords, key=lambda x: -len(x)):
            # kw 可能是短语，也可能是单词
            if " " in kw:
                if _has_phrase(core_text, kw):
                    return base
            else:
                if _has_token(core_tokens, kw):
                    return base

    return "unknown"

# 4. 映射函数
def normalize_color_name(color_str):
    """将现有颜色名称映射到统一系统"""
    if not color_str:
        return None
    
    color_lower = color_str.lower().strip()
    
    # 移除常见后缀
    color_lower = re.sub(r'\s+colors?$', '', color_lower)
    color_lower = re.sub(r'\s+color$', '', color_lower)
    
    # 1. 检查特殊类型
    for special_type, keywords in SPECIAL_TYPES.items():
        for keyword in keywords:
            if keyword in color_lower:
                if special_type == 'metallic':
                    # 提取具体金属色
                    if 'gold' in color_lower:
                        return 'metallic_gold'
                    elif 'silver' in color_lower:
                        return 'metallic_silver'
                    elif 'copper' in color_lower:
                        return 'metallic_copper'
                    elif 'bronze' in color_lower:
                        return 'metallic_bronze'
                    elif 'pewter' in color_lower:
                        return 'metallic_pewter'
                    else:
                        return 'metallic'
                elif special_type == 'pearl':
                    return 'pearl'
                elif special_type == 'glow':
                    return 'glow'
                elif special_type == 'frost':
                    return 'frost'
                elif special_type == 'multicolor':
                    return 'multicolor'
    
    # 2. 提取修饰符
    modifier = None
    for mod_type, mod_keywords in MODIFIERS.items():
        for keyword in mod_keywords:
            if keyword in color_lower:
                modifier = mod_type
                # 移除修饰符，提取基础色
                color_lower = color_lower.replace(keyword, '').strip()
                break
    
    # 3. 匹配基础色
    base_color = None
    for base, keywords in BASE_COLORS.items():
        for keyword in keywords:
            if keyword in color_lower or color_lower == keyword:
                base_color = base
                break
        if base_color:
            break
    
    # 4. 组合结果
    if base_color:
        if modifier:
            return f"{modifier}_{base_color}"
        else:
            return base_color
    else:
        # 未匹配的颜色，返回原始值（标准化后）
        return f"unknown_{color_lower.replace(' ', '_')}"

# 5. 生成完整的颜色体系文档
def generate_color_schema_doc():
    """生成颜色体系文档"""
    doc = []
    doc.append("# 统一颜色语义命名系统")
    doc.append("")
    doc.append("## 1. 基础颜色（Base Colors）")
    doc.append("")
    doc.append("| 标准名称 | 别名/变体 |")
    doc.append("|---------|----------|")
    for base, aliases in BASE_COLORS.items():
        doc.append(f"| `{base}` | {', '.join(aliases[:5])}{'...' if len(aliases) > 5 else ''} |")
    
    doc.append("")
    doc.append("## 2. 修饰符（Modifiers）")
    doc.append("")
    doc.append("| 修饰符 | 含义 | 关键词 |")
    doc.append("|-------|------|--------|")
    doc.append("| `light_` | 浅色/亮色 | light, pale, bright |")
    doc.append("| `dark_` | 深色 | dark, deep |")
    doc.append("| `vibrant_` | 鲜艳 | vibrant, brilliant, gorgeous |")
    doc.append("| `muted_` | 柔和 | muted, dull, shadow |")
    
    doc.append("")
    doc.append("## 3. 特殊类型（Special Types）")
    doc.append("")
    doc.append("| 类型 | 说明 | 示例 |")
    doc.append("|------|------|------|")
    doc.append("| `metallic_gold` | 金色金属 | gold, antique gold, aztec gold |")
    doc.append("| `metallic_silver` | 银色金属 | silver, antique silver |")
    doc.append("| `metallic_copper` | 铜色金属 | copper, antique copper, sparkling copper |")
    doc.append("| `metallic_bronze` | 青铜色 | bronze, antique bronze |")
    doc.append("| `metallic_pewter` | 白镴色 | pewter |")
    doc.append("| `metallic` | 通用金属色 | metallic |")
    doc.append("| `pearl` | 珍珠效果 | pearl, macropearl, micropearl |")
    doc.append("| `glow` | 发光效果 | glow in the dark, luminous, sparkle |")
    doc.append("| `frost` | 霜冻效果 | frost |")
    doc.append("| `multicolor` | 多色 | multicolored, assorted, variegated |")
    
    doc.append("")
    doc.append("## 4. 命名规则")
    doc.append("")
    doc.append("### 格式：`[modifier_]base_color` 或 `special_type`")
    doc.append("")
    doc.append("### 示例：")
    doc.append("- `red` - 标准红色")
    doc.append("- `light_blue` - 浅蓝色")
    doc.append("- `dark_green` - 深绿色")
    doc.append("- `vibrant_yellow` - 鲜艳黄色")
    doc.append("- `metallic_gold` - 金色金属")
    doc.append("- `pearl` - 珍珠效果")
    doc.append("- `multicolor` - 多色")
    
    doc.append("")
    doc.append("## 5. 映射示例")
    doc.append("")
    doc.append("| 原始颜色 | 标准化后 |")
    doc.append("|---------|---------|")
    
    # 测试一些示例
    examples = [
        "Black", "black", "Blue", "light Blue", "dark Brown",
        "Gold", "Antique Gold", "Metallic", "Red", "Bright Red",
        "Violet", "deep Violet", "Green", "Emerald Green",
        "Pink", "Flamingo Pink", "White", "Pearlwhite",
        "Multicolored", "Assorted Color", "Glow in the Dark"
    ]
    
    for orig in examples:
        normalized = normalize_color_name(orig)
        doc.append(f"| `{orig}` | `{normalized}` |")
    
    return "\n".join(doc)

# 6. 生成映射表（所有现有颜色 -> 标准化颜色）
def generate_mapping_table():
    """生成完整映射表（默认：只保留基础色）"""
    with open('result/product_entities.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    products = data.get('products', [])
    color_mapping = {}
    color_stats = defaultdict(int)
    
    for product in products:
        product_entities = product.get('product_entities', {})
        for key in ['Color', 'Color/Finish', 'Colour', 'Colour/Finish']:
            if key in product_entities:
                values = product_entities[key]
                if isinstance(values, list):
                    for val in values:
                        if isinstance(val, str) and val.strip():
                            orig = val.strip()
                            normalized = normalize_color_to_base(orig)
                            color_mapping[orig] = normalized
                            color_stats[normalized] += 1
    
    return color_mapping, color_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Design color schema and generate mapping.")
    parser.add_argument(
        "--base-only",
        action="store_true",
        default=True,
        help="Only keep base colors (default: True).",
    )
    args = parser.parse_args()

    # 生成文档
    doc = generate_color_schema_doc()
    with open('color_schema_design.md', 'w', encoding='utf-8') as f:
        f.write(doc)
    print("颜色体系文档已生成: color_schema_design.md")
    
    # 生成映射表
    # 当前实现：generate_mapping_table 默认使用 normalize_color_to_base
    mapping, stats = generate_mapping_table()
    
    with open('color_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({
            'mapping': mapping,
            'statistics': dict(stats)
        }, f, indent=2, ensure_ascii=False)
    print("颜色映射表已生成: color_mapping.json")
    
    # 打印统计
    print("\n标准化后的颜色分布（Top 20）:")
    for color, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {color:<30} : {count}")
