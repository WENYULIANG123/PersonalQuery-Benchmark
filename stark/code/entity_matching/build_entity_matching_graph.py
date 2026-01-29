#!/usr/bin/env python3
"""
Build a simple SKB-style graph from entity_matching_results.json.

Graph design (aligned with product_extraction.py entity categories, similar to AmazonSKB post_process):
- Node types:
  - product
  - category
  - one node type per entity category value (Scheme B):
    product --has_brand--> Brand(name="Staedtler")
    product --has_usage--> Usage(name="Shading")
- Edge types:
  - product -> category: has_category
  - product -> entity_value_node: has_<entity_category>
  - (optional) category -> category: category_parent_of (from category path)

Outputs (in output_dir):
- node_info.pkl (dict[int, dict])
- edge_index.pt (torch.LongTensor [2, E])
- edge_types.pt (torch.LongTensor [E])
- node_types.pt (torch.LongTensor [N])
- node_type_dict.pkl (dict[int, str])
- edge_type_dict.pkl (dict[int, str])
- maps.pkl (dict with useful id maps)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import os.path as osp
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


def save_files(save_path: str, **kwargs: Any) -> None:
    """
    Minimal copy of stark_qa.tools.io.save_files, kept local to avoid importing
    the whole `stark_qa` package (which can pull extra optional deps).
    """
    os.makedirs(save_path, exist_ok=True)
    for key, value in kwargs.items():
        if isinstance(value, dict):
            with open(osp.join(save_path, f"{key}.pkl"), "wb") as f:
                pickle.dump(value, f)
        elif isinstance(value, torch.Tensor):
            torch.save(value, osp.join(save_path, f"{key}.pt"))
        else:
            raise NotImplementedError(f"File type not supported for key: {key} ({type(value)})")


def _norm_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    if not x:
        return None
    # light normalization: collapse whitespace
    x = " ".join(x.split())
    return x


def _iter_str_list(values: Any) -> Iterable[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        for v in values:
            s = _norm_str(v)
            if s is not None:
                yield s
        return
    s = _norm_str(values)
    if s is not None:
        yield s


class _IdAllocator:
    def __init__(self) -> None:
        self._next_id = 0

    def next(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid


def normalize_entity_type_label(label: str) -> str:
    """
    Normalize entity type labels to keep graph schema stable.
    Currently enforces: "Color/Finish" -> "Color" (and common variants).
    """
    if label is None:
        return label
    s = str(label).strip()
    if not s:
        return s
    s_lower = s.lower().strip()
    s_compact = s_lower.replace(" ", "")
    if s_compact in {"color/finish", "colour/finish", "colorfinish", "colourfinish"}:
        return "Color"
    if s_lower in {"color", "colour"}:
        return "Color"
    if s_compact == "selling_point":
        return "Selling Point"
    if s_compact == "size":
        return "Dimensions"
    return s


def categorize_color(color_name: str) -> str:
    """
    将具体颜色名称归类到颜色系。
    返回颜色系名称，而不是具体的颜色。
    """
    if not color_name:
        return "Unknown"

    color_lower = color_name.lower().strip()

    # 蓝色系
    blue_keywords = ['blue', 'azure', 'navy', 'indigo', 'cyan', 'teal', 'turquoise', 'cobalt', 'sapphire', 'lunar', 'haze']
    if any(keyword in color_lower for keyword in blue_keywords):
        return "Blue"

    # 绿色系
    green_keywords = ['green', 'lime', 'olive', 'emerald', 'jade', 'mint', 'grass', 'cascade', 'fl green']
    if any(keyword in color_lower for keyword in green_keywords):
        return "Green"

    # 红色系
    red_keywords = ['red', 'crimson', 'scarlet', 'maroon', 'coral', 'pink', 'rose', 'carmine', 'flamingo', 'geranium', 'perylene', 'rose quartz']
    if any(keyword in color_lower for keyword in red_keywords):
        return "Red"

    # 黄色系
    yellow_keywords = ['yellow', 'gold', 'amber', 'orange', 'beige', 'cream', 'antique gold', 'aztec gold', 'brick beige']
    if any(keyword in color_lower for keyword in yellow_keywords):
        return "Yellow"

    # 紫色系
    purple_keywords = ['purple', 'violet', 'lavender', 'magenta', 'plum', 'carbazole', 'english lavender', 'misty lavender']
    if any(keyword in color_lower for keyword in purple_keywords):
        return "Purple"

    # 棕色系
    brown_keywords = ['brown', 'tan', 'coffee', 'chocolate', 'taupe', 'saddle brown', 'mid brown', 'dark brown', 'deep brown']
    if any(keyword in color_lower for keyword in brown_keywords):
        return "Brown"

    # 灰色系
    gray_keywords = ['gray', 'grey', 'silver', 'charcoal', 'ash', 'cool gray', 'mid gray', 'antique silver']
    if any(keyword in color_lower for keyword in gray_keywords):
        return "Gray"

    # 白色系
    white_keywords = ['white', 'ivory', 'pearl', 'snow']
    if any(keyword in color_lower for keyword in white_keywords):
        return "White"

    # 黑色系
    black_keywords = ['black', 'ebony', 'onyx', 'coal']
    if any(keyword in color_lower for keyword in black_keywords):
        return "Black"

    # 其他颜色归类到杂色系
    return "Other"


def categorize_material(mat_name: str) -> str:
    """
    将具体材质名称归类到材质大类。
    返回单单词的材质系名称。
    """
    if not mat_name:
        return "Unknown"

    mat_lower = mat_name.lower().strip()

    # 定义映射规则 (Single-word design)
    # Textile: 织物/纤维/毛发/帆布
    if any(kw in mat_lower for kw in ['polyester', 'cotton', 'silk', 'canvas', 'thread', 'nylon', 'wool', 'hair', 'cloth', 'muslin', 'fabric', 'rayon', 'bobbinfil']):
        return "Textile"
    # Metal: 金属
    if any(kw in mat_lower for kw in ['gold', 'silver', 'steel', 'titanium', 'metal', 'iron', 'aluminum', 'metallic']):
        return "Metal"
    # Plastic: 塑料/硅胶/聚合物
    if any(kw in mat_lower for kw in ['plastic', 'silicone', 'rubber', 'synthetic', 'fimo', 'polymer', 'resin', 'flexible']):
        return "Plastic"
    # Wood: 木质
    if any(kw in mat_lower for kw in ['wood', 'bamboo', 'oak', 'timber', 'cork', 'log']):
        return "Wood"
    # Paper: 纸质
    if any(kw in mat_lower for kw in ['paper', 'card', 'watercolor paper', 'art paper']):
        return "Paper"
    # Mineral: 矿物/陶瓷/宝石/粘土
    if any(kw in mat_lower for kw in ['stone', 'clay', 'ceramic', 'porcelain', 'crystal', 'jade', 'zisha', 'duan ni', 'amethyst', 'lapis', 'hematite', 'rhodonite', 'piemontite', 'serpentine', 'mineral']):
        return "Mineral"
    # Glass: 玻璃
    if any(kw in mat_lower for kw in ['glass', 'mirror']):
        return "Glass"
    # Medium: 媒介 (颜料/墨水/油脂/粉末)
    if any(kw in mat_lower for kw in ['ink', 'oil', 'wax', 'water', 'pigment', 'powder', 'gel', 'fluid', 'graphite', 'charcoal', 'spirits', 'aqueous']):
        return "Medium"


def categorize_usage(usage_name: str) -> str:
    """
    将具体用途名称归类到用途大类。
    返回单单词的用途场景名称。
    """
    if not usage_name:
        return "Unknown"

    u_lower = usage_name.lower().strip()

    # Art: 绘画、艺术创作
    if any(kw in u_lower for kw in ['drawing', 'sketching', 'painting', 'watercolor', 'blending', 'shading', 'art product', 'art project', 'watercolorist', 'pencil', 'professional quality']):
        return "Art"
    # Craft: 手工、DIY、装饰
    if any(kw in u_lower for kw in ['scrapbooking', 'craft', 'ornament', 'rubber stamping', 'diy', 'hobby', 'jewelry', 'model building']):
        return "Craft"
    # Sew: 缝纫、刺绣、织物处理
    if any(kw in u_lower for kw in ['sewing', 'embroidery', 'tapestry', 'needlework', 'threading', 'quilt', 'hand embroidery', 'machine embroidery', 'stitch', 'wall hanging']):
        return "Sew"
    # Write: 书写、书法
    if any(kw in u_lower for kw in ['writing', 'lettering', 'calligraphy', 'note-taking']):
        return "Write"
    # Card: 卡片、礼仪
    if any(kw in u_lower for kw in ['card', 'invitation', 'announcement', 'thank you note', 'christmas', 'birthday', 'holiday']):
        return "Card"
    # Technical: 技术制图、修补、工业
    if any(kw in u_lower for kw in ['technical drawing', 'touch-up', 'drafting', 'industrial', 'repair']):
        return "Technical"
    # Office: 办公、学习
    if any(kw in u_lower for kw in ['office', 'presentation', 'marking', 'school', 'student']):
        return "Office"
    # Storage: 储存、携带、支撑
    if any(kw in u_lower for kw in ['travel', 'storage', 'stand', 'clamp', 'protection', 'case', 'portable']):
        return "Storage"


def categorize_dimensions(dim_name: str) -> str:
    """归一化尺寸为 Small, Medium, Large"""
    if not dim_name: return "Unknown"
    d_lower = dim_name.lower().strip()
    
    # 提取所有数字（支持小数）
    import re
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", d_lower)]
    max_num = max(nums) if nums else 0
    
    # Large 逻辑: 线长度、大包装、显式关键词
    if any(kw in d_lower for kw in ['large', 'jumbo', 'oversized', 'giant', 'long', '800m', 'yard', 'meter', '800 meters']):
        return "Large"
    if max_num >= 20: # 20 inch 或 20 cm 以上倾向于算大 (或标准以上)
        return "Large"
        
    # Small 逻辑
    if any(kw in d_lower for kw in ['mini', 'small', 'pocket', 'tiny', '1/2', '3/8', '0.618']):
        return "Small"
    if "size" in d_lower:
        # 比如 Size 2 是小的
        if max_num > 0 and max_num <= 5: return "Small"
    if max_num > 0 and max_num <= 5: # 5 inch 以下算小
        return "Small"
        
    return "Medium"


def categorize_quantity(qty_name: str) -> str:
    """归一化数量为 Single, Bulk"""
    if not qty_name: return "Unknown"
    q_lower = qty_name.lower().strip()
    
    import re
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", q_lower)]
    max_num = max(nums) if nums else 0
    
    if q_lower in ['one', 'single', 'individual', '1']:
        return "Single"
    
    # 显式 Bulk 关键词
    if any(kw in q_lower for kw in ['set', 'pack', 'pcs', 'pieces', 'dozen', 'kit', 'bulk', 'spools', 'sheets', 'pages']):
        return "Bulk"
    
    if max_num > 1:
        return "Bulk"
        
    return "Single"


def categorize_safety(safety_name: str) -> str:
    """归一化安全认证为 Safe, Professional, Toxic"""
    if not safety_name: return "Unknown"
    s_lower = safety_name.lower().strip()
    
    # Toxic: 警告类
    if any(kw in s_lower for kw in ['caution', 'warning', 'toxic', 'flammable', 'harmful', 'supervision']):
        return "Toxic"
        
    # Professional: 行业标准/认证
    if any(kw in s_lower for kw in ['astm', 'ap certified', 'ap cert', 'en71', 'en-71', 'ce', 'd4236', 'd-4236', 'compliance', 'certifi']):
        return "Professional"
        
    # Safe: 环境友好/无毒
    if any(kw in s_lower for kw in ['non-toxic', 'non toxic', 'eco-friendly', 'environment', 'acid-free', 'safe', 'lead-free', 'washable', 'archival', 'lignin-free', 'azo free']):
        return "Safe"
        
    return "Other"


def categorize_design(design_name: str) -> str:
    """归一化设计为 Shape, Pattern, Style"""
    if not design_name: return "Unknown"
    d_lower = design_name.lower().strip()
    
    # Shape: 形状
    if any(kw in d_lower for kw in ['round', 'oval', 'square', 'angular', 'point', 'shape', 'flat', '3d', 'scallop']):
        return "Shape"
        
    # Pattern: 图案/纹样
    if any(kw in d_lower for kw in ['flower', 'butterfly', 'animal', 'star', 'heart', 'dot', 'stripe', 'floral', 'pattern', 'wildflower']):
        return "Pattern"
        
    # Style: 风格
    if any(kw in d_lower for kw in ['classic', 'modern', 'vintage', 'retro', 'sparkle', 'glitter', 'metallic', 'fashion', 'style', 'inspired']):
        return "Style"
        
    # Format: 格式 (如 JEF, DST 等刺绣格式)
    if any(kw in d_lower for kw in ['jef', 'dst', 'pes', 'xxx', 'hus', 'vip', 'exp', 'sew', 'pcs', 'art']):
        return "Format"
        
    return "Other"


def categorize_selling_point(sp_name: str) -> str:
    """归一化卖点为 Quality, Feature, Portable"""
    if not sp_name: return "Unknown"
    s_lower = sp_name.lower().strip()
    
    # Portable: 便携性
    if any(kw in s_lower for kw in ['portable', 'travel', 'lightweight', 'compact', 'carry']):
        return "Portable"
        
    # Quality: 品质描述
    if any(kw in s_lower for kw in ['high quality', 'professional', 'premium', 'durable', 'sturdy', 'long lasting', 'stable', 'strong', 'well done', 'best']):
        return "Quality"
        
    # Feature: 功能特性
    if any(kw in s_lower for kw in ['waterproof', 'water-soluble', 'lightfast', 'blendable', 'erasable', 'washable', 'adjustable', 'flexible', 'easy', 'versatile', 'soluble']):
        return "Feature"
        
    # Origin: 产地
    if any(kw in s_lower for kw in ['made in', 'usa', 'germany', 'japan']):
        return "Origin"
        
    return "Other"


    return "Other"


def load_entity_map(map_path: str) -> Dict[str, Dict[str, str]]:
    """Loads the LLM-generated entity resolution map"""
    try:
        with open(map_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load entity map from {map_path}: {e}")
        return {}


def build_graph(
    data: Dict[str, Any],
    entity_map_path: Optional[str] = None,
    include_main_cat: bool = True,
    add_category_hierarchy: bool = True,
) -> Dict[str, Any]:
    """
    Returns a dict compatible with save_files().
    """
    print("DEBUG: build_graph started")
    print(f"DEBUG: include_main_cat={include_main_cat}, add_category_hierarchy={add_category_hierarchy}")
    entity_type_order = [
        "Brand",
        "Material",
        "Dimensions",
        "Quantity",
        "Color",
        "Design",
        "Usage",
        "Selling Point",
        "Safety/Certification",
        "Accessories",
    ]

    def canonical_entity_type(t: str) -> str:
        # canonical for edge naming / internal keys
        t = t.strip().lower()
        out = []
        prev_us = False
        for ch in t:
            if ch.isalnum():
                out.append(ch)
                prev_us = False
            else:
                if not prev_us:
                    out.append("_")
                    prev_us = True
        s = "".join(out).strip("_")
        while "__" in s:
            s = s.replace("__", "_")
        return s

    # node types: product, category, then one type per entity category (Brand, Material, ...)
    node_type_dict: Dict[int, str] = {0: "product", 1: "category"}
    entity_type_id_by_label: Dict[str, int] = {}
    for i, label in enumerate(entity_type_order, start=2):
        node_type_dict[i] = label
        entity_type_id_by_label[label] = i

    # edge types: has_category, then one edge type per entity category, then optional category hierarchy
    edge_type_dict: Dict[int, str] = {0: "has_category"}
    edge_type_id_by_label: Dict[str, int] = {}
    next_edge_id = 1
    for label in entity_type_order:
        edge_type_dict[next_edge_id] = f"has_{canonical_entity_type(label)}"
        edge_type_id_by_label[label] = next_edge_id
        next_edge_id += 1

    category_parent_of_edge_type_id = None
    if add_category_hierarchy:
        category_parent_of_edge_type_id = next_edge_id
        edge_type_dict[category_parent_of_edge_type_id] = "category_parent_of"

    products: List[Dict[str, Any]] = data.get("products") or []
    print(f"DEBUG: Found {len(products)} products")
    if not isinstance(products, list):
        raise ValueError("Input JSON: expected top-level key 'products' to be a list")

    alloc = _IdAllocator()

    node_info: Dict[int, Dict[str, Any]] = {}
    node_types: Dict[int, int] = {}

    product_id_by_asin: Dict[str, int] = {}
    category_id_by_name: Dict[str, int] = {}
    entity_id_by_key: Dict[str, int] = {}  # "<EntityLabel>::<Value>" -> node_id

    edges_set: set[Tuple[int, int, int]] = set()

    # Load entity map if provided
    entity_map = {}
    if entity_map_path and os.path.exists(entity_map_path):
        print(f"DEBUG: Loading entity map from {entity_map_path}")
        with open(entity_map_path, 'r') as f:
            entity_map = json.load(f)

    def get_or_create_product(asin: str, title: Optional[str], product_info: Dict[str, Any]) -> int:
        if asin in product_id_by_asin:
            return product_id_by_asin[asin]
        nid = alloc.next()
        product_id_by_asin[asin] = nid
        node_types[nid] = 0
        node_info[nid] = {
            "type": "product",
            "asin": asin,
            "title": _norm_str(title) or _norm_str(product_info.get("title")) or "",
            "main_cat": _norm_str(product_info.get("main_cat")),
        }
        return nid

    def get_or_create_category(name: str) -> int:
        if name in category_id_by_name:
            return category_id_by_name[name]
        nid = alloc.next()
        category_id_by_name[name] = nid
        node_types[nid] = 1
        node_info[nid] = {
            "type": "category",
            "category_name": name,
        }
        return nid

    def get_or_create_entity(entity_label: str, value: str, sentiment: Optional[str] = None) -> int:
        # keep nodes unique per (entity_label, value) to avoid collisions
        key = f"{entity_label}::{value}"
        if key in entity_id_by_key:
            # If sentiment is provided and different, update it (prefer non-neutral)
            existing_nid = entity_id_by_key[key]
            existing_info = node_info.get(existing_nid, {})
            existing_sentiment = existing_info.get("sentiment")
            if sentiment and sentiment != "neutral":
                if not existing_sentiment or existing_sentiment == "neutral":
                    existing_info["sentiment"] = sentiment
            elif sentiment and not existing_sentiment:
                existing_info["sentiment"] = sentiment
            return existing_nid
        nid = alloc.next()
        entity_id_by_key[key] = nid
        node_types[nid] = entity_type_id_by_label[entity_label]
        node_info[nid] = {
            "type": entity_label,
            "name": value,
        }
        if sentiment:
            node_info[nid]["sentiment"] = sentiment
        return nid

    def add_edge(src: int, dst: int, edge_type_id: int) -> None:
        edges_set.add((src, dst, edge_type_id))

    print(f"DEBUG: Processing {len(products)} products")
    for i, p in enumerate(products):
        asin = _norm_str(p.get("asin"))
        if asin is None:
            print(f"DEBUG: Product {i} has no ASIN, skipping")
            continue
        print(f"DEBUG: Processing product {i}: {asin}")
        product_info = p.get("product_info") or {}
        if not isinstance(product_info, dict):
            product_info = {}
        pid = get_or_create_product(asin, p.get("product_title"), product_info)

        # categories
        cat_path = []
        for c in _iter_str_list(product_info.get("category")):
            cat_path.append(c)
            cid = get_or_create_category(c)
            add_edge(pid, cid, 0)

        if include_main_cat:
            mc = _norm_str(product_info.get("main_cat"))
            if mc is not None:
                mcid = get_or_create_category(mc)
                add_edge(pid, mcid, 0)

        # optional hierarchy edges from category path (parent -> child)
        if add_category_hierarchy and len(cat_path) >= 2:
            for parent, child in zip(cat_path[:-1], cat_path[1:]):
                parent_id = get_or_create_category(parent)
                child_id = get_or_create_category(child)
                if category_parent_of_edge_type_id is not None:
                    add_edge(parent_id, child_id, category_parent_of_edge_type_id)

        # entities: ONLY use product_entities (plain string list, no sentiment)
        product_entities = p.get("product_entities") or {}
        
        if isinstance(product_entities, dict):
            for etype_raw, vals in product_entities.items():
                etype_label = _norm_str(etype_raw)
                if etype_label is None:
                    continue
                etype_label = normalize_entity_type_label(etype_label)
                # Keep only the known categories from product_extraction.py
                if etype_label not in entity_type_order:
                    continue
                
                # product_entities values are plain strings (no sentiment)
                for v in vals:
                    entity_value = _norm_str(v)
                    if not entity_value:
                        continue

                    mapped_value = None

                    # 1. Check for product-specific normalized entities (HIGHEST PRIORITY)
                    normalized_entities_local = p.get("normalized_entities") or {}
                    # Try to find matching key in normalized_entities
                    # Keys in normalized_entities might be original field names (e.g. "Color") or normalized (e.g. "Color")
                    # We try matching etype_raw first, then etype_label
                    local_cat_map = normalized_entities_local.get(etype_raw) or normalized_entities_local.get(etype_label)
                    
                    if local_cat_map and isinstance(local_cat_map, dict):
                         if v in local_cat_map:
                             mapped_value = local_cat_map[v]
                         elif entity_value in local_cat_map:
                             mapped_value = local_cat_map[entity_value]
                    
                    # 2. Check global entity map if no local mapping found
                    if not mapped_value and etype_label in entity_map:
                        # 查找原始值对应的映射 (注意大小写或直接匹配)
                        # entity_map 结构: {Category: {RawValue: NormalizedValue}}
                        cat_map = entity_map[etype_label]
                        if v in cat_map:
                            mapped_value = cat_map[v]
                        elif entity_value in cat_map:
                             mapped_value = cat_map[entity_value]

                    if mapped_value and mapped_value != "Other":
                        entity_value = mapped_value
                        # print(f"DEBUG: Mapped '{v}' -> '{entity_value}'")
                    else:
                        # Fallback to rule-based categorization if LLM map missed it or returned Other
                        # 对于Color实体，使用颜色系而不是具体颜色名称
                        if etype_label == "Color":
                            categorized_value = categorize_color(entity_value)
                            if categorized_value != "Other": entity_value = categorized_value
    
                        # 对于Material实体，使用材质大类
                        if etype_label == "Material":
                            categorized_value = categorize_material(entity_value)
                            if categorized_value != "Other": entity_value = categorized_value
    
                        # 对于Usage实体，使用用途大类
                        if etype_label == "Usage":
                            categorized_value = categorize_usage(entity_value)
                            if categorized_value != "Other": entity_value = categorized_value
    
                        # 对于Dimensions实体
                        if etype_label == "Dimensions":
                            categorized_value = categorize_dimensions(entity_value)
                            if categorized_value != "Other": entity_value = categorized_value
    
                        # 对于Quantity实体
                        if etype_label == "Quantity":
                            categorized_value = categorize_quantity(entity_value)
                            if categorized_value != "Other": entity_value = categorized_value
    
                        # 对于Safety/Certification实体
                        if etype_label == "Safety/Certification":
                            categorized_value = categorize_safety(entity_value)
                            if categorized_value != "Other": entity_value = categorized_value
    
                        # 对于Design实体
                        if etype_label == "Design":
                            categorized_value = categorize_design(entity_value)
                            if categorized_value != "Other": entity_value = categorized_value
    
                        # 对于Selling Point实体
                        if etype_label == "Selling Point":
                            categorized_value = categorize_selling_point(entity_value)
                            if categorized_value != "Other": entity_value = categorized_value

                    # No sentiment for product_entities
                    eid = get_or_create_entity(etype_label, entity_value, None)
                    add_edge(pid, eid, edge_type_id_by_label[etype_label])

    # materialize to tensors
    num_nodes = alloc._next_id
    if num_nodes == 0:
        raise ValueError("No nodes were created (check input JSON content)")

    node_types_list = [node_types[i] for i in range(num_nodes)]
    node_types_t = torch.LongTensor(node_types_list)

    edges_sorted = sorted(edges_set)
    src_list = [s for s, _, _ in edges_sorted]
    dst_list = [d for _, d, _ in edges_sorted]
    et_list = [t for _, _, t in edges_sorted]

    edge_index = torch.LongTensor([src_list, dst_list]) if edges_sorted else torch.LongTensor([[], []])
    edge_types = torch.LongTensor(et_list) if edges_sorted else torch.LongTensor([])

    maps = {
        "product_id_by_asin": product_id_by_asin,
        "category_id_by_name": category_id_by_name,
        "entity_id_by_key": entity_id_by_key,
        "entity_type_order": entity_type_order,
    }

    result = {
        "node_info": node_info,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "node_types": node_types_t,
        "node_type_dict": node_type_dict,
        "edge_type_dict": edge_type_dict,
        "maps": maps,
    }

    print(f"DEBUG: build_graph returning dict with keys: {list(result.keys())}")
    print(f"DEBUG: node_info has {len(node_info)} nodes")
    print(f"DEBUG: edge_index shape: {edge_index.shape if hasattr(edge_index, 'shape') else 'N/A'}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SKB-style graph from product_entities.json")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/wlia0047/ar57/wenyu/result/product_entities.json",
        help="Path to product_entities.json (default points to /home/wlia0047/ar57/wenyu/result/product_entities.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.cwd() / "processed" / "entity_matching_graph"),
        help="Directory to write graph files",
    )
    parser.add_argument(
        "--include_main_cat",
        action="store_true",
        default=True,
        help="Also connect product to product_info.main_cat (default: True)",
    )
    parser.add_argument(
        "--no_include_main_cat",
        action="store_false",
        dest="include_main_cat",
        help="Do not include main_cat edges",
    )
    parser.add_argument(
        "--add_category_hierarchy",
        action="store_true",
        default=True,
        help="Add parent->child edges along product_info.category path (default: True)",
    )
    parser.add_argument(
        "--no_add_category_hierarchy",
        action="store_false",
        dest="add_category_hierarchy",
        help="Do not add category hierarchy edges",
    )
    parser.add_argument(
        "--entity_map",
        type=str,
        default=None,
        help="Path to entity_resolution_map.json (optional)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    graph_files = build_graph(
        data,
        entity_map_path=args.entity_map,
        include_main_cat=args.include_main_cat,
        add_category_hierarchy=args.add_category_hierarchy,
    )

    save_files(str(output_dir), **graph_files)

    num_nodes = len(graph_files["node_info"])
    num_edges = int(graph_files["edge_index"].shape[1]) if isinstance(graph_files["edge_index"], torch.Tensor) else 0
    num_products = len(graph_files["maps"]["product_id_by_asin"])
    num_categories = len(graph_files["maps"]["category_id_by_name"])
    num_entities = len(graph_files["maps"]["entity_id_by_key"])

    print(f"Saved graph to: {output_dir}")
    print(f"Nodes: {num_nodes} (products={num_products}, categories={num_categories}, entities={num_entities})")
    print(f"Edges: {num_edges}")
    print(f"Edge types: {graph_files['edge_type_dict']}")


if __name__ == "__main__":
    main()

