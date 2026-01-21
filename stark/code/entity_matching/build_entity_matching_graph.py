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


def build_graph(
    data: Dict[str, Any],
    include_main_cat: bool = True,
    add_category_hierarchy: bool = True,
) -> Dict[str, Any]:
    """
    Returns a dict compatible with save_files().
    """
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
    if not isinstance(products, list):
        raise ValueError("Input JSON: expected top-level key 'products' to be a list")

    alloc = _IdAllocator()

    node_info: Dict[int, Dict[str, Any]] = {}
    node_types: Dict[int, int] = {}

    product_id_by_asin: Dict[str, int] = {}
    category_id_by_name: Dict[str, int] = {}
    entity_id_by_key: Dict[str, int] = {}  # "<EntityLabel>::<Value>" -> node_id

    edges_set: set[Tuple[int, int, int]] = set()

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

    for p in products:
        asin = _norm_str(p.get("asin"))
        if asin is None:
            continue
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

    return {
        "node_info": node_info,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "node_types": node_types_t,
        "node_type_dict": node_type_dict,
        "edge_type_dict": edge_type_dict,
        "maps": maps,
    }


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
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    graph_files = build_graph(
        data,
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

