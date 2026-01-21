#!/usr/bin/env python3
"""
Visualize the entity-matching graph produced by build_entity_matching_graph.py.

Two outputs:
1) A schema diagram (product/category/entity and edge types)
2) A data-driven ego subgraph around a selected product (by ASIN)

This intentionally visualizes a *subgraph* to keep the picture readable.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import torch


@dataclass(frozen=True)
class GraphData:
    node_info: Dict[int, Dict[str, Any]]
    edge_index: torch.LongTensor  # [2, E]
    edge_types: torch.LongTensor  # [E]
    node_types: torch.LongTensor  # [N]
    node_type_dict: Dict[int, str]
    edge_type_dict: Dict[int, str]
    maps: Dict[str, Any]


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_graph(dir_path: str) -> GraphData:
    dir_path = osp.abspath(dir_path)
    node_info = _load_pickle(osp.join(dir_path, "node_info.pkl"))
    node_type_dict = _load_pickle(osp.join(dir_path, "node_type_dict.pkl"))
    edge_type_dict = _load_pickle(osp.join(dir_path, "edge_type_dict.pkl"))
    maps = _load_pickle(osp.join(dir_path, "maps.pkl"))

    edge_index = torch.load(osp.join(dir_path, "edge_index.pt"))
    edge_types = torch.load(osp.join(dir_path, "edge_types.pt"))
    node_types = torch.load(osp.join(dir_path, "node_types.pt"))

    return GraphData(
        node_info=node_info,
        edge_index=edge_index,
        edge_types=edge_types,
        node_types=node_types,
        node_type_dict=node_type_dict,
        edge_type_dict=edge_type_dict,
        maps=maps,
    )


def _iter_out_edges(g: GraphData, src: int) -> Iterable[Tuple[int, int]]:
    """
    Yield (dst, edge_type_id) for edges src -> dst.
    """
    ei = g.edge_index
    et = g.edge_types
    if ei.numel() == 0:
        return []
    srcs = ei[0]
    mask = srcs == int(src)
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    for j in idx.tolist():
        yield int(ei[1, j].item()), int(et[j].item())


def ego_subgraph(
    g: GraphData,
    center_product_id: int,
    max_entity_nodes: int = 80,
    include_category_hierarchy: bool = True,
) -> nx.DiGraph:
    Ego subgraph around a product:
    - 1-hop categories
    - 1-hop entity_category slots
    - 2-hop entity_value nodes via slots
    Limits entity_value nodes for readability.
    """
    G = nx.DiGraph()

    def add_node(nid: int) -> None:
        info = g.node_info.get(nid, {})
        ntype_id = int(g.node_types[nid].item())
        ntype = g.node_type_dict.get(ntype_id, str(ntype_id))
        G.add_node(nid, ntype=ntype, **info)

    add_node(center_product_id)

    categories: List[int] = []
    slots: List[int] = []
    values: List[int] = []
    for dst, etype in _iter_out_edges(g, center_product_id):
        # Convention from builder (new design):
        # - edge_type 0: has_category (product -> category)
        # - edge_type 1: has_entity_category (product -> entity_category slot)
        # - edge_type 2: has_entity_value (slot -> entity_value)
        # - edge_type 3: category_parent_of (category -> category) when enabled
        if etype == 0:
            categories.append(dst)
        elif etype == 1:
            slots.append(dst)

    # 2-hop: slot -> value
    for sid in slots:
        for dst, etype in _iter_out_edges(g, sid):
            if etype == 2:
                values.append(dst)

    # limit entity value nodes
    values = values[: max_entity_nodes]

    for cid in categories:
        add_node(cid)
        G.add_edge(center_product_id, cid, etype="has_category")
    for sid in slots:
        add_node(sid)
        G.add_edge(center_product_id, sid, etype="has_entity_category")
    for vid in values:
        add_node(vid)

    value_set = set(values)
    for sid in slots:
        for dst, etype in _iter_out_edges(g, sid):
            if etype == 2 and dst in value_set:
                G.add_edge(sid, dst, etype="has_entity_value")

    if include_category_hierarchy and categories:
        category_set: Set[int] = set(categories)
        # Add edges parent->child among included categories (edge_type==3)
        ei = g.edge_index
        et = g.edge_types
        if ei.numel() > 0:
            mask = et == 3
            idx = torch.nonzero(mask, as_tuple=False).view(-1)
            for j in idx.tolist():
                src = int(ei[0, j].item())
                dst = int(ei[1, j].item())
                if src in category_set and dst in category_set:
                    add_node(src)
                    add_node(dst)
                    G.add_edge(src, dst, etype="category_parent_of")

    return G


def draw_schema_png(out_path: str) -> None:
    """
    A clean, conceptual schema diagram (not data-driven).
    """
    G = nx.DiGraph()
    G.add_node("product", ntype="product")
    G.add_node("category", ntype="category")
    G.add_node("entity_category", ntype="entity_category")
    G.add_node("entity_value", ntype="entity_value")
    G.add_edge("product", "category", etype="has_category")
    G.add_edge("product", "entity_category", etype="has_entity_category")
    G.add_edge("entity_category", "entity_value", etype="has_entity_value")
    G.add_edge("category", "category", etype="category_parent_of")

    pos = {
        "product": (0.0, 0.0),
        "category": (1.8, 0.8),
        "entity_category": (1.8, -0.1),
        "entity_value": (3.6, -0.6),
    }

    fig = plt.figure(figsize=(7.5, 3.8), dpi=220)
    ax = plt.gca()
    ax.set_axis_off()

    nx.draw_networkx_nodes(G, pos, nodelist=["product"], node_color="#ff6b6b", node_size=1800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=["category"], node_color="#4dabf7", node_size=1800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=["entity_category"], node_color="#51cf66", node_size=1800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=["entity_value"], node_color="#adb5bd", node_size=1800, ax=ax)

    nx.draw_networkx_labels(
        G,
        pos,
        labels={
            "product": "product",
            "category": "category",
            "entity_category": "entity_category",
            "entity_value": "entity_value",
        },
        font_size=10,
        font_color="#111",
        ax=ax,
    )

    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=18, width=2.0, edge_color="#333", ax=ax)

    edge_labels = {
        ("product", "category"): "has_category",
        ("product", "entity_category"): "has_entity_category",
        ("entity_category", "entity_value"): "has_entity_value",
        ("category", "category"): "category_parent_of",
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.55, ax=ax)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_subgraph_png(
    G: nx.DiGraph,
    out_path: str,
    title: str,
    label_entities: bool = False,
) -> None:
    # deterministic layout
    pos = nx.spring_layout(G, seed=7, k=0.9)

    products = [n for n, d in G.nodes(data=True) if d.get("ntype") == "product"]
    categories = [n for n, d in G.nodes(data=True) if d.get("ntype") == "category"]
    entity_categories = [n for n, d in G.nodes(data=True) if d.get("ntype") == "entity_category"]
    entity_values = [n for n, d in G.nodes(data=True) if d.get("ntype") == "entity_value"]

    fig = plt.figure(figsize=(12, 9), dpi=220)
    ax = plt.gca()
    ax.set_axis_off()

    nx.draw_networkx_edges(G, pos, arrows=False, width=0.8, alpha=0.25, edge_color="#555", ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=entity_values, node_color="#adb5bd", node_size=32, alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=entity_categories, node_color="#51cf66", node_size=150, alpha=0.95, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=categories, node_color="#4dabf7", node_size=180, alpha=0.95, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=products, node_color="#ff6b6b", node_size=420, alpha=0.95, ax=ax)

    labels: Dict[Any, str] = {}
    for n in products:
        labels[n] = (G.nodes[n].get("title") or G.nodes[n].get("asin") or "product")[:45]
    for n in categories:
        labels[n] = (G.nodes[n].get("category_name") or "category")[:26]
    for n in entity_categories:
        labels[n] = (G.nodes[n].get("entity_category") or "entity_category")[:18]
    if label_entities:
        for n in entity_values:
            labels[n] = (G.nodes[n].get("name") or "entity_value")[:18]

    # Draw labels with different sizes to reduce clutter when entity labels are enabled.
    if label_entities:
        prod_cat_labels = {k: v for k, v in labels.items() if k in set(products + categories + entity_categories)}
        ent_labels = {k: v for k, v in labels.items() if k in set(entity_values)}
        nx.draw_networkx_labels(G, pos, labels=prod_cat_labels, font_size=7, font_color="#111", ax=ax)
        nx.draw_networkx_labels(G, pos, labels=ent_labels, font_size=4, font_color="#333", alpha=0.9, ax=ax)
    else:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color="#111", ax=ax)

    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize entity matching graph (schema + ego subgraph)")
    parser.add_argument(
        "--graph_dir",
        type=str,
        default=str(Path.cwd() / "processed" / "entity_matching_graph"),
        help="Directory containing node_info.pkl/edge_index.pt/etc",
    )
    parser.add_argument(
        "--asin",
        type=str,
        default="",
        help="ASIN to visualize (default: first ASIN in maps)",
    )
    parser.add_argument(
        "--max_entities",
        type=int,
        default=80,
        help="Max entity nodes to include in ego plot",
    )
    parser.add_argument(
        "--label_entities",
        action="store_true",
        default=False,
        help="Add text labels for entity nodes (may be cluttered)",
    )
    parser.add_argument(
        "--out_schema",
        type=str,
        default="",
        help="Output PNG path for schema (default: <graph_dir>/schema.png)",
    )
    parser.add_argument(
        "--out_ego",
        type=str,
        default="",
        help="Output PNG path for ego subgraph (default: <graph_dir>/ego_<asin>.png)",
    )
    args = parser.parse_args()

    g = load_graph(args.graph_dir)

    out_schema = args.out_schema or osp.join(args.graph_dir, "entity_matching_graph_schema.png")
    draw_schema_png(out_schema)

    asin = args.asin.strip()
    asin_map = g.maps.get("product_id_by_asin", {}) if isinstance(g.maps, dict) else {}
    if not asin:
        # deterministic pick: first key in sorted order
        asin = sorted(asin_map.keys())[0]
    if asin not in asin_map:
        raise ValueError(f"ASIN not found in graph maps: {asin}")
    pid = int(asin_map[asin])

    Gego = ego_subgraph(g, pid, max_entity_nodes=args.max_entities, include_category_hierarchy=True)
    out_ego = args.out_ego or osp.join(args.graph_dir, f"entity_matching_graph_ego_{asin}.png")
    title = f"Ego graph for product {asin} (categories + entity_category slots + entity_values; values<= {args.max_entities})"
    draw_subgraph_png(Gego, out_ego, title=title, label_entities=args.label_entities)
    draw_subgraph_png(Gego, out_ego, title=title, label_entities=False)

    print(f"Wrote schema PNG: {out_schema}")
    print(f"Wrote ego PNG: {out_ego}")


if __name__ == "__main__":
    main()

