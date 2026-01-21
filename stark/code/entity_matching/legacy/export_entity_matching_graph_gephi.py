#!/usr/bin/env python3
"""
Export the built entity-matching graph to a Gephi-friendly format (GEXF).

Input: directory produced by build_entity_matching_graph.py, containing:
  - node_info.pkl, node_types.pt, node_type_dict.pkl
  - edge_index.pt, edge_types.pt, edge_type_dict.pkl

Output:
  - a .gexf file that can be opened in Gephi

Notes:
  - Gephi works best with a 'label' attribute on nodes.
  - We keep the graph directed.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
from typing import Any, Dict

import networkx as nx
import torch


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_attr(v: Any) -> Any:
    """
    GEXF expects primitive attrs (str/int/float/bool). Convert complex to JSON string.
    """
    if v is None:
        # NetworkX GEXF writer does not allow None-valued attributes.
        # Caller should skip these.
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _iter_out_edges(edge_index: torch.Tensor, edge_types: torch.Tensor, src: int):
    """
    Yield (dst, edge_type_id) for edges src -> dst.
    """
    if edge_index.numel() == 0:
        return
    srcs = edge_index[0]
    mask = srcs == int(src)
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    for j in idx.tolist():
        yield int(edge_index[1, j].item()), int(edge_types[j].item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export entity matching graph to Gephi (.gexf)")
    parser.add_argument(
        "--graph_dir",
        type=str,
        default=str(os.path.join(os.getcwd(), "processed", "entity_matching_graph")),
        help="Directory containing node_info.pkl/edge_index.pt/etc",
    )
    parser.add_argument(
        "--asin",
        type=str,
        default="",
        help="If provided, export only this product's ego subgraph (product->category and product->entity nodes)",
    )
    parser.add_argument(
        "--max_values",
        type=int,
        default=200,
        help="Max entity nodes to include when --asin is set",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output .gexf path (default: <graph_dir>/entity_matching_graph.gephi.gexf)",
    )
    args = parser.parse_args()

    graph_dir = osp.abspath(args.graph_dir)
    out_path = args.out or osp.join(graph_dir, "entity_matching_graph.gephi.gexf")

    node_info: Dict[int, Dict[str, Any]] = _load_pickle(osp.join(graph_dir, "node_info.pkl"))
    node_type_dict: Dict[int, str] = _load_pickle(osp.join(graph_dir, "node_type_dict.pkl"))
    edge_type_dict: Dict[int, str] = _load_pickle(osp.join(graph_dir, "edge_type_dict.pkl"))
    node_types = torch.load(osp.join(graph_dir, "node_types.pt"))
    edge_index = torch.load(osp.join(graph_dir, "edge_index.pt"))
    edge_types = torch.load(osp.join(graph_dir, "edge_types.pt"))

    G = nx.DiGraph()

    # If exporting only one product subgraph, pick subset of node ids
    node_subset = None
    asin = args.asin.strip()
    if asin:
        maps = _load_pickle(osp.join(graph_dir, "maps.pkl"))
        asin_map = maps.get("product_id_by_asin", {}) if isinstance(maps, dict) else {}
        if asin not in asin_map:
            raise ValueError(f"ASIN not found: {asin}")
        pid = int(asin_map[asin])

        categories = []
        entities = []

        # identify category_parent_of edge type id if present
        cat_parent_ids = [k for k, v in edge_type_dict.items() if v == "category_parent_of"]
        cat_parent_id = cat_parent_ids[0] if cat_parent_ids else None

        for dst, et in _iter_out_edges(edge_index, edge_types, pid):
            if et == 0:
                categories.append(dst)
            else:
                # treat all non-category outgoing edges from product as entity edges
                entities.append(dst)

        entities = entities[: max(0, int(args.max_values))]

        node_subset = set([pid]) | set(categories) | set(entities)

        # include category hierarchy edges among selected categories if present
        if cat_parent_id is not None and categories and edge_index.numel() > 0:
            cat_set = set(categories)
            ets = edge_types.tolist()
            srcs = edge_index[0].tolist()
            dsts = edge_index[1].tolist()
            for s, d, t in zip(srcs, dsts, ets):
                if t == cat_parent_id and s in cat_set and d in cat_set:
                    node_subset.add(s)
                    node_subset.add(d)

        if not args.out:
            out_path = osp.join(graph_dir, f"entity_matching_graph.gephi_{asin}.gexf")
    else:
        node_subset = None

    # nodes
    n = int(node_types.shape[0])
    for nid in range(n):
        if node_subset is not None and nid not in node_subset:
            continue
        info = node_info.get(nid, {})
        ntype = node_type_dict.get(int(node_types[nid].item()), "unknown")

        # A readable label for Gephi
        label = None
        if ntype == "product":
            label = info.get("asin") or info.get("title") or f"product:{nid}"
        elif ntype == "category":
            label = info.get("category_name") or f"category:{nid}"
        else:
            # scheme B: entity nodes are typed by their category (Brand/Usage/...) and carry a name
            name = info.get("name")
            label = f"{ntype}:{name}" if name else f"{ntype}:{nid}"

        attrs = {"ntype": ntype, "label": str(label)}
        for k, v in info.items():
            if k == "label":
                continue
            vv = _safe_attr(v)
            if vv is None:
                continue
            attrs[k] = vv
        G.add_node(nid, **attrs)

    # edges
    if edge_index.numel() > 0:
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()
        ets = edge_types.tolist()
        for s, d, t in zip(srcs, dsts, ets):
            if node_subset is not None and (s not in node_subset or d not in node_subset):
                continue
            rel = edge_type_dict.get(int(t), str(t))
            # Use a stable key so multi-edges don't overwrite; Gephi supports parallel edges in GEXF.
            G.add_edge(int(s), int(d), rel=rel, etype=int(t))

    nx.write_gexf(G, out_path)
    print(out_path)


if __name__ == "__main__":
    main()

