#!/usr/bin/env python3
"""
Export a single product subgraph to Graphviz DOT and render to PNG.

This is designed to minimize edge crossings by enforcing layered ranks:
product -> categories / entity nodes (Scheme B)

Input graph dir should be produced by build_entity_matching_graph.py (Scheme B):
  - node_info.pkl, maps.pkl, node_types.pt, node_type_dict.pkl
  - edge_index.pt, edge_types.pt, edge_type_dict.pkl
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
import re
import subprocess
from typing import Any, Dict, List, Set

import torch


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _dot_id(prefix: str, raw: str) -> str:
    """
    Create a DOT-safe node id.
    """
    raw = str(raw)
    raw = re.sub(r"[^0-9a-zA-Z_]+", "_", raw).strip("_")
    if not raw:
        raw = "x"
    return f"{prefix}_{raw}"


def _dot_escape(label: str) -> str:
    return str(label).replace("\\", "\\\\").replace('"', '\\"')


def _iter_out_edges(edge_index: torch.Tensor, edge_types: torch.Tensor, src: int):
    if edge_index.numel() == 0:
        return
    srcs = edge_index[0]
    mask = srcs == int(src)
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    for j in idx.tolist():
        yield int(edge_index[1, j].item()), int(edge_types[j].item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one product graph to DOT/PNG (low crossings)")
    parser.add_argument(
        "--graph_dir",
        type=str,
        default=str(osp.join(os.getcwd(), "processed", "entity_matching_graph")),
        help="Directory containing built graph files",
    )
    parser.add_argument("--asin", type=str, required=True, help="ASIN to export")
    parser.add_argument("--max_values", type=int, default=200, help="Max entity_value nodes")
    parser.add_argument("--out_dot", type=str, default="", help="Output DOT path")
    parser.add_argument("--out_png", type=str, default="", help="Output PNG path (rendered by dot)")
    parser.add_argument("--rankdir", type=str, default="LR", help="Graphviz rankdir (LR/TB/RL/BT)")
    args = parser.parse_args()

    graph_dir = osp.abspath(args.graph_dir)
    asin = args.asin.strip()

    node_info: Dict[int, Dict[str, Any]] = _load_pickle(osp.join(graph_dir, "node_info.pkl"))
    maps = _load_pickle(osp.join(graph_dir, "maps.pkl"))
    node_type_dict: Dict[int, str] = _load_pickle(osp.join(graph_dir, "node_type_dict.pkl"))
    edge_type_dict: Dict[int, str] = _load_pickle(osp.join(graph_dir, "edge_type_dict.pkl"))
    node_types = torch.load(osp.join(graph_dir, "node_types.pt"))
    edge_index = torch.load(osp.join(graph_dir, "edge_index.pt"))
    edge_types = torch.load(osp.join(graph_dir, "edge_types.pt"))

    pid = int(maps["product_id_by_asin"][asin])

    # Collect nodes for subgraph (Scheme B)
    categories: List[int] = []
    entity_edges: List[tuple[int, int]] = []  # (entity_node_id, edge_type_id)

    for dst, et in _iter_out_edges(edge_index, edge_types, pid):
        if et == 0:
            categories.append(dst)
        else:
            entity_edges.append((dst, et))

    # limit entity nodes for readability
    entity_edges = entity_edges[: max(0, int(args.max_values))]
    entities = [eid for eid, _ in entity_edges]

    node_subset: Set[int] = set([pid]) | set(categories) | set(entities)

    # Add category_parent_of edges among selected categories if present
    cat_parent_ids = [k for k, v in edge_type_dict.items() if v == "category_parent_of"]
    cat_parent_id = cat_parent_ids[0] if cat_parent_ids else None
    cat_set = set(categories)
    if cat_parent_id is not None and cat_set and edge_index.numel() > 0:
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()
        ets = edge_types.tolist()
        for s, d, t in zip(srcs, dsts, ets):
            if t == cat_parent_id and s in cat_set and d in cat_set:
                node_subset.add(s)
                node_subset.add(d)

    out_dot = args.out_dot or osp.join(graph_dir, f"entity_matching_graph_{asin}.dot")
    out_png = args.out_png or osp.join(graph_dir, f"entity_matching_graph_{asin}.dot.png")

    # Build DOT
    pid_dot = _dot_id("p", asin)
    cat_dot = {cid: _dot_id("c", node_info.get(cid, {}).get("category_name", cid)) for cid in categories}
    ent_dot = {}
    ent_type_to_ids: Dict[str, List[int]] = {}
    for eid in entities:
        info = node_info.get(eid, {})
        etype = info.get("type") or node_type_dict.get(int(node_types[eid].item()), "entity")
        name = info.get("name") or str(eid)
        ent_dot[eid] = _dot_id("e", f"{etype}_{name}")
        ent_type_to_ids.setdefault(str(etype), []).append(eid)

    lines: List[str] = []
    lines.append("digraph G {")
    lines.append(f'  graph [rankdir={args.rankdir}, splines=ortho, nodesep=0.35, ranksep=0.75, concentrate=true];')
    lines.append('  node [shape=ellipse, fontsize=10];')
    lines.append('  edge [fontsize=9, color="#555555"];')

    # ranks
    lines.append("  { rank=source;")
    title = node_info.get(pid, {}).get("title", asin)
    lines.append(
        f'    {pid_dot} [label="{_dot_escape(asin)}\\n{_dot_escape(title)[:60]}", style=filled, fillcolor="#ff6b6b"];'
    )
    lines.append("  }")

    lines.append("  { rank=same;")
    for cid in categories:
        lbl = node_info.get(cid, {}).get("category_name", str(cid))
        lines.append(f'    {cat_dot[cid]} [label="{_dot_escape(lbl)}", style=filled, fillcolor="#4dabf7"];')
    lines.append("  }")

    # entity nodes grouped by entity type (clusters)
    cluster_idx = 0
    for etype, eids in sorted(ent_type_to_ids.items()):
        if not eids:
            continue
        cluster_idx += 1
        lines.append(f"  subgraph cluster_entities_{cluster_idx} {{")
        lines.append('    style="dashed"; color="#dddddd";')
        lines.append(f'    label="{_dot_escape(etype)}";')
        lines.append("    { rank=sink;")
        for eid in eids:
            name = node_info.get(eid, {}).get("name", str(eid))
            lines.append(f'      {ent_dot[eid]} [label="{_dot_escape(name)}", style=filled, fillcolor="#adb5bd"];')
        lines.append("    }")
        lines.append("  }")

    # edges
    # product -> categories
    for cid in categories:
        rel = edge_type_dict.get(0, "has_category")
        lines.append(f'  {pid_dot} -> {cat_dot[cid]} [label="{_dot_escape(rel)}"];')
    # product -> entities
    for eid, et in entity_edges:
        rel = edge_type_dict.get(int(et), str(et))
        lines.append(f'  {pid_dot} -> {ent_dot[eid]} [label="{_dot_escape(rel)}"];')

    lines.append("}")

    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # render
    subprocess.run(["dot", "-Tpng", out_dot, "-o", out_png], check=True)

    print(out_dot)
    print(out_png)


if __name__ == "__main__":
    main()

