#!/usr/bin/env python3
"""
Export the built entity matching graph to a Gephi-compatible format (.gexf).
"""

import argparse
import os
import os.path as osp
import pickle
from typing import Any, Dict

import torch
import networkx as nx


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export graph to Gephi .gexf format")
    parser.add_argument(
        "--graph_dir",
        type=str,
        default=str(osp.join(os.getcwd(), "processed", "entity_matching_graph_normalized")),
        help="Directory containing built graph files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output .gexf path (default: <graph_dir>/graph.gexf)",
    )
    args = parser.parse_args()

    graph_dir = osp.abspath(args.graph_dir)
    out_path = args.out or osp.join(graph_dir, "graph.gexf")
    out_dir = osp.dirname(out_path)
    if not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading graph from {graph_dir}...")
    
    if not osp.exists(osp.join(graph_dir, "node_info.pkl")):
         raise FileNotFoundError(f"Graph files not found in {graph_dir}.")

    node_info: Dict[int, Dict[str, Any]] = _load_pickle(osp.join(graph_dir, "node_info.pkl"))
    node_type_dict: Dict[int, str] = _load_pickle(osp.join(graph_dir, "node_type_dict.pkl"))
    edge_type_dict: Dict[int, str] = _load_pickle(osp.join(graph_dir, "edge_type_dict.pkl"))
    node_types = torch.load(osp.join(graph_dir, "node_types.pt"))
    edge_index = torch.load(osp.join(graph_dir, "edge_index.pt"))
    edge_types = torch.load(osp.join(graph_dir, "edge_types.pt"))

    # Create NetworkX Graph
    G = nx.DiGraph()

    # Add Nodes
    print("Building nodes...")
    n = int(node_types.shape[0])
    
    # Optional: Pre-define colors for Viz attributes if Gephi supports it via specific attributes
    # But usually importing 'type' column in Gephi and partitioning color by it is better.
    
    for nid in range(n):
        info = node_info.get(nid, {})
        ntype = node_type_dict.get(int(node_types[nid].item()), "unknown")
        
        # Label logic
        if ntype == "product":
            label = info.get("asin") or f"product:{nid}"
        elif ntype == "category":
            label = info.get("category_name") or f"category:{nid}"
        else:
            # For entity values, label is the value name if possible, else type
            # But wait, in previous script: label = ntype, title = value. 
            # In Gephi, we want the most descriptive text as 'Label'.
            name = info.get("name")
            if name:
                # E.g. "Blue" (Color)
                label = f"{name}" 
            else:
                label = f"{ntype}:{nid}"

        # Clean attributes for Gephi (must be simple types)
        attrs = {
            "node_type": ntype,
            "label": label
        }
        
        # Flatten info into attributes
        for k, v in info.items():
            if isinstance(v, (str, int, float, bool)):
                attrs[k] = v
            elif isinstance(v, list):
                attrs[k] = ",".join(str(x) for x in v)
            else:
                attrs[k] = str(v)

        G.add_node(nid, **attrs)

    # Add Edges
    print("Building edges...")
    if edge_index.numel() > 0:
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()
        ets = edge_types.tolist()
        
        for s, d, t in zip(srcs, dsts, ets):
            edge_label = edge_type_dict.get(int(t), str(t))
            # Rename 'type' to 'edge_type' to avoid Gephi/GEXF reserved keyword conflict
            G.add_edge(int(s), int(d), edge_type=edge_label, label=edge_label)

    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print(f"Writing GEXF to {out_path}...")
    nx.write_gexf(G, out_path)
    print("✅ Done!")

if __name__ == "__main__":
    main()
