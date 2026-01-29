#!/usr/bin/env python3
"""
Build a static HTML to display the FULL entity matching graph at once.
This is suitable for small to medium graphs (e.g. < 2000 nodes).
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
from typing import Any, Dict, List

import torch


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static HTML for FULL entity matching graph")
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
        help="Output HTML path (default: <graph_dir>/full_graph.html)",
    )
    args = parser.parse_args()

    graph_dir = osp.abspath(args.graph_dir)
    out_path = args.out or osp.join(graph_dir, "full_graph.html")

    print(f"Loading graph from {graph_dir}...")
    
    if not osp.exists(osp.join(graph_dir, "node_info.pkl")):
         raise FileNotFoundError(f"Graph files not found in {graph_dir}. Did you run build_entity_matching_graph.py?")

    node_info: Dict[int, Dict[str, Any]] = _load_pickle(osp.join(graph_dir, "node_info.pkl"))
    # maps: Dict[str, Any] = _load_pickle(osp.join(graph_dir, "maps.pkl"))
    node_type_dict: Dict[int, str] = _load_pickle(osp.join(graph_dir, "node_type_dict.pkl"))
    edge_type_dict: Dict[int, str] = _load_pickle(osp.join(graph_dir, "edge_type_dict.pkl"))
    node_types = torch.load(osp.join(graph_dir, "node_types.pt"))
    edge_index = torch.load(osp.join(graph_dir, "edge_index.pt"))
    edge_types = torch.load(osp.join(graph_dir, "edge_types.pt"))

    # Build vis-network nodes
    print("Building nodes...")
    n = int(node_types.shape[0])
    vis_nodes = []
    
    # Colors
    NODE_COLORS = {
      "product": "#ff6b6b",
      "category": "#4dabf7",
      "entity": "#adb5bd",
      "unknown": "#c0c0c0",
    }

    for nid in range(n):
        info = node_info.get(nid, {})
        ntype = node_type_dict.get(int(node_types[nid].item()), "unknown")
        
        # Label logic
        if ntype == "product":
            label = info.get("asin") or f"product:{nid}"
            size = 20
        elif ntype == "category":
            label = info.get("category_name") or f"category:{nid}"
            size = 15
        else:
            # For entity values, the type is the label (e.g. "Color"), value is in tooltip
            label = ntype
            size = 10
            
        # Tooltip logic
        title = ""
        if ntype != "product" and ntype != "category":
             v = info.get("name")
             title = f"{ntype}: {v}" if v else ntype
             # Optionally, show the value as label for small graphs? 
             # Let's append value to label for better visibility in full graph
             label = f"{v}" 
        else:
             title = label
             if ntype == "product":
                 title += f"\n{info.get('product_title', '')[:100]}..."

        color = NODE_COLORS.get(ntype, NODE_COLORS["entity"])
        if ntype not in ["product", "category"]:
             color = NODE_COLORS["entity"]

        vis_nodes.append({
            "id": nid,
            "label": label,
            "title": title,
            "group": ntype,
            "color": color,
            "size": size,
            "font": {"size": 12, "align": "middle"}
        })

    # Build vis-network edges
    print("Building edges...")
    vis_edges = []
    if edge_index.numel() > 0:
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()
        ets = edge_types.tolist()
        
        for s, d, t in zip(srcs, dsts, ets):
            label = edge_type_dict.get(int(t), str(t))
            # Optional: Hide "category_parent_of" labels to reduce clutter
            if label == "category_parent_of": label = ""
            
            vis_edges.append({
                "from": int(s),
                "to": int(d),
                "label": label,
                "arrows": "to",
                "color": {"color": "#848484", "opacity": 0.5},
                "font": {"size": 8, "align": "middle"}
            })

    print(f"Graph stats: {len(vis_nodes)} nodes, {len(vis_edges)} edges")

    html_content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Full Knowledge Graph</title>
  <style>
    body {{ margin: 0; padding: 0; overflow: hidden; background: #0b1020; color: #e8edf7; font-family: sans-serif; }}
    #mynetwork {{ width: 100vw; height: 100vh; }}
    #overlay {{
        position: absolute; top: 10px; left: 10px; z-index: 10;
        background: rgba(0,0,0,0.6); padding: 10px; border-radius: 8px;
        pointer-events: none;
    }}
    #loading {{
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        font-size: 24px; font-weight: bold; background: rgba(0,0,0,0.8);
        padding: 20px; border-radius: 10px; z-index: 100;
        text-align: center;
    }}
    .legend {{ display: flex; align-items: center; margin-bottom: 5px; font-size: 12px; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}
    .error {{ color: #ff6b6b; }}
  </style>
  <!-- Try unpkg standalone build first -->
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js" onerror="document.getElementById('loading').innerHTML='<span class=error>Failed to load vis-network.js from CDN.<br>Please check your internet connection.</span>'"></script>
</head>
<body>
  <div id="loading">Loading graph data and rendering...<br><span style="font-size:14px;font-weight:normal">(This may take a few seconds)</span></div>
  
  <div id="overlay">
    <h3 style="margin:0 0 10px 0;">Knowledge Graph ({len(vis_nodes)} nodes)</h3>
    <div class="legend"><div class="dot" style="background:#ff6b6b"></div>Product</div>
    <div class="legend"><div class="dot" style="background:#4dabf7"></div>Category</div>
    <div class="legend"><div class="dot" style="background:#adb5bd"></div>Entity Value</div>
    <div style="font-size:10px; color:#aaa; margin-top:5px;">Scroll to zoom, Drag to pan</div>
  </div>
  <div id="mynetwork"></div>

  <script type="text/javascript">
    window.onload = function() {{
        if (typeof vis === 'undefined') {{
            document.getElementById('loading').innerHTML = '<span class="error">Error: Vis.js library not loaded.<br>CDN access required.</span>';
            return;
        }}

        try {{
            const nodes = new vis.DataSet({_safe_json(vis_nodes)});
            const edges = new vis.DataSet({_safe_json(vis_edges)});

            const container = document.getElementById('mynetwork');
            const data = {{ nodes: nodes, edges: edges }};
            const options = {{
              nodes: {{
                shape: 'dot',
                scaling: {{ min: 10, max: 30 }}
              }},
              edges: {{
                color: {{ inherit: false, color: '#848484', opacity: 0.5 }},
                smooth: {{ type: 'continuous' }}
              }},
              physics: {{
                stabilization: {{ 
                    enabled: true,
                    iterations: 1000,
                    updateInterval: 25
                }},
                barnesHut: {{
                    gravitationalConstant: -10000,
                    centralGravity: 0.3,
                    springLength: 95,
                    springConstant: 0.04
                }}
              }},
              interaction: {{
                hideEdgesOnDrag: true,
                tooltipDelay: 200
              }}
            }};
            
            const network = new vis.Network(container, data, options);
            
            network.on("stabilizationProgress", function(params) {{
                const percent = Math.round((params.iterations / params.total) * 100);
                document.getElementById('loading').innerHTML = 'Stabilizing graph layout... ' + percent + '%';
            }});

            network.once("stabilizationIterationsDone", function() {{
                document.getElementById('loading').style.display = 'none';
                network.fit();
            }});
            
        }} catch (err) {{
            document.getElementById('loading').innerHTML = '<span class="error">Javascript Error: ' + err.message + '</span>';
            console.error(err);
        }}
    }};
  </script>
</body>
</html>
    """

    print(f"Writing HTML to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✅ Done!")

if __name__ == "__main__":
    main()
