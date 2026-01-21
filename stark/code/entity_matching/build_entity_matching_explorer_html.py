#!/usr/bin/env python3
"""
Build a *static* SKB-Explorer-like HTML for our entity-matching graph.

Why static:
- This environment doesn't ship with gradio/pyvis by default.
- We can still reproduce the same UX: input node id / hops / max edges,
  draw with vis-network, click node to show text — all in the browser, no backend.

Input graph dir is produced by build_entity_matching_graph.py (slot design):
  - node_info.pkl, maps.pkl, node_types.pt, node_type_dict.pkl
  - edge_index.pt, edge_types.pt, edge_type_dict.pkl

Output:
  - a single self-contained HTML file that embeds graph data + JS.
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
    parser = argparse.ArgumentParser(description="Build static HTML explorer for entity matching graph")
    parser.add_argument(
        "--graph_dir",
        type=str,
        default=str(osp.join(os.getcwd(), "processed", "entity_matching_graph")),
        help="Directory containing built graph files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output HTML path (default: <graph_dir>/entity_matching_explorer.html)",
    )
    parser.add_argument(
        "--default_asin",
        type=str,
        default="B000I7OIPI",
        help="Default ASIN to display on load (if present)",
    )
    args = parser.parse_args()

    graph_dir = osp.abspath(args.graph_dir)
    out_path = args.out or osp.join(graph_dir, "entity_matching_explorer.html")

    node_info: Dict[int, Dict[str, Any]] = _load_pickle(osp.join(graph_dir, "node_info.pkl"))
    maps: Dict[str, Any] = _load_pickle(osp.join(graph_dir, "maps.pkl"))
    node_type_dict: Dict[int, str] = _load_pickle(osp.join(graph_dir, "node_type_dict.pkl"))
    edge_type_dict: Dict[int, str] = _load_pickle(osp.join(graph_dir, "edge_type_dict.pkl"))
    node_types = torch.load(osp.join(graph_dir, "node_types.pt"))
    edge_index = torch.load(osp.join(graph_dir, "edge_index.pt"))
    edge_types = torch.load(osp.join(graph_dir, "edge_types.pt"))

    # Build compact node records
    n = int(node_types.shape[0])
    nodes: List[Dict[str, Any]] = []
    for nid in range(n):
        info = node_info.get(nid, {})
        ntype = node_type_dict.get(int(node_types[nid].item()), "unknown")
        # a readable label:
        # - product/category show their identifying value
        # - entity nodes show *type* only (e.g., Brand), value is shown via tooltip / Textual Info
        if ntype == "product":
            label = info.get("asin") or f"product:{nid}"
        elif ntype == "category":
            label = info.get("category_name") or f"category:{nid}"
        else:
            label = ntype
        nodes.append(
            {
                "id": nid,
                "ntype": ntype,
                "label": str(label),
                "info": info,
            }
        )

    # Edges as triples for compactness: [src, dst, etype_id]
    edges = []
    if edge_index.numel() > 0:
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()
        ets = edge_types.tolist()
        edges = [[int(s), int(d), int(t)] for s, d, t in zip(srcs, dsts, ets)]

    asin2id = maps.get("product_id_by_asin", {}) if isinstance(maps, dict) else {}
    default_id = asin2id.get(args.default_asin)
    if default_id is None:
        # fallback: first product id if exists
        if isinstance(asin2id, dict) and asin2id:
            default_id = asin2id[sorted(asin2id.keys())[0]]
        else:
            default_id = 0

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Entity Matching Graph Explorer</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/vis-network.min.js"
    integrity="sha512-4/EGWWWj7LIr/e+CvsslZkRk0fHDpf04dydJHoHOH32Mpw8jYU28GNI6mruO7fh/1kq15kSvwhKJftMSlgm0FA=="
    crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/dist/vis-network.min.css"
    integrity="sha512-WgxfT5LWjfszlPHXRmBWHkV2eceiWTOBvrKCNbdgDYTHrT2AeLCGbF4sZlZw3UMN3WtL0tGUoIAKsu8mllg/XA=="
    crossorigin="anonymous" referrerpolicy="no-referrer" />
  <style>
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      margin: 0;
      background: #0b1020;
      color: #e8edf7;
    }}
    header {{
      padding: 18px 22px;
      font-size: 20px;
      font-weight: 700;
      background: linear-gradient(180deg, #101833, #0b1020);
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    .wrap {{
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 14px;
      padding: 14px;
      align-items: start;
    }}
    .leftcol {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      height: calc(100vh - 120px);
      min-height: 520px;
    }}
    .text-card {{
      flex: 1;
      min-height: 260px;
      display: flex;
      flex-direction: column;
    }}
    .card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 10px;
    }}
    label {{
      display: block;
      font-size: 12px;
      color: rgba(232,237,247,0.75);
      margin-bottom: 6px;
    }}
    input, select {{
      width: 100%;
      box-sizing: border-box;
      padding: 10px 10px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(0,0,0,0.18);
      color: #e8edf7;
      outline: none;
    }}
    button {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 12px;
      border: 0;
      font-weight: 700;
      background: #ff7a2f;
      color: #111;
      cursor: pointer;
    }}
    button:hover {{ filter: brightness(1.04); }}
    .hint {{
      font-size: 12px;
      color: rgba(232,237,247,0.7);
      line-height: 1.45;
    }}
    #network {{
      width: 100%;
      height: calc(100vh - 120px);
      background: #ffffff;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    #info {{
      height: 100%;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.45;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
      background: rgba(0,0,0,0.25);
      border-radius: 10px;
      padding: 12px;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .pill {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 11px;
      margin-left: 8px;
      background: rgba(255,255,255,0.08);
      color: rgba(232,237,247,0.85);
    }}
  </style>
</head>
<body>
  <header>
    Entity Matching Graph Explorer
    <span class="pill">static HTML + vis-network</span>
  </header>
  <div class="wrap">
    <div class="leftcol">
    <div class="card">
      <div class="row">
        <div>
          <label>Number of Hops</label>
          <select id="hops">
            <option value="1">1</option>
            <option value="2" selected>2</option>
            <option value="inf">inf</option>
          </select>
        </div>
        <div>
          <label>Max Edges</label>
          <input id="maxEdges" type="number" min="10" max="5000" step="10" value="250" />
        </div>
      </div>
      <div class="row">
        <div>
          <label>Seed</label>
          <input id="seed" type="number" min="0" step="1" value="7" />
        </div>
        <div>
          <label>遍历限制（避免非product节点回溯拉入大量商品）</label>
          <select id="traversalMode">
            <option value="undirected" selected>默认（无向：允许从 category 回到商品）</option>
            <option value="no_back_to_product">限制（非product节点不走入边）</option>
          </select>
        </div>
      </div>
      <div class="row">
        <div>
          <label>选中商品时展开全部实体值</label>
          <select id="expandAllEntities">
            <option value="off" selected>关闭</option>
            <option value="on">开启</option>
          </select>
        </div>
        <div>
          <label>Max Entity Values（0=全部）</label>
          <input id="maxValues" type="number" min="0" max="20000" step="50" value="0" />
        </div>
      </div>
      <div class="row">
        <div>
          <label>自动显示商品完整信息数量（0=关闭）</label>
          <input id="maxProductInfos" type="number" min="0" max="50" step="1" value="0" />
        </div>
        <div>
          <label>指定节点 ID (覆盖默认行为)</label>
          <input id="nodeId" type="text" placeholder="例如: 107" />
        </div>
      </div>
      <div class="row">
        <div class="hint">
          填写“指定节点 ID”后，点击按钮将直接以该节点为中心进行渲染。
        </div>
      </div>
      <button id="renderBtn">Display Semi-structured Data</button>
      <div style="height:10px"></div>
      <div class="hint">
        - hops=1/2 会做 BFS；hops=inf 会随机采样边。<br/>
        - 点击节点会在左下角显示文本信息。<br/>
        - 开启“展开全部实体值”后，点击商品节点会强制显示该商品全部实体节点（可用 Max Entity Values 限制）。<br/>
        - 开启“遍历限制”后，category 节点不会通过入边回溯到大量 product，从而避免图爆炸。<br/>
        - 设置“自动显示商品完整信息数量”后，渲染后会自动展示多个商品的完整信息（categories + entities）。<br/>
      </div>
    </div>

    <div class="card text-card">
      <div style="font-weight:700; margin-bottom:10px;">Textual Info</div>
      <div id="info">点击图里的节点查看信息。</div>
    </div>
    </div>

    <div id="network"></div>
  </div>

  <script>
    // In scheme B, entity nodes are typed by category name (Brand/Usage/...)
    // We treat any non-product/category node as an "entity" visually.
    const NODE_COLORS = {{
      "product": "#ff6b6b",
      "category": "#4dabf7",
      "entity": "#adb5bd",
      "unknown": "#c0c0c0",
    }};

    const edgeTypeDict = {_safe_json(edge_type_dict)};
    const asin2id = {_safe_json(asin2id)};

    // node array indexed by original node id
    const nodesById = {_safe_json(nodes)};
    const edgesTriples = {_safe_json(edges)};

    function rand(seed) {{
      // mulberry32
      let t = seed >>> 0;
      return function() {{
        t += 0x6D2B79F5;
        let r = Math.imul(t ^ (t >>> 15), 1 | t);
        r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
      }}
    }}

    function pickFirstNProductsByAsin(n) {{
      const asins = Object.keys(asin2id || {{}});
      asins.sort();
      const out = [];
      for (const a of asins) {{
        const pid = asin2id[a];
        if (pid === undefined || pid === null) continue;
        out.push(pid);
        if (out.length >= n) break;
      }}
      return out;
    }}

    function buildAdj() {{
      const out = new Map();
      const inn = new Map();
      for (const [s,d,t] of edgesTriples) {{
        if (!out.has(s)) out.set(s, []);
        if (!inn.has(d)) inn.set(d, []);
        out.get(s).push([d,t]);
        inn.get(d).push([s,t]);
      }}
      return {{out, inn}};
    }}

    const ADJ = buildAdj();

    function neighborsByMode(nid, mode) {{
      // mode:
      // - undirected: use out + in
      // - no_back_to_product: for non-product nodes, only use out edges (prevents backtracking to many products)
      const rec = nodesById[nid];
      const ntype = rec ? rec.ntype : "unknown";

      const res = [];
      const o = ADJ.out.get(nid) || [];
      for (const [d,t] of o) res.push(d);

      if (mode === "undirected") {{
        const i = ADJ.inn.get(nid) || [];
        for (const [s,t] of i) res.push(s);
      }} else if (mode === "no_back_to_product") {{
        if (ntype === "product") {{
          const i = ADJ.inn.get(nid) || [];
          for (const [s,t] of i) res.push(s);
        }}
      }}

      return res;
    }}

    function collectSubgraph(center, hops, maxEdges, seed, traversalMode) {{
      const nodeSet = new Set();
      const edgeIdx = [];

      if (hops === "inf") {{
        const r = rand(seed);
        const idxs = [...Array(edgesTriples.length).keys()];
        for (let i = idxs.length - 1; i > 0; i--) {{
          const j = Math.floor(r() * (i + 1));
          [idxs[i], idxs[j]] = [idxs[j], idxs[i]];
        }}
        const picked = idxs.slice(0, Math.min(maxEdges, idxs.length));
        for (const k of picked) edgeIdx.push(k);
        for (let k = 0; k < edgesTriples.length; k++) {{
          const [s,d,t] = edgesTriples[k];
          if (s === center || d === center) edgeIdx.push(k);
        }}
      }} else {{
        const k = parseInt(hops, 10);
        const q = [center];
        const dist = new Map([[center, 0]]);
        nodeSet.add(center);
        while (q.length) {{
          const cur = q.shift();
          const cd = dist.get(cur);
          if (cd >= k) continue;
          for (const nb of neighborsByMode(cur, traversalMode)) {{
            if (!dist.has(nb)) {{
              dist.set(nb, cd + 1);
              q.push(nb);
              nodeSet.add(nb);
            }}
          }}
        }}
        for (let k = 0; k < edgesTriples.length; k++) {{
          const [s,d,t] = edgesTriples[k];
          if (nodeSet.has(s) && nodeSet.has(d)) edgeIdx.push(k);
        }}
        if (edgeIdx.length > maxEdges) {{
          const keep = [];
          const rest = [];
          for (const k of edgeIdx) {{
            const [s,d,t] = edgesTriples[k];
            if (s === center || d === center) keep.push(k);
            else rest.push(k);
          }}
          const r = rand(seed);
          for (let i = rest.length - 1; i > 0; i--) {{
            const j = Math.floor(r() * (i + 1));
            [rest[i], rest[j]] = [rest[j], rest[i]];
          }}
          edgeIdx.length = 0;
          for (const k of keep) edgeIdx.push(k);
          for (const k of rest.slice(0, Math.max(0, maxEdges - keep.length))) edgeIdx.push(k);
        }}
      }}

      for (const k of edgeIdx) {{
        const [s,d,t] = edgesTriples[k];
        nodeSet.add(s); nodeSet.add(d);
      }}
      return {{nodeSet, edgeIdx}};
    }}

    function collectSubgraphMulti(centers, hops, maxEdges, seed, traversalMode) {{
      // centers: array of node ids
      const nodeSet = new Set();
      const edgeIdx = [];
      const centerSet = new Set(centers || []);

      if (centerSet.size === 0) {{
        return {{nodeSet, edgeIdx}};
      }}

      if (hops === "inf") {{
        const r = rand(seed);
        const idxs = [...Array(edgesTriples.length).keys()];
        for (let i = idxs.length - 1; i > 0; i--) {{
          const j = Math.floor(r() * (i + 1));
          [idxs[i], idxs[j]] = [idxs[j], idxs[i]];
        }}
        const picked = idxs.slice(0, Math.min(maxEdges, idxs.length));
        for (const k of picked) edgeIdx.push(k);
        // ensure all centers appear by adding incident edges
        for (let k = 0; k < edgesTriples.length; k++) {{
          const [s,d,t] = edgesTriples[k];
          if (centerSet.has(s) || centerSet.has(d)) edgeIdx.push(k);
        }}
      }} else {{
        const k = parseInt(hops, 10);
        const q = [];
        const dist = new Map();
        for (const c of centerSet) {{
          q.push(c);
          dist.set(c, 0);
          nodeSet.add(c);
        }}
        while (q.length) {{
          const cur = q.shift();
          const cd = dist.get(cur);
          if (cd >= k) continue;
          for (const nb of neighborsByMode(cur, traversalMode)) {{
            if (!dist.has(nb)) {{
              dist.set(nb, cd + 1);
              q.push(nb);
              nodeSet.add(nb);
            }}
          }}
        }}
        // collect all edges inside nodeSet
        for (let k = 0; k < edgesTriples.length; k++) {{
          const [s,d,t] = edgesTriples[k];
          if (nodeSet.has(s) && nodeSet.has(d)) edgeIdx.push(k);
        }}
        // sample edges if too many, but keep those incident to ANY center
        if (edgeIdx.length > maxEdges) {{
          const keep = [];
          const rest = [];
          for (const k of edgeIdx) {{
            const [s,d,t] = edgesTriples[k];
            if (centerSet.has(s) || centerSet.has(d)) keep.push(k);
            else rest.push(k);
          }}
          const r = rand(seed);
          for (let i = rest.length - 1; i > 0; i--) {{
            const j = Math.floor(r() * (i + 1));
            [rest[i], rest[j]] = [rest[j], rest[i]];
          }}
          edgeIdx.length = 0;
          for (const k of keep) edgeIdx.push(k);
          for (const k of rest.slice(0, Math.max(0, maxEdges - keep.length))) edgeIdx.push(k);
        }}
      }}

      // derive nodeSet from chosen edges (for inf mode)
      for (const k of edgeIdx) {{
        const [s,d,t] = edgesTriples[k];
        nodeSet.add(s); nodeSet.add(d);
      }}
      return {{nodeSet, edgeIdx}};
    }}

    function collectProductAllEntities(center, maxValues) {{
      // Designed for Scheme B:
      // product --(0)--> category
      // product --(has_*)--> entity nodes (typed by category name)
      // category --(category_parent_of)--> category (optional)
      const nodeSet = new Set();
      const edgeTriples = [];
      nodeSet.add(center);

      const categories = [];
      const entities = [];

      const out = ADJ.out.get(center) || [];
      for (const [d,t] of out) {{
        if (t === 0) {{
          categories.push(d);
          nodeSet.add(d);
          edgeTriples.push([center, d, t]);
        }} else {{
          // treat all non-category edges as entity edges
          entities.push([d, t]);
        }}
      }}

      let cnt = 0;
      for (const [eid, etype] of entities) {{
        if (maxValues > 0 && cnt >= maxValues) break;
        cnt += 1;
        nodeSet.add(eid);
        edgeTriples.push([center, eid, etype]);
      }}

      // category hierarchy edges among included categories
      const catSet = new Set(categories);
      for (const c of categories) {{
        const cout = ADJ.out.get(c) || [];
        for (const [d,t] of cout) {{
          // last edge type in our builder is category_parent_of; treat by name if possible
          // but here we just keep numeric 11 (default) and ignore others.
          if (edgeTypeDict[t] === "category_parent_of" && catSet.has(d)) {{
            edgeTriples.push([c, d, t]);
          }}
        }}
      }}

      return {{nodeSet, edgeTriples}};
    }}

    function collectProductsAllEntities(centers, maxValuesPerProduct) {{
      const nodeSet = new Set();
      const edgeKey = new Set();
      const edgeTriples = [];

      for (const c of centers) {{
        const rec = nodesById[c];
        if (!rec || rec.ntype !== "product") continue;
        const one = collectProductAllEntities(c, maxValuesPerProduct);
        for (const nid of one.nodeSet) nodeSet.add(nid);
        for (const e of one.edgeTriples) {{
          const key = e[0] + "|" + e[1] + "|" + e[2];
          if (!edgeKey.has(key)) {{
            edgeKey.add(key);
            edgeTriples.push(e);
          }}
        }}
      }}
      return {{nodeSet, edgeTriples}};
    }}

    function edgeLabelWithSentiment(s, d, t) {{
      // If the edge connects a product to an entity node (non-product/non-category),
      // use the entity node's sentiment field as the edge label (the attitude towards that entity).
      const src = nodesById[s];
      const dst = nodesById[d];
      const srcType = src?.ntype || "unknown";
      const dstType = dst?.ntype || "unknown";

      const normalizeSentimentLabel = (raw) => {{
        if (typeof raw !== "string") return null;
        let s = raw.trim();
        if (!s) return null;
        // Handle "sentiment:positive" / "sentiment：positive" -> "positive"
        const lower = s.toLowerCase();
        if (lower.startsWith("sentiment:")) s = s.slice("sentiment:".length).trim();
        if (lower.startsWith("sentiment：")) s = s.slice("sentiment：".length).trim();
        // Common typo seen in some outputs
        if (s.toLowerCase() === "posotive") s = "positive";
        return s || null;
      }};

      const getSent = (node) => normalizeSentimentLabel(node?.info?.sentiment);

      const isEntity = (nt) => nt !== "product" && nt !== "category";
      const connectsProductEntity =
        (srcType === "product" && isEntity(dstType)) ||
        (dstType === "product" && isEntity(srcType));

      if (connectsProductEntity) {{
        const sent = getSent(isEntity(dstType) ? dst : src);
        if (sent) return sent;  // Directly show: positive/negative/neutral
      }}

      return edgeTypeDict[t] || String(t);
    }}

    function buildVisData(center, nodeSet, edgeIdx, centersSet) {{
      const idMap = new Map();
      let cur = 0;
      for (const nid of nodeSet) {{
        idMap.set(nid, cur++);
      }}

      const visNodes = [];
      for (const nid of nodeSet) {{
        const rec = nodesById[nid];
        const ntype = rec?.ntype || "unknown";
        const baseColor = (ntype === "product" || ntype === "category") ? (NODE_COLORS[ntype] || NODE_COLORS.unknown) : NODE_COLORS.entity;
        let size = 16;
        if (ntype === "product") size = 28;
        else if (ntype === "category") size = 20;
        else size = 14;
        const label = rec?.label || String(nid);

        let color = baseColor;
        let borderWidth = 1;
        if (centersSet && centersSet.has(nid)) {{
          borderWidth = 3;
          size = size + 6;
          color = {{ background: baseColor, border: "#111111" }};
        }}

        // show value in tooltip for entity nodes
        let title = "";
        if (ntype !== "product" && ntype !== "category") {{
          const v = rec?.info?.name;
          title = (v !== undefined && v !== null) ? (ntype + ": " + v) : ntype;
        }} else {{
          title = label;
        }}

        visNodes.push({{
          id: idMap.get(nid),
          node_id: nid,
          label: label,
          title: title,
          color: color,
          borderWidth: borderWidth,
          size: size,
          font: {{ align: "middle", size: (ntype === "entity_value" ? 10 : 12) }},
        }});
      }}

      const visEdges = [];
      for (const k of edgeIdx) {{
        const [s,d,t] = edgesTriples[k];
        if (!idMap.has(s) || !idMap.has(d)) continue;
        visEdges.push({{
          from: idMap.get(s),
          to: idMap.get(d),
          color: "#555",
          arrows: "to",
          arrowStrikethrough: false,
          label: edgeLabelWithSentiment(s, d, t),
          font: {{ align: "middle", size: 10 }},
          width: 1,
        }});
      }}
      return {{visNodes, visEdges}};
    }}

    function buildVisDataFromTriples(center, nodeSet, edgeTriples, centersSet) {{
      const idMap = new Map();
      let cur = 0;
      for (const nid of nodeSet) {{
        idMap.set(nid, cur++);
      }}

      const visNodes = [];
      for (const nid of nodeSet) {{
        const rec = nodesById[nid];
        const ntype = rec?.ntype || "unknown";
        const baseColor = (ntype === "product" || ntype === "category") ? (NODE_COLORS[ntype] || NODE_COLORS.unknown) : NODE_COLORS.entity;
        let size = 16;
        if (ntype === "product") size = 28;
        else if (ntype === "category") size = 20;
        else size = 14;
        const label = rec?.label || String(nid);

        let color = baseColor;
        let borderWidth = 1;
        if (centersSet && centersSet.has(nid)) {{
          borderWidth = 3;
          size = size + 6;
          color = {{ background: baseColor, border: "#111111" }};
        }}

        // show value in tooltip for entity nodes
        let title = "";
        if (ntype !== "product" && ntype !== "category") {{
          const v = rec?.info?.name;
          title = (v !== undefined && v !== null) ? (ntype + ": " + v) : ntype;
        }} else {{
          title = label;
        }}

        visNodes.push({{
          id: idMap.get(nid),
          node_id: nid,
          label: label,
          title: title,
          color: color,
          borderWidth: borderWidth,
          size: size,
          font: {{ align: "middle", size: (ntype === "entity_value" ? 10 : 12) }},
        }});
      }}

      const visEdges = [];
      for (const [s,d,t] of edgeTriples) {{
        if (!idMap.has(s) || !idMap.has(d)) continue;
        visEdges.push({{
          from: idMap.get(s),
          to: idMap.get(d),
          color: "#555",
          arrows: "to",
          arrowStrikethrough: false,
          label: edgeLabelWithSentiment(s, d, t),
          font: {{ align: "middle", size: 10 }},
          width: 1,
        }});
      }}
      return {{visNodes, visEdges}};
    }}

    function prettyNodeInfo(nid) {{
      const rec = nodesById[nid];
      if (!rec) return "node not found: " + nid;
      const ntype = rec.ntype || "unknown";
      const info = rec.info || {{}};
      const lines = [];
      lines.push("node_id: " + nid);
      lines.push("type: " + ntype);
      lines.push("label: " + rec.label);
      lines.push("");
      for (const [k,v] of Object.entries(info)) {{
        // matched_entities is now a dict in latest outputs; pretty print it for readability.
        if (k === "matched_entities" && v && typeof v === "object") {{
          lines.push(k + ": " + JSON.stringify(v, null, 2));
          continue;
        }}
        lines.push(k + ": " + (typeof v === "object" ? JSON.stringify(v) : v));
      }}
      return lines.join("\\n");
    }}

    function prettyProductFullInfo(pid, maxValues) {{
      const rec = nodesById[pid];
      if (!rec) return "node not found: " + pid;
      if ((rec.ntype || "unknown") !== "product") return prettyNodeInfo(pid);

      const info = rec.info || {{}};
      const lines = [];
      lines.push("======== PRODUCT ========");
      lines.push("node_id: " + pid);
      if (info.asin !== undefined) lines.push("asin: " + info.asin);
      if (info.title !== undefined) lines.push("title: " + info.title);
      lines.push("");

      // raw fields
      for (const [k,v] of Object.entries(info)) {{
        if (k === "asin" || k === "title") continue;
        lines.push(k + ": " + (typeof v === "object" ? JSON.stringify(v) : v));
      }}

      // categories + entities from outgoing edges (full, independent of maxEdges sampling)
      const out = ADJ.out.get(pid) || [];
      const catNames = [];
      const groups = new Map(); // entityType -> [values]

      let seenEnt = 0;
      let truncated = false;
      for (const [d,t] of out) {{
        const drec = nodesById[d];
        if (!drec) continue;
        if (t === 0 || drec.ntype === "category") {{
          const cn = (drec.info && drec.info.category_name) ? String(drec.info.category_name) : (drec.label || String(d));
          catNames.push(cn);
          continue;
        }}
        if (maxValues > 0 && seenEnt >= maxValues) {{
          truncated = true;
          continue;
        }}
        seenEnt += 1;
        const et = drec.ntype || "entity";
        const val = (drec.info && drec.info.name !== undefined && drec.info.name !== null) ? String(drec.info.name) : (drec.label || String(d));
        if (!groups.has(et)) groups.set(et, []);
        groups.get(et).push(val);
      }}

      // categories
      const catSet = new Set(catNames.filter(Boolean));
      const catList = Array.from(catSet);
      catList.sort();
      lines.push("");
      lines.push("-------- categories (" + catList.length + ") --------");
      for (const cn of catList) lines.push("- " + cn);

      // entities grouped
      const keys = Array.from(groups.keys());
      keys.sort();
      lines.push("");
      lines.push("-------- entities --------");
      for (const k of keys) {{
        const vals = groups.get(k) || [];
        const set = new Set(vals.filter(Boolean));
        const arr = Array.from(set);
        arr.sort();
        lines.push(k + " (" + arr.length + "): " + arr.join("; "));
      }}
      if (truncated) {{
        lines.push("");
        lines.push("[截断提示] 实体数量超过 Max Entity Values=" + maxValues + "，已截断显示。");
      }}
      return lines.join("\\n");
    }}

    function pickProductsForInfo(nodeSet, centers, maxN) {{
      const picked = [];
      const seen = new Set();
      function add(pid) {{
        if (picked.length >= maxN) return;
        if (seen.has(pid)) return;
        const r = nodesById[pid];
        if (!r || r.ntype !== "product") return;
        seen.add(pid);
        picked.push(pid);
      }}

      // prefer centers first (in order)
      for (const c of (centers || [])) add(c);
      if (picked.length >= maxN) return picked;

      // then any other product nodes in current subgraph (sorted by asin then id)
      const others = [];
      for (const nid of (nodeSet || [])) {{
        const r = nodesById[nid];
        if (r && r.ntype === "product" && !seen.has(nid)) {{
          const asin = (r.info && r.info.asin) ? String(r.info.asin) : "";
          others.push([asin, nid]);
        }}
      }}
      others.sort((a,b) => {{
        if (a[0] < b[0]) return -1;
        if (a[0] > b[0]) return 1;
        return a[1] - b[1];
      }});
      for (const [asin, nid] of others) add(nid);
      return picked;
    }}

    function maybeAutoShowProductInfos(nodeSet, centers) {{
      const n = parseInt(document.getElementById("maxProductInfos").value, 10) || 0;
      if (n <= 0) return false;
      const mv = parseInt(document.getElementById("maxValues").value, 10) || 0;
      const pids = pickProductsForInfo(nodeSet, centers, n);
      if (!pids.length) return false;
      const parts = [];
      for (const pid of pids) {{
        parts.push(prettyProductFullInfo(pid, mv));
      }}
      document.getElementById("info").textContent = parts.join("\\n\\n");
      return true;
    }}

    function drawMulti(centers, hops, maxEdges, seed, traversalMode) {{
      const centerArr = centers || [];
      const centersSet = new Set(centerArr);
      const primary = centerArr.length ? centerArr[0] : null;

      const {{nodeSet, edgeIdx}} = collectSubgraphMulti(centerArr, hops, maxEdges, seed, traversalMode);
      const {{visNodes, visEdges}} = buildVisData(primary, nodeSet, edgeIdx, centersSet);

      const container = document.getElementById("network");
      const data = {{
        nodes: new vis.DataSet(visNodes),
        edges: new vis.DataSet(visEdges),
      }};
      const options = {{
        edges: {{
          smooth: {{ enabled: true, type: "dynamic" }},
          color: {{ inherit: true }},
        }},
        interaction: {{
          dragNodes: true,
          zoomSpeed: 0.7,
        }},
        physics: {{
          enabled: true,
          stabilization: {{
            enabled: true,
            fit: true,
            iterations: 800,
          }},
        }},
      }};

      const network = new vis.Network(container, data, options);
      if (primary !== null) {{
        document.getElementById("info").textContent = prettyNodeInfo(primary);
      }}
      maybeAutoShowProductInfos(nodeSet, centerArr);

      network.on("selectNode", (e) => {{
        const vid = e.nodes[0];
        const node = data.nodes.get(vid);
        const nid = node.node_id;
        document.getElementById("info").textContent = prettyNodeInfo(nid);

        const expand = document.getElementById("expandAllEntities").value === "on";
        const rec = nodesById[nid];
        if (expand && rec && rec.ntype === "product") {{
          const mv = parseInt(document.getElementById("maxValues").value, 10) || 0;
          drawAllEntitiesForProduct(nid, mv, parseInt(document.getElementById("seed").value, 10) || 7);
        }}
      }});
    }}

    function drawAllEntitiesForProduct(center, maxValues, seed) {{
      const {{nodeSet, edgeTriples}} = collectProductAllEntities(center, maxValues);
      const {{visNodes, visEdges}} = buildVisDataFromTriples(center, nodeSet, edgeTriples, new Set([center]));

      const container = document.getElementById("network");
      const data = {{
        nodes: new vis.DataSet(visNodes),
        edges: new vis.DataSet(visEdges),
      }};
      const options = {{
        edges: {{
          smooth: {{ enabled: true, type: "dynamic" }},
          color: {{ inherit: true }},
        }},
        interaction: {{
          dragNodes: true,
          zoomSpeed: 0.7,
        }},
        physics: {{
          enabled: true,
          stabilization: {{
            enabled: true,
            fit: true,
            iterations: 800,
          }},
        }},
      }};

      const network = new vis.Network(container, data, options);
      document.getElementById("info").textContent = prettyNodeInfo(center);
      maybeAutoShowProductInfos(nodeSet, [center]);

      network.on("selectNode", (e) => {{
        const vid = e.nodes[0];
        const node = data.nodes.get(vid);
        const nid = node.node_id;
        document.getElementById("info").textContent = prettyNodeInfo(nid);

        const expand = document.getElementById("expandAllEntities").value === "on";
        const rec = nodesById[nid];
        if (expand && rec && rec.ntype === "product") {{
          const mv = parseInt(document.getElementById("maxValues").value, 10) || 0;
          drawAllEntitiesForProduct(nid, mv, parseInt(document.getElementById("seed").value, 10) || 7);
        }}
      }});
    }}

    function drawAllEntitiesForProducts(centers, maxValuesPerProduct, seed) {{
      const centerArr = centers || [];
      const centersSet = new Set(centerArr);
      const primary = centerArr.length ? centerArr[0] : null;

      const {{nodeSet, edgeTriples}} = collectProductsAllEntities(centerArr, maxValuesPerProduct);
      const {{visNodes, visEdges}} = buildVisDataFromTriples(primary, nodeSet, edgeTriples, centersSet);

      const container = document.getElementById("network");
      const data = {{
        nodes: new vis.DataSet(visNodes),
        edges: new vis.DataSet(visEdges),
      }};
      const options = {{
        edges: {{
          smooth: {{ enabled: true, type: "dynamic" }},
          color: {{ inherit: true }},
        }},
        interaction: {{
          dragNodes: true,
          zoomSpeed: 0.7,
        }},
        physics: {{
          enabled: true,
          stabilization: {{
            enabled: true,
            fit: true,
            iterations: 800,
          }},
        }},
      }};

      const network = new vis.Network(container, data, options);
      if (primary !== null) {{
        document.getElementById("info").textContent = prettyNodeInfo(primary);
      }}
      maybeAutoShowProductInfos(nodeSet, centerArr);

      network.on("selectNode", (e) => {{
        const vid = e.nodes[0];
        const node = data.nodes.get(vid);
        const nid = node.node_id;
        document.getElementById("info").textContent = prettyNodeInfo(nid);

        const expand = document.getElementById("expandAllEntities").value === "on";
        const rec = nodesById[nid];
        if (expand && rec && rec.ntype === "product") {{
          const mv = parseInt(document.getElementById("maxValues").value, 10) || 0;
          drawAllEntitiesForProduct(nid, mv, parseInt(document.getElementById("seed").value, 10) || 7);
        }}
      }});
    }}

    document.getElementById("renderBtn").addEventListener("click", () => {{
      const hops = document.getElementById("hops").value;
      const maxEdges = parseInt(document.getElementById("maxEdges").value, 10) || 250;
      const seed = parseInt(document.getElementById("seed").value, 10) || 7;
      const expand = document.getElementById("expandAllEntities").value === "on";
      const maxValues = parseInt(document.getElementById("maxValues").value, 10) || 0;
      const traversalMode = document.getElementById("traversalMode").value || "undirected";
      const n = parseInt(document.getElementById("maxProductInfos").value, 10) || 0;

      const nodeIdStr = document.getElementById("nodeId").value.trim();
      if (nodeIdStr) {{
        const customId = parseInt(nodeIdStr, 10);
        if (isNaN(customId) || !nodesById[customId]) {{
          alert("无效的节点 ID: " + nodeIdStr);
          return;
        }}
        drawMulti([customId], hops, maxEdges, seed, traversalMode);
        return;
      }}

      // If maxProductInfos>0, deterministically pick first N products by ASIN and show their full entities.
      if (n > 0) {{
        const centers = pickFirstNProductsByAsin(n);
        if (!centers.length) {{
          alert("没有可用的商品 ASIN（asin2id 为空）");
          return;
        }}
        drawAllEntitiesForProducts(centers, maxValues, seed);
      }} else {{
        // fallback: render a default product's local neighborhood
        drawMulti([{default_id}], hops, maxEdges, seed, traversalMode);
      }}
    }});

    // initial render
    drawMulti([{default_id}], "2", 250, 7, "undirected");
  </script>
</body>
</html>
"""

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(out_path)


if __name__ == "__main__":
    main()

