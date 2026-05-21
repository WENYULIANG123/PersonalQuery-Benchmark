#!/usr/bin/env python3
"""Stage 12: 用多种无监督潜因子方法分析 Query 复杂度并评估 hit@10。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from collections import Counter
from datetime import datetime
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import nltk
import numpy as np
import spacy
from benepar import InputSentence, Parser as BeneparParser
from scipy.stats import kruskal
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/fs04/ar57/wenyu")
RESULT_ROOT = REPO_ROOT / "result" / "personal_query"
QUERY_ROOT = RESULT_ROOT / "06_query"
RETRIEVAL_ROOT = RESULT_ROOT / "08_retrieval"
STAGE12_ROOT = RESULT_ROOT / "12_complexity_analysis"
STAGE6_AUDIT_PATH = REPO_ROOT / "PersoanlQuery" / "06_query" / "06_audit_attr_usage.py"

DEFAULT_CATEGORIES = (
    "Baby_Products",
    "Grocery_and_Gourmet_Food",
    "Pet_Supplies",
)
QUERY_TYPES = ("acl_query", "ccomp_query")
QUERY_TYPE_TO_RETRIEVAL_CATEGORY = {
    "acl_query": "acl",
    "ccomp_query": "ccomp",
}
RETRIEVAL_CATEGORY_TO_QUERY_TYPE = {
    "acl": "acl_query",
    "ccomp": "ccomp_query",
}

BENEPAR_MODEL_NAME = "benepar_en3"
LOCAL_NLTK_DATA = Path("/home/wlia0047/nltk_data")

CLAUSE_DEP_TYPES = {
    "acl",
    "relcl",
    "advcl",
    "ccomp",
    "xcomp",
    "csubj",
    "csubjpass",
}
ACL_DEP_TYPES = {"acl", "relcl", "advcl"}
CCOMP_DEP_TYPES = {"ccomp", "xcomp", "csubj", "csubjpass"}
PARALLEL_DEP_TYPES = {"conj", "parataxis"}
ACL_MARKERS = {
    "that",
    "whether",
    "if",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "whatever",
    "whichever",
    "whoever",
    "when",
    "where",
    "why",
    "how",
    "because",
    "although",
    "while",
    "until",
    "unless",
    "since",
    "before",
    "after",
    "though",
}
CCOMP_MARKERS = {"that", "whether", "if", "what", "which", "who", "whom", "whose"}
CLAUSE_LABELS = {"S", "SBAR", "SBARQ", "SINV", "SQ"}
TIER_LABELS = ("low", "medium-low", "medium-high", "high")
PRODUCT_ANCHOR_PRIORITY = ("A1", "A5", "A2")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_file(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl_file(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def chunked(items: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


@lru_cache(maxsize=1)
def _load_stage6_audit_module():
    spec = importlib.util.spec_from_file_location("stage6_attr_audit", STAGE6_AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 Stage 6 attr audit 模块: {STAGE6_AUDIT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_spacy_model():
    nlp = spacy.load("en_core_web_sm")
    for pipe_name in ("ner", "lemmatizer", "textcat", "textcat_multilabel", "senter", "sentencizer"):
        if pipe_name in nlp.pipe_names:
            nlp.remove_pipe(pipe_name)
    return nlp


@lru_cache(maxsize=1)
def _load_benepar_parser():
    if LOCAL_NLTK_DATA.exists():
        nltk_data_path = str(LOCAL_NLTK_DATA)
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.append(nltk_data_path)
    try:
        return BeneparParser(BENEPAR_MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(f"无法加载 benepar 模型 {BENEPAR_MODEL_NAME}: {exc}") from exc


def _doc_tokens(doc) -> list[Any]:
    return [token for token in doc if not token.is_space]


def _char_to_token_span(doc, char_start: int, char_end: int, context: str) -> tuple[int, int]:
    token_start = None
    token_end = None
    for token in _doc_tokens(doc):
        token_char_start = token.idx
        token_char_end = token.idx + len(token.text)
        if token_char_end <= char_start or token_char_start >= char_end:
            continue
        if token_start is None:
            token_start = token.i
        token_end = token.i + 1
    if token_start is None or token_end is None:
        raise ValueError(f"{context} 无法从字符位置映射到 token span")
    return token_start, token_end


def _stage6_match_spans(query: str, attrs_used: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
    stage6 = _load_stage6_audit_module()
    matches_by_key: dict[str, list[tuple[int, int]]] = {}
    for key, value in attrs_used.items():
        pattern = stage6._build_attr_value_pattern(value)
        matches: list[tuple[int, int]] = []
        if pattern is not None:
            matches.extend(match.span() for match in re.finditer(pattern, query, re.IGNORECASE))
        if isinstance(value, str):
            matches.extend(stage6._find_variant_token_spans(query, value))
        matches_by_key[key] = sorted(set(matches))
    return matches_by_key


def _dependency_tree_depth(doc) -> int:
    depth_cache: dict[int, int] = {}
    max_depth = 0
    for token in _doc_tokens(doc):
        chain = []
        current = token
        while current.i not in depth_cache and current.head != current:
            chain.append(current)
            current = current.head

        if current.i in depth_cache:
            depth = depth_cache[current.i]
        else:
            depth = 1
            depth_cache[current.i] = depth

        for chain_token in reversed(chain):
            depth += 1
            depth_cache[chain_token.i] = depth

        token_depth = depth_cache[token.i]
        if token_depth > max_depth:
            max_depth = token_depth

    if max_depth == 0:
        raise ValueError("dependency tree depth 计算失败：没有有效 token")
    return max_depth


def _clause_nesting_depth(doc) -> int:
    max_depth = 0
    for token in _doc_tokens(doc):
        depth = 0
        current = token
        visited = set()
        while True:
            if current.dep_ in CLAUSE_DEP_TYPES:
                depth += 1
            if current.head == current:
                break
            if current.i in visited:
                raise ValueError("检测到循环依存，无法计算 clause nesting depth")
            visited.add(current.i)
            current = current.head
        if depth > max_depth:
            max_depth = depth
    return max_depth


def _collect_acl_infos(doc) -> list[dict[str, Any]]:
    results = []
    for token in _doc_tokens(doc):
        acl_info = None
        if token.dep_ == "acl":
            marker_word = None
            complementizer = None
            for child in token.children:
                if child.dep_ == "mark":
                    marker_word = child.text.lower()
                    complementizer = child.text.lower()
                    break
            if marker_word is None:
                for child in token.children:
                    if child.dep_ == "comp":
                        marker_word = child.text.lower()
                        complementizer = child.text.lower()
                        break
            acl_info = {
                "acl_type": "acl",
                "marker": marker_word,
                "complementizer": complementizer,
                "head_word": token.head.text if token.head else "",
                "verb_word": token.text,
                "position": token.i,
            }
        elif token.dep_ == "relcl":
            rel_pronoun = None
            for child in token.children:
                if child.dep_ in ("nsubj", "dobj", "pobj", "attr"):
                    rel_pronoun = child.text.lower()
                    break
            if rel_pronoun is None:
                rel_pronoun = token.text.lower()
            acl_info = {
                "acl_type": "relcl_reference",
                "marker": rel_pronoun,
                "complementizer": None,
                "head_word": token.head.text if token.head else "",
                "verb_word": token.text,
                "position": token.i,
            }
        elif token.dep_ == "advcl":
            marker_word = None
            for child in token.children:
                if child.dep_ == "mark":
                    marker_word = child.text.lower()
                    break
            acl_info = {
                "acl_type": "advcl",
                "marker": marker_word,
                "complementizer": None,
                "head_word": token.head.text if token.head else "",
                "verb_word": token.text,
                "position": token.i,
            }
        elif token.dep_ == "conj":
            acl_info = {
                "acl_type": "conj",
                "marker": None,
                "complementizer": None,
                "head_word": token.head.text if token.head else "",
                "verb_word": token.text,
                "position": token.i,
            }
        elif token.dep_ == "parataxis":
            acl_info = {
                "acl_type": "parataxis",
                "marker": None,
                "complementizer": None,
                "head_word": token.head.text if token.head else "",
                "verb_word": token.text,
                "position": token.i,
            }
        elif token.dep_ == "mark":
            marker_lower = token.text.lower()
            if marker_lower in ACL_MARKERS:
                head = token.head
                if head is not None:
                    if head.dep_ == "relcl":
                        acl_type = "relcl_reference"
                    elif head.dep_ == "advcl":
                        acl_type = "advcl"
                    elif head.dep_ == "csubj":
                        acl_type = "csubj"
                    elif head.dep_ == "csubjpass":
                        acl_type = "csubjpass"
                    else:
                        acl_type = "acl"
                    acl_info = {
                        "acl_type": acl_type,
                        "marker": marker_lower,
                        "complementizer": marker_lower,
                        "head_word": head.text if head else "",
                        "verb_word": head.text if head else "",
                        "position": token.i,
                    }

        if acl_info is not None:
            results.append(acl_info)
    return results


def _collect_ccomp_infos(doc) -> list[dict[str, Any]]:
    results = []
    for token in _doc_tokens(doc):
        comp_info = None
        if token.dep_ == "ccomp":
            marker_word = None
            complementizer = None
            for child in token.children:
                if child.dep_ == "mark":
                    marker_word = child.text.lower()
                    break
            if marker_word is None:
                for child in token.children:
                    if child.dep_ == "comp":
                        complementizer = child.text.lower()
                        break
            comp_info = {
                "comp_type": "ccomp",
                "marker": marker_word,
                "complementizer": complementizer,
                "head_word": token.head.text if token.head else "",
                "verb_word": token.text,
                "position": token.i,
            }
        elif token.dep_ == "xcomp":
            marker_word = None
            for child in token.children:
                if child.dep_ == "aux" and child.text.lower() == "to":
                    marker_word = "to"
                    break
            comp_info = {
                "comp_type": "xcomp",
                "marker": marker_word if marker_word else "bare_infinitive",
                "complementizer": None,
                "head_word": token.head.text if token.head else "",
                "verb_word": token.text,
                "position": token.i,
            }
        elif token.dep_ == "csubj":
            comp_info = {
                "comp_type": "csubj",
                "marker": None,
                "complementizer": None,
                "head_word": token.head.text if token.head else "",
                "verb_word": token.text,
                "position": token.i,
            }
        elif token.dep_ == "csubjpass":
            comp_info = {
                "comp_type": "csubjpass",
                "marker": None,
                "complementizer": None,
                "head_word": token.head.text if token.head else "",
                "verb_word": token.text,
                "position": token.i,
            }
        elif token.dep_ == "mark":
            marker_lower = token.text.lower()
            if marker_lower in CCOMP_MARKERS:
                head = token.head
                if head is not None:
                    comp_info = {
                        "comp_type": "mark_" + head.dep_,
                        "marker": marker_lower,
                        "complementizer": marker_lower,
                        "head_word": head.text if head else "",
                        "verb_word": head.text if head else "",
                        "position": token.i,
                    }

        if comp_info is not None:
            results.append(comp_info)
    return results


def _parallel_modifier_count(doc) -> int:
    return sum(1 for token in _doc_tokens(doc) if token.dep_ in PARALLEL_DEP_TYPES)


def _branching_factor(doc) -> float:
    child_counts = [sum(1 for child in token.children if not child.is_space) for token in _doc_tokens(doc)]
    non_leaf = [count for count in child_counts if count > 0]
    if not non_leaf:
        return 0.0
    return float(sum(non_leaf) / len(non_leaf))


def _tree_spans_from_benepar(tree) -> list[tuple[int, int, str, bool]]:
    spans: list[tuple[int, int, str, bool]] = []

    def walk(node, start: int, is_root: bool) -> int:
        if isinstance(node, str):
            return start + 1

        index = start
        for child in node:
            index = walk(child, index, False)
        end = index
        label = node.label()
        spans.append((start, end, label, is_root))
        return end

    walk(tree, 0, True)
    return spans


def _benepar_clause_token_set(tree, tokens: list[Any]) -> set[int]:
    clause_tokens: set[int] = set()
    for start, end, label, is_root in _tree_spans_from_benepar(tree):
        if is_root and label in CLAUSE_LABELS:
            continue
        if label not in CLAUSE_LABELS:
            continue
        for idx in range(start, end):
            if idx >= len(tokens):
                raise ValueError("benepar span 超出 token 范围")
            if not tokens[idx].is_punct:
                clause_tokens.add(idx)
    return clause_tokens


def _spacy_clause_token_set(doc) -> set[int]:
    clause_tokens: set[int] = set()
    tokens = _doc_tokens(doc)
    for token in tokens:
        if token.dep_ not in CLAUSE_DEP_TYPES:
            continue
        for child in token.subtree:
            if child.is_space or child.is_punct:
                continue
            clause_tokens.add(child.i)
    return clause_tokens


def _parser_agreement(doc, tree) -> float:
    spacy_tokens = _spacy_clause_token_set(doc)
    benepar_tokens = _benepar_clause_token_set(tree, _doc_tokens(doc))
    if not spacy_tokens and not benepar_tokens:
        return 1.0
    if not spacy_tokens or not benepar_tokens:
        return 0.0
    intersection = len(spacy_tokens & benepar_tokens)
    precision = intersection / len(benepar_tokens)
    recall = intersection / len(spacy_tokens)
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _attribute_spans(doc, attrs_used: dict[str, Any]) -> dict[str, tuple[int, int, float]]:
    query_text = doc.text
    matches_by_key = _stage6_match_spans(query_text, attrs_used)
    occupied_char_spans: list[tuple[int, int]] = []
    spans: dict[str, tuple[int, int, float]] = {}
    ordered_keys = sorted(
        attrs_used,
        key=lambda key: (-len(str(attrs_used[key]).strip()), key),
    )
    for attr_key in ordered_keys:
        attr_value = attrs_used[attr_key]
        if not isinstance(attr_value, str) or not attr_value.strip():
            raise ValueError(f"{attr_key} 必须是非空字符串")
        chosen_char_span = None
        for char_span in matches_by_key[attr_key]:
            if any(not (char_span[1] <= used[0] or char_span[0] >= used[1]) for used in occupied_char_spans):
                continue
            chosen_char_span = char_span
            occupied_char_spans.append(char_span)
            break
        if chosen_char_span is None:
            raise ValueError(f"{attr_key}={attr_value} 无法按 Stage 6 规则定位唯一 span")
        start_token_index, end_token_index = _char_to_token_span(
            doc,
            chosen_char_span[0],
            chosen_char_span[1],
            f"{attr_key}={attr_value}",
        )
        spans[attr_key] = (
            start_token_index,
            end_token_index,
            (start_token_index + end_token_index - 1) / 2.0,
        )
    return spans


def _product_anchor_key(attribute_spans: dict[str, tuple[int, int, float]]) -> str:
    for key in PRODUCT_ANCHOR_PRIORITY:
        if key in attribute_spans:
            return key
    raise KeyError(
        "attrs_used 缺少可用的产品锚点属性，期望至少包含以下之一: "
        + ", ".join(PRODUCT_ANCHOR_PRIORITY)
    )


def _product_to_attribute_distance(attribute_spans: dict[str, tuple[int, int, float]], token_count: int) -> float:
    anchor_key = _product_anchor_key(attribute_spans)
    product_center = attribute_spans[anchor_key][2]
    other_centers = [span[2] for key, span in attribute_spans.items() if key != anchor_key]
    if not other_centers:
        raise ValueError("没有可用于计算 product-to-attribute distance 的属性")
    normalization = max(token_count - 1, 1)
    distances = [abs(product_center - center) / normalization for center in other_centers]
    return float(mean(distances))


def _attribute_dispersion(attribute_spans: dict[str, tuple[int, int, float]], token_count: int) -> float:
    centers = [span[2] for span in attribute_spans.values()]
    if len(centers) < 2:
        raise ValueError("属性数量不足，无法计算 attribute dispersion")
    pairwise = [abs(a - b) for a, b in combinations(centers, 2)]
    normalization = max(token_count - 1, 1)
    return float(mean(pairwise) / normalization)


def _batch_to_input_sentences(docs) -> list[InputSentence]:
    sentences = []
    for doc in docs:
        tokens = _doc_tokens(doc)
        words = [token.text for token in tokens]
        space_after = [bool(token.whitespace_) for token in tokens]
        sentences.append(InputSentence(words=words, space_after=space_after))
    return sentences


def _parse_benepar_trees_in_chunks(benepar_parser, docs, category: str, query_type: str, chunk_size: int = 256):
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正数，得到 {chunk_size}")
    input_sentences = _batch_to_input_sentences(docs)
    all_trees = []
    total = len(input_sentences)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_inputs = input_sentences[start:end]
        chunk_trees = list(benepar_parser.parse_sents(chunk_inputs))
        if len(chunk_trees) != len(chunk_inputs):
            raise RuntimeError(
                f"{category} {query_type} benepar chunk 树数量与输入不一致: "
                f"expected={len(chunk_inputs)} actual={len(chunk_trees)} start={start} end={end}"
            )
        all_trees.extend(chunk_trees)
        log(f"{category} {query_type}: benepar chunk 完成 {end}/{total}")
    return all_trees


def _safe_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("无法计算空列表的均值")
    return float(mean(values))


def _safe_pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))


def _oriented_score(values: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("values 必须是一维数组")
    if anchor.ndim != 1:
        raise ValueError("anchor 必须是一维数组")
    if len(values) != len(anchor):
        raise ValueError("values 和 anchor 长度不一致")
    if len(values) < 2:
        raise ValueError("values 长度不足，无法确定 latent score 方向")
    if np.std(values) == 0 or np.std(anchor) == 0:
        raise ValueError("latent score 或 anchor 为常数，无法计算相关性")
    corr = np.corrcoef(values, anchor)[0, 1]
    if np.isnan(corr):
        raise ValueError("latent score 与 anchor 的相关性为 NaN")
    return values if corr >= 0 else -values


def _percentile_tiers(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if scores.ndim != 1:
        raise ValueError("scores 必须是一维数组")
    order = np.argsort(np.argsort(scores, kind="mergesort"), kind="mergesort")
    percentiles = (order + 1) / len(scores)
    tiers = np.empty(len(scores), dtype=object)
    tiers[percentiles <= 0.25] = "low"
    tiers[(percentiles > 0.25) & (percentiles <= 0.5)] = "medium-low"
    tiers[(percentiles > 0.5) & (percentiles <= 0.75)] = "medium-high"
    tiers[percentiles > 0.75] = "high"
    return percentiles, tiers


def _validate_tier_counts(tiers: np.ndarray, context: str) -> dict[str, int]:
    counts = Counter(str(tier) for tier in tiers)
    for label in TIER_LABELS:
        if counts[label] == 0:
            raise ValueError(f"{context} 的 tier {label} 为空")
    return {label: int(counts[label]) for label in TIER_LABELS}


def _summarize_hit10(records: list[dict[str, Any]], context: str) -> dict[str, Any]:
    if not records:
        raise ValueError(f"{context} 没有可用于汇总的 hit@10 记录")
    hits = [float(record["hit_at10"]) for record in records]
    if any(not math.isfinite(hit) for hit in hits):
        raise ValueError(f"{context} hit@10 中存在非有限值")
    return {
        "count": len(hits),
        "mean": float(mean(hits)),
        "std": _safe_pstdev(hits),
        "min": float(min(hits)),
        "max": float(max(hits)),
    }


def _kruskal_summary(grouped_records: dict[str, list[dict[str, Any]]], context: str) -> dict[str, Any]:
    samples = []
    for label in TIER_LABELS:
        records = grouped_records.get(label, [])
        if not records:
            raise ValueError(f"{context} 的 tier {label} 为空，无法执行 Kruskal-Wallis 检验")
        samples.append([float(record["hit_at10"]) for record in records])
    stat, p_value = kruskal(*samples)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
    }


def _feature_summary(scores: np.ndarray, tiers: np.ndarray, extra: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": _safe_pstdev(scores.tolist()),
        "tier_counts": _validate_tier_counts(tiers, "tier summary"),
    }
    summary.update(extra)
    return summary


def _fit_latent_methods(feature_matrix: np.ndarray, tree_depths: np.ndarray) -> dict[str, np.ndarray]:
    scaler = StandardScaler()
    standardized = scaler.fit_transform(feature_matrix)

    pca = PCA(n_components=1, random_state=42)
    pca_scores = pca.fit_transform(standardized).ravel()
    pca_scores = _oriented_score(pca_scores, tree_depths)

    fa = FactorAnalysis(n_components=1, random_state=42)
    fa_scores = fa.fit_transform(standardized).ravel()
    fa_scores = _oriented_score(fa_scores, tree_depths)

    gmm = GaussianMixture(
        n_components=4,
        covariance_type="full",
        random_state=42,
        reg_covar=1e-6,
        n_init=5,
    )
    gmm.fit(standardized)
    gmm_probabilities = gmm.predict_proba(standardized)
    gmm_labels = gmm.predict(standardized)
    gmm_cluster_order = []
    for cluster_id in range(gmm.n_components):
        cluster_mask = gmm_labels == cluster_id
        if not np.any(cluster_mask):
            raise ValueError("GaussianMixture 产生了空簇")
        cluster_order_anchor = float(np.mean(tree_depths[cluster_mask]))
        gmm_cluster_order.append((cluster_order_anchor, cluster_id))
    gmm_cluster_order.sort()
    gmm_rank_map = {cluster_id: rank for rank, (_, cluster_id) in enumerate(gmm_cluster_order)}
    gmm_scores = np.zeros(len(standardized), dtype=float)
    for cluster_id, rank in gmm_rank_map.items():
        gmm_scores += gmm_probabilities[:, cluster_id] * rank

    ord_clust = AgglomerativeClustering(n_clusters=4, linkage="ward")
    ord_labels = ord_clust.fit_predict(standardized)
    ord_cluster_order = []
    for cluster_id in range(4):
        cluster_mask = ord_labels == cluster_id
        if not np.any(cluster_mask):
            raise ValueError("AgglomerativeClustering 产生了空簇")
        cluster_order_anchor = float(np.mean(tree_depths[cluster_mask]))
        ord_cluster_order.append((cluster_order_anchor, cluster_id))
    ord_cluster_order.sort()
    ord_rank_map = {cluster_id: rank for rank, (_, cluster_id) in enumerate(ord_cluster_order)}
    ord_scores = np.array([ord_rank_map[label] for label in ord_labels], dtype=float)

    return {
        "pca": pca_scores,
        "factor_analysis": fa_scores,
        "gmm": gmm_scores,
        "ordinal_clustering": ord_scores,
        "pca_explained_variance_ratio": np.array(pca.explained_variance_ratio_, dtype=float),
        "factor_analysis_components": fa.components_.astype(float),
    }


def _extract_query_record(category: str, query_type: str, row: dict[str, Any], query_payload: dict[str, Any], doc, tree) -> dict[str, Any]:
    query_text = query_payload.get("query")
    attrs_used = query_payload.get("attrs_used")
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError(f"{category} {query_type} 查询文本为空")
    if not isinstance(attrs_used, dict):
        raise TypeError(f"{category} {query_type} attrs_used 必须是对象")
    if len(attrs_used) != 5:
        raise ValueError(f"{category} {query_type} attrs_used 必须包含 5 个属性")

    tokens = _doc_tokens(doc)
    if not tokens:
        raise ValueError(f"{category} {query_type} 查询没有有效 token: {query_text}")

    attribute_spans = _attribute_spans(doc, attrs_used)
    tree_depth = _dependency_tree_depth(doc)
    clause_nesting_depth = _clause_nesting_depth(doc)
    acl_infos = _collect_acl_infos(doc)
    ccomp_infos = _collect_ccomp_infos(doc)
    parallel_modifier_count = _parallel_modifier_count(doc)
    branching_factor = _branching_factor(doc)
    product_to_attribute_distance = _product_to_attribute_distance(attribute_spans, len(tokens))
    attribute_dispersion = _attribute_dispersion(attribute_spans, len(tokens))
    parser_agreement = _parser_agreement(doc, tree)

    metrics = {
        "tree_depth": int(tree_depth),
        "clause_nesting_depth": int(clause_nesting_depth),
        "acl_count": int(len(acl_infos)),
        "ccomp_count": int(len(ccomp_infos)),
        "parallel_modifier_count": int(parallel_modifier_count),
        "product_to_attribute_distance": float(product_to_attribute_distance),
        "branching_factor": float(branching_factor),
        "attribute_dispersion": float(attribute_dispersion),
        "parser_agreement": float(parser_agreement),
    }

    return {
        "user_id": row["user_id"],
        "asin": row["asin"],
        "query_category": category,
        "query_type": query_type,
        "query": query_text.strip(),
        "attrs_used": attrs_used,
        "metrics": metrics,
        "tree_repr": str(tree),
    }


def _build_category_query_records(category: str, query_type: str, rows: list[dict[str, Any]], limit: int | None) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    nlp = _load_spacy_model()
    benepar_parser = _load_benepar_parser()

    selected_rows = rows if limit is None else rows[:limit]
    if not selected_rows:
        raise ValueError(f"{category} {query_type} 没有可分析的 query 行")

    query_payloads: list[dict[str, Any]] = []
    keys: list[tuple[str, str]] = []
    for row_idx, row in enumerate(selected_rows):
        if not isinstance(row, dict):
            raise TypeError(f"{category} {query_type} rows[{row_idx}] 必须是对象")
        user_id = row.get("user_id")
        asin = row.get("asin")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(f"{category} {query_type} rows[{row_idx}].user_id 必须是非空字符串")
        if not isinstance(asin, str) or not asin.strip():
            raise ValueError(f"{category} {query_type} rows[{row_idx}].asin 必须是非空字符串")
        payload = row.get(query_type)
        if not isinstance(payload, dict):
            raise TypeError(f"{category} {query_type} rows[{row_idx}].{query_type} 必须是对象")
        query_payloads.append(payload)
        keys.append((user_id, asin))

    texts = []
    for payload_idx, payload in enumerate(query_payloads):
        query_text = payload.get("query")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(f"{category} {query_type} payload[{payload_idx}] 查询文本为空")
        texts.append(query_text.strip())

    docs = list(nlp.pipe(texts, batch_size=64))
    if len(docs) != len(texts):
        raise RuntimeError(f"{category} {query_type} spaCy 文档数量与输入不一致")
    log(f"{category} {query_type}: spaCy 解析完成，queries={len(docs)}")
    trees = _parse_benepar_trees_in_chunks(benepar_parser, docs, category, query_type)
    if len(trees) != len(docs):
        raise RuntimeError(f"{category} {query_type} benepar 树数量与输入不一致")
    log(f"{category} {query_type}: benepar 解析完成，trees={len(trees)}")

    feature_rows: list[dict[str, Any]] = []
    for idx, (row, payload, doc, tree) in enumerate(zip(selected_rows, query_payloads, docs, trees)):
        feature_rows.append(
            _extract_query_record(
                category=category,
                query_type=query_type,
                row=row,
                query_payload=payload,
                doc=doc,
                tree=tree,
            )
        )
    log(f"{category} {query_type}: 指标提取完成，feature_rows={len(feature_rows)}")

    feature_index = {keys[idx]: feature_rows[idx] for idx in range(len(feature_rows))}
    if len(feature_index) != len(feature_rows):
        raise ValueError(f"{category} {query_type} 存在重复的 user_id/asin 键")
    return feature_rows, feature_index


def _build_method_analysis(
    category: str,
    query_type: str,
    feature_rows: list[dict[str, Any]],
    retrieval_summary: dict[str, Any],
    feature_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    tree_depths = np.array([float(row["metrics"]["tree_depth"]) for row in feature_rows], dtype=float)
    feature_matrix = np.array(
        [
            [
                row["metrics"]["tree_depth"],
                row["metrics"]["clause_nesting_depth"],
                row["metrics"]["acl_count"],
                row["metrics"]["ccomp_count"],
                row["metrics"]["parallel_modifier_count"],
                row["metrics"]["product_to_attribute_distance"],
                row["metrics"]["branching_factor"],
                row["metrics"]["attribute_dispersion"],
                row["metrics"]["parser_agreement"],
            ]
            for row in feature_rows
        ],
        dtype=float,
    )

    latent_outputs = _fit_latent_methods(feature_matrix, tree_depths)
    log(f"{category} {query_type}: 潜因子拟合完成")

    for method_name in ("pca", "factor_analysis", "gmm", "ordinal_clustering"):
        scores = np.asarray(latent_outputs[method_name], dtype=float)
        percentiles, tiers = _percentile_tiers(scores)
        latent_outputs[f"{method_name}_percentiles"] = percentiles
        latent_outputs[f"{method_name}_tiers"] = tiers

        for idx, row in enumerate(feature_rows):
            row.setdefault("latent_scores", {})[method_name] = float(scores[idx])
            row.setdefault("latent_percentiles", {})[method_name] = float(percentiles[idx])
            row.setdefault("complexity_tiers", {})[method_name] = str(tiers[idx])

    query_keys = {(row["user_id"], row["asin"]) for row in feature_rows}

    retrieval_results = [
        item
        for item in retrieval_summary["all_results_combined"]
        if item.get("query_category") == QUERY_TYPE_TO_RETRIEVAL_CATEGORY[query_type] and item.get("query_type") == "correct"
    ]
    if not retrieval_results:
        raise ValueError(f"{category} {query_type} retrieval summary 中没有 correct 结果")

    methods_summary: dict[str, Any] = {}
    for method_name in ("pca", "factor_analysis", "gmm", "ordinal_clustering"):
        tiers = latent_outputs[f"{method_name}_tiers"]
        tier_counts = _validate_tier_counts(tiers, f"{category} {query_type} {method_name}")
        tier_records = {
            label: []
            for label in TIER_LABELS
        }

        pooled_records = []
        for retriever_item in retrieval_results:
            retriever = str(retriever_item["retriever"])
            retriever_records = retriever_item.get("all_query_records")
            if not isinstance(retriever_records, list):
                raise TypeError(f"{category} {query_type} {retriever} all_query_records 必须是列表")

            aligned_records = []
            retriever_tier_records = {label: [] for label in TIER_LABELS}
            missing_count = 0
            for record_idx, record in enumerate(retriever_records):
                if not isinstance(record, dict):
                    raise TypeError(f"{category} {query_type} {retriever} record[{record_idx}] 必须是对象")
                key = (record.get("user_id"), record.get("asin"))
                if key not in feature_index:
                    missing_count += 1
                    continue
                feature_row = feature_index[key]
                latent_tier = feature_row["complexity_tiers"][method_name]
                hit_at10 = record.get("hit_at10")
                if not isinstance(hit_at10, (int, float)):
                    raise TypeError(f"{category} {query_type} {retriever} record[{record_idx}] 缺少 hit_at10")
                merged_record = {
                    "user_id": key[0],
                    "asin": key[1],
                    "hit_at10": float(hit_at10),
                    "tier": latent_tier,
                }
                aligned_records.append(merged_record)
                retriever_tier_records[latent_tier].append(merged_record)
                pooled_records.append(merged_record)
                tier_records[latent_tier].append(merged_record)

            methods_summary.setdefault(method_name, {"retrievers": {}, "pooled": {}})
            methods_summary[method_name]["retrievers"][retriever] = {
                "matched_count": len(aligned_records),
                "missing_count": missing_count,
                "tier_counts": {label: len(retriever_tier_records[label]) for label in TIER_LABELS},
                "tier_hit10": {
                    label: _summarize_hit10(retriever_tier_records[label], f"{category} {query_type} {method_name} {retriever} {label}")
                    for label in TIER_LABELS
                },
                "kruskal_wallis": _kruskal_summary(retriever_tier_records, f"{category} {query_type} {method_name} {retriever}"),
            }

        methods_summary[method_name]["pooled"] = {
            "matched_count": len(pooled_records),
            "tier_counts": {label: len(tier_records[label]) for label in TIER_LABELS},
            "tier_hit10": {
                label: _summarize_hit10(tier_records[label], f"{category} {query_type} {method_name} pooled {label}")
                for label in TIER_LABELS
            },
            "kruskal_wallis": _kruskal_summary(tier_records, f"{category} {query_type} {method_name} pooled"),
        }
        methods_summary[method_name]["score_summary"] = _feature_summary(
            np.asarray(latent_outputs[method_name], dtype=float),
            np.asarray(latent_outputs[f"{method_name}_tiers"], dtype=object),
            {
                "pca_explained_variance_ratio": latent_outputs["pca_explained_variance_ratio"].tolist() if method_name == "pca" else None,
                "factor_analysis_components": latent_outputs["factor_analysis_components"].tolist() if method_name == "factor_analysis" else None,
            },
        )

    return {
        "feature_rows": feature_rows,
        "feature_index_keys": [list(key) for key in sorted(query_keys)],
        "methods": methods_summary,
    }


def _load_category_query_rows(category: str) -> list[dict[str, Any]]:
    query_file = QUERY_ROOT / category / "query.json"
    rows = load_json_file(query_file, f"{category} query.json")
    if not isinstance(rows, list):
        raise TypeError(f"{category} query.json 顶层必须是列表")
    return rows


def _load_category_retrieval_summary(category: str) -> dict[str, Any]:
    summary_file = RETRIEVAL_ROOT / category / "retrieval_all_summary.json"
    data = load_json_file(summary_file, f"{category} retrieval_all_summary.json")
    if not isinstance(data, dict):
        raise TypeError(f"{category} retrieval_all_summary.json 顶层必须是对象")
    if "all_results_combined" not in data:
        raise KeyError(f"{category} retrieval_all_summary.json 缺少 all_results_combined")
    return data


def process_category(category: str, max_rows: int | None) -> dict[str, Any]:
    log(f"开始处理类别: {category}")
    rows = _load_category_query_rows(category)
    retrieval_summary = _load_category_retrieval_summary(category)
    category_output_dir = ensure_dir(STAGE12_ROOT / category)
    feature_output_file = category_output_dir / "query_features.jsonl"
    summary_output_file = category_output_dir / "stage12_summary.json"

    category_summary: dict[str, Any] = {
        "category": category,
        "source_query_file": str(QUERY_ROOT / category / "query.json"),
        "source_retrieval_summary_file": str(RETRIEVAL_ROOT / category / "retrieval_all_summary.json"),
        "query_types": {},
    }

    total_feature_rows = 0
    all_feature_rows: list[dict[str, Any]] = []
    for query_type in QUERY_TYPES:
        query_rows = []
        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"{category} rows[{row_idx}] 必须是对象")
            payload = row.get(query_type)
            if not isinstance(payload, dict):
                raise TypeError(f"{category} rows[{row_idx}].{query_type} 必须是对象")
            query_rows.append(row)

        limit = max_rows
        feature_rows, feature_index = _build_category_query_records(category, query_type, query_rows, limit)
        analysis = _build_method_analysis(category, query_type, feature_rows, retrieval_summary, feature_index)
        category_summary["query_types"][query_type] = {
            "num_queries": len(feature_rows),
            "feature_count": len(feature_rows),
            "feature_definitions": {
                "tree_depth": "spaCy 依存树最大深度",
                "clause_nesting_depth": "clausal dependency ancestor chain 的最大深度",
                "acl_count": "Stage 5 风格 ACL 计数",
                "ccomp_count": "Stage 5 风格 CCOMP 计数",
                "parallel_modifier_count": "conj/parataxis 计数",
                "product_to_attribute_distance": "A1 与其余属性的平均归一化距离",
                "branching_factor": "依存树非叶节点平均出度",
                "attribute_dispersion": "属性中心点的平均两两距离（归一化）",
                "parser_agreement": "spaCy 与 benepar 的 clause token coverage F1",
            },
            "methods": analysis["methods"],
        }
        total_feature_rows += len(feature_rows)
        all_feature_rows.extend(feature_rows)

    category_summary["num_feature_rows"] = total_feature_rows
    write_jsonl_file(feature_output_file, all_feature_rows)
    write_json_file(summary_output_file, category_summary)

    log(f"类别完成: {category}")
    log(f"  特征输出: {feature_output_file}")
    log(f"  汇总输出: {summary_output_file}")
    for query_type in QUERY_TYPES:
        method_summary = category_summary["query_types"][query_type]["methods"]
        for method_name in ("pca", "factor_analysis", "gmm", "ordinal_clustering"):
            pooled = method_summary[method_name]["pooled"]
            log(
                f"  {query_type}/{method_name}: matched={pooled['matched_count']} "
                f"low={pooled['tier_counts']['low']} medium-low={pooled['tier_counts']['medium-low']} "
                f"medium-high={pooled['tier_counts']['medium-high']} high={pooled['tier_counts']['high']} "
                f"kruskal_p={pooled['kruskal_wallis']['p_value']:.4e}"
            )

    return category_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 12 complexity analysis for Stage 6 query.json and Stage 8 retrieval summaries."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="要处理的类别，默认处理已有的三个类别。",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="每个 query_type 最多处理多少条记录，仅用于调试。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(STAGE12_ROOT)

    summaries = {}
    total_rows = 0
    for category in args.categories:
        summaries[category] = process_category(category, args.max_rows)
        total_rows += int(summaries[category]["num_feature_rows"])

    root_summary = {
        "timestamp": datetime.now().isoformat(),
        "output_root": str(STAGE12_ROOT),
        "categories": list(summaries.keys()),
        "num_categories": len(summaries),
        "num_feature_rows": total_rows,
        "summaries": {
            category: {
                "summary_file": str(STAGE12_ROOT / category / "stage12_summary.json"),
                "feature_file": str(STAGE12_ROOT / category / "query_features.jsonl"),
                "num_feature_rows": summaries[category]["num_feature_rows"],
            }
            for category in summaries
        },
    }
    write_json_file(STAGE12_ROOT / "stage12_root_summary.json", root_summary)

    log("=" * 80)
    log("Stage 12 完成")
    for category in summaries:
        log(f"  {category}: {summaries[category]['num_feature_rows']} 条特征记录")
    log(f"  总记录数: {total_rows}")
    log(f"  根汇总: {STAGE12_ROOT / 'stage12_root_summary.json'}")
    log("=" * 80)


if __name__ == "__main__":
    main()
