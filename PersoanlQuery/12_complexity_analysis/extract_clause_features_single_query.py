#!/usr/bin/env python3
"""Extract clause-related syntactic features for a single query."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import spacy


REPO_ROOT = Path("/fs04/ar57/wenyu")

RELATIVE_MARKERS = {
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
}

SUBORDINATE_DEP_TYPES = {
    "acl",
    "relcl",
    "ccomp",
    "xcomp",
    "advcl",
}

NOUNISH_POS = {
    "NOUN",
    "PROPN",
    "PRON",
}

LONG_DEPENDENCY_THRESHOLD = 5


def _is_relative_marker_token(token) -> bool:
    if token.text.lower() not in RELATIVE_MARKERS:
        return False

    current = token
    visited = set()
    while True:
        if current.dep_ == "relcl":
            return True
        if current.head == current:
            return False
        if current.i in visited:
            raise ValueError("检测到依存循环，无法判断 relative marker")
        visited.add(current.i)
        current = current.head


def _dependency_depths(tokens) -> list[int]:
    depth_cache: dict[int, int] = {}
    depths = []
    for token in tokens:
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

        depths.append(depth_cache[token.i])
    return depths


def _clause_nesting_depth(tokens) -> int:
    max_depth = 0
    for token in tokens:
        depth = 0
        current = token
        visited = set()
        while True:
            if current.dep_ in SUBORDINATE_DEP_TYPES:
                depth += 1
            if current.head == current:
                break
            if current.i in visited:
                raise ValueError("检测到循环依存，无法计算 clause_nesting_depth")
            visited.add(current.i)
            current = current.head
        if depth > max_depth:
            max_depth = depth
    return max_depth


def _is_nmod_proxy(token) -> bool:
    if token.dep_ == "poss":
        return True
    return token.dep_ == "prep" and token.head.pos_ in NOUNISH_POS


@lru_cache(maxsize=1)
def load_spacy_model():
    nlp = spacy.load("en_core_web_sm")
    for pipe_name in ("ner", "lemmatizer", "textcat", "textcat_multilabel", "senter", "sentencizer"):
        if pipe_name in nlp.pipe_names:
            nlp.remove_pipe(pipe_name)
    return nlp


def extract_clause_features_from_doc(doc, query_text: str) -> dict:
    tokens = [token for token in doc if not token.is_space]
    if not tokens:
        raise ValueError("query 没有有效 token")

    dependency_depths = _dependency_depths(tokens)
    max_dependency_depth = max(dependency_depths)
    mean_dependency_depth = float(sum(dependency_depths) / len(dependency_depths))
    dependency_tree_height = max_dependency_depth - 1
    depth_variance = float(
        sum((depth - mean_dependency_depth) ** 2 for depth in dependency_depths) / len(dependency_depths)
    )

    acl_count = sum(1 for token in tokens if token.dep_ == "acl")
    relcl_count = sum(1 for token in tokens if token.dep_ == "relcl")
    ccomp_count = sum(1 for token in tokens if token.dep_ == "ccomp")
    xcomp_count = sum(1 for token in tokens if token.dep_ == "xcomp")
    advcl_count = sum(1 for token in tokens if token.dep_ == "advcl")
    clause_nesting_depth = _clause_nesting_depth(tokens)

    dependency_distances = [
        abs(token.i - token.head.i)
        for token in tokens
        if token.head != token
    ]
    if not dependency_distances:
        raise ValueError("没有可用于计算 dependency distance 的非根节点")
    mean_dependency_distance = float(sum(dependency_distances) / len(dependency_distances))
    max_dependency_distance = int(max(dependency_distances))
    long_dependency_ratio = float(
        sum(1 for distance in dependency_distances if distance >= LONG_DEPENDENCY_THRESHOLD) / len(dependency_distances)
    )

    amod_count = sum(1 for token in tokens if token.dep_ == "amod")
    advmod_count = sum(1 for token in tokens if token.dep_ == "advmod")
    nmod_count = sum(1 for token in tokens if _is_nmod_proxy(token))
    compound_count = sum(1 for token in tokens if token.dep_ == "compound")
    modifier_density = float((amod_count + advmod_count + nmod_count + compound_count) / len(tokens))

    coordination_count = sum(
        1
        for token in tokens
        if token.dep_ == "cc" or (token.dep_ == "conj" and token.head != token)
    )
    max_branching_factor = max(
        sum(1 for child in token.children if not child.is_space)
        for token in tokens
    )

    return {
        "query": query_text,
        "word_count": len(tokens),
        "features": {
            "max_dependency_depth": int(max_dependency_depth),
            "mean_dependency_depth": mean_dependency_depth,
            "dependency_tree_height": int(dependency_tree_height),
            "depth_variance": depth_variance,
            "acl_count": int(acl_count),
            "relcl_count": int(relcl_count),
            "ccomp_count": int(ccomp_count),
            "xcomp_count": int(xcomp_count),
            "advcl_count": int(advcl_count),
            "clause_nesting_depth": int(clause_nesting_depth),
            "mean_dependency_distance": mean_dependency_distance,
            "max_dependency_distance": max_dependency_distance,
            "long_dependency_ratio": long_dependency_ratio,
            "amod_count": int(amod_count),
            "advmod_count": int(advmod_count),
            "nmod_count": int(nmod_count),
            "compound_count": int(compound_count),
            "modifier_density": modifier_density,
            "coordination_count": int(coordination_count),
            "max_branching_factor": int(max_branching_factor),
        },
        "token_debug": [
            {
                "text": token.text,
                "dep": token.dep_,
                "head": token.head.text if token.head is not None else "",
                "pos": token.pos_,
            }
            for token in tokens
        ],
    }


def extract_clause_features(query: str) -> dict:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")

    query_text = query.strip()
    nlp = load_spacy_model()
    doc = nlp(query_text)
    return extract_clause_features_from_doc(doc, query_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract clause-related syntactic features for a single query.")
    parser.add_argument("--query", type=str, required=True, help="要分析的单条 query")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = extract_clause_features(args.query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
