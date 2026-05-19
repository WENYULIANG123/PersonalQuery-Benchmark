#!/usr/bin/env python3
"""Shared syntax-depth noisy query evaluation logic."""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from json import JSONDecoder
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["HF_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_CACHE"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["XDG_CACHE_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/cache"
os.environ["TORCH_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/torch_cache"
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/ar57_scratch/wenyu/triton_cache"
for _cache_dir in (
    os.environ["XDG_CACHE_HOME"],
    os.environ["TORCH_HOME"],
    os.environ["TRITON_CACHE_DIR"],
):
    os.makedirs(_cache_dir, exist_ok=True)

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
RETRIEVAL_ROOT = CURRENT_DIR.parent / "08_retrieval"
PERSONAL_QUERY_ROOT = RETRIEVAL_ROOT.parent
sys.path.insert(0, str(RETRIEVAL_ROOT))
sys.path.insert(0, str(PERSONAL_QUERY_ROOT))

from config import get_category_config, get_global_paths


SYNTAX_DEPTH_QUERY_CATEGORY = "syntax_depth"
CORRECT_QUERY_TYPE = "correct"
NOISY_QUERY_TYPE = "noisy"
NOISY_INJECTION_SOURCE = "syntax_depth_preserve_depth"
CORRECT_QUERY_SOURCE_FIELD = "original_query"
SUMMARY_EXCLUDE_METRIC = "P@10"

DENSE_RETRIEVERS = ["bge", "e5", "minilm", "star", "ance"]
COLBERTV2_RETRIEVERS = ["colbertv2"]
SPARSE_RETRIEVERS = ["splade"]
RETRIEVERS = DENSE_RETRIEVERS + COLBERTV2_RETRIEVERS + SPARSE_RETRIEVERS + ["bm25"]

DEPTH_GROUPS = [
    ("low_complexity", 1, 3, "低复杂度(1-3)"),
    ("medium_complexity", 4, 6, "中复杂度(4-6)"),
]
DEPTH_GROUP_ORDER = [name for name, _, _, _ in DEPTH_GROUPS]
DEPTH_GROUP_DISPLAY = {name: label for name, _, _, label in DEPTH_GROUPS}

INJECTION_DEPTH_GROUPS = [
    ("low_complexity", 1, 3, "低(1-3)"),
    ("medium_complexity", 4, 6, "中(4-6)"),
]
INJECTION_DEPTH_GROUP_ORDER = [name for name, _, _, _ in INJECTION_DEPTH_GROUPS]
INJECTION_DEPTH_GROUP_DISPLAY = {name: label for name, _, _, label in INJECTION_DEPTH_GROUPS}
INJECTION_DEPTH_GROUP_SHORT_DISPLAY = {
    "low_complexity": "低",
    "medium_complexity": "中",
}


class NumpyCoreCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


COLBERTV2_CUDA_HOME = "/usr/local/cuda-12.5"
COLBERTV2_TORCH_EXTENSIONS_BASE_DIR = "/home/wlia0047/ar57_scratch/wenyu/torch_extensions"
COLBERTV2_HOST_CC = "/usr/bin/gcc"
COLBERTV2_HOST_CXX = "/usr/bin/g++"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def require_field(item: Dict, field: str, context: str):
    if field not in item:
        raise ValueError(f"{context} missing required field '{field}'")
    return item[field]


def require_str(item: Dict, field: str, context: str) -> str:
    value = require_field(item, field, context)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} field '{field}' must be a non-empty string, got {value!r}")
    return value


def require_dict(item: Dict, field: str, context: str) -> Dict:
    value = require_field(item, field, context)
    if not isinstance(value, dict):
        raise TypeError(f"{context} field '{field}' must be dict, got {type(value).__name__}")
    return value


def require_int(item: Dict, field: str, context: str) -> int:
    value = require_field(item, field, context)
    if not isinstance(value, int):
        raise TypeError(f"{context} field '{field}' must be int, got {type(value).__name__}")
    return value


def require_number(item: Dict, field: str, context: str) -> float:
    value = require_field(item, field, context)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{context} field '{field}' must be numeric, got {type(value).__name__}")
    return float(value)


def require_list(item: Dict, field: str, context: str) -> List:
    value = require_field(item, field, context)
    if not isinstance(value, list):
        raise TypeError(f"{context} field '{field}' must be list, got {type(value).__name__}")
    return value


def require_str_or_str_list(item: Dict, field: str, context: str):
    value = require_field(item, field, context)
    if isinstance(value, str) and value:
        return value
    if not isinstance(value, list):
        raise TypeError(f"{context} field '{field}' must be str or list[str], got {type(value).__name__}")
    if not value:
        raise ValueError(f"{context} field '{field}' must not be an empty list")
    for item_index, item_value in enumerate(value):
        if not isinstance(item_value, str) or not item_value:
            raise ValueError(
                f"{context} field '{field}' item {item_index} must be a non-empty string, got {item_value!r}"
            )
    return value


def require_depth_value(item: Dict, field: str, context: str) -> Tuple[int, List[int]]:
    value = require_field(item, field, context)
    if isinstance(value, int):
        return value, [value]
    if not isinstance(value, list):
        raise TypeError(f"{context} field '{field}' must be int or list[int], got {type(value).__name__}")
    if not value:
        raise ValueError(f"{context} field '{field}' must not be an empty list")
    depths = []
    for depth_index, depth in enumerate(value):
        if not isinstance(depth, int):
            raise TypeError(
                f"{context} field '{field}' item {depth_index} must be int, got {type(depth).__name__}"
            )
        depths.append(depth)
    return max(depths), depths


def depth_to_complexity_group(depth: int) -> str:
    for group_name, low, high, _ in DEPTH_GROUPS:
        if low <= depth <= high:
            return group_name
    raise ValueError(f"Unsupported syntax depth for configured groups: {depth}")


def injection_depth_to_complexity_group(depth: int) -> Optional[str]:
    for group_name, low, high, _ in INJECTION_DEPTH_GROUPS:
        if low <= depth <= high:
            return group_name
    return None


def require_single_injection_depth(item: Dict, context: str) -> Tuple[List[Dict], int, Optional[str], Optional[str]]:
    injected_errors = require_list(item, "injected_errors", context)
    if len(injected_errors) != 1:
        raise ValueError(
            f"{context} expected exactly one injected error for injection-depth grouping, "
            f"got {len(injected_errors)}"
        )
    error_context = f"{context} injected_errors[0]"
    injected_error = injected_errors[0]
    if not isinstance(injected_error, dict):
        raise TypeError(f"{error_context} must be dict, got {type(injected_error).__name__}")
    injection_depth = require_int(injected_error, "target_token_depth", error_context)
    injection_depth_group = injection_depth_to_complexity_group(injection_depth)
    injection_depth_group_display = (
        INJECTION_DEPTH_GROUP_DISPLAY[injection_depth_group]
        if injection_depth_group is not None
        else None
    )
    return (
        injected_errors,
        injection_depth,
        injection_depth_group,
        injection_depth_group_display,
    )


def load_appended_json_objects(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"JSON records file is empty: {file_path}")

    decoder = JSONDecoder()
    records = []
    index = 0
    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break

        decoded, end = decoder.raw_decode(content, index)
        if isinstance(decoded, list):
            if records:
                raise ValueError(f"{file_path} contains JSON records before a JSON array")
            if content[end:].strip():
                raise ValueError(f"{file_path} contains extra content after a JSON array")
            for row_index, row in enumerate(decoded):
                if not isinstance(row, dict):
                    raise TypeError(
                        f"{file_path} JSON array row {row_index} must be dict, got {type(row).__name__}"
                    )
            return decoded
        if not isinstance(decoded, dict):
            raise TypeError(f"{file_path} appended JSON record must be dict, got {type(decoded).__name__}")
        records.append(decoded)
        index = end

    if not records:
        raise ValueError(f"No JSON records parsed from {file_path}")
    return records


def load_syntax_depth_metadata(syntax_depth_query_file: str) -> Dict[Tuple[str, str, str], Dict]:
    data = load_appended_json_objects(syntax_depth_query_file)

    metadata = {}
    for row_index, item in enumerate(data):
        context = f"syntax-depth row {row_index}"
        if not isinstance(item, dict):
            raise TypeError(f"{context} must be dict, got {type(item).__name__}")
        user_id = require_str(item, "user_id", context)
        asin = require_str(item, "asin", context)
        syntax_query = require_dict(item, "syntax_depth_query", context)

        query_text = require_str(syntax_query, "query", context)
        actual_depth = require_int(syntax_query, "actual_depth", context)
        target_depth = require_int(syntax_query, "target_depth", context)
        word_count = require_int(syntax_query, "word_count", context)
        user_avg_depth = require_number(syntax_query, "user_avg_depth", context)

        attrs_used = syntax_query.get("attrs_used")
        if attrs_used is not None and not isinstance(attrs_used, dict):
            raise TypeError(f"{context} attrs_used must be dict when present, got {type(attrs_used).__name__}")

        key = (user_id, asin, query_text)
        if key in metadata:
            raise ValueError(f"Duplicate syntax-depth query key: user={user_id}, asin={asin}, query={query_text}")
        metadata[key] = {
            "user_id": user_id,
            "asin": asin,
            "source_query": query_text,
            "syntax_depth": actual_depth,
            "target_depth": target_depth,
            "word_count": word_count,
            "user_avg_depth": user_avg_depth,
            "attrs_used": attrs_used,
            "syntax_depth_row_index": row_index,
        }

    return metadata


def load_noisy_pairs(category_name: str, noisy_query_file: str, syntax_depth_query_file: str) -> List[Dict]:
    data = load_appended_json_objects(noisy_query_file)

    pairs = []
    skipped_non_syntax_depth = 0
    skipped_out_of_group = 0
    for row_index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"noisy row {row_index} must be dict, got {type(item).__name__}")
        context = f"noisy syntax-depth row {row_index}"
        injection_source = require_str(item, "injection_source", context)
        if injection_source != NOISY_INJECTION_SOURCE:
            skipped_non_syntax_depth += 1
            continue

        user_id = require_str(item, "user_id", context)
        asin = require_str(item, "asin", context)
        original_query = require_str(item, "original_query", context)
        noisy_query = require_str(item, "noisy_query", context)
        injection_depth, injection_depths = require_depth_value(item, "injection_target_depth", context)
        injection_depth_group = injection_depth_to_complexity_group(injection_depth)
        if injection_depth_group is None:
            skipped_out_of_group += 1
            continue
        injection_depth_group_display = INJECTION_DEPTH_GROUP_DISPLAY[injection_depth_group]
        syntax_depth_group = depth_to_complexity_group(injection_depth)
        attrs_used = {
            "noise_type": require_str_or_str_list(item, "noise_type", context),
            "correct_text": require_str_or_str_list(item, "correct_text", context),
            "noisy_text": require_str_or_str_list(item, "noisy_text", context),
            "anchor_replaced_text": require_str_or_str_list(item, "anchor_replaced_text", context),
        }

        pair = {
            "pair_id": len(pairs),
            "source_noisy_record_index": row_index,
            "category": category_name,
            "user_id": user_id,
            "asin": asin,
            "query_category": SYNTAX_DEPTH_QUERY_CATEGORY,
            "correct_query": original_query,
            "correct_query_source_field": CORRECT_QUERY_SOURCE_FIELD,
            "source_query": original_query,
            "ground_truth_query": original_query,
            "noisy_query": noisy_query,
            "original_query": original_query,
            "query_rewritten": noisy_query != original_query,
            "injection_mode": injection_source,
            "injection_source": injection_source,
            "injected_errors": [{"target_token_depth": injection_depth}],
            "injection_depth": injection_depth,
            "injection_depths": injection_depths,
            "injection_depth_group": injection_depth_group,
            "injection_depth_group_display": injection_depth_group_display,
            "syntax_depth": injection_depth,
            "target_depth": injection_depth,
            "word_count": len(original_query.split()),
            "user_avg_depth": float(injection_depth),
            "syntax_depth_group": syntax_depth_group,
            "syntax_depth_group_display": DEPTH_GROUP_DISPLAY[syntax_depth_group],
            "attrs_used": attrs_used,
        }
        pairs.append(pair)

    if not pairs:
        raise ValueError(
            f"No noisy records found with injection_source={NOISY_INJECTION_SOURCE} in {noisy_query_file}"
        )

    rewritten_count = sum(1 for pair in pairs if pair["query_rewritten"] is True)
    log(
        f"加载 noisy syntax-depth 配对: {len(pairs)} 条，"
        f"跳过非 syntax-depth 记录 {skipped_non_syntax_depth} 条，"
        f"跳过不在当前分组范围内的记录 {skipped_out_of_group} 条，"
        f"其中 query_rewritten=True 为 {rewritten_count} 条"
    )
    log("Correct 侧使用 07 noisy_query.json 的 original_query；Noisy 侧使用 noisy_query；深度字段来自 07 injection_target_depth")
    return pairs


def get_query_cache_path(query_cache_base_dir: str, retriever_name: str, query_type: str) -> str:
    if query_type not in (CORRECT_QUERY_TYPE, NOISY_QUERY_TYPE):
        raise ValueError(f"Unsupported query_type: {query_type}")
    return os.path.join(
        query_cache_base_dir,
        f"{SYNTAX_DEPTH_QUERY_CATEGORY}_{query_type}_query",
        f"{retriever_name}__{SYNTAX_DEPTH_QUERY_CATEGORY}_{query_type}_cache.pkl",
    )


def load_query_cache(query_cache_base_dir: str, retriever_name: str, query_type: str) -> Dict:
    cache_path = get_query_cache_path(query_cache_base_dir, retriever_name, query_type)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{retriever_name} {query_type} query cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = NumpyCoreCompatUnpickler(f).load()
    if not isinstance(cache, dict):
        raise TypeError(f"{retriever_name} {query_type} query cache must be dict, got {type(cache).__name__}")
    return cache


def load_dense_retriever_cache(cache_dir: str, retriever_name: str) -> Tuple[np.ndarray, List[str]]:
    embeddings_path = None
    for filename in sorted(os.listdir(cache_dir)):
        if filename.startswith(f"{retriever_name}_") and filename.endswith("_embeddings.npy"):
            embeddings_path = os.path.join(cache_dir, filename)
            break
    if embeddings_path is None:
        raise FileNotFoundError(f"{retriever_name} document embeddings not found in {cache_dir}")

    embeddings = np.load(embeddings_path, mmap_mode="r")[:].copy()
    doc_ids_path = embeddings_path.replace("_embeddings.npy", "_doc_ids.pkl")
    if not os.path.exists(doc_ids_path):
        raise FileNotFoundError(f"{retriever_name} doc id cache not found: {doc_ids_path}")
    with open(doc_ids_path, "rb") as f:
        doc_ids = pickle.load(f)
    if not isinstance(doc_ids, list):
        raise TypeError(f"{retriever_name} doc ids must be list, got {type(doc_ids).__name__}")
    if embeddings.shape[0] != len(doc_ids):
        raise ValueError(
            f"{retriever_name} document cache mismatch: embeddings={embeddings.shape[0]}, doc_ids={len(doc_ids)}"
        )
    return embeddings, doc_ids


def load_splade_retriever(cache_dir: str):
    splade_path = None
    for filename in sorted(os.listdir(cache_dir)):
        if filename.startswith("splade_") and filename.endswith(".pkl"):
            splade_path = os.path.join(cache_dir, filename)
            break
    if splade_path is None:
        raise FileNotFoundError(f"SPLADE retriever cache not found in {cache_dir}")
    with open(splade_path, "rb") as f:
        retriever = pickle.load(f)
    return retriever


def compute_metrics(relevant_asin: str, retrieved_asins: List[str], k_values: List[int]) -> Dict:
    metrics = {}
    for k in k_values:
        top_k = retrieved_asins[:k]
        metrics[f"P@{k}"] = 1.0 if relevant_asin in top_k else 0.0
        if relevant_asin in top_k:
            rank = top_k.index(relevant_asin) + 1
            metrics[f"N@{k}"] = 1.0 / np.log2(rank + 1)
            metrics[f"MR@{k}"] = 1.0 / rank
        else:
            metrics[f"N@{k}"] = 0.0
            metrics[f"MR@{k}"] = 0.0
        metrics[f"H@{k}"] = 1.0 if relevant_asin in top_k else 0.0
    return metrics


def compute_average_metrics(all_metrics: List[Dict], k_values: List[int]) -> Dict:
    if not all_metrics:
        raise ValueError("Cannot compute average metrics for an empty metric list")
    avg_metrics = {}
    for k in k_values:
        avg_metrics[f"P@{k}"] = float(np.mean([metric[f"P@{k}"] for metric in all_metrics]))
        avg_metrics[f"N@{k}"] = float(np.mean([metric[f"N@{k}"] for metric in all_metrics]))
        avg_metrics[f"MR@{k}"] = float(np.mean([metric[f"MR@{k}"] for metric in all_metrics]))
        avg_metrics[f"H@{k}"] = float(np.mean([metric[f"H@{k}"] for metric in all_metrics]))
    return avg_metrics


def compute_group_metrics(all_metrics: List[Dict], records: List[Dict], k_values: List[int]) -> Dict:
    if len(all_metrics) != len(records):
        raise ValueError(f"Metric/record length mismatch: {len(all_metrics)} vs {len(records)}")

    grouped = defaultdict(list)
    for metric, record in zip(all_metrics, records):
        grouped[record["syntax_depth_group"]].append(metric)

    result = {}
    for group_name in DEPTH_GROUP_ORDER:
        metrics = grouped[group_name]
        result[group_name] = {
            "display": DEPTH_GROUP_DISPLAY[group_name],
            "num_queries": len(metrics),
            "metrics": compute_average_metrics(metrics, k_values) if metrics else {},
        }
    return result


def compute_injection_depth_group_metrics(all_metrics: List[Dict], records: List[Dict], k_values: List[int]) -> Dict:
    if len(all_metrics) != len(records):
        raise ValueError(f"Metric/record length mismatch: {len(all_metrics)} vs {len(records)}")

    grouped = defaultdict(list)
    for metric, record in zip(all_metrics, records):
        group_name = record["injection_depth_group"]
        if group_name is None:
            raise ValueError("Encountered ungrouped record while computing injection depth metrics")
        if group_name not in INJECTION_DEPTH_GROUP_DISPLAY:
            raise ValueError(f"Unsupported injection depth group: {group_name}")
        grouped[group_name].append(metric)

    result = {}
    for group_name in INJECTION_DEPTH_GROUP_ORDER:
        metrics = grouped[group_name]
        result[group_name] = {
            "display": INJECTION_DEPTH_GROUP_DISPLAY[group_name],
            "num_queries": len(metrics),
            "metrics": compute_average_metrics(metrics, k_values) if metrics else {},
        }
    return result


def compute_exact_depth_metrics(all_metrics: List[Dict], records: List[Dict], k_values: List[int]) -> Dict:
    grouped = defaultdict(list)
    for metric, record in zip(all_metrics, records):
        grouped[record["syntax_depth"]].append(metric)

    result = {}
    for depth in sorted(grouped.keys()):
        metrics = grouped[depth]
        result[str(depth)] = {
            "num_queries": len(metrics),
            "metrics": compute_average_metrics(metrics, k_values),
        }
    return result


def compute_exact_injection_depth_metrics(all_metrics: List[Dict], records: List[Dict], k_values: List[int]) -> Dict:
    grouped = defaultdict(list)
    for metric, record in zip(all_metrics, records):
        grouped[record["injection_depth"]].append(metric)

    result = {}
    for depth in sorted(grouped.keys()):
        metrics = grouped[depth]
        result[str(depth)] = {
            "num_queries": len(metrics),
            "metrics": compute_average_metrics(metrics, k_values),
        }
    return result


def query_record_for_output(pair: Dict, query_type: str, metric: Dict) -> Dict:
    if query_type == CORRECT_QUERY_TYPE:
        query_text = pair["correct_query"]
    elif query_type == NOISY_QUERY_TYPE:
        query_text = pair["noisy_query"]
    else:
        raise ValueError(f"Unsupported query_type: {query_type}")

    return {
        "pair_id": pair["pair_id"],
        "user_id": pair["user_id"],
        "asin": pair["asin"],
        "query_type": query_type,
        "query_category": SYNTAX_DEPTH_QUERY_CATEGORY,
        "query": query_text,
        "correct_query": pair["correct_query"],
        "correct_query_source_field": pair["correct_query_source_field"],
        "source_query": pair["source_query"],
        "ground_truth_query": pair["ground_truth_query"],
        "noisy_query": pair["noisy_query"],
        "query_rewritten": pair["query_rewritten"],
        "injection_mode": pair["injection_mode"],
        "injected_errors": pair["injected_errors"],
        "injection_depth": pair["injection_depth"],
        "injection_depth_group": pair["injection_depth_group"],
        "injection_depth_group_display": pair["injection_depth_group_display"],
        "syntax_depth": pair["syntax_depth"],
        "target_depth": pair["target_depth"],
        "syntax_depth_group": pair["syntax_depth_group"],
        "syntax_depth_group_display": pair["syntax_depth_group_display"],
        "word_count": pair["word_count"],
        "user_avg_depth": pair["user_avg_depth"],
        "attrs_used": pair["attrs_used"],
        "p_at10": metric.get("P@10", 0.0),
        "metrics": dict(metric),
    }


def build_retriever_result_from_query_records(
    retriever_name: str,
    query_type: str,
    query_records: List[Dict],
    k_values: List[int],
    cache_summary: Dict,
) -> Dict:
    if not query_records:
        raise ValueError(f"{retriever_name} {query_type} produced no query records")

    all_metrics = []
    for record_index, record in enumerate(query_records):
        if "metrics" not in record:
            raise ValueError(f"{retriever_name} {query_type} query record {record_index} missing metrics")
        metrics = record["metrics"]
        if not isinstance(metrics, dict):
            raise TypeError(
                f"{retriever_name} {query_type} query record {record_index} metrics must be dict, got {type(metrics).__name__}"
            )
        all_metrics.append(metrics)

    return {
        "retriever": retriever_name,
        "query_type": query_type,
        "query_category": SYNTAX_DEPTH_QUERY_CATEGORY,
        "num_queries": len(query_records),
        "metrics": compute_average_metrics(all_metrics, k_values),
        "metrics_by_depth_group": compute_group_metrics(all_metrics, query_records, k_values),
        "metrics_by_exact_depth": compute_exact_depth_metrics(all_metrics, query_records, k_values),
        "metrics_by_injection_depth_group": compute_injection_depth_group_metrics(all_metrics, query_records, k_values),
        "metrics_by_exact_injection_depth": compute_exact_injection_depth_metrics(all_metrics, query_records, k_values),
        "cache_summary": cache_summary,
        "all_query_records": query_records,
    }


def build_retriever_result(
    retriever_name: str,
    query_type: str,
    all_metrics: List[Dict],
    records: List[Dict],
    k_values: List[int],
    cache_summary: Dict,
) -> Dict:
    if len(all_metrics) != len(records):
        raise ValueError(
            f"{retriever_name} {query_type} metric/record length mismatch: {len(all_metrics)} vs {len(records)}"
        )
    if not all_metrics:
        raise ValueError(f"{retriever_name} {query_type} produced no metrics")

    query_records = [
        query_record_for_output(pair, query_type, metric)
        for pair, metric in zip(records, all_metrics)
    ]
    return build_retriever_result_from_query_records(
        retriever_name,
        query_type,
        query_records,
        k_values,
        cache_summary,
    )


def filter_retriever_pair_results(
    correct_result: Dict,
    noisy_result: Dict,
    k_values: List[int],
    exclude_metric_name: str = SUMMARY_EXCLUDE_METRIC,
) -> Tuple[Dict, Dict, Dict]:
    correct_records = correct_result["all_query_records"]
    noisy_records = noisy_result["all_query_records"]
    if len(correct_records) != len(noisy_records):
        raise ValueError(
            f"{correct_result['retriever']} query record length mismatch: "
            f"{len(correct_records)} vs {len(noisy_records)}"
        )

    filtered_correct_records = []
    filtered_noisy_records = []
    excluded_records = []
    for record_index, (correct_record, noisy_record) in enumerate(zip(correct_records, noisy_records)):
        if correct_record["pair_id"] != noisy_record["pair_id"]:
            raise ValueError(
                f"{correct_result['retriever']} pair_id mismatch at index {record_index}: "
                f"{correct_record['pair_id']} vs {noisy_record['pair_id']}"
            )
        if exclude_metric_name not in correct_record["metrics"]:
            raise KeyError(
                f"{correct_result['retriever']} correct record pair_id={correct_record['pair_id']} missing metric {exclude_metric_name}"
            )
        if exclude_metric_name not in noisy_record["metrics"]:
            raise KeyError(
                f"{correct_result['retriever']} noisy record pair_id={noisy_record['pair_id']} missing metric {exclude_metric_name}"
            )

        correct_value = correct_record["metrics"][exclude_metric_name]
        noisy_value = noisy_record["metrics"][exclude_metric_name]
        if noisy_value > correct_value:
            excluded_records.append(
                {
                    "pair_id": correct_record["pair_id"],
                    "user_id": correct_record["user_id"],
                    "asin": correct_record["asin"],
                    "injection_depth": correct_record["injection_depth"],
                    "injection_depth_group": correct_record["injection_depth_group"],
                    "query": noisy_record["query"],
                    "correct_query": correct_record["query"],
                    "noisy_query": noisy_record["query"],
                    "correct_value": correct_value,
                    "noisy_value": noisy_value,
                }
            )
            continue

        filtered_correct_records.append(correct_record)
        filtered_noisy_records.append(noisy_record)

    if not filtered_correct_records:
        raise ValueError(
            f"{correct_result['retriever']} has no query records left after excluding noisy-better cases"
        )

    filtered_correct_result = build_retriever_result_from_query_records(
        correct_result["retriever"],
        correct_result["query_type"],
        filtered_correct_records,
        k_values,
        correct_result["cache_summary"],
    )
    filtered_noisy_result = build_retriever_result_from_query_records(
        noisy_result["retriever"],
        noisy_result["query_type"],
        filtered_noisy_records,
        k_values,
        noisy_result["cache_summary"],
    )

    filter_summary = {
        "retriever": correct_result["retriever"],
        "exclude_metric_name": exclude_metric_name,
        "excluded_count": len(excluded_records),
        "kept_count": len(filtered_correct_records),
        "original_count": len(correct_records),
        "excluded_records": excluded_records,
    }
    filtered_correct_result["final_statistics_filter"] = filter_summary
    filtered_noisy_result["final_statistics_filter"] = filter_summary
    return filtered_correct_result, filtered_noisy_result, filter_summary


def select_pairs_for_user_query_caches(
    pairs: List[Dict],
    correct_cache: Dict,
    noisy_cache: Dict,
) -> Tuple[List[Dict], List, List, Dict]:
    selected_pairs = []
    correct_values = []
    noisy_values = []
    missing_correct_user = 0
    missing_correct_query = 0
    missing_noisy_user = 0
    missing_noisy_query = 0

    for pair in pairs:
        user_id = pair["user_id"]
        correct_query = pair["correct_query"]
        noisy_query = pair["noisy_query"]

        if user_id not in correct_cache:
            missing_correct_user += 1
            continue
        if user_id not in noisy_cache:
            missing_noisy_user += 1
            continue

        correct_user_cache = correct_cache[user_id]
        noisy_user_cache = noisy_cache[user_id]
        if not isinstance(correct_user_cache, dict):
            raise TypeError(f"correct cache for user={user_id} must be dict")
        if not isinstance(noisy_user_cache, dict):
            raise TypeError(f"noisy cache for user={user_id} must be dict")

        if correct_query not in correct_user_cache:
            missing_correct_query += 1
            continue
        if noisy_query not in noisy_user_cache:
            missing_noisy_query += 1
            continue

        selected_pairs.append(pair)
        correct_values.append(correct_user_cache[correct_query])
        noisy_values.append(noisy_user_cache[noisy_query])

    summary = {
        "candidate_pairs": len(pairs),
        "selected_pairs": len(selected_pairs),
        "missing_correct_user": missing_correct_user,
        "missing_correct_query": missing_correct_query,
        "missing_noisy_user": missing_noisy_user,
        "missing_noisy_query": missing_noisy_query,
    }
    if not selected_pairs:
        raise ValueError(f"No pairs have both correct and noisy user-query cache entries: {summary}")
    return selected_pairs, correct_values, noisy_values, summary


def select_pairs_for_bm25_caches(
    pairs: List[Dict],
    correct_cache: Dict,
    noisy_cache: Dict,
) -> Tuple[List[Dict], List, List, Dict]:
    selected_pairs = []
    correct_results = []
    noisy_results = []
    missing_correct_query = 0
    missing_noisy_query = 0

    for pair in pairs:
        correct_query = pair["correct_query"]
        noisy_query = pair["noisy_query"]
        if correct_query not in correct_cache:
            missing_correct_query += 1
            continue
        if noisy_query not in noisy_cache:
            missing_noisy_query += 1
            continue
        selected_pairs.append(pair)
        correct_results.append(correct_cache[correct_query])
        noisy_results.append(noisy_cache[noisy_query])

    summary = {
        "candidate_pairs": len(pairs),
        "selected_pairs": len(selected_pairs),
        "missing_correct_query": missing_correct_query,
        "missing_noisy_query": missing_noisy_query,
    }
    if not selected_pairs:
        raise ValueError(f"No pairs have both correct and noisy BM25 cache entries: {summary}")
    return selected_pairs, correct_results, noisy_results, summary


class DenseSearcher:
    def __init__(self, embeddings: np.ndarray, doc_ids: List[str]):
        self.doc_ids = doc_ids
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_embeddings = embeddings / norms
        self.embeddings_tensor = torch.from_numpy(normalized_embeddings).float().to(self.device)

    def search_batch(self, query_embeddings: List[np.ndarray], top_k: int) -> List[List[Tuple[str, float]]]:
        if not query_embeddings:
            raise ValueError("Dense search received no query embeddings")
        query_array = np.array(query_embeddings)
        query_tensor = torch.from_numpy(query_array).float().to(self.device)
        q_norms = np.linalg.norm(query_array, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1, q_norms)
        query_tensor = query_tensor / torch.from_numpy(q_norms).float().to(self.device)
        scores = torch.mm(query_tensor, self.embeddings_tensor.T)

        results = []
        for query_index in range(len(query_embeddings)):
            top_scores, top_indices = torch.topk(scores[query_index], min(top_k, len(self.doc_ids)))
            results.append(
                [
                    (self.doc_ids[idx.item()], float(top_scores[pos].item()))
                    for pos, idx in enumerate(top_indices)
                ]
            )
        return results


def results_to_metrics(results: List[List[Tuple[str, float]]], pairs: List[Dict], k_values: List[int]) -> List[Dict]:
    if len(results) != len(pairs):
        raise ValueError(f"Result/pair length mismatch: {len(results)} vs {len(pairs)}")
    metrics = []
    for retrieved, pair in zip(results, pairs):
        retrieved_asins = [asin for asin, _ in retrieved]
        metrics.append(compute_metrics(pair["asin"], retrieved_asins, k_values))
    return metrics


def evaluate_dense_pair(
    category_config: Dict,
    retriever_name: str,
    pairs: List[Dict],
    k_values: List[int],
) -> Tuple[Dict, Dict]:
    log(f"\n{'=' * 60}")
    log(f"评估 {retriever_name.upper()} - syntax_depth correct/noisy")
    log(f"{'=' * 60}")

    query_cache_base_dir = category_config["query_cache_dir"]
    correct_cache = load_query_cache(query_cache_base_dir, retriever_name, CORRECT_QUERY_TYPE)
    noisy_cache = load_query_cache(query_cache_base_dir, retriever_name, NOISY_QUERY_TYPE)
    selected_pairs, correct_embeddings, noisy_embeddings, cache_summary = select_pairs_for_user_query_caches(
        pairs,
        correct_cache,
        noisy_cache,
    )
    log(f"  缓存配对命中: {cache_summary}")

    embeddings, doc_ids = load_dense_retriever_cache(category_config["retriever_cache_dir"], retriever_name)
    searcher = DenseSearcher(embeddings, doc_ids)

    correct_results = searcher.search_batch(correct_embeddings, top_k=max(k_values))
    noisy_results = searcher.search_batch(noisy_embeddings, top_k=max(k_values))
    correct_metrics = results_to_metrics(correct_results, selected_pairs, k_values)
    noisy_metrics = results_to_metrics(noisy_results, selected_pairs, k_values)

    del embeddings
    del searcher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        build_retriever_result(retriever_name, CORRECT_QUERY_TYPE, correct_metrics, selected_pairs, k_values, cache_summary),
        build_retriever_result(retriever_name, NOISY_QUERY_TYPE, noisy_metrics, selected_pairs, k_values, cache_summary),
    )


def evaluate_bm25_pair(
    category_config: Dict,
    pairs: List[Dict],
    k_values: List[int],
) -> Tuple[Dict, Dict]:
    retriever_name = "bm25"
    log(f"\n{'=' * 60}")
    log("评估 BM25 - syntax_depth correct/noisy")
    log(f"{'=' * 60}")

    query_cache_base_dir = category_config["query_cache_dir"]
    correct_cache = load_query_cache(query_cache_base_dir, retriever_name, CORRECT_QUERY_TYPE)
    noisy_cache = load_query_cache(query_cache_base_dir, retriever_name, NOISY_QUERY_TYPE)
    selected_pairs, correct_results, noisy_results, cache_summary = select_pairs_for_bm25_caches(
        pairs,
        correct_cache,
        noisy_cache,
    )
    log(f"  缓存配对命中: {cache_summary}")

    correct_metrics = results_to_metrics(correct_results, selected_pairs, k_values)
    noisy_metrics = results_to_metrics(noisy_results, selected_pairs, k_values)
    return (
        build_retriever_result(retriever_name, CORRECT_QUERY_TYPE, correct_metrics, selected_pairs, k_values, cache_summary),
        build_retriever_result(retriever_name, NOISY_QUERY_TYPE, noisy_metrics, selected_pairs, k_values, cache_summary),
    )


def splade_search_from_vectors(retriever, query_vectors: List[Dict], top_k: int) -> List[List[Tuple[str, float]]]:
    from scipy import sparse

    inverted_index = retriever._inverted_index
    if not inverted_index:
        raise ValueError("SPLADE inverted index is empty")
    n_docs = len(retriever.doc_ids)

    doc_rows = []
    doc_cols = []
    doc_data = []
    max_doc_term = 0
    for term_id, doc_list in inverted_index.items():
        term_id_int = int(term_id)
        max_doc_term = max(max_doc_term, term_id_int)
        for doc_idx, d_weight in doc_list:
            doc_rows.append(term_id_int)
            doc_cols.append(doc_idx)
            doc_data.append(d_weight)

    q_rows = []
    q_cols = []
    q_data = []
    max_query_term = 0
    for q_idx, q_vec in enumerate(query_vectors):
        if not isinstance(q_vec, dict):
            raise TypeError(f"SPLADE query vector must be dict, got {type(q_vec).__name__}")
        for term_id, q_weight in q_vec.items():
            term_id_int = int(term_id)
            max_query_term = max(max_query_term, term_id_int)
            q_rows.append(q_idx)
            q_cols.append(term_id_int)
            q_data.append(q_weight)

    n_terms = max(max_doc_term, max_query_term) + 1
    doc_matrix = sparse.csr_matrix(
        (doc_data, (doc_rows, doc_cols)),
        shape=(n_terms, n_docs),
        dtype=np.float32,
    )
    query_matrix = sparse.csr_matrix(
        (q_data, (q_rows, q_cols)),
        shape=(len(query_vectors), n_terms),
        dtype=np.float32,
    )

    score_matrix = query_matrix @ doc_matrix
    results = []
    for row_index in range(score_matrix.shape[0]):
        row = score_matrix.getrow(row_index)
        scores_vec = row.toarray().flatten()
        top_indices = np.argsort(scores_vec)[::-1][:top_k]
        results.append([(retriever.doc_ids[idx], float(scores_vec[idx])) for idx in top_indices])
    return results


def evaluate_splade_pair(
    category_config: Dict,
    pairs: List[Dict],
    k_values: List[int],
) -> Tuple[Dict, Dict]:
    retriever_name = "splade"
    log(f"\n{'=' * 60}")
    log("评估 SPLADE - syntax_depth correct/noisy")
    log(f"{'=' * 60}")

    query_cache_base_dir = category_config["query_cache_dir"]
    correct_cache = load_query_cache(query_cache_base_dir, retriever_name, CORRECT_QUERY_TYPE)
    noisy_cache = load_query_cache(query_cache_base_dir, retriever_name, NOISY_QUERY_TYPE)
    selected_pairs, correct_vectors, noisy_vectors, cache_summary = select_pairs_for_user_query_caches(
        pairs,
        correct_cache,
        noisy_cache,
    )
    log(f"  缓存配对命中: {cache_summary}")

    retriever = load_splade_retriever(category_config["retriever_cache_dir"])
    retriever.search(["dummy"], top_k=1)

    correct_results = splade_search_from_vectors(retriever, correct_vectors, max(k_values))
    noisy_results = splade_search_from_vectors(retriever, noisy_vectors, max(k_values))
    correct_metrics = results_to_metrics(correct_results, selected_pairs, k_values)
    noisy_metrics = results_to_metrics(noisy_results, selected_pairs, k_values)

    return (
        build_retriever_result(retriever_name, CORRECT_QUERY_TYPE, correct_metrics, selected_pairs, k_values, cache_summary),
        build_retriever_result(retriever_name, NOISY_QUERY_TYPE, noisy_metrics, selected_pairs, k_values, cache_summary),
    )


def validate_colbertv2_paths() -> None:
    required_paths = [
        os.path.join(COLBERTV2_CUDA_HOME, "include", "cuda_runtime.h"),
        os.path.join(COLBERTV2_CUDA_HOME, "bin", "nvcc"),
        COLBERTV2_HOST_CC,
        COLBERTV2_HOST_CXX,
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required ColBERTv2 runtime path not found: {path}")


def configure_colbertv2_env(category_name: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("ColBERTv2 evaluation requires CUDA")
    validate_colbertv2_paths()

    os.environ["CUDA_HOME"] = COLBERTV2_CUDA_HOME
    os.environ["CUDA_PATH"] = COLBERTV2_CUDA_HOME
    os.environ["CUDACXX"] = os.path.join(COLBERTV2_CUDA_HOME, "bin", "nvcc")
    os.environ["CC"] = COLBERTV2_HOST_CC
    os.environ["CXX"] = COLBERTV2_HOST_CXX
    os.environ["CUDAHOSTCXX"] = COLBERTV2_HOST_CXX
    for env_name, env_value in [
        ("PATH", os.path.join(COLBERTV2_CUDA_HOME, "bin")),
        ("CPATH", os.path.join(COLBERTV2_CUDA_HOME, "include")),
        ("LIBRARY_PATH", os.path.join(COLBERTV2_CUDA_HOME, "lib64")),
        ("LD_LIBRARY_PATH", os.path.join(COLBERTV2_CUDA_HOME, "lib64")),
    ]:
        existing_env_value = os.environ.get(env_name)
        os.environ[env_name] = env_value if not existing_env_value else f"{env_value}:{existing_env_value}"

    major, minor = torch.cuda.get_device_capability()
    ext_dir = os.path.join(
        COLBERTV2_TORCH_EXTENSIONS_BASE_DIR,
        f"colbertv2_cuda125_sm{major}{minor}",
    )
    os.makedirs(ext_dir, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = ext_dir
    os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "True"
    log(f"[ColBERT] shared TORCH_EXTENSIONS_DIR = {ext_dir}")
    return ext_dir


def resolve_colbertv2_output_root(category_name: str, cache_dir: str) -> str:
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Retriever cache directory not found: {cache_dir}")

    candidates = []
    for name in sorted(os.listdir(cache_dir)):
        output_root = os.path.join(cache_dir, name)
        if not name.startswith("colbertv2_") or not os.path.isdir(output_root):
            continue
        manifest_path = os.path.join(output_root, "build_manifest.json")
        doc_ids_path = os.path.join(output_root, "doc_ids.pkl")
        index_dir = os.path.join(output_root, "colbertv2_index", "indexes", "colbertv2_index")
        if os.path.isfile(manifest_path) and os.path.isfile(doc_ids_path) and os.path.isdir(index_dir):
            candidates.append(output_root)

    if not candidates:
        raise FileNotFoundError(f"No complete ColBERTv2 cache directory found under: {cache_dir}")
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one ColBERTv2 cache directory, found {len(candidates)}: {candidates}")

    output_root = candidates[0]
    manifest_path = os.path.join(output_root, "build_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise TypeError(f"ColBERTv2 build manifest must be dict, got {type(manifest).__name__}: {manifest_path}")
    for required_field in ("category", "output_root"):
        if required_field not in manifest:
            raise ValueError(f"ColBERTv2 build manifest missing required field '{required_field}': {manifest_path}")
    if manifest["category"] != category_name:
        raise ValueError(f"ColBERTv2 cache category mismatch in {manifest_path}: {manifest['category']}")
    if manifest["output_root"] != output_root:
        raise ValueError(f"ColBERTv2 cache output_root mismatch in {manifest_path}: {manifest['output_root']}")

    return output_root


def load_colbertv2_doc_ids(output_root: str) -> List[str]:
    doc_ids_path = os.path.join(output_root, "doc_ids.pkl")
    if not os.path.exists(doc_ids_path):
        raise FileNotFoundError(f"Required ColBERTv2 doc id mapping not found: {doc_ids_path}")
    with open(doc_ids_path, "rb") as f:
        doc_ids = pickle.load(f)
    if not isinstance(doc_ids, list):
        raise TypeError(f"ColBERTv2 doc_ids must be list, got {type(doc_ids).__name__}")
    if not doc_ids:
        raise ValueError(f"ColBERTv2 doc_ids is empty: {doc_ids_path}")
    return doc_ids


def resolve_local_hf_snapshot(repo_id: str) -> str:
    repo_cache_dir = os.path.join(os.environ["HF_HOME"], "models--" + repo_id.replace("/", "--"))
    ref_file = os.path.join(repo_cache_dir, "refs", "main")
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Hugging Face ref file not found for {repo_id}: {ref_file}")
    with open(ref_file, "r", encoding="utf-8") as f:
        snapshot_hash = f.read().strip()
    snapshot_dir = os.path.join(repo_cache_dir, "snapshots", snapshot_hash)
    if not os.path.exists(snapshot_dir):
        raise FileNotFoundError(f"Hugging Face snapshot not found for {repo_id}: {snapshot_dir}")
    return snapshot_dir


def build_colbertv2_searcher(category_name: str, output_root: str, doc_ids: List[str]):
    configure_colbertv2_env(category_name)

    from colbert.infra import ColBERTConfig, Run, RunConfig
    from colbert import Searcher

    checkpoint_path = resolve_local_hf_snapshot("colbert-ir/colbertv2.0")
    collection = [f"pid {pid} asin {asin}" for pid, asin in enumerate(doc_ids)]
    log(f"[ColBERT] checkpoint_path = {checkpoint_path}")
    log(f"[ColBERT] output_root = {output_root}")

    start = time.time()
    with Run().context(RunConfig(experiment="colbertv2_index", root=output_root)):
        config = ColBERTConfig(root=output_root)
        searcher = Searcher(
            index="colbertv2_index",
            checkpoint=checkpoint_path,
            collection=collection,
            config=config,
        )
    log(f"[ColBERT] Searcher loaded in {time.time() - start:.1f}s")
    return searcher


def colbertv2_search_from_cached_embedding(searcher, doc_ids: List[str], query_embedding, top_k: int):
    if not isinstance(query_embedding, np.ndarray):
        raise TypeError(f"ColBERTv2 cached query embedding must be numpy.ndarray, got {type(query_embedding).__name__}")
    if query_embedding.ndim != 2:
        raise ValueError(f"ColBERTv2 cached query embedding must be 2D, got shape {query_embedding.shape}")

    query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0)
    pids, _, scores = searcher.dense_search(query_tensor, k=top_k)

    results = []
    for pid, score in zip(pids, scores):
        pid_int = int(pid)
        if pid_int < 0 or pid_int >= len(doc_ids):
            raise IndexError(f"ColBERTv2 pid {pid_int} is outside doc_ids range 0..{len(doc_ids) - 1}")
        results.append((doc_ids[pid_int], float(score)))
    return results


def evaluate_colbertv2_pair(
    category_name: str,
    category_config: Dict,
    pairs: List[Dict],
    k_values: List[int],
) -> Tuple[Dict, Dict]:
    retriever_name = "colbertv2"
    log(f"\n{'=' * 60}")
    log("评估 COLBERTV2 - syntax_depth correct/noisy")
    log(f"{'=' * 60}")

    query_cache_base_dir = category_config["query_cache_dir"]
    correct_cache = load_query_cache(query_cache_base_dir, retriever_name, CORRECT_QUERY_TYPE)
    noisy_cache = load_query_cache(query_cache_base_dir, retriever_name, NOISY_QUERY_TYPE)
    selected_pairs, correct_embeddings, noisy_embeddings, cache_summary = select_pairs_for_user_query_caches(
        pairs,
        correct_cache,
        noisy_cache,
    )
    log(f"  缓存配对命中: {cache_summary}")

    output_root = resolve_colbertv2_output_root(category_name, category_config["retriever_cache_dir"])
    index_dir = os.path.join(output_root, "colbertv2_index", "indexes", "colbertv2_index")
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"Required ColBERTv2 index directory not found: {index_dir}")
    doc_ids = load_colbertv2_doc_ids(output_root)
    searcher = build_colbertv2_searcher(category_name, output_root, doc_ids)

    correct_results = []
    noisy_results = []
    for index, (correct_embedding, noisy_embedding) in enumerate(zip(correct_embeddings, noisy_embeddings)):
        correct_results.append(
            colbertv2_search_from_cached_embedding(searcher, doc_ids, correct_embedding, max(k_values))
        )
        noisy_results.append(
            colbertv2_search_from_cached_embedding(searcher, doc_ids, noisy_embedding, max(k_values))
        )
        if (index + 1) % 100 == 0 or index + 1 == len(correct_embeddings):
            log(f"    ColBERTv2 搜索进度: {index + 1}/{len(correct_embeddings)}")

    correct_metrics = results_to_metrics(correct_results, selected_pairs, k_values)
    noisy_metrics = results_to_metrics(noisy_results, selected_pairs, k_values)
    return (
        build_retriever_result(retriever_name, CORRECT_QUERY_TYPE, correct_metrics, selected_pairs, k_values, cache_summary),
        build_retriever_result(retriever_name, NOISY_QUERY_TYPE, noisy_metrics, selected_pairs, k_values, cache_summary),
    )


def metric_diff(correct_result: Dict, noisy_result: Dict) -> Dict:
    metrics = {}
    for metric_name, correct_value in correct_result["metrics"].items():
        metrics[metric_name] = float(noisy_result["metrics"][metric_name] - correct_value)

    by_group = {}
    for group_name in DEPTH_GROUP_ORDER:
        correct_group = correct_result["metrics_by_depth_group"][group_name]
        noisy_group = noisy_result["metrics_by_depth_group"][group_name]
        if correct_group["num_queries"] != noisy_group["num_queries"]:
            raise ValueError(
                f"Depth group query count mismatch for {correct_result['retriever']} {group_name}: "
                f"{correct_group['num_queries']} vs {noisy_group['num_queries']}"
            )
        if correct_group["num_queries"] == 0:
            by_group[group_name] = {
                "display": DEPTH_GROUP_DISPLAY[group_name],
                "num_queries": 0,
                "metrics": {},
            }
        else:
            by_group[group_name] = {
                "display": DEPTH_GROUP_DISPLAY[group_name],
                "num_queries": correct_group["num_queries"],
                "metrics": {
                    metric_name: float(noisy_group["metrics"][metric_name] - correct_group["metrics"][metric_name])
                    for metric_name in correct_group["metrics"]
                },
            }

    injection_by_group = {}
    for group_name in INJECTION_DEPTH_GROUP_ORDER:
        correct_group = correct_result["metrics_by_injection_depth_group"][group_name]
        noisy_group = noisy_result["metrics_by_injection_depth_group"][group_name]
        if correct_group["num_queries"] != noisy_group["num_queries"]:
            raise ValueError(
                f"Injection depth group query count mismatch for {correct_result['retriever']} {group_name}: "
                f"{correct_group['num_queries']} vs {noisy_group['num_queries']}"
            )
        if correct_group["num_queries"] == 0:
            injection_by_group[group_name] = {
                "display": INJECTION_DEPTH_GROUP_DISPLAY[group_name],
                "num_queries": 0,
                "metrics": {},
            }
        else:
            injection_by_group[group_name] = {
                "display": INJECTION_DEPTH_GROUP_DISPLAY[group_name],
                "num_queries": correct_group["num_queries"],
                "metrics": {
                    metric_name: float(noisy_group["metrics"][metric_name] - correct_group["metrics"][metric_name])
                    for metric_name in correct_group["metrics"]
                },
            }

    correct_exact_injection = correct_result["metrics_by_exact_injection_depth"]
    noisy_exact_injection = noisy_result["metrics_by_exact_injection_depth"]
    if set(correct_exact_injection) != set(noisy_exact_injection):
        raise ValueError(
            f"Exact injection depth keys mismatch for {correct_result['retriever']}: "
            f"{sorted(correct_exact_injection)} vs {sorted(noisy_exact_injection)}"
        )
    injection_by_exact_depth = {}
    for depth_key in sorted(correct_exact_injection, key=int):
        correct_depth = correct_exact_injection[depth_key]
        noisy_depth = noisy_exact_injection[depth_key]
        if correct_depth["num_queries"] != noisy_depth["num_queries"]:
            raise ValueError(
                f"Exact injection depth query count mismatch for {correct_result['retriever']} depth={depth_key}: "
                f"{correct_depth['num_queries']} vs {noisy_depth['num_queries']}"
            )
        injection_by_exact_depth[depth_key] = {
            "num_queries": correct_depth["num_queries"],
            "metrics": {
                metric_name: float(noisy_depth["metrics"][metric_name] - correct_depth["metrics"][metric_name])
                for metric_name in correct_depth["metrics"]
            },
        }

    return {
        "retriever": correct_result["retriever"],
        "num_queries": correct_result["num_queries"],
        "metrics_noisy_minus_correct": metrics,
        "metrics_by_depth_group_noisy_minus_correct": by_group,
        "metrics_by_injection_depth_group_noisy_minus_correct": injection_by_group,
        "metrics_by_exact_injection_depth_noisy_minus_correct": injection_by_exact_depth,
    }


def print_results_table(results: List[Dict], title: str) -> None:
    log(f"\n{'=' * 100}")
    log(title)
    log("=" * 100)
    metrics_to_show = ["P@1", "P@3", "P@5", "P@10", "N@10", "MR@10", "H@10"]
    header = f"{'检索器':<12} {'N':>8}"
    for metric_name in metrics_to_show:
        header += f" {metric_name:>10}"
    log(header)
    log("-" * 100)
    for result in results:
        row = f"{result['retriever']:<12} {result['num_queries']:>8}"
        for metric_name in metrics_to_show:
            row += f" {result['metrics'][metric_name]:>10.4f}"
        log(row)
    log("-" * 100)


def print_difference_table(differences: List[Dict]) -> None:
    log(f"\n{'=' * 120}")
    log("CORRECT vs NOISY 差异分析（NOISY - CORRECT）")
    log("=" * 120)
    metrics_to_show = ["P@1", "P@3", "P@5", "P@10", "N@10", "MR@10", "H@10"]
    header = f"{'检索器':<12} {'N':>8}"
    for metric_name in metrics_to_show:
        header += f" {metric_name:>10}"
    log(header)
    log("-" * 120)
    for diff in differences:
        row = f"{diff['retriever']:<12} {diff['num_queries']:>8}"
        for metric_name in metrics_to_show:
            value = diff["metrics_noisy_minus_correct"][metric_name]
            sign = "+" if value > 0 else ""
            row += f" {sign:>1}{value:>9.4f}"
        log(row)
    log("-" * 120)


def get_injection_depths(pairs: List[Dict]) -> List[int]:
    depths = sorted({pair["injection_depth"] for pair in pairs})
    if not depths:
        raise ValueError("No injection depths found in noisy pairs")
    return depths


def log_injection_depth_distribution(pairs: List[Dict]) -> None:
    exact_counts = defaultdict(int)
    group_counts = defaultdict(int)
    ungrouped_count = 0
    for pair in pairs:
        exact_counts[pair["injection_depth"]] += 1
        group_name = pair["injection_depth_group"]
        if group_name is None:
            ungrouped_count += 1
        else:
            group_counts[group_name] += 1

    exact_summary = ", ".join(
        f"深度{depth}: {exact_counts[depth]}"
        for depth in sorted(exact_counts)
    )
    group_summary = ", ".join(
        f"{INJECTION_DEPTH_GROUP_DISPLAY[group_name]}: {group_counts[group_name]}"
        for group_name in INJECTION_DEPTH_GROUP_ORDER
        if group_counts[group_name] > 0
    )
    if ungrouped_count > 0:
        group_summary = f"{group_summary}, 未纳入当前分组: {ungrouped_count}" if group_summary else f"未纳入当前分组: {ungrouped_count}"
    log(f"注入精确深度分布: {exact_summary}")
    log(f"注入分组分布: {group_summary}")


def print_injection_depth_group_table(
    results: List[Dict],
    title: str,
    group_metrics_key: str,
    overall_metrics_key: str,
    signed: bool = False,
) -> None:
    log(f"\n{'=' * 140}")
    log(title)
    log("=" * 140)
    metrics_to_show = ["P@1", "P@3", "P@5", "P@10", "N@10", "MR@10", "H@10"]
    col_w = 12
    header = f"{'检索器':<12}"
    for group_name in INJECTION_DEPTH_GROUP_ORDER:
        group_label = INJECTION_DEPTH_GROUP_SHORT_DISPLAY[group_name]
        header += f" {f'N_{group_label}':>{col_w}}"
    header += f" {'N_ALL':>{col_w}}"
    for metric_name in metrics_to_show:
        for group_name in INJECTION_DEPTH_GROUP_ORDER:
            group_label = INJECTION_DEPTH_GROUP_SHORT_DISPLAY[group_name]
            header += f" {f'{metric_name}_{group_label}':>{col_w}}"
        header += f" {f'{metric_name}_ALL':>{col_w}}"
    log(header)
    log("-" * 140)

    for result in results:
        row = f"{result['retriever']:<12}"
        group_metrics = result[group_metrics_key]
        overall_metrics = result[overall_metrics_key]
        for group_name in INJECTION_DEPTH_GROUP_ORDER:
            if group_name not in group_metrics:
                raise KeyError(f"{result['retriever']} missing injection depth group {group_name} in {group_metrics_key}")
            row += f" {group_metrics[group_name]['num_queries']:>{col_w}}"
        row += f" {result['num_queries']:>{col_w}}"
        for metric_name in metrics_to_show:
            for group_name in INJECTION_DEPTH_GROUP_ORDER:
                group_data = group_metrics[group_name]
                if group_data["num_queries"] == 0:
                    row += f" {'NA':>{col_w}}"
                    continue
                value = group_data["metrics"][metric_name]
                if signed and value > 0:
                    row += f" {'+' + format(value, '.4f'):>{col_w}}"
                else:
                    row += f" {value:>{col_w}.4f}"
            overall_value = overall_metrics[metric_name]
            if signed and overall_value > 0:
                row += f" {'+' + format(overall_value, '.4f'):>{col_w}}"
            else:
                row += f" {overall_value:>{col_w}.4f}"
        log(row)
    log("-" * 140)


def print_hit10_injection_depth_group_table(
    correct_results: List[Dict],
    noisy_results: List[Dict],
    differences: List[Dict],
) -> None:
    print_hit10_group_trend_table(
        correct_results,
        "CORRECT H@10 注入深度分组趋势表",
    )
    print_hit10_group_trend_table(
        noisy_results,
        "NOISY H@10 注入深度分组趋势表",
    )
    print_hit10_correct_vs_noisy_group_table(correct_results, noisy_results, differences)


def pct_change(previous_value: Optional[float], next_value: Optional[float]) -> Optional[float]:
    if previous_value is None or next_value is None:
        return None
    if previous_value == 0.0:
        return None
    return float((next_value - previous_value) / previous_value * 100.0)


def format_pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def diff_value(correct_value: Optional[float], noisy_value: Optional[float]) -> Optional[float]:
    if correct_value is None or noisy_value is None:
        return None
    return float(noisy_value - correct_value)


def format_signed_metric(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.4f}"


def mean_required(values: List[float], context: str) -> float:
    if not values:
        raise ValueError(f"Cannot compute mean percentage for empty values: {context}")
    return float(np.mean(values))


def mean_optional(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def print_hit10_group_trend_table(results: List[Dict], title: str) -> None:
    log(f"\n{'=' * 140}")
    log(title)
    log("=" * 140)
    col_w = 12
    group_names = INJECTION_DEPTH_GROUP_ORDER
    group_labels = [INJECTION_DEPTH_GROUP_SHORT_DISPLAY[group_name] for group_name in group_names]
    transition_labels = [
        f"{group_labels[index]}->{group_labels[index + 1]}Δ"
        for index in range(len(group_labels) - 1)
    ]
    header = f"{'检索器':<12}"
    for group_label in group_labels:
        header += f" {f'N_{group_label}':>{col_w}}"
    for group_label in group_labels:
        header += f" {f'H@10_{group_label}':>{col_w}}"
    for transition_label in transition_labels:
        header += f" {transition_label:>{col_w}}"
    header += f" {'均值Δ':>{col_w}}"
    log(header)
    log("-" * 140)

    transition_delta_values = [[] for _ in range(len(group_names) - 1)]
    mean_values = []
    for result in results:
        group_metrics = result["metrics_by_injection_depth_group"]
        for group_name in group_names:
            if group_name not in group_metrics:
                raise KeyError(f"{result['retriever']} missing injection depth group {group_name}")
        h10_values = []
        row = f"{result['retriever']:<12}"
        for group_name in group_names:
            group_data = group_metrics[group_name]
            row += f" {group_data['num_queries']:>{col_w}}"
            h10_value = (
                group_data["metrics"]["H@10"]
                if group_data["num_queries"] > 0
                else None
            )
            h10_values.append(h10_value)
        for h10_value in h10_values:
            row += f" {format_metric(h10_value):>{col_w}}"

        valid_pct_values = []
        transition_deltas = []
        for index in range(len(group_names) - 1):
            transition_delta = diff_value(h10_values[index], h10_values[index + 1])
            transition_deltas.append(transition_delta)
            if transition_delta is not None:
                transition_delta_values[index].append(transition_delta)
                valid_pct_values.append(transition_delta)
        two_step_mean = mean_required(valid_pct_values, f"{result['retriever']} adjacent H@10 differences")
        mean_values.append(two_step_mean)

        for transition_delta in transition_deltas:
            row += f" {format_signed_metric(transition_delta):>{col_w}}"
        row += f" {format_signed_metric(two_step_mean):>{col_w}}"
        log(row)

    mean_row = f"{'MEAN':<12}"
    for _ in group_names:
        mean_row += f" {'':>{col_w}}"
    for _ in group_names:
        mean_row += f" {'':>{col_w}}"
    for index in range(len(group_names) - 1):
        mean_row += f" {format_signed_metric(mean_optional(transition_delta_values[index])):>{col_w}}"
    mean_row += f" {format_signed_metric(mean_optional(mean_values)):>{col_w}}"
    log(mean_row)
    log("-" * 140)


def print_hit10_correct_vs_noisy_group_table(
    correct_results: List[Dict],
    noisy_results: List[Dict],
    differences: List[Dict],
) -> None:
    del differences
    log(f"\n{'=' * 180}")
    log("CORRECT vs NOISY H@10 注入深度分组对比表（NOISY - CORRECT）")
    log("=" * 180)
    col_w = 10
    header = f"{'检索器':<12}"
    for group_name in INJECTION_DEPTH_GROUP_ORDER:
        group_label = INJECTION_DEPTH_GROUP_SHORT_DISPLAY[group_name]
        header += (
            f" {f'N_{group_label}':>{col_w}}"
            f" {f'{group_label}_C':>{col_w}}"
            f" {f'{group_label}_N':>{col_w}}"
            f" {f'{group_label}_Δ':>{col_w}}"
        )
    log(header)
    log("-" * 180)

    correct_by_retriever = {result["retriever"]: result for result in correct_results}
    noisy_by_retriever = {result["retriever"]: result for result in noisy_results}
    group_delta_values = {group_name: [] for group_name in INJECTION_DEPTH_GROUP_ORDER}

    for retriever in correct_by_retriever:
        if retriever not in noisy_by_retriever:
            raise KeyError(f"Missing noisy result for retriever={retriever}")
        correct_group_metrics = correct_by_retriever[retriever]["metrics_by_injection_depth_group"]
        noisy_group_metrics = noisy_by_retriever[retriever]["metrics_by_injection_depth_group"]
        row = f"{retriever:<12}"
        for group_name in INJECTION_DEPTH_GROUP_ORDER:
            if group_name not in correct_group_metrics or group_name not in noisy_group_metrics:
                raise KeyError(f"{retriever} missing injection depth group {group_name}")
            correct_group = correct_group_metrics[group_name]
            noisy_group = noisy_group_metrics[group_name]
            if correct_group["num_queries"] != noisy_group["num_queries"]:
                raise ValueError(
                    f"H@10 group query count mismatch for {retriever} {group_name}: "
                    f"{correct_group['num_queries']} vs {noisy_group['num_queries']}"
                )
            if correct_group["num_queries"] == 0:
                correct_value = None
                noisy_value = None
            else:
                correct_value = correct_group["metrics"]["H@10"]
                noisy_value = noisy_group["metrics"]["H@10"]
            delta = diff_value(correct_value, noisy_value)
            if delta is not None:
                group_delta_values[group_name].append(delta)
            row += (
                f" {correct_group['num_queries']:>{col_w}}"
                f" {format_metric(correct_value):>{col_w}}"
                f" {format_metric(noisy_value):>{col_w}}"
                f" {format_signed_metric(delta):>{col_w}}"
            )
        log(row)

    mean_row = f"{'MEAN':<12}"
    for group_name in INJECTION_DEPTH_GROUP_ORDER:
        mean_delta = mean_optional(group_delta_values[group_name])
        mean_row += (
            f" {'':>{col_w}}"
            f" {'':>{col_w}}"
            f" {'':>{col_w}}"
            f" {format_signed_metric(mean_delta):>{col_w}}"
        )
    log(mean_row)
    log("-" * 180)


def print_exact_injection_depth_table(
    results: List[Dict],
    title: str,
    depth_metrics_key: str,
    overall_metrics_key: str,
    injection_depths: List[int],
    signed: bool = False,
) -> None:
    log(f"\n{'=' * 140}")
    log(title)
    log("=" * 140)
    metrics_to_show = ["P@1", "P@3", "P@5", "P@10", "N@10", "MR@10", "H@10"]
    col_w = 12
    header = f"{'检索器':<12} {'N':>8}"
    for metric_name in metrics_to_show:
        for depth in injection_depths:
            header += f" {f'{metric_name}_D{depth}':>{col_w}}"
        header += f" {f'{metric_name}_ALL':>{col_w}}"
    log(header)
    log("-" * 140)

    for result in results:
        row = f"{result['retriever']:<12} {result['num_queries']:>8}"
        depth_metrics = result[depth_metrics_key]
        overall_metrics = result[overall_metrics_key]
        for metric_name in metrics_to_show:
            for depth in injection_depths:
                depth_key = str(depth)
                if depth_key not in depth_metrics:
                    raise KeyError(
                        f"{result['retriever']} missing exact injection depth {depth_key} in {depth_metrics_key}"
                    )
                value = depth_metrics[depth_key]["metrics"][metric_name]
                if signed and value > 0:
                    row += f" {'+' + format(value, '.4f'):>{col_w}}"
                else:
                    row += f" {value:>{col_w}.4f}"
            overall_value = overall_metrics[metric_name]
            if signed and overall_value > 0:
                row += f" {'+' + format(overall_value, '.4f'):>{col_w}}"
            else:
                row += f" {overall_value:>{col_w}.4f}"
        log(row)
    log("-" * 140)


def print_hit10_exact_injection_depth_table(
    correct_results: List[Dict],
    noisy_results: List[Dict],
    differences: List[Dict],
    injection_depths: List[int],
) -> None:
    log(f"\n{'=' * 140}")
    log("H@10 注入精确深度对比表")
    log("=" * 140)
    col_w = 12
    header = f"{'检索器':<12}"
    for depth in injection_depths:
        header += f" {f'D{depth}_C':>{col_w}} {f'D{depth}_N':>{col_w}} {f'D{depth}_Δ':>{col_w}}"
    header += f" {'ALL_C':>{col_w}} {'ALL_N':>{col_w}} {'ALL_Δ':>{col_w}}"
    log(header)
    log("-" * 140)

    correct_by_retriever = {result["retriever"]: result for result in correct_results}
    noisy_by_retriever = {result["retriever"]: result for result in noisy_results}
    diff_by_retriever = {result["retriever"]: result for result in differences}
    for retriever in sorted(correct_by_retriever):
        if retriever not in noisy_by_retriever or retriever not in diff_by_retriever:
            raise KeyError(f"Missing paired H@10 table result for retriever={retriever}")
        correct_result = correct_by_retriever[retriever]
        noisy_result = noisy_by_retriever[retriever]
        diff_result = diff_by_retriever[retriever]
        row = f"{retriever:<12}"
        for depth in injection_depths:
            depth_key = str(depth)
            correct_depth = correct_result["metrics_by_exact_injection_depth"]
            noisy_depth = noisy_result["metrics_by_exact_injection_depth"]
            diff_depth = diff_result["metrics_by_exact_injection_depth_noisy_minus_correct"]
            for source_name, source in (
                ("correct", correct_depth),
                ("noisy", noisy_depth),
                ("diff", diff_depth),
            ):
                if depth_key not in source:
                    raise KeyError(f"{retriever} missing {source_name} H@10 value for injection depth {depth_key}")
            correct_value = correct_depth[depth_key]["metrics"]["H@10"]
            noisy_value = noisy_depth[depth_key]["metrics"]["H@10"]
            diff_value = diff_depth[depth_key]["metrics"]["H@10"]
            diff_text = f"+{diff_value:.4f}" if diff_value > 0 else f"{diff_value:.4f}"
            row += f" {correct_value:>{col_w}.4f} {noisy_value:>{col_w}.4f} {diff_text:>{col_w}}"
        overall_correct = correct_result["metrics"]["H@10"]
        overall_noisy = noisy_result["metrics"]["H@10"]
        overall_diff = diff_result["metrics_noisy_minus_correct"]["H@10"]
        overall_diff_text = f"+{overall_diff:.4f}" if overall_diff > 0 else f"{overall_diff:.4f}"
        row += f" {overall_correct:>{col_w}.4f} {overall_noisy:>{col_w}.4f} {overall_diff_text:>{col_w}}"
        log(row)
    log("-" * 140)


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(key): sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def evaluate_retriever_pair(
    category_name: str,
    category_config: Dict,
    retriever_name: str,
    pairs: List[Dict],
    k_values: List[int],
) -> Tuple[Dict, Dict]:
    if retriever_name in DENSE_RETRIEVERS:
        return evaluate_dense_pair(category_config, retriever_name, pairs, k_values)
    if retriever_name == "bm25":
        return evaluate_bm25_pair(category_config, pairs, k_values)
    if retriever_name == "splade":
        return evaluate_splade_pair(category_config, pairs, k_values)
    if retriever_name == "colbertv2":
        return evaluate_colbertv2_pair(category_name, category_config, pairs, k_values)
    raise ValueError(f"Unsupported retriever: {retriever_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate syntax-depth noisy queries against 08-style caches")
    parser.add_argument(
        "--retrievers",
        nargs="+",
        choices=RETRIEVERS,
        default=RETRIEVERS,
        help="Retrievers to evaluate",
    )
    return parser.parse_args()


def run_category_eval(category_name: str) -> None:
    args = parse_args()
    category_config = get_category_config(category_name)
    for required_key in ("retriever_cache_dir", "query_cache_dir", "corpus_file"):
        if required_key not in category_config:
            raise KeyError(f"Category config for {category_name} missing required key: {required_key}")

    global_paths = get_global_paths()
    for required_key in ("stage6_query", "inject_noisy"):
        if required_key not in global_paths:
            raise KeyError(f"Global paths missing required key: {required_key}")

    syntax_depth_query_file = os.path.join(global_paths["stage6_query"], category_name, "query_by_syntax_depth.json")
    noisy_query_file = os.path.join(global_paths["inject_noisy"], category_name, "noisy_query.json")
    output_root = os.path.join(str(Path(global_paths["inject_noisy"]).parent), "09_noisy_retrieval", category_name)
    os.makedirs(output_root, exist_ok=True)

    log("=" * 80)
    log(f"Syntax-depth noisy query evaluation - {category_name}")
    log("=" * 80)
    log(f"06 syntax-depth query file: {syntax_depth_query_file}")
    log(f"07 noisy query file: {noisy_query_file}")
    log(f"Query cache dir: {category_config['query_cache_dir']}")
    log(f"Retriever cache dir: {category_config['retriever_cache_dir']}")
    log(f"Retrievers: {', '.join(args.retrievers)}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    k_values = [1, 3, 5, 10]
    pairs = load_noisy_pairs(category_name, noisy_query_file, syntax_depth_query_file)
    injection_depths = get_injection_depths(pairs)
    log_injection_depth_distribution(pairs)

    correct_results = []
    noisy_results = []
    differences = []
    for retriever_name in args.retrievers:
        correct_result, noisy_result = evaluate_retriever_pair(
            category_name,
            category_config,
            retriever_name,
            pairs,
            k_values,
        )
        correct_results.append(correct_result)
        noisy_results.append(noisy_result)
        differences.append(metric_diff(correct_result, noisy_result))

    filtered_correct_results = []
    filtered_noisy_results = []
    filtered_differences = []
    final_statistics_filters = []
    for correct_result, noisy_result in zip(correct_results, noisy_results):
        filtered_correct_result, filtered_noisy_result, filter_summary = filter_retriever_pair_results(
            correct_result,
            noisy_result,
            k_values,
        )
        filtered_correct_results.append(filtered_correct_result)
        filtered_noisy_results.append(filtered_noisy_result)
        filtered_differences.append(metric_diff(filtered_correct_result, filtered_noisy_result))
        final_statistics_filters.append(filter_summary)
        log(
            f"{correct_result['retriever']} 最终统计剔除 noisy 优于 clean 的样本 "
            f"{filter_summary['excluded_count']} 条，保留 {filter_summary['kept_count']} 条"
        )

    print_hit10_injection_depth_group_table(filtered_correct_results, filtered_noisy_results, filtered_differences)

    output_file = os.path.join(output_root, "syntax_depth_correct_vs_noisy_results.json")
    results_to_save = {
        "timestamp": datetime.now().isoformat(),
        "category": category_name,
        "query_category": SYNTAX_DEPTH_QUERY_CATEGORY,
        "noisy_injection_source": NOISY_INJECTION_SOURCE,
        "correct_query_source_field": CORRECT_QUERY_SOURCE_FIELD,
        "syntax_depth_query_file": syntax_depth_query_file,
        "noisy_query_file": noisy_query_file,
        "query_cache_dir": category_config["query_cache_dir"],
        "retriever_cache_dir": category_config["retriever_cache_dir"],
        "retrievers": args.retrievers,
        "k_values": k_values,
        "num_noisy_pairs_loaded": len(pairs),
        "injection_depths": injection_depths,
        "depth_groups": [
            {
                "name": name,
                "low": low,
                "high": high,
                "display": display,
            }
            for name, low, high, display in DEPTH_GROUPS
        ],
        "injection_depth_groups": [
            {
                "name": name,
                "low": low,
                "high": high,
                "display": display,
            }
            for name, low, high, display in INJECTION_DEPTH_GROUPS
        ],
        "raw_correct_results": correct_results,
        "raw_noisy_results": noisy_results,
        "raw_differences": differences,
        "correct_results": filtered_correct_results,
        "noisy_results": filtered_noisy_results,
        "differences": filtered_differences,
        "final_statistics_filters": final_statistics_filters,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(results_to_save), f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存到: {output_file}")
    log("评估完成")
