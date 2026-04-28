#!/usr/bin/env python3
"""
Generate ColBERTv2 result query caches for Baby_Products.

Outputs:
- query_cache_Baby_Products/acl_correct_query/colbertv2__acl_correct_cache.pkl
- query_cache_Baby_Products/ccomp_correct_query/colbertv2__ccomp_correct_cache.pkl
"""

import importlib.util
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["HF_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_CACHE"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

CURRENT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(CURRENT_DIR))

from config import get_category_config


CATEGORY_NAME = "Baby_Products"
MODEL_NAME = "colbert-ir/colbertv2.0"
EXPERIMENT_NAME = "colbertv2_index"
TOP_K = 100
QUERY_SPECS = (
    ("acl", "acl_query", "acl_correct"),
    ("ccomp", "ccomp_query", "ccomp_correct"),
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_colbert_build_module():
    module_path = CURRENT_DIR / "08_build_colbertv2_index_Baby_Products.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Required ColBERTv2 build helper script not found: {module_path}")

    spec = importlib.util.spec_from_file_location("build_colbertv2_baby_products", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load import spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_colbertv2_output_root(category_config: Dict[str, str]) -> str:
    helper = load_colbert_build_module()
    all_metadata = helper.load_raw_metadata(category_config["raw_corpus_file"])
    documents, _ = helper.build_documents(all_metadata)
    doc_hash = helper.compute_document_hash(documents)
    output_root = os.path.join(category_config["retriever_cache_dir"], f"colbertv2_{doc_hash}")
    manifest_path = os.path.join(output_root, "build_manifest.json")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Required ColBERTv2 build manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_hash = manifest.get("document_hash")
    if manifest_hash != doc_hash:
        raise ValueError(f"ColBERTv2 manifest hash mismatch: manifest={manifest_hash}, current={doc_hash}")

    return output_root


def configure_colbert_runtime() -> None:
    helper = load_colbert_build_module()
    helper.select_cuda_toolkit_for_colbert_extension_build()
    helper.configure_host_compiler_for_colbert_extension_build()
    helper.validate_cuda_toolkit_for_colbert()
    helper.configure_cuda_env_for_colbert_extension_build()
    helper.preflight_colbert_cuda_extension_build()


def load_query_items(query_file: str) -> List[Dict]:
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"Required query file not found: {query_file}")

    with open(query_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"query.json must contain a list, got {type(data)}")
    if not data:
        raise ValueError(f"query.json is empty: {query_file}")

    return data


def collect_unique_queries(items: List[Dict], query_field: str) -> List[str]:
    unique_queries = []
    seen = set()

    for index, item in enumerate(items):
        user_id = item["user_id"]
        asin = item["asin"]
        query_data = item[query_field]
        if not isinstance(query_data, dict):
            raise TypeError(f"{query_field} must be a dict at item index {index}, user={user_id}, asin={asin}")

        query_text = query_data["query"]
        if not isinstance(query_text, str):
            raise TypeError(f"{query_field}.query must be a string at item index {index}, user={user_id}, asin={asin}")
        if not query_text:
            raise ValueError(f"{query_field}.query is empty at item index {index}, user={user_id}, asin={asin}")

        if query_text not in seen:
            seen.add(query_text)
            unique_queries.append(query_text)

    if not unique_queries:
        raise ValueError(f"No valid queries collected for field: {query_field}")

    return unique_queries


def load_doc_ids(output_root: str) -> List[str]:
    doc_ids_path = os.path.join(output_root, "doc_ids.pkl")
    if not os.path.exists(doc_ids_path):
        raise FileNotFoundError(f"Required ColBERTv2 doc id mapping not found: {doc_ids_path}")

    with open(doc_ids_path, "rb") as f:
        doc_ids = pickle.load(f)

    if not isinstance(doc_ids, list):
        raise TypeError(f"doc_ids.pkl must contain a list, got {type(doc_ids)}")
    if not doc_ids:
        raise ValueError(f"doc_ids.pkl is empty: {doc_ids_path}")

    return doc_ids


def build_searcher(output_root: str, doc_ids: List[str]):
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Searcher

    collection = [f"pid {pid} asin {asin}" for pid, asin in enumerate(doc_ids)]
    with Run().context(RunConfig(experiment=EXPERIMENT_NAME, root=output_root)):
        config = ColBERTConfig(root=output_root)
        return Searcher(
            index=EXPERIMENT_NAME,
            checkpoint=MODEL_NAME,
            collection=collection,
            config=config,
        )


def convert_ranking_to_cache(
    ranking,
    qid_to_query: Dict[int, str],
    doc_ids: List[str],
) -> Dict[str, List[Tuple[str, float]]]:
    if not hasattr(ranking, "data"):
        raise TypeError(f"ColBERT search_all returned object without data attribute: {type(ranking)}")

    cache = {}
    for qid, rows in ranking.data.items():
        query_text = qid_to_query[qid]
        converted_rows = []

        for row in rows:
            if len(row) != 3:
                raise ValueError(f"Unexpected ColBERT ranking row for qid={qid}: {row}")

            pid, _, score = row
            pid_int = int(pid)
            if pid_int < 0 or pid_int >= len(doc_ids):
                raise IndexError(f"ColBERT pid {pid_int} is outside doc_ids range 0..{len(doc_ids)-1}")

            converted_rows.append((doc_ids[pid_int], float(score)))

        cache[query_text] = converted_rows

    if len(cache) != len(qid_to_query):
        raise RuntimeError(f"Cache size mismatch: expected {len(qid_to_query)}, got {len(cache)}")

    return cache


def save_cache(cache: Dict[str, List[Tuple[str, float]]], cache_dir: str, cache_name: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"colbertv2__{cache_name}_cache.pkl")
    tmp_path = f"{cache_path}.tmp"

    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.replace(tmp_path, cache_path)
    return cache_path


def generate_cache_for_query_set(
    searcher,
    doc_ids: List[str],
    query_texts: List[str],
    cache_dir: str,
    cache_name: str,
) -> Dict[str, object]:
    qid_to_query = {qid: query_text for qid, query_text in enumerate(query_texts)}

    log(f"开始生成 {cache_name} ColBERTv2 查询缓存: queries={len(query_texts)}, top_k={TOP_K}")
    start_time = time.time()
    ranking = searcher.search_all(qid_to_query, k=TOP_K)
    cache = convert_ranking_to_cache(ranking, qid_to_query, doc_ids)
    cache_path = save_cache(cache, cache_dir, cache_name)
    elapsed = time.time() - start_time

    file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    log(f"完成 {cache_name}: cached={len(cache)}, file={cache_path}, size={file_size_mb:.2f} MB, elapsed={elapsed:.1f}s")

    return {
        "cache_name": cache_name,
        "cache_path": cache_path,
        "query_count": len(cache),
        "file_size_mb": file_size_mb,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    log("=" * 80)
    log("BUILD BABY_PRODUCTS COLBERTV2 QUERY CACHE - STARTING")
    log("=" * 80)

    category_config = get_category_config(CATEGORY_NAME)
    query_file = category_config["query_file"]
    query_cache_root = category_config["query_cache_dir"]
    output_root = resolve_colbertv2_output_root(category_config)
    index_dir = os.path.join(output_root, EXPERIMENT_NAME, "indexes", EXPERIMENT_NAME)

    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"Required ColBERTv2 index directory not found: {index_dir}")

    log(f"Category: {CATEGORY_NAME}")
    log(f"Query file: {query_file}")
    log(f"Index dir: {index_dir}")
    log(f"Query cache root: {query_cache_root}")

    configure_colbert_runtime()

    items = load_query_items(query_file)
    doc_ids = load_doc_ids(output_root)
    searcher = build_searcher(output_root, doc_ids)

    summaries = []
    for query_category, query_field, cache_name in QUERY_SPECS:
        query_texts = collect_unique_queries(items, query_field)
        cache_dir = os.path.join(query_cache_root, f"{query_category}_correct_query")
        summary = generate_cache_for_query_set(searcher, doc_ids, query_texts, cache_dir, cache_name)
        summaries.append(summary)

    log("=" * 80)
    log("BUILD BABY_PRODUCTS COLBERTV2 QUERY CACHE - COMPLETE")
    for summary in summaries:
        log(
            f"{summary['cache_name']}: queries={summary['query_count']}, "
            f"size={summary['file_size_mb']:.2f} MB, elapsed={summary['elapsed_seconds']:.1f}s"
        )
    log("当前任务已完成，请做下一个任务的指示。")


if __name__ == "__main__":
    main()
