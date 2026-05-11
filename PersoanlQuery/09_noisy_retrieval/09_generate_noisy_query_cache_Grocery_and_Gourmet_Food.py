#!/usr/bin/env python3
"""
生成并预存储检索器的 noisy 查询缓存

这个脚本将：
1. 从 07_inject_noisy 的 combined noisy_query.json 加载 ACL 和 CCOMP noisy queries
2. 为每个检索器编码每个 noisy 查询
3. 保存缓存到磁盘以加速后续评估

使用方法：
    python3 09_generate_noisy_query_cache_Grocery_and_Gourmet_Food.py > /workspace/logs/09_generate_noisy_query_cache_Grocery_and_Gourmet_Food.log 2> /workspace/logs/09_generate_noisy_query_cache_Grocery_and_Gourmet_Food.err
"""

import os
os.environ["HF_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_CACHE"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import sys
import importlib.util
import json
import pickle
import io
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

current_dir = Path(__file__).parent.resolve()
retrieval_root = current_dir.parent / "08_retrieval"
personquery_root = retrieval_root.parent

sys.path.insert(0, str(retrieval_root))
sys.path.insert(0, str(personquery_root))

from utils.retrievers import (
    E5Retriever, BGERetriever,
    STARRetriever, MiniLMRetriever, BM25,
    ANCERetriever, SPLADERetriever,
    select_cuda_toolkit_for_colbert_extension_build,
    configure_host_compiler_for_colbert_extension_build,
    validate_cuda_toolkit_for_colbert,
    configure_cuda_env_for_colbert_extension_build,
    preflight_colbert_cuda_extension_build,
)
from config import get_category_config, get_global_paths

# ============ 配置加载 ============
CATEGORY_NAME = "Grocery_and_Gourmet_Food"
CAT_CONFIG = get_category_config(CATEGORY_NAME)
GLOBAL_PATHS = get_global_paths()

# Noisy query 文件（07 生成的 combined 文件）
NOISY_QUERY_FILE = f"{GLOBAL_PATHS['inject_noisy']}/{CATEGORY_NAME}/noisy_query.json"

# 缓存目录 - 使用与 08 相同的目录结构
CACHE_DIR = CAT_CONFIG['query_cache_dir']
BM25_RETRIEVER_CACHE_DIR = CAT_CONFIG['retriever_cache_dir']

AVAILABLE_RETRIEVERS = {
    # 'GRITLM': GritLMRetriever,
    'BGE': BGERetriever,
    'E5': E5Retriever,
    'MiniLM': MiniLMRetriever,
    'STAR': STARRetriever,
    'ANCE': ANCERetriever,
    'ColBERTv2': None,
    'SPLADE': SPLADERetriever,
    'BM25': None,
}

COLBERTV2_MODEL_NAME = "colbert-ir/colbertv2.0"
COLBERTV2_QUERY_BATCH_SIZE = 128


def log_with_timestamp(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_colbertv2_build_module():
    module_path = retrieval_root / "08_build_retriever_indices_Grocery_and_Gourmet_Food.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Required ColBERTv2 build module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("build_retriever_indices_grocery_and_gourmet_food", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ColBERTv2 build module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    return module


def configure_colbertv2_runtime() -> None:
    # 为每个域的 09 作业隔离 torch extension build 目录，避免并发作业争抢同一个 lock 文件。
    os.environ["TORCH_EXTENSIONS_DIR"] = f"/home/wlia0047/ar57_scratch/wenyu/tmp/torch_extensions_09_{CATEGORY_NAME}"
    select_cuda_toolkit_for_colbert_extension_build()
    configure_host_compiler_for_colbert_extension_build()
    validate_cuda_toolkit_for_colbert()
    configure_cuda_env_for_colbert_extension_build()
    preflight_colbert_cuda_extension_build()


def collect_unique_query_texts(queries: List[Dict], mode: str) -> List[str]:
    unique_queries = []
    seen = set()

    for index, item in enumerate(queries):
        query_text = item.get('query')
        if not isinstance(query_text, str):
            raise TypeError(f"{mode} query must be a string at index {index}, got {type(query_text).__name__}")
        if not query_text:
            raise ValueError(f"{mode} query is empty at index {index}")
        if query_text not in seen:
            seen.add(query_text)
            unique_queries.append(query_text)

    if not unique_queries:
        raise ValueError(f"No valid queries collected for ColBERTv2 mode: {mode}")

    return unique_queries


def load_colbertv2_checkpoint_for_query_encoding():
    if not torch.cuda.is_available():
        raise RuntimeError("ColBERTv2 query embedding cache generation requires CUDA")

    configure_colbertv2_runtime()

    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint

    config = ColBERTConfig(
        checkpoint=COLBERTV2_MODEL_NAME,
        query_maxlen=32,
    )
    checkpoint = Checkpoint(COLBERTV2_MODEL_NAME, colbert_config=config, verbose=1)
    return checkpoint.cuda()


def encode_colbertv2_query_texts(checkpoint, query_texts: List[str]) -> Dict[str, np.ndarray]:
    if not query_texts:
        raise ValueError("ColBERTv2 query text list is empty")

    embeddings = checkpoint.queryFromText(
        query_texts,
        bsize=COLBERTV2_QUERY_BATCH_SIZE,
        to_cpu=True,
    )
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(f"ColBERTv2 query encoder returned {type(embeddings).__name__}, expected torch.Tensor")
    if embeddings.ndim != 3:
        raise ValueError(f"ColBERTv2 query embeddings must be 3D, got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != len(query_texts):
        raise RuntimeError(
            f"ColBERTv2 query embedding count mismatch: expected {len(query_texts)}, got {embeddings.shape[0]}"
        )

    return {
        query_text: np.asarray(embeddings[index].detach().cpu().numpy(), dtype=np.float32)
        for index, query_text in enumerate(query_texts)
    }


def save_colbertv2_query_embedding_cache(cache: Dict[str, Dict[str, np.ndarray]], mode: str) -> str:
    subdir = os.path.join(CACHE_DIR, f"{mode}_query")
    os.makedirs(subdir, exist_ok=True)
    cache_path = os.path.join(subdir, f"colbertv2__{mode}_cache.pkl")
    tmp_path = f"{cache_path}.tmp"

    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.replace(tmp_path, cache_path)

    file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    total_cached = sum(len(user_cache) for user_cache in cache.values())
    log_with_timestamp(f"  ✓ ColBERTv2 query embedding 缓存已保存: {cache_path}")
    log_with_timestamp(f"    - 用户数: {len(cache)}")
    log_with_timestamp(f"    - 查询数: {total_cached}")
    log_with_timestamp(f"    - 文件大小: {file_size_mb:.2f} MB")
    return cache_path


def generate_colbertv2_query_embedding_cache_for_mode(
    checkpoint,
    queries: List[Dict],
    queries_by_user: Dict[str, List[Dict]],
    mode: str,
) -> Dict[str, object]:
    query_texts = collect_unique_query_texts(queries, mode)
    log_with_timestamp(
        f"  开始生成 ColBERTv2 query embedding 缓存 ({mode}): "
        f"unique_queries={len(query_texts)}, batch_size={COLBERTV2_QUERY_BATCH_SIZE}"
    )

    start_time = time.time()
    encoded_by_query = encode_colbertv2_query_texts(checkpoint, query_texts)

    result_cache: Dict[str, Dict[str, np.ndarray]] = {uid: {} for uid in queries_by_user.keys()}
    for uid, user_queries in queries_by_user.items():
        for item in user_queries:
            query_text = item.get('query')
            if query_text not in encoded_by_query:
                raise KeyError(f"ColBERTv2 encoded query missing for user={uid}, mode={mode}, query={query_text}")
            result_cache[uid][query_text] = encoded_by_query[query_text]

    cache_path = save_colbertv2_query_embedding_cache(result_cache, mode)
    elapsed = time.time() - start_time
    total_cached = sum(len(user_cache) for user_cache in result_cache.values())

    return {
        'mode': mode,
        'cache_path': cache_path,
        'query_count': total_cached,
        'unique_query_count': len(query_texts),
        'elapsed_seconds': elapsed,
    }


def generate_colbertv2_cache_from_query_types(query_types: List[Tuple[str, List[Dict], Dict[str, List[Dict]], str]]) -> Dict[str, object]:
    checkpoint = load_colbertv2_checkpoint_for_query_encoding()

    summaries = []
    total_cached = 0
    try:
        for query_type, queries, queries_by_user, mode in query_types:
            if not queries:
                log_with_timestamp(f"  (无 {query_type} {mode} 查询，跳过)")
                continue

            summary = generate_colbertv2_query_embedding_cache_for_mode(
                checkpoint,
                queries,
                queries_by_user,
                mode,
            )
            summaries.append(summary)
            total_cached += summary['query_count']
            log_with_timestamp(
                f"  ✓ {query_type} {mode} ColBERTv2 query embedding 缓存: "
                f"{summary['query_count']} 条, unique={summary['unique_query_count']}, "
                f"耗时 {summary['elapsed_seconds']:.1f}s"
            )
    finally:
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return {
        'total_cached': total_cached,
        'summaries': summaries,
    }


def load_noisy_queries() -> Tuple[List[Dict], List[Dict]]:
    """从 combined noisy_query.json 加载 ACL 和 CCOMP noisy queries

    Returns:
        (acl_noisy_queries, ccomp_noisy_queries)
    """
    acl_noisy = []
    ccomp_noisy = []

    if not os.path.exists(NOISY_QUERY_FILE):
        log_with_timestamp(f"⚠️  noisy query 文件不存在: {NOISY_QUERY_FILE}")
        return [], []

    try:
        with open(NOISY_QUERY_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if not content:
            log_with_timestamp(f"⚠️  noisy query 文件为空: {NOISY_QUERY_FILE}")
            return [], []

        # 解析 JSON（支持 JSON Lines、pretty-printed 或 JSON array）
        if content.startswith('['):
            data = json.loads(content)
        else:
            data = []
            depth = 0
            start = -1
            for i, c in enumerate(content):
                if c == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0 and start >= 0:
                        try:
                            data.append(json.loads(content[start:i+1]))
                        except:
                            pass
                        start = -1

        for item in data:
            query_cat = item.get('query_category', '')
            noisy_text = item.get('noisy_query', '')
            if not noisy_text:
                continue

            entry = {
                'user_id': item.get('user_id', ''),
                'asin': item.get('asin', ''),
                'is_ground_truth': True,
                'query': noisy_text,
            }

            if query_cat == 'acl':
                entry['acl'] = item.get('level', 0)
                acl_noisy.append(entry)
            elif query_cat == 'ccomp':
                entry['ccomp'] = item.get('level', 0)
                ccomp_noisy.append(entry)

        log_with_timestamp(f"✓ 从 {NOISY_QUERY_FILE} 加载了 {len(acl_noisy)} 条 ACL noisy, {len(ccomp_noisy)} 条 CCOMP noisy")
    except Exception as e:
        log_with_timestamp(f"⚠️  读取 noisy query 文件失败: {e}")
        return [], []

    return acl_noisy, ccomp_noisy


def _build_queries_by_user(queries: List[Dict]) -> Dict[str, List[Dict]]:
    """将查询列表按用户ID分组"""
    by_user = defaultdict(list)
    for q in queries:
        uid = q.get('user_id', '')
        if uid:
            by_user[uid].append(q)
    return dict(by_user)


def get_retriever(retriever_name: str):
    """获取检索器实例"""
    if retriever_name == 'BM25':
        return BM25()
    retriever_class = AVAILABLE_RETRIEVERS.get(retriever_name)
    if retriever_class is None:
        raise ValueError(f"Unknown retriever: {retriever_name}")
    return retriever_class()


class _StdoutToStderr:
    """上下文管理器，将 stdout 重定向到 stderr"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = sys.stderr
        return self
    def __exit__(self, *args):
        sys.stdout = self._original_stdout


def initialize_cache_dir():
    """初始化缓存目录"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "acl_noisy_query"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "ccomp_noisy_query"), exist_ok=True)
    log_with_timestamp(f"✓ 缓存目录: {CACHE_DIR}")


def get_cache_file_path(retriever_name: str, user_id: str, mode: str) -> str:
    """获取缓存文件路径"""
    subdir = os.path.join(CACHE_DIR, f"{mode}_query")
    filename = f"{retriever_name}_{user_id}.pkl"
    return os.path.join(subdir, filename)


def _encode_and_save_cache(
    retriever_name: str,
    queries: List[Dict],
    by_user: Dict[str, List[Dict]],
    mode: str,
) -> int:
    """为检索器编码并保存查询缓存"""
    if retriever_name == 'BM25':
        # BM25 使用不同的处理方式
        return _encode_and_save_bm25_cache(queries, by_user, mode)

    # SPLADE 使用不同的缓存格式
    if retriever_name == 'SPLADE':
        return _encode_and_save_splade_cache(queries, by_user, mode)

    retriever_class = AVAILABLE_RETRIEVERS[retriever_name]
    log_with_timestamp(f"  初始化检索器 {retriever_name}...")
    with _StdoutToStderr():
        retriever = retriever_class()
    log_with_timestamp(f"  ✓ 检索器初始化完成，模型已加载")

    cache = {}
    failed_count = 0

    for user_id, user_queries in by_user.items():
        user_cache = []
        for q in user_queries:
            try:
                text = q.get('query', '')
                if not text:
                    continue
                embedding = retriever.encode_query(text)

                # Sparse retrievers (SPLADE) return dict, keep as-is
                if not isinstance(embedding, (np.ndarray, dict)):
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    else:
                        embedding = np.array(embedding)

                user_cache.append({
                    'query': text,
                    'vector': embedding,
                    'user_id': user_id,
                    'asin': q.get('asin', ''),
                    'level': q.get('acl') or q.get('ccomp', 0),
                    'is_ground_truth': q.get('is_ground_truth', True),
                })
            except Exception as e:
                log_with_timestamp(f"      ❌ 编码失败 [{retriever_name}] 查询: {text[:40]}... 错误: {str(e)[:100]}")
                failed_count += 1
                import sys
                sys.exit(1)

        if user_cache:
            cache[user_id] = user_cache

    if cache:
        cache_file = get_cache_file_path(retriever_name, "", mode)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        log_with_timestamp(f"  ✓ 缓存已保存: {cache_file}")

    if failed_count > 0:
        log_with_timestamp(f"      ⚠️  共有 {failed_count} 个查询编码失败")

    return len(cache)


def _encode_and_save_splade_cache(
    queries: List[Dict],
    by_user: Dict[str, List[Dict]],
    mode: str,
) -> int:
    """为 SPLADE 编码并保存查询缓存（稀疏向量格式）"""
    from utils.retrievers import SPLADERetriever

    log_with_timestamp(f"  初始化检索器 SPLADE...")
    with _StdoutToStderr():
        retriever = SPLADERetriever()
    log_with_timestamp(f"  ✓ SPLADE 检索器初始化完成，模型已加载")

    # SPLADE 缓存格式：{user_id: {query_text: sparse_vec_dict}}
    cache = {}
    failed_count = 0

    for user_id, user_queries in by_user.items():
        user_cache = {}
        for q in user_queries:
            try:
                text = q.get('query', '')
                if not text:
                    continue
                # SPLADE 返回 Dict[str, float] (sparse vector)
                sparse_vec = retriever.encode_query(text)
                user_cache[text] = sparse_vec
            except Exception as e:
                log_with_timestamp(f"      ❌ SPLADE 编码失败: {text[:40]}... 错误: {str(e)[:100]}")
                failed_count += 1
                continue

        if user_cache:
            cache[user_id] = user_cache

    if cache:
        # SPLADE 使用特殊路径格式，与评估代码兼容
        cache_file = os.path.join(CACHE_DIR, f"{mode}_query", f"splade__{mode}_cache.pkl")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        log_with_timestamp(f"  ✓ SPLADE 缓存已保存: {cache_file}")

    if failed_count > 0:
        log_with_timestamp(f"      ⚠️  SPLADE 共有 {failed_count} 个查询编码失败")

    return len(cache)


def _encode_and_save_bm25_cache(
    queries: List[Dict],
    by_user: Dict[str, List[Dict]],
    mode: str,
) -> int:
    """为 BM25 编码并保存查询缓存"""
    log_with_timestamp(f"  初始化 BM25...")

    # 加载 BM25 检索器
    bm25_path = None
    for f in os.listdir(BM25_RETRIEVER_CACHE_DIR):
        if f.startswith('bm25_') and f.endswith('.pkl'):
            bm25_path = os.path.join(BM25_RETRIEVER_CACHE_DIR, f)
            break
    if bm25_path is None:
        log_with_timestamp(f"  ⚠️  BM25 retriever cache not found in {BM25_RETRIEVER_CACHE_DIR}")
        return 0

    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)
    log_with_timestamp(f"  ✓ BM25 加载完成")

    cache = {}
    failed_count = 0

    for index, q in enumerate(queries):
        try:
            text = q.get('query', '')
            if not text:
                raise ValueError(f"BM25 {mode} query is empty at index {index}")
            if text in cache:
                continue
            cache[text] = bm25.search(text, top_k=100)
        except Exception as e:
            log_with_timestamp(f"      ❌ BM25 搜索失败: {text[:40]}... 错误: {str(e)[:100]}")
            failed_count += 1
            import sys
            sys.exit(1)

    if cache:
        cache_file = os.path.join(CACHE_DIR, f"{mode}_query", f"bm25__{mode}_cache.pkl")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        log_with_timestamp(f"  ✓ BM25 缓存已保存: {cache_file}")

    if failed_count > 0:
        log_with_timestamp(f"      ⚠️  共有 {failed_count} 个查询编码失败")

    return len(cache)


def clear_noisy_cache() -> int:
    """删除旧的 noisy 查询缓存文件"""
    if not os.path.exists(CACHE_DIR):
        return 0

    deleted_count = 0
    for subdir in ["acl_noisy_query", "ccomp_noisy_query"]:
        subdir_path = os.path.join(CACHE_DIR, subdir)
        if os.path.exists(subdir_path):
            for root, _, files in os.walk(subdir_path):
                for name in files:
                    if name.endswith('.pkl') or name.endswith('.json'):
                        filepath = os.path.join(root, name)
                        try:
                            os.remove(filepath)
                            deleted_count += 1
                        except Exception as e:
                            log_with_timestamp(f"  ⚠️  删除失败: {filepath} - {e}")

    if deleted_count > 0:
        log_with_timestamp(f"✓ 已清理旧缓存: {deleted_count} 个文件")
    return deleted_count


def main():
    parser = argparse.ArgumentParser(description='生成 noisy 查询缓存')
    parser.add_argument('--retrievers', type=str, nargs='+',
                        choices=list(AVAILABLE_RETRIEVERS.keys()),
                        default=list(AVAILABLE_RETRIEVERS.keys()),
                        help='指定要处理的检索器')
    parser.add_argument('--clear', action='store_true',
                        help='清理旧缓存后再生成')
    args = parser.parse_args()

    retriever_names = args.retriever_names if hasattr(args, 'retriever_names') else args.retrievers
    clear_cache_before = args.clear

    log_with_timestamp("=" * 80)
    log_with_timestamp("🚀 生成 Noisy 查询缓存")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"类别: {CATEGORY_NAME}")
    log_with_timestamp(f"Noisy 文件: {NOISY_QUERY_FILE}")
    log_with_timestamp(f"缓存目录: {CACHE_DIR}")
    log_with_timestamp(f"检索器: {', '.join(retriever_names)}")
    log_with_timestamp(
        "HF 离线模式: "
        f"HF_HOME={os.environ.get('HF_HOME')}, "
        f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}, "
        f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}, "
        f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}"
    )
    log_with_timestamp("")

    # 加载 noisy queries
    acl_noisy, ccomp_noisy = load_noisy_queries()

    if not acl_noisy and not ccomp_noisy:
        log_with_timestamp("⚠️  没有加载到任何 noisy 查询")
        return

    # 按用户分组
    acl_noisy_by_user = _build_queries_by_user(acl_noisy)
    ccomp_noisy_by_user = _build_queries_by_user(ccomp_noisy)

    log_with_timestamp("")
    log_with_timestamp(f"📋 任务配置:")
    log_with_timestamp(f"  • ACL noisy 用户: {len(acl_noisy_by_user)} 个, 查询: {sum(len(v) for v in acl_noisy_by_user.values())} 条")
    log_with_timestamp(f"  • CCOMP noisy 用户: {len(ccomp_noisy_by_user)} 个, 查询: {sum(len(v) for v in ccomp_noisy_by_user.values())} 条")
    log_with_timestamp("")

    if clear_cache_before:
        clear_noisy_cache()
    initialize_cache_dir()

    start_time = time.time()
    total_cached = 0

    # 定义查询类型
    query_types = [
        ('ACL', acl_noisy, acl_noisy_by_user, 'acl_noisy'),
        ('CCOMP', ccomp_noisy, ccomp_noisy_by_user, 'ccomp_noisy'),
    ]

    for retriever_name in retriever_names:
        if retriever_name not in AVAILABLE_RETRIEVERS:
            log_with_timestamp(f"⚠️  检索器不存在: {retriever_name}")
            continue

        log_with_timestamp(f"\n{'='*80}")
        log_with_timestamp(f"正在处理检索器: {retriever_name}")
        log_with_timestamp(f"{'='*80}")

        if retriever_name == 'ColBERTv2':
            summary = generate_colbertv2_cache_from_query_types(query_types)
            total_cached += summary['total_cached']
            log_with_timestamp(f"✓ 检索器 {retriever_name} 处理完成: {summary['total_cached']} 条")
            continue

        for query_type, queries, by_user, mode in query_types:
            if queries:
                total = _encode_and_save_cache(
                    retriever_name,
                    queries,
                    by_user,
                    mode,
                )
                total_cached += total
                log_with_timestamp(f"  ✓ {query_type} {mode} 缓存: {total} 用户")
            else:
                log_with_timestamp(f"  (无 {query_type} {mode} 查询，跳过)")

        log_with_timestamp(f"✓ 检索器 {retriever_name} 处理完成")

    elapsed = time.time() - start_time

    # 统计缓存
    cache_files = 0
    cache_dir_size = 0.0
    for subdir in ["acl_noisy_query", "ccomp_noisy_query"]:
        subdir_path = os.path.join(CACHE_DIR, subdir)
        if os.path.exists(subdir_path):
            for root, _, files in os.walk(subdir_path):
                for name in files:
                    if name.endswith('.pkl'):
                        cache_files += 1
                        cache_dir_size += os.path.getsize(os.path.join(root, name)) / (1024 * 1024)

    log_with_timestamp("")
    log_with_timestamp("=" * 80)
    log_with_timestamp("✨ 完成!")
    log_with_timestamp(f"  • 处理检索器: {len(retriever_names)} 个")
    log_with_timestamp(f"  • ACL noisy 缓存: {sum(len(v) for v in acl_noisy_by_user.values())} 条")
    log_with_timestamp(f"  • CCOMP noisy 缓存: {sum(len(v) for v in ccomp_noisy_by_user.values())} 条")
    log_with_timestamp(f"  • 缓存文件: {cache_files} 个")
    log_with_timestamp(f"  • 缓存大小: {cache_dir_size:.1f} MB")
    log_with_timestamp(f"  • 总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    main()
