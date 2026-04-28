#!/usr/bin/env python3
"""
Build the ColBERTv2 document index for Baby_Products only.

This script is intentionally narrow:
- load Baby_Products metadata
- build the document collection
- write a persistent ColBERTv2 index
- save document id mapping and build manifest
"""

import hashlib
import json
import os
import pickle
import math
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Set, Tuple

import torch

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
if "HF_HUB_CACHE" not in os.environ:
    os.environ["HF_HUB_CACHE"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"

# ColBERTv2 checkpoint is not fully cached in this environment, so allow online resolution.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils

log_with_timestamp = utils.log_with_timestamp
load_product_metadata = utils.load_product_metadata
build_document_text = utils.build_document_text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_category_config


CATEGORY_NAME = "Baby_Products"
MODEL_NAME = "colbert-ir/colbertv2.0"
NBITS = 2
TARGET_NUM_PARTITIONS = 32768
KMEANS_NITERS = 1
EXPERIMENT_NAME = "colbertv2_index"


def setup_logging() -> None:
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_with_timestamp(f"[DEBUG] Logging directory ready: {log_dir}")


def load_raw_metadata(raw_corpus_file: str) -> Dict:
    log_with_timestamp("[DEBUG] Stage 1: metadata loading started")
    if not os.path.exists(raw_corpus_file):
        raise FileNotFoundError(f"Raw corpus file not found: {raw_corpus_file}")

    log_with_timestamp(f"Loading raw metadata from {raw_corpus_file}...")
    _, all_metadata = load_product_metadata(raw_corpus_file, None)
    if not all_metadata:
        raise ValueError(f"No metadata loaded from {raw_corpus_file}")
    log_with_timestamp(f"[DEBUG] Stage 1: metadata loading complete, items={len(all_metadata)}")
    return all_metadata


def build_documents(metadata: Dict) -> Tuple[List[Dict], Set[str]]:
    log_with_timestamp("[DEBUG] Stage 2: document materialization started")
    log_with_timestamp(f"Building documents from {len(metadata)} metadata items...")
    documents = []
    asins = set()

    for idx, (asin, meta) in enumerate(metadata.items()):
        if (idx + 1) % 50000 == 0:
            log_with_timestamp(f"  Processed {idx + 1}/{len(metadata)}")

        doc = meta.copy()
        doc['asin'] = asin
        documents.append(doc)
        asins.add(asin)

    log_with_timestamp(f"Built {len(documents)} documents")
    log_with_timestamp(f"[DEBUG] Stage 2: document materialization complete, asins={len(asins)}")
    return documents, asins


def compute_document_hash(documents: List[Dict]) -> str:
    log_with_timestamp("[DEBUG] Stage 3: document hash computation started")
    doc_ids = sorted(doc.get('asin', '') for doc in documents)
    doc_hash = hashlib.md5('|'.join(doc_ids).encode()).hexdigest()
    log_with_timestamp(f"[DEBUG] Stage 3: document hash computation complete, hash={doc_hash}")
    return doc_hash


def should_skip_existing_file(path: str) -> bool:
    """如果目标文件已存在且非空，则跳过重复写入。"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        log_with_timestamp(f"[DEBUG] Skip existing file: {path}")
        return True
    return False


def write_collection_tsv(documents: List[Dict], all_metadata: Dict, collection_path: str) -> None:
    log_with_timestamp("[DEBUG] Stage 4: collection TSV writing started")
    if should_skip_existing_file(collection_path):
        return
    log_with_timestamp(f"Writing collection TSV to {collection_path}...")
    with open(collection_path, 'w', encoding='utf-8') as f:
        for pid, doc in enumerate(documents):
            text = build_document_text(doc, all_metadata)
            text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
            if not text:
                raise ValueError(f"Empty document text for pid={pid}, asin={doc.get('asin', '')}")
            f.write(f"{pid}\t{text}\n")
            if (pid + 1) % 50000 == 0:
                log_with_timestamp(f"[DEBUG] Stage 4: wrote {pid + 1}/{len(documents)} rows")
    log_with_timestamp(f"[DEBUG] Stage 4: collection TSV writing complete, total_rows={len(documents)}")


def patch_colbert_encoding_progress() -> None:
    log_with_timestamp("[DEBUG] Stage 5: installing ColBERT encoding progress hook")
    from colbert.indexing.collection_encoder import CollectionEncoder
    from colbert.utils.utils import batch

    def encode_passages_with_progress(self, passages):
        total_passages = len(passages)
        batch_size = self.config.index_bsize * 50
        total_batches = max(1, math.ceil(total_passages / batch_size))
        log_with_timestamp(
            f"[DEBUG] Stage 5: encoding hook active, total_passages={total_passages}, "
            f"batch_size={batch_size}, total_batches={total_batches}"
        )

        if total_passages == 0:
            return None, None

        from colbert.infra.run import Run
        import torch

        Run().print(f"#> Encoding {total_passages} passages..")

        with torch.inference_mode():
            embs, doclens = [], []
            for batch_idx, passages_batch in enumerate(batch(passages, batch_size), start=1):
                processed_after = min(batch_idx * batch_size, total_passages)
                pct_after = processed_after * 100.0 / total_passages
                log_with_timestamp(
                    f"[DEBUG] Stage 5: encoding batch {batch_idx}/{total_batches}, "
                    f"progress={processed_after}/{total_passages} ({pct_after:.2f}%)"
                )

                embs_, doclens_ = self.checkpoint.docFromText(
                    passages_batch,
                    bsize=self.config.index_bsize,
                    keep_dims="flatten",
                    showprogress=(not self.use_gpu),
                    pool_factor=self.config.pool_factor,
                    clustering_mode=self.config.clustering_mode,
                    protected_tokens=self.config.protected_tokens,
                )
                embs.append(embs_)
                doclens.extend(doclens_)

                if batch_idx == total_batches:
                    log_with_timestamp(
                        f"[DEBUG] Stage 5: encoding finished {processed_after}/{total_passages} passages "
                        f"({pct_after:.2f}%)"
                    )

            embs = torch.cat(embs)

        return embs, doclens

    CollectionEncoder.encode_passages = encode_passages_with_progress


def patch_colbert_indexing_params() -> None:
    log_with_timestamp(
        f"[DEBUG] Stage 5: overriding ColBERT indexing params: "
        f"partitions={TARGET_NUM_PARTITIONS}, kmeans_niters={KMEANS_NITERS}"
    )
    from colbert.indexing.collection_indexer import CollectionIndexer
    import numpy as np

    original_setup = CollectionIndexer.setup
    original_train_kmeans = CollectionIndexer._train_kmeans

    def setup_with_fixed_partitions(self):
        if self.config.resume:
            if self._try_load_plan():
                if self.verbose > 1:
                    from colbert.infra.run import Run
                    Run().print_main(f"#> Loaded plan from {self.plan_path}:")
                    Run().print_main(f"#> num_chunks = {self.num_chunks}")
                    Run().print_main(f"#> num_partitions = {self.num_partitions}")
                    Run().print_main(f"#> num_embeddings_est = {self.num_embeddings_est}")
                    Run().print_main(f"#> avg_doclen_est = {self.avg_doclen_est}")
                return

        self.num_chunks = int(np.ceil(len(self.collection) / self.collection.get_chunksize()))
        sampled_pids = self._sample_pids()
        avg_doclen_est = self._sample_embeddings(sampled_pids)

        num_passages = len(self.collection)
        self.num_embeddings_est = num_passages * avg_doclen_est
        self.num_partitions = TARGET_NUM_PARTITIONS

        if self.verbose > 0:
            from colbert.infra.run import Run
            Run().print_main(f"Creating {self.num_partitions:,} partitions.")
            Run().print_main(f"*Estimated* {int(self.num_embeddings_est):,} embeddings.")

        self._save_plan()

    CollectionIndexer.setup = setup_with_fixed_partitions

    def train_kmeans_with_checkpoint(self, sample, shared_lists):
        centroids = original_train_kmeans(self, sample, shared_lists)

        checkpoint_dir = os.path.join(self.config.index_path_, "kmeans_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        centroids_path = os.path.join(
            checkpoint_dir,
            f"iter_{self.config.kmeans_niters:03d}_centroids.pt",
        )
        metadata_path = os.path.join(
            checkpoint_dir,
            f"iter_{self.config.kmeans_niters:03d}_metadata.json",
        )

        if not should_skip_existing_file(centroids_path):
            torch.save(centroids.cpu(), centroids_path)
            log_with_timestamp(f"[DEBUG] Stage 5: saved kmeans centroids checkpoint to {centroids_path}")

        if not should_skip_existing_file(metadata_path):
            metadata = {
                "kmeans_niters": self.config.kmeans_niters,
                "num_partitions": self.num_partitions,
                "num_centroids": int(centroids.size(0)),
                "dim": int(centroids.size(1)),
                "dtype": str(centroids.dtype),
                "created_at": datetime.now().isoformat(),
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            log_with_timestamp(f"[DEBUG] Stage 5: saved kmeans metadata checkpoint to {metadata_path}")

        return centroids

    CollectionIndexer._train_kmeans = train_kmeans_with_checkpoint


def validate_cuda_toolkit_for_colbert() -> None:
    log_with_timestamp("[DEBUG] Stage 5: validating CUDA toolkit for ColBERT extension build")

    nvcc_path = shutil.which("nvcc")
    cuda_home = os.environ.get("CUDA_HOME")
    cuda_path = os.environ.get("CUDA_PATH")
    candidate_roots = []
    candidate_headers = []

    for root in (cuda_home, cuda_path):
        if root and root not in candidate_roots:
            candidate_roots.append(root)

    if nvcc_path:
        inferred_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc_path)))
        if inferred_root not in candidate_roots:
            candidate_roots.append(inferred_root)

    for root in candidate_roots:
        standard_header = os.path.join(root, "include", "cuda_runtime.h")
        if standard_header not in candidate_headers:
            candidate_headers.append(standard_header)

        targets_root = os.path.join(root, "targets")
        if os.path.isdir(targets_root):
            for target_name in sorted(os.listdir(targets_root)):
                target_header = os.path.join(
                    targets_root, target_name, "include", "cuda_runtime.h"
                )
                if target_header not in candidate_headers:
                    candidate_headers.append(target_header)

    existing_headers = [path for path in candidate_headers if os.path.exists(path)]

    log_with_timestamp(f"[DEBUG] Stage 5: CUDA_HOME={cuda_home}")
    log_with_timestamp(f"[DEBUG] Stage 5: CUDA_PATH={cuda_path}")
    log_with_timestamp(f"[DEBUG] Stage 5: nvcc_path={nvcc_path}")
    log_with_timestamp(f"[DEBUG] Stage 5: candidate_cuda_roots={candidate_roots}")

    if not nvcc_path:
        raise RuntimeError(
            "ColBERT 索引构建需要可执行的 nvcc，但当前环境中未找到。"
            "请在 sbatch_wrapper --gpu 作业环境中加载完整 CUDA toolkit。"
        )

    if not candidate_roots:
        raise RuntimeError(
            "ColBERT 索引构建需要 CUDA toolkit，但当前环境中既没有 CUDA_HOME/CUDA_PATH，"
            "也无法从 nvcc 推断 CUDA 安装目录。"
        )

    if not existing_headers:
        raise RuntimeError(
            "ColBERT 在建索引时会即时编译 CUDA 扩展，但当前环境缺少 cuda_runtime.h。"
            f"已检查路径: {candidate_headers}。"
            "请确保作业环境加载了完整 CUDA toolkit，并正确设置 CUDA_HOME。"
            "如果使用 conda 打包的 CUDA，请确认 targets/<arch>/include/cuda_runtime.h 存在。"
        )

    log_with_timestamp(f"[DEBUG] Stage 5: detected cuda_runtime.h at {existing_headers[0]}")


def configure_cuda_env_for_colbert_extension_build() -> None:
    """为 torch cpp_extension / nvcc 注入 conda CUDA 头文件路径。"""
    candidate_roots = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(key)
        if root and root not in candidate_roots:
            candidate_roots.append(root)

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        inferred_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc_path)))
        if inferred_root not in candidate_roots:
            candidate_roots.append(inferred_root)

    include_dirs = []
    for root in candidate_roots:
        standard_include = os.path.join(root, "include")
        if os.path.isdir(standard_include) and standard_include not in include_dirs:
            include_dirs.append(standard_include)

        targets_root = os.path.join(root, "targets")
        if os.path.isdir(targets_root):
            for target_name in sorted(os.listdir(targets_root)):
                target_include = os.path.join(targets_root, target_name, "include")
                if os.path.isdir(target_include) and target_include not in include_dirs:
                    include_dirs.append(target_include)

    if not include_dirs:
        raise RuntimeError(
            "无法为 ColBERT CUDA 扩展编译配置头文件目录：未找到有效的 CUDA include 路径。"
        )

    for env_key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        existing = os.environ.get(env_key, "")
        paths = [path for path in existing.split(":") if path]
        for include_dir in reversed(include_dirs):
            if include_dir not in paths:
                paths.insert(0, include_dir)
        os.environ[env_key] = ":".join(paths)

    log_with_timestamp(f"[DEBUG] Stage 5: configured CUDA include dirs={include_dirs}")
    log_with_timestamp(f"[DEBUG] Stage 5: CPATH={os.environ.get('CPATH')}")
    log_with_timestamp(f"[DEBUG] Stage 5: CPLUS_INCLUDE_PATH={os.environ.get('CPLUS_INCLUDE_PATH')}")


def preflight_colbert_cuda_extension_build() -> None:
    """在正式编码/聚类前，提前触发 ColBERT CUDA 扩展编译。"""
    log_with_timestamp("[DEBUG] Stage 5: preflight build for ColBERT CUDA extension started")

    include_candidates = []
    for env_key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        include_candidates.extend([path for path in os.environ.get(env_key, "").split(":") if path])

    thrust_header_candidates = [
        os.path.join(include_dir, "thrust", "complex.h")
        for include_dir in include_candidates
    ]
    existing_thrust_headers = [path for path in thrust_header_candidates if os.path.exists(path)]
    if not existing_thrust_headers:
        raise RuntimeError(
            "ColBERT CUDA 扩展预检查失败：未找到 thrust/complex.h。"
            f"已检查路径: {thrust_header_candidates}"
        )

    log_with_timestamp(f"[DEBUG] Stage 5: detected thrust header at {existing_thrust_headers[0]}")

    from colbert.indexing.codecs.residual import ResidualCodec

    try:
        ResidualCodec.try_load_torch_extensions(use_gpu=True)
    except Exception as e:
        log_with_timestamp(f"[ERROR] Stage 5: ColBERT CUDA extension preflight failed: {type(e).__name__}: {e}")
        raise

    log_with_timestamp("[DEBUG] Stage 5: preflight build for ColBERT CUDA extension complete")


def build_colbert_index(documents: List[Dict], all_metadata: Dict, output_root: str) -> str:
    log_with_timestamp("[DEBUG] Stage 5: ColBERT build started")
    if not torch.cuda.is_available():
        raise RuntimeError("ColBERTv2 indexing requires CUDA")

    log_with_timestamp("[DEBUG] Stage 5: importing ColBERT modules")
    try:
        from colbert.infra import Run, RunConfig, ColBERTConfig
        from colbert import Indexer
    except Exception as e:
        log_with_timestamp(f"[ERROR] Stage 5: ColBERT import failed: {type(e).__name__}: {e}")
        raise

    log_with_timestamp(f"[DEBUG] Stage 5: model checkpoint={MODEL_NAME}")
    log_with_timestamp(f"[DEBUG] Stage 5: HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
    log_with_timestamp(f"[DEBUG] Stage 5: TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")

    os.makedirs(output_root, exist_ok=True)
    log_with_timestamp(f"[DEBUG] Stage 5: output directory ready: {output_root}")
    collection_path = os.path.join(output_root, "collection.tsv")
    write_collection_tsv(documents, all_metadata, collection_path)

    nranks = torch.cuda.device_count()
    if nranks < 1:
        raise RuntimeError("CUDA is available but no visible GPU ranks were detected")

    validate_cuda_toolkit_for_colbert()
    configure_cuda_env_for_colbert_extension_build()
    preflight_colbert_cuda_extension_build()

    log_with_timestamp(f"[DEBUG] Stage 5: detected CUDA device count={nranks}")
    log_with_timestamp(
        f"Building ColBERTv2 index with model={MODEL_NAME}, nbits={NBITS}, nranks={nranks}, "
        f"partitions={TARGET_NUM_PARTITIONS}, kmeans_niters={KMEANS_NITERS}"
    )
    log_with_timestamp(f"Output root: {output_root}")

    patch_colbert_encoding_progress()
    patch_colbert_indexing_params()
    log_with_timestamp("[DEBUG] Stage 5: entering ColBERT Run context")
    with Run().context(
        RunConfig(
            nranks=nranks,
            experiment=EXPERIMENT_NAME,
            avoid_fork_if_possible=True,
            root=output_root,
        )
    ):
        config = ColBERTConfig(
            nbits=NBITS,
            kmeans_niters=KMEANS_NITERS,
            avoid_fork_if_possible=True,
            root=output_root,
        )
        log_with_timestamp(
            f"[DEBUG] Stage 5: ColBERT config ready: nbits={config.nbits}, "
            f"kmeans_niters={config.kmeans_niters}, "
            f"avoid_fork_if_possible={config.avoid_fork_if_possible}, root={config.root}"
        )
        indexer = Indexer(checkpoint=MODEL_NAME, config=config)
        log_with_timestamp("[DEBUG] Stage 5: ColBERT indexer constructed, starting index()")
        indexer.index(name=EXPERIMENT_NAME, collection=collection_path, overwrite=True)
        log_with_timestamp("[DEBUG] Stage 5: ColBERT index() returned successfully")

    log_with_timestamp("[DEBUG] Stage 5: ColBERT build complete")
    return collection_path


def save_build_artifacts(output_root: str, documents: List[Dict], all_metadata: Dict, doc_hash: str, collection_path: str) -> None:
    log_with_timestamp("[DEBUG] Stage 6: artifact saving started")
    doc_ids_path = os.path.join(output_root, "doc_ids.pkl")
    metadata_path = os.path.join(output_root, "metadata.pkl")
    manifest_path = os.path.join(output_root, "build_manifest.json")

    doc_ids = [doc.get('asin', '') for doc in documents]
    if not should_skip_existing_file(doc_ids_path):
        log_with_timestamp(f"[DEBUG] Stage 6: saving doc_ids to {doc_ids_path}")
        with open(doc_ids_path, 'wb') as f:
            pickle.dump(doc_ids, f)

    if not should_skip_existing_file(metadata_path):
        log_with_timestamp(f"[DEBUG] Stage 6: saving metadata to {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(all_metadata, f)

    manifest = {
        "category": CATEGORY_NAME,
        "model_name": MODEL_NAME,
        "nbits": NBITS,
        "num_partitions": TARGET_NUM_PARTITIONS,
        "kmeans_niters": KMEANS_NITERS,
        "experiment": EXPERIMENT_NAME,
        "document_hash": doc_hash,
        "num_documents": len(documents),
        "output_root": output_root,
        "collection_path": collection_path,
        "created_at": datetime.now().isoformat(),
    }
    if not should_skip_existing_file(manifest_path):
        log_with_timestamp(f"[DEBUG] Stage 6: saving manifest to {manifest_path}")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    log_with_timestamp(f"Saved doc_ids: {doc_ids_path}")
    log_with_timestamp(f"Saved metadata: {metadata_path}")
    log_with_timestamp(f"Saved manifest: {manifest_path}")
    log_with_timestamp("[DEBUG] Stage 6: artifact saving complete")


def main() -> None:
    setup_logging()

    log_with_timestamp("=" * 80)
    log_with_timestamp("BUILD COLBERTV2 INDEX - STARTING")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"[DEBUG] Python version: {sys.version}")
    log_with_timestamp(f"[DEBUG] Torch version: {torch.__version__}")
    log_with_timestamp(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_with_timestamp(f"[DEBUG] CUDA device count: {torch.cuda.device_count()}")

    log_with_timestamp("[DEBUG] Stage 0: configuration loading started")
    category_config = get_category_config(CATEGORY_NAME)
    raw_corpus_file = category_config['raw_corpus_file']
    output_root = os.path.join(category_config['output_dir'], "colbertv2_index")

    log_with_timestamp(f"Category: {CATEGORY_NAME}")
    log_with_timestamp(f"Raw corpus file: {raw_corpus_file}")
    log_with_timestamp(f"Output root: {output_root}")
    log_with_timestamp(f"[DEBUG] Category config keys: {sorted(category_config.keys())}")
    log_with_timestamp("[DEBUG] Stage 0: configuration loading complete")

    all_metadata = load_raw_metadata(raw_corpus_file)
    documents, asins = build_documents(all_metadata)
    doc_hash = compute_document_hash(documents)

    log_with_timestamp(f"Total documents: {len(documents)}")
    log_with_timestamp(f"Total ASINs: {len(asins)}")
    log_with_timestamp(f"Document hash: {doc_hash}")

    log_with_timestamp("[DEBUG] Stage 5/6 orchestration: index build starting")
    collection_path = build_colbert_index(documents, all_metadata, output_root)
    log_with_timestamp("[DEBUG] Stage 5/6 orchestration: index build finished")
    log_with_timestamp("[DEBUG] Stage 6 orchestration: saving artifacts starting")
    save_build_artifacts(output_root, documents, all_metadata, doc_hash, collection_path)
    log_with_timestamp("[DEBUG] Stage 6 orchestration: saving artifacts finished")

    if os.path.exists(collection_path):
        log_with_timestamp(f"[DEBUG] Removing temporary collection file: {collection_path}")
        os.remove(collection_path)

    log_with_timestamp(f"ColBERTv2 index build completed under: {output_root}")
    log_with_timestamp("=" * 80)
    log_with_timestamp("BUILD COLBERTV2 INDEX - COMPLETE")


if __name__ == "__main__":
    main()
    log_with_timestamp("当前任务已完成，请做下一个任务的指示。")
