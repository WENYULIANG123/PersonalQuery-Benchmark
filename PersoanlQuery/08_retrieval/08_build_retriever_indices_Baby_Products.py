#!/usr/bin/env python3
"""
Build all retriever indices for full-scale evaluation.
Extracts index-building logic from main evaluation script.
Only builds indices, does NOT evaluate.
"""

import os
import sys
import pickle
import hashlib
import json
import math
import shutil
import subprocess
import threading
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime

# 确保 HF_HOME 和 HF_HUB_CACHE 指向正确的缓存目录
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
if "HF_HUB_CACHE" not in os.environ:
    os.environ["HF_HUB_CACHE"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"

# ColBERTv2 checkpoint is not fully cached in this environment, so allow online resolution.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# Add utils path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils

log_with_timestamp = utils.log_with_timestamp
load_product_metadata = utils.load_product_metadata
build_document_text = utils.build_document_text

# Import retriever utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from utils import retrievers

# ============ 配置加载 ============
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_category_config

CATEGORY_NAME = "Baby_Products"
CAT_CONFIG = get_category_config(CATEGORY_NAME)

COLBERTV2_MODEL_NAME = "colbert-ir/colbertv2.0"
COLBERTV2_NBITS = 2
COLBERTV2_TARGET_NUM_PARTITIONS = 32768
COLBERTV2_KMEANS_NITERS = 1
COLBERTV2_EXPERIMENT_NAME = "colbertv2_index"



def load_fullscale_metadata(metadata_file: str) -> Dict:
    """Load full metadata"""
    log_with_timestamp(f"Loading metadata from {metadata_file}...")
    metadata, _ = load_product_metadata(metadata_file, None)
    return metadata


def build_fullscale_documents(category: str, metadata: Dict) -> Tuple[List[Dict], Set[str]]:
    """Build full-scale document set"""
    log_with_timestamp(f"Building {len(metadata)} documents from metadata...")
    
    documents = []
    asins = set()
    
    for idx, (asin, meta) in enumerate(metadata.items()):
        if (idx + 1) % 50000 == 0:
            log_with_timestamp(f"  Processed {idx + 1}/{len(metadata)}")
        
        doc = meta.copy()
        doc['asin'] = asin
        documents.append(doc)
        asins.add(asin)
    
    log_with_timestamp(f"Built document list: {len(documents)} documents")
    return documents, asins


def compute_document_hash(documents: List[Dict]) -> str:
    """Compute hash of document set to detect changes"""
    doc_ids = sorted([doc.get('asin', '') for doc in documents])
    hash_input = '|'.join(doc_ids)
    return hashlib.md5(hash_input.encode()).hexdigest()


def parse_cuda_version_text(version_text: str) -> Tuple[int, int]:
    marker = "release "
    marker_index = version_text.find(marker)
    if marker_index < 0:
        raise ValueError(f"Unable to parse CUDA release from nvcc output: {version_text}")

    version_part = version_text[marker_index + len(marker):].split(",", 1)[0].strip()
    pieces = version_part.split(".")
    if len(pieces) < 2:
        raise ValueError(f"Unable to parse CUDA major/minor from nvcc output: {version_text}")

    return int(pieces[0]), int(pieces[1])


def get_nvcc_cuda_version(nvcc_path: str) -> Tuple[int, int]:
    result = subprocess.run(
        [nvcc_path, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return parse_cuda_version_text(result.stdout)


def get_torch_cuda_version() -> Tuple[int, int]:
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise RuntimeError("PyTorch CUDA version is unavailable; ColBERTv2 indexing requires CUDA PyTorch.")

    pieces = cuda_version.split(".")
    if len(pieces) < 2:
        raise RuntimeError(f"Unable to parse PyTorch CUDA version: {cuda_version}")

    return int(pieces[0]), int(pieces[1])


def get_gcc_major_version(compiler_path: str) -> int:
    result = subprocess.run(
        [compiler_path, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    first_line = result.stdout.splitlines()[0]
    marker = "(GCC) "
    marker_index = first_line.find(marker)
    if marker_index < 0:
        raise ValueError(f"Unable to parse GCC version from compiler output: {first_line}")

    version_text = first_line[marker_index + len(marker):].split()[0]
    return int(version_text.split(".", 1)[0])


def candidate_cuda_roots_for_colbert() -> List[str]:
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

    usr_local = "/usr/local"
    if os.path.isdir(usr_local):
        for entry_name in sorted(os.listdir(usr_local)):
            if entry_name.startswith("cuda-"):
                root = os.path.join(usr_local, entry_name)
                if root not in candidate_roots:
                    candidate_roots.append(root)

    return candidate_roots


def select_cuda_toolkit_for_colbert_extension_build() -> None:
    torch_cuda_major, torch_cuda_minor = get_torch_cuda_version()
    candidate_roots = candidate_cuda_roots_for_colbert()
    if not candidate_roots:
        raise RuntimeError("No CUDA toolkit candidates found for ColBERT CUDA extension build.")

    compatible_roots = []
    rejected_roots = []
    for root in candidate_roots:
        nvcc_path = os.path.join(root, "bin", "nvcc")
        if not os.path.exists(nvcc_path):
            rejected_roots.append((root, "missing nvcc"))
            continue

        try:
            cuda_major, cuda_minor = get_nvcc_cuda_version(nvcc_path)
        except Exception as e:
            rejected_roots.append((root, f"nvcc version parse failed: {type(e).__name__}: {e}"))
            continue

        if cuda_major != torch_cuda_major:
            rejected_roots.append(
                (root, f"CUDA {cuda_major}.{cuda_minor} does not match PyTorch CUDA major {torch_cuda_major}")
            )
            continue

        compatible_roots.append((root, cuda_major, cuda_minor))

    if not compatible_roots:
        raise RuntimeError(
            "No CUDA toolkit with the same major version as PyTorch CUDA was found. "
            f"PyTorch CUDA={torch.version.cuda}; rejected candidates={rejected_roots}"
        )

    compatible_roots.sort(key=lambda item: (abs(item[2] - torch_cuda_minor), -item[2], item[0]))
    selected_root, selected_major, selected_minor = compatible_roots[0]

    os.environ["CUDA_HOME"] = selected_root
    os.environ["CUDA_PATH"] = selected_root
    selected_bin = os.path.join(selected_root, "bin")
    path_entries = [path for path in os.environ.get("PATH", "").split(":") if path]
    if selected_bin in path_entries:
        path_entries.remove(selected_bin)
    path_entries.insert(0, selected_bin)
    os.environ["PATH"] = ":".join(path_entries)

    from torch.utils import cpp_extension
    cpp_extension.CUDA_HOME = selected_root

    log_with_timestamp(
        f"[DEBUG] ColBERTv2: selected CUDA toolkit {selected_root} "
        f"(nvcc CUDA {selected_major}.{selected_minor}, PyTorch CUDA {torch.version.cuda})"
    )


def configure_host_compiler_for_colbert_extension_build() -> None:
    cc_path = "/usr/bin/gcc"
    cxx_path = "/usr/bin/g++"

    if not os.path.exists(cc_path):
        raise RuntimeError(f"Required CUDA host C compiler not found: {cc_path}")
    if not os.path.exists(cxx_path):
        raise RuntimeError(f"Required CUDA host C++ compiler not found: {cxx_path}")

    cc_major = get_gcc_major_version(cc_path)
    cxx_major = get_gcc_major_version(cxx_path)
    max_supported_major = 13
    if cc_major > max_supported_major:
        raise RuntimeError(f"{cc_path} GCC major version {cc_major} exceeds CUDA 12.5 support limit {max_supported_major}")
    if cxx_major > max_supported_major:
        raise RuntimeError(f"{cxx_path} GCC major version {cxx_major} exceeds CUDA 12.5 support limit {max_supported_major}")

    os.environ["CC"] = cc_path
    os.environ["CXX"] = cxx_path
    log_with_timestamp(
        f"[DEBUG] ColBERTv2: selected host compilers CC={cc_path} (GCC {cc_major}), "
        f"CXX={cxx_path} (GCC {cxx_major})"
    )


def should_skip_existing_file(path: str) -> bool:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        log_with_timestamp(f"[DEBUG] Skip existing file: {path}")
        return True
    return False


def write_colbertv2_collection_tsv(documents: List[Dict], all_metadata: Dict, collection_path: str) -> None:
    log_with_timestamp("[DEBUG] ColBERTv2: collection TSV writing started")
    log_with_timestamp(f"Writing ColBERTv2 collection TSV to {collection_path}...")
    with open(collection_path, 'w', encoding='utf-8') as f:
        for pid, doc in enumerate(documents):
            text = build_document_text(doc, all_metadata)
            text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
            if not text:
                raise ValueError(f"Empty document text for pid={pid}, asin={doc.get('asin', '')}")
            f.write(f"{pid}\t{text}\n")
            if (pid + 1) % 50000 == 0:
                log_with_timestamp(f"[DEBUG] ColBERTv2: wrote {pid + 1}/{len(documents)} rows")
    log_with_timestamp(f"[DEBUG] ColBERTv2: collection TSV writing complete, total_rows={len(documents)}")


def patch_colbert_encoding_progress() -> None:
    log_with_timestamp("[DEBUG] ColBERTv2: installing encoding progress hook")
    from colbert.indexing.collection_encoder import CollectionEncoder
    from colbert.utils.utils import batch

    def encode_passages_with_progress(self, passages):
        total_passages = len(passages)
        batch_size = self.config.index_bsize * 50
        total_batches = max(1, math.ceil(total_passages / batch_size))
        log_with_timestamp(
            f"[DEBUG] ColBERTv2: encoding hook active, total_passages={total_passages}, "
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
                    f"[DEBUG] ColBERTv2: encoding batch {batch_idx}/{total_batches}, "
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
                        f"[DEBUG] ColBERTv2: encoding finished {processed_after}/{total_passages} passages "
                        f"({pct_after:.2f}%)"
                    )

            embs = torch.cat(embs)

        return embs, doclens

    CollectionEncoder.encode_passages = encode_passages_with_progress


def patch_colbert_indexing_params() -> None:
    log_with_timestamp(
        f"[DEBUG] ColBERTv2: overriding indexing params: "
        f"partitions={COLBERTV2_TARGET_NUM_PARTITIONS}, kmeans_niters={COLBERTV2_KMEANS_NITERS}"
    )
    from colbert.indexing.collection_indexer import CollectionIndexer
    import numpy as np

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
        self.num_partitions = COLBERTV2_TARGET_NUM_PARTITIONS

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
            log_with_timestamp(f"[DEBUG] ColBERTv2: saved kmeans centroids checkpoint to {centroids_path}")

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
            log_with_timestamp(f"[DEBUG] ColBERTv2: saved kmeans metadata checkpoint to {metadata_path}")

        return centroids

    CollectionIndexer._train_kmeans = train_kmeans_with_checkpoint


def validate_cuda_toolkit_for_colbert() -> None:
    log_with_timestamp("[DEBUG] ColBERTv2: validating CUDA toolkit for extension build")

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

    log_with_timestamp(f"[DEBUG] ColBERTv2: CUDA_HOME={cuda_home}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: CUDA_PATH={cuda_path}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: nvcc_path={nvcc_path}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: candidate_cuda_roots={candidate_roots}")

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

    log_with_timestamp(f"[DEBUG] ColBERTv2: detected cuda_runtime.h at {existing_headers[0]}")


def configure_cuda_env_for_colbert_extension_build() -> None:
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

    def add_include_dir(path: str) -> None:
        if os.path.isdir(path) and path not in include_dirs:
            include_dirs.append(path)

    for root in candidate_roots:
        standard_include = os.path.join(root, "include")
        add_include_dir(standard_include)
        add_include_dir(os.path.join(standard_include, "cccl"))

        targets_root = os.path.join(root, "targets")
        if os.path.isdir(targets_root):
            for target_name in sorted(os.listdir(targets_root)):
                target_include = os.path.join(targets_root, target_name, "include")
                add_include_dir(target_include)
                add_include_dir(os.path.join(target_include, "cccl"))

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

    log_with_timestamp(f"[DEBUG] ColBERTv2: configured CUDA include dirs={include_dirs}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: CPATH={os.environ.get('CPATH')}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: CPLUS_INCLUDE_PATH={os.environ.get('CPLUS_INCLUDE_PATH')}")


def preflight_colbert_cuda_extension_build() -> None:
    log_with_timestamp("[DEBUG] ColBERTv2: CUDA extension preflight build started")

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

    log_with_timestamp(f"[DEBUG] ColBERTv2: detected thrust header at {existing_thrust_headers[0]}")

    from colbert.indexing.codecs.residual import ResidualCodec

    try:
        ResidualCodec.try_load_torch_extensions(use_gpu=True)
    except Exception as e:
        log_with_timestamp(f"[ERROR] ColBERTv2: CUDA extension preflight failed: {type(e).__name__}: {e}")
        raise

    log_with_timestamp("[DEBUG] ColBERTv2: CUDA extension preflight build complete")


def get_colbertv2_output_root(doc_hash: str) -> str:
    if not doc_hash:
        raise ValueError("ColBERTv2 output root requires a non-empty document hash")
    return os.path.join(CAT_CONFIG['retriever_cache_dir'], f"colbertv2_{doc_hash}")


def get_colbertv2_index_path(output_root: str) -> str:
    return os.path.join(output_root, COLBERTV2_EXPERIMENT_NAME, "indexes", COLBERTV2_EXPERIMENT_NAME)


def build_colbertv2_index(documents: List[Dict], all_metadata: Dict, output_root: str) -> str:
    log_with_timestamp("[DEBUG] ColBERTv2: build started")
    if not torch.cuda.is_available():
        raise RuntimeError("ColBERTv2 indexing requires CUDA")

    nranks = torch.cuda.device_count()
    if nranks < 1:
        raise RuntimeError("CUDA is available but no visible GPU ranks were detected")

    select_cuda_toolkit_for_colbert_extension_build()
    configure_host_compiler_for_colbert_extension_build()
    validate_cuda_toolkit_for_colbert()
    configure_cuda_env_for_colbert_extension_build()
    preflight_colbert_cuda_extension_build()

    log_with_timestamp("[DEBUG] ColBERTv2: importing ColBERT modules")
    try:
        from colbert.infra import Run, RunConfig, ColBERTConfig
        from colbert import Indexer
    except Exception as e:
        log_with_timestamp(f"[ERROR] ColBERTv2: import failed: {type(e).__name__}: {e}")
        raise

    log_with_timestamp(f"[DEBUG] ColBERTv2: model checkpoint={COLBERTV2_MODEL_NAME}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")

    os.makedirs(output_root, exist_ok=True)
    log_with_timestamp(f"[DEBUG] ColBERTv2: output directory ready: {output_root}")
    collection_path = os.path.join(output_root, "collection.tsv")
    write_colbertv2_collection_tsv(documents, all_metadata, collection_path)

    log_with_timestamp(f"[DEBUG] ColBERTv2: detected CUDA device count={nranks}")
    log_with_timestamp(
        f"Building ColBERTv2 index with model={COLBERTV2_MODEL_NAME}, nbits={COLBERTV2_NBITS}, "
        f"nranks={nranks}, partitions={COLBERTV2_TARGET_NUM_PARTITIONS}, "
        f"kmeans_niters={COLBERTV2_KMEANS_NITERS}"
    )
    log_with_timestamp(f"ColBERTv2 output root: {output_root}")

    patch_colbert_encoding_progress()
    patch_colbert_indexing_params()
    log_with_timestamp("[DEBUG] ColBERTv2: entering ColBERT Run context")
    with Run().context(
        RunConfig(
            nranks=nranks,
            experiment=COLBERTV2_EXPERIMENT_NAME,
            avoid_fork_if_possible=True,
            root=output_root,
        )
    ):
        config = ColBERTConfig(
            nbits=COLBERTV2_NBITS,
            kmeans_niters=COLBERTV2_KMEANS_NITERS,
            avoid_fork_if_possible=True,
            root=output_root,
        )
        log_with_timestamp(
            f"[DEBUG] ColBERTv2: config ready: nbits={config.nbits}, "
            f"kmeans_niters={config.kmeans_niters}, "
            f"avoid_fork_if_possible={config.avoid_fork_if_possible}, root={config.root}"
        )
        indexer = Indexer(checkpoint=COLBERTV2_MODEL_NAME, config=config)
        log_with_timestamp("[DEBUG] ColBERTv2: indexer constructed, starting index()")
        indexer.index(name=COLBERTV2_EXPERIMENT_NAME, collection=collection_path, overwrite=True)
        log_with_timestamp("[DEBUG] ColBERTv2: index() returned successfully")

    log_with_timestamp("[DEBUG] ColBERTv2: build complete")
    return collection_path


def save_colbertv2_build_artifacts(output_root: str, documents: List[Dict], all_metadata: Dict, doc_hash: str, collection_path: str) -> None:
    log_with_timestamp("[DEBUG] ColBERTv2: artifact saving started")
    doc_ids_path = os.path.join(output_root, "doc_ids.pkl")
    metadata_path = os.path.join(output_root, "metadata.pkl")
    manifest_path = os.path.join(output_root, "build_manifest.json")

    doc_ids = [doc.get('asin', '') for doc in documents]
    log_with_timestamp(f"[DEBUG] ColBERTv2: saving doc_ids to {doc_ids_path}")
    with open(doc_ids_path, 'wb') as f:
        pickle.dump(doc_ids, f)

    log_with_timestamp(f"[DEBUG] ColBERTv2: saving metadata to {metadata_path}")
    with open(metadata_path, 'wb') as f:
        pickle.dump(all_metadata, f)

    manifest = {
        "category": CATEGORY_NAME,
        "model_name": COLBERTV2_MODEL_NAME,
        "nbits": COLBERTV2_NBITS,
        "num_partitions": COLBERTV2_TARGET_NUM_PARTITIONS,
        "kmeans_niters": COLBERTV2_KMEANS_NITERS,
        "experiment": COLBERTV2_EXPERIMENT_NAME,
        "document_hash": doc_hash,
        "num_documents": len(documents),
        "output_root": output_root,
        "collection_path": collection_path,
        "created_at": datetime.now().isoformat(),
    }
    log_with_timestamp(f"[DEBUG] ColBERTv2: saving manifest to {manifest_path}")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log_with_timestamp(f"Saved ColBERTv2 doc_ids: {doc_ids_path}")
    log_with_timestamp(f"Saved ColBERTv2 metadata: {metadata_path}")
    log_with_timestamp(f"Saved ColBERTv2 manifest: {manifest_path}")
    log_with_timestamp("[DEBUG] ColBERTv2: artifact saving complete")


def build_colbertv2_retriever_index(documents: List[Dict], doc_hash: str, all_metadata: Dict) -> str:
    output_root = get_colbertv2_output_root(doc_hash)
    collection_path = build_colbertv2_index(documents, all_metadata, output_root)
    save_colbertv2_build_artifacts(output_root, documents, all_metadata, doc_hash, collection_path)

    if os.path.exists(collection_path):
        log_with_timestamp(f"[DEBUG] ColBERTv2: removing temporary collection file: {collection_path}")
        os.remove(collection_path)

    log_with_timestamp(f"ColBERTv2 index build completed under: {output_root}")
    return output_root


def get_cache_paths(retriever_name: str, doc_hash: str, cache_dir: str) -> Dict[str, str]:
    """Get cache file paths for a retriever"""
    DENSE_RETRIEVERS = ['minilm', 'star', 'e5', 'bge', 'gritlm', 'ance']
    # ColBERT 使用 token-level embeddings，需要特殊处理，不使用简单的 numpy 数组
    COLBERT_RETRIEVERS = ['colbert']
    SPARSE_RETRIEVERS = ['bm25', 'splade']

    if retriever_name in DENSE_RETRIEVERS:
        base_path = os.path.join(cache_dir, f"{retriever_name}_{doc_hash}")
        return {
            'config': f"{base_path}_config.pkl",
            'embeddings': f"{base_path}_embeddings.npy",
            'doc_ids': f"{base_path}_doc_ids.pkl",
            'metadata': f"{base_path}_metadata.pkl",
        }
    elif retriever_name == 'colbertv2':
        output_root = get_colbertv2_output_root(doc_hash)
        index_path = get_colbertv2_index_path(output_root)
        return {
            'index_root': output_root,
            'index_path': index_path,
            'index_metadata': os.path.join(index_path, "metadata.json"),
            'plan': os.path.join(index_path, "plan.json"),
            'centroids': os.path.join(index_path, "centroids.pt"),
            'ivf': os.path.join(index_path, "ivf.pid.pt"),
            'doc_ids': os.path.join(output_root, "doc_ids.pkl"),
            'metadata': os.path.join(output_root, "metadata.pkl"),
            'manifest': os.path.join(output_root, "build_manifest.json"),
        }
    elif retriever_name in COLBERT_RETRIEVERS:
        # ColBERT 使用 pickle 保存（因为是 token-level 可变长 embeddings）
        return {
            'pickle': os.path.join(cache_dir, f"{retriever_name}_{doc_hash}.pkl")
        }
    else:
        return {
            'pickle': os.path.join(cache_dir, f"{retriever_name}_{doc_hash}.pkl")
        }


def cache_exists(retriever_name: str, doc_hash: str, cache_dir: str) -> bool:
    """Check if retriever cache already exists"""
    DENSE_RETRIEVERS = ['minilm', 'star', 'e5', 'bge', 'gritlm', 'ance']
    COLBERT_RETRIEVERS = ['colbert']

    if retriever_name in DENSE_RETRIEVERS:
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
        return os.path.exists(paths['config']) and os.path.exists(paths['embeddings'])
    elif retriever_name == 'colbertv2':
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
        required_files = ['index_metadata', 'plan', 'centroids', 'ivf', 'doc_ids', 'metadata', 'manifest']
        return all(os.path.exists(paths[key]) and os.path.getsize(paths[key]) > 0 for key in required_files)
    elif retriever_name in COLBERT_RETRIEVERS:
        # ColBERT 使用 pickle 保存
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
        return os.path.exists(paths['pickle'])
    else:
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
        return os.path.exists(paths['pickle'])


def validate_retriever_cache(retriever_name: str, doc_hash: str, cache_dir: str, n_documents: int) -> Tuple[bool, str]:
    """Validate retriever cache integrity

    Returns:
        (is_valid, error_message)
    """
    DENSE_RETRIEVERS = ['minilm', 'star', 'e5', 'bge', 'gritlm', 'ance']
    COLBERT_RETRIEVERS = ['colbert']

    log_with_timestamp(f"[VALIDATE] Checking cache integrity for {retriever_name}...")

    if retriever_name in DENSE_RETRIEVERS:
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)

        # 检查文件是否存在
        required_files = ['config', 'embeddings', 'doc_ids', 'metadata']
        for key in required_files:
            if key not in paths or not os.path.exists(paths[key]):
                return False, f"Missing required file: {key}"

        # 检查文件是否为空
        for key in required_files:
            if os.path.getsize(paths[key]) == 0:
                return False, f"Empty file: {key}"

        # 检查 embeddings 和 doc_ids 数量是否匹配
        try:
            embeddings = np.load(paths['embeddings'], mmap_mode='r')
            n_embeddings = embeddings.shape[0]

            with open(paths['doc_ids'], 'rb') as f:
                doc_ids = pickle.load(f)
            n_doc_ids = len(doc_ids)

            if n_embeddings != n_doc_ids:
                return False, f"Embeddings count ({n_embeddings}) != doc_ids count ({n_doc_ids})"

            if n_embeddings != n_documents:
                return False, f"Embeddings count ({n_embeddings}) != expected document count ({n_documents})"

            # 检查 doc_ids 是否有重复
            if len(doc_ids) != len(set(doc_ids)):
                duplicates = len(doc_ids) - len(set(doc_ids))
                return False, f"Found {duplicates} duplicate doc_ids"

            log_with_timestamp(f"[VALIDATE] {retriever_name}: embeddings={n_embeddings}, doc_ids={n_doc_ids}, all checks passed ✓")
            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    elif retriever_name == 'colbertv2':
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)

        required_files = ['index_metadata', 'plan', 'centroids', 'ivf', 'doc_ids', 'metadata', 'manifest']
        for key in required_files:
            if not os.path.exists(paths[key]):
                return False, f"Missing required ColBERTv2 file: {key} ({paths[key]})"
            if os.path.getsize(paths[key]) == 0:
                return False, f"Empty ColBERTv2 file: {key} ({paths[key]})"

        try:
            with open(paths['doc_ids'], 'rb') as f:
                doc_ids = pickle.load(f)

            if len(doc_ids) != n_documents:
                return False, f"doc_ids count ({len(doc_ids)}) != expected document count ({n_documents})"

            if len(doc_ids) != len(set(doc_ids)):
                duplicates = len(doc_ids) - len(set(doc_ids))
                return False, f"Found {duplicates} duplicate ColBERTv2 doc_ids"

            with open(paths['manifest'], 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            manifest_hash = manifest.get('document_hash')
            if manifest_hash != doc_hash:
                return False, f"Manifest document_hash ({manifest_hash}) != current document_hash ({doc_hash})"

            manifest_count = manifest.get('num_documents')
            if manifest_count != n_documents:
                return False, f"Manifest num_documents ({manifest_count}) != expected document count ({n_documents})"

            log_with_timestamp(
                f"[VALIDATE] {retriever_name}: index_path={paths['index_path']}, doc_ids={len(doc_ids)}, all checks passed ✓"
            )
            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    elif retriever_name in COLBERT_RETRIEVERS:
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)

        if not os.path.exists(paths['pickle']):
            return False, f"Missing pickle file"

        if os.path.getsize(paths['pickle']) == 0:
            return False, f"Empty pickle file"

        try:
            with open(paths['pickle'], 'rb') as f:
                data = pickle.load(f)

            # 检查是否为有效对象
            if data is None:
                return False, "Pickle data is None"

            log_with_timestamp(f"[VALIDATE] {retriever_name}: pickle file valid ✓")
            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    else:
        # BM25 等其他检索器
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)

        if not os.path.exists(paths['pickle']):
            return False, f"Missing pickle file"

        if os.path.getsize(paths['pickle']) == 0:
            return False, f"Empty pickle file"

        try:
            with open(paths['pickle'], 'rb') as f:
                data = pickle.load(f)

            # 检查是否有 search 方法
            if not hasattr(data, 'search'):
                return False, f"BM25 object missing 'search' method"

            log_with_timestamp(f"[VALIDATE] {retriever_name}: pickle file valid ✓")
            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"


def _normalize_embeddings_for_save(embeddings, retriever_name: str) -> np.ndarray:
    """
    Normalize embeddings for saving, handling variable-length cases (e.g., E5 multi-window).
    
    For mixed-shape embeddings (some 1D, some 2D), average-pool multi-window embeddings
    to create uniform 2D shape (n_docs, embedding_dim).
    
    Args:
        embeddings: List or array of embeddings
        retriever_name: Name of retriever (for logging)
    
    Returns:
        Normalized numpy array with shape (n_docs, embedding_dim)
    """
    log_with_timestamp(f"[DEBUG] _normalize_embeddings_for_save called for {retriever_name}")
    log_with_timestamp(f"[DEBUG] embeddings type: {type(embeddings)}")
    if isinstance(embeddings, list):
        log_with_timestamp(f"[DEBUG] embeddings is list with {len(embeddings)} items")
    
    if not isinstance(embeddings, list):
        # Already a tensor or array
        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32)
        elif hasattr(embeddings, 'detach'):
            return np.asarray(embeddings.detach().tolist(), dtype=np.float32)
        else:
            return embeddings.numpy().astype(np.float32)
    
    # Handle list of embeddings (possibly with mixed shapes)
    normalized = []
    has_multiwindow = False
    
    for i, emb in enumerate(embeddings):
        # Convert to numpy
        log_with_timestamp(f"[DEBUG] Processing embedding {i}: type={type(emb)}")
        if hasattr(emb, 'detach'):
            emb_np = np.asarray(emb.detach().tolist(), dtype=np.float32)
        elif isinstance(emb, np.ndarray):
            emb_np = emb
        else:
            emb_np = emb.numpy()
        
        log_with_timestamp(f"[DEBUG] emb[{i}] shape={emb_np.shape}, ndim={emb_np.ndim}")
        
        # Handle multi-window embeddings (2D) vs single-window (1D)
        if emb_np.ndim == 2:
            # Multi-window: average-pool across windows
            has_multiwindow = True
            emb_pooled = emb_np.mean(axis=0)  # [num_windows, dim] -> [dim]
            log_with_timestamp(f"[DEBUG] emb[{i}] multi-window: {emb_np.shape} -> {emb_pooled.shape}")
            normalized.append(emb_pooled)
        elif emb_np.ndim == 1:
            # Single-window: keep as is
            log_with_timestamp(f"[DEBUG] emb[{i}] single-window: kept as {emb_np.shape}")
            normalized.append(emb_np)
        else:
            raise ValueError(f"Unexpected embedding shape: {emb_np.shape}")
    
    if has_multiwindow and retriever_name == 'e5':
        log_with_timestamp(f"  [INFO] E5: Applied average pooling to multi-window embeddings for uniform shape")
    
    # Stack into (n_docs, embedding_dim)
    log_with_timestamp(f"[DEBUG] Stacking {len(normalized)} normalized embeddings...")
    embeddings_np = np.stack(normalized, axis=0).astype(np.float32)
    log_with_timestamp(f"[DEBUG] Final embeddings shape: {embeddings_np.shape}")
    return embeddings_np


def save_retriever_to_cache(retriever_name: str, doc_hash: str, retriever: object, cache_dir: str) -> bool:
    """Save retriever to disk cache. Returns True if successful."""
    DENSE_RETRIEVERS = ['minilm', 'star', 'e5', 'bge', 'gritlm', 'ance']
    COLBERT_RETRIEVERS = ['colbert']
    SPARSE_RETRIEVERS = ['bm25', 'splade']

    log_with_timestamp(f"[DEBUG] save_retriever_to_cache called for {retriever_name}")

    try:
        if retriever_name in DENSE_RETRIEVERS:
            log_with_timestamp(f"[CACHE_SAVE] Saving {retriever_name} with separated embeddings...")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
            log_with_timestamp(f"[DEBUG] Paths: {paths}")

            log_with_timestamp(f"[DEBUG] Checking for doc_embeddings...")
            if hasattr(retriever, 'doc_embeddings') and retriever.doc_embeddings is not None:
                log_with_timestamp(f"[DEBUG] doc_embeddings found, normalizing...")
                embeddings = retriever.doc_embeddings
                log_with_timestamp(f"[DEBUG] embeddings type: {type(embeddings)}")

                # Use robust normalization that handles variable-length embeddings
                log_with_timestamp(f"[DEBUG] Calling _normalize_embeddings_for_save...")
                embeddings_np = _normalize_embeddings_for_save(embeddings, retriever_name)
                log_with_timestamp(f"[DEBUG] Normalization complete, shape: {embeddings_np.shape}")

                log_with_timestamp(f"[DEBUG] Saving embeddings to {paths['embeddings']}...")
                np.save(paths['embeddings'], embeddings_np)
                log_with_timestamp(f"[DEBUG] Embeddings saved successfully")
                size_gb = embeddings_np.nbytes / (1024**3)
                log_with_timestamp(f"  → Embeddings: {paths['embeddings']} ({size_gb:.2f}GB)")
                log_with_timestamp(f"  → Shape: {embeddings_np.shape}")

                log_with_timestamp(f"[DEBUG] Clearing doc_embeddings from memory...")
                retriever.doc_embeddings = None

            # 删除 model（对于 GritLM 必须在 pickle 之前删除，因为其底层 MistralForCausalLM 无法 pickle）
            log_with_timestamp(f"[DEBUG] Clearing model from retriever before pickle...")
            if hasattr(retriever, 'model') and retriever.model is not None:
                del retriever.model
                retriever.model = None
                log_with_timestamp(f"[DEBUG] Model cleared from retriever")

            log_with_timestamp(f"[DEBUG] Saving config to {paths['config']}...")
            with open(paths['config'], 'wb') as f:
                pickle.dump(retriever, f)
            log_with_timestamp(f"  → Config: {paths['config']}")
            log_with_timestamp(f"[DEBUG] Config saved")

            if hasattr(retriever, 'doc_ids'):
                log_with_timestamp(f"[DEBUG] Saving doc_ids...")
                with open(paths['doc_ids'], 'wb') as f:
                    pickle.dump(retriever.doc_ids, f)
                log_with_timestamp(f"[DEBUG] doc_ids saved")

            if hasattr(retriever, 'all_metadata'):
                log_with_timestamp(f"[DEBUG] Saving metadata...")
                with open(paths['metadata'], 'wb') as f:
                    pickle.dump(retriever.all_metadata, f)
                log_with_timestamp(f"[DEBUG] metadata saved")

            log_with_timestamp(f"[DEBUG] Dense retriever save complete")

            # 释放 GPU 模型权重和缓存
            log_with_timestamp(f"[MEMORY] Releasing {retriever_name} model from GPU...")
            if hasattr(retriever, 'model') and retriever.model is not None:
                del retriever.model
                retriever.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            log_with_timestamp(f"[MEMORY] GPU memory released for {retriever_name}")

            return True
        elif retriever_name in COLBERT_RETRIEVERS:
            # ColBERT 使用 token-level embeddings，需要特殊处理
            log_with_timestamp(f"[CACHE_SAVE] Saving {retriever_name} with token-level embeddings (as pickle)...")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)

            # ColBERT embeddings are list of tensors (possibly nested for multi-window)
            # 保留 GPU 张量后保存
            if hasattr(retriever, 'doc_embeddings') and retriever.doc_embeddings is not None:
                log_with_timestamp(f"[DEBUG] Keeping ColBERT embeddings on GPU before saving...")
                doc_embeddings_gpu = []
                for i, emb in enumerate(retriever.doc_embeddings):
                    if isinstance(emb, list):
                        # Multi-window: list of tensors
                        doc_embeddings_gpu.append([w.detach() for w in emb])
                    else:
                        # Single tensor
                        doc_embeddings_gpu.append(emb.detach())
                retriever.doc_embeddings = doc_embeddings_gpu
                log_with_timestamp(f"[DEBUG] ColBERT embeddings kept on GPU")

            log_with_timestamp(f"[DEBUG] Saving ColBERT to {paths['pickle']}...")
            with open(paths['pickle'], 'wb') as f:
                pickle.dump(retriever, f)

            log_with_timestamp(f"[DEBUG] File saved, getting size...")
            size_mb = os.path.getsize(paths['pickle']) / (1024 * 1024)
            log_with_timestamp(f"  → {paths['pickle']} ({size_mb:.1f}MB)")
            log_with_timestamp(f"[DEBUG] ColBERT save complete")

            # 释放 GPU 模型权重和缓存
            log_with_timestamp(f"[MEMORY] Releasing {retriever_name} model from GPU...")
            if hasattr(retriever, 'model') and retriever.model is not None:
                del retriever.model
                retriever.model = None
            if hasattr(retriever, 'tokenizer') and retriever.tokenizer is not None:
                del retriever.tokenizer
                retriever.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            log_with_timestamp(f"[MEMORY] GPU memory released for {retriever_name}")

            return True
        elif retriever_name in SPARSE_RETRIEVERS:
            # BM25 和 SPLADE 都使用 pickle 保存
            log_with_timestamp(f"[CACHE_SAVE] Saving {retriever_name} (sparse retriever)...")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
            log_with_timestamp(f"[DEBUG] Saving to {paths['pickle']}...")

            # SPLADE 需要在 pickle 前删除模型（FP16 CUDA tensor 无法 pickle）
            # 但保留 doc_vectors（已转换为 CPU float dict）
            if retriever_name == 'splade' and hasattr(retriever, 'model') and retriever.model is not None:
                log_with_timestamp(f"[DEBUG] Clearing SPLADE model/tokenizer before pickle...")
                if hasattr(retriever, 'model') and retriever.model is not None:
                    del retriever.model
                    retriever.model = None
                if hasattr(retriever, 'tokenizer') and retriever.tokenizer is not None:
                    del retriever.tokenizer
                    retriever.tokenizer = None
                log_with_timestamp(f"[DEBUG] SPLADE model/tokenizer cleared, doc_vectors preserved")

            with open(paths['pickle'], 'wb') as f:
                pickle.dump(retriever, f)

            log_with_timestamp(f"[DEBUG] File saved, getting size...")
            size_mb = os.path.getsize(paths['pickle']) / (1024 * 1024)
            log_with_timestamp(f"  → {paths['pickle']} ({size_mb:.1f}MB)")
            log_with_timestamp(f"[DEBUG] Sparse retriever save complete")
            return True
        else:
            log_with_timestamp(f"[CACHE_SAVE] Saving {retriever_name}...")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
            log_with_timestamp(f"[DEBUG] Saving to {paths['pickle']}...")

            with open(paths['pickle'], 'wb') as f:
                pickle.dump(retriever, f)

            log_with_timestamp(f"[DEBUG] File saved, getting size...")
            size_mb = os.path.getsize(paths['pickle']) / (1024 * 1024)
            log_with_timestamp(f"  → {paths['pickle']} ({size_mb:.1f}MB)")
            log_with_timestamp(f"[DEBUG] Retriever save complete")
            return True
    except Exception as e:
        log_with_timestamp(f"[ERROR] Error saving {retriever_name}: {type(e).__name__}: {e}")
        import traceback
        log_with_timestamp(traceback.format_exc())
        return False


def build_retriever(retriever_name: str, documents: List[Dict], doc_hash: str, cache_dir: str, all_metadata: Dict = None) -> Tuple[bool, str]:
    """Build and save retriever to cache. Returns (success, cache_path_or_error)"""
    log_with_timestamp(f"[DEBUG] build_retriever: Creating {retriever_name}...")
    try:
        if retriever_name == 'colbertv2':
            if all_metadata is None:
                raise ValueError("ColBERTv2 build requires all_metadata")
            output_root = build_colbertv2_retriever_index(documents, doc_hash, all_metadata)
            return True, output_root

        log_with_timestamp(f"[DEBUG] Creating retriever instance...")
        if retriever_name == 'bm25':
            retriever = retrievers.BM25()
            log_with_timestamp(f"[DEBUG] BM25 instance created")
        elif retriever_name == 'bge':
            retriever = retrievers.BGERetriever()
            log_with_timestamp(f"[DEBUG] BGERetriever instance created")
        elif retriever_name == 'e5':
            retriever = retrievers.E5Retriever()
            log_with_timestamp(f"[DEBUG] E5Retriever instance created")
        elif retriever_name == 'minilm':
            retriever = retrievers.MiniLMRetriever()
            log_with_timestamp(f"[DEBUG] MiniLMRetriever instance created")
        elif retriever_name == 'star':
            retriever = retrievers.STARRetriever()
            log_with_timestamp(f"[DEBUG] STARRetriever instance created")
        elif retriever_name == 'gritlm':
            retriever = retrievers.GritLMRetriever()
            log_with_timestamp(f"[DEBUG] GritLMRetriever instance created")
        elif retriever_name == 'ance':
            retriever = retrievers.ANCERetriever()
            log_with_timestamp(f"[DEBUG] ANCERetriever instance created")
        elif retriever_name == 'colbert':
            retriever = retrievers.ColBERTRetriever()
            log_with_timestamp(f"[DEBUG] ColBERTRetriever instance created")
        elif retriever_name == 'splade':
            retriever = retrievers.SPLADERetriever()
            log_with_timestamp(f"[DEBUG] SPLADERetriever instance created")
        else:
            return False, f"Unknown retriever type: {retriever_name}"
        
        log_with_timestamp(f"[DEBUG] Calling retriever.fit() on {len(documents)} documents...")
        retriever.fit(documents, all_metadata)
        log_with_timestamp(f"[DEBUG] retriever.fit() completed")
        
        log_with_timestamp(f"[DEBUG] Calling save_retriever_to_cache()...")
        if not save_retriever_to_cache(retriever_name, doc_hash, retriever, cache_dir):
            log_with_timestamp(f"[DEBUG] save_retriever_to_cache() returned False")
            return False, "Failed to save cache"
        
        log_with_timestamp(f"[DEBUG] save_retriever_to_cache() succeeded")
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
        cache_path = paths.get('config') or paths.get('pickle', '')
        return True, cache_path
    except Exception as e:
        import traceback
        log_with_timestamp(f"[ERROR] Exception in build_retriever: {type(e).__name__}: {e}")
        log_with_timestamp(traceback.format_exc())
        return False, f"{type(e).__name__}: {e}"


def main():
    """Main: Build all retriever indices"""
    setup_logging()
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("BUILD ALL RETRIEVER INDICES - STARTING")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"[DEBUG] Python version: {__import__('sys').version}")
    log_with_timestamp(f"[DEBUG] Current time: {__import__('datetime').datetime.now()}")
    
    category = CATEGORY_NAME
    log_with_timestamp(f"[DEBUG] Category: {category}")

    # Load full-scale metadata
    metadata_file = CAT_CONFIG['metadata_cache_file']
    log_with_timestamp(f"[DEBUG] Metadata file path: {metadata_file}")
    log_with_timestamp(f"[DEBUG] Metadata file exists: {os.path.exists(metadata_file)}")

    if os.path.exists(metadata_file):
        log_with_timestamp(f"[DEBUG] Loading metadata from cache...")
        try:
            with open(metadata_file, 'rb') as f:
                log_with_timestamp(f"[DEBUG] Opened metadata file successfully")
                metadata = pickle.load(f)
                log_with_timestamp(f"[DEBUG] Metadata loaded: {len(metadata)} items")
        except Exception as e:
            log_with_timestamp(f"[ERROR] Failed to load metadata: {type(e).__name__}: {e}")
            import traceback
            log_with_timestamp(traceback.format_exc())
            raise
    else:
        log_with_timestamp("[DEBUG] Metadata cache not found, loading from raw data...")
        raw_metadata_file = CAT_CONFIG['raw_corpus_file']
        log_with_timestamp(f"[DEBUG] Raw metadata file: {raw_metadata_file}")
        metadata = load_fullscale_metadata(raw_metadata_file)
    
    # Build documents
    log_with_timestamp(f"[DEBUG] Starting document building...")
    documents, asins = build_fullscale_documents(category, metadata)
    log_with_timestamp(f"[DEBUG] Document building complete")
    log_with_timestamp(f"Total documents: {len(documents)}, Total ASINs: {len(asins)}")
    
    # Compute document hash for cache keys
    log_with_timestamp(f"[DEBUG] Computing document hash...")
    doc_hash = compute_document_hash(documents)
    cache_dir = CAT_CONFIG['retriever_cache_dir']
    log_with_timestamp(f"[DEBUG] Creating cache directory...")
    os.makedirs(cache_dir, exist_ok=True)
    log_with_timestamp(f"Document hash: {doc_hash}")
    log_with_timestamp(f"Cache directory: {cache_dir}")
    
    # Define retrievers to build (按参数量从小到大排序: minilm < star < e5 < bge < gritlm < ance)
    DENSE_RETRIEVERS = [
        'minilm', 'star', 'e5', 'bge',
        # 'gritlm',
        'ance',
    ]
    COLBERT_RETRIEVERS = ['colbertv2']  # 使用 ColBERTv2 原生 late-interaction 索引
    SPARSE_RETRIEVERS = ['bm25', 'splade']
    ALL_RETRIEVERS = DENSE_RETRIEVERS + COLBERT_RETRIEVERS + SPARSE_RETRIEVERS

    log_with_timestamp(f"\nBuilding {len(ALL_RETRIEVERS)} retrievers:")
    log_with_timestamp(f"  Dense: {DENSE_RETRIEVERS}")
    log_with_timestamp(f"  ColBERT: {COLBERT_RETRIEVERS}")
    log_with_timestamp(f"  Sparse: {SPARSE_RETRIEVERS}")
    
    # Build each retriever
    log_with_timestamp(f"[DEBUG] Starting retriever build loop...")
    start_time = datetime.now()
    results = {}
    
    for retriever_name in ALL_RETRIEVERS:
        log_with_timestamp(f"[DEBUG] === Processing retriever: {retriever_name} ===")
        log_with_timestamp(f"\n[BUILD] {retriever_name}")
        log_with_timestamp(f"[DEBUG] Checking cache for {retriever_name}...")
        
        if cache_exists(retriever_name, doc_hash, cache_dir):
            log_with_timestamp(f"[CACHE_EXISTS] {retriever_name} cache already exists")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
            if retriever_name in DENSE_RETRIEVERS:
                log_with_timestamp(f"  → {paths['config']}")
                log_with_timestamp(f"  → {paths['embeddings']}")
            elif retriever_name in COLBERT_RETRIEVERS:
                log_with_timestamp(f"  → {paths['index_root']}")
                log_with_timestamp(f"  → {paths['index_path']}")
            else:
                log_with_timestamp(f"  → {paths['pickle']}")

            # 验证缓存完整性
            is_valid, error_msg = validate_retriever_cache(retriever_name, doc_hash, cache_dir, len(documents))
            if not is_valid:
                log_with_timestamp(f"[CACHE_INVALID] {retriever_name} cache validation failed: {error_msg}")
                log_with_timestamp(f"[CACHE_INVALID] Will rebuild {retriever_name}...")
                # 缓存无效，需要重建
                should_rebuild = True
            else:
                log_with_timestamp(f"[CACHE_VALID] {retriever_name} cache integrity verified ✓")
                results[retriever_name] = {'status': 'cached', 'time': 0}
                log_with_timestamp(f"[DEBUG] {retriever_name} skipped (cached)")
                should_rebuild = False
        else:
            should_rebuild = True

        if should_rebuild:
            # 在构建 ColBERT 之前，显式清理 GPU 缓存
            if retriever_name in COLBERT_RETRIEVERS and torch.cuda.is_available():
                log_with_timestamp(f"[MEMORY] ColBERT build requested, clearing GPU cache...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                log_with_timestamp(f"[MEMORY] GPU cache cleared")

            log_with_timestamp(f"[CACHE_NOT_FOUND] Building {retriever_name}...")
            log_with_timestamp(f"[DEBUG] Calling build_retriever({retriever_name})...")
            retriever_start = datetime.now()
            try:
                success, cache_path = build_retriever(retriever_name, documents, doc_hash, cache_dir, metadata)
                log_with_timestamp(f"[DEBUG] build_retriever returned: success={success}")
            except Exception as e:
                log_with_timestamp(f"[ERROR] Exception in build_retriever: {type(e).__name__}: {e}")
                import traceback
                log_with_timestamp(traceback.format_exc())
                success = False
                cache_path = str(e)
            
            elapsed = (datetime.now() - retriever_start).total_seconds()
            log_with_timestamp(f"[DEBUG] Elapsed time: {elapsed:.1f}s")
            
            if success:
                log_with_timestamp(f"[BUILD_SUCCESS] {retriever_name} built in {elapsed:.1f}s")
                paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
                if retriever_name in DENSE_RETRIEVERS:
                    log_with_timestamp(f"  → {paths['config']}")
                    log_with_timestamp(f"  → {paths['embeddings']}")
                elif retriever_name in COLBERT_RETRIEVERS:
                    log_with_timestamp(f"  → {paths['index_root']}")
                    log_with_timestamp(f"  → {paths['index_path']}")
                else:
                    log_with_timestamp(f"  → {paths['pickle']}")
                results[retriever_name] = {'status': 'success', 'time': elapsed}
            else:
                log_with_timestamp(f"[BUILD_FAILED] {retriever_name}: {cache_path}")
                results[retriever_name] = {'status': 'failed', 'error': cache_path}
                if retriever_name in COLBERT_RETRIEVERS:
                    raise RuntimeError(f"ColBERTv2 build failed: {cache_path}")
    
    # Summary
    log_with_timestamp(f"[DEBUG] Build loop complete, generating summary...")
    total_time = (datetime.now() - start_time).total_seconds()
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("BUILD SUMMARY")
    log_with_timestamp("=" * 80)
    
    for retriever_name, result in results.items():
        status = result['status']
        if status == 'success':
            time = result['time']
            log_with_timestamp(f"  ✓ {retriever_name:15} - Built in {time:7.1f}s")
        elif status == 'cached':
            log_with_timestamp(f"  ⚡ {retriever_name:15} - Already cached (skipped)")
        else:
            error = result['error']
            log_with_timestamp(f"  ✗ {retriever_name:15} - ERROR: {error}")
    
    log_with_timestamp(f"\nTotal time: {total_time:.1f}s")
    successful = sum(1 for r in results.values() if r['status'] in ['success', 'cached'])
    log_with_timestamp(f"Ready: {successful}/{len(ALL_RETRIEVERS)} (built + cached)")
    failed = sum(1 for r in results.values() if r['status'] == 'failed')
    if failed > 0:
        log_with_timestamp(f"Failed: {failed}/{len(ALL_RETRIEVERS)}")
    
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"[DEBUG] Main function complete")


def setup_logging():
    """Setup logging directory"""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)


if __name__ == '__main__':
    main()
    log_with_timestamp("当前任务已完成，请做下一个任务的指示。")
