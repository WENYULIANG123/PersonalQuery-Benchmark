#!/usr/bin/env python3
"""
STaRK ANCE Dense Retrieval Evaluation Script
=============================================

This script runs dense retrieval evaluation on the STaRK Amazon dataset using ANCE.
Uses ANCE model (castorini/ance-msmarco-passage): unified encoder for queries and documents.
Based on RoBERTa-base with ANN negative sampling for improved retrieval performance!
"""

import os
import sys
import subprocess
from datetime import datetime
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os.path as osp
from tqdm import tqdm




def create_strategy_dataset(strategy_name):
    """Create STaRK-compatible dataset for a specific strategy."""
    import pandas as pd

    if strategy_name == 'original':
        return None


    variants_file = f"/home/wlia0047/ar57/wenyu/stark/data/strategy_variants/{strategy_name}_variants_81.csv"
    stark_base_dir = f"/home/wlia0047/ar57/wenyu/stark/data/stark_strategy_{strategy_name}_dataset"

    os.makedirs(stark_base_dir, exist_ok=True)

    if not os.path.exists(variants_file):
        raise FileNotFoundError(f"Variants file not found: {variants_file}")

    df = pd.read_csv(variants_file)

    # All variant files now use the new format: id, query, answer_ids, answer_ids_source
    stark_df = pd.DataFrame({
        'id': range(len(df)),
        'query': df['query'],
        'answer_ids': df['answer_ids'],
        'query_type': ['variant'] * len(df)
    })

    # Create STaRK directory structure
    qa_dir = os.path.join(stark_base_dir, "qa", "amazon")
    split_dir = os.path.join(qa_dir, "split")
    stark_qa_dir = os.path.join(qa_dir, "stark_qa")

    os.makedirs(stark_qa_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    stark_qa_file = os.path.join(stark_qa_dir, "stark_qa.csv")
    stark_df.to_csv(stark_qa_file, index=False)

    split_file = os.path.join(split_dir, "variants.index")
    with open(split_file, 'w') as f:
        for idx in stark_df['id']:
            f.write(f"{idx}\n")

    return stark_base_dir


def load_ance_encoder():
    """Load ANCE unified encoder for encoding queries and documents."""
    print("Loading ANCE unified encoder (castorini/ance-msmarco-passage)...")
    model_name = 'castorini/ance-msmarco-passage'
    encoder = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder.eval()
    print(f"✅ ANCE model loaded successfully (RoBERTa-base backbone)")
    return encoder, tokenizer







def generate_ance_embeddings_stark_standard(dataset="amazon", emb_model="castorini/ance-msmarco-passage"):
    """Generate ANCE embeddings using STaRK's standard emb_generate.py script."""
    import subprocess
    import os.path as osp

    # Check if embeddings already exist
    # emb_generate.py generates embeddings in doc_no_rel_no_compact directory by default
    emb_dir = osp.join("emb", dataset, emb_model, "doc_no_rel_no_compact")
    emb_path = osp.join(emb_dir, "candidate_emb_dict.pt")

    if osp.exists(emb_path):
        print(f"✅ ANCE embeddings already exist at {emb_path}")
        print("🎉 Skipping embedding generation, using cached embeddings!")
        return True

    print(f"🔄 ANCE embeddings not found at {emb_path}")
    print("🚀 Generating ANCE embeddings using STaRK's standard emb_generate.py...")

    # Call STaRK's standard embedding generation script
    cmd = [
        "python", "emb_generate.py",
        "--dataset", dataset,
        "--emb_model", emb_model,
        "--mode", "doc"
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        print("🔄 Starting DPR embedding generation (output will be displayed in real-time)...")
        result = subprocess.run(cmd, cwd=".")
        if result.returncode == 0:
            print("✅ DPR embedding generation completed successfully!")
            if osp.exists(emb_path):
                print(f"📁 Embeddings saved to: {emb_path}")
                return True
            else:
                print(f"❌ Expected embedding file not found at {emb_path}")
                return False
        else:
            print(f"❌ DPR embedding generation failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Error running DPR embedding generation: {e}")
        return False









def encode_query(query_text, encoder, tokenizer):
    """Encode a single query using ANCE encoder with GPU acceleration."""
    device = next(encoder.parameters()).device  # Get the device the model is on

    inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # For ANCE (RoBERTa-based), use pooler_output like DPR
        outputs = encoder(**inputs)
        embedding = outputs.pooler_output.cpu().numpy()
    return embedding


def generate_ance_query_embeddings(encoder, tokenizer, dataset, split, strategy=None, dataset_root=None):
    """Generate ANCE query embeddings compatible with eval.py format."""
    import os.path as osp
    from stark_qa import load_qa

    print("🔍 Loading QA dataset for query embedding generation...")
    if split == 'variants' and dataset_root:
        # Load custom variants or error_aware dataset
        import pandas as pd
        csv_path = osp.join(dataset_root, "qa", "amazon", "stark_qa", "stark_qa.csv")
        qa_dataset = load_custom_qa_dataset(csv_path)
    else:
        qa_dataset = load_qa(dataset, dataset_root, human_generated_eval=split == 'human_generated_eval')

    # Create query embedding directory (compatible with eval.py)
    query_emb_dir = osp.join("emb", dataset, "castorini/ance-msmarco-passage")
    if split == 'variants' and strategy in ['error_aware', 'grammar_aware']:
        query_emb_dir = osp.join(query_emb_dir, f"query_no_rel_no_compact_stark_{strategy}_variants_dataset")
    elif split == 'variants' and strategy:
        query_emb_dir = osp.join(query_emb_dir, f"query_no_rel_no_compact_stark_{strategy}_variants_dataset")
    else:
        query_emb_dir = osp.join(query_emb_dir, f"query{('_' + split) if split == 'human_generated_eval' else ''}")

    os.makedirs(query_emb_dir, exist_ok=True)
    query_emb_path = osp.join(query_emb_dir, "query_emb_dict.pt")

    # Always regenerate embeddings (force_rerun mode)
    if osp.exists(query_emb_path):
        print(f"🔄 Force regenerating embeddings (overwriting existing file at {query_emb_path})")
    else:
        print(f"🔄 Generating ANCE query embeddings for {len(qa_dataset)} queries...")

    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    encoder.eval()

    query_emb_dict = {}

    for idx in tqdm(range(len(qa_dataset)), desc="Encoding queries with ANCE"):
        query, query_id, _, _ = qa_dataset[idx]

        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # ANCE uses RoBERTa backbone, use pooler_output for sentence embedding
            outputs = encoder(**inputs)
            embedding = outputs.pooler_output.cpu()

        query_emb_dict[query_id] = embedding.view(1, -1)

    # Save embeddings
    torch.save(query_emb_dict, query_emb_path)
    print(f"✅ Saved {len(query_emb_dict)} query embeddings to {query_emb_path}")

    return query_emb_dir


def load_custom_qa_dataset(csv_path):
    """Load QA dataset directly from CSV file."""
    import pandas as pd
    import ast

    df = pd.read_csv(csv_path)

    class CustomQADataset:
        def __init__(self, dataframe):
            self.data = []
            for idx, row in dataframe.iterrows():
                answers = ast.literal_eval(row['answer_ids']) if isinstance(row['answer_ids'], str) else row['answer_ids']
                self.data.append((row['query'], int(row['id']), answers, None))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        def get_idx_split(self, test_ratio=1.0):
            import torch
            total = len(self.data)
            indices = torch.tensor(list(range(total)))
            return {'variants': indices, 'test': indices, 'val': indices, 'train': indices}

    return CustomQADataset(df)




def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='STaRK DPR Evaluation for Query Variants')
    parser.add_argument('--strategy', type=str, default='all',
                       choices=['original', 'character', 'embedding', 'other', 'typo', 'wordnet', 'error_aware', 'grammar_aware', 'all'],
                       help='Attack strategy to evaluate (default: all)')
    args = parser.parse_args()

    stark_root = "/home/wlia0047/ar57/wenyu/stark"
    os.chdir(stark_root)

    strategies = ['original', 'character', 'embedding', 'other', 'typo', 'wordnet', 'error_aware', 'grammar_aware'] if args.strategy == 'all' else [args.strategy]

    dataset = "amazon"
    split = "variants"
    save_topk = None

    print(f"EVALUATING {len(strategies)} STRATEGIES WITH ANCE (excluding dependency): {', '.join(strategies)}")
    print("=" * 80)

    total_start_time = datetime.now()

    # Load ANCE unified encoder for query and document embedding generation
    print("🤖 Loading ANCE unified encoder for query embeddings...")
    encoder, tokenizer = load_ance_encoder()

    # Move model to GPU if available and enable optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for ANCE model: {device}")

    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ Enabled CUDA optimizations (TF32, cuDNN benchmark)")

    encoder = encoder.to(device)
    encoder.eval()

    # Generate ANCE document embeddings using STaRK's standard approach
    print("📚 Generating ANCE document embeddings using STaRK's standard emb_generate.py...")

    success = generate_ance_embeddings_stark_standard(dataset, "castorini/ance-msmarco-passage")
    if not success:
        print("❌ ANCE嵌入生成失败，退出程序")
        return

    print("✅ ANCE document embeddings ready!")

    # Set evaluation parameters
    stark_root = "/home/wlia0047/ar57/wenyu/stark"
    save_topk = None

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy.upper()} (USING eval.py)")
        print(f"{'='*60}")

        try:
            start_time = datetime.now()

            if strategy == 'original':
                dataset_root = None
                split = "human_generated_eval"
            elif strategy in ['error_aware', 'grammar_aware']:
                dataset_root = create_strategy_dataset(strategy)
                split = "variants"
            else:
                dataset_root = create_strategy_dataset(strategy)
                split = "variants"

            print(f"ANCE EVALUATION PARAMETERS - {strategy}")
            print("-" * 40)
            print(f"Dataset: {dataset}")
            print(f"Model: ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation)")
            print(f"Split: {split}")
            print(f"Dataset root: {dataset_root}")
            print(f"Embedding model: castorini/ance-msmarco-passage")
            print(f"Top-K predictions: {save_topk if save_topk is not None else 'All (default 500)'}")
            print("-" * 40)
            print("PERFORMANCE NOTES:")
            print("- Using standard eval.py for ANCE evaluation")
            print("- ANCE embeddings pre-computed and cached")
            print("- Unified encoder: RoBERTa-base backbone with ANN negative sampling")
            print("- Same encoder for queries and documents")
            print("- Dense retrieval with cosine similarity")
            if strategy == 'error_aware':
                print("- 81 error-aware queries - CUSTOM DATASET LOADING")
            elif strategy == 'grammar_aware':
                print("- 81 grammar-aware queries - CUSTOM DATASET LOADING")
            elif strategy == 'original':
                print("- 81 original queries - STANDARD STaRK LOADING")
            else:
                print("- 81 variant queries per strategy - FORCED LOCAL LOADING")
            print("- Expected runtime: ~2-5 minutes per strategy")
            print("-" * 40)

            # 生成查询嵌入（如果不存在）
            query_emb_dir = generate_ance_query_embeddings(encoder, tokenizer, dataset, split, strategy, dataset_root)

            # 使用标准的 eval.py 进行 ANCE 评估
            result = run_ance_evaluation_with_eval_py(dataset, split, dataset_root, "ANCEeval", save_topk, stark_root, strategy, query_emb_dir)

            end_time = datetime.now()
            duration = end_time - start_time

            print("=" * 60)
            print(f"✓ {strategy.upper()} ANCE EVALUATION COMPLETED!")
            print(f"Duration: {duration}")
            print("=" * 60)

        except Exception as e:
            print(f"✗ {strategy.upper()} ANCE EVALUATION FAILED!")
            print(f"Error: {e}")

    total_end_time = datetime.now()
    total_duration = total_end_time - total_start_time

    print(f"\n{'='*80}")
    print(f"🎉 ALL {len(strategies)} STRATEGIES ANCE EVALUATION COMPLETED (excluded dependency)!")
    print(f"Total Duration: {total_duration}")
    print(f"Strategies evaluated: {', '.join(strategies)}")
    print(f"{'='*80}")


def run_ance_evaluation_with_eval_py(dataset, split, dataset_root, output_dir, save_topk, stark_root, strategy, query_emb_dir):
    """Run ANCE evaluation using the standard eval.py script."""
    cmd = [
        "python", "eval.py",
        "--dataset", dataset,
        "--model", "VSS",  # 使用 VSS 模型类型来加载 ANCE 嵌入
        "--split", split,
        "--output_dir", output_dir,
        "--emb_model", "castorini/ance-msmarco-passage",
        "--force_rerun"
    ]

    if save_topk is not None:
        cmd.extend(["--save_topk", str(save_topk)])

    if dataset_root:
        cmd.extend(["--dataset_root", dataset_root])

    if strategy != 'original':
        cmd.extend(["--strategy", strategy])

    print(f"Running eval command: {' '.join(cmd)}")
    print("🔄 Starting evaluation (output displayed in real-time)...")
    result = subprocess.run(cmd, cwd=stark_root, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ ANCE eval failed for strategy {strategy} with return code {result.returncode}")
        # Try to run again with error capture to see the actual error
        print("🔍 Capturing detailed error information...")
        error_result = subprocess.run(cmd, cwd=stark_root, capture_output=True, text=True)
        print("STDERR output:")
        print(error_result.stderr)
        print("STDOUT output:")
        print(error_result.stdout)
        raise Exception(f"ANCE evaluation failed for strategy {strategy}")
    else:
        print(f"✅ ANCE eval completed for strategy {strategy}")


if __name__ == "__main__":
    main()
