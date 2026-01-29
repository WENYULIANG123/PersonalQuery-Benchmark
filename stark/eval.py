import argparse
import json
import os
import os.path as osp

try:
    print("DEBUG: Starting numpy import")
    import numpy as np
    print("DEBUG: Starting pandas import")
    import pandas as pd
    print("DEBUG: Starting torch import")
    import torch
    print("DEBUG: Starting tqdm import")
    from tqdm import tqdm

    print("DEBUG: Starting stark_qa imports")
    from stark_qa import load_qa, load_skb, load_model
    print("DEBUG: stark_qa main done")
    from stark_qa.load_qa import load_custom_qa_dataset
    print("DEBUG: load_custom_qa_dataset done")
    from stark_qa.tools.args import load_args, merge_args
    print("DEBUG: All imports successful")
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"FATAL ERROR: Import failed: {e}")
    print("This usually means the conda environment is not properly activated or torch is not installed.")
    exit(1)
except Exception as e:
    print(f"FATAL ERROR: Unexpected error during import: {e}")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset and model selection
    parser.add_argument("--dataset", default="amazon", choices=['amazon', 'prime', 'mag'])
    parser.add_argument("--model", default="VSS", choices=["BM25", "Colbertv2", "ColBERT", "GritLM", "VSS", "MultiVSS", "LLMReranker"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "test-0.1", "human_generated_eval", "variants"])
    parser.add_argument("--strategy", type=str, default="original", help="Strategy for variants evaluation")
    parser.add_argument("--categories", nargs="+", default=['Arts_Crafts_and_Sewing'], help="Categories to load for Amazon dataset")

    # Path settings
    parser.add_argument("--output_dir", type=str, default='output/')
    parser.add_argument("--download_dir", type=str, default='output/')
    parser.add_argument("--dataset_root", type=str, default=None, help="Custom dataset root directory")
    parser.add_argument("--csv_file", type=str, default=None, help="Direct CSV file path for evaluation (bypasses dataset_root)")
    parser.add_argument("--emb_dir", type=str, default='emb/')

    # Evaluation settings
    parser.add_argument("--test_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--max_queries", type=int, default=None, help="Maximum number of queries to evaluate (for testing/debugging)")

    # MultiVSS specific settings
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--multi_vss_topk", type=int, default=None)
    parser.add_argument("--aggregate", type=str, default="max")

    # VSS, MultiVSS, and LLMReranker settings
    parser.add_argument("--emb_model", type=str, default="text-embedding-ada-002")

    # LLMReranker specific settings
    parser.add_argument("--llm_model", type=str, default="gpt-4-1106-preview", help='the LLM to rerank candidates.')
    parser.add_argument("--llm_topk", type=int, default=10)
    parser.add_argument("--max_retry", type=int, default=3)

    # Prediction saving settings
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--save_topk", type=int, default=500, help="topk predicted indices to save")

    # load the embeddings stored under folder f'doc{surfix}' or f'query{surfix}', e.g., _no_compact,
    parser.add_argument("--surfix", type=str, default='')

    # Force rerun even if results exist
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun evaluation even if results already exist")

    return parser.parse_args()


if __name__ == "__main__":
    print("DEBUG: eval.py started")
    args = parse_args()
    print(f"DEBUG: args parsed, model={args.model}, dataset={args.dataset}")
    default_args = load_args(
        json.load(open("config/default_args.json", "r"))[args.dataset]
    )
    args = merge_args(args, default_args)

    # Set query embedding directory (skip for ColBERT/ColBERTv2 models)
    if args.model not in ['ColBERT', 'Colbertv2']:
        query_emb_surfix = f'_{args.split}' if args.split == 'human_generated_eval' else ''
        if args.split == 'variants' and args.strategy:
            # Try strategy-specific embeddings first, then fall back to full variants embeddings
            strategy_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, f"query_no_rel_no_compact_stark_{args.strategy}_variants_dataset")
            full_variants_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, f"query_no_rel_no_compact_stark_variants_dataset")
            if osp.exists(osp.join(strategy_emb_dir, "query_emb_dict.pt")):
                args.query_emb_dir = strategy_emb_dir
            elif osp.exists(osp.join(full_variants_emb_dir, "query_emb_dict.pt")):
                args.query_emb_dir = full_variants_emb_dir
            else:
                query_emb_surfix = f'_query_stark_{args.strategy}_variants_dataset'
                args.query_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, f"query{query_emb_surfix}{args.surfix}")
        else:
            args.query_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, f"query{query_emb_surfix}{args.surfix}")

    # Set document embedding directories (skip for ColBERT/ColBERTv2 models)
    if args.model not in ['ColBERT', 'Colbertv2']:
        # For DPR and ANCE models, use the default document embedding directory with no_rel_no_compact suffix
        if args.emb_model.startswith(('facebook/dpr', 'sentence-transformers/facebook-dpr', 'castorini/ance', 'alibaba-nlp')):
            args.node_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, "doc_no_rel_no_compact")
        else:
            args.node_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, f"doc{args.surfix}")

        args.chunk_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, f"chunk{args.surfix}")
    else:
        # For ColBERT/ColBERTv2, set dummy values to avoid AttributeError
        args.node_emb_dir = ""
        args.chunk_emb_dir = ""
        args.query_emb_dir = ""

    # Use LLMbasedeval directory for LLM-based embeddings
    if args.model in ['VSS', 'MultiVSS'] and args.emb_model.startswith(('alibaba-nlp', 'text-embedding', 'voyage', 'GritLM', 'McGill-NLP')):
        output_dir = osp.join("LLMbasedeval", args.dataset, args.model, args.emb_model)
    elif args.model.lower() == 'colbertv2':
        # Use custom Colbertv2eval directory for ColBERTv2
        output_dir = "/home/wlia0047/ar57/wenyu/stark/Colbertv2eval"
    else:
        output_dir = osp.join(args.output_dir, "eval", args.dataset, args.model)
        if args.model == 'LLMReranker':
            output_dir = osp.join(output_dir, args.llm_model)
        elif args.model in ['VSS', 'MultiVSS', 'GritLM', 'ColBERT']:
            output_dir = osp.join(output_dir, args.emb_model)
    args.output_dir = output_dir
    print(f"DEBUG: Final output_dir set to: {args.output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Create embedding directories only for models that need them
    if args.model not in ['ColBERT', 'Colbertv2']:
        os.makedirs(args.query_emb_dir, exist_ok=True)
        os.makedirs(args.chunk_emb_dir, exist_ok=True)
        os.makedirs(args.node_emb_dir, exist_ok=True)
    json.dump(vars(args), open(osp.join(output_dir, "args.json"), "w"), indent=4)

    # Generate filename with strategy info for variants
    split_name = args.split
    if args.split == 'variants' and args.strategy:
        split_name = f"{args.split}_{args.strategy}"
    elif args.split == 'variants' and not args.strategy:
        # If variants split but no strategy specified, this is an error
        raise ValueError("variants split requires a strategy to be specified")

    print(f"📁 Generated split_name: {split_name} (split={args.split}, strategy={args.strategy})")

    eval_csv_path = osp.join(output_dir, f"eval_results_{split_name}.csv")
    final_eval_path = (
        osp.join(output_dir, f"eval_metrics_{split_name}.json")
        if args.test_ratio == 1.0
        else osp.join(output_dir, f"eval_metrics_{split_name}_{args.test_ratio}.json")
    )

    print(f"📄 Results will be saved to: {eval_csv_path}")
    print(f"📊 Metrics will be saved to: {final_eval_path}")

    skb = load_skb(args.dataset, root=args.dataset_root, categories=args.categories)

    # Handle different split types
    if args.split == 'variants':
        # For variants, we need to load custom dataset
        if args.csv_file is not None:
            # Use direct CSV file - no dataset_root needed
            csv_path = args.csv_file
        elif args.dataset_root is None:
            raise ValueError("dataset_root must be specified for variants split (or use --csv_file)")
        else:
            import os.path as osp
            csv_path = osp.join(args.dataset_root, "qa", "amazon", "stark_qa", "stark_qa.csv")

        if args.strategy is None and args.csv_file is None:
            raise ValueError("strategy must be specified for variants split (unless using --csv_file)")

        # Load the dataset
        df = pd.read_csv(csv_path)

        # Handle answer_ids_source column name if present
        if 'answer_ids_source' in df.columns and 'answer_ids' not in df.columns:
            df['answer_ids'] = df['answer_ids_source']
            print("📝 Renamed 'answer_ids_source' to 'answer_ids'")

        # For strategy-specific datasets, no need to filter by attack_strategy
        # The dataset should already contain only queries for the specified strategy
        strategy_info = args.strategy or "unknown (using CSV file)"
        print(f"🔍 Using strategy-specific dataset for strategy: {strategy_info}")
        print(f"📊 Loaded {len(df)} queries for strategy '{strategy_info}'")

        qa_dataset = load_custom_qa_dataset(None, df)
    else:
        # If dataset_root is not specified, use Hugging Face cache path
        if args.dataset_root is None:
            import os
            hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            # Use the known cache path for STaRK dataset
            args.dataset_root = os.path.join(hf_cache, 'hub', 'datasets--snap-stanford--stark', 'snapshots', '88269e23e90587f99476c5dd74e235a0877e69be', 'qa')
            print(f"Using dataset root from HF cache: {args.dataset_root}")

        if args.csv_file is not None:
            # Load directly from CSV file (no dataset creation needed)
            qa_dataset = load_custom_qa_dataset(args.csv_file)
            print(f"Loaded dataset directly from CSV: {args.csv_file}")
        else:
            qa_dataset = load_qa(args.dataset, args.dataset_root, human_generated_eval=args.split == 'human_generated_eval')

    # Generate embeddings if using GritLM or ColBERT and they don't exist
    print(f"DEBUG: Processing model type: {args.model}")
    if args.model == 'GritLM':
        print("Checking GritLM embeddings...")
        from generate_gritlm_embeddings import generate_document_embeddings, generate_query_embeddings

        # Check if document embeddings exist
        doc_emb_path = osp.join(args.node_emb_dir, 'candidate_emb_dict.pt')
        if not osp.exists(doc_emb_path):
            print("Document embeddings not found, generating...")
            generate_document_embeddings(args.dataset, args.emb_model, args.device)

        # Check if query embeddings exist
        query_emb_path = osp.join(args.query_emb_dir, 'query_emb_dict.pt')
        if not osp.exists(query_emb_path):
            print("Query embeddings not found, generating...")
            generate_query_embeddings(args.dataset, args.split, args.emb_model, args.device)

        print("GritLM embeddings ready!")

    elif args.model in ['ColBERT', 'Colbertv2']:
        print(f"Checking {args.model} embeddings...")
        # Note: ColBERT/ColBERTv2 embeddings are generated on-demand in the model
        # No separate generation step needed
        print(f"{args.model} embeddings will be generated on-demand!")
    else:
        print(f"DEBUG: No special embedding handling needed for model {args.model}")

    print("DEBUG: About to call load_model...")
    print(f"DEBUG: Model type: {args.model}, Device: {args.device}")
    model = load_model(args, skb)
    print("DEBUG: load_model completed successfully")

    # For ColBERTv2, set the current strategy for dynamic score_dict updates
    if args.model == 'Colbertv2':
        current_strategy = args.strategy if args.strategy else 'original'
        if args.split == 'variants' and args.strategy:
            current_strategy = args.strategy
        elif args.split == 'human_generated_eval':
            current_strategy = 'original'
        model._current_strategy = current_strategy
        print(f"🔄 Set ColBERTv2 strategy to: {current_strategy}")

    split_idx = qa_dataset.get_idx_split(test_ratio=args.test_ratio)

    eval_metrics = [
        "mrr",
        "map",
        "rprecision",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@50",
        "recall@100",
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "hit@20",
        "hit@50",
    ]
    eval_csv = pd.DataFrame(columns=["idx", "query_id", "pred_rank"] + eval_metrics)

    existing_idx = []
    if osp.exists(eval_csv_path) and not args.force_rerun:
        eval_csv = pd.read_csv(eval_csv_path)
        existing_idx = eval_csv["idx"].tolist()
        print(f"📁 Found existing results file with {len(existing_idx)} processed queries")
    elif args.force_rerun:
        print(f"🔄 Force rerun enabled - will reprocess all queries regardless of existing results")
        # Remove existing file to start fresh
        if osp.exists(eval_csv_path):
            os.remove(eval_csv_path)
            print(f"🗑️  Removed existing results file: {eval_csv_path}")
    else:
        print("📝 No existing results file found - processing all queries")

    all_indices = split_idx[args.split].tolist()
    indices = list(set(all_indices) - set(existing_idx))

    print(f"📊 Processing {len(indices)} queries...")
    processed_count = 0

    if args.batch_size > 0 and args.model == 'VSS':
        for batch_idx in tqdm(range(0, len(indices), args.batch_size or len(indices))):
            batch_indices = [idx for idx in indices[batch_idx : min(batch_idx + args.batch_size, len(indices))] if idx not in existing_idx]
            if len(batch_indices) == 0:
                continue
            queries, query_ids, answer_ids, meta_infos = zip(
                *[qa_dataset[idx] for idx in batch_indices]
            )
            pred_ids, pred = model.forward(list(queries), list(query_ids))

            answer_ids = [torch.LongTensor(answer_id) for answer_id in answer_ids]
            results = model.evaluate_batch(pred_ids, pred, answer_ids, metrics=eval_metrics)

            for i, result in enumerate(results):
                result["idx"], result["query_id"] = batch_indices[i], query_ids[i]
                result["pred_rank"] = pred_ids[torch.argsort(pred[:,i], descending=True)[:args.save_topk]].tolist()
                eval_csv = pd.concat([eval_csv, pd.DataFrame([result])], ignore_index=True)
                processed_count += 1

                # Print progress every 10 queries
                if processed_count % 10 == 0:
                    print(f"📈 Processed {processed_count}/{len(indices)} queries...")
    else:
        for idx in tqdm(indices):
            query, query_id, answer_ids, meta_info = qa_dataset[idx]
            pred_dict = model.forward(query, query_id)

            answer_ids = torch.LongTensor(answer_ids)
            result = model.evaluate(pred_dict, answer_ids, metrics=eval_metrics)

            result["idx"], result["query_id"] = idx, query_id
            result["pred_rank"] = torch.LongTensor(list(pred_dict.keys()))[
                torch.argsort(torch.tensor(list(pred_dict.values())), descending=True)[
                    :args.save_topk
                ]
            ].tolist()

            eval_csv = pd.concat([eval_csv, pd.DataFrame([result])], ignore_index=True)
            processed_count += 1

            # Print progress every 10 queries
            if processed_count % 10 == 0:
                print(f"📈 Processed {processed_count}/{len(indices)} queries...")

            if args.save_pred:
                eval_csv.to_csv(eval_csv_path, index=False)
            for metric in eval_metrics:
                print(
                    f"{metric}: {np.mean(eval_csv[eval_csv['idx'].isin(indices)][metric])}"
                )
    if args.save_pred:
        eval_csv.to_csv(eval_csv_path, index=False)
    final_metrics = (
        eval_csv[eval_metrics].mean().to_dict()
    )
    json.dump(final_metrics, open(final_eval_path, "w"), indent=4)

    # Print save paths
    print(f"Results saved to: {eval_csv_path}")
    print(f"Metrics saved to: {final_eval_path}")