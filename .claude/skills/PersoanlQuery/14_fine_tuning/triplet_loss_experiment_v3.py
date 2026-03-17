#!/usr/bin/env python3

import json
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    'model_name': 'intfloat/e5-base-v2',
    'loocv_data_dir': '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data',
    'meta_path': '/fs04/ar57/wenyu/data/Amazon-Reviews-2018/intermediate/df_ucsd_meta.pkl',
    'output_dir': '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results_triplet',
    'batch_size': 16,
    'epochs': 3,
    'learning_rate': 2e-5,
}

def load_product_metadata(meta_path: str) -> Dict[str, str]:
    logger.info(f"Loading metadata from {meta_path}...")
    with open(meta_path, 'rb') as f:
        df_meta = pickle.load(f)
    
    asin_to_title = {}
    for idx, row in df_meta.iterrows():
        asin = str(row['asin'])
        title = row['title'] if pd.notna(row['title']) else ""
        asin_to_title[asin] = title
    
    logger.info(f"Loaded {len(asin_to_title)} products")
    return asin_to_title


def load_loocv_data(user_dir: str) -> Dict:
    with open(f'{user_dir}/holdout.json', 'r') as f:
        holdout = json.load(f)
    
    with open(f'{user_dir}/personal_train.json', 'r') as f:
        personal_train = json.load(f)
    
    with open(f'{user_dir}/global_train.json', 'r') as f:
        global_train = json.load(f)
    
    return {
        'holdout': holdout['pairs'],
        'personal_train': personal_train['pairs'],
        'global_train': global_train['pairs']
    }


def build_triplet_dataset(
    pairs: List[Dict],
    asin_to_title: Dict[str, str],
    all_asins: List[str],
    negative_strategy: str = 'random'
) -> List[InputExample]:
    triplets = []
    
    for pair in pairs:
        noisy_query = pair['positive']
        positive_asin = pair['asin']
        
        if positive_asin not in asin_to_title:
            continue
        
        positive_title = asin_to_title[positive_asin]
        if not positive_title:
            continue
        
        if negative_strategy == 'random':
            candidates = [a for a in all_asins if a != positive_asin]
            if not candidates:
                continue
            negative_asin = random.choice(candidates)
        
        negative_title = asin_to_title.get(negative_asin, "")
        if not negative_title:
            continue
        
        triplets.append(InputExample(
            texts=[noisy_query, positive_title, negative_title]
        ))
    
    return triplets


def train_model_simple(
    triplets: List[InputExample],
    checkpoint_dir: str,
    config: Dict
):
    logger.info(f"Building dataset with {len(triplets)} triplets...")
    
    from sentence_transformers.datasets import SentencesDataset
    
    logger.info(f"Loading model {config['model_name']}...")
    model = SentenceTransformer(config['model_name'])
    
    dataset = SentencesDataset(triplets, model)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    loss_func = losses.TripletLoss(model=model, triplet_margin=0.5)
    
    logger.info("Starting training...")
    model.fit(
        train_objectives=[(dataloader, loss_func)],
        epochs=config['epochs'],
        warmup_steps=0,
        output_path=checkpoint_dir,
        show_progress_bar=True,
        save_best_model=False,
        optimizer_params={'lr': config['learning_rate']}
    )
    
    logger.info(f"Model saved to {checkpoint_dir}")
    return model


def evaluate_model(model, holdout_pairs: List[Dict], asin_to_title: Dict[str, str]) -> Dict:
    from sklearn.metrics import ndcg_score
    
    corpus = list(asin_to_title.values())
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    
    mrr_scores = []
    ndcg_scores = []
    
    for pair in holdout_pairs:
        noisy_query = pair['positive']
        positive_asin = pair['asin']
        
        if positive_asin not in asin_to_title:
            continue
        
        query_embedding = model.encode(noisy_query, convert_to_tensor=True)
        
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            corpus_embeddings
        ).cpu().numpy()
        
        ranked_indices = np.argsort(-similarities)
        
        positive_rank = None
        for rank, idx in enumerate(ranked_indices):
            if corpus[idx] == asin_to_title[positive_asin]:
                positive_rank = rank
                break
        
        if positive_rank is not None:
            if positive_rank < 10:
                mrr_scores.append(1.0 / (positive_rank + 1))
            
            y_true = np.zeros(10)
            if positive_rank < 10:
                y_true[positive_rank] = 1
            y_score = similarities[ranked_indices[:10]]
            ndcg = ndcg_score([y_true], [y_score])
            ndcg_scores.append(ndcg)
    
    return {
        'mrr@10': np.mean(mrr_scores) if mrr_scores else 0.0,
        'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'n_evaluated': len(mrr_scores)
    }


def main():
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    asin_to_title = load_product_metadata(CONFIG['meta_path'])
    
    loocv_dir = Path(CONFIG['loocv_data_dir'])
    users = sorted([d.name for d in loocv_dir.iterdir() if d.is_dir()])
    
    logger.info(f"Found {len(users)} users")
    
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT A: Cross-user mixed training")
    logger.info("="*60)
    
    all_train_pairs = []
    
    for user in users:
        user_data = load_loocv_data(str(loocv_dir / user))
        all_train_pairs.extend(user_data['global_train'])
    
    logger.info(f"Total training pairs (cross-user): {len(all_train_pairs)}")
    
    all_asins = list(asin_to_title.keys())
    
    triplets_a = build_triplet_dataset(all_train_pairs, asin_to_title, all_asins)
    logger.info(f"Built {len(triplets_a)} triplets")
    
    model_a = None
    if triplets_a:
        checkpoint_a = str(output_dir / 'experiment_a_triplet')
        try:
            model_a = train_model_simple(triplets_a, checkpoint_a, CONFIG)
        except Exception as e:
            logger.error(f"Exp A training failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT B: Single-user personalized training")
    logger.info("="*60)
    
    holdout_user = 'user_A3E5V5TSTAY3R9'
    user_data_b = load_loocv_data(str(loocv_dir / holdout_user))
    personal_train = user_data_b['personal_train']
    holdout = user_data_b['holdout']
    
    logger.info(f"Personal training pairs: {len(personal_train)}")
    
    triplets_b = build_triplet_dataset(personal_train, asin_to_title, all_asins)
    logger.info(f"Built {len(triplets_b)} triplets")
    
    model_b = None
    if triplets_b:
        checkpoint_b = str(output_dir / 'experiment_b_triplet')
        try:
            model_b = train_model_simple(triplets_b, checkpoint_b, CONFIG)
        except Exception as e:
            logger.error(f"Exp B training failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION")
    logger.info("="*60)
    
    results = {}
    
    if model_a is not None:
        logger.info("\nEvaluating Exp A on holdout...")
        try:
            model_a_eval = SentenceTransformer(checkpoint_a)
            results['exp_a'] = evaluate_model(model_a_eval, holdout, asin_to_title)
            logger.info(f"Exp A: {results['exp_a']}")
        except Exception as e:
            logger.error(f"Exp A eval failed: {e}")
    
    if model_b is not None:
        logger.info("\nEvaluating Exp B on holdout...")
        try:
            model_b_eval = SentenceTransformer(checkpoint_b)
            results['exp_b'] = evaluate_model(model_b_eval, holdout, asin_to_title)
            logger.info(f"Exp B: {results['exp_b']}")
        except Exception as e:
            logger.error(f"Exp B eval failed: {e}")
    
    logger.info("\nEvaluating pretrained baseline...")
    model_pretrained = SentenceTransformer(CONFIG['model_name'])
    results['pretrained'] = evaluate_model(model_pretrained, holdout, asin_to_title)
    logger.info(f"Pretrained: {results['pretrained']}")
    
    results_file = output_dir / 'results_triplet.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    logger.info("\nSummary:")
    for exp, metrics in results.items():
        logger.info(f"  {exp}: MRR@10={metrics['mrr@10']:.4f}, NDCG@10={metrics['ndcg@10']:.4f}")


if __name__ == '__main__':
    main()
