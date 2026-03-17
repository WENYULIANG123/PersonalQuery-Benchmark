#!/usr/bin/env python3

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    'model_name': 'intfloat/e5-base-v2',
    'loocv_data_dir': '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data',
    'meta_path': '/fs04/ar57/wenyu/data/Amazon-Reviews-2018/intermediate/df_ucsd_meta.pkl',
    'output_dir': '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/experiment_4_transfer',
    'batch_size': 16,
    'epochs': 2,
}

def load_product_metadata(meta_path: str) -> Dict[str, str]:
    logger.info(f"Loading metadata...")
    with open(meta_path, 'rb') as f:
        df_meta = pickle.load(f)
    
    asin_to_title = {}
    for idx, row in df_meta.iterrows():
        asin = str(row['asin'])
        title = row['title'] if pd.notna(row['title']) else ""
        asin_to_title[asin] = title
    
    return asin_to_title


def load_user_data(user_dir: str) -> Dict:
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


def build_triplet_examples(pairs: List[Dict], asin_to_title: Dict[str, str], all_asins: List[str]) -> List[InputExample]:
    triplets = []
    
    for pair in pairs:
        noisy_query = pair['positive']
        positive_asin = pair['asin']
        
        if positive_asin not in asin_to_title:
            continue
        
        positive_title = asin_to_title[positive_asin]
        if not positive_title:
            continue
        
        candidates = [a for a in all_asins if a != positive_asin]
        if not candidates:
            continue
        
        negative_asin = random.choice(candidates)
        negative_title = asin_to_title.get(negative_asin, "")
        if not negative_title:
            continue
        
        triplets.append(InputExample(texts=[noisy_query, positive_title, negative_title]))
    
    return triplets


def train_user_model(triplets: List[InputExample], output_dir: str, config: Dict):
    if not triplets or len(triplets) < 10:
        logger.warning(f"  Not enough triplets ({len(triplets)}), skipping training")
        return None
    
    logger.info(f"  Training on {len(triplets)} triplets...")
    
    from sentence_transformers.datasets import SentencesDataset
    
    model = SentenceTransformer(config['model_name'])
    dataset = SentencesDataset(triplets, model)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    loss_func = losses.TripletLoss(model=model, triplet_margin=0.5)
    
    try:
        model.fit(
            train_objectives=[(dataloader, loss_func)],
            epochs=config['epochs'],
            warmup_steps=0,
            output_path=output_dir,
            show_progress_bar=False,
            save_best_model=False,
            optimizer_params={'lr': 2e-5}
        )
        logger.info(f"  Model trained and saved to {output_dir}")
        return model
    except Exception as e:
        logger.error(f"  Training failed: {e}")
        return None


def evaluate_model_on_user(model, user_holdout: List[Dict], asin_to_title: Dict[str, str]) -> float:
    if not user_holdout:
        return 0.0
    
    corpus = list(asin_to_title.values())
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    
    mrr_scores = []
    
    for pair in user_holdout:
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
        
        for rank, idx in enumerate(ranked_indices):
            if corpus[idx] == asin_to_title[positive_asin]:
                if rank < 10:
                    mrr_scores.append(1.0 / (rank + 1))
                break
    
    return np.mean(mrr_scores) if mrr_scores else 0.0


def main():
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loocv_dir = Path(CONFIG['loocv_data_dir'])
    users = sorted([d.name for d in loocv_dir.iterdir() if d.is_dir()])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT 4: Transfer Learning Cross-Evaluation")
    logger.info(f"{'='*60}")
    logger.info(f"Total users: {len(users)}")
    
    asin_to_title = load_product_metadata(CONFIG['meta_path'])
    all_asins = list(asin_to_title.keys())
    
    logger.info(f"\nStep 1: Train individual user models...")
    trained_models = {}
    user_to_data = {}
    
    for user in users:
        logger.info(f"\nTraining model for {user}...")
        user_data = load_user_data(str(loocv_dir / user))
        user_to_data[user] = user_data
        
        personal_train = user_data['personal_train']
        triplets = build_triplet_examples(personal_train, asin_to_title, all_asins)
        
        checkpoint_dir = str(output_dir / f'model_{user}')
        model = train_user_model(triplets, checkpoint_dir, CONFIG)
        trained_models[user] = (checkpoint_dir if model else None)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 2: Cross-evaluation matrix (User_A model on User_B data)")
    logger.info(f"{'='*60}")
    
    transfer_matrix = {}
    baseline_performance = {}
    
    for source_user in users:
        if trained_models[source_user] is None:
            logger.warning(f"Skipping {source_user}: model training failed")
            continue
        
        try:
            source_model = SentenceTransformer(trained_models[source_user])
        except Exception as e:
            logger.warning(f"Failed to load model for {source_user}: {e}")
            continue
        
        transfer_matrix[source_user] = {}
        
        for target_user in users:
            target_holdout = user_to_data[target_user]['holdout']
            mrr = evaluate_model_on_user(source_model, target_holdout, asin_to_title)
            transfer_matrix[source_user][target_user] = float(mrr)
            
            if source_user == target_user:
                baseline_performance[source_user] = mrr
    
    logger.info(f"\nTransfer matrix computed (diagonal = own user, off-diagonal = transfer):")
    logger.info(f"\nTransfer Matrix (MRR@10):")
    logger.info(f"{'User':<25} | " + " | ".join([f"{u[:15]:<15}" for u in users[:5]]) + " | ...")
    logger.info("-" * 120)
    
    for source_user in users[:5]:
        if source_user in transfer_matrix:
            vals = [transfer_matrix[source_user].get(target_user, 0.0) for target_user in users[:5]]
            logger.info(f"{source_user:<25} | " + " | ".join([f"{v:>15.4f}" for v in vals]) + " | ...")
    
    logger.info("\nDiagonal (self) vs Off-diagonal (transfer) analysis:")
    self_scores = []
    transfer_scores = []
    
    for source_user in transfer_matrix:
        for target_user in transfer_matrix[source_user]:
            score = transfer_matrix[source_user][target_user]
            if source_user == target_user:
                self_scores.append(score)
            else:
                transfer_scores.append(score)
    
    logger.info(f"  Self-trained model on own user: {np.mean(self_scores):.4f} ± {np.std(self_scores):.4f}")
    logger.info(f"  Transfer to other users: {np.mean(transfer_scores):.4f} ± {np.std(transfer_scores):.4f}")
    logger.info(f"  Transfer degradation: {(1 - np.mean(transfer_scores)/max(np.mean(self_scores), 1e-6))*100:.1f}%")
    
    logger.info(f"\nStep 3: Evaluate pretrained baseline for comparison...")
    pretrained_model = SentenceTransformer(CONFIG['model_name'])
    pretrained_performance = {}
    
    for user in users:
        user_data = user_to_data[user]
        holdout = user_data['holdout']
        mrr = evaluate_model_on_user(pretrained_model, holdout, asin_to_title)
        pretrained_performance[user] = float(mrr)
    
    logger.info(f"\nPretrained baseline MRR@10: {np.mean(list(pretrained_performance.values())):.4f}")
    
    results = {
        'transfer_matrix': transfer_matrix,
        'self_trained_avg': float(np.mean(self_scores)) if self_scores else 0.0,
        'transfer_avg': float(np.mean(transfer_scores)) if transfer_scores else 0.0,
        'transfer_degradation_pct': float((1 - np.mean(transfer_scores)/max(np.mean(self_scores), 1e-6))*100) if self_scores else 0.0,
        'pretrained_performance': pretrained_performance,
        'pretrained_avg': float(np.mean(list(pretrained_performance.values()))),
        'users': users
    }
    
    results_file = output_dir / 'transfer_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY:")
    logger.info(f"{'='*60}")
    logger.info(f"Self-trained (diagonal): {np.mean(self_scores):.4f}")
    logger.info(f"Transfer to others: {np.mean(transfer_scores):.4f}")
    logger.info(f"Degradation: {(1 - np.mean(transfer_scores)/max(np.mean(self_scores), 1e-6))*100:.1f}%")
    logger.info(f"Pretrained baseline: {np.mean(list(pretrained_performance.values())):.4f}")


if __name__ == '__main__':
    main()
