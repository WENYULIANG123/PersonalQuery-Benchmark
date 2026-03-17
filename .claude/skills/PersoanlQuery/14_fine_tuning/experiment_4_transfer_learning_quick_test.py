#!/usr/bin/env python3

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import torch

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    'model_name': 'intfloat/e5-base-v2',
    'loocv_data_dir': '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data',
    'meta_path': '/fs04/ar57/wenyu/data/Amazon-Reviews-2018/intermediate/df_ucsd_meta.pkl',
    'output_dir': '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/experiment_4_transfer_quick',
    'quick_test_users': ['user_A13OFOB1394G31', 'user_A1GYEGLX3P2Y7P', 'user_A1PAGHECG401K1']
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
    """Load and parse user data correctly"""
    with open(f'{user_dir}/holdout.json', 'r') as f:
        holdout_data = json.load(f)
        holdout = holdout_data.get('pairs', holdout_data) if isinstance(holdout_data, dict) else holdout_data
    
    with open(f'{user_dir}/personal_train.json', 'r') as f:
        train_data = json.load(f)
        personal_train = train_data.get('pairs', train_data) if isinstance(train_data, dict) else train_data
    
    return {'holdout': holdout, 'personal_train': personal_train}

def evaluate_baseline_quick(user_id: str, personal_train: List, holdout: List, 
                            asin_to_title: Dict, all_asins: List) -> Dict:
    """Quick evaluation using pretrained model"""
    logger.info(f"\n[{user_id}] Quick test (pretrained e5-base-v2):")
    
    model = SentenceTransformer(CONFIG['model_name'])
    
    # Encode corpus once
    corpus = [asin_to_title.get(asin, "") for asin in all_asins]
    corpus_embeddings = model.encode(corpus, convert_to_tensor=False)
    
    # Test on holdout
    mrr_scores = []
    for pair in holdout[:10]:  # Just first 10 for speed
        query = pair.get('positive', '')
        positive_asin = pair.get('asin', '')
        
        if not query:
            continue
        
        query_embedding = model.encode(query, convert_to_tensor=False)
        
        similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
        ranked_indices = np.argsort(-similarities)
        
        for rank, idx in enumerate(ranked_indices):
            if all_asins[idx] == positive_asin:
                if rank < 10:
                    mrr_scores.append(1.0 / (rank + 1))
                break
    
    mrr = np.mean(mrr_scores) if mrr_scores else 0.0
    logger.info(f"  MRR@10 (pretrained): {mrr:.4f} ({len(mrr_scores)} hits / {len(holdout)} queries)")
    
    return {'mrr': mrr, 'hits': len(mrr_scores), 'total': len(holdout)}

def main():
    logger.info("="*70)
    logger.info("EXPERIMENT 4 QUICK TEST: Transfer Learning Baseline (3 users)")
    logger.info("="*70)
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    asin_to_title = load_product_metadata(CONFIG['meta_path'])
    all_asins = list(asin_to_title.keys())
    
    loocv_dir = Path(CONFIG['loocv_data_dir'])
    users = CONFIG['quick_test_users']
    
    # Quick baseline evaluation
    logger.info("\nStep 1: Evaluate pretrained baseline on 3 users")
    results = {}
    
    for user in users:
        user_dir = loocv_dir / user
        if not user_dir.exists():
            logger.warning(f"  {user} data not found, skipping")
            continue
        
        data = load_user_data(str(user_dir))
        result = evaluate_baseline_quick(user, data['personal_train'], 
                                         data['holdout'], asin_to_title, all_asins)
        results[user] = result
    
    # Summary
    logger.info(f"\n" + "="*70)
    logger.info("QUICK TEST SUMMARY (Pretrained Baseline):")
    for user, res in results.items():
        logger.info(f"  {user}: MRR@10 = {res['mrr']:.4f}")
    
    avg_mrr = np.mean([r['mrr'] for r in results.values()])
    logger.info(f"  Average MRR@10: {avg_mrr:.4f}")
    logger.info("="*70)
    
    # Save results
    output_file = output_dir / 'transfer_results_quick_baseline.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    logger.info(f"✅ Results saved to {output_file}")

if __name__ == '__main__':
    main()
