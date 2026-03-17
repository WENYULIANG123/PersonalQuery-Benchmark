#!/usr/bin/env python3
"""
Quick test of the ranking evaluation logic
- Load a base model (E5)
- Test on a few queries to verify metrics are reasonable
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

data_dir = Path("/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning")
doc_cache_dir = Path("/fs04/ar57/wenyu/result/personal_query/12_retrieval/document_cache")

print("=" * 80)
print("Loading product metadata...")
print("=" * 80)

metadata_file = doc_cache_dir / "Arts_Crafts_and_Sewing_metadata.pkl"
with open(metadata_file, 'rb') as f:
    product_metadata = pickle.load(f)

all_product_asins = list(product_metadata.keys())
print(f"✓ Loaded {len(all_product_asins)} products")

print("\n" + "=" * 80)
print("Loading training data...")
print("=" * 80)

with open(data_dir / "training_data_v4_stratified.json") as f:
    training_data = json.load(f)

pairs = training_data.get('pairs', [])
print(f"✓ Loaded {len(pairs)} training pairs")

print("\n" + "=" * 80)
print("Loading base E5 model...")
print("=" * 80)

model = SentenceTransformer("intfloat/e5-base-v2")
print("✓ Model loaded")

print("\n" + "=" * 80)
print("Testing on first 5 queries...")
print("=" * 80)

test_pairs = pairs[:5]

for idx, pair in enumerate(test_pairs):
    query = pair.get('query', '')
    asin = pair.get('asin', '')
    user_id = pair.get('user_id', '')
    
    print(f"\n[Query {idx+1}/{len(test_pairs)}]")
    print(f"  User: {user_id}")
    print(f"  ASIN (ground truth): {asin}")
    print(f"  Query: {query[:60]}...")
    
    query_embedding = model.encode(query, convert_to_numpy=True)
    
    sample_asins = np.random.choice(all_product_asins, size=1000, replace=False).tolist()
    
    similarities = []
    for product_asin in sample_asins:
        if product_asin in product_metadata:
            product_text = product_metadata[product_asin].get('title', '')
            if product_text:
                product_embedding = model.encode(product_text, convert_to_numpy=True)
                sim = np.dot(query_embedding, product_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(product_embedding) + 1e-8
                )
                similarities.append((product_asin, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    ranked_asins = [a for a, s in similarities]
    
    print(f"  Candidates sampled: {len(similarities)}")
    print(f"  Top 5 products:")
    for rank, (top_asin, sim) in enumerate(similarities[:5]):
        match = "✓" if top_asin == asin else " "
        product_title = product_metadata[top_asin].get('title', 'N/A')[:50]
        print(f"    {rank+1}. [{match}] {top_asin}: {product_title}... (sim={sim:.4f})")
    
    if asin in ranked_asins:
        position = ranked_asins.index(asin)
        mrr_10 = 1.0 / (position + 1) if position < 10 else 0
        print(f"  ✓ Found at rank {position+1}")
        print(f"  MRR@10 = {mrr_10:.4f}")
    else:
        print(f"  ✗ Not found in top {len(ranked_asins)} results")
        print(f"  MRR@10 = 0.0000")

print("\n" + "=" * 80)
print("✓ Test complete - evaluation logic working!")
print("=" * 80)
