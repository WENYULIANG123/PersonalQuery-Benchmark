#!/usr/bin/env python3
"""
Evaluate Persona Diversity: Inter-persona Similarity
Computes the semantic similarity between different user personas to measure differentiation.
Lower inter-persona similarity indicates higher differentiation.
"""

import json
import os
import sys
import argparse
import numpy as np
import itertools
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Persona Diversity")
    parser.add_argument("--persona-dir",
                        default="/home/wlia0047/wenyu/result/user_profile/03_persona/results",
                        help="Directory containing user personas")
    parser.add_argument("--output-file",
                        default="/home/wlia0047/wenyu/result/user_profile/10_evaluation/diversity/diversity_metrics.json",
                        help="Output JSON file for diversity metrics")
    args = parser.parse_args()

    # Load embedding model
    log_with_timestamp("Loading embedding model (Sentence-BERT)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        log_with_timestamp("sentence-transformers not installed.")
        return

    # Load personas
    log_with_timestamp(f"Loading personas from {args.persona_dir}...")
    personas = {}
    for f in os.listdir(args.persona_dir):
        if f.startswith('persona_') and f.endswith('.json'):
            with open(os.path.join(args.persona_dir, f), 'r') as file:
                data = json.load(file)
                user_id = data['user_id']
                personas[user_id] = data.get('persona', '')
    
    user_ids = sorted(list(personas.keys()))
    log_with_timestamp(f"Loaded {len(user_ids)} personas.")

    # Compute embeddings
    log_with_timestamp("Computing embeddings for all personas...")
    embeddings = {}
    for uid in user_ids:
        embeddings[uid] = model.encode(personas[uid], convert_to_numpy=True)

    # Compute all-vs-all similarity
    log_with_timestamp("Computing pairwise similarities...")
    pairs = list(itertools.combinations(user_ids, 2))
    similarities = []
    
    matrix = np.zeros((len(user_ids), len(user_ids)))
    
    for i, uid1 in enumerate(user_ids):
        matrix[i, i] = 1.0 # Self-similarity
        for j, uid2 in enumerate(user_ids):
            if i < j:
                sim = cosine_similarity(embeddings[uid1], embeddings[uid2])
                matrix[i, j] = sim
                matrix[j, i] = sim
                similarities.append({
                    'user1': uid1,
                    'user2': uid2,
                    'similarity': round(float(sim), 4)
                })

    avg_similarity = np.mean([s['similarity'] for s in similarities])
    
    # Sort similarities to find most/least similar
    sorted_sims = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    
    # Output Results
    output = {
        'timestamp': datetime.now().isoformat(),
        'total_personas': len(user_ids),
        'avg_inter_persona_similarity': round(float(avg_similarity), 4),
        'differentiation_score': round(float(1.0 - avg_similarity), 4),
        'most_similar_pairs': sorted_sims[:5],
        'least_similar_pairs': sorted_sims[-5:],
        'full_matrix': {
            'user_ids': user_ids,
            'similarities': matrix.tolist()
        }
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("PERSONA DIVERSITY ASSESSMENT")
    log_with_timestamp("="*60)
    log_with_timestamp(f"Average Inter-persona Similarity: {avg_similarity:.4f}")
    log_with_timestamp(f"Differentiation Score (1 - Similarity): {1.0 - avg_similarity:.4f}")
    log_with_timestamp("-" * 60)
    log_with_timestamp("Top 3 Most Similar Pairs (Potential Overlap):")
    for s in sorted_sims[:3]:
        log_with_timestamp(f"  {s['user1']} <-> {s['user2']}: {s['similarity']:.4f}")
    
    log_with_timestamp("-" * 60)
    log_with_timestamp("Top 3 Most Differentiated Pairs:")
    for s in sorted_sims[-3:]:
        log_with_timestamp(f"  {s['user1']} <-> {s['user2']}: {s['similarity']:.4f}")
    log_with_timestamp("="*60)
    log_with_timestamp(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
