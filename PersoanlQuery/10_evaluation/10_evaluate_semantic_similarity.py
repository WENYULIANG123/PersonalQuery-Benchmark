#!/usr/bin/env python3
"""
Semantic Evaluation: Query-Persona Similarity
Uses sentence embeddings to compute semantic similarity between queries and persona.
More objective than LLM scoring.
"""

import json
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def get_embedding_sbert(text, model):
    """Get sentence embedding using Sentence-BERT"""
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

def get_embedding_openai(text, client):
    """Get embedding using OpenAI API (fallback)"""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        log_with_timestamp(f"OpenAI embedding error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Semantic Evaluation: Query-Persona Similarity")
    parser.add_argument("--dual-queries-dir",
                        default="/home/wlia0047/wenyu/result/user_profile/06_query",
                        help="Directory containing dual query files")
    parser.add_argument("--persona-dir",
                        default="/home/wlia0047/wenyu/result/user_profile/03_persona/results",
                        help="Directory containing user personas")
    parser.add_argument("--output-dir",
                        default="/home/wlia0047/wenyu/result/user_profile/10_evaluation/semantic",
                        help="Output directory for semantic evaluation results")
    parser.add_argument("--method", choices=['sbert', 'openai'], default='sbert',
                        help="Embedding method to use")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load embedding model
    log_with_timestamp(f"Loading embedding model ({args.method})...")

    if args.method == 'sbert':
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            log_with_timestamp("Loaded Sentence-BERT model: all-MiniLM-L6-v2")
        except ImportError:
            log_with_timestamp("sentence-transformers not installed. Install with: pip install sentence-transformers")
            return
    else:
        try:
            import openai
            client = openai.OpenAI()
            model = None
            log_with_timestamp("Using OpenAI embeddings")
        except Exception as e:
            log_with_timestamp(f"OpenAI not available: {e}")
            return

    # Load personas
    log_with_timestamp("Loading user personas...")
    personas = {}
    for f in os.listdir(args.persona_dir):
        if f.startswith('persona_') and f.endswith('.json'):
            with open(os.path.join(args.persona_dir, f), 'r') as file:
                data = json.load(file)
                personas[data['user_id']] = data.get('persona', '')
    log_with_timestamp(f"Loaded {len(personas)} personas")

    # Pre-compute persona embeddings
    log_with_timestamp("Computing persona embeddings...")
    persona_embeddings = {}
    for user_id, persona_text in personas.items():
        if args.method == 'sbert':
            persona_embeddings[user_id] = get_embedding_sbert(persona_text, model)
        else:
            persona_embeddings[user_id] = get_embedding_openai(persona_text, client)
    log_with_timestamp(f"Computed {len(persona_embeddings)} persona embeddings")

    # Load dual queries
    log_with_timestamp("Loading dual queries...")
    dual_query_files = sorted([
        os.path.join(args.queries_dir, f)
        for f in os.listdir(args.queries_dir)
        if f.startswith('queries_') and f.endswith('.json')
    ])

    # Process each user
    all_results = []

    for dual_file in dual_query_files:
        with open(dual_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            # Extract user_id from filename (queries_USERID.json)
            user_id = os.path.basename(dual_file).replace('queries_', '').replace('.json', '')
        else:
            user_id = data.get('user_id')
        persona_emb = persona_embeddings.get(user_id)

        if persona_emb is None:
            log_with_timestamp(f"Skipping {user_id} - no persona embedding")
            continue

        log_with_timestamp(f"\nEvaluating {user_id}...")

        results = []
        public_sims = []
        personalized_sims = []

        for item in data:
            # Check for query existence using the actual keys
            if not (item.get('public_query') and item.get('personalized_query')):
                continue

            asin = item.get('asin')
            public_query = item.get('public_query')
            personalized_query = item.get('personalized_query')

            # Get query embeddings
            if args.method == 'sbert':
                public_emb = get_embedding_sbert(public_query, model)
                personalized_emb = get_embedding_sbert(personalized_query, model)
            else:
                public_emb = get_embedding_openai(public_query, client)
                personalized_emb = get_embedding_openai(personalized_query, client)

            if public_emb is None or personalized_emb is None:
                continue

            # Compute similarities
            public_sim = cosine_similarity(persona_emb, public_emb)
            personalized_sim = cosine_similarity(persona_emb, personalized_emb)

            result = {
                'asin': asin,
                'category': item.get('category'),
                'public_query': public_query,
                'public_similarity': round(float(public_sim), 4),
                'personalized_query': personalized_query,
                'personalized_similarity': round(float(personalized_sim), 4),
                'similarity_diff': round(float(personalized_sim - public_sim), 4)
            }
            results.append(result)

            public_sims.append(public_sim)
            personalized_sims.append(personalized_sim)

        if not results:
            continue

        # Save user results
        user_output = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'embedding_method': args.method,
            'total_items': len(results),
            'avg_public_similarity': round(float(sum(public_sims) / len(public_sims)), 4),
            'avg_personalized_similarity': round(float(sum(personalized_sims) / len(personalized_sims)), 4),
            'improvement': round(float(sum(personalized_sims) / len(personalized_sims) - sum(public_sims) / len(public_sims)), 4),
            'results': results
        }

        output_file = os.path.join(args.output_dir, f"semantic_eval_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(user_output, f, indent=2, ensure_ascii=False)

        all_results.append({
            'user_id': user_id,
            'total': len(results),
            'avg_public': user_output['avg_public_similarity'],
            'avg_personalized': user_output['avg_personalized_similarity'],
            'improvement': user_output['improvement']
        })

        log_with_timestamp(f"  Avg: Public={user_output['avg_public_similarity']:.4f}, Personalized={user_output['avg_personalized_similarity']:.4f}, Improve={user_output['improvement']:+.4f}")

    # Summary
    log_with_timestamp("\n" + "="*70)
    log_with_timestamp("SEMANTIC EVALUATION SUMMARY")
    log_with_timestamp("="*70)
    log_with_timestamp(f"Embedding Method: {args.method}")
    log_with_timestamp(f"{'User ID':<18} {'Items':>6} {'Public':>10} {'Personal':>10} {'Improve':>10}")
    log_with_timestamp("-"*70)

    total_public = 0
    total_personalized = 0
    total_items = 0

    for r in all_results:
        log_with_timestamp(f"{r['user_id']:<18} {r['total']:>6} {r['avg_public']:>10.4f} {r['avg_personalized']:>10.4f} {r['improvement']:>+10.4f}")
        total_public += r['avg_public'] * r['total']
        total_personalized += r['avg_personalized'] * r['total']
        total_items += r['total']

    log_with_timestamp("-"*70)
    overall_public = total_public / total_items if total_items > 0 else 0
    overall_personalized = total_personalized / total_items if total_items > 0 else 0
    overall_improve = overall_personalized - overall_public

    log_with_timestamp(f"{'OVERALL':<18} {total_items:>6} {overall_public:>10.4f} {overall_personalized:>10.4f} {overall_improve:>+10.4f}")
    log_with_timestamp("="*70)

    # Improvement statistics
    improvements = [r['improvement'] for r in all_results]
    positive = sum(1 for i in improvements if i > 0)
    negative = sum(1 for i in improvements if i < 0)

    log_with_timestamp(f"\n改进情况:")
    log_with_timestamp(f"  个性化优于大众: {positive} 个用户 ({100*positive/len(improvements):.0f}%)")
    log_with_timestamp(f"  大众优于个性化: {negative} 个用户")

    # Save summary
    summary_output = {
        'timestamp': datetime.now().isoformat(),
        'embedding_method': args.method,
        'total_items': total_items,
        'overall_public_similarity': round(overall_public, 4),
        'overall_personalized_similarity': round(overall_personalized, 4),
        'overall_improvement': round(overall_improve, 4),
        'positive_count': positive,
        'negative_count': negative,
        'by_user': all_results
    }

    summary_file = os.path.join(args.output_dir, "semantic_evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_output, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
