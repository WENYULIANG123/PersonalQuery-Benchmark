#!/bin/bash
#SBATCH --job-name=test_metrics_fix
#SBATCH --output=/home/wlia0047/ar57/wenyu/stage12_test_fix_%j.log
#SBATCH --time=00:15:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4

cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval/evaluators

python3 << 'PYTHON'
import json
import os
import sys
import pickle
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from utils import utils
from retriever_manager import get_retriever_manager

STAGE9_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query"
METADATA_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl"

def load_queries_flat(user_id):
    query_file = os.path.join(STAGE9_DIR, f"noisy_queries_{user_id}.json")
    with open(query_file, 'r') as f:
        data = json.load(f)
    
    queries = data.get('queries', [])
    result = []
    for q in queries:
        asin = q.get('asin', '')
        if asin:
            pq = q.get('personalized_query', {})
            original = pq.get('original', '')
            if original:
                result.append({'asin': asin, 'query': original})
    return result[:20]

print("Testing metrics fix...")
with open(METADATA_FILE, 'rb') as f:
    metadata = pickle.load(f)

all_asins = list(metadata.keys())
docs = [dict(metadata[a], asin=a) for a in all_asins]

rm = get_retriever_manager()
bm25 = rm.get_retriever('bm25', docs, metadata)

queries = load_queries_flat('A13OFOB1394G31')
print(f"Loaded {len(queries)} queries")

metrics = utils.evaluate_retriever(bm25, queries, all_asins, [1, 3, 5])
print(f"\n✅ Metrics: {json.dumps(metrics, indent=2)}")

for k, v in metrics.items():
    if v == 0:
        print(f"❌ FAILED: {k} = 0")
        sys.exit(1)

print("\n✅ SUCCESS: All metrics non-zero!")
PYTHON
