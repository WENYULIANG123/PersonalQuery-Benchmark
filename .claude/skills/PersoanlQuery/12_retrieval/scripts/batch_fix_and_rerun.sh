#!/bin/bash
# Batch fix and re-run all retrieval evaluations with data tracking
# This script ensures all retrievers use consistent data sources

set -e

cd /home/wlia0047/ar57/wenyu

echo "================================================================"
echo "Stage 13: Batch Fix and Re-run All Retrieval Evaluations"
echo "================================================================"
echo "Start time: $(date)"
echo ""

# Activate environment
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/stark

# Configuration
BASE_DIR="/home/wlia0047/ar57/wenyu"
OUTPUT_DIR="${BASE_DIR}/result/personal_query/13_retrieval"
CACHE_DIR="${OUTPUT_DIR}/cache"
USER_ID="A13OFOB1394G31"

# Array of retriever scripts (relative to .claude/skills/PersoanlQuery/13_retrieval/evaluators/)
RETRIEVERS=(
    "sparse_retrieval/13_evaluate_bm25.py"
    "sparse_retrieval/13_evaluate_tfidf.py"
    "sparse_retrieval/13_evaluate_dirichlet.py"
    "dense_retrieval/13_evaluate_dense.py"
    "dense_retrieval/13_evaluate_e5.py"
    "dense_retrieval/13_evaluate_bge.py"
    "dense_retrieval/13_evaluate_ance.py"
    "dense_retrieval/13_evaluate_minilm.py"
    "dense_retrieval/13_evaluate_mpnet.py"
    "dense_retrieval/13_evaluate_star.py"
    "late_interaction/13_evaluate_colbert.py"
)

echo "🔧 Step 1: Deleting all caches..."
if [ -d "$CACHE_DIR" ]; then
    echo "  Deleting cache directory: $CACHE_DIR"
    rm -rf "$CACHE_DIR"
    echo "  ✅ Cache deleted"
else
    echo "  ℹ️  No cache directory found"
fi

echo ""
echo "🔧 Step 2: Running all retrievers in CLEAN mode..."
echo ""

# Run clean mode for all retrievers
for retriever in "${RETRIEVERS[@]}"; do
    script_path=".claude/skills/PersoanlQuery/13_retrieval/evaluators/$retriever"

    if [ ! -f "$script_path" ]; then
        echo "  ⚠️  Script not found: $script_path"
        continue
    fi

    echo "  🟢 Running CLEAN mode: $retriever"
    python -u "$script_path" --query-mode clean > /tmp/clean_${retriever##*/}.log 2>&1 &
done

# Wait for all clean mode jobs to complete
echo ""
echo "  ⏳ Waiting for CLEAN mode jobs to complete..."
wait
echo "  ✅ All CLEAN mode jobs completed"

echo ""
echo "🔧 Step 3: Running all retrievers in NOISY mode..."
echo ""

# Run noisy mode for all retrievers
for retriever in "${RETRIEVERS[@]}"; do
    script_path=".claude/skills/PersoanlQuery/13_retrieval/evaluators/$retriever"

    if [ ! -f "$script_path" ]; then
        echo "  ⚠️  Script not found: $script_path"
        continue
    fi

    echo "  🔴 Running NOISY mode: $retriever"
    python -u "$script_path" --query-mode noisy > /tmp/noisy_${retriever##*/}.log 2>&1 &
done

# Wait for all noisy mode jobs to complete
echo ""
echo "  ⏳ Waiting for NOISY mode jobs to complete..."
wait
echo "  ✅ All NOISY mode jobs completed"

echo ""
echo "🔧 Step 4: Verifying data consistency..."
echo ""

# Create verification script
python3 << 'VERIFY_EOF'
import json
import os
import sys

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/13_retrieval")
from utils.data_tracking import verify_data_consistency

output_dir = "/home/wlia0047/ar57/wenyu/result/personal_query/13_retrieval"
user_id = "A13OFOB1394G31"

retrievers = [
    ('bm25', 'BM25'),
    ('tfidf', 'TF-IDF'),
    ('dirichlet', 'Dirichlet'),
    ('dense', 'Dense (MiniLM)'),
    ('e5', 'E5'),
    ('bge', 'BGE'),
    ('colbert', 'ColBERT'),
    ('ance', 'ANCE'),
    ('minilm', 'MiniLM'),
    ('mpnet', 'MPNet'),
    ('star', 'STAR')
]

print("=" * 80)
print("数据一致性验证结果")
print("=" * 80)
print()

issues = []

for retriever_key, retriever_name in retrievers:
    clean_file = os.path.join(output_dir, f'retrieval_{retriever_key}_clean_{user_id}.json')
    noisy_file = os.path.join(output_dir, f'retrieval_{retriever_key}_noisy_{user_id}.json')

    if not os.path.exists(clean_file) or not os.path.exists(noisy_file):
        continue

    result = verify_data_consistency(clean_file, noisy_file)

    if 'error' in result:
        print(f'❌ {retriever_name:15s} - {result["error"]}')
        issues.append(f'{retriever_name}: {result["error"]}')
    elif not result['overall_consistent']:
        print(f'⚠️  {retriever_name:15s} - 数据不一致')
        print(f'   Clean source: {result["clean_source"]}, Noisy source: {result["noisy_source"]}')
        issues.append(f'{retriever_name}: 数据源不一致')
    else:
        print(f'✅ {retriever_name:15s} - 数据一致')
        if result.get('fingerprint_match'):
            fp_info = result['fingerprint_match']
            print(f'   指纹验证: {fp_info["matches"]}/{fp_info["common_asins"]} 匹配')

print()
if issues:
    print(f"⚠️  发现 {len(issues)} 个问题:")
    for issue in issues:
        print(f"   - {issue}")
else:
    print("✅ 所有检索器数据一致性验证通过！")

VERIFY_EOF

echo ""
echo "================================================================"
echo "批量修复和重新运行完成"
echo "================================================================"
echo "完成时间: $(date)"
echo ""
echo "📊 查看详细结果："
echo "  - 结果目录: $OUTPUT_DIR"
echo "  - 日志目录: /tmp/clean_*.log /tmp/noisy_*.log"
