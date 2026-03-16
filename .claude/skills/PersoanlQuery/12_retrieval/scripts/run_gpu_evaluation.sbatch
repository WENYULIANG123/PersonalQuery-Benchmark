#!/bin/bash
#SBATCH --job-name=stage13_retrieval_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=result/personal_query/13_retrieval/logs/gpu_eval_%j.out
#SBATCH --error=result/personal_query/13_retrieval/logs/gpu_eval_%j.err

# 确保日志目录存在
mkdir -p result/personal_query/13_retrieval/logs

echo "================================================"
echo "Stage 13: GPU 加速检索评估"
echo "================================================"
echo "节点: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "开始时间: $(date)"
echo ""

# 激活conda环境
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/stark

# 检查GPU
echo "检查GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# 切换到工作目录
cd /home/wlia0047/ar57/wenyu

# 删除缓存
echo "🔧 删除缓存..."
rm -rf result/personal_query/13_retrieval/cache
echo ""

# 清理之前的结果
echo "🔧 清理之前的结果..."
rm -f result/personal_query/13_retrieval/retrieval_*_clean_A13OFOB1394G31.json
rm -f result/personal_query/13_retrieval/retrieval_*_noisy_A13OFOB1394G31.json
echo ""

echo "🚀 开始运行 CLEAN 模式..."
echo ""

# Sparse 检索器（CPU更快，不需要GPU）
echo "▶ Sparse Retrievers (CPU):"
python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_bm25.py --query-mode clean
echo "✅ BM25 Clean 完成"
python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_tfidf.py --query-mode clean
echo "✅ TF-IDF Clean 完成"
python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_dirichlet.py --query-mode clean
echo "✅ Dirichlet Clean 完成"
echo ""

# Dense 检索器（GPU加速）
echo "▶ Dense Retrievers (GPU):"
python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_bge.py --query-mode clean
echo "✅ BGE Clean 完成"

python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_e5.py --query-mode clean
echo "✅ E5 Clean 完成"

python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_ance.py --query-mode clean
echo "✅ ANCE Clean 完成"

python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_minilm.py --query-mode clean
echo "✅ MiniLM Clean 完成"

python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_mpnet.py --query-mode clean
echo "✅ MPNet Clean 完成"

python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_star.py --query-mode clean
echo "✅ STAR Clean 完成"

python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_dense.py --query-mode clean
echo "✅ Dense Clean 完成"

# Late Interaction
python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/late_interaction/13_evaluate_colbert.py --query-mode clean
echo "✅ ColBERT Clean 完成"
echo ""

echo "🚀 开始运行 NOISY 模式..."
echo ""

# Sparse 检索器
echo "▶ Sparse Retrievers (CPU):"
python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_bm25.py --query-mode noisy
echo "✅ BM25 Noisy 完成"
python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_tfidf.py --query-mode noisy
echo "✅ TF-IDF Noisy 完成"
python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/sparse_retrieval/13_evaluate_dirichlet.py --query-mode noisy
echo "✅ Dirichlet Noisy 完成"
echo ""

# Dense 检索器
echo "▶ Dense Retrievers (GPU):"
python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_bge.py --query-mode noisy
echo "✅ BGE Noisy 完成"

python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_e5.py --query-mode noisy
echo "✅ E5 Noisy 完成"

python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_ance.py --query-mode noisy
echo "✅ ANCE Noisy 完成"

python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_minilm.py --query-mode noisy
echo "✅ MiniLM Noisy 完成"

python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_mpnet.py --query-mode noisy
echo "✅ MPNet Noisy 完成"

python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_star.py --query-mode noisy
echo "✅ STAR Noisy 完成"

python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/dense_retrieval/13_evaluate_dense.py --query-mode noisy
echo "✅ Dense Noisy 完成"

# Late Interaction
python -u .claude/skills/PersoanlQuery/13_retrieval/evaluators/late_interaction/13_evaluate_colbert.py --query-mode noisy
echo "✅ ColBERT Noisy 完成"
echo ""

echo "================================================"
echo "✅ 所有评估完成！"
echo "结束时间: $(date)"
echo "================================================"
