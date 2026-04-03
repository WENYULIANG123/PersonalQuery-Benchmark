#!/bin/bash
# Stage 5 LINGCONV Training with Semantic Pooling Launcher
# 使用方法: bash run_training_semantic_pooling.sh

set -e

# 默认参数
CKPT_DIR="/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis/checkpoints"
DATA_DIR="/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis/data"
MODEL_NAME="google/t5-v1_1-xl"
EPOCHS=2
BATCH_SIZE=4
LR=5e-5

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output_suffix)
            OUTPUT_SUFFIX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 构建输出名称
TIMESTAMP=$(date +"%m%d_%H-%M-%S")
QC_SUFFIX="semantic_pooling"
if [ ! -z "$OUTPUT_SUFFIX" ]; then
    QC_SUFFIX="${QC_SUFFIX}_${OUTPUT_SUFFIX}"
fi

OUTPUT_NAME="${TIMESTAMP}-ling_conversion-decoder_add_first_${QC_SUFFIX}"

echo "========================================"
echo "LINGCONV Training with Semantic Pooling"
echo "========================================"
echo "Output: $CKPT_DIR/$OUTPUT_NAME"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Semantic Pooling: ENABLED"
echo "========================================"

# 运行训练
cd /fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis

python main.py \
    --do_train \
    --ckpt_dir "$CKPT_DIR" \
    --data_dir "$DATA_DIR" \
    --data ling_conversion \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 32 \
    --epochs $EPOCHS \
    --lr $LR \
    --max_length 128 \
    --combine_method decoder_add_first \
    --ling2_only \
    --seed 42 \
    --use_semantic_pooling \
    --name "$OUTPUT_NAME"
