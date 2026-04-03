#!/bin/bash
#========================================
# LINGCONV 统一训练和测试脚本
#========================================
# 使用方法:
#   训练: bash run_lingconv.sh train --combine_method decoder_add_first
#   测试: bash run_lingconv.sh test --ckpt <checkpoint_path>
#========================================

set -e

# 默认参数
SCRIPT_DIR="/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis"
CKPT_DIR="/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints"
DATA_DIR="/home/wlia0047/ar57_scratch/wenyu/ling_conversion_data"
MODEL_NAME="google/flan-t5-base"
EPOCHS=2
BATCH_SIZE=256
LR=1e-3
MAX_LENGTH=128
COMBINE_METHOD="decoder_add_first"
USE_BF16="true"
CLEAN_CHECKPOINTS="true"

# SBATCH参数
SBATCH_PARTITION="m3h"
SBATCH_GPU="gpu:H100:1"
SBATCH_QOS="m3h"
SBATCH_TIME="24:00:00"
SBATCH_MEM="64G"

# 解析参数
MODE=""
CKPT_PATH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        train|test|infer)
            MODE="$1"
            shift
            ;;
        --combine_method)
            COMBINE_METHOD="$2"
            shift 2
            ;;
        --ckpt)
            CKPT_PATH="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --no-bf16)
            USE_BF16="false"
            shift
            ;;
        --keep_checkpoints)
            CLEAN_CHECKPOINTS="false"
            shift
            ;;
        --name)
            RUN_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 加载conda环境
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57/wenyu/envs/lingconv

# 进入脚本目录
cd "$SCRIPT_DIR"
export PYTHONUNBUFFERED=1

# 生成时间戳
TIMESTAMP=$(date +"%m%d_%H-%M-%S")
RUN_NAME="${RUN_NAME:-${TIMESTAMP}_${COMBINE_METHOD}_lingconv}"

#========================================
# 训练模式
#========================================
train() {
    echo "========================================"
    echo "LINGCONV Training"
    echo "========================================"
    echo "Combine Method: $COMBINE_METHOD"
    echo "Batch Size: $BATCH_SIZE"
    echo "Learning Rate: $LR"
    echo "Epochs: $EPOCHS"
    echo "Output: $CKPT_DIR/$RUN_NAME"
    echo "========================================"

    # 清理旧checkpoint
    if [ "$CLEAN_CHECKPOINTS" = "true" ]; then
        echo "清理旧checkpoint..."
        rm -rf "$CKPT_DIR/"
        mkdir -p "$CKPT_DIR"
    fi

    # 构建命令
    CMD="python -u main.py \
        --do_train \
        --data ling_conversion \
        --ckpt_dir $CKPT_DIR \
        --model_name $MODEL_NAME \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation 1 \
        --epochs $EPOCHS \
        --lr $LR \
        --max_length $MAX_LENGTH \
        --combine_method $COMBINE_METHOD \
        --name '$RUN_NAME'"

    if [ "$USE_BF16" = "true" ]; then
        CMD="$CMD --bf16"
    fi

    # 执行训练
    eval $CMD

    echo "训练完成! Checkpoint保存于: $CKPT_DIR/$RUN_NAME"
}

#========================================
# 测试模式（使用model.infer）
#========================================
test() {
    if [ -z "$CKPT_PATH" ]; then
        # 自动查找最新的checkpoint
        CKPT_PATH=$(ls -td "$CKPT_DIR"/*/ 2>/dev/null | head -1)
        if [ -d "$CKPT_PATH" ]; then
            CKPT_PATH=$(ls -d "$CKPT_PATH"checkpoint-* 2>/dev/null | head -1)
        fi
    fi

    if [ -z "$CKPT_PATH" ] || [ ! -d "$CKPT_PATH" ]; then
        echo "Error:Checkpoint路径不存在: $CKPT_PATH"
        exit 1
    fi

    echo "========================================"
    echo "LINGCONV Testing"
    echo "========================================"
    echo "Checkpoint: $CKPT_PATH"
    echo "========================================"

    /home/wlia0047/ar57/wenyu/envs/lingconv/bin/python3 -u << 'PYEOF'
import sys
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis')

import torch
from transformers import T5Tokenizer
from datasets import load_from_disk

# 从环境变量获取checkpoint路径
CKPT = """$CKPT_PATH"""

print("加载tokenizer...", flush=True)
tokenizer = T5Tokenizer.from_pretrained(CKPT)
print("tokenizer加载完成", flush=True)

print("加载模型...", flush=True)
from model import get_model

class ModelArgs:
    model_name = CKPT
    lng_dim = 40
    hidden_dim = 500
    disc_lng_dim = 40
    ling_dropout = 0.1
    initializer_range = 0.02
    combine_method = 'decoder_add_first'
    ling2_only = True
    ling_embed_type = 'one-layer'
    injection_type = 'first'
    injection_layer = 1
    combine_weight = 1.0
    use_semantic_pooling = False
    sem_loss = False
    sem_loss_type = 'dedicated'
    disc_loss = False
    pretrain_disc = False
    pretrain_sem = False
    pretrain_gen = False
    ling_vae = False
    linggen_type = 'none'
    max_length = 128
    ckpt = CKPT
    disc_ckpt = None
    sem_ckpt = None

m_args = ModelArgs()
model, _, _ = get_model(m_args, tokenizer, torch.device('cuda:0'))
model.eval()
model = model.cuda()
print("模型加载完成", flush=True)

print("加载测试数据...", flush=True)
raw_data = load_from_disk('/home/wlia0047/ar57_scratch/wenyu/ling_conversion_data')
print("数据加载完成", flush=True)

print("\n" + "="*80, flush=True)
print("测试结果", flush=True)
print("="*80, flush=True)

for i in range(10):
    sample = raw_data['test'][i]
    src = sample['sentence1']
    tgt = sample['sentence2']
    ling = sample['sentence2_ling']

    inputs = tokenizer(src, return_tensors='pt', padding=True, truncation=True, max_length=128)
    ling_tensor = torch.tensor([ling], dtype=torch.float32)

    # 使用model.infer()方法（正确调用方式）
    batch = {
        "input_ids": inputs['input_ids'].cuda(),
        "attention_mask": inputs['attention_mask'].cuda(),
        "sentence1_ling": ling_tensor.cuda(),
        "sentence2_ling": ling_tensor.cuda(),
        "labels": inputs['input_ids'].cuda(),
    }

    with torch.no_grad():
        pred = model.infer(batch)

    generated = tokenizer.decode(pred[0], skip_special_tokens=True)

    print(f"\n[样本 {i+1}]", flush=True)
    print(f"输入:   {src}", flush=True)
    print(f"目标:   {tgt}", flush=True)
    print(f"生成:   {generated}", flush=True)

print("\n" + "="*80, flush=True)
print("测试完成!", flush=True)
PYEOF
}

#========================================
# 推理模式（单条测试）
#========================================
infer() {
    if [ -z "$CKPT_PATH" ]; then
        CKPT_PATH=$(ls -td "$CKPT_DIR"/*/ 2>/dev/null | head -1)
        if [ -d "$CKPT_PATH" ]; then
            CKPT_PATH=$(ls -d "$CKPT_PATH"checkpoint-* 2>/dev/null | head -1)
        fi
    fi

    if [ -z "$CKPT_PATH" ] || [ ! -d "$CKPT_PATH" ]; then
        echo "Error:Checkpoint路径不存在: $CKPT_PATH"
        exit 1
    fi

    echo "使用checkpoint: $CKPT_PATH"

    /home/wlia0047/ar57/wenyu/envs/lingconv/bin/python3 -u << 'PYEOF'
import sys
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis')

import torch
from transformers import T5Tokenizer
from model import get_model

CKPT = """$CKPT_PATH"""

print("加载tokenizer...", flush=True)
tokenizer = T5Tokenizer.from_pretrained(CKPT)
print("tokenizer加载完成", flush=True)

print("加载模型...", flush=True)

class ModelArgs:
    model_name = CKPT
    lng_dim = 40
    hidden_dim = 500
    disc_lng_dim = 40
    ling_dropout = 0.1
    initializer_range = 0.02
    combine_method = 'decoder_add_first'
    ling2_only = True
    ling_embed_type = 'one-layer'
    injection_type = 'first'
    injection_layer = 1
    combine_weight = 1.0
    use_semantic_pooling = False
    sem_loss = False
    sem_loss_type = 'dedicated'
    disc_loss = False
    pretrain_disc = False
    pretrain_sem = False
    pretrain_gen = False
    ling_vae = False
    linggen_type = 'none'
    max_length = 128
    ckpt = CKPT
    disc_ckpt = None
    sem_ckpt = None

m_args = ModelArgs()
model, _, _ = get_model(m_args, tokenizer, torch.device('cuda:0'))
model.eval()
model = model.cuda()
print("模型加载完成", flush=True)

print("\n输入句子进行测试 (输入quit退出):", flush=True)

while IFS= read -r -p "> " line; do
    if [ "$line" = "quit" ] || [ "$line" = "q" ]; then
        break
    fi

    # 默认复杂度向量
    ling = torch.ones(40) * 0.5

    inputs = tokenizer(line, return_tensors='pt', padding=True, truncation=True, max_length=128)
    ling_tensor = ling.unsqueeze(0).cuda()

    batch = {
        "input_ids": inputs['input_ids'].cuda(),
        "attention_mask": inputs['attention_mask'].cuda(),
        "sentence1_ling": ling_tensor,
        "sentence2_ling": ling_tensor,
        "labels": inputs['input_ids'].cuda(),
    }

    with torch.no_grad():
        pred = model.infer(batch)

    generated = tokenizer.decode(pred[0], skip_special_tokens=True)
    print(f"生成: {generated}", flush=True)
done
PYEOF
}

#========================================
# 主入口
#========================================
case "$MODE" in
    train)
        train
        ;;
    test)
        test
        ;;
    infer)
        infer
        ;;
    *)
        echo "使用方法:"
        echo "  训练: bash $0 train --combine_method decoder_add_first [--epochs 2] [--batch_size 256] [--lr 1e-3]"
        echo "  测试: bash $0 test --ckpt <checkpoint_path>"
        echo "  推理: bash $0 infer [--ckpt <checkpoint_path>]"
        echo ""
        echo "示例:"
        echo "  bash $0 train --combine_method decoder_add_first"
        echo "  bash $0 train --combine_method bos_replace --epochs 3"
        echo "  bash $0 test"
        echo "  bash $0 infer"
        exit 1
        ;;
esac
