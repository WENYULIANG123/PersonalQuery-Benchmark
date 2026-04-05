#!/usr/bin/env python3
"""
LINGCONV 14维特征训练脚本
=========================
使用回译数据（14维spaCy特征）训练LINGCONV模型

使用方法:
    python 05_lingconv_14d.py
"""
import os
import sys
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    T5Tokenizer,
)

# 添加LingConv路径
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/LingConv')
from data import load_backtrans_data, LingDataCollator14D
from model import get_model


def log_with_timestamp(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main():
    # ============ 硬编码参数 ============
    DATA_FILE = "/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/user_sentences/all_users_merged.json"
    MODEL_NAME = "google/flan-t5-base"
    OUTPUT_DIR = "/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints_14d"
    MAX_LENGTH = 200
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION = 4  # 等效 batch_size = 32 * 4 = 128
    EPOCHS = 10
    LR = 1e-3
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    RNG_SEED = 42
    TEST_SIZE = 0.2

    log_with_timestamp("=" * 80)
    log_with_timestamp("LINGCONV 14D Training (Back-translation Data)")
    log_with_timestamp("=" * 80)

    # 设置随机种子
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    # 加载tokenizer
    log_with_timestamp(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    # 加载回译数据
    log_with_timestamp(f"Loading back-translation data: {DATA_FILE}")
    dataset = load_backtrans_data(
        json_file=DATA_FILE,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        test_size=TEST_SIZE,
    )

    if "train" not in dataset:
        raise ValueError("No training data found!")

    log_with_timestamp(f"Train samples: {len(dataset['train'])}")
    if "test" in dataset:
        log_with_timestamp(f"Test samples: {len(dataset['test'])}")

    # 创建模型参数
    class Args:
        combine_method = "decoder_add_first"
        ling2_only = True
        use_semantic_pooling = False
        sem_loss = False
        disc_loss = False
        combine_weight = 1.0
        hidden_dim = 500
        lng_dim = 14  # 关键改变：14维特征
        ling_dropout = 0.1
        initializer_range = 0.02
        model_name = MODEL_NAME
        pretrain_disc = False
        disc_ckpt = None
        disc_type = "t5"
        ckpt = None
        ling_embed_type = "one-layer"
        injection_type = "first"
        injection_layer = 1
        ling_vae = False
        latent_dim = 150
        sem_loss_type = "dedicated"
        use_lingpred = False
        process_lingpred = False
        aug_same = False
        freeze_lm = False
        max_length = MAX_LENGTH
        feedback_param = "logits"
        sem_ckpt = None
        disc_lng_dim = 40

    args = Args()

    # 获取模型
    log_with_timestamp("Creating model...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log_with_timestamp(f"Device: {device}")

    model, _, _ = get_model(args, tokenizer, device)
    model.train()

    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if "test" in dataset else "no",
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        save_total_limit=2,
        fp16=False,  # t5-base不支持fp16
        dataloader_num_workers=4,
        seed=RNG_SEED,
        report_to="none",
    )

    # DataCollator
    data_collator = LingDataCollator14D(tokenizer)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 开始训练
    log_with_timestamp("Starting training...")
    trainer.train()

    # 保存模型
    final_ckpt = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_ckpt)
    tokenizer.save_pretrained(final_ckpt)
    log_with_timestamp(f"Model saved to: {final_ckpt}")

    log_with_timestamp("=" * 80)
    log_with_timestamp("Training Complete!")
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    main()
