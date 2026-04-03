#!/usr/bin/env python3
"""
LingConv Training with Semantic Loss (联合训练版)

这个脚本实现了语义损失的联合训练：
1. LM Loss: 标准语言建模损失
2. Sem Loss: 语义相似度损失 - 确保源句和目标句的语义一致

原理:
- 使用共享的 T5 encoder 同时编码源句和目标句
- 计算两者表示的余弦相似度，最大化语义一致性

用法:
    # 基础训练 (无语义损失)
    python train_with_sem_loss.py

    # 启用语义损失 (权重 0.1)
    python train_with_sem_loss.py --sem_loss --sem_loss_weight 0.1

    # 从检查点继续
    python train_with_sem_loss.py --ckpt ./checkpoints/xxx --sem_loss
"""

import os
import sys
import json
import argparse
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, set_seed

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import LingDataCollator, load_data
from model import EncoderDecoderVAE
from options import parse_args


class SemanticLossComputer:
    """计算语义相似度损失的辅助类"""

    def __init__(self, model, device, loss_weight=0.1):
        self.model = model
        self.device = device
        self.loss_weight = loss_weight

    def compute(self, input_ids, attention_mask, labels):
        """计算语义损失

        通过 encoder 编码源句和目标句，计算余弦相似度
        """
        try:
            # 编码源句
            source_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            source_emb = source_outputs.last_hidden_state.mean(1)  # [batch, hidden]

            # 获取目标句的表示 (使用 decoder 的嵌入)
            if labels is not None:
                # 移位得到 decoder 输入
                decoder_input_ids = self.model._shift_right(labels)
            else:
                return None, 0

            # 通过 decoder (不更新梯度)
            with torch.no_grad():
                decoder_embeds = self.model.decoder.embed_tokens(decoder_input_ids)
                decoder_outputs = self.model.decoder(
                    inputs_embeds=decoder_embeds,
                    encoder_hidden_states=source_outputs.last_hidden_state,
                    encoder_attention_mask=attention_mask,
                )
                target_emb = decoder_outputs.last_hidden_state.mean(1)

            # 计算余弦相似度
            cos_sim = F.cosine_similarity(source_emb, target_emb, dim=-1)
            # 目标：最大化相似度，损失为 1 - sim
            sem_loss = (1 - cos_sim).mean()

            return sem_loss, self.loss_weight * sem_loss

        except Exception as e:
            print(f"Warning: Semantic loss failed: {e}")
            return None, 0


class LingConvTrainer:
    """LingConv 训练器，支持语义损失"""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir,
        sem_loss_weight=0.0,
        lr=1e-3,
        weight_decay=1e-2,
        epochs=2,
        batch_size=16,
        eval_batch_size=32,
        eval_steps=500,
        max_grad_norm=1.0,
        device=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.sem_loss_weight = sem_loss_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_steps = eval_steps
        self.max_grad_norm = max_grad_norm

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # 数据整理器
        self.collator = LingDataCollator(tokenizer)

        # 语义损失计算器
        self.sem_computer = SemanticLossComputer(model, self.device, sem_loss_weight)

        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float("inf")

    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        if is_best:
            path = os.path.join(self.output_dir, "best_model")
        else:
            path = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Saved checkpoint to {path}")

    def evaluate(self):
        """评估"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                sentence2_ling = batch.get("sentence2_ling")
                if sentence2_ling is not None:
                    sentence2_ling = sentence2_ling.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    sentence2_ling=sentence2_ling,
                )

                total_loss += outputs.loss.item()
                num_batches += 1

        self.model.train()
        return {"eval_loss": total_loss / num_batches}

    def train_step(self, batch):
        """单步训练"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        sentence2_ling = batch.get("sentence2_ling")
        if sentence2_ling is not None:
            sentence2_ling = sentence2_ling.to(self.device)

        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            sentence2_ling=sentence2_ling,
        )

        # LM 损失
        lm_loss = outputs.loss

        # 语义损失
        sem_loss_value = 0
        if self.sem_loss_weight > 0:
            sem_loss, weighted_sem = self.sem_computer.compute(input_ids, attention_mask, labels)
            if sem_loss is not None:
                total_loss = lm_loss + weighted_sem
                sem_loss_value = sem_loss.item()
            else:
                total_loss = lm_loss
        else:
            total_loss = lm_loss

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "lm_loss": lm_loss.item(),
            "sem_loss": sem_loss_value,
            "total_loss": total_loss.item(),
        }

    def train(self):
        """完整训练"""
        self.model.train()
        print("=" * 70)
        print("LingConv Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Semantic Loss Weight: {self.sem_loss_weight}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Output: {self.output_dir}")
        print("=" * 70)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses = []
            epoch_sem_losses = []

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collator,
            )

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for batch in pbar:
                loss_dict = self.train_step(batch)
                self.global_step += 1

                epoch_losses.append(loss_dict["lm_loss"])
                if loss_dict["sem_loss"] > 0:
                    epoch_sem_losses.append(loss_dict["sem_loss"])

                # 更新进度条
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "sem": f"{sum(epoch_sem_losses)/len(epoch_sem_losses) if epoch_sem_losses else 0:.4f}" if epoch_sem_losses else "N/A"
                })

                # 定期评估
                if self.global_step % self.eval_steps == 0 and self.eval_dataset is not None:
                    eval_metrics = self.evaluate()
                    print(f"\nStep {self.global_step}: {eval_metrics}")

                    if eval_metrics["eval_loss"] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics["eval_loss"]
                        self.save_checkpoint(is_best=True)
                        print(f"New best eval_loss: {self.best_eval_loss:.4f}")

            # Epoch 结束
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch+1} complete: avg_loss={avg_epoch_loss:.4f}")

            # 每个 epoch 结束后保存
            self.save_checkpoint()

        print("=" * 70)
        print("Training complete!")
        print("=" * 70)


def main():
    args, _, _ = parse_args()

    # 构建输出目录
    timestamp = datetime.now().strftime("%m%d_%H-%M-%S")
    suffix = "_sem" if args.sem_loss else ""
    output_name = f"{timestamp}-ling_conversion{suffix}"
    output_dir = os.path.join(args.ckpt_dir, output_name)

    sem_loss_weight = getattr(args, 'sem_loss_weight', 0.1)
    print("=" * 70)
    print("LingConv Training with Semantic Loss")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Semantic Loss: {args.sem_loss} (weight: {sem_loss_weight})")
    print("=" * 70)

    # 设置随机种子
    set_seed(args.seed)

    # 加载 tokenizer
    print("\nLoading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    # 加载数据
    print("Loading data...")
    data, _, _ = load_data(args, tokenizer, return_data=True)
    print(f"Train: {len(data['train'])} samples")
    if 'dev' in data:
        print(f"Dev: {len(data['dev'])} samples")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    config = {
        "sem_loss": args.sem_loss,
        "sem_loss_weight": sem_loss_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "combine_method": args.combine_method,
        "ling2_only": args.ling2_only,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 加载模型
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用 get_model 获取模型
    from model import get_model
    model, _, _ = get_model(args, tokenizer, device)
    model.train()

    # 创建训练器
    trainer = LingConvTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data.get("dev"),
        output_dir=output_dir,
        sem_loss_weight=sem_loss_weight if args.sem_loss else 0.0,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_steps=500,
        max_grad_norm=args.max_grad_norm,
        device=device,
    )

    # 开始训练
    trainer.train()

    # 保存最终模型
    print(f"\nSaving final model to {output_dir}...")
    trainer.save_checkpoint()
    print("Done!")


if __name__ == "__main__":
    main()
