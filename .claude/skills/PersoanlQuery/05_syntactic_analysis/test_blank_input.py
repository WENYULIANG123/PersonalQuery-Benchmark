#!/usr/bin/env python3
"""
测试：使用空白/最小输入 + 目标复杂度 来生成句子
这样可以避免模型复制原文前缀
"""

import torch
from transformers import T5Tokenizer, set_seed
from model import get_model
from options import parse_args
import json


def main():
    args, _, _ = parse_args()

    args.ckpt = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis/checkpoints/0402_23-52-15-ling_conversion-decoder_add_first/checkpoint-17482"
    args.disc_ckpt = None
    args.sem_ckpt = None
    args.seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model, _, _ = get_model(args, tokenizer, device)
    model.eval()
    model.to(device)

    print("模型加载完成\n")

    # 不同复杂度目标
    ling_configs = [
        ("极低0.2", torch.ones(40) * 0.2),
        ("中0.5", torch.ones(40) * 0.5),
        ("高0.8", torch.ones(40) * 0.8),
    ]

    # 测试句子
    base_sentence = "I bought an easel that is large and versatile."

    print("=" * 70)
    print("方案1: 使用原始句子作为输入")
    print("=" * 70)
    target_enc = tokenizer(base_sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)

    for name, ling in ling_configs:
        batch = {
            "input_ids": target_enc["input_ids"].to(device),
            "attention_mask": target_enc["attention_mask"].to(device),
            "sentence1_input_ids": target_enc["input_ids"].to(device),
            "sentence1_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_ling": ling.unsqueeze(0).to(device),
            "sentence1_ling": ling.unsqueeze(0).to(device),
            "labels": target_enc["input_ids"].to(device),
        }

        set_seed(42)
        with torch.no_grad():
            pred = model.infer(batch)

        pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
        print(f"[{name}] {pred_text}")

    print("\n" + "=" * 70)
    print("方案2: 使用通用提示词作为输入（不包含具体内容）")
    print("=" * 70)

    # 通用提示词
    prompt = "I bought something that is good and useful."
    target_enc = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=128)

    for name, ling in ling_configs:
        batch = {
            "input_ids": target_enc["input_ids"].to(device),
            "attention_mask": target_enc["attention_mask"].to(device),
            "sentence1_input_ids": target_enc["input_ids"].to(device),
            "sentence1_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_ling": ling.unsqueeze(0).to(device),
            "sentence1_ling": ling.unsqueeze(0).to(device),
            "labels": target_enc["input_ids"].to(device),
        }

        set_seed(42)
        with torch.no_grad():
            pred = model.infer(batch)

        pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
        print(f"[{name}] {pred_text}")

    print("\n" + "=" * 70)
    print("方案3: 使用最小输入 + 强ling控制")
    print("=" * 70)

    # 最小输入
    min_prompt = "It is good."
    target_enc = tokenizer(min_prompt, return_tensors='pt', padding=True, truncation=True, max_length=128)

    # 增强ling特征（×1.5，但限制在0.95）
    for name, ling in ling_configs:
        enhanced_ling = torch.clamp(ling * 1.5, 0, 0.95)
        batch = {
            "input_ids": target_enc["input_ids"].to(device),
            "attention_mask": target_enc["attention_mask"].to(device),
            "sentence1_input_ids": target_enc["input_ids"].to(device),
            "sentence1_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_ling": enhanced_ling.unsqueeze(0).to(device),
            "sentence1_ling": enhanced_ling.unsqueeze(0).to(device),
            "labels": target_enc["input_ids"].to(device),
        }

        set_seed(42)
        with torch.no_grad():
            pred = model.infer(batch)

        pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
        print(f"[{name}×1.5] {pred_text}")

    print("\n" + "=" * 70)
    print("方案4: 更高温度采样（增加多样性）")
    print("=" * 70)

    # 回到原始句子测试
    target_enc = tokenizer(base_sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)

    # 注意：当前代码使用固定的temperature=0.7
    # 如需更高多样性，需要修改model.py中的temperature值

    for name, ling in ling_configs:
        batch = {
            "input_ids": target_enc["input_ids"].to(device),
            "attention_mask": target_enc["attention_mask"].to(device),
            "sentence1_input_ids": target_enc["input_ids"].to(device),
            "sentence1_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_ling": ling.unsqueeze(0).to(device),
            "sentence1_ling": ling.unsqueeze(0).to(device),
            "labels": target_enc["input_ids"].to(device),
        }

        # 多次采样
        results = []
        for seed in [42, 100, 200]:
            set_seed(seed)
            with torch.no_grad():
                pred = model.infer(batch)
            pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
            results.append(pred_text)

        print(f"[{name}]")
        for r in results:
            print(f"  - {r}")


if __name__ == "__main__":
    main()
