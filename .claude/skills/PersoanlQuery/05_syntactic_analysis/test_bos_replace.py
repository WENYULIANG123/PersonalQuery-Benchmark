#!/usr/bin/env python3
"""
测试 bos_replace 方法训练出来的模型的生成效果
"""

import torch
from transformers import T5Tokenizer, set_seed
import sys
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis')

from model import get_model
from options import parse_args


def main():
    args, _, _ = parse_args()

    # 使用训练好的 bos_replace 模型
    args.ckpt = "/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints/0403_13-58-13-ling_conversion-bos_replace/checkpoint-17000"
    args.disc_ckpt = None
    args.sem_ckpt = None
    args.seed = 42
    args.combine_method = "bos_replace"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model, _, _ = get_model(args, tokenizer, device)
    model.eval()
    model.to(device)

    print(f"combine_method = {model.args.combine_method}")
    print("模型加载完成\n")

    # 不同复杂度目标
    ling_configs = [
        ("极低0.2", torch.ones(40) * 0.2),
        ("中0.5", torch.ones(40) * 0.5),
        ("高0.8", torch.ones(40) * 0.8),
    ]

    # 测试句子
    test_sentences = [
        "I bought an easel that is large and versatile.",
        "The product has a great price and good quality.",
        "It is perfect for my art studio.",
        "I would recommend this to anyone.",
        "The quality exceeded my expectations.",
    ]

    print("=" * 70)
    print("BOS Replace 方法测试: 不同复杂度")
    print("=" * 70)

    for target_text in test_sentences:
        print(f"\n[输入] {target_text}")

        target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

        for name, ling in ling_configs:
            batch = {
                "input_ids": target_enc["input_ids"].to(device),
                "attention_mask": target_enc["attention_mask"].to(device),
                "sentence1_input_ids": target_enc["input_ids"].to(device),
                "sentence1_attention_mask": target_enc["attention_mask"].to(device),
                "sentence2_input_ids": target_enc["input_ids"].to(device),
                "sentence2_attention_mask": target_enc["attention_mask"].to(device),
                "sentence2_ling": ling.unsqueeze(0).to(device),
                "sentence1_ling": ling.unsqueeze(0).to(device),
                "labels": target_enc["input_ids"].to(device),
            }

            set_seed(42)
            with torch.no_grad():
                pred = model.infer(batch)

            pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
            print(f"  [{name}] {pred_text}")

    print("\n" + "=" * 70)
    print("对比: 同一句子，不同seed")
    print("=" * 70)

    base_sentence = "The product is good and useful."
    target_enc = tokenizer(base_sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)

    ling = torch.ones(40) * 0.8

    for seed in [42, 100, 200, 300]:
        batch = {
            "input_ids": target_enc["input_ids"].to(device),
            "attention_mask": target_enc["attention_mask"].to(device),
            "sentence1_input_ids": target_enc["input_ids"].to(device),
            "sentence1_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_input_ids": target_enc["input_ids"].to(device),
            "sentence2_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_ling": ling.unsqueeze(0).to(device),
            "sentence1_ling": ling.unsqueeze(0).to(device),
            "labels": target_enc["input_ids"].to(device),
        }

        set_seed(seed)
        with torch.no_grad():
            pred = model.infer(batch)

        pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
        print(f"[seed={seed}] {pred_text}")


if __name__ == "__main__":
    main()
