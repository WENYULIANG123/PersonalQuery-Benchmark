#!/usr/bin/env python3
"""
测试语义池化效果：pooling后decoder只能看到语义摘要，必须依赖ling embedding
"""

import torch
from transformers import T5Tokenizer, set_seed
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/LingConv')

from model import get_model
from options import parse_args


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
    print("模式1: 原始模式 (decoder能看到完整encoder输出)")
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
    print("模式2: 语义池化模式 (decoder只能看到语义摘要)")
    print("=" * 70)

    for name, ling in ling_configs:
        # 手动调用encode和decode，使用语义池化
        batch = {
            "input_ids": target_enc["input_ids"].to(device),
            "attention_mask": target_enc["attention_mask"].to(device),
            "sentence1_input_ids": target_enc["input_ids"].to(device),
            "sentence1_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_ling": ling.unsqueeze(0).to(device),
            "sentence1_ling": ling.unsqueeze(0).to(device),
            "labels": target_enc["input_ids"].to(device),
        }

        # Encode
        model.eval()
        encoder_outputs, encoder_attention_mask, cache = model.encode(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            sentence1_ling=batch.get("sentence1_ling"),
            sentence2_ling=batch.get("sentence2_ling"),
        )

        # Decode with semantic pooling
        set_seed(42)
        with torch.no_grad():
            dec_output, cache2 = model.decode(
                sentence2_input_ids=batch.get("labels"),
                sentence1_ling=batch.get("sentence1_ling"),
                sentence2_ling=batch.get("sentence2_ling"),
                encoder_outputs=encoder_outputs,
                encoder_attention_mask=encoder_attention_mask,
                generate=True,
                use_semantic_pooling=True,  # 开启语义池化
            )

        pred_text = tokenizer.decode(dec_output[0], skip_special_tokens=True)
        print(f"[{name}] {pred_text}")

    print("\n" + "=" * 70)
    print("模式2效果验证: 同一个句子，不同复杂度")
    print("=" * 70)

    same_sentence = "The product is good and useful."
    target_enc = tokenizer(same_sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)

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

        # Encode
        encoder_outputs, encoder_attention_mask, cache = model.encode(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            sentence1_ling=batch.get("sentence1_ling"),
            sentence2_ling=batch.get("sentence2_ling"),
        )

        # Decode with semantic pooling
        set_seed(42)
        with torch.no_grad():
            dec_output, cache2 = model.decode(
                sentence2_input_ids=batch.get("labels"),
                sentence1_ling=batch.get("sentence1_ling"),
                sentence2_ling=batch.get("sentence2_ling"),
                encoder_outputs=encoder_outputs,
                encoder_attention_mask=encoder_attention_mask,
                generate=True,
                use_semantic_pooling=True,
            )

        pred_text = tokenizer.decode(dec_output[0], skip_special_tokens=True)
        print(f"[{name}] {pred_text}")


if __name__ == "__main__":
    main()
