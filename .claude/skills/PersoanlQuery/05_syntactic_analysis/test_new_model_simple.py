#!/usr/bin/env python3
"""简单测试新 LingConv 模型的生成质量"""

import torch
from transformers import T5Tokenizer, set_seed
from model import get_model
from options import parse_args


def main():
    args, _, _ = parse_args()

    # 使用新的 T5-v1.1-xl 模型
    args.ckpt = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis/checkpoints/0402_23-52-15-ling_conversion-decoder_add_first/checkpoint-17482"
    args.disc_ckpt = None  # 不使用 LingDisc
    args.sem_ckpt = None    # 不使用 SemEmb
    args.seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model, _, _ = get_model(args, tokenizer, device)
    model.eval()
    model.to(device)

    print("模型加载完成\n")

    # 测试句子
    test_sentences = [
        "The dog is running after the cat",
        "She opened the door and walked in",
        "The book that I bought yesterday is interesting",
        "He sang while he was walking",
        "The teacher explained the lesson to the students",
    ]

    for idx, target_text in enumerate(test_sentences):
        print("=" * 70)
        print(f"测试 {idx+1}: {target_text}")
        print("-" * 70)

        target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

        # 使用不同的ling特征值生成3个不同版本
        ling_values = [
            ("原始特征", torch.ones(40) * 0.5),
            ("高复杂度", torch.ones(40) * 0.8),
            ("低复杂度", torch.ones(40) * 0.3),
        ]

        for name, ling in ling_values:
            batch = {
                "input_ids": target_enc["input_ids"].to(device),
                "attention_mask": target_enc["attention_mask"].to(device),
                "sentence1_input_ids": target_enc["input_ids"].to(device),
                "sentence1_attention_mask": target_enc["attention_mask"].to(device),
                "sentence2_ling": ling.unsqueeze(0).to(device),
                "sentence1_ling": ling.unsqueeze(0).to(device),
                "labels": target_enc["input_ids"].to(device),
            }

            # 使用 infer 方法
            with torch.no_grad():
                pred = model.infer(batch)

            pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
            print(f"  [{name}] {pred_text}")

        print()


if __name__ == "__main__":
    main()
