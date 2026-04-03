#!/usr/bin/env python3
"""使用用户提供的例句测试"""

import torch
from transformers import T5Tokenizer, set_seed
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

    # 用户提供的例句
    user_sentences = [
        "The black dog is running through the grass with the other dogs running.",
        "A black dog is running behind a large black cat, running through the air.",
        "The dog is running slowly behind a cat while running in the air.",
        "A black dog is running on a stray cat in a field with trees behind.",
        "A dog is running behind a large black cat with its mouth closed.",
    ]

    # 不同复杂度
    ling_configs = [
        ("中0.5", torch.ones(40) * 0.5),
        ("高0.8", torch.ones(40) * 0.8),
    ]

    print("=" * 70)
    print("使用用户例句测试不同复杂度")
    print("=" * 70)

    for target_text in user_sentences:
        print(f"\n[输入] {target_text}")

        target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

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


if __name__ == "__main__":
    main()
