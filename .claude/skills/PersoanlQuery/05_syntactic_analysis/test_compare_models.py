#!/usr/bin/env python3
"""对比不同特征值下的生成效果"""

import torch
from transformers import T5Tokenizer, set_seed
from model import get_model
from options import parse_args


def main():
    args, _, _ = parse_args()

    # 使用新的 T5-v1.1-xl 模型
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

    # 测试句子
    test_sentences = [
        "The dog is running after the cat",
        "She opened the door and walked in",
        "I bought an easel that is large and versatile",
    ]

    # 特征值对比
    ling_configs = [
        ("全0.2(极低复杂度)", torch.ones(40) * 0.2),
        ("全0.5(中等复杂度)", torch.ones(40) * 0.5),
        ("全0.8(高复杂度)", torch.ones(40) * 0.8),
        ("真实用户值", torch.tensor([0.018, 0.228, 0.168, 0.207, 0.012, 0.071, 0.512, 0.024, 0.052, 0.470, 0.001, 0.061, 0.004] + [0.5]*27)),
    ]

    for target_text in test_sentences:
        print("=" * 70)
        print(f"输入: {target_text}")
        print("=" * 70)

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

            with torch.no_grad():
                pred = model.infer(batch)

            pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
            print(f"[{name:20s}] {pred_text}")
        print()


if __name__ == "__main__":
    main()
