#!/usr/bin/env python3
"""使用高复杂度特征值生成查询 - 简化版"""

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

    # 高复杂度特征值 - 全0.8
    high_ling = torch.ones(40) * 0.8
    med_ling = torch.ones(40) * 0.5

    # 测试句子
    test_sentences = [
        "I bought an easel that is large and versatile",
        "The product has a great price and good quality",
        "It is perfect for my art studio",
        "I would recommend this to anyone",
        "The quality exceeded my expectations",
    ]

    print("=" * 70)
    print("高复杂度(0.8) vs 中等复杂度(0.5) 生成对比")
    print("=" * 70)

    for target_text in test_sentences:
        print(f"\n[输入] {target_text}")
        target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

        # 中等复杂度
        batch_med = {
            "input_ids": target_enc["input_ids"].to(device),
            "attention_mask": target_enc["attention_mask"].to(device),
            "sentence1_input_ids": target_enc["input_ids"].to(device),
            "sentence1_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_ling": med_ling.unsqueeze(0).to(device),
            "sentence1_ling": med_ling.unsqueeze(0).to(device),
            "labels": target_enc["input_ids"].to(device),
        }

        # 高复杂度
        batch_high = {
            "input_ids": target_enc["input_ids"].to(device),
            "attention_mask": target_enc["attention_mask"].to(device),
            "sentence1_input_ids": target_enc["input_ids"].to(device),
            "sentence1_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_ling": high_ling.unsqueeze(0).to(device),
            "sentence1_ling": high_ling.unsqueeze(0).to(device),
            "labels": target_enc["input_ids"].to(device),
        }

        set_seed(42)
        with torch.no_grad():
            pred_med = model.infer(batch_med)

        set_seed(42)
        with torch.no_grad():
            pred_high = model.infer(batch_high)

        med_text = tokenizer.decode(pred_med[0], skip_special_tokens=True)
        high_text = tokenizer.decode(pred_high[0], skip_special_tokens=True)

        print(f"[中0.5] {med_text}")
        print(f"[高0.8] {high_text}")

    # 再用用户真实复杂度值×2
    print("\n" + "=" * 70)
    print("使用用户真实复杂度×2")
    print("=" * 70)

    query_file = "/home/wlia0047/ar57/wenyu/result/personal_query/06_query/queries_A0069667HW6WCU78SMHF.json"

    try:
        with open(query_file) as f:
            query_data = json.load(f)

        target_complexity = query_data.get("target_complexity", {})

        user_ling = torch.tensor([
            min(target_complexity.get("subordinate_clause_freq", 0.5) * 2, 0.85),
            min(target_complexity.get("dep_distance", 0.5) * 2, 0.85),
            min(target_complexity.get("modifier_density", 0.5) * 2, 0.85),
            min(target_complexity.get("coord_chain", 0.5) * 2, 0.85),
            min(target_complexity.get("negation_scope", 0.5) * 2, 0.85),
            min(target_complexity.get("voice_ratio", 0.5) * 2, 0.85),
            min(target_complexity.get("branching_direction", 0.5) * 2, 0.85),
            min(target_complexity.get("advcl_freq", 0.5) * 2, 0.85),
            min(target_complexity.get("comp_clause_freq", 0.5) * 2, 0.85),
            min(target_complexity.get("fanout", 2.3) / 5.0 * 2, 0.85),
            min(target_complexity.get("parataxis_freq", 0.5) * 2, 0.85),
            min(target_complexity.get("prep_density", 0.5) * 2, 0.85),
            min(target_complexity.get("appos_freq", 0.5) * 2, 0.85),
        ] + [0.7] * 27)

        print(f"用户复杂度(×2): {user_ling[:13].tolist()}")
        print()

        for target_text in test_sentences[:3]:
            print(f"[输入] {target_text}")
            target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

            batch = {
                "input_ids": target_enc["input_ids"].to(device),
                "attention_mask": target_enc["attention_mask"].to(device),
                "sentence1_input_ids": target_enc["input_ids"].to(device),
                "sentence1_attention_mask": target_enc["attention_mask"].to(device),
                "sentence2_ling": user_ling.unsqueeze(0).to(device),
                "sentence1_ling": user_ling.unsqueeze(0).to(device),
                "labels": target_enc["input_ids"].to(device),
            }

            set_seed(42)
            with torch.no_grad():
                pred = model.infer(batch)

            pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
            print(f"[用户×2] {pred_text}")
            print()
    except Exception as e:
        print(f"读取用户配置失败: {e}")


if __name__ == "__main__":
    main()
