#!/usr/bin/env python3
"""使用查询语句生成"""

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

    # 加载查询文件
    query_file = "/home/wlia0047/ar57/wenyu/result/personal_query/06_query/all_queries_summary_v2.json"
    with open(query_file) as f:
        data = json.load(f)

    results = data.get("results", [])

    # 选取前3个用户的查询
    for user_data in results[:3]:
        user_id = user_data.get("user_id", "unknown")
        style_desc = user_data.get("style_description", "")[:200]
        target_complexity = user_data.get("target_complexity", {})
        attributes = user_data.get("attributes", {})

        print("=" * 70)
        print(f"用户ID: {user_id}")
        print(f"风格: {style_desc}...")
        print(f"属性: {attributes}")
        print("=" * 70)

        # 构建高复杂度特征值 (用户真实值×2)
        ling_values = torch.tensor([
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

        # 生成基于属性的查询句子
        query_templates = [
            f"I bought an {attributes.get('A1', 'item')} that is {attributes.get('A4', 'good')} and {attributes.get('A5', 'useful')}.",
            f"The {attributes.get('A1', 'product')} from {attributes.get('A2', 'this company')} has a great price of {attributes.get('A3', 'reasonable')}.",
            "It is perfect for my needs and very easy to use.",
            "I would recommend this to anyone looking for quality.",
            "The quality exceeded my expectations.",
        ]

        for target_text in query_templates:
            print(f"\n[原始查询] {target_text}")

            target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

            # 用高复杂度生成3次
            for seed in [42, 123, 456]:
                set_seed(seed)

                batch = {
                    "input_ids": target_enc["input_ids"].to(device),
                    "attention_mask": target_enc["attention_mask"].to(device),
                    "sentence1_input_ids": target_enc["input_ids"].to(device),
                    "sentence1_attention_mask": target_enc["attention_mask"].to(device),
                    "sentence2_ling": ling_values.unsqueeze(0).to(device),
                    "sentence1_ling": ling_values.unsqueeze(0).to(device),
                    "labels": target_enc["input_ids"].to(device),
                }

                with torch.no_grad():
                    pred = model.infer(batch)

                pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
                print(f"  [seed={seed}] {pred_text}")
        print()


if __name__ == "__main__":
    main()
