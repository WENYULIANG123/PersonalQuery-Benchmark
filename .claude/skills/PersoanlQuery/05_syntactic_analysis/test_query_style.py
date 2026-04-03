#!/usr/bin/env python3
"""使用真实查询复杂度配置测试 LingConv"""

import torch
from transformers import T5Tokenizer, set_seed
from model import get_model
from options import parse_args
import json


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

    # 从真实查询配置加载目标复杂度
    query_file = "/home/wlia0047/ar57/wenyu/result/personal_query/06_query/queries_A0069667HW6WCU78SMHF.json"
    with open(query_file) as f:
        query_data = json.load(f)

    target_complexity = query_data.get("target_complexity", {})
    style_desc = query_data.get("style_description", "")
    attributes = query_data.get("attributes", {})

    print("=" * 70)
    print("用户查询配置")
    print("=" * 70)
    print(f"风格描述: {style_desc[:150]}...")
    print(f"目标复杂度: {json.dumps(target_complexity, indent=2)}")
    print(f"属性: {attributes}")
    print()

    # 将40维复杂度配置转换为tensor (只取前40维)
    ling_values = torch.tensor([
        target_complexity.get("subordinate_clause_freq", 0.5),
        target_complexity.get("dep_distance", 0.5),
        target_complexity.get("modifier_density", 0.5),
        target_complexity.get("coord_chain", 0.5),
        target_complexity.get("negation_scope", 0.5),
        target_complexity.get("voice_ratio", 0.5),
        target_complexity.get("branching_direction", 0.5),
        target_complexity.get("advcl_freq", 0.5),
        target_complexity.get("comp_clause_freq", 0.5),
        target_complexity.get("fanout", 0.5) / 5.0,  # 归一化
        target_complexity.get("parataxis_freq", 0.5),
        target_complexity.get("prep_density", 0.5),
        target_complexity.get("appos_freq", 0.5),
    ] + [0.5] * 27)  # 补足40维

    print("测试句子 (基于用户属性):")
    print("-" * 70)

    # 基于属性构建测试句子
    test_sentences = [
        f"I bought an {attributes.get('A1', 'easel')} that is {attributes.get('A4', 'large')} and {attributes.get('A5', 'versatile')}.",
        f"The {attributes.get('A1', 'product')} from {attributes.get('A2', 'this brand')} has a great price of {attributes.get('A3', 'reasonable')}.",
        "It is perfect for my art studio and very easy to set up.",
        "I would recommend this to anyone who enjoys creative work.",
        "The quality exceeded my expectations and arrived quickly.",
    ]

    for idx, target_text in enumerate(test_sentences):
        print(f"\n测试 {idx+1}: {target_text}")

        target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

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
        print(f"  -> {pred_text}")

    # 也测试一下不同复杂度级别
    print("\n" + "=" * 70)
    print("不同复杂度级别对比:")
    print("-" * 70)

    base_sentence = "I bought an easel that is large and versatile."
    target_enc = tokenizer(base_sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)

    complexity_levels = [
        ("极低复杂度", torch.ones(40) * 0.2),
        ("中等复杂度", torch.ones(40) * 0.5),
        ("高复杂度", torch.ones(40) * 0.8),
    ]

    for name, ling in complexity_levels:
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
        print(f"[{name}] {pred_text}")


if __name__ == "__main__":
    main()
