#!/usr/bin/env python3
"""使用高复杂度特征值生成查询"""

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

    # 从真实查询配置加载
    query_file = "/home/wlia0047/ar57/wenyu/result/personal_query/06_query/queries_A0069667HW6WCU78SMHF.json"
    with open(query_file) as f:
        query_data = json.load(f)

    attributes = query_data.get("attributes", {})
    style_desc = query_data.get("style_description", "")

    print("=" * 70)
    print("用户风格: ", style_desc[:200])
    print("=" * 70)

    # 高复杂度特征值 - 将用户真实值乘以2-3倍并限制在0.8以内
    target_complexity = query_data.get("target_complexity", {})

    def scale_val(val, scale=3.0, max_val=0.85):
        return min(val * scale, max_val)

    ling_values = torch.tensor([
        scale_val(target_complexity.get("subordinate_clause_freq", 0.5), 3.0),
        scale_val(target_complexity.get("dep_distance", 0.5), 2.0),
        scale_val(target_complexity.get("modifier_density", 0.5), 2.0),
        scale_val(target_complexity.get("coord_chain", 0.5), 2.0),
        scale_val(target_complexity.get("negation_scope", 0.5), 3.0),
        scale_val(target_complexity.get("voice_ratio", 0.5), 2.0),
        scale_val(target_complexity.get("branching_direction", 0.5), 2.0),
        scale_val(target_complexity.get("advcl_freq", 0.5), 3.0),
        scale_val(target_complexity.get("comp_clause_freq", 0.5), 3.0),
        scale_val(target_complexity.get("fanout", 0.5) / 5.0, 2.0),
        scale_val(target_complexity.get("parataxis_freq", 0.5), 3.0),
        scale_val(target_complexity.get("prep_density", 0.5), 2.0),
        scale_val(target_complexity.get("appos_freq", 0.5), 3.0),
    ] + [0.7] * 27)  # 其他维度设为0.7

    print(f"缩放后特征值(前13维): {ling_values[:13].tolist()}")
    print()

    # 测试句子 - 基于用户属性的多样化句子
    test_sentences = [
        f"I bought an {attributes.get('A1', 'easel')} that is {attributes.get('A4', 'large')} and {attributes.get('A5', 'versatile')}.",
        f"The {attributes.get('A1', 'product')} from {attributes.get('A2', 'this brand')} has a great price of {attributes.get('A3', 'reasonable')}.",
        "It is perfect for my art studio and very easy to set up.",
        "I would recommend this to anyone who enjoys creative work.",
        "The quality exceeded my expectations and arrived quickly.",
        "Setting up the easel was simple and the instructions were clear.",
        "It folds nicely and stores away when not in use.",
    ]

    print("=" * 70)
    print("高复杂度生成结果 (采样5次取最优)")
    print("=" * 70)

    for idx, target_text in enumerate(test_sentences):
        print(f"\n[输入] {target_text}")

        target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

        # 运行5次采样，取loss最低的结果
        best_pred = None
        best_score = float('-inf')

        for seed in range(5):
            set_seed(seed * 100)

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
                pred, cache = model.infer_with_cache(batch)
                logits = cache.get('logits')

            pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)

            # 计算伪log prob分数
            if logits is not None:
                probs = torch.log_softmax(logits, dim=-1)
                seq_len = pred[0].shape[0] if len(pred[0].shape) > 0 else 0
                if seq_len > 0:
                    score = probs[0, :seq_len].diagonal().mean().item()
                    if score > best_score:
                        best_score = score
                        best_pred = pred_text

        if best_pred:
            print(f"[输出] {best_pred}")

    print("\n" + "=" * 70)
    print("对比: 不同复杂度级别")
    print("=" * 70)

    base = "I bought an easel that is large and versatile."
    target_enc = tokenizer(base, return_tensors='pt', padding=True, truncation=True, max_length=128)

    for ling_name, ling in [
        ("原始(×1)", ling_values),
        ("×1.5", ling_values * 1.5),
        ("×2.0", ling_values * 2.0),
    ]:
        ling_clamped = torch.clamp(ling, 0, 0.95)
        batch = {
            "input_ids": target_enc["input_ids"].to(device),
            "attention_mask": target_enc["attention_mask"].to(device),
            "sentence1_input_ids": target_enc["input_ids"].to(device),
            "sentence1_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_ling": ling_clamped.unsqueeze(0).to(device),
            "sentence1_ling": ling_clamped.unsqueeze(0).to(device),
            "labels": target_enc["input_ids"].to(device),
        }

        set_seed(42)
        with torch.no_grad():
            pred = model.infer(batch)

        pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
        print(f"[{ling_name}] {pred_text}")


if __name__ == "__main__":
    main()
