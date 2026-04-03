#!/usr/bin/env python3
"""多候选生成+多样性选择"""

import torch
from transformers import T5Tokenizer, set_seed
from model import get_model
from options import parse_args


def compute_prefix_overlap(text1, text2):
    """计算两个文本的单词级前缀重叠度"""
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    prefix_len = 0
    for j in range(min(len(words1), len(words2))):
        if words1[j] == words2[j]:
            prefix_len += 1
        else:
            break
    return prefix_len / len(words1) if words1 else 0


def analyze_sentence(text):
    """分析句子结构"""
    words = text.lower().split()
    clause_count = 0
    for i, word in enumerate(words):
        if word in ['which', 'that', 'where', 'when', 'because', 'although', 'while', 'if', 'unless', 'since', 'though', 'after', 'before', 'until']:
            clause_count += 1
        if word == 'and' and i > 0 and words[i-1] == ',':
            clause_count += 1
        if word == 'but' and i > 0 and words[i-1] == ',':
            clause_count += 1
    return clause_count


def main():
    args, _, _ = parse_args()
    args.ckpt = "/home/wlia0047/ar57_scratch/wenyu/LingConv_models/0402_17-19-36-ling_conversion_sem/best_model"
    args.disc_ckpt = "/home/wlia0047/ar57_scratch/wenyu/LingConv_models/0402_18-02-15-ling_disc-t5/best_ling_disc"
    args.disc_type = "t5"
    args.sem_ckpt = None
    args.sem_loss = True
    args.sem_loss_type = "shared"
    args.predict_with_feedback = True
    args.feedback_param = 'l'
    args.seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model, ling_disc, sem_emb = get_model(args, tokenizer, device)
    model.eval()
    model.to(device)
    ling_disc.eval()
    ling_disc.to(device)
    if sem_emb is not None:
        sem_emb.eval()
        sem_emb.to(device)

    print("模型加载完成\n")

    target_text = "The dog is running after the cat"
    target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

    print("=" * 70)
    print(f"输入: {target_text}")
    print("=" * 70)

    # 多次生成，使用不同种子
    n_candidates = 10
    all_candidates = []

    for seed in range(n_candidates):
        set_seed(seed * 10 + 42)  # 不同种子
        ling_val = 0.70
        mod_ling = torch.ones(40) * ling_val

        batch = {
            "input_ids": target_enc["input_ids"].to(device),
            "attention_mask": target_enc["attention_mask"].to(device),
            "sentence1_input_ids": target_enc["input_ids"].to(device),
            "sentence1_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_ling": mod_ling.unsqueeze(0).to(device),
            "sentence1_ling": mod_ling.unsqueeze(0).to(device),
            "labels": target_enc["input_ids"].to(device),
        }

        try:
            with torch.no_grad():
                prediction_ids, feedback_trace = model.infer_with_feedback_BP(
                    ling_disc=ling_disc,
                    sem_emb=sem_emb,
                    batch=batch,
                    tokenizer=tokenizer
                )
            pred_text = tokenizer.decode(prediction_ids[0], skip_special_tokens=True)
            all_candidates.append(pred_text)
        except Exception as e:
            pass

    # 去重
    unique_candidates = list(dict.fromkeys(all_candidates))

    print(f"\n生成了 {len(unique_candidates)} 个不同结果:\n")

    # 分析每个候选
    results = []
    for pred_text in unique_candidates:
        prefix_ratio = compute_prefix_overlap(target_text, pred_text)
        clause_count = analyze_sentence(pred_text)
        results.append({
            'text': pred_text,
            'prefix_ratio': prefix_ratio,
            'clause_count': clause_count,
            'diversity': 1 - prefix_ratio  # 多样性 = 1 - 前缀重叠
        })

    # 按多样性降序排列
    results.sort(key=lambda x: (-x['diversity'], -x['clause_count']))

    print("结果（按多样性降序）:")
    print("-" * 70)
    for i, r in enumerate(results):
        print(f"\n[{i+1}] 多样性: {r['diversity']:.2f}, 从句: {r['clause_count']}")
        print(f"    {r['text']}")
        print(f"    (前缀重叠: {r['prefix_ratio']*100:.1f}%)")

    # 展示最佳（最多样）结果
    print("\n" + "=" * 70)
    print("最佳结果（最多样）:")
    best = results[0]
    print(f"  {best['text']}")
    print(f"  多样性: {best['diversity']:.2f}, 前缀重叠: {best['prefix_ratio']*100:.1f}%")


if __name__ == "__main__":
    main()
