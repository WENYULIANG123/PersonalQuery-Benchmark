#!/usr/bin/env python3
"""
测试不同语义相似度权重下的从句生成效果
"""

import os
import torch
from transformers import T5Tokenizer, set_seed
from model import get_model
from options import parse_args


def analyze_sentence_structure(text):
    """分析句子结构"""
    subordinator_count = 0
    subordinators = ['which', 'that', 'where', 'when', 'because', 'although', 'while', 'if', 'unless', 'since', 'though', 'after', 'before', 'until', 'for', 'nor', 'but', 'or', 'yet', 'so']
    words = text.lower().split()

    for i, word in enumerate(words):
        if word in ['which', 'that', 'where', 'when']:
            subordinator_count += 1
        if word in ['because', 'although', 'while', 'if', 'unless', 'since', 'though', 'after', 'before', 'until']:
            subordinator_count += 1
        if word == 'and' and i > 0 and words[i-1] == ',':
            subordinator_count += 1
        if word == 'but' and i > 0 and words[i-1] == ',':
            subordinator_count += 1

    return subordinator_count


def main():
    args, _, _ = parse_args()

    # 设置参数
    args.ckpt = "/home/wlia0047/ar57_scratch/wenyu/LingConv_models/0402_17-19-36-ling_conversion_sem/best_model"
    args.disc_ckpt = "/home/wlia0047/ar57_scratch/wenyu/LingConv_models/0402_18-02-15-ling_disc-t5/best_ling_disc"
    args.disc_type = "t5"
    args.sem_ckpt = None
    args.sem_loss = True
    args.sem_loss_type = "shared"
    args.predict_with_feedback = True
    args.feedback_param = 'l'
    args.seed = 42

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 加载 tokenizer 和模型
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

    # 测试句子
    target_text = "The dog is running after the cat"

    target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

    print("=" * 70)
    print(f"输入句子: {target_text}")
    print("=" * 70)

    # 测试不同的ling_value
    results = []

    print("\n【不同ling_value测试】:")
    for ling_val in [0.50, 0.60, 0.70, 0.80, 0.90]:
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
            word_count = len(pred_text.split())
            clause_count = analyze_sentence_structure(pred_text)

            # 检查是否与原句前半部分相同
            orig_words = target_text.lower().split()
            pred_words = pred_text.lower().split()

            # 计算前缀相同长度
            prefix_len = 0
            for i in range(min(len(orig_words), len(pred_words))):
                if orig_words[i] == pred_words[i]:
                    prefix_len += 1
                else:
                    break

            results.append({
                'ling_val': ling_val,
                'pred': pred_text,
                'word_count': word_count,
                'clause_count': clause_count,
                'prefix_len': prefix_len,
                'prefix_ratio': prefix_len / len(orig_words) if orig_words else 0
            })

            print(f"\nling={ling_val:.2f} -> {pred_text}")
            print(f"  [{word_count}词, {clause_count}从句, 前缀相同比例: {prefix_len}/{len(orig_words)}]")

        except Exception as e:
            print(f"\nling={ling_val:.2f} 错误: {e}")

    # 总结
    print("\n" + "=" * 70)
    print("结果分析:")
    print("=" * 70)
    print(f"\n原句词数: {len(target_text.split())}")
    print(f"\n| ling_value | 生成词数 | 从句数 | 前缀相同 | 相同比例 |")
    print("|------------|---------|-------|---------|---------|")
    for r in results:
        print(f"| {r['ling_val']:.2f} | {r['word_count']:>7} | {r['clause_count']:>5} | {r['prefix_len']:>6} | {r['prefix_ratio']*100:>5.1f}% |")


if __name__ == "__main__":
    main()
