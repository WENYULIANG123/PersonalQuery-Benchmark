#!/usr/bin/env python3
"""
微调 linguistic features 值，生成更多从句的句子
"""

import os
import torch
from transformers import T5Tokenizer, set_seed
from model import get_model
from options import parse_args


def analyze_sentence_structure(text):
    """简单分析句子结构"""
    # 检测从句连接词
    subordinator_count = 0
    subordinators = ['which', 'that', 'where', 'when', 'because', 'although', 'while', 'if', 'unless', 'since', 'though', 'after', 'before', 'until', 'for', 'nor', 'but', 'or', 'yet', 'so']
    words = text.lower().split()

    for i, word in enumerate(words):
        # 检测 "which/that/where" 等关系词
        if word in ['which', 'that', 'where', 'when']:
            subordinator_count += 1
        # 检测 "because", "although" 等从属连词
        if word in ['because', 'although', 'while', 'if', 'unless', 'since', 'though', 'after', 'before', 'until']:
            subordinator_count += 1
        # 检测 ", and" 或 ", but" 等并列结构
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

    # 用户指定的句子
    target_text = "Robert Kaufman Fabrics Pre-Cut Quilt Squares - Green, 16.50, for Children"

    # 编码
    target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

    print("=" * 70)
    print(f"原始句子: {target_text}")
    print("=" * 70)

    # 细粒度值测试
    values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    print("\n细粒度调整（聚焦于从句结构）:\n")

    results = []
    for val in values:
        mod_ling = torch.ones(40) * val

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

            results.append((val, word_count, clause_count, pred_text))

            print(f"【{val:.2f}】{word_count}词, {clause_count}从句: {pred_text}")

        except Exception as e:
            print(f"【{val:.2f}】错误: {e}")

    # 总结
    print("\n" + "=" * 70)
    print("复杂度排序（按从句数量）:")
    print("=" * 70)
    results.sort(key=lambda x: (-x[2], x[1]))  # 先按从句数量降序，再按词数升序
    for val, word_count, clause_count, text in results:
        print(f"{val:.2f}: {clause_count}从句, {word_count}词 - {text}")


if __name__ == "__main__":
    main()
