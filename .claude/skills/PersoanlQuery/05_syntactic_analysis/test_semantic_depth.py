#!/usr/bin/env python3
"""
使用有语义深度的句子进行QC优化，生成带从句的句子
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

    # 有语义深度的测试句子 - 包含动作、对象、状态等多种元素
    test_sentences = [
        # 动作+对象
        "A boy is riding a horse",
        # 人物+动作+物品
        "A woman is holding an umbrella",
        # 人物+动作+地点
        "The child is playing in the garden",
        # 人物+动作+穿着
        "A man is wearing a blue jacket",
        # 多动作
        "The dog is running after the cat",
        # 人物+动作+目的
        "The boy is reading a story book",
        # 人物+状态+位置
        "The girl is sitting near the window",
    ]

    # 使用0.70作为优化值
    ling_value = 0.70

    print("=" * 70)
    print(f"QC优化测试 - 语义深度句子 (ling_value={ling_value})")
    print("=" * 70)

    all_results = []

    for target_text in test_sentences:
        target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

        print(f"\n输入: {target_text}")

        mod_ling = torch.ones(40) * ling_value

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
            all_results.append((target_text, pred_text, word_count, clause_count))
            print(f"  -> {pred_text}")
            print(f"     [{word_count}词, {clause_count}从句]")
        except Exception as e:
            print(f"  错误: {e}")

    # 总结
    print("\n" + "=" * 70)
    print("结果汇总（按从句数量降序）:")
    print("=" * 70)
    all_results.sort(key=lambda x: (-x[3], x[2]))
    for orig, pred, word_count, clause_count in all_results:
        print(f"\n原文: {orig}")
        print(f"  -> {pred}")
        print(f"     [{word_count}词, {clause_count}从句]")


if __name__ == "__main__":
    main()
