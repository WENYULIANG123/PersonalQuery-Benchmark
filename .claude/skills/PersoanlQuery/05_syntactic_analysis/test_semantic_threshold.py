#!/usr/bin/env python3
"""
测试语义相似度阈值对生成的影响
模拟sem_prob >= 0.90的约束效果
"""

import os
import torch
from transformers import T5Tokenizer, set_seed
from model import get_model
from options import parse_args


def compute_cosine_similarity(sem_emb, text1_ids, text2_ids, mask1, mask2):
    """计算两个文本的余弦相似度"""
    with torch.no_grad():
        enc_out1 = sem_emb(input_ids=text1_ids, attention_mask=mask1)
        s1_emb = enc_out1.last_hidden_state.mean(1)

        enc_out2 = sem_emb(input_ids=text2_ids, attention_mask=mask2)
        s2_emb = enc_out2.last_hidden_state.mean(1)

        cos_sim = torch.nn.functional.cosine_similarity(s1_emb, s2_emb, dim=-1)
        return cos_sim.item()


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

    # 测试不同ling_value，查看语义相似度
    print("\n【不同ling_value下的语义相似度】:")
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

            # 计算语义相似度
            pred_enc = tokenizer(pred_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            sem_prob = compute_cosine_similarity(
                sem_emb,
                target_enc["input_ids"].to(device),
                pred_enc["input_ids"].to(device),
                target_enc["attention_mask"].to(device),
                pred_enc["attention_mask"].to(device)
            )

            # 计算前缀相同比例
            orig_words = target_text.lower().split()
            pred_words = pred_text.lower().split()
            prefix_len = 0
            for j in range(min(len(orig_words), len(pred_words))):
                if orig_words[j] == pred_words[j]:
                    prefix_len += 1
                else:
                    break

            ratio = prefix_len / len(orig_words) if orig_words else 0

            print(f"\nling={ling_val:.2f}:")
            print(f"  生成: {pred_text}")
            print(f"  语义相似度: {sem_prob:.4f} (需要≥0.90)")
            print(f"  前缀相同: {prefix_len}/{len(orig_words)} ({ratio*100:.1f}%)")

        except Exception as e:
            print(f"\nling={ling_val:.2f} 错误: {e}")


if __name__ == "__main__":
    main()
