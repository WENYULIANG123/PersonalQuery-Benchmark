#!/usr/bin/env python3
"""对比贪婪解码和采样解码的40维特征距离"""

import torch
import numpy as np
from transformers import T5Tokenizer, set_seed
from model import get_model
from options import parse_args


def get_ling_features(text, tokenizer, ling_disc, device):
    """获取句子的40维语言特征"""
    enc = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 直接用LingDisc预测特征
    with torch.no_grad():
        features = ling_disc(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    return features[0].cpu().numpy()


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

    # 测试句子
    target_text = "The dog is running after the cat"
    target_enc = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

    print("=" * 70)
    print(f"输入: {target_text}")
    print("=" * 70)

    # 获取原句特征
    original_features = get_ling_features(target_text, tokenizer, ling_disc, device)
    print(f"\n原句40维特征: {original_features}")

    # 准备batch
    mod_ling = torch.ones(40) * 0.70
    batch = {
        "input_ids": target_enc["input_ids"].to(device),
        "attention_mask": target_enc["attention_mask"].to(device),
        "sentence1_input_ids": target_enc["input_ids"].to(device),
        "sentence1_attention_mask": target_enc["attention_mask"].to(device),
        "sentence2_ling": mod_ling.unsqueeze(0).to(device),
        "sentence1_ling": mod_ling.unsqueeze(0).to(device),
        "labels": target_enc["input_ids"].to(device),
    }

    # ============ 贪婪解码 ============
    print("\n" + "=" * 70)
    print("贪婪解码 (temperature=0)")
    print("=" * 70)

    # 临时修改temperature=0进行贪婪解码
    original_temp = 0.7
    model._greedy_decode_with_logits.__globals__['temperature'] = 0

    set_seed(42)
    try:
        with torch.no_grad():
            pred_greedy, _ = model.infer_with_feedback_BP(
                ling_disc=ling_disc,
                sem_emb=sem_emb,
                batch=batch,
                tokenizer=tokenizer
            )
        pred_text_greedy = tokenizer.decode(pred_greedy[0], skip_special_tokens=True)
        pred_features_greedy = get_ling_features(pred_text_greedy, tokenizer, ling_disc, device)

        # 计算距离
        dist_greedy = np.linalg.norm(pred_features_greedy - original_features)
        print(f"贪婪解码结果: {pred_text_greedy}")
        print(f"与原句特征距离: {dist_greedy:.4f}")
    except Exception as e:
        print(f"贪婪解码错误: {e}")
        pred_text_greedy = None
        dist_greedy = None

    # ============ 采样解码 ============
    print("\n" + "=" * 70)
    print("采样解码 (temperature=0.7)")
    print("=" * 70)

    # 恢复temperature=0.7
    model._greedy_decode_with_logits.__globals__['temperature'] = 0.7

    results_sampling = []
    for i in range(5):
        set_seed(i * 100)
        try:
            with torch.no_grad():
                pred_samp, _ = model.infer_with_feedback_BP(
                    ling_disc=ling_disc,
                    sem_emb=sem_emb,
                    batch=batch,
                    tokenizer=tokenizer
                )
            pred_text_samp = tokenizer.decode(pred_samp[0], skip_special_tokens=True)
            pred_features_samp = get_ling_features(pred_text_samp, tokenizer, ling_disc, device)
            dist_samp = np.linalg.norm(pred_features_samp - original_features)

            results_sampling.append({
                'text': pred_text_samp,
                'features': pred_features_samp,
                'distance': dist_samp
            })
            print(f"采样{i+1}: {pred_text_samp}")
            print(f"       与原句特征距离: {dist_samp:.4f}")
        except Exception as e:
            print(f"采样{i+1}错误: {e}")

    # ============ 对比统计 ============
    print("\n" + "=" * 70)
    print("特征距离对比")
    print("=" * 70)

    if dist_greedy is not None:
        print(f"贪婪解码特征距离: {dist_greedy:.4f}")

    if results_sampling:
        avg_dist = np.mean([r['distance'] for r in results_sampling])
        min_dist = np.min([r['distance'] for r in results_sampling])
        max_dist = np.max([r['distance'] for r in results_sampling])
        print(f"采样解码特征距离 (5次平均): {avg_dist:.4f}")
        print(f"采样解码特征距离 (min): {min_dist:.4f}")
        print(f"采样解码特征距离 (max): {max_dist:.4f}")

    # 展示特征差异最大的维度
    if results_sampling and dist_greedy is not None:
        print("\n" + "=" * 70)
        print("特征维度差异分析 (采样解码 vs 贪婪解码)")
        print("=" * 70)

        # 获取最后一次采样的特征进行对比
        last_samp_features = results_sampling[-1]['features']
        diff = np.abs(last_samp_features - pred_features_greedy)
        top_indices = np.argsort(diff)[-10:][::-1]  # 前10个差异最大的维度

        from const import sca_names
        print(f"{'维度':<10} {'特征名':<10} {'贪婪解码':<12} {'采样解码':<12} {'差异':<10}")
        print("-" * 60)
        for idx in top_indices:
            name = sca_names[idx] if idx < len(sca_names) else f"dim_{idx}"
            print(f"{idx:<10} {name:<10} {pred_features_greedy[idx]:<12.4f} {last_samp_features[idx]:<12.4f} {diff[idx]:<10.4f}")


if __name__ == "__main__":
    main()
