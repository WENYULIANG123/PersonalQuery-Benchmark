#!/usr/bin/env python3
"""重新测试，验证sem_prob阈值修改效果"""

import torch
from transformers import T5Tokenizer, set_seed
from model import get_model
from options import parse_args


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

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"sem_prob阈值: 0.70 (已修改)")
    print()

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

    # 测试不同ling_value
    for ling_val in [0.80, 0.90, 1.00]:
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
            print(f"ling={ling_val:.2f} -> {pred_text}")
        except Exception as e:
            print(f"ling={ling_val:.2f} 错误: {e}")


if __name__ == "__main__":
    main()
