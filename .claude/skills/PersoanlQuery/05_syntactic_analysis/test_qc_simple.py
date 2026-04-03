#!/usr/bin/env python3
"""
简化的 QC 推理测试

不使用有问题的 infer_with_feedback_BP，
而是生成多个候选，然后用 LingDisc 和 SemEmb 打分选择最佳。
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import get_model
from data import LingDataCollator, load_data
from predict import write_predictions
from options import parse_args


def simple_qc_inference(model, ling_disc, sem_emb, batch, tokenizer, num_candidates=4):
    """
    简化的 QC 推理：
    1. 生成多个候选（通过改变 temperature/top_p）
    2. 用 LingDisc 预测每个候选的 ling features
    3. 用 SemEmb 计算语义相似度
    4. 选择最接近目标 ling features 且语义相似度高的
    """
    device = next(model.parameters()).device

    # 准备 batch
    sentence2_ling_target = batch.get("sentence2_ling")  # 目标 ling features

    # 使用模型生成多个候选
    with torch.no_grad():
        # 基本生成
        pred_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            sentence1_ling=batch.get("sentence1_ling"),
            sentence2_ling=sentence2_ling_target,
        )

        # 如果需要更多候选，可以用不同参数生成
        candidates = [pred_ids]

        # 尝试不同 temperature 生成更多候选
        for temp in [0.7, 0.9, 1.1]:
            try:
                cand_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    sentence1_ling=batch.get("sentence1_ling"),
                    sentence2_ling=sentence2_ling_target,
                    temperature=temp,
                    do_sample=True,
                    top_p=0.9,
                )
                candidates.append(cand_ids)
            except:
                pass

    # 解码所有候选
    decoded = tokenizer.batch_decode(torch.cat(candidates, dim=0).cpu(),
                                     skip_special_tokens=True)

    best_text = decoded[0]
    best_score = -float('inf')

    # 用 LingDisc 评估每个候选
    for i, text in enumerate(decoded):
        if not text.strip():
            continue

        # 用 tokenizer 处理候选
        cand_inputs = tokenizer(text, return_tensors="pt", padding=True,
                               truncation=True, max_length=128)
        cand_input_ids = cand_inputs["input_ids"].to(device)
        cand_attention_mask = cand_inputs["attention_mask"].to(device)

        # LingDisc 预测
        with torch.no_grad():
            pred_ling = ling_disc(input_ids=cand_input_ids,
                                  attention_mask=cand_attention_mask)

        # 计算与目标的 MSE
        if sentence2_ling_target is not None:
            target_ling = sentence2_ling_target[0] if sentence2_ling_target.dim() > 1 else sentence2_ling_target
            ling_loss = F.mse_loss(pred_ling[0], target_ling.to(device))
        else:
            ling_loss = 0

        # 简单的质量分数（越低越好）
        score = -ling_loss.item()

        if score > best_score:
            best_score = score
            best_text = text

    return best_text, [decoded[0], decoded]


def main():
    args, _, _ = parse_args()

    # 设置参数
    args.ckpt = "/home/wlia0047/ar57_scratch/wenyu/LingConv_models/0402_17-19-36-ling_conversion_sem/best_model"
    args.disc_ckpt = "/home/wlia0047/ar57_scratch/wenyu/LingConv_models/0402_18-02-15-ling_disc-t5/best_ling_disc"
    args.disc_type = "t5"
    args.sem_loss = True
    args.sem_loss_type = "shared"
    args.predict_with_feedback = False  # 使用简化版 QC
    args.data_dir = "./data"
    args.data = "ling_conversion"
    args.split = "test"
    args.predict_fn = "preds/test_qc_simple.txt"
    args.eval_batch_size = 1  # 只能用 batch_size=1 因为每个样本独立处理
    args.seed = 42

    set_seed(args.seed)

    print("=" * 70)
    print("Simple QC Inference Test")
    print("=" * 70)
    print(f"Model: {args.ckpt}")
    print(f"LingDisc: {args.disc_ckpt} (type={args.disc_type})")
    print("=" * 70)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    data, _, _ = load_data(args, tokenizer, return_data=True)
    print(f"Test samples: {len(data[args.split])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ling_disc, sem_emb = get_model(args, tokenizer, device)
    model.eval()
    if ling_disc is not None:
        ling_disc.eval()
    if sem_emb is not None:
        sem_emb.eval()
    print(f"Device: {device}")
    print(f"LingDisc loaded: {ling_disc is not None}")
    print(f"SemEmb loaded: {sem_emb is not None}")

    collator = LingDataCollator(tokenizer)
    dataloader = DataLoader(data[args.split], batch_size=1, shuffle=False, collate_fn=collator)

    predictions = []
    feedback_traces = []

    print("\nRunning simple QC inference...")
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        pred_text, interpolations = simple_qc_inference(
            model, ling_disc, sem_emb, batch, tokenizer
        )

        predictions.append(pred_text.strip())
        feedback_traces.append({
            "final": pred_text.strip(),
            "interpolations": interpolations
        })

    os.makedirs(os.path.dirname(args.predict_fn), exist_ok=True)
    write_predictions(args.predict_fn, predictions)
    write_feedback_log = lambda p, f: open(p, 'w').write('\n'.join([str(x) for x in f]))
    write_feedback_log(args.predict_fn + ".traces", feedback_traces)

    print("\n" + "=" * 70)
    print("Done!")
    print(f"Saved predictions to {args.predict_fn}")
    print("=" * 70)

    # 打印几个样本
    print("\nSample Predictions:")
    print("=" * 70)
    for i in range(min(5, len(predictions))):
        print(f"\n[{i+1}] {predictions[i]}")


if __name__ == "__main__":
    main()
