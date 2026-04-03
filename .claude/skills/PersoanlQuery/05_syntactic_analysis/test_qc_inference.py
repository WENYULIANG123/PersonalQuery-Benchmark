#!/usr/bin/env python3
"""QC 推理测试脚本"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import get_model
from data import LingDataCollator, load_data
from predict import build_feedback_batch, write_predictions, write_feedback_log
from options import parse_args


def main():
    # 解析参数
    args, _, _ = parse_args()

    # 设置参数
    args.ckpt = "/home/wlia0047/ar57_scratch/wenyu/LingConv_models/0402_17-19-36-ling_conversion_sem/best_model"
    args.disc_ckpt = "/home/wlia0047/ar57_scratch/wenyu/LingConv_models/0402_18-02-15-ling_disc-t5/best_ling_disc"
    args.disc_type = "t5"  # 强制使用 t5
    args.sem_ckpt = None  # 不使用 dedicated sem_emb
    args.sem_loss = True  # 启用 sem_loss 以获取共享 encoder
    args.sem_loss_type = "shared"  # 使用共享 encoder
    args.predict_with_feedback = True
    args.data_dir = "./data"
    args.data = "ling_conversion"
    args.split = "test"
    args.predict_fn = "preds/test_qc.txt"
    args.fb_log = "feedback_logs/test_qc.txt"
    args.eval_batch_size = 8
    args.seed = 42

    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(os.path.dirname(args.predict_fn), exist_ok=True)
    os.makedirs(os.path.dirname(args.fb_log), exist_ok=True)

    print("=" * 70)
    print("QC Inference Test")
    print("=" * 70)
    print(f"Model: {args.ckpt}")
    print(f"LingDisc: {args.disc_ckpt} (type={args.disc_type})")
    print(f"SemEmb: shared encoder")
    print("=" * 70)

    # 加载 tokenizer
    print("\nLoading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    # 加载数据
    print("Loading data...")
    data, _, _ = load_data(args, tokenizer, return_data=True)
    print(f"Test samples: {len(data[args.split])}")

    # 加载模型
    print("\nLoading model...")
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

    # 数据整理器
    collator = LingDataCollator(tokenizer)
    dataloader = DataLoader(data[args.split], batch_size=1, shuffle=False, collate_fn=collator)

    # 推理
    print("\nRunning QC inference...")
    predictions = []
    feedback_traces = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        prediction_ids, feedback_trace = model.infer_with_feedback_BP(
            ling_disc=ling_disc,
            sem_emb=sem_emb,
            batch=build_feedback_batch(batch),
            tokenizer=tokenizer,
        )
        decoded = tokenizer.batch_decode(prediction_ids.cpu(), skip_special_tokens=True)
        predictions.extend(text.strip() for text in decoded)
        feedback_traces.append({
            "final": feedback_trace[0],
            "interpolations": feedback_trace[1]
        })

    # 保存结果
    write_predictions(args.predict_fn, predictions)
    write_feedback_log(args.fb_log, feedback_traces)

    print("\n" + "=" * 70)
    print("Done!")
    print(f"Saved predictions to {args.predict_fn}")
    print(f"Saved feedback traces to {args.fb_log}")
    print("=" * 70)


if __name__ == "__main__":
    main()
