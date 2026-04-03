#!/usr/bin/env python3
"""普通生成模式测试"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import get_model
from data import LingDataCollator, load_data
from predict import write_predictions
from options import parse_args


def main():
    args, _, _ = parse_args()

    # 设置参数
    args.ckpt = "/home/wlia0047/ar57_scratch/wenyu/LingConv_models/0402_17-19-36-ling_conversion_sem/best_model"
    args.disc_ckpt = None  # 不使用 LingDisc
    args.sem_loss = True
    args.sem_loss_type = "shared"
    args.predict_with_feedback = False
    args.data_dir = "./data"
    args.data = "ling_conversion"
    args.split = "test"
    args.predict_fn = "preds/test_basic.txt"
    args.eval_batch_size = 16
    args.seed = 42

    set_seed(args.seed)

    print("=" * 70)
    print("Basic Generation Test (No QC)")
    print("=" * 70)
    print(f"Model: {args.ckpt}")
    print("=" * 70)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    data, _, _ = load_data(args, tokenizer, return_data=True)
    print(f"Test samples: {len(data[args.split])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ling_disc, sem_emb = get_model(args, tokenizer, device)
    model.eval()
    print(f"Device: {device}")
    print(f"SemEmb (shared encoder): {sem_emb is not None}")

    collator = LingDataCollator(tokenizer)
    dataloader = DataLoader(data[args.split], batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)

    predictions = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            prediction_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                sentence1_ling=batch.get("sentence1_ling"),
                sentence2_ling=batch.get("sentence2_ling"),
            )
        decoded = tokenizer.batch_decode(prediction_ids.cpu(), skip_special_tokens=True)
        predictions.extend(text.strip() for text in decoded)

    os.makedirs(os.path.dirname(args.predict_fn), exist_ok=True)
    write_predictions(args.predict_fn, predictions)
    print(f"\nSaved predictions to {args.predict_fn}")

    # 打印几个样本
    print("\n" + "=" * 70)
    print("Sample Predictions:")
    print("=" * 70)
    for i in range(min(3, len(predictions))):
        print(f"\n[{i+1}] {predictions[i]}")


if __name__ == "__main__":
    main()
