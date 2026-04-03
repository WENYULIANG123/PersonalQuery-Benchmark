#!/usr/bin/env python3
"""Quick test for QC inference - only tests 3 batches"""

import os
import torch
from transformers import T5Tokenizer, set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import get_model
from predict import build_feedback_batch, write_predictions, write_feedback_log, move_to_device, LingDataCollator
from data import load_data
from options import parse_args


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
    args.data_dir = "./data"
    args.data = "ling_conversion"
    args.split = "test"
    args.predict_fn = "preds/test_qc_quick.txt"
    args.fb_log = "feedback_logs/test_qc_quick.txt"
    args.eval_batch_size = 1
    args.seed = 42

    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.predict_fn), exist_ok=True)
    os.makedirs(os.path.dirname(args.fb_log), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer and data
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    data, _, _ = load_data(args, tokenizer, return_data=True)

    print(f"Total test samples: {len(data[args.split])}")

    # Load model
    model, ling_disc, sem_emb = get_model(args, tokenizer, device)
    model.eval()
    model.to(device)

    ling_disc.eval()
    ling_disc.to(device)

    if sem_emb is not None:
        sem_emb.eval()
        sem_emb.to(device)

    print(f"LingDisc loaded: {ling_disc is not None}")
    print(f"SemEmb loaded: {sem_emb is not None}")
    print(f"SemEmb type: {type(sem_emb)}")

    # Create dataloader with batch_size=1 for QC
    collator = LingDataCollator(tokenizer)
    dataloader = DataLoader(data[args.split], batch_size=1, shuffle=False, collate_fn=collator)

    # Run QC inference on first 3 batches
    predictions = []
    feedback_traces = []

    max_batches = 3
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        print(f"\n{'='*60}")
        print(f"Testing batch {i+1}/{max_batches}")

        batch = move_to_device(batch, device)

        try:
            with torch.no_grad():
                prediction_ids, feedback_trace = model.infer_with_feedback_BP(
                    ling_disc=ling_disc,
                    sem_emb=sem_emb,
                    batch=build_feedback_batch(batch),
                    tokenizer=tokenizer
                )

            decoded = tokenizer.batch_decode(prediction_ids.cpu(), skip_special_tokens=True)
            pred_text = decoded[0].strip()
            predictions.append(pred_text)

            print(f"Prediction: {pred_text}")
            print(f"Feedback trace (first 3): {feedback_trace[1][:3]}")

            feedback_traces.append({
                "final": pred_text,
                "interpolations": feedback_trace[1]
            })

        except Exception as e:
            print(f"Error on batch {i}: {e}")
            import traceback
            traceback.print_exc()
            predictions.append("")
            feedback_traces.append({"final": "", "interpolations": []})

    # Save results
    write_predictions(args.predict_fn, predictions)
    write_feedback_log(args.fb_log, feedback_traces)

    print(f"\n{'='*60}")
    print("QC Quick Test Results:")
    print(f"{'='*60}")
    for i, pred in enumerate(predictions):
        print(f"[{i+1}] {pred}")

    print(f"\nPredictions saved to: {args.predict_fn}")
    print(f"Feedback logs saved to: {args.fb_log}")


if __name__ == "__main__":
    main()
