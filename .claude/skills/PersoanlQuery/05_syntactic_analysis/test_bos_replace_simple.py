#!/usr/bin/env python3
"""
简单测试 bos_replace 方法
"""

import torch
from transformers import T5Tokenizer, set_seed, AutoModelForSeq2SeqLM
import sys
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis')

from model import get_model
from options import parse_args


def main():
    args, _, _ = parse_args()

    # 使用正确的检查点
    args.ckpt = "/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints/0403_13-58-13-ling_conversion-bos_replace/checkpoint-17482"
    args.disc_ckpt = None
    args.sem_ckpt = None
    args.seed = 42
    args.combine_method = "bos_replace"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model, _, _ = get_model(args, tokenizer, device)
    model.eval()
    model.to(device)

    print(f"combine_method = {model.args.combine_method}")
    print(f"ling2_only = {model.args.ling2_only}")
    print("模型加载完成\n")

    # 测试句子
    test_sentence = "I bought an easel that is large and versatile."
    target_enc = tokenizer(test_sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)

    # 目标复杂度
    ling = torch.ones(40) * 0.8

    # 构造 batch
    batch = {
        "input_ids": target_enc["input_ids"].to(device),
        "attention_mask": target_enc["attention_mask"].to(device),
        "sentence2_input_ids": target_enc["input_ids"].to(device),
        "sentence2_attention_mask": target_enc["attention_mask"].to(device),
        "sentence2_ling": ling.unsqueeze(0).to(device),
        "sentence1_ling": ling.unsqueeze(0).to(device),
        "labels": target_enc["input_ids"].to(device),
    }

    print("Batch keys:", batch.keys())
    print("sentence2_ling shape:", batch["sentence2_ling"].shape)
    print()

    # 尝试推理
    print("尝试使用 model.infer...")
    try:
        set_seed(42)
        with torch.no_grad():
            pred = model.infer(batch)
        print(f"原始输出: {pred}")
        print(f"原始输出shape: {pred.shape}")
        pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
        print(f"结果: {pred_text}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n尝试使用 model.generate...")
    try:
        set_seed(42)
        with torch.no_grad():
            # 直接调用generate
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                sentence2_ling=batch["sentence2_ling"],
                sentence1_ling=batch["sentence1_ling"],
                max_length=128,
                do_sample=True,
                temperature=0.7,
            )
        print(f"原始输出: {outputs}")
        print(f"原始输出shape: {outputs.shape}")
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"结果: {pred_text}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
