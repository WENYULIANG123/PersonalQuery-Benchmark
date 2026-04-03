#!/usr/bin/env python3
"""
测试独立生成模式：decoder不看原始句子，只靠ling embedding生成
"""

import torch
from transformers import T5Tokenizer, set_seed
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/LingConv')

from model import get_model
from options import parse_args


def main():
    args, _, _ = parse_args()

    args.ckpt = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis/checkpoints/0402_23-52-15-ling_conversion-decoder_add_first/checkpoint-17482"
    args.disc_ckpt = None
    args.sem_ckpt = None
    args.seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model, _, _ = get_model(args, tokenizer, device)
    model.eval()
    model.to(device)

    print("模型加载完成\n")

    # 不同复杂度目标
    ling_configs = [
        ("极低0.2", torch.ones(40) * 0.2),
        ("中0.5", torch.ones(40) * 0.5),
        ("高0.8", torch.ones(40) * 0.8),
    ]

    # 测试句子
    base_sentence = "I bought an easel that is large and versatile."

    print("=" * 70)
    print("模式1: 原始模式 (decoder可以看到原始句子)")
    print("=" * 70)

    target_enc = tokenizer(base_sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)

    for name, ling in ling_configs:
        batch = {
            "input_ids": target_enc["input_ids"].to(device),
            "attention_mask": target_enc["attention_mask"].to(device),
            "sentence1_input_ids": target_enc["input_ids"].to(device),
            "sentence1_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_ling": ling.unsqueeze(0).to(device),
            "sentence1_ling": ling.unsqueeze(0).to(device),
            "labels": target_enc["input_ids"].to(device),
            "use_original_decoder_input": True,  # 原始模式
        }

        set_seed(42)
        with torch.no_grad():
            pred = model.infer(batch)

        pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
        print(f"[{name}] {pred_text}")

    print("\n" + "=" * 70)
    print("模式2: 独立生成模式 (decoder只看pad token，不看原始句子)")
    print("=" * 70)

    # 这里需要手动调用decode，因为infer不支持use_original_decoder_input参数
    # 我们直接修改model.prepare_inputs_for_generation的行为

    for name, ling in ling_configs:
        batch = {
            "input_ids": target_enc["input_ids"].to(device),
            "attention_mask": target_enc["attention_mask"].to(device),
            "sentence1_input_ids": target_enc["input_ids"].to(device),
            "sentence1_attention_mask": target_enc["attention_mask"].to(device),
            "sentence2_ling": ling.unsqueeze(0).to(device),
            "sentence1_ling": ling.unsqueeze(0).to(device),
            "labels": target_enc["input_ids"].to(device),
        }

        # 手动调用encode
        model.eval()
        encoder_outputs, encoder_attention_mask, cache = model.encode(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            sentence1_ling=batch.get("sentence1_ling"),
            sentence2_ling=batch.get("sentence2_ling"),
        )

        # 修改prepare_inputs_for_generation来使用独立模式
        from types import SimpleNamespace
        original_prepare = model.prepare_inputs_for_generation

        # 临时替换为独立模式
        def independent_prepare(
            input_ids,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            sentence1_ling=None,
            sentence2_ling=None,
            **kwargs
        ):
            # 使用pad token作为decoder输入
            bs = input_ids.shape[0]
            decoder_input_ids = torch.full((bs, 1), model.pad_token_id, dtype=torch.long, device=input_ids.device)
            decoder_inputs_embeds = model.shared(decoder_input_ids)

            # 添加ling embedding
            if sentence1_ling is not None and sentence2_ling is not None:
                ling_combined = sentence1_ling + sentence2_ling
            elif sentence2_ling is not None:
                ling_combined = sentence2_ling
            else:
                ling_combined = None

            decoder_inputs_embeds = decoder_inputs_embeds + ling_combined

            return {
                "decoder_inputs_embeds": decoder_inputs_embeds,
                "past_key_values": past_key_values,
                "encoder_outputs": encoder_outputs,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
            }

        model.prepare_inputs_for_generation = independent_prepare

        # 生成
        set_seed(42)
        with torch.no_grad():
            dec_output = model.generate(
                attention_mask=encoder_attention_mask,
                encoder_outputs=encoder_outputs,
                sentence1_ling=batch.get("sentence1_ling"),
                sentence2_ling=batch.get("sentence2_ling"),
                max_length=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.85,
            )

        pred_text = tokenizer.decode(dec_output[0], skip_special_tokens=True)
        print(f"[{name}] {pred_text}")

        # 恢复原始prepare函数
        model.prepare_inputs_for_generation = original_prepare

    print("\n" + "=" * 70)
    print("不同seed的独立生成效果")
    print("=" * 70)

    ling = torch.ones(40) * 0.8  # 高复杂度

    for seed in [42, 100, 200, 300]:
        # 重新encode
        encoder_outputs, encoder_attention_mask, cache = model.encode(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            sentence1_ling=ling.unsqueeze(0).to(device),
            sentence2_ling=ling.unsqueeze(0).to(device),
        )

        model.prepare_inputs_for_generation = independent_prepare

        set_seed(seed)
        with torch.no_grad():
            dec_output = model.generate(
                attention_mask=encoder_attention_mask,
                encoder_outputs=encoder_outputs,
                sentence1_ling=ling.unsqueeze(0).to(device),
                sentence2_ling=ling.unsqueeze(0).to(device),
                max_length=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.85,
            )

        pred_text = tokenizer.decode(dec_output[0], skip_special_tokens=True)
        print(f"[seed={seed}] {pred_text}")

        model.prepare_inputs_for_generation = original_prepare


if __name__ == "__main__":
    main()
