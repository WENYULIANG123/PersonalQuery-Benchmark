#!/usr/bin/env python3
"""简单推理测试"""

import torch
from transformers import T5Tokenizer
import sys
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis')

from data import load_data
from options import parse_args

def main():
    args, _, _ = parse_args()
    args.ckpt = '/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints/0403_13-58-13-ling_conversion-bos_replace/checkpoint-17000'
    args.combine_method = 'bos_replace'

    device = torch.device('cuda')
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    data, _, _ = load_data(args, tokenizer)

    # 获取几个样本进行测试
    print("=" * 70)
    print("查看训练数据样本和对应的复杂度")
    print("=" * 70)

    for i in [0, 10, 20]:
        sample = data['train'][i]
        src = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        tgt = tokenizer.decode(sample['labels'], skip_special_tokens=True)
        ling = sample['sentence2_ling']

        print(f"\n[样本 {i}]")
        print(f"输入: {src[:100]}")
        print(f"目标: {tgt[:100]}")
        print(f"复杂度: {ling[:10]}... (共{len(ling)}维)")

    print("\n" + "=" * 70)
    print("模型评估损失 (forward pass，不触发generate)")
    print("=" * 70)

    from model import get_model
    model, _, _ = get_model(args, tokenizer, device)
    model.eval()

    # 取几个样本计算损失
    total_loss = 0
    n_samples = 10

    for i in range(n_samples):
        sample = data['train'][i]
        batch = {
            'input_ids': torch.tensor([sample['input_ids']]).to(device),
            'attention_mask': torch.tensor([sample['attention_mask']]).to(device),
            'labels': torch.tensor([sample['labels']]).to(device),
            'sentence1_ling': torch.tensor([sample['sentence1_ling']]).to(device),
            'sentence2_ling': torch.tensor([sample['sentence2_ling']]).to(device),
        }

        with torch.no_grad():
            output = model(**batch)

        # 计算简单的交叉熵损失
        logits = output.logits[:, :-1]
        labels = batch['labels'][:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100
        )
        total_loss += loss.item()

        if i < 3:
            print(f"样本 {i}: loss = {loss.item():.4f}")

    avg_loss = total_loss / n_samples
    print(f"\n平均损失: {avg_loss:.4f}")
    print("\n注意: forward pass 可以正常工作，generate 路径有 bug 需要修复。")


if __name__ == "__main__":
    main()
