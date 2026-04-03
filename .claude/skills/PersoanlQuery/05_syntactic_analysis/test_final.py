#!/usr/bin/env python3
"""最终测试：sem_prob阈值0.70 + 不同ling_value"""

import torch
from transformers import T5Tokenizer, set_seed
from model import get_model
from options import parse_args


def analyze_sentence(text):
    """分析句子结构"""
    words = text.lower().split()
    clause_count = 0
    for i, word in enumerate(words):
        if word in ['which', 'that', 'where', 'when', 'because', 'although', 'while', 'if', 'unless', 'since', 'though', 'after', 'before', 'until']:
            clause_count += 1
        if word == 'and' and i > 0 and words[i-1] == ',':
            clause_count += 1
        if word == 'but' and i > 0 and words[i-1] == ',':
            clause_count += 1
    return clause_count


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
    print(f"sem_prob阈值: 0.70 (从0.90降低)")
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

    results = []
    for ling_val in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
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
            clause_count = analyze_sentence(pred_text)

            orig_words = target_text.lower().split()
            pred_words = pred_text.lower().split()
            prefix_len = 0
            for j in range(min(len(orig_words), len(pred_words))):
                if orig_words[j] == pred_words[j]:
                    prefix_len += 1
                else:
                    break

            results.append({
                'ling': ling_val,
                'pred': pred_text,
                'clause': clause_count,
                'prefix': f"{prefix_len}/{len(orig_words)}",
                'prefix_ratio': prefix_len / len(orig_words)
            })
            print(f"ling={ling_val:.2f}: {pred_text}")
            print(f"        [从句{clause_count}, 前缀相同{prefix_len}/{len(orig_words)}]")

        except Exception as e:
            print(f"ling={ling_val:.2f}: 错误 - {e}")

    print("\n" + "=" * 70)
    print("结果汇总（按前缀相同比例升序）:")
    results.sort(key=lambda x: x['prefix_ratio'])
    for r in results:
        print(f"ling={r['ling']:.2f}: [{r['clause']}从句, 前缀{r['prefix']}] {r['pred']}")


if __name__ == "__main__":
    main()
