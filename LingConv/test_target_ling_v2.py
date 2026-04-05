"""
测试直接指定目标语言复杂度的 QC 反馈 - 基于数据范围内
使用正确归一化的语言特征
"""
import sys
import os
import torch
import numpy as np
import importlib.util

# 动态加载官方 model.py
model_path = os.path.join(os.path.dirname(__file__), "model.py")
spec = importlib.util.spec_from_file_location("model", model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

# 动态加载官方 data.py
data_path = os.path.join(os.path.dirname(__file__), "data.py")
spec = importlib.util.spec_from_file_location("data", data_path)
data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_module)

from transformers import T5Tokenizer
import joblib


def test_target_ling_v2():
    ckpt_dir = "/home/wlia0047/ar57_scratch/wenyu/lingconv_official6/0405_01-47-33-ling_conversion-decoder_add_first"
    disc_ckpt = "/home/wlia0047/ar57_scratch/wenyu/lingconv_disc"

    print(f"加载 checkpoint: {ckpt_dir}")
    print(f"加载 LingDisc: {disc_ckpt}")

    tokenizer = T5Tokenizer.from_pretrained(ckpt_dir)

    # 加载 scaler
    scaler = joblib.load("assets/scaler.bin")
    print(f"Scaler loaded: mean[:5]={scaler.mean_[:5]}")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--combine_method", default="decoder_add_first")
    parser.add_argument("--ling2_only", type=bool, default=True)
    parser.add_argument("--use_semantic_pooling", type=bool, default=False)
    parser.add_argument("--sem_loss", type=bool, default=False)
    parser.add_argument("--disc_loss", type=bool, default=False)
    parser.add_argument("--combine_weight", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=500)
    parser.add_argument("--lng_dim", type=int, default=40)
    parser.add_argument("--ling_dropout", type=float, default=0.1)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--pretrain_disc", action="store_true")
    parser.add_argument("--disc_ckpt", default=disc_ckpt)
    parser.add_argument("--disc_type", default="t5")
    parser.add_argument("--ckpt", default=ckpt_dir)
    parser.add_argument("--ling_embed_type", default="one-layer")
    parser.add_argument("--injection_type", default="first")
    parser.add_argument("--injection_layer", type=int, default=1)
    parser.add_argument("--ling_vae", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=150)
    parser.add_argument("--sem_loss_type", default="dedicated")
    parser.add_argument("--use_lingpred", action="store_true")
    parser.add_argument("--process_lingpred", action="store_true")
    parser.add_argument("--aug_same", type=bool, default=False)
    parser.add_argument("--freeze_lm", action="store_true")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--feedback_param", default="logits")
    parser.add_argument("--sem_ckpt", default=None)
    args = parser.parse_args([])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, ling_disc, _ = model_module.get_model(args, tokenizer, device)
    model.eval()
    ling_disc.eval()

    # 加载数据
    parser2 = argparse.ArgumentParser()
    parser2.add_argument("--data_dir", default="/home/wlia0047/ar57_scratch/wenyu/ling_conversion_official")
    parser2.add_argument("--data", default="ling_conversion")
    parser2.add_argument("--data_sources", default=["qqp", "mrpc", "stsb"])
    parser2.add_argument("--src_lng", default="ling")
    parser2.add_argument("--quantize_lng", type=bool, default=False)
    parser2.add_argument("--quant_nbins", type=int, default=20)
    parser2.add_argument("--do_imputation", type=bool, default=False)
    parser2.add_argument("--imputation_percentage", type=int, default=20)
    parser2.add_argument("--imputation_seed", type=int, default=0)
    parser2.add_argument("--use_ica", type=bool, default=False)
    parser2.add_argument("--n_ica", type=int, default=10)
    parser2.add_argument("--max_length", type=int, default=128)
    parser2.add_argument("--prepend_prompt", type=bool, default=False)
    parser2.add_argument("--prompt_text", default="")
    parser2.add_argument("--use_lingpred", type=bool, default=False)
    parser2.add_argument("--lng_ids", default=None)
    parser2.add_argument("--lng_ids_idx", type=int, default=None)
    parser2.add_argument("--lng_ids_path", default="./indices")
    parser2.add_argument("--aug_same", type=bool, default=False)
    parser2.add_argument("--max_eval_samples", type=int, default=3000)
    parser2.add_argument("--seed", type=int, default=0)
    args2 = parser2.parse_args([])

    data, _, _ = data_module.load_data(args2, tokenizer)
    test_split = data.get("test", data.get("dev"))
    if test_split is None:
        raise ValueError("未找到测试集")

    # 获取测试集的语言特征（已归一化，load_data 已经处理过）
    test_ling_scaled = np.array([item['sentence2_ling'] for item in test_split])
    l2_norms = np.linalg.norm(test_ling_scaled, axis=1)

    print(f"\n归一化后 L2 范数范围: [{l2_norms.min():.2f}, {l2_norms.max():.2f}]")
    print(f"归一化后 L2 范数均值: {l2_norms.mean():.2f}")

    # 设计测试用例
    low_threshold = np.percentile(l2_norms, 25)
    high_threshold = np.percentile(l2_norms, 75)

    low_idx = np.where(l2_norms < low_threshold)[0][0]
    high_idx = np.where(l2_norms > high_threshold)[0][0]
    mid_candidates = np.where((l2_norms >= low_threshold) & (l2_norms <= high_threshold))[0]
    mid_idx = mid_candidates[len(mid_candidates)//2]

    low_sample = test_split[low_idx]
    mid_sample = test_split[mid_idx]
    high_sample = test_split[high_idx]

    # 直接使用已归一化的语言特征
    target_low = np.array(low_sample['sentence2_ling'])
    target_mid = np.array(mid_sample['sentence2_ling'])
    target_high = np.array(high_sample['sentence2_ling'])

    print(f"\n低复杂度目标 L2: {np.linalg.norm(target_low):.2f} | {low_sample['sentence2'][:50]}...")
    print(f"中复杂度目标 L2: {np.linalg.norm(target_mid):.2f} | {mid_sample['sentence2'][:50]}...")
    print(f"高复杂度目标 L2: {np.linalg.norm(target_high):.2f} | {high_sample['sentence2'][:50]}...")

    test_cases = [
        {
            "name": "降低复杂度 → 低复杂度目标",
            "input": "A man with a hard hat is dancing.",
            "target_ling": torch.tensor(target_low, dtype=torch.float32),
        },
        {
            "name": "保持复杂度 → 中等复杂度目标",
            "input": "A man with a hard hat is dancing.",
            "target_ling": torch.tensor(target_mid, dtype=torch.float32),
        },
        {
            "name": "提高复杂度 → 高复杂度目标",
            "input": "A man is dancing.",
            "target_ling": torch.tensor(target_high, dtype=torch.float32),
        },
    ]

    print("\n" + "="*70)
    print("开始 QC 反馈测试")
    print("="*70)

    for i, case in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"【测试 {i+1}】{case['name']}")
        print(f"输入: {case['input']}")
        print(f"目标 L2: {torch.norm(case['target_ling']).item():.2f}")
        print(f"{'='*70}")

        inputs = tokenizer(case["input"], return_tensors='pt', truncation=True, max_length=128)

        # 构建 batch
        ling_tensor = torch.zeros(1, 40, dtype=torch.float32)
        batch = {
            "input_ids": inputs['input_ids'].to(device),
            "sentence1_input_ids": inputs['input_ids'].to(device),
            "sentence1_attention_mask": inputs['attention_mask'].to(device),
            "attention_mask": inputs['attention_mask'].to(device),
            "sentence1_ling": ling_tensor.to(device),
            "sentence2_ling": ling_tensor.to(device),
            "labels": inputs['input_ids'].to(device),
        }

        with torch.no_grad():
            pred, info = model.infer_with_target_ling(
                ling_disc,
                case["target_ling"].to(device),
                batch,
                tokenizer,
                max_iter=10,
                loss_threshold=1.0
            )

        generated = tokenizer.decode(pred[0], skip_special_tokens=True)

        print(f"\n📊 结果:")
        print(f"   原始输入: {case['input']}")
        print(f"   QC 生成:  {generated}")

        print(f"\n📈 迭代轨迹 ({len(info[1])} 步):")
        for j, text in enumerate(info[1]):
            marker = "→" if j < len(info[1]) - 1 else "★"
            print(f"   {j+1}. {marker} {text}")

        print(f"{'='*70}")

    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_target_ling_v2()
