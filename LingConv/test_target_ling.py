"""
测试直接指定目标语言复杂度的 QC 反馈
"""
import sys
import os
import torch
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


def test_target_ling():
    ckpt_dir = "/home/wlia0047/ar57_scratch/wenyu/lingconv_official6/0405_01-47-33-ling_conversion-decoder_add_first"
    disc_ckpt = "/home/wlia0047/ar57_scratch/wenyu/lingconv_disc"

    print(f"加载 checkpoint: {ckpt_dir}")
    print(f"加载 LingDisc: {disc_ckpt}")

    tokenizer = T5Tokenizer.from_pretrained(ckpt_dir)

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

    print("\n" + "="*70)
    print("测试直接指定目标语言复杂度的 QC 反馈")
    print("="*70 + "\n")

    # 测试用例：指定不同的复杂度目标
    test_cases = [
        {
            "name": "降低复杂度（简单句）",
            "input": "The quick brown fox jumps over the lazy dog.",
            "target_ling": torch.tensor([0.0] * 40, dtype=torch.float32),  # 全部归零 = 最简单
        },
        {
            "name": "保持当前复杂度",
            "input": "A man with a hard hat is dancing.",
            "target_ling": torch.tensor([0.5] * 40, dtype=torch.float32),  # 中等复杂度
        },
        {
            "name": "提高复杂度（复杂句）",
            "input": "A man is dancing.",
            "target_ling": torch.tensor([2.0] * 40, dtype=torch.float32),  # 高复杂度
        },
    ]

    for i, case in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"【测试 {i+1}】{case['name']}")
        print(f"输入: {case['input']}")
        print(f"目标语言特征: {case['target_ling'][:5].tolist()}... (前5维)")
        print(f"{'='*70}")

        inputs = tokenizer(case["input"], return_tensors='pt', truncation=True, max_length=128)

        # 构建 batch（sentence2_ling 只是占位，实际目标由 target_ling 指定）
        ling_tensor = torch.zeros(1, 40, dtype=torch.float32)  # 占位
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
                case["target_ling"],
                batch,
                tokenizer,
                max_iter=5,
                loss_threshold=0.5
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
    print("目标语言复杂度 QC 反馈测试完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_target_ling()
