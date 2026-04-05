"""
Stage 5: 测试官方 LINGCONV 模型
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


def test_model():
    ckpt_dir = "/home/wlia0047/ar57_scratch/wenyu/lingconv_official6/0405_01-47-33-ling_conversion-decoder_add_first"
    print(f"加载 checkpoint: {ckpt_dir}")

    # 加载 tokenizer
    tokenizer = T5Tokenizer.from_pretrained(ckpt_dir)

    # 创建模型
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
    parser.add_argument("--disc_ckpt")
    parser.add_argument("--disc_type", default="deberta")
    parser.add_argument("--ckpt", default="/home/wlia0047/ar57_scratch/wenyu/lingconv_official6/0405_01-47-33-ling_conversion-decoder_add_first")
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
    parser.add_argument("--sem_ckpt")
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args([])

    model, _, _ = model_module.get_model(args, tokenizer, torch.device('cuda:0'))
    model.eval()

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

    print(f"\n测试集大小: {len(test_split)}")
    print("=" * 60)

    # 测试前 10 个样本
    for i in range(min(10, len(test_split))):
        sample = test_split[i]
        src = sample.get('sentence1', None)
        tgt = sample.get('sentence2', None)

        if src is None:
            continue

        inputs = tokenizer(src, return_tensors='pt', truncation=True, max_length=128)
        ling_tensor = torch.tensor([sample['sentence2_ling']], dtype=torch.float32)

        batch = {
            "input_ids": inputs['input_ids'].cuda(),
            "attention_mask": inputs['attention_mask'].cuda(),
            "sentence1_ling": ling_tensor.cuda(),
            "sentence2_ling": ling_tensor.cuda(),
            "labels": inputs['input_ids'].cuda(),
        }

        with torch.no_grad():
            pred = model.infer(batch)

        generated = tokenizer.decode(pred[0], skip_special_tokens=True)

        print(f"\n[样本 {i+1}]")
        print(f"输入: {src}")
        print(f"目标: {tgt}")
        print(f"生成: {generated}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_model()
