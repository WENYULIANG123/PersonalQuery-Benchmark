"""测试 LingDisc 预测准确性 - 对比真实值与预测值"""
import torch
import numpy as np
import importlib.util
import argparse
from scipy.spatial.distance import cosine

sys_path = '/home/wlia0047/ar57/wenyu/LingConv'
import sys
sys.path.insert(0, sys_path)

model_path = f'{sys_path}/model.py'
spec = importlib.util.spec_from_file_location("model", model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

ckpt_dir = "/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints/0405_20-08-56-ling_conversion-decoder_add_first"

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
parser.add_argument("--disc_ckpt", default=None)
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
parser.add_argument("--disc_lng_dim", type=int, default=40)
args = parser.parse_args([])

from transformers import T5Tokenizer
from data import load_data

tokenizer = T5Tokenizer.from_pretrained(ckpt_dir)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Loading LingDisc model...")
ling_disc = model_module.LingDisc(
    model_name=args.model_name,
    disc_type=args.disc_type,
    disc_ckpt=args.disc_ckpt,
    lng_dim=args.lng_dim,
    disc_lng_dim=args.disc_lng_dim,
).to(device)
ling_disc.eval()
print("LingDisc model loaded successfully!")

# 加载数据
data_args = argparse.Namespace(
    data_dir='/home/wlia0047/ar57_scratch/wenyu/ling_conversion_official',
    data='ling_conversion',
    data_sources=['qqp', 'mrpc', 'stsb'],
    src_lng='ling',
    quantize_lng=False,
    quant_nbins=20,
    do_imputation=False,
    imputation_percentage=20,
    imputation_seed=0,
    use_ica=False,
    n_ica=10,
    max_length=128,
    prepend_prompt=False,
    prompt_text='',
    use_lingpred=False,
    lng_ids=None,
    lng_ids_idx=None,
    lng_ids_path='./indices',
    aug_same=False,
    max_eval_samples=3000,
    seed=0
)
data, scaler, _ = load_data(data_args, tokenizer)
test_split = data.get("test", data.get("dev"))

print(f"\n数据集样本数: {len(test_split)}")

# 测试不同复杂度的样本
test_ling = np.array([item['sentence2_ling'] for item in test_split])
l2_norms = np.linalg.norm(test_ling, axis=1)

low_idx = np.argmin(l2_norms)
high_idx = np.argmax(l2_norms)
mid_idx = np.median(l2_norms)
mid_idx = np.argmin(np.abs(l2_norms - mid_idx))

# 选取不同复杂度的样本进行测试
test_indices = [low_idx, mid_idx, high_idx]

print("\n" + "="*80)
print("测试 LingDisc 预测准确性")
print("="*80)

for idx in test_indices:
    sample = test_split[idx]
    true_ling = sample['sentence2_ling']
    sentence2 = sample['sentence2']

    print(f"\n--- 样本 idx={idx} ---")
    print(f"句子: {sentence2}")
    print(f"真实 L2: {np.linalg.norm(true_ling):.4f}")

    # 用 LingDisc 预测
    inputs = tokenizer(sentence2, return_tensors='pt', truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        pred_ling = ling_disc(**inputs).cpu().numpy()[0]

    # 计算差异
    l2_error = np.linalg.norm(pred_ling - true_ling)
    cos_sim = cosine(pred_ling, true_ling)

    print(f"预测 L2: {np.linalg.norm(pred_ling):.4f}")
    print(f"L2 误差: {l2_error:.4f}")
    print(f"余弦相似度: {cos_sim:.4f}")
    print(f"\n真实向量前10维: {true_ling[:10]}")
    print(f"预测向量前10维: {pred_ling[:10]}")

print("\n" + "="*80)
print("统计整体误差 (采样100个样本)")
print("="*80)

sample_size = min(100, len(test_split))
indices = np.random.choice(len(test_split), sample_size, replace=False)

l2_errors = []
cos_sims = []

for idx in indices:
    sample = test_split[idx]
    true_ling = sample['sentence2_ling']
    sentence2 = sample['sentence2']

    inputs = tokenizer(sentence2, return_tensors='pt', truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        pred_ling = ling_disc(**inputs).cpu().numpy()[0]

    l2_error = np.linalg.norm(pred_ling - true_ling)
    cos_sim = cosine(pred_ling, true_ling)

    l2_errors.append(l2_error)
    cos_sims.append(cos_sim)

l2_errors = np.array(l2_errors)
cos_sims = np.array(cos_sims)

print(f"\nL2 误差统计:")
print(f"  平均值: {l2_errors.mean():.4f}")
print(f"  标准差: {l2_errors.std():.4f}")
print(f"  最小值: {l2_errors.min():.4f}")
print(f"  最大值: {l2_errors.max():.4f}")

print(f"\n余弦相似度统计:")
print(f"  平均值: {cos_sims.mean():.4f}")
print(f"  标准差: {cos_sims.std():.4f}")
print(f"  最小值: {cos_sims.min():.4f}")
print(f"  最大值: {cos_sims.max():.4f}")

print("\n测试完成")
