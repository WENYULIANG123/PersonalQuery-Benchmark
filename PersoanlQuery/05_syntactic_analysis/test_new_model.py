"""测试 LingDisc 预测准确性"""
import torch
import numpy as np
import importlib.util
import argparse
import sys
from scipy.spatial.distance import cosine

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/LingConv')

model_path = '/home/wlia0047/ar57/wenyu/LingConv/model.py'
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

print("Loading LINGCONV model...")
model, _, _ = model_module.get_model(args, tokenizer, device)
model.eval()
print("LINGCONV model loaded successfully!")

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

print("Loading SemEmb model...")
from transformers import T5EncoderModel
sem_encoder = T5EncoderModel.from_pretrained(args.model_name).to(device)
sem_encoder.eval()
print("SemEmb model loaded successfully!")


def get_semantic_embedding(text, tokenizer, encoder):
    """获取句子的语义向量"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = encoder(**inputs)
        # 使用 mean pooling 作为句子嵌入
        emb = outputs.last_hidden_state.mean(dim=1)
    return emb.cpu().numpy()[0]


def compute_semantic_sim(text1, text2, tokenizer, encoder):
    """计算两个文本之间的语义相似度"""
    emb1 = get_semantic_embedding(text1, tokenizer, encoder)
    emb2 = get_semantic_embedding(text2, tokenizer, encoder)
    # 余弦相似度
    sim = 1 - cosine(emb1, emb2)
    return sim

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

# 获取不同复杂度的样本
test_ling = np.array([item['sentence2_ling'] for item in test_split])
l2_norms = np.linalg.norm(test_ling, axis=1)

low_idx = np.argmin(l2_norms)
high_idx = np.argmax(l2_norms)
mid_idx = np.median(l2_norms)
mid_idx = np.argmin(np.abs(l2_norms - mid_idx))

print(f"\n数据集复杂度分布:")
print(f"  低复杂度 L2: {l2_norms[low_idx]:.2f}")
print(f"  中复杂度 L2: {l2_norms[mid_idx]:.2f}")
print(f"  高复杂度 L2: {l2_norms[high_idx]:.2f}")

test_cases = [
    ("低复杂度", test_split[low_idx], low_idx),
    ("中复杂度", test_split[mid_idx], mid_idx),
    ("高复杂度", test_split[high_idx], high_idx),
]

# 测试输入
input_text = "I am looking for Omnigrid Yellow Ruler Racks designed for scrapbooking purposes, priced around $19.44, and I would like to know more details about this product."
input_ids = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=128)

for name, sample, idx in test_cases:
    print(f"\n{'='*60}")
    print(f"测试: {name} (idx={idx})")
    print(f"  目标 L2: {np.linalg.norm(sample['sentence2_ling']):.2f}")
    print(f"  sentence1 (原句): {sample['sentence1']}")
    print(f"  sentence2 (目标句): {sample['sentence2']}")
    print(f"{'='*60}")

    sentence1_ling = torch.zeros(1, 40).to(device)
    sentence2_ling = torch.tensor([sample['sentence2_ling']]).float().to(device)

    # 构建 batch
    batch = {
        "input_ids": input_ids['input_ids'].to(device),
        "attention_mask": input_ids['attention_mask'].to(device),
        "sentence1_ling": sentence1_ling,
        "sentence2_ling": sentence2_ling,
    }

    print(f"  输入: {input_text}")
    print(f"  目标 sentence2_ling L2: {np.linalg.norm(sample['sentence2_ling']):.4f}")
    print(f"  生成 (10个候选, do_sample=True, temperature=2.0):")

    candidates = []
    for i in range(10):
        with torch.no_grad():
            dec_output = model(**batch, generate=True, do_sample=True, temperature=2.0, top_p=0.95)

        if hasattr(dec_output, 'sequences'):
            output_ids = dec_output.sequences
        else:
            output_ids = dec_output

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        candidates.append(generated)

    # 用 LingDisc 预测每个候选的复杂度向量，并计算与目标的距离
    target_ling = sample['sentence2_ling']
    print(f"\n  === 候选句子与目标复杂度距离 ===")

    dist_results = []
    for i, cand in enumerate(candidates):
        # 用 LingDisc 预测候选句子的复杂度
        cand_input = tokenizer(cand, return_tensors='pt', truncation=True, max_length=128, padding=True)
        cand_input = {k: v.to(device) for k, v in cand_input.items()}

        with torch.no_grad():
            pred_ling = ling_disc(**cand_input).cpu().numpy()[0]

        # 计算复杂度距离
        ling_l2_dist = np.linalg.norm(pred_ling - target_ling)
        ling_cos_sim = 1 - cosine(pred_ling, target_ling)

        # 计算语义相似度（与原始输入）
        sem_sim = compute_semantic_sim(input_text, cand, tokenizer, sem_encoder)

        dist_results.append((i+1, cand, ling_l2_dist, ling_cos_sim, sem_sim, pred_ling))
        print(f"    [{i+1}] Ling_L2={ling_l2_dist:.4f}, Ling_Cos={ling_cos_sim:.4f}, Sem_Sim={sem_sim:.4f}:")
        print(f"        {cand}")

    # 按复杂度 L2 距离排序
    dist_results.sort(key=lambda x: x[2])
    print(f"\n  === 按复杂度 L2 距离排序 ===")
    for rank, (i, cand, ling_l2_dist, ling_cos_sim, sem_sim, _) in enumerate(dist_results, 1):
        print(f"    #{rank} [{i}] Ling_L2={ling_l2_dist:.4f}, Sem_Sim={sem_sim:.4f}:")
        print(f"        {cand}")

print("\n测试完成")
