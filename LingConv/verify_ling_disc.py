"""验证新的 LingDisc 输出范围"""
import torch
import importlib.util
import sys
sys.path.insert(0, '.')

# Load model
model_path = 'model.py'
spec = importlib.util.spec_from_file_location('model', model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

from transformers import T5Tokenizer
import argparse
import numpy as np

ckpt_dir = '/home/wlia0047/ar57_scratch/wenyu/lingconv_official6/0405_01-47-33-ling_conversion-decoder_add_first'
disc_ckpt = '/home/wlia0047/ar57_scratch/wenyu/lingconv_disc'

tokenizer = T5Tokenizer.from_pretrained(ckpt_dir)
parser = argparse.ArgumentParser()
parser.add_argument('--combine_method', default='decoder_add_first')
parser.add_argument('--ling2_only', type=bool, default=True)
parser.add_argument('--use_semantic_pooling', type=bool, default=False)
parser.add_argument('--sem_loss', type=bool, default=False)
parser.add_argument('--disc_loss', type=bool, default=False)
parser.add_argument('--combine_weight', type=float, default=1.0)
parser.add_argument('--hidden_dim', type=int, default=500)
parser.add_argument('--lng_dim', type=int, default=40)
parser.add_argument('--ling_dropout', type=float, default=0.1)
parser.add_argument('--initializer_range', type=float, default=0.02)
parser.add_argument('--model_name', default='google/flan-t5-base')
parser.add_argument('--pretrain_disc', action='store_true')
parser.add_argument('--disc_ckpt', default=disc_ckpt)
parser.add_argument('--disc_type', default='t5')
parser.add_argument('--ckpt', default=ckpt_dir)
parser.add_argument('--ling_embed_type', default='one-layer')
parser.add_argument('--injection_type', default='first')
parser.add_argument('--injection_layer', type=int, default=1)
parser.add_argument('--ling_vae', action='store_true')
parser.add_argument('--latent_dim', type=int, default=150)
parser.add_argument('--sem_loss_type', default='dedicated')
parser.add_argument('--use_lingpred', action='store_true')
parser.add_argument('--process_lingpred', action='store_true')
parser.add_argument('--aug_same', type=bool, default=False)
parser.add_argument('--freeze_lm', action='store_true')
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--feedback_param', default='logits')
parser.add_argument('--sem_ckpt', default=None)
args = parser.parse_args([])

device = torch.device('cuda:0')
model, ling_disc, _ = model_module.get_model(args, tokenizer, device)
ling_disc.eval()

print('Loading data...')
from data import load_data
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
    max_eval_samples=100,
    seed=0
)
data, _, _ = load_data(data_args, tokenizer)
test_data = data['test']

# Get a sample and its features
sample = test_data[0]
sentence = sample['sentence2']
print(f'Sample sentence: {sentence}')
print(f'True ling features (first 5): {sample["sentence2_ling"][:5]}')
print(f'True ling L2 norm: {np.linalg.norm(sample["sentence2_ling"]):.4f}')

# Get model logits
inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=128)
batch = {
    'input_ids': inputs['input_ids'].to(device),
    'sentence1_ling': torch.tensor([sample['sentence2_ling']]).to(device),
    'sentence2_ling': torch.tensor([sample['sentence2_ling']]).to(device),
}
scores = model.infer_with_cache(batch)[1]['scores']

with torch.no_grad():
    pred = ling_disc(logits=scores)
pred_cpu = pred[0].cpu().tolist()
print(f'Predicted features (first 5): {pred_cpu[:5]}')
print(f'Predicted L2 norm: {np.linalg.norm(pred_cpu):.4f}')

# MSE loss
mse = torch.mean((pred[0] - torch.tensor(sample['sentence2_ling']).to(device))**2)
print(f'MSE: {mse.item():.4f}')

# Try with different target scales
print()
print('--- Testing with different target scales ---')
for target_val in [0.5, 1.0, 2.0, 3.0, 5.0]:
    target = torch.tensor([target_val] * 40).to(device)
    mse_target = torch.mean((pred[0] - target)**2)
    print(f'MSE with target={target_val}: {mse_target.item():.4f}')

print()
print('--- Verifying scaler is correct ---')
import joblib
scaler = joblib.load('assets/scaler.bin')
print(f'Scaler mean[:5]: {scaler.mean_[:5]}')
print(f'Scaler scale_[:5]: {scaler.scale_[:5]}')
