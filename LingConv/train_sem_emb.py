"""
Stage 5: 训练 SemEmb 语义嵌入模型
用于 QC 反馈优化，确保生成句子与原句保持语义相似
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, T5Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from data import load_data
from model import SemEmb
from options import parse_args


class SemEmbDataset(torch.utils.data.Dataset):
    """SemEmb 专用数据集"""
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize sentence1
        s1 = self.tokenizer(
            item['sentence1'],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )

        # Tokenize sentence2
        s2 = self.tokenizer(
            item['sentence2'],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )

        # 标签：QQP/MRPC 有 label 字段，STSB 有 similarity 字段
        # 其他情况默认是 paraphrase (label=1)
        if 'label' in item:
            label = float(item['label'])
        elif 'similarity' in item:
            label = float(item['similarity']) / 5.0  # normalize to [0, 1]
        else:
            label = 1.0  # 默认 paraphrase

        return {
            'sentence1_input_ids': s1['input_ids'].squeeze(0),
            'sentence1_attention_mask': s1['attention_mask'].squeeze(0),
            'sentence2_input_ids': s2['input_ids'].squeeze(0),
            'sentence2_attention_mask': s2['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32),
        }


def collate_fn(batch):
    """自定义 collate 函数"""
    max_len1 = max(x['sentence1_input_ids'].shape[0] for x in batch)
    max_len2 = max(x['sentence2_input_ids'].shape[0] for x in batch)

    s1_ids = []
    s1_mask = []
    s2_ids = []
    s2_mask = []
    labels = []

    for item in batch:
        # Pad sentence1
        pad_len1 = max_len1 - item['sentence1_input_ids'].shape[0]
        s1_ids.append(F.pad(item['sentence1_input_ids'], (0, pad_len1), value=0))
        s1_mask.append(F.pad(item['sentence1_attention_mask'], (0, pad_len1), value=0))

        # Pad sentence2
        pad_len2 = max_len2 - item['sentence2_input_ids'].shape[0]
        s2_ids.append(F.pad(item['sentence2_input_ids'], (0, pad_len2), value=0))
        s2_mask.append(F.pad(item['sentence2_attention_mask'], (0, pad_len2), value=0))

        labels.append(item['label'])

    return {
        'sentence1_input_ids': torch.stack(s1_ids),
        'sentence1_attention_mask': torch.stack(s1_mask),
        'sentence2_input_ids': torch.stack(s2_ids),
        'sentence2_attention_mask': torch.stack(s2_mask),
        'labels': torch.stack(labels),
    }


class SemEmbTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        self.sep_token_id = self.tokenizer.get_vocab()['</s>']

        # 加载数据
        data, _, _ = load_data(args, self.tokenizer)
        self.train_data = data.get('train')
        self.dev_data = data.get('dev')

        if self.train_data is None:
            raise ValueError("训练数据中没有 train 集")

        print(f"训练集大小: {len(self.train_data)}")
        print(f"验证集大小: {len(self.dev_data) if self.dev_data else 0}")

        # 初始化模型
        t5_encoder = T5EncoderModel.from_pretrained(args.model_name)
        self.model = SemEmb(t5_encoder.config, self.sep_token_id).to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    def compute_loss(self, batch, training=True):
        """计算 BCE loss - 预测语义相似度"""
        with torch.set_grad_enabled(training):
            logits = self.model.compare_sem(**batch)
            probs = torch.sigmoid(logits).squeeze(-1)
            labels = batch['labels']

            # BCE Loss
            loss = F.binary_cross_entropy(probs, labels)

        return loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        dataloader = DataLoader(
            SemEmbDataset(self.train_data, self.tokenizer, self.args.max_length),
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        for batch in tqdm(dataloader, desc=f'Epoch {epoch}'):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad()
            loss = self.compute_loss(batch, training=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        dataloader = DataLoader(
            SemEmbDataset(self.dev_data, self.tokenizer, self.args.max_length),
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                loss = self.compute_loss(batch, training=False)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self):
        best_loss = float('inf')
        train_loader = DataLoader(
            SemEmbDataset(self.train_data, self.tokenizer, self.args.max_length),
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        num_training_steps = len(train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            eval_loss = self.evaluate() if self.dev_data else train_loss

            print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")

            # 保存最佳模型
            if eval_loss < best_loss:
                best_loss = eval_loss
                self.save_model()
                print(f"保存最佳模型 (loss={best_loss:.4f})")

            scheduler.step()

    def save_model(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        # 保存完整模型目录结构
        model_dir = os.path.join(self.args.output_dir, 'sememb_model')
        os.makedirs(model_dir, exist_ok=True)

        # 直接保存模型完整对象
        torch.save(self.model, os.path.join(model_dir, 'model.pt'))

        # 保存 tokenizer
        self.tokenizer.save_pretrained(model_dir)

        print(f"模型已保存到: {model_dir}")


def main():
    args, _, _ = parse_args()

    # SemEmb 专用参数
    args.batch_size = getattr(args, 'batch_size', 32)
    args.eval_batch_size = getattr(args, 'eval_batch_size', 64)
    args.lr = getattr(args, 'lr', 1e-4)
    args.weight_decay = getattr(args, 'weight_decay', 0.01)
    args.warmup_steps = getattr(args, 'warmup_steps', 500)
    args.max_grad_norm = getattr(args, 'max_grad_norm', 1.0)
    args.epochs = getattr(args, 'epochs', 3)
    args.output_dir = getattr(args, 'output_dir', '/home/wlia0047/ar57_scratch/wenyu/lingconv_sem')
    args.data_sources = ["qqp", "mrpc", "stsb"]

    # 默认数据配置
    args.data_dir = '/home/wlia0047/ar57_scratch/wenyu/ling_conversion_official'
    args.data = 'ling_conversion'
    args.src_lng = 'ling'
    args.quantize_lng = False
    args.quant_nbins = 1
    args.do_imputation = False
    args.imputation_percentage = 20
    args.imputation_seed = 0
    args.use_ica = False
    args.n_ica = 10
    args.max_length = 128
    args.prepend_prompt = False
    args.prompt_text = ''
    args.use_lingpred = False
    args.lng_ids = None
    args.lng_ids_idx = None
    args.lng_ids_path = './indices'
    args.aug_same = False
    args.max_eval_samples = 3000
    args.seed = 42

    trainer = SemEmbTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
