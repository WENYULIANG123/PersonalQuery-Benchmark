"""
Stage 5: 训练 LingDisc 语言判别器
用于 QC 反馈优化，根据生成句子的 logits 预测语言特征
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, T5Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from data import load_data, LingDataCollator
from model import LingDisc
from options import parse_args


class LingDiscTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_name)

        # 加载数据
        data, _, _ = load_data(args, self.tokenizer)
        self.train_data = data.get('train')
        self.dev_data = data.get('dev')

        if self.train_data is None:
            raise ValueError("训练数据中没有 train 集")

        print(f"训练集大小: {len(self.train_data)}")
        print(f"验证集大小: {len(self.dev_data) if self.dev_data else 0}")

        # 初始化模型
        self.model = LingDisc(
            model_name=args.model_name,
            disc_type=args.disc_type,
            disc_ckpt=None,  # 从头训练
            lng_dim=40,
            quant_nbins=1,
        ).to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # 数据整理器
        self.collator = LingDataCollator(self.tokenizer)

    def compute_loss(self, batch, training=True):
        """计算 MSE loss - 预测语言特征与目标特征的差距"""
        input_ids = batch.get('input_ids')
        labels = batch.get('labels')  # sentence2 的 token ids

        if labels is None:
            return None

        # 获取模型预测
        # LingDisc 接收 logits（来自主模型的输出）
        # 训练时我们用 labels 作为 proxy
        with torch.set_grad_enabled(training):
            # 对于训练，我们直接用 sentence2 作为输入预测其语言特征
            # 注意：LingDisc 实际期望的是 logits，但训练时我们用交叉熵作为 proxy
            output = self.model(input_ids=input_ids, attention_mask=batch.get('attention_mask'))

            # 目标语言特征
            targets = batch.get('sentence2_ling_40') or batch.get('sentence2_ling')
            if targets is None:
                return None

            # MSE Loss
            loss = F.mse_loss(output, targets)

        return loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        dataloader = DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )

        for batch in tqdm(dataloader, desc=f'Epoch {epoch}'):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad()
            loss = self.compute_loss(batch, training=True)

            if loss is None:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        dataloader = DataLoader(
            self.dev_data,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                loss = self.compute_loss(batch, training=False)
                if loss is not None:
                    total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self):
        best_loss = float('inf')
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collator,
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
        model_dir = os.path.join(self.args.output_dir, 'lingdisc_model')
        os.makedirs(model_dir, exist_ok=True)

        # 直接保存模型完整对象
        torch.save(self.model, os.path.join(model_dir, 'model.pt'))

        # 保存 tokenizer
        self.tokenizer.save_pretrained(model_dir)

        print(f"模型已保存到: {model_dir}")


def main():
    args, _, _ = parse_args()

    # LingDisc 专用参数
    args.disc_type = 't5'  # 使用 t5 类型而非 deberta
    args.batch_size = getattr(args, 'batch_size', 32)
    args.eval_batch_size = getattr(args, 'eval_batch_size', 64)
    args.lr = getattr(args, 'lr', 1e-4)
    args.weight_decay = getattr(args, 'weight_decay', 0.01)
    args.warmup_steps = getattr(args, 'warmup_steps', 500)
    args.max_grad_norm = getattr(args, 'max_grad_norm', 1.0)
    args.epochs = getattr(args, 'epochs', 3)
    args.output_dir = getattr(args, 'output_dir', '/home/wlia0047/ar57_scratch/wenyu/lingconv_disc')
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

    trainer = LingDiscTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
