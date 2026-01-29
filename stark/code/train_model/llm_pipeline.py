import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import ast
import logging
from tqdm import tqdm

# --- Configuration ---
class Config:
    def __init__(self):
        # Using the base Qwen2-1.5B model as requested
        self.model_name = "Qwen/Qwen2-1.5B" 
        self.max_len = 128
        self.batch_size = 4  # Reduced batch size for LLM
        self.epochs = 5
        self.lr = 1e-5 # Lowered LR for stability
        self.seed = 42
        self.use_lora = False # Disabled LoRA due to environment incompatibility with peft/torch_dist
        self.clean_csv = "/home/wlia0047/ar57/wenyu/result/query/kg_queries_clean.csv"
        self.noisy_csv = "/home/wlia0047/ar57/wenyu/result/query/kg_queries_noisy.csv"
        self.output_dir = "/home/wlia0047/ar57/wenyu/stark/code/train_model/results_qwen"

config = Config()

# --- Logging Setup ---
os.makedirs(config.output_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.output_dir, 'training_qwen.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# --- Data Loading ---
class TripletDataset(Dataset):
    def __init__(self, data, tokenizer, mode='train'):
        self.data = data
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        clean_q = str(row.get('clean_query', ''))
        noisy_q = str(row.get('noisy_query', ''))
        item_id = str(row.get('positive_item', ''))
        item_text = f"Item {item_id}" 

        return {
            'clean_q': clean_q,
            'noisy_q': noisy_q,
            'item_text': item_text
        }

def prepare_data(test_size=0.2):
    logging.info("Loading Data...")
    df_clean = pd.read_csv(config.clean_csv)
    df_noisy = pd.read_csv(config.noisy_csv)
    
    df_clean = df_clean.rename(columns={'query': 'clean_query'})
    df_noisy = df_noisy.rename(columns={'query': 'noisy_query'})
    
    df_merged = pd.merge(df_clean[['id', 'clean_query', 'answer_ids']], 
                         df_noisy[['id', 'noisy_query']], 
                         on='id', how='inner')
    
    def get_first_item(ids_str):
        try:
            ids = ast.literal_eval(str(ids_str))
            return ids[0] if ids else None
        except:
            return None
            
    df_merged['positive_item'] = df_merged['answer_ids'].apply(get_first_item)
    df_merged = df_merged.dropna(subset=['positive_item'])
    df_merged['user_id'] = "AG7EF0SVBQOUX"

    train_df, test_df = train_test_split(df_merged, test_size=test_size, random_state=config.seed)
    
    logging.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df

# --- Model ---
class MultiTaskQwenEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Load base Qwen2 model
        # attn_implementation="eager" to avoid flash_attn dependency
        self.encoder = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, # Use bfloat16 for better stability
            attn_implementation="eager"
        )
        
        # We keep the encoder frozen mostly if we want to save memory, 
        # but since user wants to "use LLM", we'll allow gradients for the whole thing
        # but with a very small batch size.
        
        hidden_size = self.encoder.config.hidden_size
        self.projector = nn.Linear(hidden_size, 128).to(torch.bfloat16).to(self.encoder.device)

    def last_token_pool(self, last_hidden_states, attention_mask):
        """Standard last token pooling for causal LMs."""
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs.last_hidden_state[:, -1, :] # Simple last token
        pooled_output = self.last_token_pool(outputs.last_hidden_state, attention_mask)
        return self.projector(pooled_output)

# --- Training wrapper ---
def contrastive_loss(q_emb, i_emb, temp=0.07):
    sim_matrix = F.cosine_similarity(q_emb.unsqueeze(1), i_emb.unsqueeze(0), dim=2) / temp
    labels = torch.arange(q_emb.size(0)).to(q_emb.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def alignment_loss(clean_emb, noisy_emb):
    return 1 - F.cosine_similarity(clean_emb, noisy_emb).mean()

def train(mode='baseline'):
    logging.info(f"Starting Training: Mode = {mode.upper()}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = MultiTaskQwenEncoder(config.model_name).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    train_df, test_df = prepare_data()
    train_ds = TripletDataset(train_df, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    
    test_ds = TripletDataset(test_df, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            def encode_text(texts):
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=config.max_len, return_tensors='pt').to('cuda')
                return model(inputs['input_ids'], inputs['attention_mask'])
            
            # Using half precision
            clean_emb = encode_text(batch['clean_q'])
            noisy_emb = encode_text(batch['noisy_q'])
            item_emb = encode_text(batch['item_text'])
            
            loss = 0
            if mode == 'baseline':
                loss = contrastive_loss(clean_emb, item_emb)
            elif mode == 'noise_aug':
                loss = contrastive_loss(noisy_emb, item_emb)
            elif mode == 'multitask':
                l_align = alignment_loss(clean_emb, noisy_emb)
                l_retrieval_noisy = contrastive_loss(noisy_emb, item_emb)
                l_retrieval_clean = contrastive_loss(clean_emb, item_emb)
                loss = l_retrieval_noisy + 0.5 * l_retrieval_clean + 0.5 * l_align
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Add gradient clipping
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
    logging.info("Evaluating on Noisy Test Set...")
    model.eval()
    mrr_sum = 0
    recall_10 = 0
    count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            noisy_emb = encode_text(batch['noisy_q'])
            item_emb = encode_text(batch['item_text'])
            
            sim_matrix = F.cosine_similarity(noisy_emb.unsqueeze(1), item_emb.unsqueeze(0), dim=2)
            batch_size = noisy_emb.size(0)
            
            for i in range(batch_size):
                scores = sim_matrix[i]
                target_score = scores[i]
                rank = (scores > target_score).sum().item() + 1
                
                mrr_sum += 1.0 / rank
                if rank <= 10:
                    recall_10 += 1
                count += 1
    
    mrr = mrr_sum / count
    r10 = recall_10 / count
    logging.info(f"Mode: {mode} | MRR: {mrr:.4f} | Recall@10: {r10:.4f}")
    return {'mrr': mrr, 'recall_10': r10}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_all', action='store_true')
    args = parser.parse_args()
    
    results = {}
    if args.run_all:
        modes = ['baseline', 'noise_aug', 'multitask']
        for m in modes:
            results[m] = train(mode=m)
            
        print("\n" + "="*60)
        print("FINAL QWEN2 PIPELINE RESULTS")
        print("="*60)
        print(f"{'Mode':<15} | {'MRR':<10} | {'Recall@10':<10}")
        print("-" * 45)
        for m in modes:
            res = results[m]
            print(f"{m:<15} | {res['mrr']:.4f}     | {res['recall_10']:.4f}")
        print("-" * 45)
    else:
        train('baseline')
