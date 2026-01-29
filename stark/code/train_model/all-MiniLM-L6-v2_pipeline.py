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
import json
import logging
from tqdm import tqdm

# --- Configuration ---
class Config:
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2" # Lightweight for demo, can swap
        self.max_len = 128
        self.batch_size = 16
        self.epochs = 5
        self.lr = 2e-5
        self.seed = 42
        self.clean_csv = "/home/wlia0047/ar57/wenyu/result/generated_kg_queries.csv"
        self.noisy_csv = "/home/wlia0047/ar57/wenyu/result/query/generated_kg_queries_with_errors.csv"
        self.output_dir = "/home/wlia0047/ar57/wenyu/stark/code/train_model/results"

config = Config()

# --- Logging Setup ---
os.makedirs(config.output_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.output_dir, 'training.log'),
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
        
        # Inputs depending on mode/experiment type
        # We always return all available fields, model decides what to use
        clean_q = str(row.get('clean_query', ''))
        noisy_q = str(row.get('noisy_query', ''))
        # Represent item as text ID for now, in real dense retrieval we'd map to item content
        # For this PoC, we used contrastive learning with in-batch negatives or hard negatives
        # Here we simulate by just encoding query. 
        # WAIT: To do retrieval training we need positives.
        # Assuming answer_ids is list of item IDs. We treat the first answer as the positive item.
        # In a real setup, we need item text. Let's placeholders item text = "Item " + ID
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
    
    # Merge on ID
    # Clean: id, query, answer_ids_source
    # Noisy: id, query, answer_ids_source
    df_clean = df_clean.rename(columns={'query': 'clean_query', 'answer_ids_source': 'answer_ids'})
    df_noisy = df_noisy.rename(columns={'query': 'noisy_query'})
    
    # Select relevant columns
    df_merged = pd.merge(df_clean[['id', 'clean_query', 'answer_ids']], 
                         df_noisy[['id', 'noisy_query']], 
                         on='id', how='inner')
    
    # Extract positive item (first one)
    def get_first_item(ids_str):
        try:
            ids = ast.literal_eval(str(ids_str))
            return ids[0] if ids else None
        except:
            return None
            
    df_merged['positive_item'] = df_merged['answer_ids'].apply(get_first_item)
    df_merged = df_merged.dropna(subset=['positive_item'])
    
    # Add User ID (Mock for single user)
    df_merged['user_id'] = "AG7EF0SVBQOUX"

    # Split
    # We want "Noisy Test Set" -> seen items but unseen queries (noisy versions)
    train_df, test_df = train_test_split(df_merged, test_size=test_size, random_state=config.seed)
    
    logging.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df

# --- Model ---
class MultiTaskEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # Shared projection head
        self.projector = nn.Linear(self.encoder.config.hidden_size, 128) 

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.last_hidden_state[:, 0] # CLS token
        return self.projector(pooler_output)

# --- Training wrapper ---
def contrastive_loss(q_emb, i_emb, temp=0.07):
    # SimCLR style / InfoNCE
    sim_matrix = F.cosine_similarity(q_emb.unsqueeze(1), i_emb.unsqueeze(0), dim=2) / temp
    labels = torch.arange(q_emb.size(0)).to(q_emb.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def alignment_loss(clean_emb, noisy_emb):
    # MSE or Cosine distance to pull representations close
    return 1 - F.cosine_similarity(clean_emb, noisy_emb).mean()

def train(mode='baseline'):
    logging.info(f"Starting Training: Mode = {mode.upper()}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = MultiTaskEncoder(config.model_name).cuda()
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
            
            # Prepare Inputs
            # In a real implementation we would tokenize properly with padding/truncating
            # Simplifying for brevity: tokenize on fly
            
            def encode_text(texts):
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=config.max_len, return_tensors='pt').to('cuda')
                return model(inputs['input_ids'], inputs['attention_mask'])
            
            # 1. Clean Query
            clean_emb = encode_text(batch['clean_q'])
            # 2. Noisy Query
            noisy_emb = encode_text(batch['noisy_q'])
            # 3. Item
            item_emb = encode_text(batch['item_text'])
            
            # --- Logic per Mode ---
            loss = 0
            
            if mode == 'baseline':
                # Train only on Clean <-> Item
                loss = contrastive_loss(clean_emb, item_emb)
                
            elif mode == 'noise_aug':
                # Train on Noisy <-> Item (Augmentation)
                loss = contrastive_loss(noisy_emb, item_emb)
                
            elif mode == 'multitask':
                # Task A: Alignment (Clean <-> Noisy)
                l_align = alignment_loss(clean_emb, noisy_emb)
                
                # Task B: Retrieval (Noisy <-> Item AND Clean <-> Item)
                l_retrieval_noisy = contrastive_loss(noisy_emb, item_emb)
                l_retrieval_clean = contrastive_loss(clean_emb, item_emb)
                
                # Combined Loss
                loss = l_retrieval_noisy + 0.5 * l_retrieval_clean + 0.5 * l_align
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
    # --- Evaluation (Inline) ---
    logging.info("Evaluating on Noisy Test Set...")
    model.eval()
    mrr_sum = 0
    recall_10 = 0
    count = 0
    
    # Pre-encode all test items (in this simplified set, item set = test batch items for simplicity,
    # usually we encode ALL candidate items)
    # For robust eval: we use in-batch ranking on test set (Simulated Retrieval)
    
    with torch.no_grad():
        for batch in test_loader:
            noisy_emb = encode_text(batch['noisy_q'])
            item_emb = encode_text(batch['item_text'])
            
            # Similarity matrix [Batch, Batch]
            # Row i is query i, Col j is item j
            sim_matrix = F.cosine_similarity(noisy_emb.unsqueeze(1), item_emb.unsqueeze(0), dim=2)
            
            # Correct index is diagonal (0, 1, 2...)
            batch_size = noisy_emb.size(0)
            
            for i in range(batch_size):
                scores = sim_matrix[i]
                target_score = scores[i]
                
                # Rank: how many scores are greater than target?
                # Using strict inequality
                rank = (scores > target_score).sum().item() + 1
                
                mrr_sum += 1.0 / rank
                if rank <= 10:
                    recall_10 += 1
                count += 1
    
    mrr = mrr_sum / count
    r10 = recall_10 / count
    logging.info(f"Mode: {mode} | MRR: {mrr:.4f} | Recall@10: {r10:.4f}")
    
    return {'mrr': mrr, 'recall_10': r10}

# --- Main Runner ---
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
        print("FINAL PIPELINE RESULTS")
        print("="*60)
        print(f"{'Mode':<15} | {'MRR':<10} | {'Recall@10':<10}")
        print("-" * 45)
        for m in modes:
            res = results[m]
            print(f"{m:<15} | {res['mrr']:.4f}     | {res['recall_10']:.4f}")
        print("-" * 45)
    else:
        # Default run just one for testing
        train('baseline')
