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
        self.lr = 5e-5 # Increased LR for faster convergence
        self.temp = 0.1 # Increased temperature
        self.seed = 42
        self.use_lora = False # Disabled LoRA due to environment incompatibility with peft/torch_dist
        self.freeze_encoder = True # Freeze encoder to prevent overfitting on small dataset
        self.clean_csv = "/home/wlia0047/ar57/wenyu/result/clean_query/clean_queries.csv"
        self.noisy_csv = "/home/wlia0047/ar57/wenyu/result/noisy_query/noisy_queries.csv"
        self.meta_path = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz"
        self.output_dir = "/home/wlia0047/ar57/wenyu/stark/code/train_model/results_qwen"

        # Dynamic Loss Weights
        self.w_noisy = 1.0
        self.w_clean = 0.5
        self.w_align_start = 1.0
        self.w_align_end = 0.1

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
def load_metadata(path):
    import gzip
    import json
    logging.info(f"Loading metadata from {path}...")
    meta_map = {}
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    asin = item.get('asin')
                    title = item.get('title')
                    if asin and title:
                        meta_map[asin] = title
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")
    logging.info(f"Loaded {len(meta_map)} items from metadata.")
    return meta_map

class TripletDataset(Dataset):
    def __init__(self, data, tokenizer, meta_map, mode='train'):
        self.data = data
        self.tokenizer = tokenizer
        self.meta_map = meta_map
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        clean_q = str(row.get('clean_query', ''))
        noisy_q = str(row.get('noisy_query', ''))
        item_id = str(row.get('positive_item', ''))
        
        # Use title if available, otherwise fallback to ID
        title = self.meta_map.get(item_id, f"Item {item_id}")
        item_text = f"Product: {title}" 
        
        # DEBUG: Log first few items to ensure title loading works
        if idx < 3 and self.mode == 'train':
            logging.info(f"[DEBUG] Q: {clean_q[:30]}... | Item: {item_text[:50]}...")

        return {
            'clean_q': clean_q,
            'noisy_q': noisy_q,
            'item_text': item_text
        }

def prepare_data(test_size=0.2):
    logging.info("Loading Data...")
    try:
        df_clean = pd.read_csv(config.clean_csv)
        df_noisy = pd.read_csv(config.noisy_csv)
    except Exception as e:
        logging.error(f"Error reading CSVs: {e}")
        return None, None, {}

    if df_noisy.empty:
        logging.error("Noisy CSV is empty!")
        return None, None, {}

    # Ensure IDs are strings and stripped
    df_clean['id'] = df_clean['id'].astype(str).str.strip()
    df_noisy['id'] = df_noisy['id'].astype(str).str.strip()

    # Rename `query` columns
    df_clean = df_clean.rename(columns={'query': 'clean_query'})
    df_noisy = df_noisy.rename(columns={'query': 'noisy_query'})
    
    # Merge on 'id'
    df_merged = pd.merge(df_clean[['id', 'clean_query', 'answer_ids_source']], 
                         df_noisy[['id', 'noisy_query']], 
                         on='id', how='inner')
    
    logging.info(f"Merged Data Size: {len(df_merged)}")
    
    def get_first_item(ids_str):
        import ast
        import json
        try:
            # First try literal_eval
            try:
                ids = ast.literal_eval(str(ids_str))
            except (ValueError, SyntaxError):
                # Fallback to json.loads if literal_eval fails (e.g. for JSON strings)
                try:
                    ids = json.loads(str(ids_str))
                except:
                    # If both fail, treat the string itself as the ID
                    return str(ids_str)
            return ids[0] if isinstance(ids, list) and ids else ids
        except:
            # If all parsing fails, return the string value
            return str(ids_str) if pd.notna(ids_str) else None
            
    df_merged['positive_item'] = df_merged['answer_ids_source'].apply(get_first_item)
    df_merged = df_merged.dropna(subset=['positive_item'])
    df_merged['user_id'] = "AG7EF0SVBQOUX"

    train_df, test_df = train_test_split(df_merged, test_size=test_size, random_state=config.seed)
    
    # Load metadata once
    meta_map = load_metadata(config.meta_path)
    
    logging.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df, meta_map

# --- Model ---
class MultiTaskQwenEncoder(nn.Module):
    def __init__(self, model_name, freeze_encoder=True):
        super().__init__()
        # Load base Qwen2 model
        # attn_implementation="eager" to avoid flash_attn dependency
        self.encoder = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, # Use bfloat16 for better stability
            attn_implementation="eager"
        )

        # Freeze encoder parameters to prevent overfitting on small dataset
        if freeze_encoder:
            logging.info("Freezing encoder parameters - only projector will be trained")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            logging.info("Encoder is trainable - full model fine-tuning")

        hidden_size = self.encoder.config.hidden_size
        self.projector = nn.Linear(hidden_size, 128).to(torch.bfloat16)

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
def contrastive_loss(q_emb, i_emb, temp=None):
    if temp is None: temp = config.temp
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

    model = MultiTaskQwenEncoder(config.model_name, freeze_encoder=config.freeze_encoder).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    train_df, test_df, meta_map = prepare_data()
    train_ds = TripletDataset(train_df, tokenizer, meta_map)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    
    test_ds = TripletDataset(test_df, tokenizer, meta_map)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        # Dynamic weight calculation for multitask
        w_align = config.w_align_start
        if mode == 'multitask':
            w_align = config.w_align_start - (config.w_align_start - config.w_align_end) * (epoch / (config.epochs - 1 if config.epochs > 1 else 1))
            logging.info(f"Epoch {epoch+1} Dynamic Weights - w_align: {w_align:.4f}")

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
                loss = config.w_noisy * l_retrieval_noisy + config.w_clean * l_retrieval_clean + w_align * l_align
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Add gradient clipping
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # Evaluate on both Clean and Noisy queries
    results_clean = evaluate_model(test_loader, tokenizer, model, query_type='clean')
    results_noisy = evaluate_model(test_loader, tokenizer, model, query_type='noisy')

    logging.info(f"\n{'='*95}")
    logging.info(f"EVALUATION RESULTS - Mode: {mode.upper()}")
    logging.info(f"{'='*95}")
    logging.info(f"{'Query Type':<15} | {'MRR':<8} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'Hit@10':<8} | {'NDCG@10':<8}")
    logging.info(f"{'-'*95}")
    logging.info(f"{'Clean':<15} | {results_clean['mrr']:.4f}   | {results_clean['hit_1']:.4f}   | {results_clean['hit_3']:.4f}   | {results_clean['hit_5']:.4f}   | {results_clean['hit_10']:.4f}   | {results_clean['ndcg_10']:.4f}")
    logging.info(f"{'Noisy':<15} | {results_noisy['mrr']:.4f}   | {results_noisy['hit_1']:.4f}   | {results_noisy['hit_3']:.4f}   | {results_noisy['hit_5']:.4f}   | {results_noisy['hit_10']:.4f}   | {results_noisy['ndcg_10']:.4f}")
    logging.info(f"{'='*95}\n")

    # Return combined results
    return {
        'clean': results_clean,
        'noisy': results_noisy
    }

def evaluate_model(test_loader, tokenizer, model, query_type='noisy'):
    """
    Evaluate model on test set.
    Args:
        query_type: 'clean' or 'noisy' - which query type to use for evaluation
    """
    model.eval()
    mrr_sum = 0
    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    hit_10 = 0
    ndcg_10_sum = 0
    count = 0

    logging.info(f"Evaluating on {query_type.upper()} Test Set...")

    # helper to encode all queries and items
    def encode_all_queries_and_items(loader, q_type):
        all_q_embs = []
        all_i_embs = []
        with torch.no_grad():
            for batch in loader:
                # Encode queries based on query_type
                if q_type == 'clean':
                    inputs_q = tokenizer(batch['clean_q'], padding=True, truncation=True,
                                        max_length=config.max_len, return_tensors='pt').to('cuda')
                else:
                    inputs_q = tokenizer(batch['noisy_q'], padding=True, truncation=True,
                                        max_length=config.max_len, return_tensors='pt').to('cuda')
                q_emb = model(inputs_q['input_ids'], inputs_q['attention_mask'])

                # Encode items
                inputs_i = tokenizer(batch['item_text'], padding=True, truncation=True,
                                    max_length=config.max_len, return_tensors='pt').to('cuda')
                i_emb = model(inputs_i['input_ids'], inputs_i['attention_mask'])

                all_q_embs.append(q_emb)
                all_i_embs.append(i_emb)
        return torch.cat(all_q_embs), torch.cat(all_i_embs)

    # Encode all test queries and items
    q_embs_all, i_embs_all = encode_all_queries_and_items(test_loader, query_type)

    # Compute similarity matrix (Num_Queries x Num_Items)
    q_embs_norm = F.normalize(q_embs_all, p=2, dim=1)
    i_embs_norm = F.normalize(i_embs_all, p=2, dim=1)

    # similarity [Q, I]
    sim_matrix = torch.matmul(q_embs_norm, i_embs_norm.T)

    num_queries = sim_matrix.shape[0]

    for i in range(num_queries):
        # The correct item for query i is item i (1-to-1 mapping)
        scores = sim_matrix[i]
        target_score = scores[i]

        # Calculate rank
        rank = (scores > target_score).sum().item() + 1

        mrr_sum += 1.0 / rank
        if rank <= 1: hit_1 += 1
        if rank <= 3: hit_3 += 1
        if rank <= 5: hit_5 += 1
        if rank <= 10:
            hit_10 += 1
            ndcg_10_sum += 1.0 / np.log2(rank + 1)
        count += 1

    mrr = mrr_sum / count
    h1 = hit_1 / count
    h3 = hit_3 / count
    h5 = hit_5 / count
    h10 = hit_10 / count
    ndcg10 = ndcg_10_sum / count

    logging.info(f"Mode: {query_type} | MRR: {mrr:.4f} | Hit@1: {h1:.4f} | Hit@3: {h3:.4f} | Hit@5: {h5:.4f} | Hit@10: {h10:.4f} | NDCG@10: {ndcg10:.4f}")

    return {'mrr': mrr, 'hit_1': h1, 'hit_3': h3, 'hit_5': h5, 'hit_10': h10, 'ndcg_10': ndcg10}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_all', action='store_true', help='Run all three modes (baseline, noise_aug, multitask) sequentially')
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'noise_aug', 'multitask'], help='Training mode (only used if --run_all is not set)')
    parser.add_argument('--epochs', type=int, default=9, help='Number of training epochs (default: 9)')
    args = parser.parse_args()

    config.epochs = args.epochs

    results = {}
    if args.run_all:
        print("\n" + "="*95)
        print(f"RUNNING ALL THREE MODES ({args.epochs} epochs each)")
        print("="*95)

        modes = ['baseline', 'noise_aug', 'multitask']
        for idx, m in enumerate(modes, 1):
            print(f"\n[{idx}/{len(modes)}] Starting {m.upper()} mode...")
            print("-" * 95)
            results[m] = train(mode=m)
            print(f"\n[{idx}/{len(modes)}] {m.upper()} mode completed!")
            # Show both clean and noisy results summary
            clean_res = results[m]['clean']
            noisy_res = results[m]['noisy']
            print(f"  Clean  - MRR: {clean_res['mrr']:.4f} | Hit@10: {clean_res['hit_10']:.4f} | NDCG@10: {clean_res['ndcg_10']:.4f}")
            print(f"  Noisy  - MRR: {noisy_res['mrr']:.4f} | Hit@10: {noisy_res['hit_10']:.4f} | NDCG@10: {noisy_res['ndcg_10']:.4f}")

        print("\n" + "="*130)
        print("FINAL RESULTS - ALL MODES COMPLETED")
        print("="*130)

        # Print Clean Query Results
        print(f"\n{'='*130}")
        print("CLEAN QUERY EVALUATION")
        print(f"{'='*130}")
        print(f"{'Mode':<15} | {'MRR':<8} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'Hit@10':<8} | {'NDCG@10':<8}")
        print("-" * 125)
        for m in modes:
            res = results[m]['clean']
            print(f"{m:<15} | {res['mrr']:.4f}   | {res['hit_1']:.4f}   | {res['hit_3']:.4f}   | {res['hit_5']:.4f}   | {res['hit_10']:.4f}   | {res['ndcg_10']:.4f}")

        # Print Noisy Query Results
        print(f"\n{'='*130}")
        print("NOISY QUERY EVALUATION")
        print(f"{'='*130}")
        print(f"{'Mode':<15} | {'MRR':<8} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'Hit@10':<8} | {'NDCG@10':<8}")
        print("-" * 125)
        for m in modes:
            res = results[m]['noisy']
            print(f"{m:<15} | {res['mrr']:.4f}   | {res['hit_1']:.4f}   | {res['hit_3']:.4f}   | {res['hit_5']:.4f}   | {res['hit_10']:.4f}   | {res['ndcg_10']:.4f}")

        print("="*130)
        print("\nTraining completed successfully!")
    else:
        train(mode=args.mode)
