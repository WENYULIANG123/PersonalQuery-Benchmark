import os.path as osp
import torch
from typing import Any, Union, List, Dict
from tqdm import tqdm
from stark_qa.models.base import ModelForSTaRKQA
from stark_qa.evaluator import Evaluator


class VSS(ModelForSTaRKQA):
    """
    Vector Similarity Search (VSS) model for knowledge base retrieval.
    
    This model performs similarity search by comparing query embeddings against precomputed
    candidate embeddings from a knowledge base using vector similarity measures (e.g., dot product).
    """
    
    def __init__(self, 
                 skb: Any, 
                 query_emb_dir: str, 
                 candidates_emb_dir: str, 
                 emb_model: str = 'text-embedding-ada-002',
                 device: str = 'cuda') -> None:
        """
        Initialize the VSS model.

        Args:
            skb (Any): Knowledge base containing semi-structured data.
            query_emb_dir (str): Directory where query embeddings are stored.
            candidates_emb_dir (str): Directory where candidate embeddings are stored.
            emb_model (str, optional): Embedding model used for similarity search. Defaults to 'text-embedding-ada-002'.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        super(VSS, self).__init__(skb, query_emb_dir=query_emb_dir)
        self.emb_model = emb_model
        self.candidates_emb_dir = candidates_emb_dir
        self.device = device
        self.evaluator = Evaluator(self.candidate_ids, device)

        # Load candidate embeddings
        candidate_emb_path = osp.join(candidates_emb_dir, 'candidate_emb_dict.pt')
        candidate_emb_dict = torch.load(candidate_emb_path)
        print(f'Loaded candidate_emb_dict from {candidate_emb_path}!')

        # assert len(candidate_emb_dict) == len(self.candidate_ids), "Mismatch in candidate embedding count."
        if len(candidate_emb_dict) != len(self.candidate_ids):
             print(f"Warning: Mismatch in candidate embedding count. candidates: {len(self.candidate_ids)}, embeddings: {len(candidate_emb_dict)}")

        # Stack candidate embeddings into a tensor
        # Filter to only use candidates that exist in both candidate_ids and candidate_emb_dict if necessary, 
        # or assuming candidate_ids are keys in candidate_emb_dict.
        # Based on typical usage, candidate_emb_dict keys are the IDs.
        
        # Safe loading: Only load embeddings for IDs that exist in the dictionary
        valid_candidate_ids = []
        candidate_embs = []
        missing_count = 0
        
        for idx in self.candidate_ids:
            if idx in candidate_emb_dict:
                candidate_embs.append(candidate_emb_dict[idx].view(1, -1))
                valid_candidate_ids.append(idx)
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"Warning: {missing_count} candidate IDs missing from embedding dictionary. Using {len(valid_candidate_ids)} valid embeddings.")
            # Update candidate_ids to strictly match the available embeddings to avoid shape mismatch later
            self.candidate_ids = valid_candidate_ids
            
        self.candidate_embs = torch.cat(candidate_embs, dim=0).to(device)

        # L2 normalize embeddings for DPR and ANCE models (cosine similarity via dot product)
        print(f'🔍 Checking emb_model: "{self.emb_model}" (lowercase: "{self.emb_model.lower()}")')
        if 'dpr' in self.emb_model.lower() or 'ance' in self.emb_model.lower():
            print(f'🔄 L2 normalizing embeddings for model: {self.emb_model}')
            self.candidate_embs = torch.nn.functional.normalize(self.candidate_embs, p=2, dim=1)
            print('✅ Candidate embeddings L2 normalized')
        else:
            print(f'ℹ️ Using raw embeddings for model: {self.emb_model}')
    
    def forward(self, 
                query: Union[str, List[str]], 
                query_id: Union[int, List[int]], 
                **kwargs: Any) -> Dict[int, float]:
        """
        Compute similarity scores for the given query using vector similarity.

        Args:
            query (Union[str, list]): Query string or list of query strings.
            query_id (Union[int, list]): Query index or list of indices.
            
        Returns:
            Dict[int, float]: A dictionary mapping candidate IDs to their similarity scores.
        """
        # Get query embeddings
        query_emb = self.get_query_emb(query, query_id, emb_model=self.emb_model, **kwargs)

        # L2 normalize query embeddings for DPR and ANCE models
        if 'dpr' in self.emb_model.lower() or 'ance' in self.emb_model.lower():
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            print(f"🔄 Query embeddings L2 normalized for model: {self.emb_model}")

        # Compute similarity between query and candidate embeddings
        similarity = torch.matmul(query_emb.to(self.device).float(),
                                  self.candidate_embs.T.float()).cpu()

        # Debug: print similarity statistics
        if isinstance(query, str):  # Only for single queries
            sim_stats = {
                'min': similarity.min().item(),
                'max': similarity.max().item(),
                'mean': similarity.mean().item(),
                'std': similarity.std().item()
            }
            print(f"🔍 Similarity stats: min={sim_stats['min']:.4f}, max={sim_stats['max']:.4f}, mean={sim_stats['mean']:.4f}, std={sim_stats['std']:.4f}")

        # Return dictionary mapping candidate IDs to similarity scores for a single query
        if isinstance(query, str):
            return dict(zip(self.candidate_ids, similarity.view(-1)))
        # Return candidate IDs and similarity scores for multiple queries
        else:
            return torch.LongTensor(self.candidate_ids), similarity.t()
