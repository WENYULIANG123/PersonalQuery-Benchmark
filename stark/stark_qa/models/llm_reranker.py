import torch
from typing import Any, Union, List, Dict
import re
import gc
import os

from stark_qa.models.vss import VSS
from stark_qa.models.base import ModelForSTaRKQA
from stark_qa.tools.llm_lib.get_llm_outputs import get_llm_output


class LLMReranker(ModelForSTaRKQA):
    """
    LLM-based reranking model for knowledge base candidate reranking.

    This model uses a combination of vector similarity and an LLM (Language Model)
    to rerank the top candidates from a query.
    """

    def __init__(self,
                 skb: Any,
                 llm_model: str,
                 emb_model: str,
                 query_emb_dir: str,
                 candidates_emb_dir: str,
                 sim_weight: float = 0.1,
                 max_cnt: int = 3,
                 max_k: int = 100,
                 device: str = 'cuda') -> None:
        """
        Initialize the LLMReranker model.

        Args:
            skb (Any): Knowledge base with semi-structured information.
            llm_model (str): Name of the LLM model used for reranking.
            emb_model (str): Name of the embedding model used for vector similarity.
            query_emb_dir (str): Directory where query embeddings are stored.
            candidates_emb_dir (str): Directory where candidate embeddings are stored.
            sim_weight (float, optional): Weight for similarity score in the final ranking. Defaults to 0.1.
            max_cnt (int, optional): Maximum number of retry attempts for LLM response. Defaults to 3.
            max_k (int, optional): Maximum number of top candidates to rerank. Defaults to 100.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        super(LLMReranker, self).__init__(skb)
        self.max_k = max_k
        self.emb_model = emb_model
        self.llm_model = llm_model
        self.sim_weight = sim_weight
        self.max_cnt = max_cnt

        self.query_emb_dir = query_emb_dir
        self.candidates_emb_dir = candidates_emb_dir
        self.parent_vss = VSS(skb, query_emb_dir, candidates_emb_dir, emb_model=emb_model, device=device)

        # Aggressive memory cleanup before LLM initialization
        print("🧹 Performing aggressive GPU memory cleanup...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset memory allocator
            torch.cuda.reset_peak_memory_stats()
            print(f"📊 GPU memory after cleanup: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total, "
                  f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB allocated, "
                  f"{torch.cuda.memory_reserved() / 1024**3:.1f}GB reserved")

            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            # Set memory fraction to leave some headroom
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory for safety
            # Enable expandable segments for better memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    def forward(self, 
                query: Union[str, List[str]],
                query_id: Union[int, List[int]] = None,
                **kwargs: Any) -> Dict[int, float]:
        """
        Compute predictions for the given query using LLM reranking.

        Args:
            query (Union[str, List[str]]): Query string or a list of query strings.
            query_id (Union[int, List[int]], optional): Query index (optional).

        Returns:
            Dict[int, float]: A dictionary mapping candidate IDs to reranked scores.
        """
        # Retrieve initial candidate scores using VSS (Vector Search System)
        initial_score_dict = self.parent_vss(query, query_id)
        node_ids = list(initial_score_dict.keys())
        node_scores = list(initial_score_dict.values())

        # Get the top k candidates based on the initial similarity scores
        top_k_idx = torch.topk(
            torch.FloatTensor(node_scores), 
            min(self.max_k, len(node_scores)), 
            dim=-1
        ).indices.view(-1).tolist()
        top_k_node_ids = [node_ids[i] for i in top_k_idx]
        cand_len = len(top_k_node_ids)

        pred_dict = {}
        for idx, node_id in enumerate(top_k_node_ids):
            node_type = self.skb.get_node_type_by_id(node_id)
            prompt = (
                f'You are an expert search relevance evaluator. Your task is to score the relevance of a product to a user query on a continuous scale from 0.0 to 1.0.\n'
                f'Scoring Rubric:\n'
                f'- 0.0: Completely Irrelevant. Does not match the query.\n'
                f'- 0.5: Partially Relevant. Matches some keywords but misses core intent (e.g. wrong brand or function).\n'
                f'- 1.0: Perfectly Relevant. Matches all constraints (brand, function, attributes) in the query.\n\n'
                f'Query: "{query}"\n'
                f'Product Info:\n{self.skb.get_doc_info(node_id, add_rel=True)}\n\n'
                f'Constraint: Output ONLY a single floating point number between 0.0 and 1.0. Do not output text.\n'
                f'Relevance Score:'
                )

            success = False
            for attempt in range(self.max_cnt):
                try:
                    # Aggressive memory cleanup before each attempt
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # Force garbage collection
                        import gc
                        gc.collect()

                    llm_response = get_llm_output(prompt,
                                                  self.llm_model,
                                                  max_tokens=5
                                                  )
                    print(f"🔍 LLM attempt {attempt + 1}: '{llm_response[:100]}...'")
                    answer = find_floating_number(llm_response)
                    if len(answer) == 1:
                        answer = answer[0]
                        success = True
                        print(f"✅ Parsed score: {answer}")
                        break
                    else:
                        print(f"❌ Could not parse score from: '{llm_response}'")
                except Exception as e:
                    print(f'💥 Exception in attempt {attempt + 1}: {e}')

            if success:
                llm_score = float(answer)
                sim_score = (cand_len - idx) / cand_len
                score = llm_score + self.sim_weight * sim_score
                pred_dict[node_id] = score
            else:
                print(f'⚠️ LLM reranking parsing failed for query "{query}" after {self.max_cnt} attempts. Assigning 0.0.')
                sim_score = (cand_len - idx) / cand_len
                pred_dict[node_id] = 0.0 + self.sim_weight * sim_score
        return pred_dict


def find_floating_number(text):
    pattern = r'0\.\d+|1\.0'
    matches = re.findall(pattern, text)
    return [round(float(match), 4) for match in matches if float(match) <= 1.1]

