import os
import os.path as osp
import subprocess
from typing import Any, Union, List, Dict, Optional
from collections import defaultdict
import pandas as pd
from datetime import datetime

import torch
from tqdm import tqdm

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

from stark_qa.models.base import ModelForSTaRKQA
from stark_qa import load_qa


def create_strategy_dataset(strategy_name):
    """Create STaRK-compatible dataset for a specific strategy."""
    import pandas as pd
    import os.path as osp

    if strategy_name == 'original':
        return None


    variants_file = f"/home/wlia0047/ar57/wenyu/stark/data/strategy_variants/{strategy_name}_variants_81.csv"
    stark_base_dir = f"/home/wlia0047/ar57/wenyu/stark/data/stark_strategy_{strategy_name}_dataset"

    os.makedirs(stark_base_dir, exist_ok=True)

    if not os.path.exists(variants_file):
        raise FileNotFoundError(f"Variants file not found: {variants_file}")

    df = pd.read_csv(variants_file)

    # All variant files now use the new format: id, query, answer_ids, answer_ids_source
    stark_df = pd.DataFrame({
        'id': range(len(df)),
        'query': df['query'],
        'answer_ids': df['answer_ids'],
        'query_type': ['variant'] * len(df)
    })

    # Create STaRK directory structure
    qa_dir = os.path.join(stark_base_dir, "qa", "amazon")
    split_dir = os.path.join(qa_dir, "split")
    stark_qa_dir = os.path.join(qa_dir, "stark_qa")

    os.makedirs(stark_qa_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    stark_qa_file = os.path.join(stark_qa_dir, "stark_qa.csv")
    stark_df.to_csv(stark_qa_file, index=False)

    # Create split file
    split_file = os.path.join(split_dir, "variants.index")
    with open(split_file, 'w') as f:
        for i in range(len(stark_df)):
            f.write(f"{i}\n")

    print(f"Created STaRK dataset for strategy '{strategy_name}' at: {stark_base_dir}")
    return stark_base_dir


class Colbertv2(ModelForSTaRKQA):
    """
    ColBERTv2 Model for STaRK QA.

    This model integrates the ColBERTv2 dense retrieval model to rank candidates based on their relevance
    to a query from a question-answering dataset.
    """
    
    url = "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz"
    
    def __init__(self,
                 skb: Any,
                 dataset_name: str,
                 human_generated_eval: bool,
                 add_rel: bool = False,
                 download_dir: str = 'output',
                 save_dir: str = 'output/colbertv2.0',
                 experiments_dir: str = './experiments',  # Custom experiments directory
                 nbits: int = 2,
                 k: int = 100,
                 kmeans_iterations: int = 10,
                 strategy: str = 'original',  # Strategy for query variants
                 dataset_root: str = None):  # Dataset root for variants
        """
        Initialize the ColBERTv2 model with the given knowledge base and parameters.

        Args:
            skb (Any): The knowledge base containing candidate documents.
            dataset_name (str): The name of the dataset being used.
            human_generated_eval (bool): Whether to use human-generated queries for evaluation.
            add_rel (bool, optional): Whether to add relational information to the document. Defaults to False.
            download_dir (str, optional): Directory where the ColBERTv2 model is downloaded. Defaults to 'output'.
            save_dir (str, optional): Directory where the experiment output is saved. Defaults to 'output/colbertv2.0'.
            experiments_dir (str, optional): Directory where experiments and indexes are stored. Defaults to './experiments'.
            nbits (int, optional): Number of bits for indexing. Defaults to 2.
            k (int, optional): Number of top candidates to retrieve. Defaults to 100.
            kmeans_iterations (int, optional): Number of K-means iterations. Defaults to 10.
            strategy (str, optional): Strategy for query variants ('original', 'character', 'embedding', 'other', 'typo', 'wordnet', 'error_aware', 'all'). Defaults to 'original'.
        """
        super(Colbertv2, self).__init__(skb)

        self.k = k
        self.nbits = nbits
        self.kmeans_iterations = kmeans_iterations

        # Handle strategy parameter
        valid_strategies = ['original', 'character', 'embedding', 'other', 'typo', 'wordnet', 'error_aware', 'grammar_aware', 'kg_query', 'all']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of: {valid_strategies}")
        self.strategy = strategy
        self.strategies = ['original', 'character', 'embedding', 'other', 'typo', 'wordnet', 'error_aware', 'grammar_aware', 'kg_query'] if strategy == 'all' else [strategy]
        self.dataset_root = dataset_root

        # Determine query file name based on strategy
        if strategy != 'original':
            query_tsv_name = f'query_{strategy}.tsv'
        else:
            query_tsv_name = 'query_hg.tsv' if human_generated_eval else 'query.tsv'

        self.exp_name = dataset_name + '_hg'  # Unified index path for all strategies

        self.save_dir = save_dir
        self.download_dir = download_dir
        self.experiments_dir = experiments_dir
        
        self.model_ckpt_dir = osp.join(self.download_dir, 'colbertv2.0') 
        self.query_tsv_path = osp.join(self.save_dir, query_tsv_name)
        self.doc_tsv_path = osp.join(self.save_dir, 'doc.tsv')
        self.index_ckpt_path = osp.join(self.save_dir, 'index.faiss')
        self.ranking_path = osp.join(self.save_dir, 'ranking.tsv')

        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        # Load the question-answer dataset and check for required files
        qa_dataset = load_qa(dataset_name, self.dataset_root, human_generated_eval=human_generated_eval)
        self._check_query_csv(qa_dataset, self.query_tsv_path)
        self._check_doc_csv(skb, self.doc_tsv_path, add_rel)

        # Download and set up the ColBERTv2 model
        self._download()

        # Load the queries and documents into ColBERTv2 format
        self.queries = Queries(self.query_tsv_path)
        self.collection = Collection(self.doc_tsv_path)

        # Prepare the indexer and build the index
        self._prepare_indexer()

        # Run the model and store the results
        # Initialize score_dict for the default strategy
        self._current_strategy = self.strategy if isinstance(self.strategy, str) else self.strategy[0] if self.strategy else 'original'
        self._last_strategy = None
        self.score_dict = self.run_all()
    
    def _check_query_csv(self, qa_dataset: Any, query_tsv_path: str) -> None:
        """
        Check if the query TSV file exists; if not, create it from the QA dataset.

        Args:
            qa_dataset (Any): The question-answer dataset.
            query_tsv_path (str): Path to the query TSV file.
        """
        if not osp.exists(query_tsv_path):
            queries = {qa_dataset[i][1]: qa_dataset[i][0].replace('\n', ' ') 
                       for i in range(len(qa_dataset))}
            lines = [f"{qid}\t{q}" for qid, q in queries.items()]
            with open(query_tsv_path, 'w') as file:
                file.write('\n'.join(lines))
        else:
            print(f'Loaded existing queries from {query_tsv_path}')

    def _check_doc_csv(self, skb: Any, doc_tsv_path: str, add_rel: bool) -> None:
        """
        Check if the document TSV file exists; if not, create it from the knowledge base.

        Args:
            skb (Any): The knowledge base containing candidate documents.
            doc_tsv_path (str): Path to the document TSV file.
            add_rel (bool): Whether to add relational information to the documents.
        """
        indices = skb.candidate_ids
        self.docid2pid = {idx: i for i, idx in enumerate(indices)}
        self.pid2docid = {i: idx for i, idx in enumerate(indices)}

        regenerate = False
        if osp.exists(doc_tsv_path):
            # Verify existing doc.tsv matches current candidate count
            try:
                with open(doc_tsv_path, 'r') as f:
                    line_count = sum(1 for _ in f)
                if line_count != len(indices):
                    print(f"⚠️ Warning: Existing {doc_tsv_path} has {line_count} lines, but SKB has {len(indices)} candidates. Regenerating...")
                    regenerate = True
                else:
                    print(f"✅ Existing {doc_tsv_path} matches candidate count ({line_count}).")
            except Exception as e:
                print(f"⚠️ Error checking {doc_tsv_path}: {e}. Regenerating...")
                regenerate = True

        if not osp.exists(doc_tsv_path) or regenerate:
            corpus = {self.docid2pid[idx]: skb.get_doc_info(idx, add_rel=add_rel, compact=True)
                      for idx in tqdm(indices, desc="Gathering documents")}
            
            lines = [f"{idx}\t{doc}" for idx, doc in corpus.items()]
            with open(doc_tsv_path, 'w') as file:
                file.write('\n'.join(lines))
        else:
            print(f'Loaded existing documents from {doc_tsv_path}')
    
    def _download(self) -> None:
        """
        Download the ColBERTv2 model if not already available.
        """
        if not osp.exists(osp.join(self.download_dir, 'colbertv2.0')):
            # Download the ColBERTv2 checkpoint
            download_command = f"wget {self.url} -P {self.download_dir}"
            subprocess.run(download_command, shell=True, check=True)

            # Extract the downloaded tar.gz file
            tar_command = f"tar -xvzf {osp.join(self.download_dir, 'colbertv2.0.tar.gz')} -C {self.download_dir}"
            subprocess.run(tar_command, shell=True, check=True)

    def _prepare_indexer(self) -> None:
        """
        Prepare the BM25 indexer for the document corpus.
        """
        import time
        from datetime import datetime

        def log_progress(step, message, start_time=None):
            timestamp = datetime.now().strftime("%H:%M:%S")
            if start_time:
                elapsed = time.time() - start_time
                print(f"[{timestamp}] 🔄 Step {step}: {message} (+{elapsed:.1f}s)")
            else:
                print(f"[{timestamp}] 🔄 Step {step}: {message}")

        print("\n🏗️  Starting ColBERTv2 Index Building Process")
        print("=" * 60)
        overall_start = time.time()

        log_progress("1/6", "Initializing GPU configuration", overall_start)
        nranks = torch.cuda.device_count()
        log_progress("1/6", f"Detected {nranks} GPU(s) available", overall_start)

        log_progress("2/6", "Setting up ColBERT configuration", overall_start)
        with Run().context(RunConfig(nranks=nranks, experiment=self.exp_name, root=self.experiments_dir)):
            config = ColBERTConfig(nbits=self.nbits, root=self.experiments_dir, resume=True)
            log_progress("2/6", f"Config created: nbits={self.nbits}, experiment={self.exp_name}, resume=True", overall_start)

            log_progress("3/6", "Loading ColBERT model checkpoint", overall_start)
            model_start = time.time()
            indexer = Indexer(checkpoint=self.model_ckpt_dir, config=config)
            log_progress("3/6", "Model loaded successfully", model_start)

            log_progress("4/6", "Checking existing index", overall_start)
            index_name = f"{self.exp_name}.nbits={self.nbits}"
            index_path = f"{self.experiments_dir}/{self.exp_name}/indexes/{index_name}"

            # Check for key index files to ensure index is complete
            key_files = ['centroids.pt', 'ivf.pid.pt', 'metadata.json', 'plan.json']
            index_complete = os.path.exists(index_path) and all(os.path.exists(os.path.join(index_path, f)) for f in key_files)

            if index_complete:
                log_progress("4/6", f"Found complete existing index at {index_path}, will reuse", overall_start)
                log_progress("5/6", "Skipping index building - using cached index", overall_start)
                # Exit early - don't build index
                overall_end = time.time()
                total_duration = overall_end - overall_start
                print("=" * 60)
                print(f"✓ INDEX BUILDING COMPLETED (reused existing)!")
                print(f"⏱️  Total time: {total_duration:.1f} seconds")
                print("=" * 60)
                return
            else:
                log_progress("4/6", f"No complete existing index found, will create new one", overall_start)

            log_progress("5/6", "Starting document indexing process", overall_start)

            # Count documents in TSV file for progress tracking
            doc_count = 0
            try:
                with open(self.doc_tsv_path, 'r', encoding='utf-8') as f:
                    doc_count = sum(1 for _ in f) - 1  # Subtract header line
                print(f"📚 Processing {doc_count:,} documents from: {self.doc_tsv_path}")
            except:
                print(f"📚 Processing documents from: {self.doc_tsv_path} (count unavailable)")

            print(f"🎯 Target index: {index_name}")
            print(f"🗜️  Compression: {self.nbits}-bit quantization")

            # Show processing phases that will be monitored
            print(f"📊 Processing phases:")
            print(f"   1. Document encoding (batch processing)")
            print(f"   2. K-means clustering (iterative optimization)")
            print(f"   3. Vector quantization (compression)")
            print(f"   4. Index construction (final assembly)")
            print(f"⏳ This may take 10-30 minutes depending on document count...")

            # Start timing and provide phase updates
            index_start = time.time()
            phase_start = time.time()

            print(f"\n🔄 Phase 1/4: Starting document encoding...")
            try:
                # Build the index (force overwrite since we already checked for completeness)
                indexer.index(name=index_name, collection=self.doc_tsv_path, overwrite=True)

                # Phase completions (estimated based on typical ColBERT processing)
                phase_times = {
                    1: "Document encoding",
                    2: "K-means clustering",
                    3: "Vector quantization",
                    4: "Index construction"
                }

                for phase in range(1, 5):
                    if phase < 4:  # Not the last phase
                        estimated_time = (time.time() - phase_start) * (4-phase) / phase
                        print(f"🔄 Phase {phase}/4: {phase_times[phase]} completed (estimated {estimated_time:.1f}s remaining)")

                index_time = time.time() - index_start
                log_progress("5/6", f"Document indexing completed in {index_time:.1f} seconds", index_start)
            except Exception as e:
                log_progress("5/6", f"Indexing failed: {e}", index_start)
                raise

            log_progress("6/6", "Index building process completed", overall_start)
            total_time = time.time() - overall_start
            print(f"⏱️  Total indexing time: {total_time:.1f} seconds")
            print("📂 Index files saved to:", index_path)

            # Show index statistics if available
            if os.path.exists(index_path):
                files = os.listdir(index_path)
                print(f"📁 Index contains {len(files)} files:")
                for file in sorted(files)[:5]:  # Show first 5 files
                    size = os.path.getsize(os.path.join(index_path, file)) / (1024*1024)  # MB
                    print(f"   • {file} ({size:.1f} MB)")
                if len(files) > 5:
                    print(f"   ... and {len(files)-5} more files")

        print("✅ ColBERTv2 index building completed successfully!")
        print("=" * 60)

    def _build_index_if_missing(self) -> None:
        """
        Build the ColBERT index if it doesn't exist.
        This is a simplified version of _prepare_indexer for on-demand building.
        """
        index_name = f"{self.exp_name}.nbits={self.nbits}"
        index_path = osp.join(self.experiments_dir, self.exp_name, "indexes", index_name)

        # Check for key index files to ensure index is complete and valid
        key_files = ['centroids.pt', 'ivf.pid.pt', 'metadata.json', 'plan.json']
        index_complete = (os.path.exists(index_path) and
                        all(os.path.exists(os.path.join(index_path, f)) for f in key_files))

        if index_complete:
            print(f"✅ Found complete existing index at {index_path}")
            print("🎉 Skipping index building - using cached index!")
            return

        print("🏗️  Building missing ColBERT index...")
        print(f"⚠️  Index {index_name} not found or incomplete at {index_path}")

        with Run().context(RunConfig(nranks=1, experiment=self.exp_name, root=self.experiments_dir)):  # Force single GPU
            config = ColBERTConfig(nbits=self.nbits, root=self.experiments_dir)

            # Use the model checkpoint directory directly
            checkpoint = self.model_ckpt_dir
            if not osp.exists(checkpoint):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

            indexer = Indexer(checkpoint=checkpoint, config=config)

            # Build the index - don't overwrite if it exists
            indexer.index(name=index_name, collection=self.doc_tsv_path, overwrite=False)

        print("✅ Index building completed")

    def run_all(self, ranking_suffix: str = "") -> Dict[int, Dict[int, float]]:
        """
        Run the retrieval for all strategies and store the rankings.

        Args:
            ranking_suffix (str): Optional suffix to append to ranking filenames to avoid conflicts.

        Returns:
            Dict[int, Dict[int, float]]: A dictionary mapping query IDs to a dictionary of candidate scores.
        """
        def find_file_path_by_name(name: str, path: str) -> Optional[str]:
            """
            Find the file path by its name in a given directory.

            Args:
                name (str): The name of the file to find.
                path (str): The directory to search.

            Returns:
                Optional[str]: The file path if found, None otherwise.
            """
            for root, dirs, files in os.walk(path):
                if name in files:
                    return osp.join(root, name)
            return None

        print(f"🔍 Starting ColBERTv2 retrieval evaluation for {len(self.strategies)} strategies: {', '.join(self.strategies)}")
        print("=" * 80)

        total_start_time = datetime.now()
        all_score_dicts = {}

        for strategy in self.strategies:
            print(f"\n{'='*60}")
            print(f"🔍 EVALUATING STRATEGY: {strategy.upper()} (FORCED, NO CACHING)")
            print("=" * 60)

            strategy_start_time = datetime.now()

            # Set up strategy-specific parameters
            if strategy == 'original':
                strategy_dataset_root = None
                strategy_split = "human_generated_eval"
                query_tsv_path = self.query_tsv_path
            elif strategy == 'error_aware':
                strategy_dataset_root = create_strategy_dataset(strategy)
                strategy_split = "variants"
                # Create strategy-specific query file using the variants file
                variants_file = f"/home/wlia0047/ar57/wenyu/stark/data/strategy_variants/{strategy}_variants_81.csv"
                if os.path.exists(variants_file):
                    df = pd.read_csv(variants_file)
                    query_tsv_path = f"/home/wlia0047/ar57/wenyu/stark/Colbertv2eval/query_{strategy}_variants.tsv"
                    with open(query_tsv_path, 'w') as f:
                        for i, row in enumerate(df.itertuples()):
                            # All variant files now use the new format with 'query' column
                            f.write(f"{i}\t{row.query}\n")
                else:
                    print(f"Warning: Variants file not found for strategy {strategy}, skipping...")
                    continue
            elif strategy == 'kg_query':
                # Custom handling for kg_query
                strategy_split = "variants"
                # For kg_query, we need to ensure the query TSV is created from our source CSV
                # Source: /home/wlia0047/ar57/wenyu/result/generated_kg_queries.csv
                source_csv = "/home/wlia0047/ar57/wenyu/result/generated_kg_queries.csv"
                if os.path.exists(source_csv):
                    df = pd.read_csv(source_csv)
                    query_tsv_path = f"/home/wlia0047/ar57/wenyu/stark/Colbertv2eval/query_{strategy}_variants.tsv"
                    with open(query_tsv_path, 'w') as f:
                        for i, row in enumerate(df.itertuples()):
                            # Handle different column names if necessary
                            query_text = row.query
                            f.write(f"{i}\t{query_text}\n")
                else:
                    print(f"Warning: Source CSV not found for strategy {strategy}: {source_csv}, skipping...")
                    continue
            else:
                strategy_dataset_root = create_strategy_dataset(strategy)
                strategy_split = "variants"
                # Create strategy-specific query file
                variants_file = f"/home/wlia0047/ar57/wenyu/stark/data/strategy_variants/{strategy}_variants_81.csv"
                if os.path.exists(variants_file):
                    df = pd.read_csv(variants_file)
                    query_tsv_path = f"/home/wlia0047/ar57/wenyu/stark/Colbertv2eval/query_{strategy}_variants.tsv"
                    with open(query_tsv_path, 'w') as f:
                        for i, row in enumerate(df.itertuples()):
                            # All variant files now use the new format with 'query' column
                            f.write(f"{i}\t{row.query}\n")
                else:
                    print(f"Warning: Variants file not found for strategy {strategy}, skipping...")
                    continue

            # Always run evaluation for this strategy - USE SHARED INDEX
            nranks = torch.cuda.device_count()
            # Use the SAME index for all strategies (amazon_hg)
            exp_name = self.exp_name

            # Build index if missing (this method now checks for existing index)
            self._build_index_if_missing()

            # Perform retrieval with existing index
            index_name = f"{self.exp_name}.nbits={self.nbits}"
            with Run().context(RunConfig(nranks=nranks, experiment=exp_name, root=self.experiments_dir)):
                config = ColBERTConfig(nbits=self.nbits, root=self.experiments_dir)
                searcher = Searcher(index=index_name, collection=self.doc_tsv_path, config=config)
                queries = Queries(query_tsv_path)
                ranking = searcher.search_all(queries, k=self.k)
                # Save with strategy-specific name to avoid conflicts
                ranking_filename = f'ranking_{strategy}{ranking_suffix}.tsv'

                # Delete any existing ranking files with the same name to allow overwrite
                # ColBERT saves files in timestamped directories under experiments_dir/exp_name/eval/
                eval_dir = osp.join(self.experiments_dir, exp_name, 'eval')
                if osp.exists(eval_dir):
                    for root, dirs, files in os.walk(eval_dir):
                        for file in files:
                            if file == ranking_filename:
                                filepath = osp.join(root, file)
                                os.remove(filepath)
                                print(f"🗑️  Removed existing ranking file: {filepath}")

                ranking.save(ranking_filename)

            # Find the ranking file in the experiments directory
            ranking_path = find_file_path_by_name(ranking_filename, self.experiments_dir)

            if ranking_path:
                print(f"✅ Strategy {strategy} evaluation completed! Ranking saved to: {ranking_path}")
            else:
                print(f"❌ Strategy {strategy} evaluation failed - ranking file not found")

            strategy_end_time = datetime.now()
            strategy_duration = strategy_end_time - strategy_start_time
            print(f"⏱️  Strategy {strategy} time: {strategy_duration}")

        total_end_time = datetime.now()
        total_duration = total_end_time - total_start_time
        print("=" * 80)
        print(f"✓ ALL STRATEGIES ColBERTv2 EVALUATION COMPLETED!")
        print(f"⏱️  Total time: {total_duration}")
        print("=" * 80)

        # Load and return the score dictionary for the current strategy
        score_dict = defaultdict(dict)

        # Try multiple possible locations for the ranking file
        possible_paths = [
            osp.join(self.experiments_dir, self.exp_name, 'eval'),  # Colbertv2eval/amazon_hg/eval
            './experiments',  # fallback
            osp.join(self.experiments_dir),  # fallback
        ]

        # Determine which strategy ranking file to load
        # If we have a current strategy set, use it; otherwise fall back to original
        current_strategy = getattr(self, 'strategy', 'original')
        ranking_filename = f'ranking_{current_strategy}.tsv'

        ranking_path = None
        for base_path in possible_paths:
            # Look for the most recent ranking file for this specific strategy
            # Search subdirectories for the specific ranking file
            candidate_paths = []
            if osp.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    if ranking_filename in files:
                        candidate_paths.append(osp.join(root, ranking_filename))

            # Sort by modification time (newest first) and pick the most recent
            if candidate_paths:
                candidate_paths.sort(key=lambda x: osp.getmtime(x), reverse=True)
                ranking_path = candidate_paths[0]
                break

        if ranking_path:
            print(f"✅ Found ranking file for strategy '{current_strategy}': {ranking_path}")
            with open(ranking_path) as f:
                for line in f:
                    qid, pid, rank, *score = line.strip().split('\t')
                    qid, pid, rank = int(qid), int(pid), int(rank)
                    if len(score) > 0:
                        assert len(score) == 1
                        score = float(score[0])
                        score_dict[qid][pid] = score
                    else:
                        score_dict[qid][pid] = -999
        else:
            print(f"❌ Warning: Could not find ranking file '{ranking_filename}' for strategy '{current_strategy}' in any expected location")
            print("Expected locations checked:")
            for path in possible_paths:
                print(f"  - {path}")
            print("Returning empty score dict - this will cause evaluation failures!")

        return score_dict

        score_dict = defaultdict(dict)
        with open(self.ranking_path) as f:
            for line in f:
                qid, pid, rank, *score = line.strip().split('\t')
                qid, pid, rank = int(qid), int(pid), int(rank)
                if len(score) > 0:
                    assert len(score) == 1
                    score = float(score[0])
                    score_dict[qid][pid] = score
                else:
                    score_dict[qid][pid] = -999

        return score_dict

    def forward(self,
                query: Union[str, None],
                query_id: int,
                **kwargs: Any) -> Dict[int, float]:
        """
        Forward pass to retrieve rankings for the given query.

        Args:
            query (str): The query string.
            query_id (int): The query index.

        Returns:
            Dict[int, float]: A dictionary of candidate IDs and their corresponding similarity scores.
        """
        # Check if we need to update score_dict for different strategies
        if hasattr(self, '_current_strategy') and self._current_strategy != getattr(self, '_last_strategy', None):
            print(f"🔄 Updating score_dict for strategy: {self._current_strategy}")
            self._update_score_dict_for_strategy(self._current_strategy)
            self._last_strategy = self._current_strategy

        score_dict = self.score_dict[query_id]

        # Create a dictionary with scores for all candidate documents
        # Retrieved documents get their actual scores, others get a very low score
        all_scores = {}
        min_score = min(score_dict.values()) if score_dict else -float('inf')


        # Set very low scores for all candidates first
        for doc_id in self.candidate_ids:
            all_scores[doc_id] = min_score - 1.0  # Even lower than the lowest retrieved score

        # Update scores for retrieved documents
        # Skip PIDs that are not in pid2docid (e.g., attribute nodes)
        for pid, score in score_dict.items():
            if pid in self.pid2docid:
                doc_id = self.pid2docid[pid]
                all_scores[doc_id] = score

        return all_scores

    def _update_score_dict_for_strategy(self, strategy: str) -> None:
        """
        Update the score_dict for a specific strategy by re-running the retrieval.

        Args:
            strategy (str): The strategy to update score_dict for.
        """
        print(f"🔄 Re-running retrieval for strategy: {strategy}")

        # Temporarily change the strategy and strategies
        old_strategy = self.strategy
        old_strategies = self.strategies
        self.strategy = strategy
        self.strategies = [strategy]

        try:
            # Re-run the retrieval process for this single strategy with a unique filename
            # to avoid conflicts with existing ranking files
            import time
            timestamp = int(time.time() * 1000)  # millisecond precision
            self.score_dict = self.run_all(ranking_suffix=f"_{timestamp}")
        finally:
            # Restore original strategy and strategies
            self.strategy = old_strategy
            self.strategies = old_strategies
