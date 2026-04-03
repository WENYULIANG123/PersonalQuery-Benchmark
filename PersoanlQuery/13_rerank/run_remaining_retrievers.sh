#!/usr/bin/env bash
set -euo pipefail

META_FILE="/fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz"
USER_ID="A2U6VP21H9UVV3"

python3 13_batch_llm_rerank_all.py --config 15_config_bm25.json --meta-file "$META_FILE" --user-id "$USER_ID"
python3 13_batch_llm_rerank_all.py --config 15_config_tfidf.json --meta-file "$META_FILE" --user-id "$USER_ID"
python3 13_batch_llm_rerank_all.py --config 15_config_e5.json --meta-file "$META_FILE" --user-id "$USER_ID"
python3 13_batch_llm_rerank_all.py --config 15_config_bge.json --meta-file "$META_FILE" --user-id "$USER_ID"
python3 13_batch_llm_rerank_all.py --config 15_config_minilm.json --meta-file "$META_FILE" --user-id "$USER_ID"
python3 13_batch_llm_rerank_all.py --config 15_config_mpnet.json --meta-file "$META_FILE" --user-id "$USER_ID"
python3 13_batch_llm_rerank_all.py --config 15_config_star.json --meta-file "$META_FILE" --user-id "$USER_ID"
