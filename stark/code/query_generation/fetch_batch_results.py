import json
import csv
import os
import sys
from openai import OpenAI
import requests

# Config
SILICONFLOW_API_KEY = 'sk-drezmfyckjkmxixpiblvbwdhypjbrsoyvmeertajtupiqnnj'
SILICONFLOW_BASE_URL = 'https://api.siliconflow.cn/v1'
BATCH_ID = 'batch_lfgkivrqut' # The completed one
OUTPUT_CSV = '/home/wlia0047/ar57/wenyu/result/generated_kg_queries.csv'
PATH_SOURCE = '/home/wlia0047/ar57/wenyu/result/generated_kg_queries.json' # This contains path info if saved

def fetch_results(batch_id):
    client = OpenAI(base_url=SILICONFLOW_BASE_URL, api_key=SILICONFLOW_API_KEY)
    batch = client.batches.retrieve(batch_id)
    if batch.status != 'completed':
        print(f"Batch {batch_id} is {batch.status}")
        return None
    
    file_id = batch.output_file_id
    url = f"{SILICONFLOW_BASE_URL}/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}"}
    
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    
    results = []
    for line in resp.text.strip().split('\n'):
        if line:
            results.append(json.loads(line))
    return results

def process_and_save():
    results = fetch_results(BATCH_ID)
    if not results: return
    
    # We need the path info to get anchor products / SKB IDs.
    # But wait, Job ID 50433336 restarted, so the JSON file might be empty or wrong.
    # Let me check if there's any backup of path_map.
    # Actually, the batch results have NO context of what the prompt was about easily.
    
    # This is a problem with the batch approach if the job restarts.
    # I should have saved the path_map to a file BEFORE submitting the batch.
    
    # Let's try to find an earlier log that has the paths? No.
    
    print(f"Fetched {len(results)} results.")
    for res in results[:3]:
        content = res['response']['body']['choices'][0]['message']['content']
        print(f"Query: {content}")

if __name__ == "__main__":
    process_and_save()
