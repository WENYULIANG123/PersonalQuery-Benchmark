#!/usr/bin/env python3
"""Download GECToR pretrained model from S3"""
import os
import urllib.request
import sys
from pathlib import Path

# Model URLs from official GECToR README
MODELS = {
    "roberta": "https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th",
    "bert": "https://grammarly-nlp-data-public.s3.amazonaws.com/gector/bert_0_gectorv2.th",
    "xlnet": "https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gectorv2.th",
}

def download_model(model_name="roberta", output_dir="/fs04/ar57/wenyu/models/gector"):
    """Download GECToR pretrained model"""
    if model_name not in MODELS:
        print(f"Available models: {list(MODELS.keys())}")
        sys.exit(1)
    
    url = MODELS[model_name]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filename = f"{model_name}_gectorv2.th"
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        print(f"✓ Model already exists: {filepath}")
        return filepath
    
    print(f"Downloading {model_name} model from S3...")
    print(f"URL: {url}")
    print(f"Output: {filepath}")
    
    try:
        urllib.request.urlretrieve(url, filepath)
        file_size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"✓ Downloaded successfully ({file_size_mb:.1f}MB)")
        return filepath
    except Exception as e:
        print(f"✗ Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        sys.exit(1)

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "roberta"
    download_model(model_name)
