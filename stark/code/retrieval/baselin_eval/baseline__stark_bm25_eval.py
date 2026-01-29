
#!/usr/bin/env python3
"""
Baseline STaRK BM25 Evaluation Script
=====================================
Uses:
- Queries: /home/wlia0047/ar57/wenyu/stark/data/stark_qa_human_generated_eval.csv
- SKB: /home/wlia0047/ar57/wenyu/stark/data/amazon/processed
"""

import os
import sys
import subprocess
from datetime import datetime
import select

def main():
    # STaRK project root
    stark_root = "/home/wlia0047/ar57/wenyu/stark"
    os.chdir(stark_root)
    sys.path.insert(0, stark_root)

    dataset = "amazon"
    model = "BM25"
    split = "human_generated_eval"
    
    # Paths specified by user
    # Note: eval.py typically expects SKB at specific locations or via dataset_root
    # We will point dataset_root to the parent of 'qa' folder if possible, 
    # BUT here we need to point to where the 'processed' data is for SKB loading.
    # SKB loading usually looks for `dataset_root/processed/{dataset}/...` or similiar depending on `load_skb`.
    # Let's check where `load_skb` looks. It usually looks at `root` argument.
    
    # User specified SKB path:
    skb_root = "/home/wlia0047/ar57/wenyu/stark/data/amazon/processed"
    # Wait, load_skb usually expects the root containing the 'processed' folder or the folder itself?
    # Let's try passing the folder directly as dataset_root based on previous runs.
    
    # Query file:
    query_file = "/home/wlia0047/ar57/wenyu/stark/data/stark_qa_synthesized_100.csv"
    
    output_dir = "BM25eval_baseline_synthesized"
    
    print(f"STARTING BASELINE EVALUATION (SYNTHESIZED 100)")
    print(f"Queries: {query_file}")
    print(f"SKB Root: {skb_root}")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['OMP_NUM_THREADS'] = '1' 

    # Set TQDM_MININTERVAL to reduce log spam
    env['TQDM_MININTERVAL'] = '30'  # Update only every 30 seconds
    
    # Construct command
    # We use --csv_file to load the specific query file
    # We use --dataset_root to point to the SKB location
    cmd = ["python", "-u", "eval.py",
           "--dataset", dataset,
           "--model", model,
           "--split", "variants", # Use 'variants' mode to allow custom CSV loading
           "--strategy", "baseline_human", # Just a label
           "--csv_file", query_file,
           "--dataset_root", skb_root,
           "--output_dir", output_dir,
           "--save_pred",
           "--batch_size", "1",
           "--device", "cpu",
           "--force_rerun"]

    print(f"Executing: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
        universal_newlines=True,
        env=env
    )

    # Stream output
    while True:
        ready, _, _ = select.select([process.stdout], [], [], 0.1)
        if ready:
            char = process.stdout.read(1)
            if char:
                sys.stdout.write(char)
                sys.stdout.flush()
            else:
                if process.poll() is not None:
                    break
        else:
            if process.poll() is not None:
                remaining = process.stdout.read()
                if remaining:
                    sys.stdout.write(remaining)
                    sys.stdout.flush()
                break

    return_code = process.poll()
    if return_code != 0:
        print(f"Evaluation failed with return code {return_code}")
        sys.exit(return_code)
    else:
        print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()
