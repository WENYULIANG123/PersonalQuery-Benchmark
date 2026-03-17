#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from typing import List

from gector_detector_v2 import GectorDetectorHF


def load_json_comments(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return [
            item.get("text", item) if isinstance(item, dict) else str(item)
            for item in data
        ]
    elif isinstance(data, dict):
        if "comments" in data:
            return [item if isinstance(item, str) else item.get("text", str(item)) 
                    for item in data["comments"]]
        else:
            return [data.get("text", str(data))]
    else:
        return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_detect_hf.py <input.json> [output.json] [model_id]")
        print("  input.json: JSON file with comments (list or dict with 'text'/'comments')")
        print("  output.json: output file for results (default: gector_results.json)")
        print("  model_id: HuggingFace model ID (default: prithivida/BERT_based_uncased_English_Grammar_Corrector)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "gector_results.json"
    model_id = sys.argv[3] if len(sys.argv) > 3 else "sahilnishad/BERT-GED-FCE-FT"
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    print(f"[1/3] Loading comments from {input_file}...")
    texts = load_json_comments(input_file)
    print(f"  Loaded {len(texts)} comments")
    
    print(f"[2/3] Initializing GECToR detector on GPU...")
    print(f"  Model: {model_id}")
    detector = GectorDetectorHF(
        model_id=model_id,
        min_error_probability=0.1
    )
    
    print(f"[3/3] Processing {len(texts)} comments...")
    results = detector.detect_errors(texts, batch_size=32)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    error_count = sum(len(r.get("errors", [])) for r in results)
    print(f"✓ Total errors detected: {error_count} across {len(texts)} comments")
    
    if error_count > 0:
        print(f"\nFirst error example:")
        for r in results:
            if r.get("errors"):
                print(f"  Text: {r['text']}")
                print(f"  Error: {r['errors'][0]}")
                break


if __name__ == "__main__":
    main()
