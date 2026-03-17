#!/usr/bin/env python3
import torch

model_candidates = [
    "prithivida/BERT_based_uncased_English_Grammar_Corrector",
    "grammarly/coedit",
]

print("=" * 60)
print("Testing HuggingFace Grammar Correction Models")
print("=" * 60)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print()

for model_id in model_candidates:
    print(f"Testing: {model_id}")
    print("-" * 60)
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        print(f"  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"  ✓ Tokenizer loaded")
        
        print(f"  Loading model...")
        model = AutoModelForTokenClassification.from_pretrained(model_id, trust_remote_code=True)
        print(f"  ✓ Model loaded")
        print(f"    - Num labels: {model.config.num_labels}")
        print(f"    - Hidden size: {model.config.hidden_size}")
        
        if hasattr(model.config, 'id2label'):
            id2label = model.config.id2label
            print(f"    - Sample labels: {dict(list(id2label.items())[:5])}")
        
        print(f"  ✓ Model ready for inference")
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:200]}")
    
    print()

print("=" * 60)
print("Recommendation: Use 'prithivida/BERT_based_uncased_English_Grammar_Corrector'")
print("=" * 60)
