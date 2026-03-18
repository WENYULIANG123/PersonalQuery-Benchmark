#!/usr/bin/env python3
"""Test device alignment fix for retrievers"""
import sys
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval')

import torch
import numpy as np
from utils.retrievers import DenseRetriever, ANCERetriever, BGERetriever

print("=" * 60)
print("Testing Device Alignment Fixes")
print("=" * 60)

# Test 1: DenseRetriever device setup
print("\n[Test 1] DenseRetriever device setup...")
try:
    dense = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"  ✓ Device: {dense.device}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    print("  ✓ DenseRetriever initialized successfully")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: ANCE device setup
print("\n[Test 2] ANCERetriever device setup...")
try:
    ance = ANCERetriever(model_name="intfloat/e5-base-v2")
    print(f"  ✓ Device: {ance.device}")
    print("  ✓ ANCERetriever initialized successfully")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: BGE device setup
print("\n[Test 3] BGERetriever device setup...")
try:
    bge = BGERetriever(model_name="BAAI/bge-large-en-v1.5")
    print(f"  ✓ Device: {bge.device}")
    print("  ✓ BGERetriever initialized successfully")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All device setup tests passed!")
print("=" * 60)
