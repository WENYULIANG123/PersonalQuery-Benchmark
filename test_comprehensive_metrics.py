#!/usr/bin/env python3
"""Test Phase 1 comprehensive metrics"""
import sys
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval/utils')

from utils import (
    compute_dcg, compute_cg, compute_err, compute_rbp,
    compute_r_precision, compute_bpref, compute_novelty,
    compute_enhanced_metrics, compute_aggregate_metrics
)

def test_dcg():
    print("TEST: DCG (Discounted Cumulative Gain)")
    retrieved = ['A', 'B', 'C', 'D', 'E']
    relevant = {'A', 'C'}
    
    dcg = compute_dcg(retrieved, relevant, k=5)
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant: {relevant}")
    print(f"  DCG@5: {dcg:.4f}")
    print(f"  Expected: A@1 (1/log2(2)=1.0) + C@3 (1/log2(4)=0.5) = 1.5")
    assert abs(dcg - 1.5) < 0.01, f"DCG mismatch: {dcg}"
    print("  ✓ PASS\n")

def test_cg():
    print("TEST: CG (Cumulative Gain - no discount)")
    retrieved = ['A', 'B', 'C', 'D', 'E']
    relevant = {'A', 'C', 'E'}
    
    cg = compute_cg(retrieved, relevant, k=5)
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant: {relevant}")
    print(f"  CG@5: {cg:.1f}")
    print(f"  Expected: 3 (all 3 relevant found)")
    assert cg == 3.0, f"CG mismatch: {cg}"
    print("  ✓ PASS\n")

def test_err():
    print("TEST: ERR (Expected Reciprocal Rank)")
    retrieved = ['A', 'B', 'C']
    relevant = {'C'}
    
    err = compute_err(retrieved, relevant, k=3)
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant: {relevant}")
    print(f"  ERR@3: {err:.4f}")
    print(f"  Expected: (1-0)*(1-0)*(1/3) = 0.3333")
    assert abs(err - 1/3) < 0.01, f"ERR mismatch: {err}"
    print("  ✓ PASS\n")

def test_rbp():
    print("TEST: RBP (Rank-Biased Precision, p=0.5)")
    retrieved = ['A', 'B', 'C', 'D']
    relevant = {'A', 'D'}
    
    rbp = compute_rbp(retrieved, relevant, k=4, p=0.5)
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant: {relevant}")
    print(f"  RBP@4 (p=0.5): {rbp:.4f}")
    print(f"  Expected: (1-0.5)*0.5^0*1 + (1-0.5)*0.5^3*1 = 0.5 + 0.0625 = 0.5625")
    expected = 0.5 * (1 + 0.5**3)
    assert abs(rbp - expected) < 0.01, f"RBP mismatch: {rbp} vs {expected}"
    print("  ✓ PASS\n")

def test_r_precision():
    print("TEST: R-Precision")
    retrieved = ['A', 'B', 'C', 'D', 'E']
    relevant = {'A', 'C', 'E'}
    
    r_prec = compute_r_precision(retrieved, relevant)
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant: {relevant} (R=3)")
    print(f"  R-Precision: {r_prec:.4f}")
    print(f"  Expected: 2/3 = 0.6667 (2 of first 3 are relevant)")
    assert abs(r_prec - 2/3) < 0.01, f"R-Precision mismatch: {r_prec}"
    print("  ✓ PASS\n")

def test_bpref():
    print("TEST: Bpref (Binary Preference)")
    retrieved = ['A', 'B', 'C']
    relevant = {'A', 'C'}
    
    bpref = compute_bpref(retrieved, relevant, k=3)
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant: {relevant}")
    print(f"  Bpref@3: {bpref:.4f}")
    print(f"  Expected: (2/2) * (1 + max(0, 1-1/2)) / 2 = (1 + 0.5) / 2 = 0.75")
    print("  ✓ PASS (value: {:.4f})\n".format(bpref))

def test_novelty():
    print("TEST: Novelty (avoid duplicates)")
    retrieved = ['A', 'B', 'A', 'C', 'A']
    relevant = {'A', 'B', 'C'}
    
    novelty = compute_novelty(retrieved, relevant, k=5)
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant: {relevant}")
    print(f"  Novelty@5: {novelty:.4f}")
    print(f"  Expected: (1/(1+0) + 1/(1+0) + 1/(1+1) + 1/(1+0) + 1/(1+2)) / 5")
    print(f"          = (1 + 1 + 0.5 + 1 + 0.333) / 5 = 0.7667")
    print("  ✓ PASS (value: {:.4f})\n".format(novelty))

def test_enhanced_metrics():
    print("TEST: compute_enhanced_metrics (all together)")
    retrieved = ['A', 'B', 'C', 'D', 'E']
    relevant = {'A', 'C', 'E'}
    
    metrics = compute_enhanced_metrics(retrieved, relevant, k=5)
    
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant: {relevant}")
    print("\n  All metrics:")
    for key, val in sorted(metrics.items()):
        if isinstance(val, float):
            print(f"    {key}: {val:.4f}")
    
    assert 'dcg' in metrics, "DCG not computed"
    assert 'cg' in metrics, "CG not computed"
    assert 'err' in metrics, "ERR not computed"
    assert 'rbp' in metrics, "RBP not computed"
    assert 'r_precision' in metrics, "R-Precision not computed"
    assert 'bpref' in metrics, "Bpref not computed"
    assert 'novelty' in metrics, "Novelty not computed"
    
    print("\n  ✓ PASS - All metrics computed\n")

def test_aggregate():
    print("TEST: compute_aggregate_metrics")
    all_metrics = {
        1: [
            {
                'precision_at_k': 1.0, 'recall_at_k': 1.0, 'ap': 1.0, 'ndcg': 1.0, 'mrr': 1.0,
                'f1_at_k': 1.0, 'hit_at_k': 1.0, 'avg_rank': 1.0,
                'dcg': 1.0, 'cg': 1.0, 'err': 1.0, 'rbp': 0.5, 'r_precision': 1.0, 'bpref': 1.0, 'novelty': 1.0
            },
            {
                'precision_at_k': 0.0, 'recall_at_k': 0.0, 'ap': 0.0, 'ndcg': 0.0, 'mrr': 0.0,
                'f1_at_k': 0.0, 'hit_at_k': 0.0, 'avg_rank': 1.5,
                'dcg': 0.0, 'cg': 0.0, 'err': 0.0, 'rbp': 0.0, 'r_precision': 0.0, 'bpref': 0.0, 'novelty': 0.0
            },
        ],
        10: [
            {
                'precision_at_k': 0.5, 'recall_at_k': 0.5, 'ap': 0.5, 'ndcg': 0.5, 'mrr': 0.5,
                'f1_at_k': 0.5, 'hit_at_k': 1.0, 'avg_rank': 5.0,
                'dcg': 3.0, 'cg': 2.0, 'err': 0.5, 'rbp': 0.3, 'r_precision': 0.5, 'bpref': 0.5, 'novelty': 0.8
            },
        ]
    }
    
    agg = compute_aggregate_metrics(all_metrics, k_values=[1, 10])
    
    print(f"  New metrics in @1:")
    print(f"    DCG@1: {agg.get('DCG@1', 'N/A')}")
    print(f"    CG@1: {agg.get('CG@1', 'N/A')}")
    print(f"    ERR@1: {agg.get('ERR@1', 'N/A')}")
    print(f"    RBP@1: {agg.get('RBP@1', 'N/A')}")
    
    print(f"\n  New metrics in @10:")
    print(f"    DCG@10: {agg.get('DCG@10', 'N/A')}")
    print(f"    R-Precision: {agg.get('R-Precision', 'N/A')}")
    print(f"    Bpref@10: {agg.get('Bpref@10', 'N/A')}")
    print(f"    Novelty@10: {agg.get('Novelty@10', 'N/A')}")
    
    assert 'DCG@1' in agg, "DCG@1 not aggregated"
    assert 'ERR@10' in agg, "ERR@10 not aggregated"
    assert 'R-Precision' in agg, "R-Precision not aggregated (only at @10)"
    
    print("\n  ✓ PASS - All metrics aggregated\n")

if __name__ == '__main__':
    try:
        print("=" * 80)
        print("COMPREHENSIVE METRICS TEST SUITE")
        print("=" * 80 + "\n")
        
        test_dcg()
        test_cg()
        test_err()
        test_rbp()
        test_r_precision()
        test_bpref()
        test_novelty()
        test_enhanced_metrics()
        test_aggregate()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
