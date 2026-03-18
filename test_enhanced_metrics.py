#!/usr/bin/env python3
"""
Test script for enhanced metrics computation
"""
import sys
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval/utils')
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval')

from utils import compute_enhanced_metrics, compute_aggregate_metrics, compute_noise_robustness

def test_compute_enhanced_metrics():
    """Test compute_enhanced_metrics function"""
    print("=" * 80)
    print("TEST 1: compute_enhanced_metrics()")
    print("=" * 80)
    
    # Test case 1: Query with 1 relevant result at position 1
    retrieved = ['ASIN_A', 'ASIN_B', 'ASIN_C', 'ASIN_D', 'ASIN_E']
    relevant = {'ASIN_A'}
    
    metrics = compute_enhanced_metrics(retrieved, relevant, k=5)
    print(f"\nTest Case 1: 1 relevant at position 1")
    print(f"  Retrieved: {retrieved[:5]}")
    print(f"  Relevant: {relevant}")
    print(f"  Metrics@5:")
    for key, val in metrics.items():
        print(f"    {key}: {val:.4f}")
    
    assert metrics['precision_at_k'] == 0.2, f"Precision should be 0.2, got {metrics['precision_at_k']}"
    assert metrics['recall_at_k'] == 1.0, f"Recall should be 1.0, got {metrics['recall_at_k']}"
    assert metrics['hit_at_k'] == 1.0, f"Hit should be 1.0, got {metrics['hit_at_k']}"
    print("  ✓ Test Case 1 passed")
    
    # Test case 2: Query with 2 relevant results
    retrieved = ['ASIN_A', 'ASIN_B', 'ASIN_C', 'ASIN_D', 'ASIN_E']
    relevant = {'ASIN_A', 'ASIN_C'}
    
    metrics = compute_enhanced_metrics(retrieved, relevant, k=5)
    print(f"\nTest Case 2: 2 relevant at positions 1, 3")
    print(f"  Retrieved: {retrieved[:5]}")
    print(f"  Relevant: {relevant}")
    print(f"  Metrics@5:")
    for key, val in metrics.items():
        print(f"    {key}: {val:.4f}")
    
    assert metrics['precision_at_k'] == 0.4, f"Precision should be 0.4, got {metrics['precision_at_k']}"
    assert metrics['recall_at_k'] == 1.0, f"Recall should be 1.0, got {metrics['recall_at_k']}"
    assert metrics['hit_at_k'] == 1.0, f"Hit should be 1.0, got {metrics['hit_at_k']}"
    assert abs(metrics['avg_rank'] - 2.0) < 0.01, f"AvgRank should be 2.0, got {metrics['avg_rank']}"
    print("  ✓ Test Case 2 passed")
    
    # Test case 3: Query with no relevant results
    retrieved = ['ASIN_B', 'ASIN_C', 'ASIN_D', 'ASIN_E', 'ASIN_F']
    relevant = {'ASIN_A'}
    
    metrics = compute_enhanced_metrics(retrieved, relevant, k=5)
    print(f"\nTest Case 3: 0 relevant")
    print(f"  Retrieved: {retrieved[:5]}")
    print(f"  Relevant: {relevant}")
    print(f"  Metrics@5:")
    for key, val in metrics.items():
        print(f"    {key}: {val:.4f}")
    
    assert metrics['precision_at_k'] == 0.0, f"Precision should be 0.0, got {metrics['precision_at_k']}"
    assert metrics['recall_at_k'] == 0.0, f"Recall should be 0.0, got {metrics['recall_at_k']}"
    assert metrics['hit_at_k'] == 0.0, f"Hit should be 0.0, got {metrics['hit_at_k']}"
    print("  ✓ Test Case 3 passed")

def test_compute_aggregate_metrics():
    """Test compute_aggregate_metrics function"""
    print("\n" + "=" * 80)
    print("TEST 2: compute_aggregate_metrics()")
    print("=" * 80)
    
    all_metrics = {
        1: [
            {'precision_at_k': 1.0, 'recall_at_k': 1.0, 'ap': 1.0, 'ndcg': 1.0, 'mrr': 1.0, 'f1_at_k': 1.0, 'hit_at_k': 1.0, 'avg_rank': 1.0},
            {'precision_at_k': 0.0, 'recall_at_k': 0.0, 'ap': 0.0, 'ndcg': 0.0, 'mrr': 0.0, 'f1_at_k': 0.0, 'hit_at_k': 0.0, 'avg_rank': 1.5},
        ],
        3: [
            {'precision_at_k': 0.33, 'recall_at_k': 1.0, 'ap': 0.5, 'ndcg': 0.5, 'mrr': 1.0, 'f1_at_k': 0.5, 'hit_at_k': 1.0, 'avg_rank': 1.67},
        ],
        5: [],
        10: [],
    }
    
    aggregated = compute_aggregate_metrics(all_metrics, k_values=[1, 3, 5, 10])
    
    print(f"\nAggregated Metrics:")
    for key, val in sorted(aggregated.items()):
        if isinstance(val, dict):
            print(f"  {key}:")
            for k2, v2 in val.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {key}: {val}")
    
    # Verify some aggregated values
    assert 'P@1' in aggregated, "Should have P@1"
    assert 'F1@1' in aggregated, "Should have F1@1 (new metric)"
    assert 'Hit@1' in aggregated, "Should have Hit@1 (new metric)"
    assert 'NDCG@1_stats' in aggregated, "Should have NDCG@1_stats (distribution)"
    assert 'Performance_Distribution@1' in aggregated, "Should have Performance_Distribution@1 (classification)"
    
    print(f"\n  Expected P@1=0.5, got {aggregated['P@1']}")
    assert aggregated['P@1'] == 0.5, f"P@1 should be 0.5"
    print(f"  Expected F1@1=0.5, got {aggregated['F1@1']}")
    assert aggregated['F1@1'] == 0.5, f"F1@1 should be 0.5"
    print(f"  Expected Hit@1=0.5, got {aggregated['Hit@1']}")
    assert aggregated['Hit@1'] == 0.5, f"Hit@1 should be 0.5"
    
    print("  ✓ Test Case passed")

def test_compute_noise_robustness():
    """Test compute_noise_robustness function"""
    print("\n" + "=" * 80)
    print("TEST 3: compute_noise_robustness()")
    print("=" * 80)
    
    clean_metrics = {'NDCG@10': 0.5}
    noisy_metrics = {'NDCG@10': 0.45}
    
    robustness = compute_noise_robustness(clean_metrics, noisy_metrics, key='NDCG@10')
    
    print(f"\nClean NDCG@10: {clean_metrics['NDCG@10']}")
    print(f"Noisy NDCG@10: {noisy_metrics['NDCG@10']}")
    print(f"Robustness Results:")
    for key, val in robustness.items():
        print(f"  {key}: {val}")
    
    assert robustness['delta'] == -0.05, f"Delta should be -0.05, got {robustness['delta']}"
    assert robustness['rel_change_pct'] == -10.0, f"Rel_change_pct should be -10.0, got {robustness['rel_change_pct']}"
    assert robustness['robustness'] == 0.9, f"Robustness should be 0.9, got {robustness['robustness']}"
    
    print("  ✓ Test Case passed")

if __name__ == '__main__':
    try:
        test_compute_enhanced_metrics()
        test_compute_aggregate_metrics()
        test_compute_noise_robustness()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
