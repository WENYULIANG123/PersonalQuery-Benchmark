#!/usr/bin/env python3
"""
Benchmark script to compare performance between original and enhanced metrics computation
"""
import sys
import time
import random
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval/utils')
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval')

from utils import compute_metrics, compute_enhanced_metrics, compute_aggregate_metrics

def generate_test_data(num_queries=1000, k_max=10):
    """Generate realistic test data"""
    test_data = []
    k_values = [1, 3, 5, 10]
    
    for q_idx in range(num_queries):
        retrieved = [f"ASIN_{i:06d}" for i in random.sample(range(100000), k_max)]
        num_relevant = random.randint(0, min(3, len(retrieved)))
        relevant = set(random.sample(retrieved, num_relevant)) if num_relevant > 0 else set()
        
        test_data.append((retrieved, relevant, k_values))
    
    return test_data

def benchmark_original_metrics(test_data):
    """Benchmark original compute_metrics"""
    start = time.time()
    
    all_metrics = {k: [] for k in [1, 3, 5, 10]}
    
    for retrieved, relevant, k_values in test_data:
        for k in k_values:
            metrics = compute_metrics(retrieved, relevant, k)
            all_metrics[k].append(metrics)
    
    elapsed = time.time() - start
    return elapsed, all_metrics

def benchmark_enhanced_metrics(test_data):
    """Benchmark enhanced compute_enhanced_metrics"""
    start = time.time()
    
    all_metrics = {k: [] for k in [1, 3, 5, 10]}
    
    for retrieved, relevant, k_values in test_data:
        for k in k_values:
            metrics = compute_enhanced_metrics(retrieved, relevant, k)
            all_metrics[k].append(metrics)
    
    elapsed = time.time() - start
    return elapsed, all_metrics

def benchmark_aggregation(all_metrics_orig, all_metrics_enhanced):
    """Benchmark aggregation function"""
    k_values = [1, 3, 5, 10]
    
    start = time.time()
    aggregated = compute_aggregate_metrics(all_metrics_enhanced, k_values)
    elapsed = time.time() - start
    
    return elapsed

def main():
    print("=" * 80)
    print("PERFORMANCE BENCHMARK: Original vs Enhanced Metrics")
    print("=" * 80)
    
    num_queries = 1000
    print(f"\nGenerating {num_queries} test queries...")
    test_data = generate_test_data(num_queries, k_max=10)
    print(f"✓ Generated {len(test_data)} test cases")
    
    print("\n" + "-" * 80)
    print("BENCHMARK 1: compute_metrics() vs compute_enhanced_metrics()")
    print("-" * 80)
    
    print("\nBenchmarking original compute_metrics()...")
    time_orig, all_metrics_orig = benchmark_original_metrics(test_data)
    print(f"  Time: {time_orig:.3f}s")
    print(f"  Rate: {len(test_data) * 4 / time_orig:.0f} metrics/s")
    
    print("\nBenchmarking enhanced compute_enhanced_metrics()...")
    time_enhanced, all_metrics_enhanced = benchmark_enhanced_metrics(test_data)
    print(f"  Time: {time_enhanced:.3f}s")
    print(f"  Rate: {len(test_data) * 4 / time_enhanced:.0f} metrics/s")
    
    overhead = (time_enhanced - time_orig) / time_orig * 100
    print(f"\nOverhead: {overhead:.1f}%")
    print(f"Slowdown factor: {time_enhanced / time_orig:.2f}x")
    
    print("\n" + "-" * 80)
    print("BENCHMARK 2: compute_aggregate_metrics()")
    print("-" * 80)
    
    print("\nBenchmarking compute_aggregate_metrics()...")
    time_agg = benchmark_aggregation(all_metrics_orig, all_metrics_enhanced)
    print(f"  Time: {time_agg:.3f}s")
    
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    
    total_time_orig = time_orig
    total_time_enhanced = time_enhanced + time_agg
    
    print(f"\nOriginal pipeline (compute_metrics only):")
    print(f"  Total time: {total_time_orig:.3f}s")
    
    print(f"\nEnhanced pipeline (compute_enhanced_metrics + aggregation):")
    print(f"  Total time: {total_time_enhanced:.3f}s")
    
    total_overhead = (total_time_enhanced - total_time_orig) / total_time_orig * 100
    print(f"\nTotal overhead: {total_overhead:.1f}%")
    print(f"Total slowdown factor: {total_time_enhanced / total_time_orig:.2f}x")
    
    print("\n" + "=" * 80)
    
    if total_overhead < 30:
        print(f"✅ PASS: Overhead is {total_overhead:.1f}% (< 30% acceptable threshold)")
    else:
        print(f"⚠️  WARNING: Overhead is {total_overhead:.1f}% (> 30% acceptable threshold)")
    
    print("=" * 80)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
